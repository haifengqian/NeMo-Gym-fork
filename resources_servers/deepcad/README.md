# DeepCAD reconstruction

The model receives either three dimensioned **front**, **side**, and **top** images or a composite of four diagonal orthographic views, and writes Python with CadQuery. Preparation defaults to `p = 0.75` for diagonal views and `1 - p = 0.25` for front/side/top. Views, annotation visibility, and foreground color are sampled once per example from its task ID and seed, so all GRPO responses for that example use identical input bytes. The prompt names the selected views and their camera directions, requires one Python fenced block, and assigns the final `cadquery.Workplane` or `cadquery.Shape` to `result`. When annotations are enabled, the axis views include overall sizes, circle/arc radius leaders (up to six distinct radii per view), and axis-aligned rectangular additive extrusion depths (up to two per view), where available. Labels use millimetres. Axis prompts request millimetres; diagonal prompts request normalized coordinates with bounding-box center `[0.5, 0.5, 0.5]` and longest side spanning `[0,1]`.

The verifier executes code in a disposable NeMo Gym sandbox, exports the result as STEP, and computes filled voxel intersection-over-union against the hidden reference. Syntax errors, runtime errors, timeouts, missing results, and invalid or empty solids earn **0**. Dataset and sandbox infrastructure failures return `mask_sample: true` with a reason.

The default scorer independently centers the predicted and reference mesh bounding boxes at `[0.5, 0.5, 0.5]`, uniformly scales each longest bounding-box dimension to `1`, and voxelizes and fills at pitch `1/64`. Like BenchCAD, the filled matrices are center-padded into a shared `68×68×68` grid. Reward is occupied-voxel intersection divided by union. Normalization removes translation and uniform size; it preserves orientation, aspect ratios, and reflections. There is no Procrustes or ICP rotation matching. Features below the voxel pitch may be lost or merged. `voxel_resolution` defaults to `64`; changing it changes the reward metric. Responses include the resolution and occupied/intersection/union voxel counts.

## Setup

From the Gym checkout:

```bash
uv pip install --python .venv/bin/python -r resources_servers/deepcad/requirements.txt
docker build -t deepcad-cadquery:2.8.0 resources_servers/deepcad
gym env start --config resources_servers/deepcad/configs/deepcad.yaml --model-type openai_model
gym eval run --no-serve --agent deepcad_simple_agent \
  --input resources_servers/deepcad/data/example.jsonl \
  --output /tmp/deepcad-rollouts.jsonl --limit 5 --max-output-tokens 4096
```

Configure a vision-capable model server and credentials using its normal Gym configuration. Sandbox images contain CadQuery; Gym installs verifier Python dependencies from `requirements.txt`. Default Docker execution disables networking and drops capabilities. Never mount the reference directory, credentials, or the host home into the generated-code sandbox. Enroot uses the cluster's rootless container isolation; review site hooks before executing untrusted code. Each request uses a fresh container, a bounded concurrency slot, execution/geometry timeouts, and a STEP size limit.

HSG uses the separate CadQuery image `~/data/deepcad/gym/cadquery-2.8.0.sqsh`. Use the standalone `configs/deepcad-hsg.yaml` configuration on HSG. The parent `cad-rl` rem project uses `rl.nightly.sep30.2026.sqsh`.

## View augmentation and data

The conversion utility, dimension renderer, and prompt live in this environment's `dataset/prepare_gym.py` and `dataset/gym-prompt.txt`. Run preparation from the Gym checkout. Supply a JSONL manifest with local image paths relative to the manifest and a trusted reference STEP:

```json
{"task_id":"part-001","cad_json":"source.json","reference_step":"part.step","images":{"front":"front.png","side":"side.png","top":"top.png"}}
```

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.prepare_gym \
  --manifest /path/tasks.jsonl --limit 5 \
  --reference-root /path/references --output /path/tasks-gym.jsonl
```

Set the resources server's `reference_root` to the same reference directory. The utility embeds the selected views in Responses API messages; target paths appear only in `verifier_metadata`. No text descriptions are required. Supplied views should use the labeled orthographic directions and a common scale. Headless Linux preparation uses CPU off-screen VTK and requires `libosmesa6` (`apt-get install -y libosmesa6`); the provided Dockerfile includes it. For an existing DeepCAD split, the utility can reconstruct unquantized JSON sketches/extrusions using modern OCP and render the selected views:

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.prepare_gym \
  --data-root ~/data/deepcad --split validation --limit 100 \
  --reference-root ~/data/deepcad/gym/references \
  --output ~/data/deepcad/gym/validation.jsonl \
  --diagonal-view-probability 0.75 --no-annotations-probability 0.75 --seed 42
```

Each individual view is 256×256 pixels at a shared orthographic scale. Front/side/top are three separate images. Diagonal views use BenchCAD's quadrant directions: top-left `(-1,-1,-1)`, top-right `(+1,+1,+1)`, bottom-left `(+1,-1,+1)`, and bottom-right `(-1,+1,-1)`, all looking toward the center with world Z up. These vectors are camera directions relative to the part center. The four 256px views form a 524×524 composite with 4px white gutters. Geometry is normalized by its longest bounding-box side and centered at `[0.5, 0.5, 0.5]`; the prompt also supplies overall XYZ sizes in millimetres for physical proportions. The off-screen VTK renderer starts with BenchCAD's 0.55 half-height and widens the shared frame only when needed to keep all four views unclipped. Overall sizes and circular radii come from the final B-rep, with the source JSON's metres converted to millimetres. Radii are only labeled in views normal to their circle plane; projected duplicates are counted once. Depth labels describe rectangular additive features in the construction history, rather than inferred wall thickness. Dimensions describe the target geometry; reward still normalizes uniform size independently. Supplied-image manifests use resized 256px input images for the axis branch, and render diagonal views from their STEP for the diagonal branch. They should include dimension annotations if desired.

Preparation reports each failed reconstruction rather than silently replacing it with an incorrect target. The committed `data/example.jsonl` contains five validation tasks and their reference STEP files. Large training/validation files stay outside Git. Geometry preparation is offline; only prediction execution and scoring happen during verification. Regenerate JSONL after changing the renderer: images are embedded in each task, and an already-running training process keeps its loaded inputs.

Set `--diagonal-view-probability 0` for front/side/top only or `1` for diagonal only. The same flag is available in `prepare_gym_bucket` and defaults to 0.75 there too. Changing `--seed` creates a different reproducible assignment; worker count, task order, and construction difficulty do not affect a task's choice. `verifier_metadata.view_mode` records `axis` or `diagonal`, and preparation reports include the probability, seed, and actual view counts.

To assess a model's existing CadQuery API knowledge separately from CAD generation, use the
[closed-book API probe](knowledge_eval/README.md). It asks about each documented symbol using only its name,
saves individual Markdown answers and raw responses, and compares explicit signatures against the reference snapshot.

To augment an existing original front/side/top JSONL while preserving its target STEP files, difficulty values, membership, and ordering:

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.augment_views \
  --input /path/original.jsonl --output /path/augmented.jsonl \
  --reference-root /path/references --views-root /path/augmented-views \
  --diagonal-view-probability 0.75 --no-annotations-probability 0.75 --seed 42 --workers 4
```

Use a separate output file to retain the original dataset. Read its `.report.json` for the realized mix. Train with that new JSONL; image selection happens before tokenization and multimodal preprocessing.

Annotation visibility is sampled independently, with `--no-annotations-probability 0.75` by default: 75% omit all engineering labels and leaders from axis images and omit physical dimension text from both prompt variants. The remaining 25% retain available annotations. Diagonal composites do not have dimension overlays; their annotation option controls the supplied overall XYZ dimension text. Set this probability to `0` to keep annotations or `1` to omit them. Omitted axis annotations are removed by rendering clean geometry from the reference STEP, rather than masking parts of the image. `verifier_metadata.engineering_annotations` records the fixed choice and reports include `annotation_counts`.

All embedded input images have white backgrounds. Foreground augmentation selects one of blue, green, teal, steel blue, violet, or amber and applies it to all images in that example. The color ramp preserves black edges, shading, dimension labels, and white antialiased boundaries. Transparent supplied images are composited onto white. View, annotation, and color selection use separate hash domains (`deepcad-views`, `deepcad-annotations`, and `deepcad-colors`), so selecting difficulty buckets or diagonal views does not force a particular color. The task ID and `--seed` determine all three choices; no sampling occurs during training or response generation. Metadata records `foreground_color` and `input_augmentation_seed`, and reports include `color_counts`.

To update prompts and color augmentation in an already prepared axis/diagonal JSONL without rendering geometry or changing the saved view or annotation choice:

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.augment_inputs \
  --input /path/views.jsonl --output /path/colored.jsonl --seed 42
```

This command preserves target metadata, task membership, difficulty, image sizes, and ordering. Reapplying it with the same seed preserves the image bytes exactly. To change the color seed, start from the original uncolored JSONL. Preparation via `prepare_gym`, `prepare_gym_bucket`, or `augment_views` includes the same prompt, annotation, and color augmentation automatically.

## Construction difficulty

The utility scores the entire chosen split, then sorts from easiest to hardest before applying `--limit`. Ties use task ID. It computes the requested metric from the profiles actually used by extrusion features:

```text
difficulty = 4 * extrusions + sketch_curves + 2 * additional_loops + 4 * cuts_and_intersections
```

An extrusion counts once per feature. Additional loops are `max(loops - 1, 0)` per referenced profile; unused sketches/profiles do not count. Reused profiles count once per use. Arc/circle counts are recorded but have no separate weight beyond sketch curves. Each JSON row carries `difficulty` and the counts in `verifier_metadata`; neither is included in the model prompt or affects IoU. A companion `.difficulty.jsonl` records all ranked split IDs, including tasks beyond the rendering limit. Supplied-image manifests must identify the source `cad_json` so construction difficulty can be measured.

## Live rollout viewer

Set `live_rollout_root` on the resources server to a shared directory and
`live_rollout_experiment` to the experiment name. Each verifier call saves the
fixed prompt images, generated code and response, reward, and failure reason;
successful execution also saves a downloadable prediction STEP. Prompt images
are deduplicated across responses. Recording failures are logged and do not
change rewards. Four Gym replicas can append to the same experiment safely.

Start the lightweight viewer on a persistent CPU host with access to that directory:

```bash
python3 -m resources_servers.deepcad.rollout_viewer \
  --root /data/deepcad/gym/live-rollouts --advertise-host viewer-host \
  --experiment my-experiment
```

The viewer prints its URL, chooses a free port on first launch, and saves it in
`.viewer-state.json` for reuse on restart. It creates an experiment folder and
serves `/<experiment>`, with live refresh, search, reward sorting, prompt images,
code, response, and STEP downloads. New experiment folders appear automatically.
Run one viewer for the shared root; a duplicate launch returns the existing port.
It requires only Python's standard library and can stay up between GPU allocations.
Recording is disabled when `live_rollout_root` is unset. The Super mixed recipe
enables it and uses the W&B run name as the experiment folder.

## Validation and licensing

```bash
.venv/bin/pytest resources_servers/deepcad/tests
gym dataset collate --config resources_servers/deepcad/configs/deepcad.yaml \
  --mode example_validation --output-dir /tmp/deepcad-validated
```

Code: Apache-2.0. DeepCAD source/data: [upstream repository and license](https://github.com/rundiwu/DeepCAD) (MIT). CadQuery: Apache-2.0; OCP/OpenCascade: LGPL-2.1 with its exception; NumPy/SciPy: BSD-3-Clause; Trimesh: MIT; Pillow: HPND. This new environment remains `verified: false` until benchmark baselines are reviewed.
