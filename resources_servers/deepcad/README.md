# DeepCAD reconstruction

The model receives exactly three dimensioned images, labeled **front**, **side**, and **top**, and writes Python with CadQuery. It must assign a `cadquery.Workplane` or `cadquery.Shape` containing valid solids to `result`. Generated DeepCAD views include overall sizes, circle/arc radius leaders (up to six distinct radii per view), and axis-aligned rectangular additive extrusion depths (up to two per view), where available. All labels use millimetres; the prompt asks for CadQuery lengths in millimetres.

The verifier executes code in a disposable NeMo Gym sandbox, exports the result as STEP, and computes solid-volume intersection-over-union against the hidden reference. Syntax errors, runtime errors, timeouts, missing results, and invalid or empty solids earn **0**. Dataset and sandbox infrastructure failures return `mask_sample: true` with a reason.

Alignment removes translation, proper rotation, and **uniform scale**; it never reflects or independently scales axes. Surface registration tries 24 proper PCA-axis orientations and an identity start, using nearest-neighbor ICP with similarity Procrustes fits. This is an approximate registration and can reach a local optimum for ambiguous shapes. IoU uses OpenCascade solid intersections rather than voxel occupancy. The response includes the fitted transform and surface alignment error.

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

## Three-view data

The conversion utility, dimension renderer, and prompt live in this environment's `dataset/prepare_gym.py` and `dataset/gym-prompt.txt`. Run preparation from the Gym checkout. Supply a JSONL manifest with local image paths relative to the manifest and a trusted reference STEP:

```json
{"task_id":"part-001","cad_json":"source.json","reference_step":"part.step","images":{"front":"front.png","side":"side.png","top":"top.png"}}
```

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.prepare_gym \
  --manifest /path/tasks.jsonl --limit 5 \
  --reference-root /path/references --output /path/tasks-gym.jsonl
```

Set the resources server's `reference_root` to the same reference directory. The utility embeds all three PNG/JPEG/WebP images in Responses API messages; target paths appear only in `verifier_metadata`. No text descriptions are required. Supplied views should use the labeled orthographic directions and a common scale. For an existing DeepCAD split, the utility can reconstruct unquantized JSON sketches/extrusions using modern OCP and render the three views:

```bash
uv run --no-sync python -m resources_servers.deepcad.dataset.prepare_gym \
  --data-root ~/data/deepcad --split validation --limit 100 \
  --reference-root ~/data/deepcad/gym/references \
  --output ~/data/deepcad/gym/validation.jsonl
```

The renderer produces 640×640 images at a shared orthographic scale. Overall sizes and circular radii come from the final B-rep, with the source JSON's metres converted to millimetres. Radii are only labeled in views normal to their circle plane; projected duplicates are counted once. Depth labels describe rectangular additive features in the construction history, rather than inferred wall thickness. Dimensions describe the target geometry; reward still permits uniform-scale alignment. Supplied-image manifests preserve their input images and should include dimension annotations if desired.

Preparation reports each failed reconstruction rather than silently replacing it with an incorrect target. The committed `data/example.jsonl` contains five validation tasks and their reference STEP files. Large training/validation files stay outside Git. Geometry preparation is offline; only prediction execution and scoring happen during verification. Regenerate JSONL after changing the renderer: images are embedded in each task, and an already-running training process keeps its loaded inputs.

## Construction difficulty

The utility scores the entire chosen split, then sorts from easiest to hardest before applying `--limit`. Ties use task ID. It computes the requested metric from the profiles actually used by extrusion features:

```text
difficulty = 4 * extrusions + sketch_curves + 2 * additional_loops + 4 * cuts_and_intersections
```

An extrusion counts once per feature. Additional loops are `max(loops - 1, 0)` per referenced profile; unused sketches/profiles do not count. Reused profiles count once per use. Arc/circle counts are recorded but have no separate weight beyond sketch curves. Each JSON row carries `difficulty` and the counts in `verifier_metadata`; neither is included in the model prompt or affects IoU. A companion `.difficulty.jsonl` records all ranked split IDs, including tasks beyond the rendering limit. Supplied-image manifests must identify the source `cad_json` so construction difficulty can be measured.

## Validation and licensing

```bash
.venv/bin/pytest resources_servers/deepcad/tests
gym dataset collate --config resources_servers/deepcad/configs/deepcad.yaml \
  --mode example_validation --output-dir /tmp/deepcad-validated
```

Code: Apache-2.0. DeepCAD source/data: [upstream repository and license](https://github.com/rundiwu/DeepCAD) (MIT). CadQuery: Apache-2.0; OCP/OpenCascade: LGPL-2.1 with its exception; NumPy/SciPy: BSD-3-Clause; Pillow: HPND. This new environment remains `verified: false` until benchmark baselines are reviewed.
