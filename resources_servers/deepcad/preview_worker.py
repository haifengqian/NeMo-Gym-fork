# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render saved prediction STEPs on demand, separately from the verifier."""

import argparse
import fcntl
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from resources_servers.deepcad.live_rollouts import write_json
from resources_servers.deepcad.rollout_viewer import contained_path


def render_preview(step: Path, output: Path, *, mode: str, color: str) -> None:
    """Render the prompt's camera layout and color without executing generated code."""
    # CAD and rendering dependencies are optional for the lightweight HTTP viewer.
    from PIL import Image

    from resources_servers.deepcad.dataset.augment_inputs import colorize_image
    from resources_servers.deepcad.dataset.prepare_gym import render_diagonal_views, render_views
    from resources_servers.deepcad.geometry import load_solid

    shape = load_solid(step)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".preview-") as temporary:
        directory = Path(temporary)
        if mode == "diagonal":
            images = render_diagonal_views(shape, directory)
            with Image.open(images["diagonal"]) as image:
                composite = image.convert("RGB")
        elif mode == "axis":
            images = render_views(shape, directory, engineering_annotations=False)
            composite = Image.new("RGB", (784, 264), "white")
            for index, name in enumerate(("front", "side", "top")):
                with Image.open(images[name]) as image:
                    composite.paste(image, (4 + index * 260, 4))
        else:
            raise ValueError(f"Unknown view mode: {mode}")
        preview = directory / "prediction.png"
        colorize_image(composite, color=color).save(preview)
        preview.replace(output)


def process_request(root: Path, request: Path) -> None:
    """Isolate each STEP render and preserve optional preview failures for the viewer."""
    job = json.loads(request.read_text())
    directory = contained_path(root, job["experiment"], "rollouts", job["id"])
    status = directory / "preview-status.json"
    write_json(status, {"state": "rendering"})
    try:
        record = json.loads((directory / "reward.json").read_text())
        metadata = record["verifier_metadata"]
        step = contained_path(root, job["experiment"], "rollouts", job["id"], "prediction.step")
        output = contained_path(root, job["experiment"], "rollouts", job["id"], "prediction.png")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "resources_servers.deepcad.preview_worker",
                "--step",
                str(step),
                "--output",
                str(output),
                "--mode",
                metadata.get("view_mode", "axis"),
                "--color",
                metadata.get("foreground_color", "teal"),
            ],
            check=True,
            capture_output=True,
            timeout=60,
        )
        write_json(status, {"state": "ready"})
    except subprocess.CalledProcessError as exc:
        write_json(status, {"state": "failed", "message": exc.stderr.decode(errors="replace")[-1600:]})
    except Exception as exc:
        write_json(status, {"state": "failed", "message": str(exc)})
    finally:
        request.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path)
    parser.add_argument("--step", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mode", choices=("diagonal", "axis"), default="diagonal")
    parser.add_argument("--color", default="teal")
    args = parser.parse_args()
    if args.step:
        if args.output is None:
            parser.error("--step requires --output")
        render_preview(args.step, args.output, mode=args.mode, color=args.color)
        return
    if args.root is None:
        parser.error("Specify --root to watch viewer requests, or --step and --output to render one model")
    root = args.root.expanduser().resolve()
    requests = root / ".preview-requests"
    requests.mkdir(parents=True, exist_ok=True)
    with (root / ".preview-worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        print(f"Watching render requests in {requests}", flush=True)
        while True:
            for request in sorted(requests.glob("*.json"), key=lambda path: path.stat().st_mtime):
                process_request(root, request)
            time.sleep(1)


if __name__ == "__main__":
    main()
