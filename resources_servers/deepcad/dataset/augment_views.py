# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Augment an existing DeepCAD JSONL without changing targets or example order."""

import argparse
import base64
import io
import json
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cadquery as cq
from PIL import Image

from resources_servers.deepcad.dataset.prepare_gym import (
    dimension_data,
    render_diagonal_views,
    sample_view_mode,
    task_row,
)


def augment(arguments: tuple[dict, Path, Path, float, int]) -> dict:
    """Replace only image inputs, their description, and view-mode metadata."""
    row, reference_root, views_root, probability, seed = arguments
    metadata = row["verifier_metadata"]
    mode = sample_view_mode(metadata["task_id"], diagonal_view_probability=probability, seed=seed)
    if mode == "diagonal":
        reference = (reference_root / metadata["reference_step"]).resolve()
        if not reference.is_relative_to(reference_root.resolve()):
            raise ValueError("Reference STEP escapes reference-root")
        shape = cq.importers.importStep(str(reference)).val()
        images = render_diagonal_views(shape, views_root / metadata["task_id"])
        replacement = task_row(
            metadata["task_id"],
            metadata["reference_step"],
            images,
            metrics={**metadata["construction_metrics"], "difficulty": metadata["difficulty"]},
            dimensions_mm=dimension_data(shape)["size_mm"].tolist(),
        )
        row["responses_create_params"]["input"] = replacement["responses_create_params"]["input"]
    else:
        if metadata.get("view_mode", "axis") != "axis":
            raise ValueError("Axis views require an original front/side/top input dataset")
        count = 0
        for message in row["responses_create_params"]["input"]:
            for item in message["content"] if isinstance(message["content"], list) else []:
                if item["type"] != "input_image":
                    continue
                payload = base64.b64decode(item["image_url"].split(",", 1)[1])
                with Image.open(io.BytesIO(payload)) as image:
                    output = io.BytesIO()
                    image.convert("RGB").resize((256, 256), Image.Resampling.LANCZOS).save(output, format="PNG")
                item["image_url"] = "data:image/png;base64," + base64.b64encode(output.getvalue()).decode()
                count += 1
        if count != 3:
            raise ValueError("Expected three original axis images")
    metadata["view_mode"] = mode
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "output", "reference-root", "views-root"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--diagonal-view-probability", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.diagonal_view_probability <= 1 or args.workers <= 0:
        parser.error("Require probability between 0 and 1 and positive workers")
    if args.input.resolve() == args.output.resolve():
        parser.error("Use a separate output to preserve the original views")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    arguments = [
        (row, args.reference_root, args.views_root, args.diagonal_view_probability, args.seed) for row in rows
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    counts = Counter()
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with ProcessPoolExecutor(max_workers=args.workers) as pool, temporary.open("w") as output:
        for index, row in enumerate(pool.map(augment, arguments), 1):
            output.write(json.dumps(row) + "\n")
            counts[row["verifier_metadata"]["view_mode"]] += 1
            if index % 32 == 0 or index == len(rows):
                print(json.dumps({"prepared": index, "total": len(rows)}), flush=True)
    temporary.replace(args.output)
    report = {
        "input": str(args.input),
        "output": str(args.output),
        "diagonal_view_probability": args.diagonal_view_probability,
        "view_seed": args.seed,
        "view_counts": dict(counts),
        "written": len(rows),
    }
    args.output.with_suffix(".report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report), flush=True)


if __name__ == "__main__":
    main()
