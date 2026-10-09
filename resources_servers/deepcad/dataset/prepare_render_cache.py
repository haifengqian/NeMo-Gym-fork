# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Prepare verification-only reference maps without changing model inputs."""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from resources_servers.deepcad.geometry import reference_maps


def prepare(arguments: tuple[dict, Path, Path, int, int]) -> dict:
    """Validate one reference path and materialize its deterministic cached maps."""
    metadata, root, cache, voxel_resolution, render_resolution = arguments
    path = (root / metadata["reference_step"]).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("Reference STEP escapes reference-root")
    _, masks, _, _, hit = reference_maps(
        path,
        mode=metadata.get("view_mode", "axis"),
        resolution=voxel_resolution,
        render_resolution=render_resolution,
        cache_root=cache,
    )
    return {"task_id": metadata["task_id"], "views": len(masks), "cache_hit": hit}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--reference-root", type=Path, required=True)
    parser.add_argument("--cache-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--voxel-resolution", type=int, default=64)
    parser.add_argument("--render-resolution", type=int, default=256)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    rows = [json.loads(line) for line in args.input.read_text().splitlines() if line.strip()]
    jobs = [
        (row["verifier_metadata"], args.reference_root, args.cache_root, args.voxel_resolution, args.render_resolution)
        for row in rows
    ]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for index, result in enumerate(pool.map(prepare, jobs), 1):
            print(json.dumps({"prepared": index, "total": len(rows), **result}), flush=True)


if __name__ == "__main__":
    main()
