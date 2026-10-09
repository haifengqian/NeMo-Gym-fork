# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted geometry worker: independently normalized voxel and render rewards."""

import fcntl
import hashlib
import json
import sys
import time
import uuid
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh

from resources_servers.deepcad.surface import Camera, cameras, compare_maps, render_maps, save_diagnostics


COMPONENT_NAMES = ("voxel_iou", "silhouette_iou", "normal_alignment", "hole_iou")
SCORER_VERSION = "render-maps-v2"


def aggregate_rewards(result: dict, weights: dict[str, float]) -> dict:
    """Expose weighted components whose sum is exactly the scalar reward.

    A hole score with no nonempty union is unavailable, and contributes zero.
    Keep its component key for GDPO's stable batch schema; a group with no hole
    scores has a constant zero column and contributes no hole advantage.
    """
    if set(weights) != set(COMPONENT_NAMES) or any(not np.isfinite(w) or w < 0 for w in weights.values()):
        raise ValueError("Require a finite nonnegative weight for each reward component")
    result["reward_components"] = {name: weights[name] * (result.get(name) or 0.0) for name in COMPONENT_NAMES}
    result["reward"] = sum(result["reward_components"].values())
    return result


def reference_maps(
    path: Path, *, mode: str, resolution: int, render_resolution: int, cache_root: Path | None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Camera, ...], bool]:
    """Cache reference voxels and render maps by STEP content and scorer settings."""

    def prepare() -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[Camera, ...]]:
        shape = load_solid(path)
        mesh = normalized_mesh(shape)
        views = cameras(mesh, mode=mode)
        masks, normals = render_maps(mesh, views, resolution=render_resolution)
        if not masks.any(axis=(1, 2)).all():
            raise ValueError("Reference has an empty surface view")
        return normalized_voxels(shape, resolution=resolution), masks, normals, views

    if cache_root is None:
        return (*prepare(), False)
    settings = json.dumps([SCORER_VERSION, mode, resolution, render_resolution, 0.002, 0.05])
    digest = hashlib.sha256(path.read_bytes() + settings.encode()).hexdigest()
    cache_root.mkdir(parents=True, exist_ok=True)
    path_cache = cache_root / f"{digest}.npz"
    with (cache_root / f"{digest}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path_cache.exists():
            with np.load(path_cache, allow_pickle=False) as saved:
                voxels, masks, normals = saved["voxels"], saved["masks"], saved["normals"]
                view_data = json.loads(str(saved["cameras"]))
            views = tuple(Camera(**item) for item in view_data)
            expected = (4 if mode == "diagonal" else 3, render_resolution, render_resolution)
            if (
                voxels.shape != (resolution + 4,) * 3
                or masks.shape != expected
                or normals.shape != (*expected, 3)
                or not masks.any(axis=(1, 2)).all()
                or not np.isfinite(normals).all()
            ):
                raise ValueError("Invalid reference render cache")
            return voxels, masks, normals, views, True
        voxels, masks, normals, views = prepare()
        temporary = cache_root / f".{digest}.{uuid.uuid4().hex}.tmp"
        try:
            with temporary.open("wb") as output:
                np.savez_compressed(
                    output,
                    voxels=voxels,
                    masks=masks,
                    normals=normals,
                    cameras=np.array(json.dumps([vars(view) for view in views])),
                )
            temporary.replace(path_cache)
        finally:
            temporary.unlink(missing_ok=True)
        return voxels, masks, normals, views, False


def load_solid(path: Path) -> cq.Shape:
    """Read a STEP file and reject invalid, empty, or non-solid geometry."""
    shape = cq.importers.importStep(str(path)).val()
    volume = shape.Volume()
    if not shape.Solids() or not shape.isValid() or not np.isfinite(volume) or volume <= 0:
        raise ValueError("STEP must contain valid solids with positive finite volume")
    return shape


def normalized_mesh(shape: cq.Shape) -> trimesh.Trimesh:
    """Center the mesh bbox at 0.5 and scale its longest dimension to one.

    Tessellation tolerance follows the input size, so metre and millimetre
    programs have the same mesh accuracy. Axes and orientation are preserved.
    """
    vertices, faces = shape.tessellate(shape.BoundingBox().DiagonalLength * 0.002, 0.05)
    points = np.array([vertex.toTuple() for vertex in vertices], dtype=float)
    if not len(points) or not len(faces) or not np.isfinite(points).all():
        raise ValueError("Empty or non-finite mesh")
    lower, upper = points.min(axis=0), points.max(axis=0)
    longest = float(np.max(upper - lower))
    if not np.isfinite(longest) or longest <= 0:
        raise ValueError("Degenerate mesh bounding box")
    points = (points - (lower + upper) / 2) / longest + 0.5
    return trimesh.Trimesh(vertices=points, faces=np.asarray(faces), process=False)


def normalized_voxels(shape: cq.Shape, *, resolution: int = 64) -> np.ndarray:
    """Voxelize and fill at pitch 1/resolution, then center-pad like BenchCAD."""
    if resolution <= 0:
        raise ValueError("Voxel resolution must be positive")
    voxels = normalized_mesh(shape).voxelized(pitch=1.0 / resolution).fill()
    occupied = voxels.matrix.astype(bool)
    size = resolution + 4
    if any(dimension > size for dimension in occupied.shape):
        raise ValueError("Normalized voxel grid exceeds its padded bounds")
    grid = np.zeros((size, size, size), dtype=bool)
    start = (size - np.array(occupied.shape)) // 2
    slices = tuple(
        slice(int(offset), int(offset + length)) for offset, length in zip(start, occupied.shape, strict=True)
    )
    grid[slices] = occupied
    if not grid.any():
        raise ValueError("Empty voxelized solid")
    return grid


def voxel_iou(prediction: np.ndarray, reference: np.ndarray, *, resolution: int = 64) -> dict:
    """Compute filled-grid overlap without rotation, reflection or axis rescaling."""
    intersection = int(np.count_nonzero(prediction & reference))
    union = int(np.count_nonzero(prediction | reference))
    if not union:
        raise ValueError("Empty voxel union")
    return {
        "reward": intersection / union,
        "scoring_method": "voxel_iou",
        "voxel_resolution": resolution,
        "intersection_voxels": intersection,
        "union_voxels": union,
        "prediction_voxels": int(np.count_nonzero(prediction)),
        "reference_voxels": int(np.count_nonzero(reference)),
    }


def score(prediction: cq.Shape, reference: cq.Shape, *, resolution: int = 64) -> dict:
    """Score independently normalized meshes on filled grids at pitch 1/64 by default."""
    return voxel_iou(
        normalized_voxels(prediction, resolution=resolution),
        normalized_voxels(reference, resolution=resolution),
        resolution=resolution,
    )


def main() -> None:
    """Invalid reference is infrastructure; invalid prediction earns zero."""
    resolution = int(sys.argv[3]) if len(sys.argv) > 3 else 64
    options = json.loads(sys.argv[4]) if len(sys.argv) > 4 else {}
    rendered = options.get("render_rewards", False)
    started = time.monotonic()
    try:
        if rendered:
            reference, ref_masks, ref_normals, views, cache_hit = reference_maps(
                Path(sys.argv[2]),
                mode=options.get("view_mode", "axis"),
                resolution=resolution,
                render_resolution=options.get("render_resolution", 256),
                cache_root=Path(options["cache_root"]) if options.get("cache_root") else None,
            )
        else:
            reference = normalized_voxels(load_solid(Path(sys.argv[2])), resolution=resolution)
    except Exception as exc:
        print(
            json.dumps(
                {
                    "reward": 0.0,
                    "mask_sample": True,
                    "failure_kind": "deepcad:reference",
                    "failure_reason": str(exc),
                }
            )
        )
        return
    try:
        shape = load_solid(Path(sys.argv[1]))
        prediction = normalized_voxels(shape, resolution=resolution)
    except Exception as exc:
        print(json.dumps({"reward": 0.0, "failure_kind": "deepcad:geometry", "failure_reason": str(exc)}))
        return
    try:
        result = voxel_iou(prediction, reference, resolution=resolution)
        result["voxel_iou"] = result["reward"]
        if rendered:
            pred_masks, pred_normals = render_maps(
                normalized_mesh(shape), views, resolution=options.get("render_resolution", 256)
            )
            result.update(compare_maps(pred_masks, ref_masks, pred_normals, ref_normals, views))
            result["scoring_method"] = "cad_multireward"
            result["reference_cache_hit"] = cache_hit
            result["render_resolution"] = options.get("render_resolution", 256)
            result["scorer_version"] = SCORER_VERSION
            if options.get("artifact_dir"):
                try:
                    save_diagnostics(Path(options["artifact_dir"]), pred_masks, ref_masks, pred_normals, ref_normals)
                except OSError as exc:
                    print(f"Could not save verification diagnostics: {exc}", file=sys.stderr)
        if options:
            aggregate_rewards(result, options.get("reward_weights", dict.fromkeys(COMPONENT_NAMES, 1.0)))
        result["scoring_time_s"] = time.monotonic() - started
    except Exception as exc:
        result = {
            "reward": 0.0,
            "mask_sample": True,
            "failure_kind": "deepcad:infrastructure",
            "failure_reason": str(exc),
        }
    print(json.dumps(result))


if __name__ == "__main__":
    main()
