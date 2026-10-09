# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted geometry worker: independently normalized, filled voxel IoU."""

import json
import sys
from pathlib import Path

import cadquery as cq
import numpy as np
import trimesh


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
    try:
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
        prediction = normalized_voxels(load_solid(Path(sys.argv[1])), resolution=resolution)
        result = voxel_iou(prediction, reference, resolution=resolution)
    except Exception as exc:
        result = {"reward": 0.0, "failure_kind": "deepcad:geometry", "failure_reason": str(exc)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
