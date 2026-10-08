# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Trusted geometry worker: similarity Procrustes registration and solid-volume IoU."""

import itertools
import json
import sys
from pathlib import Path

import cadquery as cq
import numpy as np
from OCP.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCP.gp import gp_Trsf
from scipy.spatial import cKDTree


def load_solid(path: Path) -> cq.Shape:
    """Read a STEP file and reject invalid, empty, or non-solid geometry."""
    shape = cq.importers.importStep(str(path)).val()
    volume = shape.Volume()
    if not shape.Solids() or not shape.isValid() or not np.isfinite(volume) or volume <= 0:
        raise ValueError("STEP must contain valid solids with positive finite volume")
    return shape


def surface_points(shape: cq.Shape, count: int) -> np.ndarray:
    """Deterministically sample triangles by area, in normalized size units."""
    size = shape.BoundingBox().DiagonalLength
    vertices, faces = shape.tessellate(size * 0.002, 0.05)
    triangles = np.asarray([v.toTuple() for v in vertices])[np.asarray(faces)]
    area = np.linalg.norm(np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]), axis=1)
    if not len(area) or not np.isfinite(area).all() or area.sum() <= 0:
        raise ValueError("Cannot sample the solid surface")
    rng = np.random.default_rng(1729)
    chosen = triangles[rng.choice(len(triangles), count, p=area / area.sum())]
    u = np.sqrt(rng.random((count, 1)))
    v = rng.random((count, 1))
    return (1 - u) * chosen[:, 0] + u * (1 - v) * chosen[:, 1] + u * v * chosen[:, 2]


def procrustes(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Least-squares translation, proper rotation, and positive uniform scale for paired points."""
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    x, y = source - src_center, target - dst_center
    u, singular, vt = np.linalg.svd(x.T @ y)
    signs = np.ones(3)
    signs[-1] = np.linalg.det(vt.T @ u.T)
    rotation = vt.T @ np.diag(signs) @ u.T
    denominator = np.sum(x * x)
    scale = np.sum(singular * signs) / denominator if denominator > 0 else 0
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Degenerate Procrustes point set")
    matrix = np.eye(4)
    matrix[:3, :3] = scale * rotation
    matrix[:3, 3] = dst_center - scale * rotation @ src_center
    return matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def alignment(source: np.ndarray, target: np.ndarray, iterations: int) -> tuple[np.ndarray, float]:
    """Try proper PCA-axis orientations, then refine correspondences with ICP.

    DeepCAD and generated solids do not share vertices. Each ICP step fits a
    similarity Procrustes transform to nearest surface points. This multistart
    approximation is deterministic, but does not guarantee a global optimum.
    """
    src_center, dst_center = source.mean(axis=0), target.mean(axis=0)
    x, y = source - src_center, target - dst_center
    src_axes = np.linalg.eigh(x.T @ x)[1]
    dst_axes = np.linalg.eigh(y.T @ y)[1]
    scale = np.sqrt(np.sum(y * y) / np.sum(x * x))
    starts = [np.eye(3)]
    for permutation in itertools.permutations(range(3)):
        for signs in itertools.product((-1, 1), repeat=3):
            rotation = dst_axes[:, permutation] @ np.diag(signs) @ src_axes.T
            if np.linalg.det(rotation) > 0:
                starts.append(rotation)
    tree = cKDTree(target)
    best_error, best_matrix = float("inf"), np.eye(4)
    for rotation in starts:
        matrix = np.eye(4)
        matrix[:3, :3] = scale * rotation
        matrix[:3, 3] = dst_center - scale * rotation @ src_center
        last_error = float("inf")
        for _ in range(iterations):
            moved = transform_points(source, matrix)
            distance, indices = tree.query(moved)
            error = np.mean(distance**2)
            matrix = procrustes(moved, target[indices]) @ matrix
            if abs(last_error - error) < 1e-10 * np.sum(y * y) / len(y):
                break
            last_error = error
        moved = transform_points(source, matrix)
        error = float(np.mean(tree.query(moved)[0] ** 2) + np.mean(cKDTree(moved).query(target)[0] ** 2))
        if error < best_error:
            best_error, best_matrix = error, matrix
    return best_matrix, best_error


def score(prediction: cq.Shape, reference: cq.Shape, *, samples: int = 2048, iterations: int = 40) -> dict:
    """Compute BRep volume IoU after sampled-surface similarity registration."""
    matrix, error = alignment(surface_points(prediction, samples), surface_points(reference, samples), iterations)
    transform = gp_Trsf()
    transform.SetValues(*matrix[:3].ravel().tolist())
    aligned = cq.Shape.cast(BRepBuilderAPI_Transform(prediction.wrapped, transform, True).Shape())
    intersection = aligned.intersect(reference).Volume()
    pred_volume, ref_volume = aligned.Volume(), reference.Volume()
    intersection = min(max(intersection, 0.0), pred_volume, ref_volume)
    reward = intersection / (pred_volume + ref_volume - intersection)
    if not np.isfinite(reward):
        raise ValueError("Non-finite IoU")
    return {"reward": float(np.clip(reward, 0, 1)), "alignment_error": error, "transform": matrix.tolist()}


def main() -> None:
    """Invalid reference is infrastructure; invalid prediction earns zero."""
    try:
        reference = load_solid(Path(sys.argv[2]))
    except Exception as exc:
        print(
            json.dumps(
                {"reward": 0.0, "mask_sample": True, "failure_kind": "deepcad:reference", "failure_reason": str(exc)}
            )
        )
        return
    try:
        prediction = load_solid(Path(sys.argv[1]))
        result = score(prediction, reference, samples=int(sys.argv[3]), iterations=int(sys.argv[4]))
    except Exception as exc:
        result = {"reward": 0.0, "failure_kind": "deepcad:geometry", "failure_reason": str(exc)}
    print(json.dumps(result))


if __name__ == "__main__":
    main()
