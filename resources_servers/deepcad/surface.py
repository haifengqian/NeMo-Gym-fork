# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Deterministic orthographic silhouettes and floating-point world-normal maps."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import trimesh
from PIL import Image
from scipy import ndimage


@dataclass(frozen=True)
class Camera:
    name: str
    direction: tuple[float, float, float]
    right: tuple[float, float, float]
    up: tuple[float, float, float]
    half_height: float


def cameras(mesh: trimesh.Trimesh, *, mode: str) -> tuple[Camera, ...]:
    """Match the prompt's camera axes and framing, deriving scale from the reference."""
    if mode == "axis":
        return (
            Camera("front", (0, -1, 0), (1, 0, 0), (0, 0, 1), 640 / 840),
            Camera("side", (1, 0, 0), (0, -1, 0), (0, 0, 1), 640 / 840),
            Camera("top", (0, 0, 1), (1, 0, 0), (0, 1, 0), 640 / 840),
        )
    if mode != "diagonal":
        raise ValueError(f"Unknown view mode: {mode}")
    axes = []
    maximum = 0.0
    for name, position in zip(
        ("top-left", "top-right", "bottom-left", "bottom-right"),
        ((-1, -1, -1), (1, 1, 1), (1, -1, 1), (-1, 1, -1)),
        strict=True,
    ):
        direction = np.asarray(position, dtype=float) / np.sqrt(3)
        right = np.cross([0, 0, 1], direction)
        right /= np.linalg.norm(right)
        up = np.cross(direction, right)
        projected = np.stack([(mesh.vertices - 0.5) @ right, (mesh.vertices - 0.5) @ up], axis=1)
        maximum = max(maximum, float(np.abs(projected).max()))
        axes.append((name, tuple(direction), tuple(right), tuple(up)))
    half_height = max(0.55, maximum * 1.05)
    return tuple(Camera(*axis, half_height) for axis in axes)


def render_maps(
    mesh: trimesh.Trimesh, views: tuple[Camera, ...], *, resolution: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize visible triangles into masks and unlit XYZ normals.

    A CPU depth buffer keeps the nearest surface at each pixel. No antialiasing,
    RGB encoding, smoothing across CAD edges, or graphics driver is involved.
    Conservative triangle/pixel intersection preserves subpixel-thin projections.
    Equal-depth ties use the first triangle, making repeated renders deterministic.
    """
    if resolution <= 0:
        raise ValueError("Render resolution must be positive")
    masks = np.zeros((len(views), resolution, resolution), dtype=bool)
    normal_maps = np.zeros((*masks.shape, 3), dtype=np.float32)
    normals = mesh.face_normals
    for index, camera in enumerate(views):
        scale = resolution / (2 * camera.half_height)
        centered = mesh.vertices - 0.5
        projected = np.stack(
            [resolution / 2 + centered @ camera.right * scale, resolution / 2 - centered @ camera.up * scale],
            axis=1,
        )
        depths = centered @ camera.direction
        zbuffer = np.full((resolution, resolution), -np.inf)
        for face_index, face in enumerate(mesh.faces):
            if np.linalg.norm(normals[face_index]) < 0.5:
                continue
            triangle = projected[face]
            lower = np.maximum(np.floor(triangle.min(axis=0)).astype(int), 0)
            upper = np.minimum(np.ceil(triangle.max(axis=0)).astype(int) - 1, resolution - 1)
            if np.any(lower > upper):
                continue
            a, b, c = triangle
            denominator = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1])
            if abs(denominator) < 1e-12:
                continue
            y, x = np.mgrid[lower[1] : upper[1] + 1, lower[0] : upper[0] + 1]
            x, y = x + 0.5, y + 0.5
            first = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / denominator
            second = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / denominator
            third = 1 - first - second
            # Separating-axis tests for a triangle against each unit pixel square.
            # The bbox supplies X/Y axes; these margins supply the three edge axes.
            margins = [
                0.5 * np.abs(end - start).sum() / abs(denominator)
                for start, end in ((b, c), (c, a), (a, b))
            ]
            inside = (first >= -margins[0]) & (second >= -margins[1]) & (third >= -margins[2])
            # Edge pixels can have their centers outside the triangle. Keep their
            # depth on its surface instead of extrapolating past a silhouette.
            weights = np.maximum(np.stack([first, second, third]), 0)
            weights /= weights.sum(axis=0)
            depth = np.einsum("i,ijk->jk", depths[face], weights)
            region = (slice(lower[1], upper[1] + 1), slice(lower[0], upper[0] + 1))
            visible = inside & (depth > zbuffer[region] + 1e-12)
            zbuffer[region][visible] = depth[visible]
            masks[index][region][visible] = True
            normal_maps[index][region][visible] = normals[face_index]
    return masks, normal_maps


def hole_masks(masks: np.ndarray) -> np.ndarray:
    """Keep inverted-mask components that do not touch any image edge (8-connected)."""
    holes = np.zeros_like(masks, dtype=bool)
    for index, mask in enumerate(masks):
        labels, _ = ndimage.label(~mask, structure=np.ones((3, 3), dtype=bool))
        border = np.unique(np.concatenate([labels[0], labels[-1], labels[:, 0], labels[:, -1]]))
        holes[index] = (labels != 0) & ~np.isin(labels, border)
    return holes


def compare_maps(
    prediction: np.ndarray,
    reference: np.ndarray,
    prediction_normals: np.ndarray,
    reference_normals: np.ndarray,
    views: tuple[Camera, ...],
) -> dict:
    """Average per-view IoU and signed normal dot products; skip empty hole unions."""
    if prediction.shape != reference.shape or prediction_normals.shape != (*prediction.shape, 3):
        raise ValueError("Prediction maps do not match reference dimensions")
    if reference_normals.shape != prediction_normals.shape or len(views) != len(reference):
        raise ValueError("Normal maps or cameras do not match")
    if not np.isfinite(prediction_normals).all() or not np.isfinite(reference_normals).all():
        raise ValueError("Non-finite surface normals")
    pred_holes, ref_holes = hole_masks(prediction), hole_masks(reference)
    per_view = []
    for index, camera in enumerate(views):
        common = prediction[index] & reference[index]
        union = prediction[index] | reference[index]
        if not reference[index].any():
            raise ValueError(f"Empty reference render: {camera.name}")
        alignment = 0.0
        if common.any():
            pred, ref = prediction_normals[index][common], reference_normals[index][common]
            pred_length, ref_length = np.linalg.norm(pred, axis=1), np.linalg.norm(ref, axis=1)
            if np.any(pred_length < 1e-8) or np.any(ref_length < 1e-8):
                raise ValueError("Foreground pixel has no surface normal")
            dots = np.einsum("ij,ij->i", pred / pred_length[:, None], ref / ref_length[:, None])
            alignment = float(np.clip(dots, -1, 1).mean())
        hole_union = int(np.count_nonzero(pred_holes[index] | ref_holes[index]))
        hole_intersection = int(np.count_nonzero(pred_holes[index] & ref_holes[index]))
        per_view.append(
            {
                "view": camera.name,
                "silhouette_iou": int(common.sum()) / int(union.sum()),
                "normal_dot": alignment,
                "common_pixels": int(common.sum()),
                "union_pixels": int(union.sum()),
                "prediction_pixels": int(prediction[index].sum()),
                "reference_pixels": int(reference[index].sum()),
                "hole_iou": hole_intersection / hole_union if hole_union else None,
                "hole_union_pixels": hole_union,
                "hole_intersection_pixels": hole_intersection,
            }
        )
    silhouette = float(np.mean([view["silhouette_iou"] for view in per_view]))
    alignment = float(np.mean([view["normal_dot"] for view in per_view]))
    valid_holes = [view["hole_iou"] for view in per_view if view["hole_iou"] is not None]
    return {
        "silhouette_iou": silhouette,
        "normal_dot": alignment,
        "normal_alignment": silhouette * alignment,
        "hole_iou": float(np.mean(valid_holes)) if valid_holes else None,
        "hole_valid_views": len(valid_holes),
        "surface_common_pixels": sum(view["common_pixels"] for view in per_view),
        "surface_union_pixels": sum(view["union_pixels"] for view in per_view),
        "view_scores": per_view,
    }


def save_diagnostics(
    directory: Path,
    prediction: np.ndarray,
    reference: np.ndarray,
    prediction_normals: np.ndarray,
    reference_normals: np.ndarray,
) -> None:
    """Save verification-only PNG composites; numerical scores use float normals."""
    directory.mkdir(parents=True, exist_ok=True)
    pred_holes, ref_holes = hole_masks(prediction), hole_masks(reference)
    arrays = {}
    for name, mask in (
        ("prediction-mask", prediction),
        ("reference-mask", reference),
        ("prediction-holes", pred_holes),
        ("reference-holes", ref_holes),
    ):
        arrays[name] = np.where(mask[..., None], 0, 255).repeat(3, axis=-1).astype(np.uint8)
    for name, mask, normals in (
        ("prediction-normals", prediction, prediction_normals),
        ("reference-normals", reference, reference_normals),
    ):
        arrays[name] = np.where(mask[..., None], np.rint((np.clip(normals, -1, 1) + 1) * 127.5), 255).astype(np.uint8)
    overlay = np.full((*prediction.shape, 3), 255, dtype=np.uint8)
    overlay[prediction & reference] = (90, 170, 100)
    overlay[prediction & ~reference] = (225, 90, 90)
    overlay[reference & ~prediction] = (75, 125, 220)
    arrays["mask-overlap"] = overlay
    count, size = prediction.shape[:2]
    width = 2 if count == 4 else count
    height = 2 if count == 4 else 1
    for name, array in arrays.items():
        composite = Image.new("RGB", (width * (size + 4) + 4, height * (size + 4) + 4), "white")
        for index, pixels in enumerate(array):
            composite.paste(Image.fromarray(pixels), (4 + index % width * (size + 4), 4 + index // width * (size + 4)))
        temporary = directory / f".{name}.tmp.png"
        composite.save(temporary)
        temporary.replace(directory / f"{name}.png")
