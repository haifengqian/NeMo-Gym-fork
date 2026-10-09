# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import patch

import cadquery as cq
import numpy as np
import pytest

from resources_servers.deepcad.geometry import aggregate_rewards, main, normalized_mesh, reference_maps
from resources_servers.deepcad.surface import Camera, cameras, compare_maps, hole_masks, render_maps


@pytest.mark.parametrize("mode", ["axis", "diagonal"])
def test_actual_solid_render_identity_and_scale(mode: str) -> None:
    part = cq.Workplane("XY").box(1, 2, 3).val()
    reference = normalized_mesh(part)
    views = cameras(reference, mode=mode)
    mask, normals = render_maps(reference, views)
    moved, moved_normals = render_maps(normalized_mesh(part.scale(1000).translate((7, -18, 50))), views)
    assert mask.shape == (3 if mode == "axis" else 4, 256, 256)
    np.testing.assert_array_equal(mask, moved)
    np.testing.assert_allclose(normals, moved_normals)
    np.testing.assert_allclose(np.linalg.norm(normals[mask], axis=1), 1, atol=1e-6)
    result = compare_maps(moved, mask, moved_normals, normals, views)
    assert result["silhouette_iou"] == pytest.approx(1)
    assert result["normal_dot"] == pytest.approx(1)
    assert result["normal_alignment"] == pytest.approx(1)
    assert result["hole_iou"] is None and result["hole_valid_views"] == 0
    if mode == "axis":
        for index, expected in enumerate(((0, -1, 0), (1, 0, 0), (0, 0, 1))):
            np.testing.assert_allclose(normals[index, 128, 128], expected)


def test_actual_hole_and_wrong_orientation() -> None:
    box = cq.Workplane("XY").box(1, 1, 1).val()
    ring = cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().hole(0.5).val()
    mesh = normalized_mesh(ring)
    views = cameras(mesh, mode="axis")
    reference, ref_normals = render_maps(mesh, views)
    assert reference[0, 128, 128] and not reference[2, 128, 128]
    holes = hole_masks(reference)
    assert holes[2, 128, 128] and not holes[:2].any()
    same = compare_maps(reference, reference, ref_normals, ref_normals, views)
    assert same["hole_iou"] == 1 and same["hole_valid_views"] == 1
    prediction, pred_normals = render_maps(normalized_mesh(box), views)
    missing = compare_maps(prediction, reference, pred_normals, ref_normals, views)
    assert missing["hole_iou"] == 0 and missing["hole_valid_views"] == 1
    assert missing["silhouette_iou"] < 1
    rotated = ring.rotate((0, 0, 0), (0, 1, 0), 90)
    prediction, pred_normals = render_maps(normalized_mesh(rotated), views)
    result = compare_maps(prediction, reference, pred_normals, ref_normals, views)
    assert result["silhouette_iou"] < 0.95 and result["normal_alignment"] < 1


def test_subpixel_thin_solid_has_nonempty_axis_views() -> None:
    mesh = normalized_mesh(cq.Workplane("XY").box(1, 1, 0.00001).val())
    views = cameras(mesh, mode="axis")
    masks, normals = render_maps(mesh, views)
    assert masks.any(axis=(1, 2)).all()
    result = compare_maps(masks, masks, normals, normals, views)
    assert result["normal_alignment"] == pytest.approx(1)


def test_background_components_at_every_edge_are_excluded() -> None:
    mask = np.ones((1, 16, 16), dtype=bool)
    mask[0, 5:9, 5:9] = False
    for y, x in ((0, 12), (15, 2), (2, 0), (12, 15)):
        mask[0, y, x] = False
    holes = hole_masks(mask)
    assert holes.sum() == 16 and holes[0, 6, 6]
    opened = mask.copy()
    opened[0, :6, 5] = False
    assert not hole_masks(opened).any()
    assert not hole_masks(np.zeros_like(mask)).any()


def test_diagonal_background_connection_is_not_a_hole() -> None:
    mask = np.ones((1, 5, 5), dtype=bool)
    mask[0, np.arange(3), np.arange(3)] = False
    assert not hole_masks(mask).any()


def artificial_views(count: int) -> tuple[Camera, ...]:
    return tuple(Camera(str(i), (0, 0, 1), (1, 0, 0), (0, 1, 0), 1) for i in range(count))


def test_tiny_overlap_does_not_receive_large_normal_reward() -> None:
    reference = np.ones((3, 10, 10), dtype=bool)
    prediction = np.zeros_like(reference)
    prediction[:, :1] = True
    normals = np.zeros((*reference.shape, 3), dtype=np.float32)
    normals[..., 2] = 1
    result = compare_maps(prediction, reference, normals, normals, artificial_views(3))
    assert result["normal_dot"] == 1
    assert result["silhouette_iou"] == pytest.approx(0.1)
    assert result["normal_alignment"] == pytest.approx(0.1)
    reversed_result = compare_maps(prediction, reference, -normals, normals, artificial_views(3))
    assert reversed_result["normal_dot"] == -1 and reversed_result["normal_alignment"] == pytest.approx(-0.1)
    prediction[:] = False
    empty = compare_maps(prediction, reference, normals, normals, artificial_views(3))
    assert empty["normal_dot"] == empty["normal_alignment"] == empty["silhouette_iou"] == 0


def test_hole_iou_only_averages_nonempty_unions() -> None:
    ref = np.ones((4, 12, 12), dtype=bool)
    pred = ref.copy()
    ref[0, 3:7, 3:7] = pred[0, 3:7, 3:7] = False
    ref[1, 3:7, 3:7] = False
    normals = np.zeros((*ref.shape, 3), dtype=np.float32)
    normals[..., 2] = 1
    result = compare_maps(pred, ref, normals, normals, artificial_views(4))
    assert result["hole_iou"] == 0.5 and result["hole_valid_views"] == 2
    assert [view["hole_iou"] for view in result["view_scores"]] == [1, 0, None, None]
    # An invented hole is also scored, even if the reference has none.
    result = compare_maps(ref, pred, normals, normals, artificial_views(4))
    assert result["hole_iou"] == 0.5


def test_rewards_are_independent_and_sum_to_scalar() -> None:
    raw = {"voxel_iou": 0.8, "silhouette_iou": 0.6, "normal_alignment": 0.3, "hole_iou": None}
    result = aggregate_rewards(raw.copy(), dict.fromkeys(raw, 1.0))
    assert result["reward"] == pytest.approx(1.7) and result["reward_components"]["hole_iou"] == 0
    result = aggregate_rewards(raw.copy(), dict(zip(raw, (2, 0, 3, 4), strict=True)))
    assert result["reward"] == pytest.approx(2.5)
    assert result["reward_components"]["normal_alignment"] == pytest.approx(0.9)
    assert result["silhouette_iou"] == 0.6 and result["hole_iou"] is None
    with pytest.raises(ValueError):
        aggregate_rewards(raw.copy(), dict.fromkeys(raw, float("nan")))


def test_cache_content_and_settings_invalidation(tmp_path: Path) -> None:
    path = tmp_path / "reference.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(path))
    arguments = {"mode": "axis", "resolution": 16, "render_resolution": 64, "cache_root": tmp_path / "cache"}
    first = reference_maps(path, **arguments)
    second = reference_maps(path, **arguments)
    assert not first[-1] and second[-1]
    for a, b in zip(first[:3], second[:3], strict=True):
        np.testing.assert_array_equal(a, b)
    cq.exporters.export(cq.Workplane("XY").box(2, 2, 3), str(path))
    assert not reference_maps(path, **arguments)[-1]
    assert not reference_maps(path, **{**arguments, "mode": "diagonal"})[-1]
    assert not reference_maps(path, **{**arguments, "render_resolution": 32})[-1]
    assert len(list((tmp_path / "cache").glob("*.npz"))) == 4


def test_worker_render_protocol_and_verification_artifacts(tmp_path: Path, capsys) -> None:
    path = tmp_path / "part.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 1, 1).faces(">Z").workplane().hole(0.5), str(path))
    options = {"render_rewards": True, "view_mode": "diagonal", "artifact_dir": str(tmp_path / "artifacts")}
    with patch("sys.argv", ["geometry", str(path), str(path), "32", json.dumps(options)]):
        main()
    result = json.loads(capsys.readouterr().out)
    assert result["reward"] == pytest.approx(sum(result["reward_components"].values()))
    assert result["voxel_iou"] == result["silhouette_iou"] == 1
    assert result["normal_alignment"] == pytest.approx(1)
    assert len(list((tmp_path / "artifacts").glob("*.png"))) == 7


def test_foreground_requires_valid_normals() -> None:
    mask = np.ones((1, 8, 8), dtype=bool)
    normals = np.zeros((*mask.shape, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="no surface normal"):
        compare_maps(mask, mask, normals, normals, artificial_views(1))
