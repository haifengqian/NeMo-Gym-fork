# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import copy
import json
from pathlib import Path
from unittest.mock import patch

import cadquery as cq
import numpy as np
import pytest
from PIL import Image

from resources_servers.deepcad.dataset.prepare_gym import (
    construction_metrics,
    dimension_data,
    main,
    rank_split,
    render_views,
)


def document(holes: int = 0) -> dict:
    loops = [{"profile_curves": [{"type": "Line3D"}] * 4}]
    loops += [{"profile_curves": [{"type": "Circle3D"}]} for _ in range(holes)]
    return {
        "sequence": [
            {"type": "Sketch", "entity": "s"},
            {"type": "ExtrudeFeature", "entity": "e"},
        ],
        "entities": {
            "s": {"profiles": {"p": {"loops": loops}, "unused": {"loops": loops * 100}}},
            "e": {
                "operation": "NewBodyFeatureOperation",
                "profiles": [{"sketch": "s", "profile": "p"}],
            },
        },
    }


def test_single_extrusion_many_holes_and_unused_profile() -> None:
    counts = construction_metrics(document(8))
    assert counts == {
        "extrusions": 1,
        "sketch_curves": 12,
        "additional_loops": 8,
        "cuts_and_intersections": 0,
        "arcs_and_circles": 8,
        "difficulty": 32,
    }


def test_reused_profile_cut_and_intersection() -> None:
    model = document()
    for name, operation in [
        ("cut", "CutFeatureOperation"),
        ("intersect", "IntersectFeatureOperation"),
    ]:
        model["entities"][name] = {
            **copy.deepcopy(model["entities"]["e"]),
            "operation": operation,
        }
        model["sequence"].append({"type": "ExtrudeFeature", "entity": name})
    counts = construction_metrics(model)
    assert counts["extrusions"] == 3
    assert counts["sketch_curves"] == 12
    assert counts["cuts_and_intersections"] == 2
    assert counts["difficulty"] == 32


def test_arc_count_has_no_extra_weight() -> None:
    model = document()
    model["entities"]["s"]["profiles"]["p"]["loops"][0]["profile_curves"][0] = {"type": "Arc3D"}
    counts = construction_metrics(model)
    assert counts["arcs_and_circles"] == 1 and counts["difficulty"] == 8


def test_multi_profile_feature() -> None:
    model = document()
    model["entities"]["s"]["profiles"]["p2"] = {"loops": [{"profile_curves": [{"type": "Circle3D"}]}]}
    model["entities"]["e"]["profiles"].append({"sketch": "s", "profile": "p2"})
    counts = construction_metrics(model)
    assert counts["extrusions"] == 1 and counts["sketch_curves"] == 5 and counts["difficulty"] == 9


def test_ranking_entire_split_and_stable_ties(tmp_path: Path) -> None:
    (tmp_path / "cad_json").mkdir()
    for identifier, holes in [("hard", 8), ("b", 0), ("a", 0)]:
        (tmp_path / "cad_json" / f"{identifier}.json").write_text(json.dumps(document(holes)))
    ranked, failures = rank_split(tmp_path, ["hard", "b", "missing", "a"])
    assert [record["task_id"] for record in ranked] == ["a", "b", "hard"]
    assert [record["difficulty"] for record in ranked] == [8, 8, 32]
    assert failures[0]["task_id"] == "missing"


def test_cli_sorts_before_limit_and_preserves_images(tmp_path: Path) -> None:
    (tmp_path / "cad_json").mkdir()
    for identifier, holes in [("hard", 8), ("easy", 0)]:
        (tmp_path / "cad_json" / f"{identifier}.json").write_text(json.dumps(document(holes)))
    (tmp_path / "train_val_test_split.json").write_text(json.dumps({"validation": ["hard", "easy"]}))
    image = tmp_path / "input.png"
    Image.new("RGB", (16, 16)).save(image)
    output = tmp_path / "tasks.jsonl"
    args = [
        "prepare_gym",
        "--data-root",
        str(tmp_path),
        "--output",
        str(output),
        "--reference-root",
        str(tmp_path / "refs"),
        "--limit",
        "1",
    ]
    with (
        patch("sys.argv", args),
        patch(
            "resources_servers.deepcad.dataset.prepare_gym.reconstruct",
            return_value=cq.Workplane("XY").box(1, 2, 3).val(),
        ),
        patch(
            "resources_servers.deepcad.dataset.prepare_gym.render_views",
            return_value={k: image for k in ("front", "side", "top")},
        ),
    ):
        main()
    row = json.loads(output.read_text())
    assert row["verifier_metadata"]["task_id"] == "easy"
    assert row["verifier_metadata"]["difficulty"] == 8
    content = row["responses_create_params"]["input"][1]["content"]
    assert len([item for item in content if item["type"] == "input_image"]) == 3
    assert all(item["detail"] == "high" for item in content if item["type"] == "input_image")
    gym = pytest.importorskip("nemo_gym.base_resources_server")
    gym.BaseRunRequest.model_validate(row)
    assert "reference_step" not in json.dumps(row["responses_create_params"])
    assert len(output.with_suffix(".difficulty.jsonl").read_text().splitlines()) == 2


def test_manifest_requires_cad_history(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"task_id": "example"}) + "\n")
    with (
        patch(
            "sys.argv",
            [
                "prepare",
                "--manifest",
                str(manifest),
                "--output",
                str(tmp_path / "out.jsonl"),
                "--reference-root",
                str(tmp_path / "refs"),
            ],
        ),
        pytest.raises(ValueError, match="cad_json"),
    ):
        main()


def test_dimensions_use_final_geometry_and_convert_metres_to_mm(tmp_path: Path) -> None:
    # A drilled box exercises true radius and overall dimensions, with cap-edge duplicates.
    shape = cq.Workplane("XY").box(0.02, 0.03, 0.004).faces(">Z").workplane().hole(0.006).val()
    measured = dimension_data(shape)
    np.testing.assert_allclose(measured["size_mm"], [20, 30, 4], atol=1e-5)
    assert {round(c["radius"] * 1000, 6) for c in measured["circles"]} == {3}
    paths = render_views(shape, tmp_path)
    assert set(paths) == {"front", "side", "top"}
    for name, path in paths.items():
        image = Image.open(path)
        assert image.size == (640, 640)
        pixels = np.asarray(image)
        # View title, overall size labels and radius leaders occupy reserved margins.
        assert np.any(pixels[10:40, 10:530] < 100)
        assert np.any(pixels[565:605, 100:540] < 100)
        assert np.any(pixels[110:530, 565:610] < 100)
        if name == "top":
            assert np.any(pixels[40:80, 10:600] < 100)


def test_plate_depth_with_mislabeled_outer_loops() -> None:
    model = document()
    points = [(0, 0), (0.02, 0), (0.02, 0.03), (0, 0.03)]
    curves = [
        {
            "type": "Line3D",
            "start_point": {"x": a[0], "y": a[1], "z": 0},
            "end_point": {"x": b[0], "y": b[1], "z": 0},
        }
        for a, b in zip(points, points[1:] + points[:1])
    ]
    model["entities"]["s"]["profiles"]["p"]["loops"] = [
        {
            "is_outer": True,
            "profile_curves": [
                {
                    "type": "Circle3D",
                    "radius": 0.003,
                    "center_point": {"x": 0.01, "y": 0.01, "z": 0},
                }
            ],
        },
        {"is_outer": True, "profile_curves": curves},
    ]
    model["entities"]["s"]["transform"] = {
        "origin": {"x": 0, "y": 0, "z": 0},
        "z_axis": {"x": 0, "y": 0, "z": 1},
    }
    model["entities"]["e"].update(
        {
            "extent_type": "OneSideFeatureExtentType",
            "extent_one": {"distance": {"value": 0.004}},
        }
    )
    measurements = dimension_data(cq.Workplane("XY").box(0.02, 0.03, 0.004).val(), model)
    assert len(measurements["plates"]) == 1
    assert measurements["plates"][0]["depth_mm"] == 4
    assert measurements["plates"][0]["axis"] == 2
