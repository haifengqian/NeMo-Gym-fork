# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import cadquery as cq
import numpy as np
import pytest

from resources_servers.deepcad import geometry, runner


@pytest.mark.parametrize(
    "code, valid",
    [
        ("import cadquery as cq\nresult=cq.Workplane('XY').box(1,2,3)", True),
        ("result=cq.Workplane('XY').box(1,2,3).val()", True),
        ("result=cq.Workplane('XY')", False),
        ("result=42", False),
        ("raise RuntimeError('bad program')", False),
        ("result=cq.Workplane('XY').rect(1,2).wire()", False),
    ],
)
def test_runner_solid_contract(tmp_path: Path, code: str, valid: bool) -> None:
    (tmp_path / "model.py").write_text(code)
    output = tmp_path / "prediction.step"
    real_export = cq.exporters.export
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with patch.object(runner.cq.exporters, "export", side_effect=lambda shape, _: real_export(shape, str(output))):
            if valid:
                runner.main()
                assert geometry.load_solid(output).Volume() == pytest.approx(6)
            else:
                with pytest.raises((ValueError, RuntimeError)):
                    runner.main()
                assert not output.exists()


def test_runner_nonfinite_volume(tmp_path: Path) -> None:
    (tmp_path / "model.py").write_text("result=cq.Workplane('XY').box(1,2,3).val()")
    with patch.object(runner, "Path", side_effect=lambda path: tmp_path / Path(path).name):
        with patch.object(cq.Shape, "Volume", return_value=float("nan")):
            with pytest.raises(ValueError, match="finite volume"):
                runner.main()


@pytest.mark.parametrize("missing", [None, "reference", "prediction"])
def test_worker_protocol(tmp_path: Path, capsys, missing: str | None) -> None:
    reference, prediction = tmp_path / "reference.step", tmp_path / "prediction.step"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(reference))
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(prediction))
    if missing == "reference":
        reference = tmp_path / "missing.step"
    elif missing == "prediction":
        prediction = tmp_path / "missing.step"
    with patch("sys.argv", ["geometry", str(prediction), str(reference), "256", "10"]):
        geometry.main()
    result = json.loads(capsys.readouterr().out)
    if missing:
        assert result["reward"] == 0
        assert result.get("mask_sample", False) == (missing == "reference")
    else:
        assert result["reward"] == pytest.approx(1)


def test_load_invalid_solid() -> None:
    imported = MagicMock()
    imported.val.return_value.Solids.return_value = [1]
    imported.val.return_value.Volume.return_value = 1
    imported.val.return_value.isValid.return_value = False
    with patch.object(cq.importers, "importStep", return_value=imported):
        with pytest.raises(ValueError, match="valid solids"):
            geometry.load_solid(Path("invalid.step"))


def test_zero_area_surface() -> None:
    shape = MagicMock()
    shape.BoundingBox.return_value.DiagonalLength = 1
    shape.tessellate.return_value = ([cq.Vector(i, 0, 0) for i in range(3)], [(0, 1, 2)])
    with pytest.raises(ValueError, match="sample"):
        geometry.surface_points(shape, 32)


def test_nonfinite_iou() -> None:
    shape = cq.Workplane("XY").box(1, 2, 3).val()
    invalid = MagicMock()
    invalid.Volume.return_value = 6.0
    invalid.intersect.return_value.Volume.return_value = float("nan")
    with patch.object(geometry, "alignment", return_value=(np.eye(4), 0)):
        with patch.object(cq.Shape, "cast", return_value=invalid):
            with pytest.raises(ValueError, match="Non-finite"):
                geometry.score(shape, shape, samples=32)
