# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from pathlib import Path

import cadquery as cq
import numpy as np
from PIL import Image

from resources_servers.deepcad.live_rollouts import save_rollout, write_json
from resources_servers.deepcad.preview_worker import process_request, render_preview


def test_preview_renders_saved_geometry_with_white_background(tmp_path: Path) -> None:
    step = tmp_path / "prediction.step"
    output = tmp_path / "prediction.png"
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(step))
    render_preview(step, output, mode="axis", color="blue")
    with Image.open(output) as image:
        assert image.size == (784, 264)
        pixels = np.array(image)
    assert np.all(pixels[:4] == 255)
    for offset in (4, 264, 524):
        view = pixels[4:260, offset : offset + 256]
        assert np.any(view < 255)
        assert np.any(view[:, :, 2] > view[:, :, 0])
    assert not list(tmp_path.glob(".preview-*"))


def test_invalid_step_preview_preserves_the_recorded_reward(tmp_path: Path) -> None:
    save_rollout(
        tmp_path,
        "experiment",
        "rollout",
        metadata={"task_id": "part", "view_mode": "axis", "foreground_color": "blue"},
        inputs=[],
        outputs=[],
        code="saved program",
        result={"reward": 0.8},
    )
    directory = tmp_path / "experiment/rollouts/rollout"
    reward_before = (directory / "reward.json").read_bytes()
    (directory / "prediction.step").write_text("invalid STEP")
    request = tmp_path / ".preview-requests/request.json"
    write_json(request, {"experiment": "experiment", "id": "rollout"})
    process_request(tmp_path, request)
    assert json.loads((directory / "preview-status.json").read_text())["state"] == "failed"
    assert (directory / "reward.json").read_bytes() == reward_before
    assert not request.exists() and not (directory / "prediction.png").exists()
