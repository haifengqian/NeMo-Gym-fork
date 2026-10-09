# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import cadquery as cq
import pytest

from nemo_gym.sandbox.providers.base import SandboxExecResult
from nemo_gym.server_utils import ServerClient
from resources_servers.deepcad.app import (
    DeepCADResourcesServer,
    DeepCADResourcesServerConfig,
    DeepCADVerifyRequest,
    RewardWeights,
    assistant_text,
    extract_code,
)
from resources_servers.deepcad.geometry import load_solid, score


def request(text: str) -> DeepCADVerifyRequest:
    return DeepCADVerifyRequest.model_validate(
        {
            "responses_create_params": {"input": [{"role": "user", "content": "three views"}]},
            "verifier_metadata": {"task_id": "box", "reference_step": "box.step"},
            "response": {
                "id": "resp_test",
                "created_at": 0,
                "model": "test",
                "object": "response",
                "output": [
                    {
                        "id": "msg",
                        "type": "message",
                        "role": "assistant",
                        "status": "completed",
                        "content": [{"type": "output_text", "text": text, "annotations": []}],
                    }
                ],
                "parallel_tool_calls": False,
                "tool_choice": "auto",
                "tools": [],
            },
        }
    )


@pytest.fixture
def server(tmp_path: Path) -> DeepCADResourcesServer:
    cq.exporters.export(cq.Workplane("XY").box(1, 2, 3), str(tmp_path / "box.step"))
    config = DeepCADResourcesServerConfig(
        host="0.0.0.0",
        port=8080,
        entrypoint="app.py",
        name="deepcad",
        reference_root=str(tmp_path),
        sandbox_provider={"docker": {}},
        voxel_resolution=64,
    )
    client = MagicMock(spec=ServerClient)
    client.global_config_dict = {}
    return DeepCADResourcesServer(config=config, server_client=client)


@pytest.mark.parametrize(
    "text, expected",
    [
        ("result = cq.Workplane('XY').box(1,2,3)", "result = cq.Workplane('XY').box(1,2,3)"),
        ("prose\n```python\nresult = 1\n```", "result = 1"),
        ("```py\nresult = 1\n```\n```python\nresult = 2\n```", "result = 2"),
        ("<think>```python\nbad\n```</think>\n```python\ngood\n```", "good"),
        ("hidden reasoning</thinking>\n```\ngood\n```", "good"),
        ("<thinking>secret</thinking>good", "good"),
        ("", ""),
    ],
)
def test_extract_code(text: str, expected: str) -> None:
    assert extract_code(text) == expected


def test_last_assistant_message() -> None:
    body = request("first")
    second = body.response.output[0].model_copy(deep=True)
    second.content[0].text = "last"
    body.response.output.append(second)
    assert assistant_text(body) == "last"
    body.response.output.clear()
    assert assistant_text(body) == ""


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "text, kind", [("", "code"), ("if:", "syntax"), ("x" * 65537, "code")], ids=["empty", "syntax", "oversized"]
)
async def test_invalid_code_zero(server: DeepCADResourcesServer, text: str, kind: str) -> None:
    response = await server.verify(request(text))
    assert response.reward == 0 and not response.mask_sample
    assert response.failure_kind == f"deepcad:{kind}"
    assert set(response.reward_components) == {"voxel_iou", "silhouette_iou", "normal_alignment", "hole_iou"}
    assert sum(response.reward_components.values()) == 0


@pytest.mark.asyncio
async def test_continuous_reward(server: DeepCADResourcesServer) -> None:
    with patch.object(DeepCADResourcesServer, "_evaluate", AsyncMock(return_value={"reward": 0.37})):
        assert (await server.verify(request("result = 1"))).reward == 0.37


@pytest.mark.asyncio
@pytest.mark.parametrize("reference", ["missing.step", "../outside.step"])
async def test_reference_mask(server: DeepCADResourcesServer, reference: str) -> None:
    result = await server._evaluate("result=1", reference)
    assert result["mask_sample"] and result["reward"] == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "probe, execution, check, kind, mask",
    [
        (SandboxExecResult("", "", 1), None, None, "infrastructure", True),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 1), None, "execution", False),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 124, "timeout"), None, "execution", False),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 125, "runtime"), None, "infrastructure", True),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 0), SandboxExecResult("0", "", 0), "artifact", False),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("huge", "", 0),
            "artifact",
            False,
        ),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("999999999", "", 0),
            "artifact",
            False,
        ),
        (
            SandboxExecResult("", "", 0),
            SandboxExecResult("", "", 0),
            SandboxExecResult("100", "", 1),
            "artifact",
            False,
        ),
        (SandboxExecResult("", "", 0), SandboxExecResult("", "", 0), SandboxExecResult("100", "", 0), None, False),
    ],
)
async def test_sandbox_outcomes(server: DeepCADResourcesServer, probe, execution, check, kind, mask) -> None:
    sandbox = AsyncMock()
    sandbox.exec.side_effect = [probe, execution, check]
    context = MagicMock()
    context.__aenter__ = AsyncMock(return_value=sandbox)
    context.__aexit__ = AsyncMock(return_value=False)
    with patch("resources_servers.deepcad.app.AsyncSandbox", return_value=context) as factory:
        with patch.object(DeepCADResourcesServer, "_score", AsyncMock(return_value={"reward": 0.82})):
            result = await server._evaluate("result=1", "box.step")
    assert result.get("mask_sample", False) is mask
    if kind:
        assert result["reward"] == 0 and result["failure_kind"] == f"deepcad:{kind}"
    else:
        assert result["reward"] == 0.82
        sandbox.download.assert_awaited_once()
    spec = factory.call_args.args[1]
    assert set(spec.files) == {"/sandbox/model.py", "/sandbox/runner.py"}
    assert "box.step" not in str(spec.files)
    assert spec.env.get("NEMO_GYM_CONFIG_DICT", "") == ""
    assert all(
        not spec.env[key]
        for key in os.environ
        if key not in {"PATH", "HOME", "LANG", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "ENROOT_MOUNT_HOME"}
    )


def test_identical_translated_scaled_and_wrong_orientation() -> None:
    reference = cq.Workplane("XY").box(1, 2, 3).val()
    result = score(reference, reference)
    assert result["reward"] == pytest.approx(1)
    assert result["scoring_method"] == "voxel_iou" and result["voxel_resolution"] == 64
    moved = reference.scale(2.5).translate((9, -4, 3))
    assert score(moved, reference)["reward"] == pytest.approx(1)
    rotated = reference.rotate((0, 0, 0), (0, 0, 1), 90)
    assert 0.25 < score(rotated, reference)["reward"] < 0.45
    wrong = cq.Workplane("XY").sphere(1).val()
    assert 0 < score(wrong, reference)["reward"] < 0.85


def test_load_solid_rejects_shell(tmp_path: Path) -> None:
    path = tmp_path / "face.step"
    cq.exporters.export(cq.Workplane("XY").rect(1, 1).wire().val(), str(path))
    with pytest.raises(ValueError, match="solids"):
        load_solid(path)


@pytest.mark.asyncio
async def test_real_worker(server: DeepCADResourcesServer) -> None:
    path = Path(server.config.reference_root) / "box.step"
    assert (await server._score(path, path))["reward"] == pytest.approx(1)
    assert (await server._score(path, path.parent / "missing.step"))["mask_sample"]
    result = await server._score(path.parent / "missing.step", path)
    assert result["reward"] == 0 and not result.get("mask_sample", False)


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["axis", "diagonal"])
async def test_real_multireward_worker(server: DeepCADResourcesServer, tmp_path: Path, mode: str) -> None:
    server.config.render_rewards = True
    server.config.reward_weights = RewardWeights(voxel_iou=2, silhouette_iou=3, normal_alignment=4, hole_iou=5)
    server.config.reference_cache_root = str(tmp_path / "cache")
    path = Path(server.config.reference_root) / "box.step"
    result = await server._score(path, path, view_mode=mode, artifact_dir=tmp_path / "diagnostics")
    assert result["voxel_iou"] == result["silhouette_iou"] == 1
    assert result["normal_alignment"] == pytest.approx(1)
    assert result["hole_iou"] is None and result["hole_valid_views"] == 0
    assert result["reward"] == pytest.approx(9)
    assert result["reward_components"] == {"voxel_iou": 2, "silhouette_iou": 3, "normal_alignment": 4, "hole_iou": 0}
    assert not result["reference_cache_hit"]
    assert (await server._score(path, path, view_mode=mode))["reference_cache_hit"]
    assert len(list((tmp_path / "diagnostics").glob("*.png"))) == 7


def test_config_rejects_disabled_render_rewards_with_nonzero_weights() -> None:
    with pytest.raises(ValueError, match="render_rewards=true"):
        DeepCADResourcesServerConfig(
            host="localhost", port=8080, entrypoint="app.py", name="deepcad", reward_weights={"silhouette_iou": 1}
        )
    with pytest.raises(ValueError):
        RewardWeights(normal_alignment=-1)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["timeout", "crash", "cancel", "bad_json"])
async def test_worker_cleanup(server: DeepCADResourcesServer, outcome: str) -> None:
    process = MagicMock()
    process.returncode = 1 if outcome == "crash" else 0
    process.communicate = AsyncMock()
    if outcome == "timeout":
        process.communicate.side_effect = [TimeoutError(), (b"", b"")]
        expected = TimeoutError
    elif outcome == "cancel":
        process.communicate.side_effect = [asyncio.CancelledError(), (b"", b"")]
        expected = asyncio.CancelledError
    else:
        process.communicate.return_value = (b"not-json", b"failed\xff")
        expected = RuntimeError if outcome == "crash" else json.JSONDecodeError
    with patch("asyncio.create_subprocess_exec", AsyncMock(return_value=process)):
        with pytest.raises(expected):
            await server._score(Path("a"), Path("b"))
    if outcome in ("timeout", "cancel"):
        process.kill.assert_called_once()
