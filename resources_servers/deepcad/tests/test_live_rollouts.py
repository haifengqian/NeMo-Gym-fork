# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch

import pytest

from resources_servers.deepcad.live_rollouts import save_rollout, write_json
from resources_servers.deepcad.rollout_viewer import RolloutHandler
from resources_servers.deepcad.tests.test_app import request, server  # noqa: F401 - shared real server fixture


def record(root: Path, identifier: str, reward: float = 0.5) -> None:
    save_rollout(
        root,
        "experiment",
        identifier,
        metadata={"task_id": "part-1", "difficulty": 8},
        inputs=[{"role": "user", "content": "fixed inputs"}],
        outputs=[{"role": "assistant", "content": "result = cq.Workplane('XY').box(1,2,3)"}],
        code="result = cq.Workplane('XY').box(1,2,3)",
        result={"reward": reward},
    )


@pytest.fixture
def viewer(tmp_path: Path):
    handler = type("TestRolloutHandler", (RolloutHandler,), {"root": tmp_path, "indexes": {}})
    http = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    thread = threading.Thread(target=http.serve_forever, daemon=True)
    thread.start()
    yield tmp_path, f"http://127.0.0.1:{http.server_port}"
    http.shutdown()
    http.server_close()
    thread.join(timeout=2)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=5) as response:
        return json.load(response)


def test_replicas_share_fixed_inputs_and_keep_separate_outputs(tmp_path: Path) -> None:
    threads = [threading.Thread(target=record, args=(tmp_path, str(index))) for index in range(20)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=5)
    assert len(list((tmp_path / "experiment/inputs").glob("*.json"))) == 1
    assert len(list((tmp_path / "experiment/rollouts").glob("*/reward.json"))) == 20
    index = [json.loads(line) for line in (tmp_path / "experiment/index.jsonl").read_text().splitlines()]
    assert len(index) == 20 and len({row["id"] for row in index}) == 20


def test_viewer_lists_live_outputs_and_keeps_partial_index_lines(viewer) -> None:
    root, url = viewer
    record(root, "a", 0.2)
    record(root, "b", 0.8)
    assert get(url + "/api/experiments") == {"experiments": ["experiment"]}
    result = get(url + "/api/rollouts/experiment?sort=highest")
    assert [row["id"] for row in result["rows"]] == ["b", "a"]
    assert get(url + "/api/rollouts/experiment?q=missing")["matched"] == 0
    detail = get(url + "/api/rollout/experiment/a")
    assert detail["reward"] == 0.2 and detail["input"][0]["content"] == "fixed inputs"
    assert "result =" in detail["code"] and not detail["has_step"]
    index = root / "experiment/index.jsonl"
    extra = json.dumps({"id": "c", "task_id": "part-1", "reward": 0.4, "created_at": 123})
    with index.open("a") as output:
        output.write(extra[:10])
    assert get(url + "/api/rollouts/experiment")["total"] == 2
    with index.open("a") as output:
        output.write(extra[10:] + "\n")
    assert get(url + "/api/rollouts/experiment")["total"] == 3


def test_viewer_blocks_paths_outside_rollout_root(viewer) -> None:
    _, url = viewer
    for route in ("/api/rollouts/%2e%2e", "/file/%2e%2e/%2e%2e/prediction.step"):
        with pytest.raises(urllib.error.HTTPError) as error:
            get(url + route)
        assert error.value.code == 404


def test_viewer_requests_prediction_preview_and_serves_completed_image(viewer) -> None:
    root, url = viewer
    record(root, "a")
    directory = root / "experiment/rollouts/a"
    (directory / "prediction.step").write_bytes(b"saved STEP")
    detail = get(url + "/api/rollout/experiment/a")
    assert detail["has_step"] and not detail["has_preview"]
    assert detail["preview_status"]["state"] == "queued"
    jobs = list((root / ".preview-requests").glob("*.json"))
    assert len(jobs) == 1 and json.loads(jobs[0].read_text()) == {"experiment": "experiment", "id": "a"}
    get(url + "/api/rollout/experiment/a")
    assert len(list((root / ".preview-requests").glob("*.json"))) == 1
    (directory / "prediction.png").write_bytes(b"rendered PNG")
    write_json(directory / "preview-status.json", {"state": "ready"})
    assert get(url + "/api/rollout/experiment/a")["has_preview"]
    with urllib.request.urlopen(url + "/file/experiment/a/prediction.png", timeout=5) as response:
        assert response.headers["Content-Type"] == "image/png"
        assert response.read() == b"rendered PNG"


def test_viewer_keeps_preview_failure_separate_from_reward(viewer) -> None:
    root, url = viewer
    record(root, "a", 0.8)
    directory = root / "experiment/rollouts/a"
    (directory / "prediction.step").write_bytes(b"saved STEP")
    write_json(directory / "preview-status.json", {"state": "failed", "message": "renderer unavailable"})
    detail = get(url + "/api/rollout/experiment/a")
    assert detail["reward"] == 0.8 and detail["has_step"] and not detail["has_preview"]
    assert not (root / ".preview-requests").exists()


def test_viewer_serves_verification_artifacts_outside_model_inputs(viewer) -> None:
    root, url = viewer
    record(root, "verification")
    directory = root / "experiment/rollouts/verification"
    for name in ("reference-mask.png", "prediction-normals.png", "reference-holes.png"):
        (directory / name).write_bytes(b"verification PNG")
    detail = get(url + "/api/rollout/experiment/verification")
    assert len(detail["verification_images"]) == 3
    assert detail["input"] == [{"role": "user", "content": "fixed inputs"}]
    for name in detail["verification_images"]:
        with urllib.request.urlopen(url + "/file/experiment/verification/" + name) as response:
            assert response.read() == b"verification PNG" and response.headers["Content-Type"] == "image/png"


def test_saved_port_is_reused_on_restart(tmp_path: Path) -> None:
    command = [
        sys.executable,
        "-m",
        "resources_servers.deepcad.rollout_viewer",
        "--root",
        str(tmp_path),
        "--host",
        "127.0.0.1",
        "--advertise-host",
        "127.0.0.1",
        "--experiment",
        "experiment",
    ]
    ports = []
    for _ in range(2):
        process = subprocess.Popen(command, stdout=subprocess.PIPE, text=True)
        try:
            info = json.loads(process.stdout.readline())
            ports.append(info["port"])
            assert get(info["url"] + "api/health")["service"] == "deepcad-rollouts"
            # A second start attaches to the existing saved viewer rather than binding another port.
            other = subprocess.run(command, capture_output=True, text=True, timeout=5, check=True)
            assert json.loads(other.stdout)["already_running"]
        finally:
            process.terminate()
            process.wait(timeout=5)
    assert ports[0] == ports[1]


@pytest.mark.asyncio
async def test_verifier_records_invalid_code_without_changing_reward(server, tmp_path: Path) -> None:
    root = tmp_path / "live"
    server.config.live_rollout_root = str(root)
    server.config.live_rollout_experiment = "test"
    write_json(root / ".viewer-state.json", {"host": "localhost", "port": 12345})
    result = await server.verify(request("result ="))
    assert result.reward == 0 and not result.mask_sample
    assert result.rollout_url.startswith("http://localhost:12345/test#")
    records = list((root / "test/rollouts").glob("*/reward.json"))
    assert len(records) == 1
    saved = json.loads(records[0].read_text())
    assert saved["failure_kind"] == "deepcad:syntax" and saved["code"] == "result ="
    with patch("resources_servers.deepcad.app.save_rollout", side_effect=OSError("disk unavailable")):
        result = await server.verify(request("result ="))
    assert result.reward == 0 and not result.mask_sample
