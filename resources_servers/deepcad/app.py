# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Orthographic CAD reconstruction environment for NeMo Gym."""

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import tempfile
import uuid
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.openai_utils import TOKEN_METADATA_FIELDS
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec
from resources_servers.deepcad.live_rollouts import save_rollout
from resources_servers.deepcad.task_data import TaskData


LOGGER = logging.getLogger(__name__)
HERE = Path(__file__).resolve().parent


class DeepCADResourcesServerConfig(BaseResourcesServerConfig):
    reference_root: str = "resources_servers/deepcad/data"
    sandbox_provider: str | dict[str, Any] = "sandbox"
    sandbox_image: str = "deepcad-cadquery:2.8.0"
    sandbox_python: str = "python"
    max_concurrency: int = Field(default=4, ge=1)
    execution_timeout_s: int = Field(default=60, ge=1)
    geometry_timeout_s: int = Field(default=120, ge=1)
    max_code_bytes: int = Field(default=65536, ge=1)
    max_step_bytes: int = Field(default=32 * 1024 * 1024, ge=1)
    voxel_resolution: int = Field(default=64, ge=8, le=256)
    live_rollout_root: str | None = None
    live_rollout_experiment: str = Field(default="deepcad", pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


class DeepCADVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class DeepCADVerifyResponse(BaseVerifyResponse):
    scoring_method: str | None = None
    voxel_resolution: int | None = None
    intersection_voxels: int | None = None
    union_voxels: int | None = None
    prediction_voxels: int | None = None
    reference_voxels: int | None = None
    rollout_url: str | None = None


def extract_code(text: str) -> str:
    """Accept Python fences or raw code; ignore model reasoning blocks."""
    text = re.sub(r"<think(?:ing)?>.*?</think(?:ing)?>", "", text, flags=re.S | re.I)
    text = re.split(r"</think(?:ing)?>", text, flags=re.I)[-1]
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, flags=re.S | re.I)
    return (blocks[-1] if blocks else text).strip()


def assistant_text(body: BaseVerifyRequest) -> str:
    messages = [item for item in body.response.output if item.type == "message" and item.role == "assistant"]
    if not messages:
        return ""
    return "\n".join(part.text for part in messages[-1].content if part.type == "output_text")


class DeepCADResourcesServer(SimpleResourcesServer):
    config: DeepCADResourcesServerConfig
    _semaphore: asyncio.Semaphore = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        self._semaphore = asyncio.Semaphore(self.config.max_concurrency)
        if self.config.live_rollout_root:
            (Path(self.config.live_rollout_root).expanduser() / self.config.live_rollout_experiment).mkdir(
                parents=True, exist_ok=True
            )

    async def verify(self, body: DeepCADVerifyRequest) -> DeepCADVerifyResponse:
        """Execute code in isolation, then score the exported solid in a trusted worker."""
        result: dict[str, Any]
        rollout_id = uuid.uuid4().hex if self.config.live_rollout_root else None
        code = extract_code(assistant_text(body))
        if not code or len(code.encode()) > self.config.max_code_bytes:
            result = {"reward": 0.0, "failure_kind": "deepcad:code", "failure_reason": "Empty or oversized code"}
        else:
            try:
                compile(code, "model.py", "exec")
            except SyntaxError as exc:
                result = {"reward": 0.0, "failure_kind": "deepcad:syntax", "failure_reason": str(exc)}
            else:
                async with self._semaphore:
                    if rollout_id:
                        result = await self._evaluate(
                            code, body.verifier_metadata.reference_step, rollout_id=rollout_id
                        )
                    else:
                        result = await self._evaluate(code, body.verifier_metadata.reference_step)
        if rollout_id:
            try:
                url = await asyncio.to_thread(
                    save_rollout,
                    Path(self.config.live_rollout_root).expanduser(),
                    self.config.live_rollout_experiment,
                    rollout_id,
                    metadata=body.verifier_metadata.model_dump(mode="json"),
                    inputs=body.responses_create_params.model_dump(mode="json")["input"],
                    outputs=[
                        item.model_dump(mode="json", exclude=TOKEN_METADATA_FIELDS) for item in body.response.output
                    ],
                    code=code,
                    result=result,
                )
                if url:
                    result["rollout_url"] = url
            except (OSError, ValueError, KeyError):
                LOGGER.exception("Could not save DeepCAD live rollout")
        return DeepCADVerifyResponse(**body.model_dump(), **result)

    async def _evaluate(self, code: str, reference_step: str, *, rollout_id: str | None = None) -> dict[str, Any]:
        root = Path(self.config.reference_root).expanduser().resolve()
        reference = (root / reference_step).resolve()
        if not reference.is_relative_to(root) or not reference.is_file():
            return {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "deepcad:reference",
                "failure_reason": "Reference STEP missing or outside reference_root",
            }
        try:
            provider = resolve_provider_config(self.config.sandbox_provider, self.server_client.global_config_dict)
            spec = SandboxSpec(
                image=self.config.sandbox_image,
                workdir="/sandbox",
                env={
                    # Enroot can inherit the launcher's environment, including serialized Gym config.
                    **dict.fromkeys(os.environ, ""),
                    "PATH": "/usr/local/bin:/usr/bin:/bin",
                    "HOME": "/tmp",
                    "LANG": "C.UTF-8",
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "ENROOT_MOUNT_HOME": "no",
                },
                resources=SandboxResources(cpu=1, memory_mib=2048),
                files={"/sandbox/model.py": code, "/sandbox/runner.py": (HERE / "runner.py").read_text()},
            )
            with tempfile.TemporaryDirectory(prefix="deepcad-verify-") as directory:
                prediction = Path(directory) / "prediction.step"
                async with AsyncSandbox(provider, spec) as sandbox:
                    await sandbox.start()
                    python = shlex.quote(self.config.sandbox_python)
                    probe = await sandbox.exec(f"{python} -c 'import cadquery'", timeout_s=30)
                    if probe.return_code or probe.error_type:
                        raise RuntimeError("Sandbox image cannot import CadQuery")
                    execution = await sandbox.exec(
                        f"{python} /sandbox/runner.py > /sandbox/run.log 2>&1",
                        timeout_s=self.config.execution_timeout_s,
                    )
                    if execution.error_type and execution.error_type != "timeout":
                        raise RuntimeError(f"Sandbox execution failed: {execution.error_type}")
                    if execution.return_code or execution.error_type:
                        return {
                            "reward": 0.0,
                            "failure_kind": "deepcad:execution",
                            "failure_reason": "Code timed out or did not produce valid solid geometry",
                        }
                    check = await sandbox.exec(
                        "test -f /sandbox/prediction.step && wc -c < /sandbox/prediction.step", timeout_s=10
                    )
                    try:
                        size = int((check.stdout or "").strip())
                    except ValueError:
                        size = 0
                    if check.return_code or not 0 < size <= self.config.max_step_bytes:
                        return {
                            "reward": 0.0,
                            "failure_kind": "deepcad:artifact",
                            "failure_reason": "Missing or oversized STEP artifact",
                        }
                    await sandbox.download("/sandbox/prediction.step", prediction)
                if rollout_id and self.config.live_rollout_root:
                    try:
                        artifact = (
                            Path(self.config.live_rollout_root).expanduser()
                            / self.config.live_rollout_experiment
                            / "rollouts"
                            / rollout_id
                            / "prediction.step"
                        )
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        artifact.write_bytes(prediction.read_bytes())
                    except OSError:
                        LOGGER.exception("Could not save generated STEP for live rollout")
                return await self._score(prediction, reference)
        except Exception as exc:
            LOGGER.exception("DeepCAD infrastructure failure")
            return {
                "reward": 0.0,
                "mask_sample": True,
                "failure_kind": "deepcad:infrastructure",
                "failure_reason": str(exc),
            }

    async def _score(self, prediction: Path, reference: Path) -> dict[str, Any]:
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            str(HERE / "geometry.py"),
            str(prediction),
            str(reference),
            str(self.config.voxel_resolution),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), self.config.geometry_timeout_s)
        except (TimeoutError, asyncio.CancelledError):
            process.kill()
            await process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(f"Geometry worker failed: {stderr.decode(errors='replace')[-2000:]}")
        return json.loads(stdout.decode(errors="replace"))


if __name__ == "__main__":
    DeepCADResourcesServer.run_webserver()
