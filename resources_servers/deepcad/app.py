# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Three-view CAD reconstruction environment for NeMo Gym."""

import asyncio
import json
import logging
import os
import re
import shlex
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import Field, PrivateAttr

from nemo_gym.base_resources_server import (
    BaseResourcesServerConfig,
    BaseVerifyRequest,
    BaseVerifyResponse,
    SimpleResourcesServer,
)
from nemo_gym.sandbox import AsyncSandbox
from nemo_gym.sandbox.config import resolve_provider_config
from nemo_gym.sandbox.providers.base import SandboxResources, SandboxSpec
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
    alignment_samples: int = Field(default=2048, ge=32)
    alignment_iterations: int = Field(default=40, ge=1)


class DeepCADVerifyRequest(BaseVerifyRequest):
    verifier_metadata: TaskData


class DeepCADVerifyResponse(BaseVerifyResponse):
    alignment_error: float | None = None
    transform: list[list[float]] | None = None


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

    async def verify(self, body: DeepCADVerifyRequest) -> DeepCADVerifyResponse:
        """Execute code in isolation, then score the exported solid in a trusted worker."""
        result: dict[str, Any]
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
                    result = await self._evaluate(code, body.verifier_metadata.reference_step)
        return DeepCADVerifyResponse(**body.model_dump(), **result)

    async def _evaluate(self, code: str, reference_step: str) -> dict[str, Any]:
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
            str(self.config.alignment_samples),
            str(self.config.alignment_iterations),
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
