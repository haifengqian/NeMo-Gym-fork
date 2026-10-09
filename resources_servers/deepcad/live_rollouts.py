# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Save verifier outputs for the persistent DeepCAD rollout viewer."""

import fcntl
import hashlib
import json
import time
import uuid
from pathlib import Path


def write_json(path: Path, value: object) -> None:
    """Publish complete JSON atomically across processes and Gym replicas."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False))
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def save_rollout(
    root: Path,
    experiment: str,
    rollout_id: str,
    *,
    metadata: dict,
    inputs: list,
    outputs: list,
    code: str,
    result: dict,
) -> str | None:
    """Deduplicate fixed prompt images and save every response as a separate rollout."""
    directory = root / experiment
    encoded = json.dumps(inputs, sort_keys=True, ensure_ascii=False)
    digest = hashlib.sha256(encoded.encode()).hexdigest()
    input_path = directory / "inputs" / f"{digest}.json"
    if not input_path.exists():
        write_json(input_path, inputs)
    record = {
        "id": rollout_id,
        "experiment": experiment,
        "created_at": time.time(),
        "verifier_metadata": metadata,
        "input_hash": digest,
        "output": outputs,
        "code": code,
        **result,
    }
    write_json(directory / "rollouts" / rollout_id / "reward.json", record)
    summary = {
        "id": rollout_id,
        "created_at": record["created_at"],
        "task_id": metadata["task_id"],
        "difficulty": metadata.get("difficulty"),
        "reward": result["reward"],
        "failure_kind": result.get("failure_kind"),
        "mask_sample": result.get("mask_sample", False),
        "reward_components": result.get("reward_components"),
        "voxel_iou": result.get("voxel_iou"),
        "silhouette_iou": result.get("silhouette_iou"),
        "normal_alignment": result.get("normal_alignment"),
        "hole_iou": result.get("hole_iou"),
    }
    # Multiple Gym replicas share this append-only index on Lustre. Keep large
    # prompts and model outputs out of it so the viewer can tail new entries.
    with (directory / "index.jsonl").open("a") as index:
        fcntl.flock(index, fcntl.LOCK_EX)
        index.write(json.dumps(summary) + "\n")
        index.flush()
        fcntl.flock(index, fcntl.LOCK_UN)
    state = root / ".viewer-state.json"
    if state.exists():
        info = json.loads(state.read_text())
        return f"http://{info['host']}:{info['port']}/{experiment}#{rollout_id}"
    return None
