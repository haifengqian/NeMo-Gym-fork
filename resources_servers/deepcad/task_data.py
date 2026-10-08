# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Task fields shared by dataset validation and the runtime verifier."""

from pydantic import BaseModel, ConfigDict, Field


class TaskData(BaseModel):
    model_config = ConfigDict(extra="allow")
    task_id: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    reference_step: str = Field(json_schema_extra={"legacy_location": "verifier_metadata"})
    difficulty: int | None = Field(default=None, ge=0, json_schema_extra={"legacy_location": "verifier_metadata"})
    construction_metrics: dict[str, int] | None = Field(
        default=None, json_schema_extra={"legacy_location": "verifier_metadata"}
    )
