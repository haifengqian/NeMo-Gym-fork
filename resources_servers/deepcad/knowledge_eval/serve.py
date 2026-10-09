# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Launch vLLM with the same source compatibility patches as NeMo-RL."""

import runpy

from nemo_rl.models.generation.vllm.patches import ensure_vllm_source_compat


if __name__ == "__main__":
    ensure_vllm_source_compat()
    runpy.run_module("vllm.entrypoints.openai.api_server", run_name="__main__")
