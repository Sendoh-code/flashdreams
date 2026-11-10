# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A40 (sm86) LingBot 1.3B app for KV window / host-offload sweeps.

Run as ``flashdreams-run-v2 lingbot.apps.cam2v.kv_sweep``; the KV policy comes
from ``KV_SWEEP_WINDOW_T``, ``KV_SWEEP_SINK_T``, and ``KV_SWEEP_OFFLOAD=1``.
"""

from __future__ import annotations

import os

from flashdreams.api_v2.application import IApplication
from flashdreams.infra.config import derive_config
from lingbot.apps.cam2v.adapter import LingbotCam2VApplication
from lingbot.config import PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF


def create_app() -> IApplication:
    """Return the bf16 1.3B app with the KV policy read from the environment."""
    window_t = int(os.environ.get("KV_SWEEP_WINDOW_T", "12"))
    sink_t = int(os.environ.get("KV_SWEEP_SINK_T", "6"))
    offload = os.environ.get("KV_SWEEP_OFFLOAD") == "1"
    config = derive_config(
        PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF,
        name=f"lingbot-1p3b-a40-w{window_t}-s{sink_t}{'-offload' if offload else ''}",
        diffusion_model=dict(
            transformer=dict(
                # sm86 has no FP8: use bf16 linears and Wan attention.
                network=dict(linear_backend="torch", self_attention_backend="wan"),
                window_size_t=window_t,
                sink_size_t=sink_t,
                kv_offload=offload,
            ),
        ),
    )
    return LingbotCam2VApplication(pipeline_config=config)
