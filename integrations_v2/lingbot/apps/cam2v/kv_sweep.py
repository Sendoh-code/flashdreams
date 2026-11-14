# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A40 (sm86) LingBot 1.3B app for KV window / host-offload sweeps.

Run as ``flashdreams-run-v2 lingbot.apps.cam2v.kv_sweep``; the KV policy comes
from ``KV_SWEEP_WINDOW_T``, ``KV_SWEEP_SINK_T``, and ``KV_SWEEP_OFFLOAD=1``.
``KV_SWEEP_RETRIEVE_T>0`` (implies offload) adds a camera-pose KV bank with
``KV_SWEEP_BANK_GB`` (default 8) and ``KV_SWEEP_BANK_GRANULARITY``
(``chunk``/``frame``). ``KV_SWEEP_ROPE=pre`` caches K before RoPE
(cache-relative positions) instead of the default ``post``.
"""

from __future__ import annotations

import os

from flashdreams.api_v2.application import IApplication
from flashdreams.core.attention import CameraPoseRetrieval, KVBankConfig
from flashdreams.infra.config import derive_config
from lingbot.apps.cam2v.adapter import LingbotCam2VApplication
from lingbot.config import PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF


def create_app() -> IApplication:
    """Return the bf16 1.3B app with the KV policy read from the environment."""
    window_t = int(os.environ.get("KV_SWEEP_WINDOW_T", "12"))
    sink_t = int(os.environ.get("KV_SWEEP_SINK_T", "6"))
    retrieve_t = int(os.environ.get("KV_SWEEP_RETRIEVE_T", "0"))
    offload = os.environ.get("KV_SWEEP_OFFLOAD") == "1" or retrieve_t > 0
    rope = os.environ.get("KV_SWEEP_ROPE", "post")
    if rope not in ("pre", "post"):
        raise ValueError(f"KV_SWEEP_ROPE must be 'pre' or 'post', got {rope!r}")
    bank = None
    if retrieve_t > 0:
        bank = KVBankConfig(
            budget_gb=float(os.environ.get("KV_SWEEP_BANK_GB", "8")),
            granularity=os.environ.get("KV_SWEEP_BANK_GRANULARITY", "chunk"),  # type: ignore[arg-type]
            retrieve_size_t=retrieve_t,
            policy=CameraPoseRetrieval(),
        )
    config = derive_config(
        PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF,
        name=f"lingbot-1p3b-a40-w{window_t}-s{sink_t}{'-offload' if offload else ''}"
        f"{f'-r{retrieve_t}' if retrieve_t else ''}{'-prerope' if rope == 'pre' else ''}",
        diffusion_model=dict(
            transformer=dict(
                # sm86 has no FP8: use bf16 linears and Wan attention.
                network=dict(
                    linear_backend="torch",
                    self_attention_backend="wan",
                    apply_rope_before_kvcache=rope == "post",
                ),
                window_size_t=window_t,
                sink_size_t=sink_t,
                kv_offload=offload,
                kv_bank=bank,
            ),
        ),
    )
    return LingbotCam2VApplication(pipeline_config=config)
