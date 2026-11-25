# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""bf16 LingBot app for KV window / host-offload sweeps (1.3B or 14B).

Run as ``flashdreams-run-v2 lingbot.apps.cam2v.kv_sweep``; the KV policy comes
from ``KV_SWEEP_WINDOW_T``, ``KV_SWEEP_SINK_T``, and ``KV_SWEEP_OFFLOAD=1``.
``KV_SWEEP_RETRIEVE_T>0`` adds a camera-pose KV bank with ``KV_SWEEP_BANK_GB``
(default 8), ``KV_SWEEP_BANK_GRANULARITY`` (``chunk``/``frame``), and
``KV_SWEEP_RETRIEVE_GAP_T`` (``CameraPoseRetrieval.min_gap_t``, default 0);
without offload the caches stay resident and only retrieved entries are copied
in. ``KV_SWEEP_ROPE=pre`` caches K before RoPE (cache-relative positions)
instead of the default ``post``. ``KV_SWEEP_MODEL`` picks the base preset:
``1p3b`` (default), ``14b`` (v2 14B causal-fast), or ``v1fast`` (v1
LingBot-World-Fast 14B, the model WorldKV uses).
"""

from __future__ import annotations

import os

from flashdreams.api_v2.application import IApplication
from flashdreams.core.attention import CameraPoseRetrieval, KVBankConfig
from flashdreams.infra.config import derive_config
from lingbot.apps.cam2v.adapter import LingbotCam2VApplication
from lingbot.config import (
    PIPELINE_LINGBOT_WORLD_FAST,
    PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF,
    PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
)

_BASE_CONFIGS = {
    "1p3b": PIPELINE_LINGBOT_WORLD_V2_1P3B_CAUSAL_FAST_PERF,
    "14b": PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST,
    "v1fast": PIPELINE_LINGBOT_WORLD_FAST,
}


def create_app() -> IApplication:
    """Return the bf16 app with the model and KV policy read from the environment."""
    model = os.environ.get("KV_SWEEP_MODEL", "1p3b")
    if model not in _BASE_CONFIGS:
        raise ValueError(
            f"KV_SWEEP_MODEL must be one of {sorted(_BASE_CONFIGS)}, got {model!r}"
        )
    window_t = int(os.environ.get("KV_SWEEP_WINDOW_T", "12"))
    sink_t = int(os.environ.get("KV_SWEEP_SINK_T", "6"))
    retrieve_t = int(os.environ.get("KV_SWEEP_RETRIEVE_T", "0"))
    offload = os.environ.get("KV_SWEEP_OFFLOAD") == "1"
    rope = os.environ.get("KV_SWEEP_ROPE", "post")
    if rope not in ("pre", "post"):
        raise ValueError(f"KV_SWEEP_ROPE must be 'pre' or 'post', got {rope!r}")
    bank = None
    if retrieve_t > 0:
        bank = KVBankConfig(
            budget_gb=float(os.environ.get("KV_SWEEP_BANK_GB", "8")),
            granularity=os.environ.get("KV_SWEEP_BANK_GRANULARITY", "chunk"),  # type: ignore[arg-type]
            retrieve_size_t=retrieve_t,
            policy=CameraPoseRetrieval(
                min_gap_t=int(os.environ.get("KV_SWEEP_RETRIEVE_GAP_T", "0"))
            ),
        )
    config = derive_config(
        _BASE_CONFIGS[model],
        name=f"lingbot-{model}-w{window_t}-s{sink_t}{'-offload' if offload else ''}"
        f"{f'-r{retrieve_t}' if retrieve_t else ''}{'-prerope' if rope == 'pre' else ''}",
        diffusion_model=dict(
            transformer=dict(
                # bf16 linears and Wan attention: runs on sm86 (no FP8) and
                # keeps numbers comparable across GPUs.
                network=dict(
                    linear_backend="torch",
                    self_attention_backend="wan",
                    apply_rope_before_kvcache=rope == "post",
                ),
                window_size_t=window_t,
                sink_size_t=sink_t,
                kv_offload=offload,
                kv_bank=bank,
                # Offload and the bank hook every layer; keep every arm
                # graph-free (the 1.3B preset already is, the 14B ones are not).
                use_cuda_graph=False,
            ),
        ),
    )
    return LingbotCam2VApplication(pipeline_config=config)
