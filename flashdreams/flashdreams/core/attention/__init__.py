# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Attention primitives and KV cache for streaming inference."""

from flashdreams.core.attention.cp import ContextParallelAttention
from flashdreams.core.attention.kvbank import (
    CameraPoseRetrieval,
    KVBank,
    KVBankConfig,
    KVBankEntry,
    KVRetrievalPolicy,
    RecentRetrieval,
)
from flashdreams.core.attention.kvcache import (
    BlockKVCache,
    FixedSlotKVCache,
    LayerKV,
    LayerwiseKVOffload,
    SlotRegion,
    TokenWindow,
)
from flashdreams.core.attention.native import NativeAttention
from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
    apply_rope_freqs,
)

__all__ = [
    "BlockKVCache",
    "CameraPoseRetrieval",
    "ContextParallelAttention",
    "FixedSlotKVCache",
    "KVBank",
    "KVBankConfig",
    "KVBankEntry",
    "KVCacheRelativeRotaryPositionEmbedding3D",
    "KVRetrievalPolicy",
    "LayerKV",
    "LayerwiseKVOffload",
    "NativeAttention",
    "RecentRetrieval",
    "RotaryPositionEmbedding3D",
    "SlotRegion",
    "TokenWindow",
    "apply_rope_freqs",
]
