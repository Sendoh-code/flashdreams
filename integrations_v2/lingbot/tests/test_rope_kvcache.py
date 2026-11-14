# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pre-RoPE (cache-relative) and post-RoPE K caching agree before the window rolls."""

import pytest
import torch

from flashdreams.core.attention.rope import (
    KVCacheRelativeRotaryPositionEmbedding3D,
    RotaryPositionEmbedding3D,
)
from integrations_v2.lingbot.impl.transformer.impl.modules import CamCtrlBlock

pytestmark = [
    pytest.mark.ci_gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
]

_FP8 = torch.cuda.is_available() and torch.cuda.get_device_capability() >= (8, 9)


@pytest.mark.parametrize(
    "backend",
    [
        "wan",
        pytest.param("scaled_fp8", marks=pytest.mark.skipif(not _FP8, reason="needs sm89+")),
    ],
)
def test_pre_and_post_rope_cache_match_without_roll(backend: str) -> None:
    """With sink 0 and no roll, cache slot == rollout position, so both modes match."""
    torch.manual_seed(0)
    dim, heads, len_t, len_h, len_w, chunks = 256, 2, 2, 2, 4, 3
    tokens = len_t * len_h * len_w
    dtype, device = torch.bfloat16, torch.device("cuda")
    blocks = {
        pre: CamCtrlBlock(
            dim=dim,
            ffn_dim=dim,
            num_heads=heads,
            self_attention_backend=backend,  # type: ignore[arg-type]
            apply_rope_before_kvcache=not pre,
        )
        for pre in (False, True)
    }
    blocks[True].load_state_dict(blocks[False].state_dict())
    rope_args = dict(head_dim=dim // heads, len_h=len_h, len_w=len_w, len_t=len_t, device=device)
    ropes = {
        False: RotaryPositionEmbedding3D(**rope_args),  # type: ignore[arg-type]
        True: KVCacheRelativeRotaryPositionEmbedding3D(
            sink_size_t=0, window_size_t=chunks * len_t, **rope_args  # type: ignore[arg-type]
        ),
    }
    context = torch.randn(1, 3, dim, device=device, dtype=dtype)
    caches = {}
    for pre, block in blocks.items():
        block.to(device, dtype)
        caches[pre] = block.initialize_cache(tokens, chunks * tokens, 0, context)
        block.prepare_camera_cache(torch.randn(1, tokens, dim, device=device, dtype=dtype) * 0, caches[pre])

    for chunk in range(chunks):
        x = torch.randn(1, tokens, dim, device=device, dtype=dtype)
        e = torch.randn(1, 6, dim, device=device, dtype=dtype) * 0.1
        for _step in range(2):  # repeated denoising writes overwrite the same rows
            out = {}
            for pre, block in blocks.items():
                caches[pre].before_update(chunk)
                out[pre] = block(x.clone(), e, caches[pre], ropes[pre].shift_t(chunk))
                caches[pre].after_update(chunk)
            torch.testing.assert_close(out[True], out[False], atol=3e-2, rtol=3e-2)
        # Post-RoPE stores rotated K; pre-RoPE stores raw K, so the caches differ.
        assert not torch.equal(caches[True].self_attn._k, caches[False].self_attn._k)
