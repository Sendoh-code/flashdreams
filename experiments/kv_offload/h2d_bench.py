# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Pinned host <-> device copy bandwidth at one-layer LingBot 1.3B K/V sizes.

Sizes are ``V`` latent frames x 9.26 MB (K+V, 1508 tokens x 1536 dim, bf16).
Prints one JSON line per (direction, frames).
"""

from __future__ import annotations

import json
import sys

import torch

FRAME_MB = 1508 * 1536 * 2 * 2 / 1e6
FRAMES = (1, 4, 18, 40, 80, 160)


def bench(nbytes: int, h2d: bool, iters: int = 10) -> float:
    host = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dev = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    src, dst = (host, dev) if h2d else (dev, host)
    for _ in range(2):
        dst.copy_(src, non_blocking=True)
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    end.record()
    end.synchronize()
    return nbytes * iters / (start.elapsed_time(end) / 1e3) / 1e9


def main() -> None:
    tag = sys.argv[1] if len(sys.argv) > 1 else ""
    for frames in FRAMES:
        nbytes = int(frames * FRAME_MB * 1e6)
        for h2d in (True, False):
            gbps = bench(nbytes, h2d)
            print(json.dumps({"tag": tag, "dir": "h2d" if h2d else "d2h",
                              "frames": frames, "mb": nbytes / 1e6, "gbps": round(gbps, 2)}))


if __name__ == "__main__":
    main()
