# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Dump the latents a run decodes, or compare two dumps bit for bit.

Run (same args as ``flashdreams-run-v2``; add ``--no-compile`` after ``--``,
compiled runs are not reproducible)::

    LATENT_DIR=<dir> python experiments/kv_offload/latent_parity.py \
        lingbot.apps.cam2v.kv_sweep --mode mp4 ... -- ... --no-compile

Compare::

    python experiments/kv_offload/latent_parity.py --compare <dir_a> <dir_b>
"""

import os
import sys
from pathlib import Path

import torch


def compare(a: Path, b: Path) -> bool:
    files = sorted(a.glob("*.pt"), key=lambda p: int(p.stem))
    assert files and len(files) == len(list(b.glob("*.pt"))), (
        f"dump counts differ: {a} {b}"
    )
    same = True
    for f in files:
        x, y = torch.load(f).float(), torch.load(b / f.name).float()
        if torch.equal(x, y):
            print(f"{f.name} bit-equal")
        else:
            same = False
            print(
                f"{f.name} max|d|={(x - y).abs().max():.3e} rel={(x - y).norm() / x.norm():.2e}"
            )
    return same


def run() -> int:
    from flashdreams.recipes.wan.autoencoder import vae
    from flashdreams.runtime_v2.cli import entrypoint

    out = Path(os.environ["LATENT_DIR"])
    out.mkdir(parents=True, exist_ok=True)
    decode, count = vae.WanVAE.decode, [0]

    def dumping_decode(self, z, cache):  # type: ignore[no-untyped-def]
        torch.save(z.detach().cpu(), out / f"{count[0]}.pt")
        count[0] += 1
        return decode(self, z, cache)

    vae.WanVAE.decode = dumping_decode
    sys.argv[0] = "flashdreams-run-v2"
    return entrypoint()


if __name__ == "__main__":
    if sys.argv[1:2] == ["--compare"]:
        sys.exit(0 if compare(Path(sys.argv[2]), Path(sys.argv[3])) else 1)
    sys.exit(run())
