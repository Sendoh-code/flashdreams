# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a LingBot KV-policy comparison written by ``compare.slurm``.

Usage: ``python compare_metrics.py <out_dir> --poses <poses.npy> [--baseline NAME]``
or ``python compare_metrics.py --self-check``.

Per run (``<out_dir>/<run>/`` with ``out.mp4``, ``stats.json``, ``run.log``):

- speed: median / p90 chunk time after warmup and the implied FPS;
- memory: peak GPU memory (``mem_peak_gib``) and host max RSS (``/usr/bin/time -v``);
- revisit consistency: PSNR / SSIM between frame ``j`` and the first earlier frame
  ``i`` with the same camera pose (rotation < ``--max-deg``) at least ``--min-gap``
  frames back, skipping sources inside the sink;
- no-reference quality proxies: Laplacian sharpness and its last/first quarter
  ratio (blur drift), plus PSNR vs the baseline run (divergence, not quality).

Writes ``metrics.json`` and ``metrics.md`` into ``<out_dir>``.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

FRAMES_PER_CHUNK = 16
"""Pixel frames per steady-state AR chunk (LingBot len_t=4, VAE stride 4)."""


def _gray(frames: torch.Tensor) -> torch.Tensor:
    """``[T, H, W, 3]`` uint8 -> ``[T, 1, H, W]`` float in [0, 1]."""
    f = frames.float() / 255
    return (0.299 * f[..., 0] + 0.587 * f[..., 1] + 0.114 * f[..., 2])[:, None]


def psnr(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame PSNR (dB) of uint8 ``[T, H, W, 3]`` stacks."""
    mse = ((a.float() - b.float()) / 255).pow(2).flatten(1).mean(1)
    return 10 * torch.log10(1 / mse.clamp_min(1e-10))


def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Per-frame grayscale SSIM (11x11 Gaussian, sigma 1.5)."""
    x, y = _gray(a), _gray(b)
    g = torch.exp(-((torch.arange(11) - 5.0) ** 2) / (2 * 1.5**2))
    w = (g[:, None] * g[None, :] / g.sum() ** 2)[None, None]
    mu_x, mu_y = F.conv2d(x, w), F.conv2d(y, w)
    sxx = F.conv2d(x * x, w) - mu_x**2
    syy = F.conv2d(y * y, w) - mu_y**2
    sxy = F.conv2d(x * y, w) - mu_x * mu_y
    c1, c2 = 0.01**2, 0.03**2
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / ((mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2))
    return s.flatten(1).mean(1)


def sharpness(frames: torch.Tensor) -> torch.Tensor:
    """Per-frame variance of the Laplacian (higher = sharper)."""
    k = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    return F.conv2d(_gray(frames), k).flatten(1).var(1)


def revisit_pairs(
    poses: np.ndarray, *, min_gap: int, max_deg: float, min_src: int
) -> list[tuple[int, int]]:
    """``(i, j)`` pairs: ``i`` is the first frame >= ``min_src`` matching ``j``'s pose."""
    rot, trans = poses[:, :3, :3], poses[:, :3, 3]
    pairs = []
    for j in range(len(poses)):
        for i in range(min_src, j - min_gap + 1):
            cos = (np.trace(rot[i].T @ rot[j]) - 1) / 2
            deg = np.degrees(np.arccos(np.clip(cos, -1, 1)))
            if deg < max_deg and np.linalg.norm(trans[i] - trans[j]) < 1e-3:
                pairs.append((i, j))
                break
    return pairs


def _steady(stats: dict, name: str, warmup: int) -> list[float]:
    rows = [s for s in stats["samples"] if s["name"] == name]
    return [s["value"] for s in rows if s["step_index"] >= warmup] or [s["value"] for s in rows]


def run_metrics(run: Path, pairs: list[tuple[int, int]], warmup: int) -> dict:
    """Speed, memory, and video metrics for one run directory."""
    import imageio.v3 as iio

    out: dict = {"run": run.name}
    stats = json.loads((run / "stats.json").read_text())
    total = sorted(_steady(stats, "total_s", warmup))
    out["chunk_s_median"] = statistics.median(total)
    out["chunk_s_p90"] = total[min(len(total) - 1, int(0.9 * len(total)))]
    out["fps"] = FRAMES_PER_CHUNK / out["chunk_s_median"]
    out["gpu_peak_gib"] = max(_steady(stats, "mem_peak_gib", 0))
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", (run / "run.log").read_text())
    out["host_rss_gib"] = int(rss.group(1)) / 2**20 if rss else None
    frames = torch.from_numpy(iio.imread(run / "out.mp4", plugin="pyav"))
    out["frames"] = len(frames)
    pairs = [(i, j) for i, j in pairs if j < len(frames)]
    if pairs:
        src, dst = frames[[i for i, _ in pairs]], frames[[j for _, j in pairs]]
        out["revisit_pairs"] = len(pairs)
        out["revisit_psnr"] = psnr(src, dst).mean().item()
        out["revisit_ssim"] = ssim(src, dst).mean().item()
    sharp = sharpness(frames)
    q = max(1, len(frames) // 4)
    out["sharpness"] = sharp.mean().item()
    out["sharpness_late_over_early"] = (sharp[-q:].mean() / sharp[:q].mean()).item()
    out["_frames"] = frames
    return out


def summarize(out_dir: Path, poses: np.ndarray, args: argparse.Namespace) -> list[dict]:
    """Compute metrics for every finished run and write metrics.json / metrics.md."""
    pairs = revisit_pairs(poses, min_gap=args.min_gap, max_deg=args.max_deg, min_src=args.min_src)
    rows = []
    for run in sorted(p for p in out_dir.iterdir() if (p / "ok").exists()):
        rows.append(run_metrics(run, pairs, args.warmup))
    base = next((r for r in rows if r["run"] == args.baseline), None)
    for r in rows:
        if base is not None and r is not base:
            n = min(len(r["_frames"]), len(base["_frames"]))
            r["psnr_vs_baseline"] = psnr(r["_frames"][:n], base["_frames"][:n]).mean().item()
    for r in rows:
        del r["_frames"]
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    cols = [
        "run", "chunk_s_median", "chunk_s_p90", "fps", "gpu_peak_gib", "host_rss_gib",
        "revisit_psnr", "revisit_ssim", "sharpness", "sharpness_late_over_early",
        "psnr_vs_baseline",
    ]  # fmt: skip
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str("-" if v is None else v)  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    md.append(f"\n{len(pairs)} revisit pairs (min gap {args.min_gap}, < {args.max_deg} deg).")
    (out_dir / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def self_check() -> None:
    """Metric sanity on synthetic data; no GPU or video files needed."""
    yaw = np.radians([0, 30, 60, 30, 0, 0])
    poses = np.tile(np.eye(4), (6, 1, 1))
    poses[:, 0, 0] = poses[:, 2, 2] = np.cos(yaw)
    poses[:, 0, 2], poses[:, 2, 0] = np.sin(yaw), -np.sin(yaw)
    assert revisit_pairs(poses, min_gap=2, max_deg=1, min_src=0) == [(1, 3), (0, 4), (0, 5)]
    assert revisit_pairs(poses, min_gap=2, max_deg=1, min_src=1) == [(1, 3)]
    g = torch.Generator().manual_seed(0)
    a = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8, generator=g)
    b = torch.randint(0, 256, (2, 32, 32, 3), dtype=torch.uint8, generator=g)
    assert psnr(a, a).min() > 90 and psnr(a, b).max() < 10
    assert ssim(a, a).min() > 0.999 and ssim(a, b).max() < 0.1
    flat = torch.full_like(a, 128)
    assert sharpness(flat).max() == 0 < sharpness(a).min()
    print("self-check ok")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out_dir", nargs="?", type=Path)
    parser.add_argument("--poses", type=Path, help="raw [T, 4, 4] poses given to --pose-path")
    parser.add_argument("--len-t", type=int, default=4)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--min-gap", type=int, default=64, help="> baseline window (48 frames)")
    parser.add_argument("--max-deg", type=float, default=2.0)
    parser.add_argument("--min-src", type=int, default=24, help="skip sink frames as sources")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if args.out_dir is None or args.poses is None:
        parser.error("out_dir and --poses are required")
    # Same pose resampling the model sees, so frame j lines up with poses[j].
    from lingbot.impl.encoder.utils import preprocess_example_poses

    poses, _ = preprocess_example_poses(np.load(args.poses), transformer_len_t=args.len_t)
    summarize(args.out_dir, poses, args)


if __name__ == "__main__":
    sys.exit(main())
