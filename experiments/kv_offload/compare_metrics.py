# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a LingBot KV-policy comparison written by ``compare.slurm``.

Usage: ``python compare_metrics.py <out_dir> --poses <poses.npy> [--baseline NAME]``
or ``python compare_metrics.py --self-check``.

Per run (``<out_dir>/<run>/`` with ``out.mp4``; ``stats.json`` and ``run.log`` add
speed and memory, so videos from other code bases can be scored too):

- speed: median / p90 chunk time after warmup and the implied FPS;
- memory: peak GPU memory (``mem_peak_gib``) and host max RSS (``/usr/bin/time -v``);
- revisit consistency: PSNR / SSIM between frame ``j`` and the first earlier frame
  ``i`` with the same camera pose (rotation < ``--max-deg``) at least ``--min-gap``
  frames back, skipping sources inside the sink; also split into near / far
  pairs at ``--far-gap`` frames apart; ``--perceptual`` adds WorldTrace's pair
  metrics, DINOv2 ViT-S/14 CLS cosine and LPIPS-VGG (needs ``lpips``, e.g.
  ``uv run --with lpips``; downloads both models);
- no-reference quality proxies: Laplacian sharpness and its last/first quarter
  ratio (blur drift), plus PSNR vs the baseline run (divergence, not quality).

All video metrics use only columns ``x >= --x0 * width``: flashdreams cam2v mp4s have
the "Camera controls" UI panel burned in at the top left (``x < ~0.45 W``).

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

VAE_STRIDE_T = 4
"""Pixel frames per latent frame; a steady-state chunk has ``len_t * 4`` pixel frames."""


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
    s = ((2 * mu_x * mu_y + c1) * (2 * sxy + c2)) / (
        (mu_x**2 + mu_y**2 + c1) * (sxx + syy + c2)
    )
    return s.flatten(1).mean(1)


def sharpness(frames: torch.Tensor) -> torch.Tensor:
    """Per-frame variance of the Laplacian (higher = sharper)."""
    k = torch.tensor([[0.0, 1, 0], [1, -4, 1], [0, 1, 0]])[None, None]
    return F.conv2d(_gray(frames), k).flatten(1).var(1)


def perceptual(
    src: torch.Tensor, dst: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-pair DINOv2 CLS cosine and LPIPS-VGG of uint8 ``[N, H, W, 3]`` frames (WorldTrace App. E)."""
    import lpips
    from transformers import AutoModel

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    dino = AutoModel.from_pretrained("facebook/dinov2-small").to(dev).eval()
    vgg = lpips.LPIPS(net="vgg", verbose=False).to(dev)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    cos, lp = [], []
    with torch.no_grad():
        for a, b in zip(src.split(16), dst.split(16)):
            a, b = (x.to(dev).permute(0, 3, 1, 2).float() / 255 for x in (a, b))
            lp.append(vgg(a, b, normalize=True).flatten().cpu())
            # The paper feeds 252x448; keep that height and round the width to the
            # patch size, since the UI-panel crop changes the aspect ratio.
            size = (252, max(14, round(252 * a.shape[3] / a.shape[2] / 14) * 14))
            fa, fb = (
                dino(
                    pixel_values=(
                        F.interpolate(x, size, mode="bilinear", antialias=True) - mean
                    )
                    / std
                ).last_hidden_state[:, 0]
                for x in (a, b)
            )
            cos.append(F.cosine_similarity(fa, fb).cpu())
    return torch.cat(cos), torch.cat(lp)


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


def video_poses(raw: np.ndarray, n_frames: int, stretch: bool) -> np.ndarray:
    """Raw pose of each decoded pixel frame.

    flashdreams keeps the first ``4k + 1`` raw poses, so frame ``f`` is ``raw[f]``;
    with ``stretch`` (WorldKV ``generate_fast.py``) the frames span all raw poses.
    """
    if stretch:
        return raw[np.round(np.linspace(0, len(raw) - 1, n_frames)).astype(int)]
    return raw[:n_frames]


def _steady(stats: dict, name: str, warmup: int) -> list[float]:
    rows = [s for s in stats["samples"] if s["name"] == name]
    return [s["value"] for s in rows if s["step_index"] >= warmup] or [
        s["value"] for s in rows
    ]


def run_metrics(
    run: Path, raw_poses: np.ndarray, args: argparse.Namespace
) -> tuple[dict, list[tuple[int, int]]]:
    """Speed, memory, and video metrics for one run directory, plus its revisit pairs."""
    import imageio.v3 as iio

    far_gap, x0 = args.far_gap, args.x0
    out: dict = {"run": run.name}
    if (run / "stats.json").exists():
        stats = json.loads((run / "stats.json").read_text())
        total = sorted(_steady(stats, "total_s", args.warmup))
        out["chunk_s_median"] = statistics.median(total)
        out["chunk_s_p90"] = total[min(len(total) - 1, int(0.9 * len(total)))]
        out["fps"] = args.len_t * VAE_STRIDE_T / out["chunk_s_median"]
        out["gpu_peak_gib"] = max(_steady(stats, "mem_peak_gib", 0))
    log = (run / "run.log").read_text() if (run / "run.log").exists() else ""
    rss = re.search(r"Maximum resident set size \(kbytes\): (\d+)", log)
    out["host_rss_gib"] = int(rss.group(1)) / 2**20 if rss else None
    frames = torch.from_numpy(iio.imread(run / "out.mp4", plugin="pyav"))
    # The VAE decodes 4k + 1 frames; anything before that is a presentation
    # frame (flashdreams records one "waiting" frame first).
    frames = frames[(len(frames) - 1) % VAE_STRIDE_T :]
    frames = frames[:, :, int(x0 * frames.shape[2]) :]  # drop the burned-in UI panel
    out["frames"] = len(frames)
    pairs = revisit_pairs(
        video_poses(raw_poses, len(frames), args.stretch_poses),
        min_gap=args.min_gap,
        max_deg=args.max_deg,
        min_src=args.min_src,
    )
    if pairs:
        src, dst = frames[[i for i, _ in pairs]], frames[[j for _, j in pairs]]
        p, s = psnr(src, dst), ssim(src, dst)
        far = torch.tensor([j - i >= far_gap for i, j in pairs])
        out["revisit_pairs"] = len(pairs)
        out["revisit_psnr"], out["revisit_ssim"] = p.mean().item(), s.mean().item()
        metrics = {"psnr": p, "ssim": s}
        if args.perceptual:
            metrics["dino"], metrics["lpips"] = perceptual(src, dst)
            out["revisit_dino"] = metrics["dino"].mean().item()
            out["revisit_lpips"] = metrics["lpips"].mean().item()
        for tag, m in (("near", ~far), ("far", far)):  # split by revisit distance
            if m.any():
                for name, v in metrics.items():
                    out[f"revisit_{name}_{tag}"] = v[m].mean().item()
    sharp = sharpness(frames)
    q = max(1, len(frames) // 4)
    out["sharpness"] = sharp.mean().item()
    out["sharpness_late_over_early"] = (sharp[-q:].mean() / sharp[:q].mean()).item()
    out["_frames"] = frames
    return out, pairs


def summarize(out_dir: Path, poses: np.ndarray, args: argparse.Namespace) -> list[dict]:
    """Compute metrics for every finished run (``ok`` marker) from raw ``poses``."""
    rows, pairs = [], []
    for run in sorted(p for p in out_dir.iterdir() if (p / "ok").exists()):
        row, pairs = run_metrics(run, poses, args)
        rows.append(row)
    base = next((r for r in rows if r["run"] == args.baseline), None)
    for r in rows:
        if base is not None and r is not base:
            n = min(len(r["_frames"]), len(base["_frames"]))
            r["psnr_vs_baseline"] = (
                psnr(r["_frames"][:n], base["_frames"][:n]).mean().item()
            )
    for r in rows:
        del r["_frames"]
    (out_dir / "metrics.json").write_text(json.dumps(rows, indent=2))
    cols = [
        "run", "chunk_s_median", "chunk_s_p90", "fps", "gpu_peak_gib", "host_rss_gib",
        "revisit_psnr", "revisit_ssim", "revisit_psnr_near", "revisit_psnr_far",
        "sharpness", "sharpness_late_over_early", "psnr_vs_baseline",
    ]  # fmt: skip
    if args.perceptual:
        cols[8:8] = [
            "revisit_dino",
            "revisit_lpips",
            "revisit_dino_far",
            "revisit_lpips_far",
        ]
    fmt = lambda v: f"{v:.3f}" if isinstance(v, float) else str("-" if v is None else v)  # noqa: E731
    md = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    md += ["| " + " | ".join(fmt(r.get(c)) for c in cols) + " |" for r in rows]
    n_far = sum(j - i >= args.far_gap for i, j in pairs)
    md.append(
        f"\n{len(pairs)} revisit pairs (min gap {args.min_gap}, < {args.max_deg} deg; "
        f"near < {args.far_gap} frames: {len(pairs) - n_far}, far: {n_far})."
    )
    (out_dir / "metrics.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    return rows


def self_check() -> None:
    """Metric sanity on synthetic data; no GPU or video files needed."""
    yaw = np.radians([0, 30, 60, 30, 0, 0])
    poses = np.tile(np.eye(4), (6, 1, 1))
    poses[:, 0, 0] = poses[:, 2, 2] = np.cos(yaw)
    poses[:, 0, 2], poses[:, 2, 0] = np.sin(yaw), -np.sin(yaw)
    assert revisit_pairs(poses, min_gap=2, max_deg=1, min_src=0) == [
        (1, 3),
        (0, 4),
        (0, 5),
    ]
    assert revisit_pairs(poses, min_gap=2, max_deg=1, min_src=1) == [(1, 3)]
    raw = np.arange(9)
    assert video_poses(raw, 5, False).tolist() == [0, 1, 2, 3, 4]
    assert video_poses(raw, 5, True).tolist() == [0, 2, 4, 6, 8]
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
    parser.add_argument(
        "--poses", type=Path, help="raw [T, 4, 4] poses given to --pose-path"
    )
    parser.add_argument("--len-t", type=int, default=4)
    parser.add_argument("--baseline", default="baseline")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument(
        "--min-gap", type=int, default=64, help="> baseline window (48 frames)"
    )
    parser.add_argument("--max-deg", type=float, default=2.0)
    parser.add_argument(
        "--min-src", type=int, default=24, help="skip sink frames as sources"
    )
    parser.add_argument(
        "--far-gap", type=int, default=200,
        help="near/far split of revisit pairs (pixel frames); 200 sits between "
        "WorldKV R_L_R_L_R_L90_360's two gap clusters (~120-190 and ~240-300)",
    )  # fmt: skip
    parser.add_argument(
        "--x0", type=float, default=0.5,
        help="keep only columns x >= x0 * width (UI panel ends at ~0.45 W); 0 = full frame",
    )  # fmt: skip
    parser.add_argument(
        "--stretch-poses", action="store_true",
        help="frames span all raw poses (WorldKV generate_fast.py videos)",
    )  # fmt: skip
    parser.add_argument(
        "--perceptual",
        action="store_true",
        help="add revisit DINOv2 cosine and LPIPS-VGG",
    )
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args()
    if args.self_check:
        return self_check()
    if args.out_dir is None or args.poses is None:
        parser.error("out_dir and --poses are required")
    # Raw poses, one per pixel frame: the camera moves smoothly within a latent
    # frame, so the per-latent poses the model takes would misalign pairs.
    summarize(args.out_dir, np.load(args.poses), args)


if __name__ == "__main__":
    sys.exit(main())
