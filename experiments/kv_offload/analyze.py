# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Summarize a KV offload sweep and find the H2D crossover V*.

Usage: ``python analyze.py <sweep_dir>`` or ``python analyze.py --self-check``.

``<sweep_dir>/<run>/`` holds ``meta.json`` (written by ``sweep.slurm``),
``stats.json`` (``--stats-path``), optionally ``kv_timing.jsonl`` and
``dmon.log``. Writes ``forwards.csv``, ``summary.csv`` and, when matplotlib is
installed, ``crossover.png`` into ``<sweep_dir>``.
"""

from __future__ import annotations

import csv
import json
import statistics
import sys
from pathlib import Path

FRAME_MB = 1508 * 1536 * 2 * 2 / 1e6
"""K+V MB per latent frame per layer (LingBot 1.3B, 464x832, bf16)."""


def forward_rows(jsonl: Path, meta: dict) -> list[dict]:
    """One row per timed forward, warmup chunks dropped."""
    forwards = [json.loads(line) for line in jsonl.read_text().splitlines() if line]
    per_chunk = max(1, len(forwards) // meta["blocks"])
    rows = []
    for record in forwards[meta["warmup"] * per_chunk :]:
        layers = record["layers"]
        compute = sum(r["compute_ms"] for r in layers)
        fetch = sum(r["fetch_ms"] for r in layers)
        rows.append(
            {
                "run": meta["run"],
                "forward": record["forward"],
                "chunk": record["forward"] // per_chunk,
                "V": round(statistics.median(r["fetch_mb"] for r in layers) / FRAME_MB, 1),
                "fetch_ms": statistics.median(r["fetch_ms"] for r in layers),
                "compute_ms": statistics.median(r["compute_ms"] for r in layers),
                "writeback_ms": statistics.median(r["writeback_ms"] for r in layers),
                "stall_total_ms": sum(r["stall_ms"] for r in layers),
                "compute_total_ms": compute,
                "h2d_gbps": sum(r["fetch_mb"] for r in layers) / fetch if fetch else 0.0,
            }
        )
    return rows


def crossover(rows: list[dict]) -> dict:
    """Fit fetch/compute per layer as lines in V; V* is where they meet."""
    vs = [r["V"] for r in rows]
    f1, f0 = statistics.linear_regression(vs, [r["fetch_ms"] for r in rows])
    c1, c0 = statistics.linear_regression(vs, [r["compute_ms"] for r in rows])
    v_fit = (c0 - f0) / (f1 - c1) if f1 > c1 else float("inf")
    # Empirical: smallest V whose median stall exceeds 5% of compute.
    by_v: dict[float, list[dict]] = {}
    for r in rows:
        by_v.setdefault(r["V"], []).append(r)
    stalled = [
        v
        for v, group in sorted(by_v.items())
        if statistics.median(g["stall_total_ms"] / g["compute_total_ms"] for g in group) > 0.05
    ]
    return {
        "fetch_ms_per_frame": f1,
        "fetch_ms_at_0": f0,
        "compute_ms_per_frame": c1,
        "compute_ms_at_0": c0,
        "v_star_fit": v_fit,
        "v_star_stall5pct": stalled[0] if stalled else None,
    }


def step_wall_s(stats: Path, warmup: int) -> tuple[float, float]:
    """Median per-chunk model wall time after warmup, and last steady FPS."""
    samples = json.loads(stats.read_text())["samples"]
    walls = [
        s["value"]
        for s in samples
        if s["name"] == "model_step_wall_s" and (s["step_index"] or 0) >= warmup
    ]
    fps = [s["value"] for s in samples if s["name"] == "steady_state_fps"]
    return (statistics.median(walls) if walls else float("nan"), fps[-1] if fps else float("nan"))


def dmon_rx_gbps(log: Path) -> float:
    """Peak PCIe rx (host->device) from ``nvidia-smi dmon -s t`` in GB/s."""
    rx = [
        float(cols[1])
        for line in log.read_text().splitlines()
        if not line.startswith("#") and len(cols := line.split()) >= 3 and cols[1].isdigit()
    ]
    return max(rx) / 1e3 if rx else float("nan")


def write_csv(path: Path, rows: list[dict]) -> None:
    if rows:
        with path.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def analyze(sweep: Path) -> None:
    forwards: list[dict] = []
    summary: list[dict] = []
    for meta_path in sorted(sweep.glob("*/meta.json")):
        run_dir = meta_path.parent
        meta = json.loads(meta_path.read_text()) | {"run": run_dir.name}
        rows = []
        if (run_dir / "kv_timing.jsonl").exists():
            rows = forward_rows(run_dir / "kv_timing.jsonl", meta)
            forwards += rows
        stats = run_dir / "stats.json"
        wall, fps = step_wall_s(stats, meta["warmup"]) if stats.exists() else (float("nan"),) * 2
        dmon = run_dir / "dmon.log"
        med = lambda key: statistics.median(r[key] for r in rows) if rows else ""  # noqa: E731
        summary.append(
            {
                "run": meta["run"],
                "W": meta["window"],
                "S": meta["sink"],
                "offload": meta["offload"],
                "blocks": meta["blocks"],
                "timing": meta["timing"],
                "status": meta.get("status", ""),
                "V_max": max((r["V"] for r in rows), default=""),
                "fetch_ms_layer": med("fetch_ms"),
                "compute_ms_layer": med("compute_ms"),
                "stall_ms_forward": med("stall_total_ms"),
                "h2d_gbps": med("h2d_gbps"),
                "dmon_rx_peak_gbps": dmon_rx_gbps(dmon) if dmon.exists() else "",
                "chunk_wall_s": wall,
                "steady_fps": fps,
            }
        )
    base = {
        (s["blocks"]): s["chunk_wall_s"]
        for s in summary
        if s["W"] == 12 and s["S"] == 6 and not s["offload"] and not s["timing"]
    }
    for s in summary:
        s["slowdown_vs_B0"] = s["chunk_wall_s"] / base[s["blocks"]] if s["blocks"] in base else ""
    write_csv(sweep / "forwards.csv", forwards)
    write_csv(sweep / "summary.csv", summary)
    if not forwards:
        print("no timed offload runs found")
        return
    fit = crossover(forwards)
    (sweep / "crossover.json").write_text(json.dumps(fit, indent=2) + "\n")
    print(json.dumps(fit, indent=2))
    plot(sweep / "crossover.png", forwards, fit)


def plot(path: Path, rows: list[dict], fit: dict) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot")
        return
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    vs = [r["V"] for r in rows]
    ax.scatter(vs, [r["fetch_ms"] for r in rows], s=8, label="fetch / layer")
    ax.scatter(vs, [r["compute_ms"] for r in rows], s=8, label="compute / layer")
    if fit["v_star_fit"] != float("inf"):
        ax.axvline(fit["v_star_fit"], color="gray", ls="--", label=f"V* = {fit['v_star_fit']:.1f}")
    ax.set(xlabel="visible KV frames V", ylabel="ms", title="Per-layer fetch vs compute")
    ax.legend()
    ax2.scatter(vs, [r["stall_total_ms"] for r in rows], s=8)
    ax2.set(xlabel="visible KV frames V", ylabel="ms", title="Compute stall per forward")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def self_check() -> None:
    # fetch = 0.42 V, compute = 3 + 0.15 V  ->  V* = 3 / 0.27
    rows = [
        {"V": v, "fetch_ms": 0.42 * v, "compute_ms": 3 + 0.15 * v,
         "stall_total_ms": 30 * max(0.0, 0.42 * v - 3 - 0.15 * v),
         "compute_total_ms": 30 * (3 + 0.15 * v)}
        for v in range(2, 164, 4)
    ]
    fit = crossover(rows)
    assert abs(fit["v_star_fit"] - 3 / 0.27) < 1e-6, fit
    assert fit["v_star_stall5pct"] == 14, fit
    print("self-check ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--self-check"]:
        self_check()
    else:
        analyze(Path(sys.argv[1]))
