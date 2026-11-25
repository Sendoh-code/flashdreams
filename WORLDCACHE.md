# WorldCache: KV-Centric Serving for Causal World Models

Working notes for the WorldCache fork of FlashDreams. Read this first when
resuming the work on a new machine or in a new agent session. Update the
**Status** and **Next Steps** sections whenever they change. File/line map of
every implemented component: `WORLDCACHE_IMPL.md` (update it with each change).

## Goal

Turn FlashDreams into an **LMCache-style, KV-centric serving engine for causal
(autoregressive) video/world models** with **pluggable KV management
policies**.

World-model rollouts are long: fully resident K/V exhausts GPU memory, while
short windows lose long-range consistency (e.g. revisiting a place). The thesis
is to manage KV the way LMCache does for LLMs:

- **Storage hierarchy**: GPU HBM → pinned CPU → local disk / remote, with async
  layerwise transfer overlapped with compute.
- **Pluggable policies chosen by config, not model code**:
  sliding window + sink (exists), full history (exists via a large window),
  host offload (prototype), and future eviction/compression (token importance,
  quantized KV, frame subsampling), retrieval of past KV (pose/spatially
  indexed memory), and reuse/sharing across sessions or CFG branches.
- **Chunk-level KV identity** (chunk index, camera pose, prompt) so KV can be
  stored, looked up, and reloaded instead of recomputed.
- **Serving**: multi-session, memory-budget accounting, per-chunk latency.

Design constraint (see `AGENTS.md` → Dependencies): policies live in
`core`/`infra` as generic hooks; models opt in through config slots. No
model-specific branches in `core`.

## Existing KV Primitives

All in `flashdreams/flashdreams/core/attention/kvcache.py`:

- `BlockKVCache`: per-layer causal cache, layout `[sink | rolling window]`.
  Per chunk: `before_update(chunk_idx) → update(k, v) → cached_k/v() →
  after_update(chunk_idx)`. Same `chunk_idx` overwrites (denoising steps);
  `+1` appends or rolls. Fixed addresses, CUDA-graph safe.
- `FixedSlotKVCache` + `SlotRegion`: fixed prefill plus named circular slot
  regions (rotating controls / generated history).
- `TokenWindow`: `[V, T, S, D]` circular multi-view token buffer.
- `LayerwiseKVOffload`: host-offload prototype (below).
- Multi-view variants: `core/attention/multiview/{cache,rollout}.py`.

Consumers: recipes `wan/transformer/impl/{modules,network}.py`,
`cosmos/transformer/impl/modules.py`, `template/transformer/network.py`;
`accelerated/multi_head_attention/{torch,optimized}.py`; integrations
`lingbot`, `omnidreams`, `flashvsr`, `cosmos_interactive`,
`cosmos3_nano_sim_auto`, `cosmos3_nano_sim_bimanual`.

Today the only policy knobs are `window_size_t`, `sink_size_t`, and
`kv_offload` in transformer configs. There is no shared policy abstraction;
building one is the main refactor toward the goal.

## Status

### Layerwise host KV offload (prototype)

- `LayerwiseKVOffload` streams `BlockKVCache` layers through two GPU slots
  (reusing the first two caches' buffers to keep custom layouts such as
  LingBot's TMA padding). Host storage per layer is a pinned sink buffer plus
  `KVBank` entries, the only host copy of everything past the sink; without a
  configured bank the offload builds a window-only one. `before_layer(i)`
  assembles layer `i+1` (sink, retrieved, window gathered from bank entries by
  index) on a side stream; `after_layer(i)` writes back only the current write
  range. Between layers caches hold meta tensors, so the window roll moves no
  memory. One offload is shared by CFG cond and uncond.
- `begin_chunk()` (synchronize + bank row allocation/eviction) must run before
  each forward's `before_update`; it is wired into
  `WanDiTNetworkCache.before_update`.
- Timing: `FLASHDREAMS_KV_OFFLOAD_TIMING=1`, optional
  `FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=<path>`.
- Wiring: `Wan21TransformerConfig.kv_offload` (rejects `use_cuda_graph=True`);
  `WanDiTNetwork.initialize_cache(kv_offload=...)` and per-block hooks in the
  forward.
- LingBot: offload disables `reuse_key_history` (the Sage key sum would be read
  from a host buffer). Preset and runner
  `lingbot-world-v2-1p3b-causal-fast-perf-fullkv-offload`
  (`window_size_t=80`, `sink_size_t=0`).
- Tests: `flashdreams/tests/test_kvcache.py` —
  `test_layerwise_offload_matches_resident_caches` (`ci_cpu`, parity vs
  resident caches) and `test_layerwise_offload_timing_reports_every_layer`
  (`ci_gpu`).

### A40 H2D crossover experiment (in progress)

- Question: at what visible KV length V does per-layer fetch exceed per-layer
  compute on A40 48GB (Delta `gpuA40x4`)?
- `lingbot.apps.cam2v.kv_sweep`: bf16 1.3B app (sm86 has no FP8, so
  `torch` linears + `wan` attention) with `KV_SWEEP_WINDOW_T`,
  `KV_SWEEP_SINK_T`, `KV_SWEEP_OFFLOAD` from the environment.
- `experiments/kv_offload/`: `h2d_bench.py` (pinned copy GB/s at layer sizes),
  `sweep.slurm` (window × resident/offload matrix, NUMA-bound, serial),
  `analyze.py` (CSV + fitted crossover V*; `--self-check`).
- Results so far (job 23683973, 2026-11-09, outputs under
  `/work/nvme/bicu/ytian6/kv_offload/23683973/`):
  - Pinned H2D 24.5 GB/s, D2H 26.3 GB/s (NUMA-bound, flat from 1 to 40 frames).
  - F20 (W=80) and F40 (W=160): every resident run OOMs on 48 GB; F40 offload
    runs were host-OOM-killed (exit137, ~44 GB pinned K/V vs `--mem`).
  - Only data point: F20 offload-timed at V=40, fetch 9.06 ms/layer vs
    compute 19.2 ms/layer, stall 9.2 ms/forward. Single window → V* fit is
    meaningless (reported ∞).
  - Next: rerun with smaller windows (e.g. W=20/40/60) that fit resident so
    there is a baseline, and raise `--mem` for large offload windows.

### KV bank + pre/post-RoPE option (implemented, validating)

- `core/attention/kvbank.py`: budgeted pinned-host `KVBank` with per-entry
  metadata (chunk or frame granularity), FIFO eviction, pluggable
  `KVRetrievalPolicy` (`RecentRetrieval`, `CameraPoseRetrieval`). Fed through
  `LayerwiseKVOffload(bank=...)`; attended layout `[sink | retrieved | window]`.
  Config: `Wan21TransformerConfig.kv_bank` (needs `kv_offload`).
- RoPE: `network.apply_rope_before_kvcache` now also works for LingBot.
  Post-RoPE keeps retrieved entries at their original positions; pre-RoPE
  places them at their retrieve slots.
- GPU validation job: `experiments/kv_offload/bank_test.slurm`
  (`sbatch` from the repo root; summary in `$SWEEP_ROOT/bank-<jobid>/summary.txt`).
  - Job 23840663 (2026-11-13): `ci_cpu` 2 failed —
    `test_kv_bank_fills_retrieve_region[2-chunk|2-frame]` raise
    `kv bank budget ... holds 1 entries; one chunk needs 2` (test budget vs
    the `kvbank.py` capacity check; unresolved). `ci_gpu` showed `..FF.Fs`
    but the log and all 5 generation runs died with `Errno 122` because the
    shared `/work/nvme/bicu` quota (550 GB hard) was full.
  - Freed 120 GB and resubmitted unchanged as job 23900078 (COMPLETED,
    results `/work/nvme/bicu/ytian6/kv_offload/bank-23900078/`):
    - `ci_cpu` still 2 failed (same budget check). `ci_gpu` 3 failed: the same
      budget error in both `_cuda` bank tests, plus
      `test_pre_and_post_rope_cache_match_without_roll[wan]` raising
      `Must call before_update() before computing visible cache size`
      (`test_rope_kvcache.py:72`); the lingbot variant is skipped (needs sm89+).
    - resident-post hit the 600 s `timeout` (first chunk took 227 s of
      compile/autotune), so its mp4 holds only the warmup chunk and the PSNRs
      below are against that partial video: not meaningful.
    - bank-chunk-post ok (steady 4.0 s/chunk, 3.97 fps, peak 21.9 GiB);
      bank-frame-post ok; "PSNR" 16.2 / 15.9 dB.
    - bank-chunk-pre and resident-pre: `InductorError: AssertionError` in
      torch.compile, so pre-RoPE fails under compile even without the bank.
  - Fixes (2026-11-13): `KVBank._allocate` capacity used `int()` on a float
    (239.99… B floored to one entry short) → `round()`; the RoPE test read
    `cached_k()` after `after_update()` → compares `_k`; pre-RoPE cached-K
    rotation now goes through `apply_rope_freqs_copy` (`rope.py`, eager,
    graph break, `ponytail:`), since Inductor asserts on that Triton kernel;
    `bank_test.slurm` runs resident first, PSNR vs the same-RoPE resident only
    if it finished, persistent inductor/Triton cache on `/work`, and splits via
    `RUNS`/`TESTS`/`OUT` into two jobs that fit interactive's 1 h cap (regular
    `gpuA40x4` queues for a week; interactive allows 1 job per user, so job B
    is submitted after A ends). `ci_cpu` bank tests 6/6 pass on the login node.
    Rerun: job A 23910329 (tests + resident-post/pre), then job B (bank runs),
    shared output `/work/nvme/bicu/ytian6/kv_offload/bank-1113-1640/`.
- Reference implementation: WorldKV (`git clone
  git@github.com:cvlab-kaist/WorldKV.git WorldKV && git -C WorldKV checkout
  046f6d1`, gitignored). `CameraPoseRetrieval` mirrors its
  `image2video_fast.py`.
- File/line map: `WORLDCACHE_IMPL.md` §3–§5.

### Bank as the only host store (2026-11-18, `8c4a9932`, A100-validated)

- Why: with a full pinned home per layer, `BlockKVCache`'s window roll ran
  on host memory (clone + copy of the whole window): ~1.84 s of a 5.1 s chunk
  for LingBot 1.3B W40/S6 (`compare-24082157/offload-w40/run.log`, timing
  lines from `WanDiTNetworkCache.before_update`). With a bank, window chunks
  also existed twice on host and were written back twice.
- Now: home is gone (sink buffer only); the window is `KVBank.window()` index
  lookups over the newest entries; bank capacity is at least the window and
  evicting an in-window entry raises. Details and line map:
  `WORLDCACHE_IMPL.md` §6.
- `ci_cpu` `test_kvcache.py` 19 passed (offload parity now parametrized over
  chunk-unaligned sink/window). `ci_gpu` tests still pending.
- A100 job 24131186 (`compare-1118-1550`, commit `99b072de`): `roll` < 2 ms
  per chunk; offload-w40 5.10 → 3.29 s/chunk (4.86 fps), ours 4.81 → 3.74
  s/chunk (4.28 fps), ours host RSS 50.9 → 39.5 GiB, GPU peak unchanged
  (22.3 / 22.5 GiB). Revisit PSNR 13.15 / 13.41 is within run-to-run noise
  (repeat runs of one config differ by ~22–25 dB PSNR).
- Steady-state per forward (30 layers): offload-w40 fetch 523 / compute 544 /
  stall 39 ms, 12.8 GB; ours 615 / 622 / 54 ms, 15.0 GB. H2D runs at
  ~24.5 GB/s (pinned peak) but is hidden behind compute: stall is ~7% of a
  forward, and offload-w40 is only 4% slower than resident-w40 (3.29 vs 3.15
  s/chunk). ours vs offload-w40 (+0.45 s/chunk) is entirely the 8 retrieved
  frames: attention covers 54 vs 46 frames (15.0/12.8 GB = 54/46), +78 ms
  compute × 5 forwards ≈ the +0.40 s diffuse gap; retrieval host cost is
  negligible.

### RTX PRO 6000 Blackwell rerun (2026-11-20, `4049e1d4`)

All numbers in this section are from a dedicated **RTX PRO 6000 Blackwell
Server Edition** (96 GB, sm120, PCIe Gen5 x16, one NUMA node), not A100/A40.
Same bf16 `kv_sweep` app and `compare.slurm` runs as the A100 job, run with
`bash` (no Slurm). Outputs: `/mnt/datastore/yiti6755/kv_offload/`
(`h2d-1120-2155/`, `compare-1120-rtx6000/`).

- Pinned H2D 57.8 GB/s, D2H 57.3 GB/s, flat from 1 to 160 frames (2.4× A100).
- s/chunk (median after 2 warmup chunks): baseline W12 1.16 (13.8 fps),
  resident-w40 2.04, offload-w40 2.08, ours 2.32 (6.9 fps). GPU peak 26.4 /
  33.6 / 22.4 / 22.5 GiB; ours host RSS 39.9 GiB. Revisit PSNR 15.5–15.7 for
  all four on full frames; 14.3–14.5 with the UI panel masked (see below).
  Noise either way.
- Steady state per layer: offload-w40 fetch 7.40 / compute 11.85 ms
  (ratio 0.62), ours 8.69 / 13.66 ms (0.64); stall 2.1% of a forward in both
  (A100: ratio ≈ 0.96, stall ~7%). H2D runs at 57.6 GB/s. offload-w40 is 1.7%
  slower than resident-w40 (A100: 4%). **Compute still hides fetch**, with
  more margin than on A100: bandwidth grew 2.4×, bf16 compute only 1.5×.
- Caveat: sm120 has FP8; FP8 linears/attention would shrink compute and could
  bring fetch back to parity. Not measured yet.

### Window study, camera-follow check, UI-panel metric fix (2026-11-24)

- **flashdreams `out.mp4` has the cam2v "Camera controls" UI panel burned in**
  (top left, x < ~0.45 W, changing text) and is encoded at ~4.5 Mbps
  (official LingBot: no panel, ~15 Mbps). Every video metric on full frames
  was contaminated. `compare_metrics.py` now keeps only x >= `--x0` × W
  (default 0.5; `--x0 0` for full frames). Old full-frame numbers are kept as
  `metrics_fullframe.*` / `results_fullframe.md` next to the new ones.
- Window study (`/mnt/datastore/yiti6755/kv_offload/win-study/`, seeds
  42/43/44, `compare.slurm` `res-w<W>` / `ret-w<W>-r<R>`, panel masked):
  resident W 12–84 revisit PSNR 13.7–14.2 dB, std 0.5–1.0, no trend. At
  equal total T = 16/24/32, retrieval `ret-w8-r<T-8>` vs sliding `res-w<T>`,
  paired by seed: far pairs (>= 200 frames apart) +0.45 / +0.27 / +0.34 dB,
  better in 7 of 9 pairs; GPU 22 vs 27–31 GiB at +2–3% s/chunk. A weak
  signal; WorldKV's own code on the 14B model gives ~+0.9 dB at window 18,
  winning on all 3 seeds (`WorldKV/repro/out/results.md`, gitignored clone).
  More seeds and clean recordings are needed before calling it a gain.
- Camera following vs official LingBot-World-V2 1.3B (upstream `1895d30` +
  SDPA fallback, same image/prompt/poses, sink 6 + window 12, seeds
  42/43/44; scripts and outputs in `/mnt/datastore/yiti6755/lingbot-upstream/`).
  Optical-flow yaw over x >= 0.5 W: gain (measured / commanded yaw) flashdreams
  0.99, official 1.00; per-frame correlation 0.83 vs 0.85. Same for every
  win-study config (0.98–1.05). An earlier "flashdreams turns 20% less"
  reading (0.78) came from the panel inside the flow ROI. The CPU checks done
  while chasing it still hold: Plücker inputs match upstream to bf16 rounding
  (same 88 latent poses, intrinsics, rays), and the 4-step sampler matches
  (timesteps 999/937/833/624, sigmas within 6e-4, clean-latent KV refresh).
- Blur drift (Laplacian sharpness, last / first quarter, x >= 0.5 W): ~0.67
  for flashdreams baseline, ~0.74 for official (0.69 / 0.86 / 0.68). Both
  models do it; the ratio also tracks which way the camera faces, so it is
  only a rough cue. The panel's text had hidden it (0.94 on full frames).

### LingBot v2 14B: preset fix and offload (2026-11-24)

- `PIPELINE_LINGBOT_WORLD_V2_14B_CAUSAL_FAST` was a copy of v1 Fast:
  `len_t` 3, timesteps 999/978/947/825, window 63 / sink 0. Upstream v2
  `generate.py` + README run 14B with chunk 4, timesteps_index
  [0, 250, 500, 750] (999/967/908/768 at shift 10), local_attn_size 18 /
  sink 6. The preset now uses `len_t=4`, `denoising_timesteps=[1000, 750,
  500, 250]`, window 12 / sink 6 (18 frames in all, as in the 1.3B preset).
- `kv_sweep` takes `KV_SWEEP_MODEL=1p3b|14b` and forces
  `use_cuda_graph=False` (offload needs it; the 14B preset had graphs on).
  `compare.slurm` passes `MODEL`.
- RTX PRO 6000, bf16, `compare-14b-1124/` (seed 42, 22 chunks, timing on):
  baseline W12 5.96 s/chunk, 69.3 GiB GPU; **resident-w40 OOMs** (95 GiB);
  offload-w40 9.58 s/chunk, **51.2 GiB**, host RSS 93 GiB; ours (+8
  retrieved, bank 112 GB) 10.52 s/chunk, 51.6 GiB, host RSS 183 GiB.
- Steady state per layer (40 layers): offload-w40 fetch 24.6 / compute
  43.2 ms (ratio 0.57), ours 28.9 / 50.6 ms (0.57); stall 1.4% of a forward;
  56.8 / 66.7 GB H2D per forward at 57.7 GB/s. **Compute still hides fetch
  on 14B**, same margin as 1.3B: fetch grows with the window, but so does
  attention. Offload lets 14B run a window that does not fit resident, with
  less GPU memory than the W12 baseline.
- Quality (one seed, panel masked): revisit PSNR 9.25 / 9.60 / 9.06 dB
  (baseline / offload-w40 / ours). Noise; 14B frames have a large moving
  horse and rider in the foreground, so these are not comparable to 1.3B
  numbers. Camera following is fine (flow yaw gain 1.04, corr 0.87).

### WorldKV setting, resident bank, and run-to-run determinism (2026-11-25)

- New: `kv_bank` without `kv_offload` keeps caches on the device
  (`LayerwiseKVOffload(resident=True)`, WorldKV's layout): only bank entries
  cross PCIe, the retrieve region is filled once per chunk.
  `CameraPoseRetrieval.min_gap_t` skips candidates ending within that many
  frames of the query (WorldKV excludes its 6 "recent" frames). `kv_sweep`
  has `KV_SWEEP_MODEL=v1fast` (v1 LingBot-World-Fast 14B, `len_t` 3) and
  `KV_SWEEP_RETRIEVE_GAP_T`; `compare.slurm` has `wkv-baseline` (W12/S6),
  `wkv-resident` (S3 + 9 retrieved + W6, gap 6), `wkv-ours` (same + offload).
- RTX PRO 6000, v1fast, 30 chunks, seeds 42/43/44, panel masked
  (`/mnt/datastore/yiti6755/kv_offload/wkv-1125{,-bank120}/`): s/chunk 4.39–4.42
  / 4.48–4.52 / 4.43–4.45, GPU 66.8 / 66.8 / **47.5 GiB** (baseline /
  resident / ours). Mean revisit PSNR, bank 24 GB: 9.81 / 9.57 / 8.70 dB;
  bank 120 GB (all 30 chunks retrievable): resident 9.00, ours 7.86. Host RSS
  ~180 GB with the 120 GB bank; its pinned allocation stretches a run from
  ~4 to 8 (resident) / 24 (ours) min, per-chunk speed unchanged. No retrieval
  gain over baseline, unlike WorldKV's own +0.9 dB.
- ours < resident in 6/6 seed×bank pairs (mean −1.0 dB) looked like an
  offload bug. **It is not: offload and resident are bit-identical.** Eager
  1.3B, WorldKV layout scaled to `len_t` 4 (S4 + R8 + W8, gap 8, chunk bank),
  8 chunks: decoded latents of resident vs offload are bit-equal for every
  chunk, with and without the bank; bank vs no bank diverge exactly at chunk 5
  when retrieval starts, and the offload arm streamed 5.56 GB H2D per forward.
  `test_kvcache.py` `ci_gpu` 15 passed (first run since bank-as-only-store).
- **Compiled runs are not reproducible.** Same config, same seed, twice
  (frame PSNR of the two videos): compiled 33 dB at frame 1 → 25 dB by chunk
  3, drifting (Inductor `max-autotune` kernel choice, amplified by the
  rollout); `--no-compile` ~38 dB flat with **bit-equal latents** (the rest is
  VAE decode / mp4 encode, which never feeds back). Deterministic algorithms
  and an eager VAE decoder change nothing. Every earlier quality comparison
  (window study's +0.3–0.45 dB, 14B, wkv-1125) is one compiled recording per
  seed, so differences below ~1 dB there are noise, not evidence.
- Tool: `experiments/kv_offload/latent_parity.py` dumps decoded latents and
  compares two dumps; `compare.slurm` passes `APP_ARGS` (e.g. `--no-compile`).

### Known gaps

- End-to-end bank runs (`bank-1113-1640`, all five ok; A40 bank ≈ +26%
  s/chunk vs resident, −3.8 GiB GPU) are in `WORLDCACHE_EXPERIMENTS.md` §3.
- Offload vs resident parity is shown bit-exact only on eager 1.3B, 8 chunks,
  no bank eviction; not on v1fast 14B / 30 chunks.
- No quality gain from retrieval shown yet: all quality runs so far are
  compiled, single-recording, and within noise.
- Each prefetch re-copies the whole visible prefix on every denoising step
  (marked `ponytail:` in `_prefetch`). Inherent to two device slots: keeping
  history across steps means keeping every layer resident. Hidden at W40 on
  A100 (fetch ≈ compute) and on RTX PRO 6000 bf16 (fetch ≈ 0.62× compute);
  bytes grow linearly with visible length, so longer windows or faster
  compute (e.g. FP8) will make it the bottleneck.
- Not CUDA-graph compatible (slot rebinding + side stream).
- Covers only `BlockKVCache` in the Wan recipe; not a general policy interface.

## Resuming On A New Machine

1. Read this file, then `WORLDCACHE_IMPL.md` (file/line map; keep it updated
   with every KV change and bump its baseline commit).
   To rerun tests and experiments (setup, path overrides, every run's meaning,
   historical numbers to compare against): `WORLDCACHE_EXPERIMENTS.md`.
2. Clone WorldKV as above if you need the reference code.
3. Slurm scripts hard-code Delta paths (`/work/nvme/bicu/ytian6`,
   account `bicu-delta-gpu`); override `SWEEP_ROOT`/`HF_HOME` elsewhere.
   Current host (since 2026-11-20): a dedicated RTX PRO 6000 without Slurm;
   run the `.slurm` scripts with `bash` (see `WORLDCACHE_EXPERIMENTS.md` §0).
4. On Delta, watch the group quota (`quota`): `/work/nvme/bicu` is shared and
   was full.

## Next Steps

1. Validate: done (`test_kvcache.py` `ci_cpu` 23 / `ci_gpu` 15 passed,
   2026-11-25); rerun both after every KV change.
2. GPU run of the full-KV offload runner with timing enabled; compare output
   and latency against `lingbot-world-v2-1p3b-causal-fast-perf`
   (`skills/profile-model-performance`, `skills/validate-performance-quality`).
3. Deferred until H2D stalls dominate (not at W40 on A100 or RTX PRO 6000
   bf16, 1.3B or 14B, see Status; recheck with FP8 on RTX PRO 6000): a
   `resident_layers` knob (keep the first M layers' history on device; trades
   memory for H2D). Skipping the write range in prefetch only saves ~9% of
   bytes and no wall time while fetch is hidden.
4. Bank-as-only-store change: done (`compare.slurm` job 24131186; `ci_gpu`
   `test_kvcache.py` 15 passed 2026-11-25).
5. Quality comparisons: run eager (`APP_ARGS=--no-compile`, deterministic, so
   arms pair exactly per seed) or use >= 8 seeds compiled. First rerun the
   WorldKV setting (`wkv-*`, v1fast) eager to answer whether retrieval helps
   at all; then the T = 16/24/32 retrieval vs sliding comparison. Also turn
   off the cam2v UI overlay and raise the mp4 bitrate.
6. Design a model-agnostic KV policy interface in `core` (storage tier +
   eviction/retrieval policy) and port window/sink, full, and offload onto it.
7. Add chunk-level KV identity and a store/lookup path (LMCache-style reuse).
