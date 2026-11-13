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

- `LayerwiseKVOffload` keeps each `BlockKVCache` K/V home in pinned host
  memory and streams layers through two GPU slots (reusing the first two
  caches' buffers to keep custom layouts such as LingBot's TMA padding).
  `before_layer(i)` prefetches layer `i+1` on a side stream; `after_layer(i)`
  writes back only the current write range. One offload is shared by CFG
  cond and uncond.
- `synchronize()` must run before the host-side window roll; it is wired into
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
  - Freed 120 GB and resubmitted unchanged as job 23900078; check its
    `experiments/kv_offload/logs/kv-bank-test-23900078.out`.
- Reference implementation: WorldKV (`git clone
  git@github.com:cvlab-kaist/WorldKV.git WorldKV && git -C WorldKV checkout
  046f6d1`, gitignored). `CameraPoseRetrieval` mirrors its
  `image2video_fast.py`.
- File/line map: `WORLDCACHE_IMPL.md` §3–§5.

### Known gaps

- `ci_cpu` KV tests run on Delta; 2 bank tests fail (see above). GPU tests
  and the end-to-end bank runs have no clean result yet.
- No quality parity vs resident KV, no perf numbers beyond the one sweep point.
- H2D-bandwidth bound: each prefetch re-copies the whole visible prefix on
  every denoising step (marked `ponytail:` in `_prefetch`).
- Not CUDA-graph compatible (slot rebinding + side stream).
- Covers only `BlockKVCache` in the Wan recipe; not a general policy interface.

## Resuming On A New Machine

1. Read this file, then `WORLDCACHE_IMPL.md` (file/line map; keep it updated
   with every KV change and bump its baseline commit).
2. Clone WorldKV as above if you need the reference code.
3. Slurm scripts hard-code Delta paths (`/work/nvme/bicu/ytian6`,
   account `bicu-delta-gpu`); override `SWEEP_ROOT`/`HF_HOME` elsewhere.
4. Watch the group quota (`quota`): `/work/nvme/bicu` is shared and was full.

## Next Steps

1. Validate: `uv run pytest -m ci_cpu flashdreams/tests/test_kvcache.py -k layerwise`,
   then the `ci_gpu` timing test on a GPU host.
2. GPU run of the full-KV offload runner with timing enabled; compare output
   and latency against `lingbot-world-v2-1p3b-causal-fast-perf`
   (`skills/profile-model-performance`, `skills/validate-performance-quality`).
3. If H2D dominates, remove the `ponytail:` ceiling: skip the write range in
   prefetch and keep a chunk's history resident across its denoising steps.
4. Design a model-agnostic KV policy interface in `core` (storage tier +
   eviction/retrieval policy) and port window/sink, full, and offload onto it.
5. Add chunk-level KV identity and a store/lookup path (LMCache-style reuse).
