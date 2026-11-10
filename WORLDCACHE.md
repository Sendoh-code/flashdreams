# WorldCache: KV-Centric Serving for Causal World Models

Working notes for the WorldCache fork of FlashDreams. Read this first when
resuming the work on a new machine or in a new agent session. Update the
**Status** and **Next Steps** sections whenever they change.

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

### Known gaps

- Tests have not been run yet (the original machine had `uv` < 0.11.8).
- No GPU end-to-end run, no quality parity vs resident KV, no perf numbers.
- H2D-bandwidth bound: each prefetch re-copies the whole visible prefix on
  every denoising step (marked `ponytail:` in `_prefetch`).
- Not CUDA-graph compatible (slot rebinding + side stream).
- Covers only `BlockKVCache` in the Wan recipe; not a general policy interface.

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
