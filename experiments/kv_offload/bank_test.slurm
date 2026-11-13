#!/bin/bash
# KV bank + pre/post-RoPE validation on one Delta A40.
#
#   sbatch experiments/kv_offload/bank_test.slurm            # everything (needs >1h)
#
# Interactive A40 caps jobs at 1h, so split into two jobs sharing one output dir and
# a persistent inductor cache (job B reuses job A's compile):
#   OUT=/work/nvme/bicu/ytian6/kv_offload/bank-$(date +%m%d-%H%M); export OUT
#   A=$(sbatch --parsable --export=ALL,RUNS="resident-post resident-pre" \
#       experiments/kv_offload/bank_test.slurm)
#   sbatch --dependency=afterany:$A --export=ALL,TESTS=0,RUNS="bank-chunk-post bank-chunk-pre bank-frame-post" \
#       experiments/kv_offload/bank_test.slurm
#
# 1. Unit tests: ci_cpu bank/offload + ci_gpu bank (CUDA streams), offload timing,
#    LingBot pre- vs post-RoPE block parity.
# 2. LingBot 1.3B end-to-end (window 12, sink 4, 16 blocks): resident post/pre-RoPE
#    baselines, then KV bank (camera-pose retrieval, 8 retrieved frames) at chunk
#    and frame granularity, post- and pre-RoPE. Each run writes out.mp4,
#    stats.json and (bank runs) kv_timing.jsonl; summary.txt has status and PSNR
#    vs the resident baseline with the same RoPE mode (only if that one finished).
#SBATCH --job-name=kv-bank-test
#SBATCH --account=bicu-delta-gpu
#SBATCH --partition=gpuA40x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=64G
#SBATCH --cpus-per-task=16
#SBATCH --constraint=scratch
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/kv_offload/logs/%x-%j.out

set -uo pipefail
module reset >/dev/null 2>&1 || true
module load ffmpeg/7.1 >/dev/null 2>&1 || true
unset VIRTUAL_ENV

REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
OUT=${OUT:-${SWEEP_ROOT:-/work/nvme/bicu/ytian6/kv_offload}/bank-${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/work/nvme/bicu/ytian6/hf_cache}
# Persistent compile caches: a cold resident run spent 227 s compiling its first chunk.
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/work/nvme/bicu/ytian6/inductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR/triton}
PY="$REPO/.venv/bin"
nvidia-smi -L | tee -a "$OUT/env.txt"

if [[ ${TESTS:-1} == 1 ]]; then
echo "=== unit tests ($(date +%T))"
"$PY/pytest" -q -p no:cacheprovider -m ci_cpu flashdreams/tests/test_kvcache.py \
  -k "bank or camera or layerwise" > "$OUT/pytest_cpu.log" 2>&1
echo "    ci_cpu -> rc=$? ($(tail -1 "$OUT/pytest_cpu.log"))"
"$PY/pytest" -q -p no:cacheprovider -m ci_gpu flashdreams/tests/test_kvcache.py \
  integrations_v2/lingbot/tests/test_rope_kvcache.py -k "bank or timing or rope" \
  -rs > "$OUT/pytest_gpu.log" 2>&1
echo "    ci_gpu -> rc=$? ($(tail -1 "$OUT/pytest_gpu.log"))"
fi

BLOCKS=${BLOCKS:-16}
run_one() {  # name rope retrieve granularity offload
  local name=$1 dir="$OUT/$1"
  mkdir -p "$dir"
  echo "=== $name ($(date +%T))"
  KV_SWEEP_WINDOW_T=12 KV_SWEEP_SINK_T=4 KV_SWEEP_ROPE=$2 KV_SWEEP_RETRIEVE_T=$3 \
  KV_SWEEP_BANK_GRANULARITY=$4 KV_SWEEP_OFFLOAD=$5 KV_SWEEP_BANK_GB=4 \
  FLASHDREAMS_KV_OFFLOAD_TIMING=$5 \
  FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=$([[ $5 == 1 ]] && echo "$dir/kv_timing.jsonl") \
  timeout "${RUN_TIMEOUT:-900}" "$PY/flashdreams-run-v2" lingbot.apps.cam2v.kv_sweep \
    --mode mp4 --output-path "$dir/out.mp4" --timeout unbound \
    --backpressure-mode block --presentation-mode on_demand \
    --stats-path "$dir/stats.json" \
    -- --example-data --total-blocks "$BLOCKS" --warmup-blocks 2 --seed 42 \
    > "$dir/run.log" 2>&1
  local rc=$? status=ok
  [[ $rc -ne 0 ]] && status="exit$rc"
  grep -q "OutOfMemoryError\|CUDA out of memory" "$dir/run.log" && status=oom
  local psnr=- ref="$OUT/resident-$2"
  if [[ $status == ok && $name != resident-* && -f "$ref/ok" ]]; then
    psnr=$(ffmpeg -nostats -i "$dir/out.mp4" -i "$ref/out.mp4" \
      -lavfi psnr -f null - 2>&1 | grep -o "average:[^ ]*" | tail -1)
  fi
  [[ $status == ok ]] && touch "$dir/ok"
  echo "$name status=$status psnr_vs_resident_$2=$psnr" | tee -a "$OUT/summary.txt"
}

declare -A ARGS=(  # name -> rope retrieve granularity offload
  [resident-post]="post 0 chunk 0" [resident-pre]="pre 0 chunk 0"
  [bank-chunk-post]="post 8 chunk 1" [bank-chunk-pre]="pre 8 chunk 1"
  [bank-frame-post]="post 8 frame 1"
)
for name in ${RUNS:-resident-post resident-pre bank-chunk-post bank-chunk-pre bank-frame-post}; do
  run_one "$name" ${ARGS[$name]}
done
echo "results: $OUT"
