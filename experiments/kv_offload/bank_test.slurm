#!/bin/bash
# KV bank + pre/post-RoPE validation on one Delta A40 (interactive partition, 1h cap).
#
#   sbatch experiments/kv_offload/bank_test.slurm
#
# 1. Unit tests: ci_cpu bank/offload + ci_gpu bank (CUDA streams), offload timing,
#    LingBot pre- vs post-RoPE block parity.
# 2. LingBot 1.3B end-to-end (window 12, sink 4, 16 blocks): resident post/pre-RoPE
#    baselines, then KV bank (camera-pose retrieval, 8 retrieved frames) at chunk
#    and frame granularity, post- and pre-RoPE. Each run writes out.mp4,
#    stats.json and (bank runs) kv_timing.jsonl; summary.txt has status and PSNR
#    vs the resident post-RoPE baseline.
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
OUT=${SWEEP_ROOT:-/work/nvme/bicu/ytian6/kv_offload}/bank-${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/work/nvme/bicu/ytian6/hf_cache}
PY="$REPO/.venv/bin"
nvidia-smi -L | tee "$OUT/env.txt"

echo "=== unit tests ($(date +%T))"
"$PY/pytest" -q -p no:cacheprovider -m ci_cpu flashdreams/tests/test_kvcache.py \
  -k "bank or camera or layerwise" > "$OUT/pytest_cpu.log" 2>&1
echo "    ci_cpu -> rc=$? ($(tail -1 "$OUT/pytest_cpu.log"))"
"$PY/pytest" -q -p no:cacheprovider -m ci_gpu flashdreams/tests/test_kvcache.py \
  integrations_v2/lingbot/tests/test_rope_kvcache.py -k "bank or timing or rope" \
  -rs > "$OUT/pytest_gpu.log" 2>&1
echo "    ci_gpu -> rc=$? ($(tail -1 "$OUT/pytest_gpu.log"))"

BLOCKS=${BLOCKS:-16}
run_one() {  # name rope retrieve granularity offload
  local name=$1 dir="$OUT/$1"
  mkdir -p "$dir"
  echo "=== $name ($(date +%T))"
  KV_SWEEP_WINDOW_T=12 KV_SWEEP_SINK_T=4 KV_SWEEP_ROPE=$2 KV_SWEEP_RETRIEVE_T=$3 \
  KV_SWEEP_BANK_GRANULARITY=$4 KV_SWEEP_OFFLOAD=$5 KV_SWEEP_BANK_GB=4 \
  FLASHDREAMS_KV_OFFLOAD_TIMING=$5 \
  FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=$([[ $5 == 1 ]] && echo "$dir/kv_timing.jsonl") \
  timeout "${RUN_TIMEOUT:-600}" "$PY/flashdreams-run-v2" lingbot.apps.cam2v.kv_sweep \
    --mode mp4 --output-path "$dir/out.mp4" --timeout unbound \
    --backpressure-mode block --presentation-mode on_demand \
    --stats-path "$dir/stats.json" \
    -- --example-data --total-blocks "$BLOCKS" --warmup-blocks 2 --seed 42 \
    > "$dir/run.log" 2>&1
  local rc=$? status=ok
  [[ $rc -ne 0 ]] && status="exit$rc"
  grep -q "OutOfMemoryError\|CUDA out of memory" "$dir/run.log" && status=oom
  local psnr=-
  if [[ $status == ok && $name != resident-post && -f "$OUT/resident-post/out.mp4" ]]; then
    psnr=$(ffmpeg -nostats -i "$dir/out.mp4" -i "$OUT/resident-post/out.mp4" \
      -lavfi psnr -f null - 2>&1 | grep -o "average:[^ ]*" | tail -1)
  fi
  echo "$name status=$status psnr_vs_resident_post=$psnr" | tee -a "$OUT/summary.txt"
}

#       name             rope retrieve granularity offload
run_one resident-post    post 0 chunk 0
run_one bank-chunk-post  post 8 chunk 1
run_one bank-chunk-pre   pre  8 chunk 1
run_one resident-pre     pre  0 chunk 0
run_one bank-frame-post  post 8 frame 1
echo "results: $OUT"
