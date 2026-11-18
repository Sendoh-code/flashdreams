#!/bin/bash
# LingBot 1.3B: original KV window vs layerwise offload + camera-pose KV bank, one A100.
#
#   sbatch experiments/kv_offload/compare.slurm          # all runs (may exceed 1 h cold)
#
# Interactive caps jobs at 1 h, so split into two jobs sharing OUT and the compile cache:
#   OUT=/work/nvme/bicu/ytian6/kv_offload/compare-$(date +%m%d-%H%M); export OUT
#   A=$(sbatch --parsable --export=ALL,RUNS="baseline resident-w40" experiments/kv_offload/compare.slurm)
#   sbatch --dependency=afterany:$A --export=ALL,RUNS="offload-w40 ours" experiments/kv_offload/compare.slurm
#
# Runs (bf16 torch linears + wan attention in both arms: A100 has no FP8; post-RoPE):
#   baseline     resident window 12 / sink 6 (the shipped 1.3B KV policy)
#   resident-w40 resident window 40 / sink 6 (what the long window costs without offload)
#   offload-w40  layerwise host offload, window 40 / sink 6
#   ours         offload-w40 + camera-pose KV bank, 8 retrieved frames per forward
#   ours-w8      offload, window 8 / sink 6 + 8 retrieved frames: WorldKV's
#                sink + retrieved + recent layout (recent = current + 1 chunk)
# Diagnostics: FLASHDREAMS_KV_OFFLOAD_TIMING=1 writes <run>/kv_timing.jsonl (per-layer
# fetch/stall/compute/writeback); H2D_BENCH=1 appends h2d_bench.py results.
# Trajectory: WorldKV R_L_R_L_R_L90_360 (in-place yaw sweeps that revisit the same
# views ~120 frames apart, beyond the baseline window). Each run writes out.mp4,
# stats.json, run.log; metrics.{json,md} compare speed, memory, revisit PSNR/SSIM.
#SBATCH --job-name=kv-compare
#SBATCH --account=bicu-delta-gpu
#SBATCH --partition=gpuA100x4-interactive
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --mem=128G
#SBATCH --cpus-per-task=16
#SBATCH --time=01:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/kv_offload/logs/%x-%j.out

set -uo pipefail
module reset >/dev/null 2>&1 || true
module load ffmpeg/7.1 >/dev/null 2>&1 || true
unset VIRTUAL_ENV

REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
OUT=${OUT:-${SWEEP_ROOT:-/work/nvme/bicu/ytian6/kv_offload}/compare-${SLURM_JOB_ID:-local}}
mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/work/nvme/bicu/ytian6/hf_cache}
export TORCHINDUCTOR_CACHE_DIR=${TORCHINDUCTOR_CACHE_DIR:-/work/nvme/bicu/ytian6/inductor_cache}
export TRITON_CACHE_DIR=${TRITON_CACHE_DIR:-$TORCHINDUCTOR_CACHE_DIR/triton}
PY="$REPO/.venv/bin"
POSES=${POSES:-$REPO/WorldKV/traj/R_L_R_L_R_L90_360/R_L_R_L_R_L90_360_poses.npy}
BLOCKS=${BLOCKS:-22}  # 360 poses -> 88 latent frames -> 22 chunks of 4
{ nvidia-smi -L; git rev-parse HEAD; echo "POSES=$POSES BLOCKS=$BLOCKS"; } | tee -a "$OUT/env.txt"

run_one() {  # name window sink retrieve offload
  local name=$1 dir="$OUT/$1"
  mkdir -p "$dir"
  echo "=== $name ($(date +%T))"
  KV_SWEEP_WINDOW_T=$2 KV_SWEEP_SINK_T=$3 KV_SWEEP_RETRIEVE_T=$4 KV_SWEEP_OFFLOAD=$5 \
  KV_SWEEP_ROPE=post KV_SWEEP_BANK_GRANULARITY=chunk KV_SWEEP_BANK_GB=${BANK_GB:-24} \
  FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=${FLASHDREAMS_KV_OFFLOAD_TIMING:+$dir/kv_timing.jsonl} \
  timeout "${RUN_TIMEOUT:-1500}" /usr/bin/time -v "$PY/flashdreams-run-v2" lingbot.apps.cam2v.kv_sweep \
    --mode mp4 --output-path "$dir/out.mp4" --timeout unbound \
    --backpressure-mode block --presentation-mode on_demand \
    --stats-path "$dir/stats.json" \
    -- --example-data --example-idx 0 --pose-path "$POSES" \
    --total-blocks "$BLOCKS" --warmup-blocks 2 --seed 42 \
    > "$dir/run.log" 2>&1
  local rc=$? status=ok
  if [[ $rc -ne 0 ]]; then
    status="exit$rc"
    # Only a failed run counts as OOM; Triton autotune logs "OutOfMemoryError: out of resource".
    grep -q "CUDA out of memory\|torch.OutOfMemoryError" "$dir/run.log" && status=oom
  else
    touch "$dir/ok"
  fi
  echo "$name status=$status" | tee -a "$OUT/summary.txt"
}

declare -A ARGS=(  # name -> window sink retrieve offload
  [baseline]="12 6 0 0" [resident-w40]="40 6 0 0"
  [offload-w40]="40 6 0 1" [ours]="40 6 8 1"
  [ours-w8]="8 6 8 1"
)
for name in ${RUNS:-baseline resident-w40 offload-w40 ours}; do
  run_one "$name" ${ARGS[$name]}
done
"$PY/python" experiments/kv_offload/compare_metrics.py "$OUT" --poses "$POSES" \
  > "$OUT/metrics.log" 2>&1 || echo "metrics failed (see $OUT/metrics.log)"
if [[ -n ${H2D_BENCH:-} ]]; then  # raw pinned-copy GB/s, NUMA-bound and unbound
  BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader | head -1 | tr 'A-F' 'a-f' | sed 's/^0000//')
  NODE=$(cat "/sys/bus/pci/devices/$BUS/numa_node" 2>/dev/null || echo -1)
  { nvidia-smi topo -m; numactl -H; echo "BUS=$BUS NUMA=$NODE"; } >> "$OUT/env.txt" 2>&1
  [[ $NODE -ge 0 ]] && numactl --cpunodebind="$NODE" --preferred="$NODE" \
    "$PY/python" experiments/kv_offload/h2d_bench.py bound > "$OUT/h2d.jsonl" 2>&1
  "$PY/python" experiments/kv_offload/h2d_bench.py unbound >> "$OUT/h2d.jsonl" 2>&1
  echo "h2d_bench -> $OUT/h2d.jsonl"
fi
echo "results: $OUT"
