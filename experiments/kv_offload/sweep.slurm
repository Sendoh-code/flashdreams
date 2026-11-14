#!/bin/bash
# KV window / host-offload H2D sweep for LingBot 1.3B on one Delta A40 48GB.
#
#   sbatch --export=ALL,SWEEP_GROUPS="F40" experiments/kv_offload/sweep.slurm
#   sbatch --export=ALL,SWEEP_GROUPS="B0 F20 M1 M2 M3 M4" experiments/kv_offload/sweep.slurm
#   # Short version on the interactive partition (1h cap, shorter queue):
#   sbatch -p gpuA40x4-interactive -t 01:00:00 -J kv-offload-setup \
#     --export=ALL,SWEEP_GROUPS="F40 F20",RUN_TIMEOUT=900 experiments/kv_offload/sweep.slurm
#
# First time, in an interactive job (downloads weights, checks GPU timing and
# raw PCIe bandwidth, bound to the GPU's NUMA node and unbound):
#   salloc -A bicu-delta-gpu -p gpuA40x4-interactive --gpus-per-node=1 \
#     --cpus-per-task=16 --mem=64G -t 01:00:00
#   export HF_HOME=/work/nvme/bicu/ytian6/hf_cache
#   KV_SWEEP_WINDOW_T=12 KV_SWEEP_SINK_T=6 srun .venv/bin/flashdreams-run-v2 \
#     lingbot.apps.cam2v.kv_sweep --preload-application -- --example-data
#   srun .venv/bin/pytest -m ci_gpu flashdreams/tests/test_kvcache.py -k timing
#   srun numactl --cpunodebind=N --preferred=N .venv/bin/python \
#     experiments/kv_offload/h2d_bench.py bound   # N: see nvidia-smi topo -m
#   srun .venv/bin/python experiments/kv_offload/h2d_bench.py unbound
#
# Runs are serial on purpose: sibling-GPU jobs would share host memory
# bandwidth and skew H2D numbers. Results: $SWEEP_ROOT/<jobid>/<run>/.
#SBATCH --job-name=kv-offload-sweep
#SBATCH --account=bicu-delta-gpu
#SBATCH --partition=gpuA40x4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --gpu-bind=closest
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --cpus-per-task=16
#SBATCH --constraint=scratch
#SBATCH --time=08:00:00
#SBATCH --no-requeue
#SBATCH --output=experiments/kv_offload/logs/%x-%j.out

set -uo pipefail
module reset >/dev/null 2>&1 || true
module load ffmpeg/7.1 >/dev/null 2>&1 || true  # mp4 sink needs ffmpeg on PATH
unset VIRTUAL_ENV

REPO=${SLURM_SUBMIT_DIR:-$PWD}
cd "$REPO"
OUT=${SWEEP_ROOT:-/work/nvme/bicu/ytian6/kv_offload}/${SLURM_JOB_ID:-local}
mkdir -p "$OUT"
export HF_HOME=${HF_HOME:-/work/nvme/bicu/ytian6/hf_cache}
PY="$REPO/.venv/bin"

# Pin torch and dmon to the same (only allocated) GPU, and bind to its NUMA node.
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
export CUDA_VISIBLE_DEVICES=$GPU_UUID
BUS=$(nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader -i "$GPU_UUID" | tr 'A-F' 'a-f' | sed 's/^0000//')
NODE=$(cat "/sys/bus/pci/devices/$BUS/numa_node" 2>/dev/null || echo -1)
BIND=()
# --preferred, not --membind: F40 offload pins ~44 GB of host K/V (160 frames x
# 9.26 MB x 30 layers) on top of umt5/weights, which overflows the 64 GB node and
# gets the run SIGKILLed (exit137). Preferred keeps pages local and spills over.
[[ $NODE -ge 0 ]] && BIND=(numactl --cpunodebind="$NODE" --preferred="$NODE")
{
  hostname; date; nvidia-smi -L; nvidia-smi topo -m; numactl -H; free -g; ulimit -l
  nvidia-smi -q -i "$GPU_UUID" | grep -A12 "GPU Link Info"
  echo "GPU_UUID=$GPU_UUID BUS=$BUS NUMA=$NODE"
} > "$OUT/env.txt" 2>&1
# Raw pinned-copy bandwidth at layer sizes, NUMA-bound and unbound (seconds).
"${BIND[@]}" "$PY/python" experiments/kv_offload/h2d_bench.py bound > "$OUT/h2d.jsonl" 2>&1
"$PY/python" experiments/kv_offload/h2d_bench.py unbound >> "$OUT/h2d.jsonl" 2>&1

# group -> "window sink blocks..."
declare -A MATRIX=(
  [B0]="12 6 20 40" [M1]="24 6 20 40" [M2]="36 6 20 40" [M3]="52 6 20 40"
  [M4]="76 6 40" [F20]="80 0 20" [F40]="160 0 40"
)
WARMUP=2

run_one() {  # name window sink blocks offload timing
  local name=$1 window=$2 sink=$3 blocks=$4 offload=$5 timing=$6 dir="$OUT/$1"
  mkdir -p "$dir"
  echo "=== $name ($(date +%T))"
  nvidia-smi dmon -s t -d 1 -i "$GPU_UUID" > "$dir/dmon.log" 2>&1 &
  local dmon=$!
  KV_SWEEP_WINDOW_T=$window KV_SWEEP_SINK_T=$sink KV_SWEEP_OFFLOAD=$offload \
  FLASHDREAMS_KV_OFFLOAD_TIMING=$timing \
  FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=$([[ $timing == 1 ]] && echo "$dir/kv_timing.jsonl") \
  timeout "${RUN_TIMEOUT:-5400}" "${BIND[@]}" "$PY/flashdreams-run-v2" lingbot.apps.cam2v.kv_sweep \
    --mode mp4 --output-path "$dir/out.mp4" --timeout unbound \
    --backpressure-mode block --presentation-mode on_demand \
    --stats-path "$dir/stats.json" \
    -- --example-data --total-blocks "$blocks" --warmup-blocks "$WARMUP" --seed 42 \
    > "$dir/run.log" 2>&1
  local rc=$?
  kill "$dmon" 2>/dev/null
  local status=ok
  [[ $rc -ne 0 ]] && status="exit$rc"
  grep -q "OutOfMemoryError\|CUDA out of memory" "$dir/run.log" && status=oom
  [[ $rc -eq 137 ]] && status=killed  # host OOM / SIGKILL
  printf '{"window": %d, "sink": %d, "blocks": %d, "offload": %s, "timing": %s, "warmup": %d, "status": "%s"}\n' \
    "$window" "$sink" "$blocks" "$([[ $offload == 1 ]] && echo true || echo false)" \
    "$([[ $timing == 1 ]] && echo true || echo false)" "$WARMUP" "$status" > "$dir/meta.json"
  echo "    -> $status"
}

for group in ${SWEEP_GROUPS:-F40}; do
  read -r window sink blocks_list <<< "${MATRIX[$group]}"
  for blocks in $blocks_list; do
    tag="$group-w$window-s$sink-b$blocks"
    run_one "$tag-offload-timed" "$window" "$sink" "$blocks" 1 1
    run_one "$tag-offload" "$window" "$sink" "$blocks" 1 0
    run_one "$tag-resident" "$window" "$sink" "$blocks" 0 0
  done
done

"$PY/python" experiments/kv_offload/analyze.py "$OUT"
echo "results: $OUT"
