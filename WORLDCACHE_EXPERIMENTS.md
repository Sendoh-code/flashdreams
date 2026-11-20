# WorldCache 实验与测试复现手册

目的：把整个 repo 搬到另一台有 GPU 的节点后，照着本文件就能重跑之前的单测和实验
（baseline / ours 对比、bank 验证、H2D sweep），并拿到可以和历史结果对照的数字。
目标与状态见 `WORLDCACHE.md`，代码位置见 `WORLDCACHE_IMPL.md`。

---

## 0. 新节点准备（一次性）

```bash
cd <repo>
uv sync --package flashdreams-lingbot --inexact --extra dev   # 生成 .venv（脚本固定用 .venv/bin/*）
git clone git@github.com:cvlab-kaist/WorldKV.git WorldKV && git -C WorldKV checkout 046f6d1
# ↑ 已在 .gitignore；compare.slurm 用它的相机轨迹 WorldKV/traj/R_L_R_L_R_L90_360/
```

运行时依赖（脚本会用到，缺了会失败）：

| 依赖 | 用途 | 检查 |
| --- | --- | --- |
| `ffmpeg` 在 PATH | mp4 输出、`bank_test` 的 PSNR | `ffmpeg -version`（Delta 上脚本自动 `module load ffmpeg/7.1`） |
| `/usr/bin/time` | `compare.slurm` 记录 host 峰值 RSS | `/usr/bin/time -v true` |
| `numactl`（可选） | `sweep.slurm` 和 `H2D_BENCH=1` 绑定 GPU 所在的 NUMA 节点 | `numactl -H` |
| HF 权重约 29 GB | `robbyant/lingbot-world-v2-1.3b-causal-fast`、`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`、`lightx2v/Autoencoders`，首次运行自动下载 | 需要时 `export HF_TOKEN=...` |

已验证环境：torch 2.12.1+cu130、Python 3.12.16；A40 48 GB（sm86）和 A100-SXM4 40 GB。
sm86 / A100 没有 FP8，所以 `kv_sweep` app 固定用 bf16 `torch` linear 和 `wan` attention。

### 0.1 路径变量（脚本默认写死为 Delta 路径，换节点必须覆盖）

| 变量 | Delta 默认值 | 含义 |
| --- | --- | --- |
| `SWEEP_ROOT` | `/work/nvme/bicu/ytian6/kv_offload` | 所有实验输出的根目录 |
| `OUT` | `$SWEEP_ROOT/<前缀>-<jobid>` | 单次实验的输出目录；拆成两个 job 时两边共用同一个 |
| `HF_HOME` | `/work/nvme/bicu/ytian6/hf_cache` | 权重缓存 |
| `TORCHINDUCTOR_CACHE_DIR` / `TRITON_CACHE_DIR` | `/work/nvme/bicu/ytian6/inductor_cache[/triton]` | 编译缓存。冷启动第一个 chunk 要编译约 230 s，缓存热了才能放进 1 h 的 job |

```bash
export SWEEP_ROOT=/path/to/kv_offload HF_HOME=/path/to/hf_cache \
       TORCHINDUCTOR_CACHE_DIR=/path/to/inductor_cache
```

### 0.2 有 Slurm 和没有 Slurm 两种跑法

- **其它 Slurm 集群**：改脚本头部的 `#SBATCH --account/--partition/--constraint`，
  以及 `--output`（日志目录 `experiments/kv_offload/logs/` 已存在）。必须在 repo 根目录下 `sbatch`。
- **没有 Slurm 的 GPU 机器**：`#SBATCH` 行只是注释，脚本可以直接 `bash` 运行
  （`SLURM_SUBMIT_DIR` 为空时用 `$PWD`；`SLURM_JOB_ID` 为空时目录名用 `local`；`module` 失败会被忽略）。
  这时最好显式设 `OUT`，避免多次运行写进同一个 `*-local` 目录：
  ```bash
  OUT=$SWEEP_ROOT/compare-$(date +%m%d-%H%M) bash experiments/kv_offload/compare.slurm
  ```
- 多卡机器上用 `CUDA_VISIBLE_DEVICES=<id>` 固定一张卡。H2D 相关数字要串行跑，不要和别的 GPU 作业共享这个节点的 host 带宽。

---

## 1. 单元测试（先跑，CPU 几分钟，GPU 一分钟左右）

全部在 `flashdreams/tests/test_kvcache.py`，另有一个在 LingBot 集成里。

| 测试 | 标记 | 验证什么 | 坏了说明什么 |
| --- | --- | --- | --- |
| `test_block_kvcache_*`（多个） | `ci_cpu` / `ci_gpu` | `BlockKVCache` 的 sink + 滚动窗口语义与参考实现一致，含 compile / CUDA graph | 基础 KV cache 回归 |
| `test_layerwise_offload_matches_resident_caches[sink,window]` | `ci_cpu` | offload（两个 slot + sink buffer + 以 bank 为窗口）在每层、每个去噪步看到的 K/V 和常驻 cache 完全一样；参数覆盖 sink、window 与 chunk 不对齐的情况；host 上不再有整层 home | offload 正确性、窗口索引、滚窗 |
| `test_kv_bank_fills_retrieve_region[granularity-capacity]` | `ci_cpu` | attention 看到的是 `[sink | 策略选中的 bank 条目 | 窗口]`，FIFO 驱逐与独立模型一致；chunk / frame 粒度，预算无限 / 很小（会被抬到窗口大小） | bank 检索、驱逐、retrieve 区 |
| `test_kv_bank_fills_retrieve_region_cuda[...]` | `ci_gpu` | 同上，但走 CUDA side stream + pinned 内存（真实的异步路径） | 流同步或竞态 |
| `test_camera_pose_retrieval_picks_nearest_pose` | `ci_cpu` | `CameraPoseRetrieval` 按位姿距离取 top-k | 检索打分 |
| `test_layerwise_offload_timing_reports_every_layer` | `ci_gpu` | `FLASHDREAMS_KV_OFFLOAD_TIMING` 每层都有 fetch/compute 计时，`fetch_mb` 正确，JSONL 落盘 | 计时诊断 |
| `integrations_v2/lingbot/tests/test_rope_kvcache.py::test_pre_and_post_rope_cache_match_without_roll` | `ci_gpu` | 不滚窗时，pre-RoPE 与 post-RoPE 两种 K 缓存的 block 输出一致（`scaled_fp8` 参数需要 sm89+，A40/A100 上会 skip） | RoPE 位置语义 |

```bash
.venv/bin/pytest -q -p no:cacheprovider -m ci_cpu flashdreams/tests/test_kvcache.py
.venv/bin/pytest -q -p no:cacheprovider -m ci_gpu flashdreams/tests/test_kvcache.py \
  integrations_v2/lingbot/tests/test_rope_kvcache.py -rs
```

历史结果：`ci_cpu` 19 passed（2026-11-18，bank 成为唯一 host 存储之后）；
`ci_gpu` 6 passed / 1 skipped（job 23910329，2026-11-13，那次改动之前）。
`bank_test.slurm` 第 1 步跑的就是这些测试的子集。

---

## 2. 主实验：baseline vs ours（`compare.slurm`）

**问题**：长窗口 / host offload / 相机位姿 KV bank 相对原始短窗口，在速度、显存和
「回到看过的视角时的一致性」上各有什么代价和收益。

- 模型：LingBot World v2 1.3B causal-fast，bf16，post-RoPE，`--seed 42`。
- 轨迹：WorldKV `R_L_R_L_R_L90_360`（原地左右来回 yaw 扫，约 120 帧后回到同一视角，超出 baseline 窗口）。
  360 个位姿对应 88 个 latent 帧，即 `BLOCKS=22` 个 chunk（每个 4 latent 帧 / 16 像素帧），前 2 个 chunk 是 warmup。
- 帧数单位：`window/sink/retrieve` 都按 patchify 前的 latent 帧计。

| run | window | sink | retrieve | offload | 含义 |
| --- | --- | --- | --- | --- | --- |
| `baseline` | 12 | 6 | 0 | 否 | 当前 1.3B 默认的 KV 策略 |
| `resident-w40` | 40 | 6 | 0 | 否 | 不用 offload、直接开长窗口的代价（显存） |
| `offload-w40` | 40 | 6 | 0 | 是 | 长窗口放到 host，逐层流式 H2D |
| `ours` | 40 | 6 | 8 | 是 | offload-w40 + 相机位姿 KV bank，每次 forward 检索 8 帧（`BANK_GB` 默认 24） |
| `ours-w8` | 8 | 6 | 8 | 是 | WorldKV 的布局：sink + 检索 + 最近 2 个 chunk |

```bash
# 一次全跑（冷编译缓存时可能超过 1 h）
sbatch experiments/kv_offload/compare.slurm
# 交互分区限 1 h：拆成两个 job，共用 OUT 和编译缓存
OUT=$SWEEP_ROOT/compare-$(date +%m%d-%H%M); export OUT
A=$(sbatch --parsable --export=ALL,RUNS="baseline resident-w40" experiments/kv_offload/compare.slurm)
sbatch --dependency=afterany:$A --export=ALL,RUNS="offload-w40 ours" experiments/kv_offload/compare.slurm
```

可选开关：`RUNS="..."`（选择要跑的 run）、`BLOCKS`、`BANK_GB`、`RUN_TIMEOUT`（每个 run 的超时，默认 1500 s）、
`FLASHDREAMS_KV_OFFLOAD_TIMING=1`（每层 fetch/stall/compute/writeback 写入 `<run>/kv_timing.jsonl`，
同时在 `run.log` 打印每个 chunk 的 `begin_chunk` / `roll` 的 host 耗时）、
`H2D_BENCH=1`（结束后测原始 pinned 拷贝带宽，写入 `$OUT/h2d.jsonl`）。

**输出**：`$OUT/<run>/{out.mp4, stats.json, run.log, ok}`，`$OUT/summary.txt`（每个 run 的状态），
`$OUT/metrics.{json,md}`（由 `compare_metrics.py` 生成）。

**指标**（`compare_metrics.py`，也可以单独运行：
`.venv/bin/python experiments/kv_offload/compare_metrics.py $OUT --poses <poses.npy>`；用 `--self-check` 自检）：

| 列 | 含义 |
| --- | --- |
| `chunk_s_median` / `chunk_s_p90` / `fps` | 去掉 warmup 后每个 chunk 的耗时，以及换算出的 FPS |
| `gpu_peak_gib` / `host_rss_gib` | 显存峰值 / host 峰值 RSS（来自 `/usr/bin/time -v`） |
| `revisit_psnr` / `revisit_ssim` | **核心质量指标**：第 j 帧与之前同位姿（< 2°）、至少相隔 64 帧（超出 baseline 窗口）的第 i 帧之间的 PSNR / SSIM；跳过 sink 内的源帧 |
| `sharpness` / `sharpness_late_over_early` | Laplacian 清晰度，以及最后四分之一 / 最前四分之一的比值（越低说明越来越糊） |
| `psnr_vs_baseline` | 与 baseline 视频的差异程度（只说明偏离，不代表质量） |

**历史结果**（A100-SXM4-40GB，commit `711ff4f5`，208 个 revisit pair）：

| 来源 | run | chunk s（中位 / p90） | fps | GPU GiB | host RSS GiB | revisit PSNR / SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| `compare-1116-1310` | baseline | 冷编译超时（exit124），**没有有效结果** | | | | |
| `compare-1116-1310` | resident-w40 | 3.15 / 103（p90 包含编译） | 5.08 | 33.5 | 17.9 | 13.52 / 0.566 |
| `compare-1116-1310` | offload-w40 | 5.45 / 7.88 | 2.94 | 22.3 | 21.0 | 13.52 / 0.552 |
| `compare-1116-1310` | ours | 4.77 / 7.70 | 3.36 | 22.5 | 50.9 | 13.46 / 0.556 |
| `compare-24082157`（timing） | offload-w40 | 5.10 / 5.34 | 3.14 | 22.3 | 20.9 | 13.67 / 0.565 |
| `diag-1117-1048`（timing + H2D） | ours | 4.81 / 5.84 | 3.33 | 22.5 | 50.9 | 13.49 / 0.559 |
| `compare-1118-1550`（`99b072de`，bank 作为唯一 host 存储，timing） | offload-w40 | 3.29 / 5.22 | 4.86 | 22.3 | 22.5 | 13.15 / 0.557 |
| `compare-1118-1550`（同上） | ours | 3.74 / 6.52 | 4.28 | 22.5 | 39.5 | 13.41 / 0.558 |

结果目录在 Delta 的 `/work/nvme/bicu/ytian6/kv_offload/` 下。

结论与待办：
- 改动前 offload-w40 的 5.1 s/chunk 里约 1.84 s 是 host 上的滚窗（`roll` 日志）。「bank 作为唯一 host 存储」
  （`WORLDCACHE_IMPL.md` §6，job 24131186）之后 `roll` < 2 ms：offload-w40 5.10 → 3.29 s/chunk，
  ours 4.81 → 3.74 s/chunk，ours host RSS 50.9 → 39.5 GiB。第 0 个 chunk 的 `begin_chunk` 包含一次性的
  pinned bank 分配（ours 17.5 s，offload-w40 8.6 s），属于 warmup。
- 质量：同一配置重复跑的视频之间 PSNR 只有约 22–25 dB（编译 / autotune 带来的非确定性），revisit PSNR
  在 13.15–13.67 之间波动，改动前后的差异在噪声范围内。
- 稳态每层 fetch（offload-w40 17.4 ms，ours 20.5 ms）≈ compute（18.1 / 20.7 ms），H2D 已跑到 pinned 峰值
  约 24.5 GB/s，但被 compute 掩盖：每次 forward 的 stall 只有 39 / 54 ms（约 7%），offload-w40 只比 resident-w40
  慢 4%（3.29 vs 3.15 s/chunk）。所以现在**不是** H2D 瓶颈，去掉 `_prefetch` 的 `ponytail:` 暂不做；
  拷贝量随可见长度线性增长，窗口更长或 compute 更快时才会成为瓶颈（第 4 节的交叉点）。
- ours 比 offload-w40 慢 0.45 s/chunk，完全来自多出的 8 个检索帧（两者 window 都是 40）：attention 覆盖
  54 vs 46 帧，每次 forward 拷贝 15.0 vs 12.8 GB（比值 = 54/46），compute 多 78 ms × 每 chunk 5 次 forward
  ≈ diffuse 多出的 0.40 s；检索本身的 host 开销可忽略。
- 统计稳态要取窗口填满后（chunk ≥ 12）的 forward，混入填充阶段会把每层时间算低。
- baseline 还没有有效结果：在热缓存下重跑，或者调大 `RUN_TIMEOUT`。

---

## 3. KV bank 与 pre/post-RoPE 端到端（`bank_test.slurm`）

**问题**：bank 在 chunk / frame 粒度、pre-RoPE / post-RoPE 下能否端到端跑通，以及与同 RoPE 模式的常驻版本差多少。
配置：window 12、sink 4、16 个 block、`--example-data`（不用位姿轨迹）、bank 4 GB、检索 8 帧。原来设计在 A40 上跑。

| run | RoPE | retrieve | 粒度 | offload |
| --- | --- | --- | --- | --- |
| `resident-post` / `resident-pre` | post / pre | 0 | – | 否 |
| `bank-chunk-post` / `bank-chunk-pre` | post / pre | 8 | chunk | 是 |
| `bank-frame-post` | post | 8 | frame | 是 |

```bash
OUT=$SWEEP_ROOT/bank-$(date +%m%d-%H%M); export OUT
A=$(sbatch --parsable --export=ALL,RUNS="resident-post resident-pre" experiments/kv_offload/bank_test.slurm)
sbatch --dependency=afterany:$A --export=ALL,TESTS=0,RUNS="bank-chunk-post bank-chunk-pre bank-frame-post" \
  experiments/kv_offload/bank_test.slurm
```

`TESTS=0` 跳过第 1 节的单测。`summary.txt` 里每行是 `status` 以及与同 RoPE 模式 resident 的 PSNR
（只有那个 resident 留下了 `ok` 标记时才计算）。

**历史结果**（A40，`bank-1113-1640`，16 个 block，去掉 2 个 warmup 后的中位 chunk 耗时）：

| run | 来源 job | chunk s | GPU 峰值 GiB | PSNR vs 同 RoPE resident |
| --- | --- | --- | --- | --- |
| `resident-post` | 24046877（11-16） | 2.94 | 25.7 | – |
| `resident-pre` | 23910329（11-13） | 3.02 | 25.7 | – |
| `bank-chunk-post` | 24046879（11-18） | 3.72 | 21.9 | 19.95 dB |
| `bank-chunk-pre` | 24046879 | 3.79 | 21.9 | 20.23 dB |
| `bank-frame-post` | 24046879 | 3.72 | 21.9 | 19.93 dB |

- 五个 run 全部跑通，pre-RoPE（`apply_rope_freqs_copy`）与 frame 粒度端到端可用；同一节、两个 job 的
  单测 `ci_cpu` 6 passed、`ci_gpu` 6 passed / 1 skipped（23910329）。
- 24046879 在 11-16 以 `711ff4f5` 提交，11-18 18:15 才开始，跑的是当时的工作树（「bank 作为唯一 host 存储」之后）；
  两个 resident 来自更早的 commit。resident 不 offload，与这次改动无关，可以直接对比。
- 23910329 的 `summary.txt` 把 `resident-pre` 记成 oom，但目录里有 `ok` 和完整输出；这是 `711ff4f5` 修掉的误判。
- bank 比 resident 慢约 0.8 s/chunk（+26%），省 3.8 GiB 显存：多出的是 offload 的 stall 和 8 个检索帧的 attention。
- PSNR ≈ 20 dB 只说明偏离程度（检索改变了 attention 的内容），不代表质量；同配置重复跑也只有 22–25 dB。
  pre / post 与 chunk / frame 之间没有可区分的差异。
- 早先 job 23900078 的 `bank-chunk-post`：稳态 4.0 s/chunk，显存 21.9 GiB。

---

## 4. H2D 交叉点 sweep（`sweep.slurm` + `analyze.py` + `h2d_bench.py`）

**问题**：可见 KV 长度 V 到多大时，每层的 H2D fetch 时间会超过每层的计算时间（交叉点 V*）。
设计在 A40 上整节点独占、串行运行（节点上的其他作业会占用 host 带宽）。

| 组 | window | sink | blocks |
| --- | --- | --- | --- |
| B0 / M1 / M2 / M3 | 12 / 24 / 36 / 52 | 6 | 20、40 |
| M4 | 76 | 6 | 40 |
| F20 / F40 | 80 / 160 | 0 | 20 / 40 |

每个组合跑三种：`offload-timed` / `offload` / `resident`。

```bash
sbatch --export=ALL,SWEEP_GROUPS="B0 M1 M2 M3" experiments/kv_offload/sweep.slurm
.venv/bin/python experiments/kv_offload/analyze.py $SWEEP_ROOT/<jobid>   # 生成 forwards.csv / summary.csv / crossover.png
.venv/bin/python experiments/kv_offload/h2d_bench.py bound|unbound       # 只测原始 pinned 拷贝带宽
```

历史结果（job 23683973，A40）：H2D 24.5 GB/s，D2H 26.3 GB/s。F20/F40 的 resident 全部 OOM；
F40 offload 被 host OOM kill（约 44 GB pinned 内存）。只有 F20 offload 在 V=40 有一个点：
fetch 9.06 ms/层，compute 19.2 ms/层。还缺能放进显存的窗口（B0–M3）作为 baseline。
新节点的 host 内存决定最大可用窗口：F40 需要 64 GB 以上。

---

## 5. 在新节点上的推荐顺序

1. 第 0 节环境准备 → 第 1 节 `ci_cpu` 和 `ci_gpu` 全部通过。
2. 先单独跑一个 run 预热编译缓存：`RUNS=resident-w40 bash experiments/kv_offload/compare.slurm`
   （这一步也会下载权重）。
3. `compare.slurm`：先 `RUNS="offload-w40 ours"` 并开 `FLASHDREAMS_KV_OFFLOAD_TIMING=1`
   （验证滚窗开销是否消失），再跑 `RUNS="baseline resident-w40"` 补齐对照。
4. `bank_test.slurm` 跑 pre/post-RoPE 与粒度矩阵。
5. 需要带宽结论时再跑第 4 节的 sweep。

新的结果写回 `WORLDCACHE.md` 的 Status，并在本文件的历史结果表里追加一行（注明 GPU 型号和 commit）。
