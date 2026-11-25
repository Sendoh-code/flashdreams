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
| `ffmpeg` 在 PATH | mp4 输出、`bank_test` 的 PSNR | `ffmpeg -version`（Delta 上脚本自动 `module load ffmpeg/7.1`；没有 sudo 时用 lockfile 里的 `imageio-ffmpeg`，见下方） |
| `imageio`、`av` | `compare_metrics.py` 读 mp4 | `uv sync` 不会装；缺了 `metrics.log` 报 `No module named 'imageio'` |
| `/usr/bin/time` | `compare.slurm` 记录 host 峰值 RSS | `/usr/bin/time -v true` |
| `numactl`（可选） | `sweep.slurm` 和 `H2D_BENCH=1` 绑定 GPU 所在的 NUMA 节点 | `numactl -H` |
| HF 权重约 29 GB | `robbyant/lingbot-world-v2-1.3b-causal-fast`、`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`、`lightx2v/Autoencoders`，首次运行自动下载 | 需要时 `export HF_TOKEN=...` |

已验证环境：torch 2.12.1+cu130、Python 3.12.16；A40 48 GB（sm86）、A100-SXM4 40 GB，
以及 RTX PRO 6000 Blackwell Server Edition 96 GB（sm120，2026-11-20 起的独占实验机，无 Slurm）。
sm86 / A100 没有 FP8，所以 `kv_sweep` app 固定用 bf16 `torch` linear 和 `wan` attention
（RTX PRO 6000 上也保持 bf16，以便和历史数据对比）。

RTX PRO 6000 上的补充步骤（无 sudo、无 Slurm）：

```bash
uv pip install --python .venv/bin/python imageio-ffmpeg==0.6.0 imageio==2.37.3 av==16.1.0  # 版本取自 uv.lock
mkdir -p .venv/ffbin && ln -sf "$(.venv/bin/python -c 'import imageio_ffmpeg;print(imageio_ffmpeg.get_ffmpeg_exe())')" .venv/ffbin/ffmpeg
export PATH=$PWD/.venv/ffbin:$PATH HF_HUB_DOWNLOAD_TIMEOUT=60
```

- 首次下载 `Wan-AI/Wan2.1-T2V-1.3B-Diffusers` 的 umT5 分片时，Xet 传输在文件已完整写入后卡住
  （连接 `CLOSE-WAIT`，进程 0% CPU，不报错）。处理：杀掉进程，对 `blobs/*.incomplete` 做 `sha256sum`，
  与文件名（LFS oid）一致就去掉 `.incomplete` 后缀，再跑一次 `snapshot_download` 补 snapshot 链接。
- Triton autotune 会打印 `out of resource ... Hardware limit:101376`（sm120 每 SM 共享内存约 99 KB），
  只是跳过该配置，不影响运行。

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
| `test_kv_bank_fills_retrieve_region[granularity-capacity-resident]` | `ci_cpu` | attention 看到的是 `[sink | 策略选中的 bank 条目 | 窗口]`，FIFO 驱逐与独立模型一致；chunk / frame 粒度，预算无限 / 很小（会被抬到窗口大小），offload 与常驻 bank 两种模式，含空检索时恢复 sink | bank 检索、驱逐、retrieve 区 |
| `test_kv_bank_fills_retrieve_region_cuda[...]` | `ci_gpu` | 同上，但走 CUDA side stream + pinned 内存（真实的异步路径） | 流同步或竞态 |
| `test_camera_pose_retrieval_picks_nearest_pose` | `ci_cpu` | `CameraPoseRetrieval` 按位姿距离取 top-k | 检索打分 |
| `test_layerwise_offload_timing_reports_every_layer` | `ci_gpu` | `FLASHDREAMS_KV_OFFLOAD_TIMING` 每层都有 fetch/compute 计时，`fetch_mb` 正确，JSONL 落盘 | 计时诊断 |
| `integrations_v2/lingbot/tests/test_rope_kvcache.py::test_pre_and_post_rope_cache_match_without_roll` | `ci_gpu` | 不滚窗时，pre-RoPE 与 post-RoPE 两种 K 缓存的 block 输出一致（`scaled_fp8` 参数需要 sm89+，A40/A100 上会 skip） | RoPE 位置语义 |

```bash
.venv/bin/pytest -q -p no:cacheprovider -m ci_cpu flashdreams/tests/test_kvcache.py
.venv/bin/pytest -q -p no:cacheprovider -m ci_gpu flashdreams/tests/test_kvcache.py \
  integrations_v2/lingbot/tests/test_rope_kvcache.py -rs
```

历史结果：`test_kvcache.py` `ci_cpu` 23 passed、`ci_gpu` 15 passed（2026-11-25，RTX PRO 6000，含常驻 bank）；
`ci_cpu` 19 passed（2026-11-18，bank 成为唯一 host 存储之后）；
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

**注意 UI 面板**：flashdreams 录出的 `out.mp4` 左上角烧进了 cam2v 的 "Camera controls" 面板
（x < 约 0.45 W，文字会变），码率约 4.5 Mbps（官方 LingBot 视频没有面板，约 15 Mbps）。
所有视频指标默认只用 x ≥ `--x0` × W 的部分（`--x0` 默认 0.5，传 `--x0 0` 用整帧）。
2026-11-24 之前的数字都是整帧算的：revisit PSNR 被面板抬高约 1.5 dB，`sharpness_late_over_early`
被面板文字拉到接近 1，掩盖了画面变糊。

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

以上 A100 结果目录在 Delta 的 `/work/nvme/bicu/ytian6/kv_offload/` 下。

**RTX PRO 6000 Blackwell 结果**（96 GB，sm120，commit `4049e1d4`，bf16，timing 开启，208 个 revisit pair；
目录 `/mnt/datastore/yiti6755/kv_offload/compare-1120-rtx6000/`）。p90 含前两个 chunk 之外的编译抖动，
baseline / resident-w40 的 p90 偏大是因为它们先跑、编译缓存还是冷的：

| GPU | run | chunk s（中位 / p90） | fps | GPU GiB | host RSS GiB | revisit PSNR / SSIM |
| --- | --- | --- | --- | --- | --- | --- |
| RTX PRO 6000 | baseline | 1.16 / 59.7 | 13.81 | 26.4 | 18.2 | 15.70 / 0.552 |
| RTX PRO 6000 | resident-w40 | 2.04 / 28.0 | 7.83 | 33.6 | 18.2 | 15.66 / 0.559 |
| RTX PRO 6000 | offload-w40 | 2.08 / 5.63 | 7.70 | 22.4 | 23.1 | 15.58 / 0.579 |
| RTX PRO 6000 | ours | 2.32 / 4.49 | 6.89 | 22.5 | 39.9 | 15.53 / 0.570 |

上表 revisit 是整帧算的；遮掉面板（x ≥ 0.5 W）后 baseline / resident-w40 / offload-w40 / ours 是
14.49 / 14.29 / 14.51 / 14.43 dB（SSIM 0.387 / 0.392 / 0.426 / 0.415），仍在噪声范围内。
整帧版本保存在同目录的 `metrics_fullframe.{json,md}`。

RTX PRO 6000 稳态逐层计时（窗口填满后的 forward，30 层）：offload-w40 fetch 7.40 / compute 11.85 ms
（比值 0.62），ours 8.69 / 13.66 ms（0.64）；每次 forward stall 7.4 / 8.7 ms（约 2.1%），H2D 57.6 GB/s，
每次 forward 拷贝 12.8 / 15.0 GB。offload-w40 只比 resident-w40 慢 1.7%。**compute 仍掩盖 fetch，
余量比 A100 更大**（带宽是 A100 的 2.4 倍，bf16 compute 只快 1.5 倍）。sm120 有 FP8，换 FP8 后 compute
变短，需要重测。

结论与待办：
- 改动前 offload-w40 的 5.1 s/chunk 里约 1.84 s 是 host 上的滚窗（`roll` 日志）。「bank 作为唯一 host 存储」
  （`WORLDCACHE_IMPL.md` §6，job 24131186）之后 `roll` < 2 ms：offload-w40 5.10 → 3.29 s/chunk，
  ours 4.81 → 3.74 s/chunk，ours host RSS 50.9 → 39.5 GiB。第 0 个 chunk 的 `begin_chunk` 包含一次性的
  pinned bank 分配（ours 17.5 s，offload-w40 8.6 s），属于 warmup。
- 质量：同一配置重复跑的视频之间 PSNR 只有约 22–25 dB（编译 / autotune 带来的非确定性，§2.3 已确认：
  eager 下 latent 逐位相同），revisit PSNR 在 13.15–13.67 之间波动，改动前后的差异在噪声范围内。
- 稳态每层 fetch（offload-w40 17.4 ms，ours 20.5 ms）≈ compute（18.1 / 20.7 ms），H2D 已跑到 pinned 峰值
  约 24.5 GB/s，但被 compute 掩盖：每次 forward 的 stall 只有 39 / 54 ms（约 7%），offload-w40 只比 resident-w40
  慢 4%（3.29 vs 3.15 s/chunk）。所以现在**不是** H2D 瓶颈，去掉 `_prefetch` 的 `ponytail:` 暂不做；
  拷贝量随可见长度线性增长，窗口更长或 compute 更快时才会成为瓶颈（第 4 节的交叉点）。
- ours 比 offload-w40 慢 0.45 s/chunk，完全来自多出的 8 个检索帧（两者 window 都是 40）：attention 覆盖
  54 vs 46 帧，每次 forward 拷贝 15.0 vs 12.8 GB（比值 = 54/46），compute 多 78 ms × 每 chunk 5 次 forward
  ≈ diffuse 多出的 0.40 s；检索本身的 host 开销可忽略。
- 统计稳态要取窗口填满后（chunk ≥ 12）的 forward，混入填充阶段会把每层时间算低。
- A100 上 baseline 还没有有效结果：在热缓存下重跑，或者调大 `RUN_TIMEOUT`。
  RTX PRO 6000 上 baseline 已有结果（上表，`RUN_TIMEOUT=3600`）。

### 2.1 窗口长度实验与官方对照（RTX PRO 6000，2026-11-20 至 11-24）

**窗口长度实验**：`/mnt/datastore/yiti6755/kv_offload/win-study/`，驱动脚本 `run.sh`，汇总脚本
`aggregate.py`（`python3 aggregate.py . > results.md`）。seed 42/43/44，每个 seed 跑一遍 `compare.slurm`：
`baseline`（W12）、`res-w16…w84`、`ret-w8-r8/r16/r24`。遮掉面板后的结论：

- resident 窗口 12–84：revisit PSNR 13.7–14.2 dB，std 0.5–1.0，没有趋势。
- 总长 T 相同时，检索 `ret-w8-r<T-8>` 对比滑窗 `res-w<T>`（同 seed 配对）：far pair（间隔 ≥ 200 帧）
  分别 +0.45 / +0.27 / +0.34 dB（T = 16 / 24 / 32），9 对里 7 对检索更好；显存 22 vs 27–31 GiB，
  速度只慢 2–3%。信号弱，需要更多 seed 和干净录制（关掉 overlay、提高码率）才能下结论。

**WorldKV 官方代码复现**（14B `lingbot-world-base-cam`，不是 1.3B）：`WorldKV/repro/run.sh`，结果在
`WorldKV/repro/out/results.md`（gitignored）。窗口 18 时，检索（`worldkv` / `wr9`）比同长滑窗
`sink3-recent15` 高约 0.9 dB，3 个 seed 都赢；去掉 sink 的 `fifo18` 差 3 dB 以上。

**相机跟随对照（官方 LingBot-World-V2 1.3B vs flashdreams）**：`/mnt/datastore/yiti6755/lingbot-upstream/`。
`SEEDS="42 43 44" bash run_ab.sh` 用官方 `generate.py`（upstream `1895d30` + SDPA fallback）在相同输入
（example 00 图片/prompt/内参 + `R_L_R_L_R_L90_360` 位姿，`local_attn_size 18`、`sink_size 6`）上生成，
已有视频不重跑，然后跑 `flow_yaw.py`。`flow_yaw.py` 用 Farneback 光流估计每帧 yaw，`homog_yaw.py`
用 ORB 匹配加单应矩阵交叉检查，两者都只看 x ≥ 0.5 W。

| | 相关系数 | 增益（实际 / 命令 yaw） |
| --- | --- | --- |
| flashdreams baseline | 0.83 | 0.99（0.98 / 0.95 / 1.03） |
| 官方 1.3B | 0.85 | 1.00（0.97 / 1.02 / 1.02） |

win-study 所有配置的增益都在 0.98–1.05（`win-study/flow_yaw_roi50.txt`）。之前测出的 0.78
（"flashdreams 少转 20%"）是因为 ROI 从 0.3 W 开始，压到了面板。排查过程中确认过两边的
Plücker 输入（位姿、内参、射线，只差 bf16 舍入）和 4 步采样器（timestep 999/937/833/624，
sigma 差 < 6e-4）一致。

**画面变糊**（锐度最后 1/4 / 最前 1/4，x ≥ 0.5 W）：flashdreams baseline 约 0.67，官方约 0.74
（0.69 / 0.86 / 0.68）。两边都有，不是 flashdreams 独有；这个比值也受相机朝向影响，只能作粗略参考。

### 2.2 LingBot v2 14B（RTX PRO 6000，2026-11-24）

**准备**：checkpoint `robbyant/lingbot-world-v2-14b-causal-fast` 的 `transformers/*` 约 74 GB（fp32 分片，
加载成 bf16）。T5 / VAE 和 1.3B 共用。下载时同样会碰到 Xet 卡死（分片已完整，进程不退出），按 §0
的方法处理：杀进程，校验 sha256，去掉 `.incomplete` 后缀，再跑一次 `snapshot_download`。

**preset 修正**：`lingbot-world-v2-14b-causal-fast` 原来照搬了 v1 Fast（`len_t` 3、timestep
999/978/947/825、window 63 / sink 0），现在按官方 v2 改为 `len_t` 4、timestep 999/967/908/768
（shift 10）、window 12 / sink 6（官方 `local_attn_size 18`、`sink_size 6`，18 帧里包含 sink 和当前 chunk）。

**运行**：

```bash
MODEL=14b BANK_GB=112 FLASHDREAMS_KV_OFFLOAD_TIMING=1 RUN_TIMEOUT=3600 \
  RUNS="baseline resident-w40 offload-w40 ours" OUT=$SWEEP_ROOT/compare-14b-$(date +%m%d) \
  bash experiments/kv_offload/compare.slurm
```

每个 run 要重新加载约 74 GB 权重（约 4 分钟），第一次还要编译。`BANK_GB=112` 足够放下 88 个
latent 帧 × 每帧约 1.24 GB 的全部历史；`ours` 的 host RSS 约 183 GiB，需要 200 GB 以上内存。

**结果**（seed 42，22 个 chunk，`compare-14b-1124/`）：

| run | chunk s（中位） | GPU GiB | host RSS GiB | revisit PSNR / SSIM（遮罩） |
| --- | --- | --- | --- | --- |
| baseline（W12） | 5.96 | 69.3 | 70.2 | 9.25 / 0.266 |
| resident-w40 | OOM（95 GiB 用满） | | | |
| offload-w40 | 9.58 | 51.2 | 92.9 | 9.60 / 0.272 |
| ours（+8 检索帧） | 10.52 | 51.6 | 182.9 | 9.06 / 0.233 |

稳态逐层计时（最后 40 次 forward，40 层）：offload-w40 fetch 24.6 / compute 43.2 ms（比值 0.57），
ours 28.9 / 50.6 ms（0.57）；stall 约占每次 forward 的 1.4%；每次 forward 拷贝 56.8 / 66.7 GB，
H2D 57.7 GB/s。**14B 上 compute 仍然掩盖 fetch**，余量和 1.3B 相当。offload 让 W40 在 14B 上能跑，
显存比 W12 baseline 还少 18 GiB；offload-w40 比 baseline 慢 61%，主要是 W40 本身的 attention 计算，
不是 H2D。ours 比 offload-w40 慢 10%，来自多出的 8 个检索帧（attention 54 vs 46 帧）。

质量：14B 画面前景有一匹会动的大马和骑手，revisit PSNR 只有约 9 dB，不能和 1.3B 的数字比；
单 seed 下三者差异属于噪声。相机跟随正常（光流 yaw 增益 1.04，相关系数 0.87）。

### 2.3 WorldKV 设置、常驻 bank 与可复现性（RTX PRO 6000，2026-11-25）

**问题**：按 WorldKV 自己的设置（v1 LingBot-World-Fast 14B，`len_t` 3，共 18 个 attended 帧）对比
滑窗与检索，并看 offload 有没有改变结果。

| run | 布局 | offload | 含义 |
| --- | --- | --- | --- |
| `wkv-baseline` | sink 6 + window 12 | 否 | v2 README 的策略 |
| `wkv-resident` | sink 3 + 检索 9 + window 6，gap 6 | 否 | WorldKV `--use_retrieval --retrieval_frames 9`；cache 常驻显存，只有 bank 在 host（`kv_bank` 不带 `kv_offload`） |
| `wkv-ours` | 同上 | 是 | 同上 + 逐层 offload |

gap 6 = `CameraPoseRetrieval.min_gap_t`：WorldKV 把最近 6 帧排除在检索之外，虽然它只 attend 其中 3 帧。

```bash
MODEL=v1fast BLOCKS=30 RUN_TIMEOUT=3600 FLASHDREAMS_KV_OFFLOAD_TIMING=1 \
  METRIC_ARGS="--len-t 3 --min-src 12 --min-gap 80 --baseline wkv-baseline" \
  RUNS="wkv-baseline wkv-resident wkv-ours" SEED=42 OUT=$SWEEP_ROOT/wkv-$(date +%m%d)/seed42 \
  bash experiments/kv_offload/compare.slurm
```

驱动脚本：`/mnt/datastore/yiti6755/kv_offload/wkv-1125/run.sh`（3 个 seed，bank 24 GB）和
`wkv-1125-bank120/run.sh`（`BANK_GB=120`，30 个 chunk 全部可检索，外加一次 seed 42 resident 重复；
这次重复没设 `BANK_GB`，实际是 24 GB）。161 个 revisit pair，遮掉面板。

| | s/chunk | GPU GiB | revisit PSNR（42 / 43 / 44，均值） |
| --- | --- | --- | --- |
| wkv-baseline | 4.39–4.42 | 66.8 | 10.36 / 9.10 / 9.96，**9.81** |
| wkv-resident，bank 24 GB | 4.48–4.52 | 66.8 | 9.24 / 10.17 / 9.29，9.57 |
| wkv-ours，bank 24 GB | 4.43–4.45 | **47.5** | 7.93 / 9.68 / 8.49，8.70 |
| wkv-resident，bank 120 GB | 4.49 | 66.8 | 8.82 / 9.36 / 8.83，9.00 |
| wkv-ours，bank 120 GB | 4.44–4.45 | 47.5 | 7.53 / 8.61 / 7.43，7.86 |
| wkv-resident 重复（seed 42） | 4.50 | 66.8 | 8.64（同配置第一次 9.24） |

bank 120 GB 时 host RSS 约 180 GB，pinned 分配让单个 run 从约 4 分钟变成 8 分钟（resident）/ 24 分钟（ours），
每 chunk 速度不变。检索没有赢过 baseline（WorldKV 自己的代码在 14B 上约 +0.9 dB）。

**ours 比 resident 低 1 dB 不是 bug。** 6 对里 6 对 ours 更低，看起来像 offload 路径有错，于是做了逐位 parity：

1. `test_kvcache.py`：`ci_cpu` 23 passed，`ci_gpu` 15 passed（含 resident 参数化的 bank 测试）。
2. 同配置同 seed 跑两次，比较两段视频（x ≥ 0.5 W 的逐帧 PSNR），1.3B baseline，3 个 chunk：

   | 设置 | 两次运行之间 | decode 前的 latent |
   | --- | --- | --- |
   | 默认编译（`max-autotune`） | 第 1 帧 33 dB，第 3 个 chunk 降到 25 dB，持续漂移 | – |
   | `--no-compile` | 约 38 dB，不随时间变化 | **逐位相同** |
   | `--no-compile` + `torch.use_deterministic_algorithms` / VAE decoder 也不编译 | 约 38 dB，无变化 | – |

   编译后 Inductor 按计时挑 kernel，两次运行的 bf16 舍入不同，并在自回归里放大；第 0 个 chunk 还没有任何
   历史就已经分叉。eager 下 latent 逐位相同，剩下的约 38 dB 来自 VAE decode / mp4 编码，不会回灌进 rollout。
3. eager 1.3B，把 WorldKV 布局按 `len_t` 4 缩放（sink 4 + 检索 8 + window 8，gap 8，chunk bank），8 个 chunk：
   resident vs offload 每个 chunk 的 latent **逐位相同**，有 bank、无 bank 都是。有效性检查：有 bank 与无 bank
   恰好在第 5 个 chunk（开始检索）分叉；offload 那一路每次 forward 实际 H2D 5.56 GB。
   没覆盖到：v1fast 14B、30 个 chunk、bank 驱逐。

**结论**：编译过的单次录制不可复现，之前所有低于约 1 dB 的质量差异（§2 的 revisit PSNR、§2.1 的 +0.3–0.45 dB、
§2.2、本节）都只能算噪声。以后做质量对比用 eager（`APP_ARGS=--no-compile`，同 seed 两组可以逐位配对；
1.3B 约 1.6 s/chunk，编译后约 1.16），或者编译但用 ≥ 8 个 seed 做统计。

**latent parity 工具**：`experiments/kv_offload/latent_parity.py` 包装 `flashdreams-run-v2`，把每次传给
`WanVAE.decode` 的 latent 存到 `$LATENT_DIR/<n>.pt`，`--compare` 逐位比较两次 dump（全部相同返回 0）：

```bash
P=experiments/kv_offload/latent_parity.py
for off in 0 1; do
  KV_SWEEP_MODEL=1p3b KV_SWEEP_WINDOW_T=8 KV_SWEEP_SINK_T=4 KV_SWEEP_RETRIEVE_T=8 KV_SWEEP_RETRIEVE_GAP_T=8 \
  KV_SWEEP_OFFLOAD=$off KV_SWEEP_ROPE=post KV_SWEEP_BANK_GRANULARITY=chunk KV_SWEEP_BANK_GB=24 \
  LATENT_DIR=$OUT/lat-$off .venv/bin/python $P lingbot.apps.cam2v.kv_sweep --mode mp4 \
    --output-path $OUT/off$off.mp4 --timeout unbound --backpressure-mode block --presentation-mode on_demand \
    -- --example-data --example-idx 0 --pose-path WorldKV/traj/R_L_R_L_R_L90_360/R_L_R_L_R_L90_360_poses.npy \
    --total-blocks 8 --warmup-blocks 0 --seed 42 --no-compile
done
.venv/bin/python $P --compare $OUT/lat-0 $OUT/lat-1
```

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

RTX PRO 6000 Blackwell（2026-11-20，`/mnt/datastore/yiti6755/kv_offload/h2d-1120-2155/h2d.jsonl`）：
pinned H2D 57.8 GB/s，D2H 57.3 GB/s（PCIe Gen5 x16，1–160 帧都一样；单 NUMA 节点，只测了 unbound）。
sweep 本身还没在这台机器上跑。

---

## 5. 在新节点上的推荐顺序

1. 第 0 节环境准备 → 第 1 节 `ci_cpu` 和 `ci_gpu` 全部通过。
2. 先单独跑一个 run 预热编译缓存：`RUNS=resident-w40 bash experiments/kv_offload/compare.slurm`
   （这一步也会下载权重）。
3. `compare.slurm`：先 `RUNS="offload-w40 ours"` 并开 `FLASHDREAMS_KV_OFFLOAD_TIMING=1`
   （验证滚窗开销是否消失），再跑 `RUNS="baseline resident-w40"` 补齐对照。
4. `bank_test.slurm` 跑 pre/post-RoPE 与粒度矩阵。
5. 需要带宽结论时再跑第 4 节的 sweep。
6. 质量对比加 `APP_ARGS=--no-compile`（§2.3：编译后的运行不可复现）。

新的结果写回 `WORLDCACHE.md` 的 Status，并在本文件的历史结果表里追加一行（注明 GPU 型号和 commit）。
