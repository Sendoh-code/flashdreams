# WorldCache 实现索引（代码位置 + 行号）

`WORLDCACHE.md` 记录目标/状态/下一步；本文件记录**每个已实现组件改在了哪里**，
精确到文件与行号。行号基于分支 `kv-offload-sweep` 的 `711ff4f5` 加上「bank 作为唯一 host 存储」改动（§6，2026-11-18，未提交）。
**每新增或修改一个组件，都要在本文件追加/更新对应小节和行号**（见文末「维护规则」）。

---

## 1. Layerwise host KV offload（offload + fetch + 逐层重叠）

引入于 `0712e9b7`。核心思想：每层 self-attention 的 K/V 在 pinned host
内存，GPU 上只有两个 slot（双缓冲）。第 `i` 层计算时，side stream 把第 `i+1`
层 H2D 拼进另一个 slot；第 `i` 层算完后只把本次写入的 chunk D2H 写回 host。
host 侧只有 sink buffer + `KVBank` 条目（§6），不再有整层 home。

### 1.1 核心类 `LayerwiseKVOffload`

文件：`flashdreams/flashdreams/core/attention/kvcache.py`

| 行号 | 内容 | 说明 |
| --- | --- | --- |
| 8–21 | `json` / `os` / `statistics` / `nullcontext` 等 import；21 运行时 `from ...kvbank import KVBank, KVBankConfig` | `kvbank.py` 只在 `TYPE_CHECKING` 下 import `kvcache`，无循环 |
| 890–925 | 类定义 + docstring | host 布局（sink buffer + bank）、meta 绑定、用法；每个 forward 前必须 `begin_chunk` |
| 927–941 | `__init__(bank=None)` | `bank`（调用方的 bank，可为 None，LingBot 用它 `annotate`）、`_bank`（实际存储，`add` 时缺省建一个）、`_slots`、`_ready`、`_sink`（`id(cache)→` pinned sink K/V）、`_idle`（`id(cache)→` meta K/V）、`_stream`、计时状态 |
| 943–980 | `add(cache)` | 前两个 cache 的 GPU buffer 复用为 slot（保留 LingBot TMA padding）；形状不一致报错；无 bank 时建**只装窗口**的 token 单位 bank（961–969）；分配 `sink_size` 长的 pinned sink buffer（971–976）；`cache._k/_v` 换成 meta tensor（977–980） |
| 982–990 | `synchronize()` | 等 side stream；结算计时 |
| 992–996 | `begin_chunk(chunk_idx)` | `synchronize()` 后 `_bank.begin_chunk`（驱逐把 host 行交给新 chunk，必须无拷贝在途） |
| 998–1002 | `reset()` | 同步后清空 bank 条目 |
| 1004–1019 | `before_layer(caches, i)` | 预取 `i+1` 层；计算流 `wait_event(_ready[slot])`（1015）；绑定 GPU slot |
| 1021–1046 | `after_layer(caches, i)` | `wait_stream`（1032）；写入范围落在 sink 内的部分拷回 sink buffer；当前 chunk 经 `_bank.store` 拷进 bank 行（1040）；cache 绑回 meta（1044） |
| 1048–1051 | `_side_stream()` | 无 CUDA 时退化为 `nullcontext` |
| 1053–1089 | `_prefetch(caches, i)` | slot = retrieve 区（`bank.fetch`）+ 窗口（`bank.window`，1070）+ sink buffer 其余两段；`_ready[slot].record`（1082）。`ponytail:`（1055–1058）：每个去噪步重拷整段历史 |
| 1091–1094 | `_record()` | 计时 CUDA event |
| 1096–1150 | `_log_forward()` | 每层 `fetch_ms / stall_ms / compute_ms / writeback_ms / fetch_mb`；JSONL（1126） |
| 1153– | `__all__` | |

依赖的 `BlockKVCache` 方法（未改动）：
`size`（600），`_seq_slice`（655），`_current_write_bounds`（695），
`before_update`（766）中的窗口左滚 `_roll_local_window_left`（661）——offload 下它作用在 meta tensor 上，不搬任何内存。

导出：`flashdreams/flashdreams/core/attention/__init__.py` 中 import 与 `__all__` 各加一行 `LayerwiseKVOffload`。

### 1.2 重叠的时序与正确性

一次 forward（L 层，单个 CFG 分支）：

```
side stream : [fetch 0][fetch 1]      [wb 0][fetch 2]      [wb 1][fetch 3] ...
compute     :          wait r0 [layer 0]    wait r1 [layer 1]    wait r2 ...
```

- **slot 复用安全**：`before_layer(i)` 往 slot `(i+1)%2` 预取，该 slot 上一次装的是第
  `i-1` 层；`after_layer(i-1)` 已让 side stream 等待第 `i-1` 层计算（1017），并在同一个
  in-order side stream 上先排了它的 write-back，所以预取不会覆盖未写回/未算完的数据（注释 996–997）。
- **计算等数据**：计算流只在 `_ready[slot]` event 上等待（1001），其余时间与 H2D 并行。
- **跨 forward**：同一 chunk 的多个去噪步、CFG cond/uncond 共享同一个 offload 和两个 slot；
  side stream 的顺序保证上一 forward 最后的 write-back 先于下一 forward 的 fetch。
- **驱逐前必须同步**：bank 驱逐把 host 行交给新 chunk，所以
  `WanDiTNetworkCache.before_update` 先调 `begin_chunk()`（内含 `synchronize()`，见 1.3）。窗口左滚本身已不碰 host 内存（meta）。
- **不兼容 CUDA graph**：slot 重绑定（Python 侧换 tensor）+ side stream；`@torch.compiler.disable`（1004, 1021）。

### 1.3 Wan recipe 接线

`flashdreams/flashdreams/recipes/wan/transformer/impl/network.py`

| 行号 | 内容 |
| --- | --- |
| 19 | `from flashdreams.core.attention import LayerwiseKVOffload` |
| 41 | `WanDiTNetworkCache.kv_offload` 字段 |
| 48–62 | `before_update`：offload 存在时先 `kv_offload.begin_chunk()`（52–54），再逐 block 滚窗口；`FLASHDREAMS_KV_OFFLOAD_TIMING=1` 时打印 `begin_chunk` / `roll` 的 host 耗时（57–62） |
| 69–72 | `reset`：先 `kv_offload.reset()` |
| 340 | `initialize_cache(..., kv_offload=None)` 参数 |
| 372–373 | 每建一个 block cache 立刻 `kv_offload.add(block_cache.self_attn)`，峰值显存只到几层 |
| 534–547 | forward 中的逐层 hook：`before_layer` / `after_layer` 包住每个 block |

`flashdreams/flashdreams/recipes/wan/transformer/wan21.py`

| 行号 | 内容 |
| --- | --- |
| 72–83 | `Wan21TransformerCache.start()` 调 `network_cache(_uncond).before_update` —— 因为 `predict_flow` 用 `eager_mode=False`（685），同步点实际在这里触发 |
| 194 | `Wan21TransformerConfig.kv_offload: bool = False` |
| 294–295 | `kv_offload` 与 `use_cuda_graph=True` 互斥，报错 |
| 416–456 | `_build_network_cache(..., kv_offload)` 透传 |
| 510–536 | 创建**一个** `LayerwiseKVOffload`（可带 `KVBank`，见 §3），cond 与 uncond 两个 network cache 共享（524, 536） |

### 1.4 LingBot 集成

| 文件:行号 | 内容 |
| --- | --- |
| `integrations_v2/lingbot/impl/transformer/impl/network.py:151–178` | 覆写 `initialize_cache`：offload 时关闭 `reuse_key_history`（174–177），因为 Sage key-sum 会在 `before_update` 里从（已在 host 的）K buffer 读取 |
| `integrations_v2/lingbot/impl/transformer/impl/modules.py:264, 280, 376–382, 437` | `reuse_key_history` 的定义与使用（被上面关闭；pre-RoPE 时也关闭，见 §4） |
| `integrations_v2/lingbot/config.py:204–217` | preset `lingbot-world-v2-1p3b-causal-fast-perf-fullkv-offload`：`window_size_t=80, sink_size_t=0, kv_offload=True`；注册进 `PIPELINE_CONFIGS`（219–） |
| `integrations_v2/lingbot/apps/cam2v/adapter.py:128–132, 141` | `create_app_v2_1p3b_causal_fast_perf_fullkv_offload` |
| `integrations_v2/lingbot/pyproject.toml:37` | 对应 app entry point |

### 1.5 计时 / 观测

- 环境变量：`FLASHDREAMS_KV_OFFLOAD_TIMING=1`（kvcache.py:936），
  `FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=<path>`（kvcache.py:1126）。
- 事件记录点在 `_prefetch` / `before_layer` / `after_layer` 内；`fetch_mb` 包含 sink、检索和窗口三类拷贝。
- 每 chunk 的 host 侧 `begin_chunk` / `roll` 耗时：Wan `network.py:57–62` 的日志行。

### 1.6 测试

`flashdreams/tests/test_kvcache.py`

| 行号 | 测试 |
| --- | --- |
| 181–237 | `test_layerwise_offload_matches_resident_caches`（`ci_cpu`）：参数化 `(sink, window)` ∈ (2,4)/(1,4)/(0,5)/(1,5)，含 sink、window 与 chunk 不对齐；3 层、6 个 chunk、每 chunk 2 个去噪步，逐步对比 offload 与常驻 cache 的 `cached_k/v`；最后断言 cache 是 meta、sink buffer 长度 = sink、bank 容量 = ⌈window/chunk⌉ |
| 367–422 | `test_layerwise_offload_timing_reports_every_layer`（`ci_gpu`）：每层都有 fetch/compute 计时，`fetch_mb` 符合预期，JSONL 内容一致 |

---

## 2. A40 H2D 交叉点实验（`944cdf13`）

| 文件:行号 | 内容 |
| --- | --- |
| `integrations_v2/lingbot/apps/cam2v/kv_sweep.py:25–` | `create_app()`：从 `KV_SWEEP_WINDOW_T / KV_SWEEP_SINK_T / KV_SWEEP_OFFLOAD` 读 KV 策略，bank 与 RoPE 开关见 §3.4 / §4；sm86 用 `linear_backend="torch"`、`self_attention_backend="wan"`（49–53） |
| `experiments/kv_offload/h2d_bench.py:17–47` | `FRAME_MB`（每帧每层 K+V MB）、`bench()`（21）测 pinned H2D/D2H GB/s、`main()`（36） |
| `experiments/kv_offload/sweep.slurm:1–114` | 窗口 × 常驻/offload 矩阵，NUMA 绑定，串行执行（用法见 1–22 行注释） |
| `experiments/kv_offload/analyze.py` | `forward_rows`（26，丢弃 warmup）、`crossover`（52，线性拟合 fetch/compute 求 V*，及 stall>5% 的经验 V*）、`step_wall_s`（77）、`dmon_rx_gbps`（89）、`analyze`（107）、`plot`（158）、`self_check`（181） |

---

## 3. KV bank（预算化 host K/V 存储 + 元数据检索）

思想：offload 写回时，把每个 chunk（或每个 latent frame）存进一个按预算分配的
pinned host bank，并带上元数据（位姿、动作……）。bank 也是窗口的唯一 host 副本（§6）。每个 chunk 开始时，可插拔的
`KVRetrievalPolicy` 从「已离开 sink 和窗口」的条目中选 k 个，prefetch 时拷进
cache 的 **retrieve 区**。每层 attention 看到 `[sink | retrieved | recent window]`。
retrieve 区就是 `BlockKVCache` sink 的尾部，所以窗口滚动不会动它。

### 3.1 核心 `flashdreams/flashdreams/core/attention/kvbank.py`

| 行号 | 内容 | 说明 |
| --- | --- | --- |
| 30–47 | `KVBankEntry` | `chunk_idx`、`frame`（全局 latent 帧号）、`num_frames`、`meta`、`slot`（bank 存储行，查询为 -1） |
| 51–73 | `KVRetrievalPolicy.select(bank, query, candidates, k)` | 策略基类；candidates 已去掉 sink 和窗口内条目，按时间从旧到新 |
| 77–87 | `RecentRetrieval` | 取最新 k 个；默认策略 |
| 91–127 | `CameraPoseRetrieval` | 平移 L2 + 旋转测地距离，max 归一化后加权取 top-k（与 WorldKV `image2video_fast.py` 一致） |
| 131–147 | `KVBankConfig` | `budget_gb`（低于窗口所需时自动抬到窗口大小，`0` = 只装窗口）、`granularity`、`retrieve_size_t`、`policy` |
| 150–189 | `KVBank.__init__` | `entry_frames`、`k = retrieve_size_t / entry_frames` |
| 191–202 | `add` / `entry_tokens` / `memory_start` | `memory_start = sink_size - k * entry_tokens`，retrieve 区起点 |
| 204–215 | `reset` / `capacity` | |
| 217–259 | `begin_chunk(chunk_idx)` | 分配当前 chunk 的条目；只跳过**完全**在真 sink 内的条目（241，跨 sink 边界的 chunk 也存，因为它的尾部在窗口里）；满了 FIFO 驱逐（`ponytail:` 243），驱逐仍在窗口（235–238）内的条目时 `RuntimeError`（246–250） |
| 261–293 | `annotate(chunk_idx, per_frame, **shared)` | 写入查询与条目元数据 |
| 295–316 | `plan()` | 候选 = 起点 ≥ 真 sink（308）且整体在窗口之前的条目 → `policy.select` → 按时间排序 |
| 318–332 | `fetch(cache, slot_k, slot_v)` | retrieve 区的 `(dst, src)` H2D 拷贝 |
| 334–370 | `window(cache, slot_k, slot_v)` | **窗口 = bank 索引**：从最新条目倒序，按 rollout token 位置拼出 slot 的 `[sink_size, size)`；末尾断言覆盖的 token 数 = 窗口长度 |
| 372–386 | `store(cache)` | 当前写入范围 → bank 行的 D2H 拷贝 |
| 388–418 | `_allocate()` | 容量 = `max(round(budget)//entry_bytes, ⌈window_frames/entry_frames⌉)`（406–408） |

导出：`flashdreams/flashdreams/core/attention/__init__.py` 7–14、`__all__` 加 6 个名字。

### 3.2 接入 offload（`kvcache.py`）

`LayerwiseKVOffload(bank=...)`（927），`add` 注册（969），`begin_chunk` / `reset`（992–1002），
`after_layer` 存 bank（1040），`_prefetch` 用 bank 填 retrieve 区和窗口，sink buffer 拷剩下两段（1053–1089）。
检索不足 k 条时，retrieve 区剩余位置是 sink buffer 的内容（即 filling 阶段写进去的、真 sink 之后几帧）。

### 3.3 Wan recipe 接线

| 文件:行号 | 内容 |
| --- | --- |
| `recipes/wan/transformer/wan21.py:204–217` | `Wan21TransformerConfig.kv_bank: KVBankConfig \| None`，docstring 写明 RoPE 语义（§4） |
| `wan21.py:296–304` | 校验：需要 `kv_offload`；`retrieve_size_t % patch_t == 0`；frame 粒度要求 `cp_size == 1` |
| `wan21.py:409–413` | `_retrieve_size_t` 属性 |
| `wan21.py:439` | `BlockKVCache` 的 sink = `sink_size_t + retrieve_size_t` |
| `wan21.py:510–520` | 创建 `KVBank(frames_per_chunk, sink_frames, window_frames)` 并传给 offload |
| `wan21.py:553` | cache-relative RoPE 的 `sink_size_t` 同样加上 retrieve 区 |
| `recipes/wan/transformer/impl/network.py:50, 61–62` | `before_update → begin_chunk`，`reset → kv_offload.reset` |

### 3.4 LingBot 集成

| 文件:行号 | 内容 |
| --- | --- |
| `integrations_v2/lingbot/impl/encoder/camctrl.py:76–78, 184` | `I2VCamCtrlEmbeddings.poses`：每个 latent 帧的绝对 c2w 位姿 `[..., T, 4, 4]` |
| `integrations_v2/lingbot/impl/transformer/__init__.py:219–224` | `predict_flow` 每 chunk 一次 `bank.annotate(per_frame={"poses": ...})`（`start()` → `begin_chunk` 先于它） |
| `integrations_v2/lingbot/impl/transformer/__init__.py:248` | patchify 时保留 `poses` |
| `integrations_v2/lingbot/apps/cam2v/kv_sweep.py:29–40` | `KV_SWEEP_RETRIEVE_T>0` 时启用 `CameraPoseRetrieval` bank（隐含 offload）；`KV_SWEEP_BANK_GB`（默认 8）、`KV_SWEEP_BANK_GRANULARITY` |

### 3.5 测试

| 文件:行号 | 测试 |
| --- | --- |
| `flashdreams/tests/test_kvcache.py:240–251` | `_MatchAction`：按动作匹配的测试策略 |
| `test_kvcache.py:254–259` | `test_kv_bank_fills_retrieve_region`（`ci_cpu`）：chunk/frame × 无限/容量 2（frame 粒度时被抬到窗口的 4 条），10 个 chunk × 2 去噪步，逐层对照独立模型检查 attended K/V 与 FIFO 驱逐 |
| `test_kvcache.py:262–268` | `test_kv_bank_fills_retrieve_region_cuda`（`ci_gpu`）：同上，CUDA side stream + pinned 存储 |
| `test_kvcache.py:271–349` | `_check_kv_bank(granularity, capacity, device)` 共用主体 |
| `test_kvcache.py:352–364` | `test_camera_pose_retrieval_picks_nearest_pose`（`ci_cpu`） |

已知限制：FIFO 驱逐；每个去噪步都写 bank（只有最后一步有用）；`CameraPoseRetrieval` 把 batch 维也平均（B>1 不对）。

---

## 4. Pre-RoPE / Post-RoPE K 缓存可选

开关就是已有的 `network.apply_rope_before_kvcache`（`recipes/wan/transformer/impl/network.py:104`）。
bank 原样存 cache 里的 K，所以不另设 bank 级开关，以免两个开关不一致：

- `True`（post-RoPE，默认）：K 写入 cache 前用全局位置旋转（`RotaryPositionEmbedding3D`）。
  检索回来的条目**保留它原来的 rollout 时间位置**。
- `False`（pre-RoPE）：cache 存未旋转 K，读时按 cache 槽位旋转
  （`KVCacheRelativeRotaryPositionEmbedding3D`）。检索条目**放在 retrieve 槽位的位置上**，
  要求 `sink_size_t + retrieve_size_t + window_size_t` 是 `len_t` 的整数倍（RoPE 类里有 assert）。

| 文件:行号 | 内容 |
| --- | --- |
| `core/attention/rope.py:471–480` | `apply_rope_freqs_copy`：out-of-place、`@torch.compiler.disable` 的 cached-K 旋转。`ponytail:` 每次调用一个 graph break；torch 2.11 Inductor 的 `decompose_triton_kernel_wrapper_functional` 对 compile 内的该 Triton kernel 报 AssertionError（job 23900078 的 pre-RoPE 两个 run） |
| `recipes/wan/transformer/impl/modules.py:217, 274–278, 286–301` | base `SelfAttention` 的两种路径与 `_slice_rope_freqs`；pre-RoPE 读时旋转改用 `apply_rope_freqs_copy`（278） |
| `recipes/wan/transformer/wan21.py:550–555` | 已有：按开关选 RoPE adapter；本次让 relative 的 sink 包含 retrieve 区 |
| `integrations_v2/lingbot/impl/transformer/impl/modules.py:139, 146` | `OptimizedSelfAttention(apply_rope_before_kvcache=...)` |
| `lingbot/.../modules.py:215–224` | fused 路径：`_slice_rope_freqs` 取 q/k 频率；post 旋转后写 cache；pre 写未旋转 K，attention 前用 `apply_rope_freqs_copy(cached_k())` 旋转（224）；三个后端统一用 `key` |
| `lingbot/.../modules.py:310, 318, 328` | `CamCtrlBlock(apply_rope_before_kvcache=...)` 透传给 `Block` 与 `OptimizedSelfAttention` |
| `lingbot/.../modules.py:376–382` | pre-RoPE 时关闭 Sage `reuse_key_history`（存的是未旋转 K 的和） |
| `integrations_v2/lingbot/impl/transformer/impl/network.py:190` | `_build_block` 传入 `self.apply_rope_before_kvcache`（此前 LingBot 一直强制 post） |
| `integrations_v2/lingbot/apps/cam2v/kv_sweep.py:31–33, 45, 49–53` | `KV_SWEEP_ROPE=pre\|post`（默认 post），名字后缀 `-prerope` |

测试：`integrations_v2/lingbot/tests/test_rope_kvcache.py:30–` `test_pre_and_post_rope_cache_match_without_roll`
（`ci_gpu`）：sink 0、不滚窗时槽位位置 = 全局位置，两种模式的 `CamCtrlBlock` 输出应一致，
而 cache 内容不同（最后比较 `_k` 原始 buffer，`after_update` 之后不能调 `cached_k()`）。参数化 `wan`，以及 sm89+ 才跑的 `scaled_fp8`（A40 会 skip）。

---

## 5. GPU 验证脚本

`experiments/kv_offload/bank_test.slurm`（`gpuA40x4-interactive`，1h；每个 run `RUN_TIMEOUT` 默认 900 s；`TORCHINDUCTOR_CACHE_DIR` 持久化到 `/work`。`RUNS`=要跑的 run 名单，`TESTS=0` 跳过单测，`OUT` 指定共享输出目录，用法见脚本头注释——两个 job 串行拆分以满足 1h 上限）：
1. `ci_cpu` bank/offload 测试；`ci_gpu` bank（CUDA）、offload 计时、LingBot RoPE 一致性。
2. LingBot 1.3B 端到端（window 12、sink 4、16 blocks、seed 42）：`resident-post`、
   `resident-pre`、`bank-chunk-post`、`bank-chunk-pre`、`bank-frame-post`（retrieve 8 帧，
   `CameraPoseRetrieval`，bank 4 GB）。输出在 `/work/nvme/bicu/ytian6/kv_offload/bank-<jobid>/`，
   `summary.txt` 汇总状态和相对**同 RoPE 模式** resident 的 PSNR（只在该 resident 成功、留下 `ok` 标记时计算）。

---

## 6. bank 作为唯一 host 存储（窗口 = bank 索引，2026-11-18）

**问题**：offload 下 `cache._k/_v` 在两层之间指向整层 pinned host home，`before_update`
的窗口左滚（`_roll_local_window_left`，`clone` + 回写）就在 CPU 上搬整个窗口：
LingBot 1.3B W40/S6 实测每 chunk 约 1.84 s，占 5.1 s/chunk 的 36%
（`/work/nvme/bicu/ytian6/kv_offload/compare-24082157/offload-w40/run.log`）。
同时有 bank 时每个窗口 chunk 在 host 上存两份（home + bank），D2H 也写两次。

**做法**：
- host 只剩 per-cache 的 sink buffer（`sink_size` = 真 sink + retrieve 区）+ `KVBank` 条目；
  没有调用方 bank 时，`add` 建一个 token 单位、只装窗口的 bank（kvcache.py 961–969）。
- 两层之间 cache 绑 meta tensor（kvcache.py 977–980, 1044），`BlockKVCache` 的滚窗逻辑不改，只是不再碰内存。
- prefetch 时 `KVBank.window`（kvbank.py 334–370）按 rollout token 位置
  `g = p + (curr_chunk+1)·chunk_size − size` 从最新条目拼出 slot 的窗口区；slot 里仍是逻辑顺序，
  所以 attention、pre/post-RoPE 都不用改。
- 写回：写入范围在 sink 内的部分进 sink buffer，整个 chunk 进 bank（kvcache.py 1021–1046）。
- bank 存储规则：只跳过完全在真 sink 内的条目；候选要求起点 ≥ 真 sink（与旧行为一致）；
  容量至少 ⌈window/entry⌉；FIFO 驱逐窗口内条目直接报错。

**预期**：滚窗 ~1.84 s → ~0；`ours`（W40/S6/R8，24 GB bank）host 内存从 ~15 GB home + 24 GB bank
降到 ~4 GB sink + 24 GB bank（窗口已在 bank 预算内，可检索的旧条目相应少约 10 个 chunk）；D2H 减半。
H2D 字节数不变（每层每条目一次拷贝，frame 粒度下拷贝调用数 ×4，未合并相邻行）。

**测试**：§1.6 的参数化 parity 测试（含 sink/window 与 chunk 不对齐）与 §3.5 的 bank 测试；
`uv run pytest -m ci_cpu flashdreams/tests/test_kvcache.py` 19 passed（2026-11-18，login node）。GPU 端到端尚未跑。

---

## 7. 待实现组件（实现后在此追加小节，写清文件与行号）

对应 `WORLDCACHE.md` → Next Steps：

- [x] host home 与 bank 合并、窗口 = bank 索引（§6）
- [ ] 去掉 `_prefetch` 的 `ponytail:` 上限：预取跳过写入范围 / 同一 chunk 的去噪步间保留历史（`kvcache.py` `_prefetch` 1053–1089、`after_layer` 1021–1046）
- [ ] `core` 中模型无关的 KV policy 接口（存储层级 + 驱逐/检索策略），把 window/sink、full、offload 迁移上去
- [ ] chunk 级 KV identity（chunk index、相机位姿、prompt）与 store/lookup 路径
- [ ] 多 session / 显存预算 / 每 chunk 延迟

## 维护规则

1. 新组件：追加一个编号小节，按「文件 → 行号表 → 设计要点 → 测试」组织。
2. 改动已有组件：同步更新受影响的行号；更新顶部的基准 commit。
3. 检查行号：`grep -n "<符号>" <文件>`，不要凭记忆。
