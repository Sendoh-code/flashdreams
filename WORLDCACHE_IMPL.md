# WorldCache 实现索引（代码位置 + 行号）

`WORLDCACHE.md` 记录目标/状态/下一步；本文件记录**每个已实现组件改在了哪里**，
精确到文件与行号。行号基于分支 `kv-offload-sweep` 上 KV bank / RoPE 提交（`944cdf13` 之后的下一个提交，2026-11-13）。
**每新增或修改一个组件，都要在本文件追加/更新对应小节和行号**（见文末「维护规则」）。

---

## 1. Layerwise host KV offload（offload + fetch + 逐层重叠）

引入于 `0712e9b7`。核心思想：每层 self-attention 的 K/V「家」在 pinned host
内存，GPU 上只有两个 slot（双缓冲）。第 `i` 层计算时，side stream 把第 `i+1`
层 H2D 拷进另一个 slot；第 `i` 层算完后只把本次写入的 chunk D2H 写回 host。

### 1.1 核心类 `LayerwiseKVOffload`

文件：`flashdreams/flashdreams/core/attention/kvcache.py`

| 行号 | 内容 | 说明 |
| --- | --- | --- |
| 8–15 | `json` / `os` / `statistics` / `nullcontext` / `TYPE_CHECKING` 等 import | 为 offload、计时、bank 类型注解新增 |
| 22–23 | `if TYPE_CHECKING: from ...kvbank import KVBank` | 避免与 `kvbank.py` 循环 import |
| 892–924 | 类定义 + docstring | 用法：`before_layer → block → after_layer`；920–923 说明 bank 行为 |
| 926–938 | `__init__(bank=None)` | `bank`（927，可选 `KVBank`）、`_slots`（2 个 GPU K/V 槽）、`_ready`（每槽一个 CUDA event）、`_home`（`id(cache)→` pinned host K/V）、`_stream`（side stream）、计时状态 |
| 940–966 | `add(cache)` | **offload**：前两个 cache 的 GPU buffer 直接复用为 slot（保留 LingBot TMA padding 等自定义布局，949）；之后每个 cache 分配 pinned host home（958–962），并把 `cache._k/_v` 换成 home（964）；有 bank 时同时 `bank.add(cache)`（965–966）。形状不一致直接报错（953–957） |
| 968–976 | `synchronize()` | `self._stream.synchronize()` 等待所有异步拷贝；顺带结算挂起的计时 |
| 978–982 | `begin_chunk(chunk_idx)` | 先 `synchronize()`，再 `bank.begin_chunk`（分配/驱逐 entry 要求无拷贝在途） |
| 984–988 | `reset()` | 同步后清空 bank entry（存储不释放） |
| 990–1005 | `before_layer(caches, i)` | **fetch + 重叠**：`i==0` 时先预取第 0 层（993–994）；预取第 `i+1` 层到 slot `(i+1)%2`（995–998）；计算流 `wait_event(_ready[i%2])`（1001）；把 cache 绑定到 GPU slot（1005） |
| 1007–1032 | `after_layer(caches, i)` | **write-back**：side stream `wait_stream(当前计算流)`（1017）；只拷 `_current_write_bounds()` 写入范围回 host（1012, 1021–1022）；有 bank 时再把当前 chunk 拷进 bank（1023–1026）；cache 绑回 host home（1029） |
| 1034–1037 | `_side_stream()` | CPU（无 CUDA）时退化为 `nullcontext`，拷贝变同步，便于 CPU 测试 |
| 1039–1075 | `_prefetch(caches, i)` | 拷 `[0, cache.size)` 可见前缀到 slot；有 bank 检索结果时，retrieve 区由 `bank.fetch` 填充，home 只拷其余两段（1047–1060）；然后 `_ready[slot].record(stream)`（1068）。`ponytail:`（1040–1043）：每个去噪步重拷整段历史，是已知 H2D 带宽上限 |
| 1077–1081 | `_record()` | 创建并记录带计时的 CUDA event |
| 1083–1136 | `_log_forward()` | 每次 forward 汇总每层 `fetch_ms / stall_ms / compute_ms / writeback_ms / fetch_mb`；`stall`（1096–1100）；可写 JSONL（1112–1118） |
| 1139– | `__all__` | 导出 `LayerwiseKVOffload` |

依赖的 `BlockKVCache` 已有方法（未改动）：
`size`（602），`_seq_slice`（657），`_current_write_bounds`（697），
`before_update`（768）中的窗口左滚 `_roll_local_window_left`（663，在 host home 上执行）。

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
- **host 侧修改前必须同步**：窗口左滚直接改 host home，所以
  `WanDiTNetworkCache.before_update` 先调 `begin_chunk()`（内含 `synchronize()`，见 1.3）。
- **不兼容 CUDA graph**：slot 重绑定（Python 侧换 tensor）+ side stream；`@torch.compiler.disable`（990, 1007）。

### 1.3 Wan recipe 接线

`flashdreams/flashdreams/recipes/wan/transformer/impl/network.py`

| 行号 | 内容 |
| --- | --- |
| 17 | `from flashdreams.core.attention import LayerwiseKVOffload` |
| 39 | `WanDiTNetworkCache.kv_offload` 字段 |
| 46–52 | `before_update`：offload 存在时先 `kv_offload.begin_chunk()`（48–50），再逐 block 滚窗口 |
| 59–64 | `reset`：先 `kv_offload.reset()`（61–62） |
| 330, 340 | `initialize_cache(..., kv_offload=None)` 参数与文档 |
| 362–363 | 每建一个 block cache 立刻 `kv_offload.add(block_cache.self_attn)`，峰值显存只到几层 |
| 522–537 | forward 中的逐层 hook：`before_layer` / `after_layer` 包住每个 block |

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

- 环境变量：`FLASHDREAMS_KV_OFFLOAD_TIMING=1`（kvcache.py:933），
  `FLASHDREAMS_KV_OFFLOAD_TIMING_JSONL=<path>`（kvcache.py:1112）。
- 事件记录点：fetch（1064, 1071–1075），compute（1004, 1015），writeback（1020, 1028）。
  `fetch_mb` 包含 bank 检索拷贝（1070）。

### 1.6 测试

`flashdreams/tests/test_kvcache.py`

| 行号 | 测试 |
| --- | --- |
| 182–233 | `test_layerwise_offload_matches_resident_caches`（`ci_cpu`）：3 层、6 个 chunk（越过 sink 触发滚窗）、每 chunk 2 个去噪步，逐步对比 offload 与常驻 cache 的 `cached_k/v` |
| 361–415 | `test_layerwise_offload_timing_reports_every_layer`（`ci_gpu`）：每层都有 fetch/compute 计时，`fetch_mb` 符合预期，JSONL 内容一致 |

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

思想：offload 写回时，把每个 chunk（或每个 latent frame）额外存进一个按预算分配的
pinned host bank，并带上元数据（位姿、动作……）。每个 chunk 开始时，可插拔的
`KVRetrievalPolicy` 从「已离开 sink 和窗口」的条目中选 k 个，prefetch 时拷进
cache 的 **retrieve 区**。每层 attention 看到 `[sink | retrieved | recent window]`。
retrieve 区就是 `BlockKVCache` sink 的尾部，所以窗口滚动不会动它。

### 3.1 核心 `flashdreams/flashdreams/core/attention/kvbank.py`

| 行号 | 内容 | 说明 |
| --- | --- | --- |
| 29–46 | `KVBankEntry` | `chunk_idx`、`frame`（全局 latent 帧号）、`num_frames`、`meta`、`slot`（bank 存储行，查询为 -1） |
| 50–72 | `KVRetrievalPolicy.select(bank, query, candidates, k)` | 策略基类；candidates 已去掉 sink 和窗口内条目，按时间从旧到新 |
| 76–86 | `RecentRetrieval` | 取最新 k 个，相当于更长的滑窗；默认策略 |
| 90–126 | `CameraPoseRetrieval` | `meta["poses"]` 每条目平均位姿；平移 L2 + 旋转测地距离，各自 max 归一化后按 `translation_weight` 加权，取 top-k 最近（与 WorldKV `image2video_fast.py` 一致） |
| 130–143 | `KVBankConfig` | `budget_gb`、`granularity`（`chunk`/`frame`）、`retrieve_size_t`（patchify 前帧数，须为条目大小的整数倍）、`policy` |
| 156–185 | `KVBank.__init__` | `entry_frames`、`k = retrieve_size_t / entry_frames`；倍数校验（167–171） |
| 187–198 | `add` / `entry_tokens` / `memory_start` | `memory_start = sink_size - k * entry_tokens`，即 retrieve 区起点 |
| 200–211 | `reset` / `capacity` | |
| 213–244 | `begin_chunk(chunk_idx)` | 为当前 chunk 分配条目（真 sink 内的帧跳过，231–232）；满了 FIFO 驱逐（`ponytail:` 234–236） |
| 246–278 | `annotate(chunk_idx, per_frame, **shared)` | 写入查询与条目元数据；`per_frame` 首维 = `frames_per_chunk`，frame 粒度时按帧切片 |
| 280–298 | `plan()` | 每 chunk 算一次：窗口起点之前的条目作为候选 → `policy.select` → 按时间排序 |
| 300–314 | `fetch(cache, slot_k, slot_v)` | 返回 retrieve 区的 `(dst, src)` H2D 拷贝 |
| 316–330 | `store(cache)` | 返回把当前写入范围拷进 bank 行的 D2H 拷贝 |
| 333–363 | `_allocate()` | 首次 `begin_chunk` 时按预算算容量（350 用 `round()`，`int()` 会把 239.99… B 截成少一个条目），为每个注册的 cache 分配 `[capacity, *entry_shape]` pinned K/V；要求 `window_size >= chunk_size` |

导出：`flashdreams/flashdreams/core/attention/__init__.py` 7–14、`__all__` 加 6 个名字。

### 3.2 接入 offload（`kvcache.py`）

`LayerwiseKVOffload(bank=...)`（926），`add` 注册（965–966），`begin_chunk` / `reset`（978–988），
`after_layer` 存 bank（1023–1026），`_prefetch` 用 bank 填 retrieve 区，home 拷剩下两段（1047–1060）。
检索不足 k 条时，retrieve 区剩余位置仍是 host home 的内容（即最早的 sink 之后几帧）。

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
| `flashdreams/tests/test_kvcache.py:236–247` | `_MatchAction`：按动作匹配的测试策略 |
| `test_kvcache.py:250–255` | `test_kv_bank_fills_retrieve_region`（`ci_cpu`）：chunk/frame × 无限/容量 2，10 个 chunk × 2 去噪步，逐层对照独立模型检查 attended K/V 与 FIFO 驱逐 |
| `test_kvcache.py:258–264` | `test_kv_bank_fills_retrieve_region_cuda`（`ci_gpu`）：同上，CUDA side stream + pinned 存储 |
| `test_kvcache.py:267–343` | `_check_kv_bank(granularity, capacity, device)` 共用主体 |
| `test_kvcache.py:346–358` | `test_camera_pose_retrieval_picks_nearest_pose`（`ci_cpu`） |

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

## 6. 待实现组件（实现后在此追加小节，写清文件与行号）

对应 `WORLDCACHE.md` → Next Steps：

- [ ] 去掉 `_prefetch` 的 `ponytail:` 上限：预取跳过写入范围 / 同一 chunk 的去噪步间保留历史（预计改 `kvcache.py` `_prefetch` 1010–1035、`after_layer` 982–1003）
- [ ] `core` 中模型无关的 KV policy 接口（存储层级 + 驱逐/检索策略），把 window/sink、full、offload 迁移上去
- [ ] chunk 级 KV identity（chunk index、相机位姿、prompt）与 store/lookup 路径
- [ ] 多 session / 显存预算 / 每 chunk 延迟

## 维护规则

1. 新组件：追加一个编号小节，按「文件 → 行号表 → 设计要点 → 测试」组织。
2. 改动已有组件：同步更新受影响的行号；更新顶部的基准 commit。
3. 检查行号：`grep -n "<符号>" <文件>`，不要凭记忆。
