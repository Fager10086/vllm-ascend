╭─── Claude Code v2.1.92 ──────────────────────────────────────────────────────╮
│                                    │ Tips for getting started                │
│            Welcome back!           │ Run /init to create a CLAUDE.md file w… │
│                                    │ ─────────────────────────────────────── │
│               ▐▛███▜▌              │ Recent activity                         │
│              ▝▜█████▛▘             │ No recent activity                      │
│                ▘▘ ▝▝               │                                         │
│                                    │                                         │
│   Sonnet 4.6 · API Usage Billing   │                                         │
│          /vllm-workspace           │                                         │
╰──────────────────────────────────────────────────────────────────────────────╯

✻ Conversation compacted (ctrl+o for history)

  ⎿  Read 
     vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py 
     (224 lines)
  ⎿  Read vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.p
     y.v18_correct (224 lines)
  ⎿  Read vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.p
     y.v9_correct (240 lines)
  ⎿  Read vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.p
     y.baseline (293 lines)
  ⎿  Read test_rope/test_split_qkv_rmsnorm_rope_raw_perf.py (275 lines)
  ⎿  Skills restored (triton-ascend-ops-optimizer)

● V18 实现优化要点全面总结

  一、Kernel 层面优化

  1. Grid 维度简化：2D → 1D

  - 基线：使用 (n_rows, n_cols, 1) 三维 Grid，其中 n_cols = kv_hidden_size / 
  KV_BLOCK_SIZE，Grid 大小与输入 shape 关联
  - V18：使用固定 (get_vectorcore_num(), ) 一维 Grid（40 核）
  - 收益：消除 Grid 维度计算的 Scalar 开销，简化 program_id 索引逻辑；符合
  Ascend NPU 的 1D Grid 最佳实践

  2. Kernel 入参数量大幅精简：20 → 16（其中动态指针 10 → 5）

  - 基线：10 个动态指针参数（input, cos_sin, pos, q_ptr, k_ptr, v_ptr,
  q_weight_ptr, q_bias_ptr, k_weight_ptr, k_bias_ptr）
  - V18：5 个动态指针参数（input_ptr, cos_sin_ptr, pos_ptr, output_ptr,
  weights_ptr）
  - 收益：每个动态参数约增加 ~0.011ms 启动开销，减少 5 个动态参数节省约 ~0.055ms
   kernel launch 时间

  3. 权重打包（Weight Packing）

  - 基线：q_weight、k_weight、q_bias、k_bias 作为 4 个独立指针传入 kernel
  - V18：在 Python 端预先 torch.cat([q_weight, k_weight]) 或
  torch.cat([q_weight, k_weight, q_bias, k_bias])，打包为单一 weights_ptr
  - Kernel 内部：通过偏移量访问各权重段
  q_weight_vals = tl.load(weights_ptr + hd_arange)              # offset 0
  k_weight_vals = tl.load(weights_ptr + HEAD_DIM + hd_arange)   # offset 
  HEAD_DIM
  q_bias_vals = tl.load(weights_ptr + 2 * HEAD_DIM + hd_arange) # offset 
  2*HEAD_DIM
  k_bias_vals = tl.load(weights_ptr + 3 * HEAD_DIM + hd_arange) # offset 
  3*HEAD_DIM
  - 收益：减少指针参数数量；权重在连续内存中，访存更高效

  4. 输出合并：3 个输出 buffer → 1 个

  - 基线：分别分配 q_output(batch, q_hidden_size)、k_output(batch, 
  kv_hidden_size)、v_output(batch, kv_hidden_size)，3 个独立指针传入 kernel
  - V18：分配单一 output(batch, total_hidden_size)，传入 1 个 output_ptr
  - Kernel 内部：通过偏移写入 Q/K/V 区域
  out_row_base = row_idx * total_hidden_size
  # Q: output_ptr + out_row_base + q_arange
  # K: output_ptr + out_row_base + q_hidden_size + kv_arange
  # V: output_ptr + out_row_base + q_hidden_size + kv_hidden_size + kv_arange
  - 收益：减少 2 个指针参数；Python 端减少 2 次 torch.empty 调用（每次
  ~0.012ms）

  5. 三循环合并为单循环

  - 基线：3 个独立的 tl.range 循环——分别处理
  Q（RMSNorm+RoPE）、K（RMSNorm+RoPE）、V（纯拷贝），每个循环独立遍历所有 batch
  - V18：单一 tl.range(pid, batch_size, num_cores) 循环，在每次迭代中依次处理 V
  → Q → K
  - 收益：
    - 循环管理开销减少 2/3
    - 消除了基线中 K 和 V 循环重复加载 pos_idx、cos/sin 的冗余
    - 每 token 的 cos/sin 仅加载一次，Q 和 K 共享使用

  6. cos/sin 共享加载

  - 基线：Q 循环和 K 循环各自独立加载 pos_idx、cos、sin（每 token 加载 2 次）
  - V18：在单循环内，每 token 仅加载一次 pos_idx、cos_val、sin_val，Q 和 K 的
  RoPE 计算共享使用
  - 收益：减少一半的 cos/sin 访存操作

  7. V 数据提前写入（Write Early / 边算边写）

  - 基线：V 的拷贝放在最后一个独立循环中
  - V18：在每个 token 的处理循环中，V 数据最先加载并立即写入输出
  # V: load and store immediately
  v_data = tl.load(input_ptr + input_row_base + q_hidden_size + kv_hidden_size +
   kv_arange)
  tl.store(output_ptr + out_row_base + q_hidden_size + kv_hidden_size +
  kv_arange, v_data)
  - 收益：V 的 MTE（Memory Transfer Engine）写入与后续 Q/K
  的向量计算并行执行，利用 MTE/Vector 流水并行隐藏访存延迟

  8. 消除所有 mask 和 other 参数

  - 基线：所有 tl.load 使用 mask=valid_mask, other=0.0，所有 tl.store 使用
  mask=valid_mask
  - V18：所有 tl.load 和 tl.store 均无 mask 和 other 参数
  - 前提：因为 q_hidden_size 和 kv_hidden_size 本身就是 head_dim
  的整数倍，q_arange = tl.arange(0, q_hidden_size) 不会越界，无需 mask
  - 收益：
    - other=0.0 在底层触发 tl.where，消除后避免了 load 后的额外 Scalar 操作
    - 无 mask 的 load/store 可更好地流水线并行

  9. do_not_specialize 防止重编译

  - 基线：无 do_not_specialize，batch_size 作为普通标量参数，每次 batch_size
  变化都触发重编译
  - V18：@triton.jit(do_not_specialize=["batch_size"])
  - 收益：在推理场景下 batch_size 频繁变化时，避免秒级重编译开销

  10. 循环内偏移计算简化

  - 基线：维护 input_offset、output_offset，在循环末尾手动 += offset_step
  - V18：在循环内直接通过 row_idx * total_hidden_size 计算偏移，无需维护累加变量
  - 收益：减少循环内 Scalar 运算，代码更清晰

  ---
  二、Python Wrapper 层面优化

  1. 单次内存分配替代三次

  - 基线：
  q_output = torch.empty(batch_size, q_hidden_size, ...)
  k_output = torch.empty(batch_size, kv_hidden_size, ...)
  v_output = torch.empty(batch_size, kv_hidden_size, ...)
  - V18：
  output = torch.empty(batch_size, total_hidden_size, ...)
  - 返回时零拷贝切片：
  q_output = output[:, :q_hidden_size]
  k_output = output[:, q_hidden_size:q_hidden_size + kv_hidden_size]
  v_output = output[:, q_hidden_size + kv_hidden_size:]
  - 收益：节省约 ~0.023ms（2 次 torch.empty 调用，每次 ~0.012ms）

  2. Grid 全局缓存

  - 基线：每次调用都计算 num_vectorcore = get_vectorcore_num() 和 Grid
  - V18：
  _GRID = None
  if _GRID is None:
      _GRID = (get_vectorcore_num(), )
  - 收益：避免重复调用 get_vectorcore_num() 的开销

  3. 权重打包缓存（双路缓存）

  - V18 通过 data_ptr() 进行身份检查，且为 bias/no-bias 场景维护独立缓存：
  _PACKED_WEIGHTS_NOBIAS = None
  _PACKED_WEIGHTS_NOBIAS_KEY = None
  _PACKED_WEIGHTS_BIAS = None
  _PACKED_WEIGHTS_BIAS_KEY = None
  - 关键设计：bias 和 no-bias
  使用分离的缓存变量，避免同一模型中两种调用路径互相覆盖缓存导致的错误（这是 V18
   初始版本 1/40 测试失败的 bug 修复点）
  - 收益：torch.cat 仅在首次或权重变更时执行，后续调用直接复用打包后的 tensor

  4. 移除不必要的断言

  - 基线：包含 assert head_dim == KV_BLOCK_SIZE、assert q_hidden_size % 
  kv_hidden_size == 0、assert num_vectorcore % n_cols == 0
  - V18：移除所有断言
  - 收益：微量减少 Python 解释器开销

  ---
  三、性能效果

  ┌──────┬───────────────┬────────┐
  │ 版本 │ 中位耗时 (ms) │ 加速比 │
  ├──────┼───────────────┼────────┤
  │ 基线 │ 0.271         │ 1.00x  │
  ├──────┼───────────────┼────────┤
  │ V18  │ 0.261         │ 1.04x  │
  └──────┴───────────────┴────────┘

  性能瓶颈分析：该算子在 num_tokens=1024 场景下，94% 的执行时间（~0.24ms）为
  Triton kernel launch + NPU dispatch 的固定开销，实际计算时间仅 ~0.01ms。kernel
   内部优化已接近极限（计算量本身极小），进一步提升需要从 Triton 运行时或 NPU
  驱动层面减少 launch 开销。

✻ Cogitated for 3m 19s
