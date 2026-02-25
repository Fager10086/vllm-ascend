# split_qkv_rmsnorm_mrope_kernel 逐行解析

> 源文件：`vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py`
> PR：https://github.com/vllm-project/vllm-ascend/pull/6730

---

## 1. 整体功能

这个 Triton kernel 是一个 **三合一融合算子**：

1. **Split QKV** — 把 fused QKV 张量按 `[q_size, kv_size, kv_size]` 切分成 Q、K、V
2. **RMSNorm** — 对 Q 和 K 分别做 RMSNorm（逐 head 归一化）
3. **MRoPE** — 对归一化后的 Q 和 K 施加多分辨率旋转位置编码（Multiresolution Rotary Position Embedding），支持 interleaved 和 non-interleaved 两种频率排布

---

## 2. Program_id 维度解析

```python
split_qkv_rmsnorm_mrope_kernel[(block_dim,)](...)
```

该 kernel **只有 1 个 `program_id` 维度** — `tl.program_id(0)`，代表 **核心编号（vector core index）**。

- `block_dim` 等于昇腾 NPU 上的 vector core 数量（通过 `get_vectorcore_num()` 获取，通常为 20 或更多）
- 没有第二或第三维度
- 调度策略：把 `num_tokens` 个 token **均匀分配**到所有 core 上

### 前端 core vs 尾端 core 的负载均衡

```python
front_core_num = core_num
if num_tokens % core_num != 0:
    front_core_num = num_tokens % core_num

num_tokens_each_front_core = ceil(num_tokens / core_num)

tail_core_num = core_num - front_core_num   # (if num_tokens > core_num)
num_tokens_each_tail_core = num_tokens // core_num
```

当 `num_tokens` 不能被 `core_num` 整除时：

- **前 `front_core_num` 个 core** 每个处理 `ceil(num_tokens/core_num)` 个 token（多分 1 个）
- **后 `tail_core_num` 个 core** 每个处理 `floor(num_tokens/core_num)` 个 token

**举例**：`num_tokens=1024, core_num=20` → 前 4 个 core 各处理 52 个 token，后 16 个 core 各处理 51 个 token。

---

## 3. 每个 pid 负责的 Tile 形状

每个 core 在 for 循环中 **逐 token 处理**，每次迭代处理 1 个 token 的全部 heads：

| 数据 | Tile 形状 | 说明 |
|------|-----------|------|
| Q 加载 | `(num_q_heads, head_size)` | 如 `(8, 128)` = 1024 元素 |
| K 加载 | `(num_kv_heads, head_size)` | 如 `(2, 128)` = 256 元素 |
| V 加载 | `(kv_size,)` | 一维，直接搬运 |
| cos/sin | `(1, half_rope_dim)` → broadcast 到 `(1, rope_dim)` | 如 `(1, 64)` → `(1, 128)` |
| Q RMSNorm | `(num_q_heads, head_size)` | 逐 head 归一化 |
| K RMSNorm | `(num_kv_heads, head_size)` | 逐 head 归一化 |
| Q RoPE 切片 | `(num_q_heads, half_rope_dim)` × 2 | x1, x2 各取一半 |
| K RoPE 切片 | `(num_kv_heads, half_rope_dim)` × 2 | y1, y2 各取一半 |

**总结：每个 pid 的 tile 是 `1 token × 所有 heads × head_size`，以 token 维度分 tile。**

---

## 4. 逐行分段解析：搬运 vs 计算

### 段1：参数加载（搬运/状态初始化）— 循环外

```python
block_idx = tl.program_id(0)

loop_num = num_tokens_each_front_core
if block_idx >= front_core_num:
    loop_num = num_tokens_each_tail_core

block_offset = num_tokens_each_front_core * block_idx
if block_idx >= front_core_num:
    block_offset = (
        num_tokens_each_front_core * front_core_num
        + (block_idx - front_core_num) * num_tokens_each_tail_core
    )
```

→ **状态更新**：根据 core 编号计算该 core 负责的 token 起始偏移和循环次数。

```python
q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

if has_bias:
    q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
    k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))
```

→ **搬运**：把 RMSNorm 的 weight/bias 一次性加载到寄存器，所有 token 复用。形状为 `(head_size,)`。

---

### 段2：循环体 — Load 阶段（搬运）

```python
for index in range(loop_num):
    ## load ##
    # q
    in_q_offset = in_qkv_ptr + (block_offset + index) * (q_size + 2 * kv_size)
    in_q_tensor = tl.load(in_q_offset + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)

    # k
    in_k_offset = in_qkv_ptr + (block_offset + index) * (q_size + 2 * kv_size) + q_size
    in_k_tensor = tl.load(in_k_offset + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)

    # v
    in_v_offset = in_qkv_ptr + (block_offset + index) * (q_size + 2 * kv_size) + q_size + kv_size
    in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))
```

→ **搬运（Split QKV）**：从 fused QKV 张量中按偏移读取 Q、K、V。这就是 "split" 操作的实现——通过计算不同的偏移来分离三部分。注意 Q 和 K 转为 float32（为了后续高精度计算），V 保持原始 bf16。

---

### 段3：cos/sin MRoPE 频率加载（搬运 + 状态更新）

```python
    cos_offsets = tl.arange(0, half_rope_dim)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets < mrope_section_t
        h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
        w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
            cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
        )
```

→ **状态更新（mask 计算）**：根据 `mrope_section` 的 `[t, h, w]` 分段信息，生成三个 mask，分别标记哪些频率位置属于 temporal(T)、height(H)、width(W)。

- **Interleaved 模式**：频率按 `[T, H, W, T, H, W, ...]` 交错排布，用 `%3` 选择
- **Non-interleaved 模式**：频率按 `[TTT...HHH...WWW]` 连续排布，用区间选择

```python
    t_cos_offset = cos_sin_ptr + (block_offset + index) * rope_dim
    h_cos_offset = t_cos_offset + num_tokens * rope_dim
    w_cos_offset = h_cos_offset + num_tokens * rope_dim

    t_sin_offset = cos_sin_ptr + (block_offset + index) * rope_dim + half_rope_dim
    h_sin_offset = t_sin_offset + num_tokens * rope_dim
    w_sin_offset = h_sin_offset + num_tokens * rope_dim

    t_cos_tensor = tl.load(t_cos_offset + cos_offsets, mask=t_mask, other=0)
    h_cos_tensor = tl.load(h_cos_offset + cos_offsets, mask=h_mask, other=0)
    w_cos_tensor = tl.load(w_cos_offset + cos_offsets, mask=w_mask, other=0)
    t_sin_tensor = tl.load(t_sin_offset + cos_offsets, mask=t_mask, other=0)
    h_sin_tensor = tl.load(h_sin_offset + cos_offsets, mask=h_mask, other=0)
    w_sin_tensor = tl.load(w_sin_offset + cos_offsets, mask=w_mask, other=0)
```

→ **搬运**：`cos_sin_ptr` 的内存布局是 `[3, num_tokens, rope_dim]`（3 对应 T/H/W），每行 `rope_dim` 中前 `half_rope_dim` 是 cos，后 `half_rope_dim` 是 sin。用 mask 选择性加载，未命中的位置填 0。

```python
    cos_tensor = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32).reshape(1, half_rope_dim)
    cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

    sin_tensor = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32).reshape(1, half_rope_dim)
    sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(1, rope_dim)
```

→ **算术计算**：把 T/H/W 三路 cos/sin 加在一起（因为 mask 互斥，所以 add 等价于 select-merge），然后 broadcast 成 `(1, rope_dim)` 供所有 heads 共用。broadcast 的技巧：`(half_rope_dim,)` → `(2, half_rope_dim)` → `(rope_dim,)`，这样 cos/sin 被复制了两份，与后续 `[-x2, x1]` 拼接的 rope 公式配合。

---

### 段4：Q-RMSNorm（算术计算）

```python
    squares = in_q_tensor * in_q_tensor
    variances = tl.sum(squares, axis=1) / head_size
    reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
    q_normalized = in_q_tensor * reciprocal_std
    q_normalized = q_normalized * q_rmsnorm_weight
    if has_bias:
        q_normalized = q_normalized + q_bias
```

→ **纯算术计算**：标准 RMSNorm。对每个 head 的 `head_size` 维度求均方根，然后乘归一化因子和 weight。

- `axis=1` 表示沿 `head_size` 维度 reduce
- 形状变化：`(num_q_heads, head_size)` → reduce → `(num_q_heads,)` → reshape → `(num_q_heads, 1)` 用于广播

---

### 段5：K-RMSNorm（算术计算）

```python
    squares = in_k_tensor * in_k_tensor
    variances = tl.sum(squares, axis=1) / head_size
    reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
    k_normalized = in_k_tensor * reciprocal_std
    k_normalized = k_normalized * k_rmsnorm_weight
    if has_bias:
        k_normalized = k_normalized + k_bias
```

→ 与 Q-RMSNorm 完全对称，只是 head 数量为 `num_kv_heads`。

---

### 段6：Q-MRoPE（算术计算）

```python
    x1 = tl.extract_slice(
        q_normalized, offsets=(0, 0),
        sizes=(num_q_heads, half_rope_dim), strides=(1, 1),
    )
    x2 = tl.extract_slice(
        q_normalized, offsets=(0, half_rope_dim),
        sizes=(num_q_heads, half_rope_dim), strides=(1, 1),
    )
```

→ 把 Q 的 rotary 部分切成前半 `x1` 和后半 `x2`。

```python
    cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
    cat_x = tl.insert_slice(cat_x, -x2, offsets=(0, 0), ...)
    cat_x = tl.insert_slice(cat_x, x1, offsets=(0, half_rope_dim), ...)
```

→ 构造 `[-x2, x1]`，这是旋转位置编码公式的一部分。

```python
    if IS_PARTIAL_ROPE:
        orig_qk = tl.extract_slice(
            q_normalized, offsets=(0, 0),
            sizes=(num_q_heads, rope_dim), ...
        )
    else:
        orig_qk = q_normalized
    roped_q = cat_x * sin_tensor + orig_qk * cos_tensor
```

→ **核心 RoPE 公式**：`roped = [-x2, x1] * sin + [x1, x2] * cos`，等价于标准旋转矩阵。`IS_PARTIAL_ROPE` 处理 `rope_dim < head_size` 的情况（只对前 `rope_dim` 维做旋转，剩余维度不变）。

---

### 段7：K-MRoPE（算术计算）

```python
    y1 = tl.extract_slice(k_normalized, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), ...)
    y2 = tl.extract_slice(k_normalized, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), ...)
    cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
    cat_y = tl.insert_slice(cat_y, -y2, offsets=(0, 0), ...)
    cat_y = tl.insert_slice(cat_y, y1, offsets=(0, half_rope_dim), ...)
    ...
    roped_k = cat_y * sin_tensor + orig_qk * cos_tensor
```

→ 与 Q-MRoPE 完全对称，只是 head 数量为 `num_kv_heads`。

---

### 段8：结果回写（搬运）

```python
    if IS_PARTIAL_ROPE:
        q_normalized = tl.insert_slice(q_normalized, roped_q, ...).to(tl.bfloat16)
        k_normalized = tl.insert_slice(k_normalized, roped_k, ...).to(tl.bfloat16)
    else:
        q_normalized = roped_q.to(tl.bfloat16)
        k_normalized = roped_k.to(tl.bfloat16)
```

→ **状态更新**：如果是 partial rope，把 roped 部分插回原始张量（保留未旋转的尾部维度），然后转回 bf16。

```python
    ## store ##
    out_q_offset = out_q_ptr + (block_offset + index) * q_size
    tl.store(out_q_offset + tl.arange(0, q_size), q_normalized.reshape(q_size))

    out_k_offset = out_k_ptr + (block_offset + index) * kv_size
    tl.store(out_k_offset + tl.arange(0, kv_size), k_normalized.reshape(kv_size))

    out_v_offset = out_v_ptr + (block_offset + index) * kv_size
    tl.store(out_v_offset + tl.arange(0, kv_size), in_v_tensor)
```

→ **搬运（Store）**：写出 Q（经过 RMSNorm + MRoPE）、K（经过 RMSNorm + MRoPE）、V（原样搬运）。

---

## 5. 总结速查表

| 代码段 | 行为类型 | 描述 |
|--------|----------|------|
| `block_idx` / `block_offset` / `loop_num` 计算 | 状态更新 | 确定本 core 处理哪些 token |
| 加载 `q/k_rmsnorm_weight`, `bias` | 搬运 | 循环外一次性加载 |
| 加载 Q/K/V from fused QKV | 搬运 | Split 操作（通过偏移实现） |
| mask 计算（interleaved/non-interleaved） | 状态更新 | 构建 T/H/W 选择 mask |
| 加载 cos/sin（T/H/W 三路 masked load） | 搬运 | 从 cos_sin_ptr 加载 |
| cos/sin 合并 + broadcast | 算术计算 | 三路 add + 形状变换 |
| Q-RMSNorm（mean-square → reciprocal_std → scale） | 算术计算 | 逐 head 归一化 |
| K-RMSNorm | 算术计算 | 同上 |
| Q-MRoPE（extract → [-x2,x1]\*sin + [x1,x2]\*cos） | 算术计算 | 旋转位置编码 |
| K-MRoPE | 算术计算 | 同上 |
| partial rope insert_slice + bf16 cast | 状态更新 | 回填未旋转维度 |
| store Q/K/V 到输出 | 搬运 | 写回全局内存 |
