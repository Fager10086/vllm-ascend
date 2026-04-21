---
name: triton-ascend-ops-optimizer
description: 昇腾（Ascend） NPU 上 Triton 算子深度性能优化技能（Skill），致力于实现用户要求的 Triton 算子性能提升。核心技术包括但不限于 Unified Buffer (UB) 容量规划、访存优化、多 Tokens 并行处理、MTE/Vector 流水并行、mask（掩码）优化等。当用户提及以下内容时，务必触发此技能（Skill）：昇腾（Ascend）NPU 上 Triton 算子性能优化。
---

# Triton 算子性能优化

## 目标与概述

确保在**昇腾（Ascend）NPU** 上，可靠地实现对 Triton 算子的深度性能优化目标。Grid 大小禁止与输入数据的形状相关联，**必须使用固定大小**，确保泛化性。

**优化工作应聚焦于 Triton 算子 Kernel 本身的性能优化**，禁止通过取巧方式或针对测试用例或测试代码逻辑进行 `Hacking` 来实现性能提升。同时，禁止修改测试用例；禁止以降低算子变量的数据精度为手段进行性能优化；禁止通过字典、缓存（cache）或任何形式的存储机制记录历史输入与输出，以避免利用相同输入复用结果来提升性能。

**核心目标**：将指定的 Triton 算子性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好，追求极致性能。

**工作模式**：单算子优化模式。禁止以降低算子变量的数据精度为手段进行性能优化；禁止通过字典、缓存（cache）或任何形式的存储机制记录历史输入与输出，以避免利用相同输入复用结果来提升性能；**禁止使用入图方式**来提升性能（模型侧会通过整网入图或 Piecewise 方式进行图优化，这里只关注单算子的独立优化）。

**工作原则**：
- **正确性优先**：每次修改后都必须进行正确性验证和性能测量，注意：精度与功能正确性验证全部通过后，才能进行性能测量。
- **目标导向**：性能提升未达到目标前，持续优化，不停止迭代。
- **迭代优化**：可以反复修改、测试、迭代，直至达成目标。在调测过程中，第一次修改算子源代码时务必备份，之后还要注意保存有性能提升的代码版本，以便需要时恢复；准确记录每次修改算子代码后的性能测试结果，并概括相应的修改要点，以便从经验中提升。
- **精准修改**：追求“手术级”的精准修改，避免引入新问题。

## 工作流程

0. 在昇腾 NPU 环境中，执行以下命令完成**环境配置**：`export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/driver:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64:$LD_LIBRARY_PATH && source /usr/local/Ascend/ascend-toolkit/set_env.sh`

1. **基线性能验证**：首先，深入分析算子的输入参数、数据类型、Shape 范围、功能逻辑、计算流程及输出结果；然后，运行精度与功能正确性验证，验证算子的正确性和精度；最后，执行性能测试，输出中的 Task Duration: 即为当前算子的耗时，将其记录为基线性能数据。注意：精度与功能正确性验证，以及性能测试所用的**脚本和运行命令**，**严格按照用户提供的执行**；如果用户未提供，必须主动提示用户输入。每次运行新的测试之前，必须预先清除缓存，且只能使用如下命令清除指定内容：
```bash
rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/
```

2. **深度性能优化**：根据基线分析结果，对 `<op_name>.py` 算子进行针对性优化，确保性能提升至少 **x 倍**（用户要求的性能提升），在满足要求的基础上，性能越高越好，追求极致性能。每次对算子源码进行完整修改后，需按顺序依次执行以下测试：
    - 精度与功能正确性验证。注意：精度与功能正确性验证全部通过后，才能进行性能测量；如果没通过，继续修改优化。
    - 性能测试（与基线对比）

3. **迭代调优过程中按需参考的文档**：`references/hardware_constraints.md`、`references/troubleshooting.md`

## 性能优化参考

1 - 必须遵循的 Grid 大小的计算方式

**关键优化点**：Grid 大小禁止与输入数据的形状相关联，**必须使用固定大小**：对于纯 Vector 算子，只能使用 **grid = (get_vectorcore_num(), )**，禁止 min、max 操作；而对于 Kernel 内包含 `tl.dot` 的 Triton 算子，只能使用 **grid = (get_vectorcore_num() // 2, )**，禁止 min、max 操作。**示意如下**：
```python
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

# must first be imported and initialized: 
init_device_properties_triton()
print(f"NPU Vector 核心数: {get_vectorcore_num()}")

# pure vector operator
grid = (get_vectorcore_num(), )
vector_kernel[grid]()

# cube operator: has tl.dot
grid = (get_vectorcore_num() // 2, )
cube_kernel[grid]()
```

优先使用 **1D Grid**；适配 NPU 的 2D Grid 写法也会合并为 1D。

2 - Ascend NPU 在架构上访存能力相对较弱，而计算能力较强，因此在设计时需要尽可能减少频繁的内存访问。**关键优化点是 Kernel 内批量处理多个 Tokens**，必须优先思考和调试，从而避免因逐个加载而产生的大量访存开销；由于受限于硬件内存容量，无法一次性处理完整的序列，仍需采用**分批次**计算。

一次循环里能处理的**最大 Token 数 N**，由 Kernel 内 **UB 可用容量**决定。**设：**
- 单 Kernel 内 UB 总容量为 **192 KB**
- 为留安全余量，仅使用 170 KB 的 **50%**（为确保启用 Double Buffering），即 **85 KB**
- 单个 Token 在 Kernel 内同时占用的 UB 空间峰值为 $S_{\text{token}}$（包含所有 load、中间变量的内存占用）

则需满足：$N \times S_{\text{token}} \le 85 \times 1024$；因此：$N \le \frac{85 \times 1024}{S_{\text{token}}}$
**示例：** 若 Kernel 只做一次 load 和一次 store，加载形状为 `(batch_size, hidden_size)` 的 **BF16** Tensor（每元素 2 Bytes），且不引入其他中间变量，则单个 Token 的 UB 占用峰值为：
$$
S_{\text{token}} = \text{hidden\_size} \times 2
$$

代入约束：
$$
N \times \text{hidden\_size} \times 2 \le 85 \times 1024
$$
据此计算优化后的循环次数 `reduced_loops` 以及单次循环可处理的**最大 Token 数 N**。单次循环应尽可能占满 UB，但需控制在 UB 大小的约一半以内，以利用 `Double Buffering` 机制实现流水并行。计算最大处理量时应使用**整数除法**（//）而非 `tl.cdiv`，否则易引发 UB 溢出问题。

3 - 在昇腾（Ascend）NPU 上使用 Triton 进行算子性能优化时，必须避免在算子调用过程中触发重编译。Triton 的单次编译耗时在秒级，重编译会显著影响计算性能。**通过以下方式避免重编译**：

在 Kernel 函数定义前，对于会变化的标量参数，可使用 `@triton.jit(do_not_specialize=["T", "B",  ...])` 进行装饰，其中 `T`、`B` 为变量名。这是因为底层编译器会自动对这样的变量做特定优化；在 Kernel 函数定义中，声明为 `constexpr` 的常量必须保持值不变，不应作为可变值使用；反之，可变值也不应声明为 `constexpr` 类型。因为一旦常量值发生变化，Triton 便会触发重编译。

4 - 掩码（mask）与尾块处理：每次核函数加载和存储 tensor 时都需使用 `mask` 来处理不需要计算的尾块。经过 mask 处理后，每个核上的 tensor 形状保持一致。

5 - 减少 kernel 内 Scalar 运算：将与 pid 或循环变量无关的计算移至辅助函数或循环外部；能合并的计算尽量合并，减少冗余操作。

6 - 对于 `index_select` 这类涉及非连续地址访问的操作，只能通过循环逐行读取数据；否则会引入大量标量（Scalar）计算（计算二维 mask），严重影响性能。

7 - 加载与计算交织：当需要多次从同一全局内存地址加载数据并进行计算（如加法）时，需采用 “加载一次、计算一次” 的方式，而不是全部加载完再统一计算。后者会导致计算流水线等待所有 tensor 加载完成，效率较低；前者可有效隐藏访存延迟。

8 - 若存在多个写入流，建议边计算边写入数据。写入流通常不会相互冲突，计算完提前写入可以增大并行的可能，提升整体性能。

9 - 使用 `tl.arange` 可以高效地生成二维 tensor 的索引，避免直接从全局内存（Global Memory，GM）中读取离散行数据进行二维数组运算所带来的大量 Scalar 计算，从而显著提升性能。

10 - 尽量避免使用 `tl.where`，因其主要适用于离散数据处理，性能较差。

11 - 避免对同一 tensor 多次调用 `insert_slice`，以提升执行效率。

12 - 执行规约操作时，优先选择最大的维度进行规约，有助于提升性能。

13 - **kernel 入参**：对于同一模型调用期间保持不变的参数，推荐声明为 `tl.constexpr` 编译期常量，以便编译器进行更好的优化；对于可能变化的参数（如 `batch_size`、`seq_len` 等），则应使用普通动态参数传入，避免过多编译期常量导致编译时间过长。

14 - **消除冗余 reshape/transpose**：当 kernel 内部对加载的数据执行 reshape → transpose → 计算 → transpose → reshape 流程时，应分析计算操作（如 `tl.cumsum`、`tl.sum` 等规约操作）是否支持通过 `axis` 参数直接在目标维度上执行。若支持，则可在 reshape 后直接选择正确的 axis 进行计算，从而消除 transpose 操作。transpose 涉及 UB 内数据搬移，消耗 Vector 周期且增加 UB 峰值占用。**示例**：
```python
# 优化前：reshape → transpose → cumsum(axis=0) → transpose → reshape
b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
b_s = tl.trans(b_s, (1, 0, 2))         # transpose 到 (CHUNK_SIZE, N_CHUNKS, H)
b_o = tl.cumsum(b_s, axis=0)           # 在 axis=0 (CHUNK_SIZE) 上 cumsum
b_o = tl.trans(b_o, (1, 0, 2))         # transpose 回 (N_CHUNKS, CHUNK_SIZE, H)
b_o = tl.reshape(b_o, (BLOCK_T, H))

# 优化后：reshape → cumsum(axis=1) → reshape，零 transpose
b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
b_o = tl.cumsum(b_s, axis=1)           # 直接在 axis=1 (CHUNK_SIZE) 上 cumsum
b_o = tl.reshape(b_o, (BLOCK_T, H))
```

15 - **批量加载优先于逐块加载**：昇腾 NPU 访存能力相对较弱，应尽量一次性加载尽可能多的数据到 UB，在 UB 内完成全部计算后再写回。**禁止将大块加载拆分为逐小块循环加载**（如逐 chunk 加载），因为这会成倍增加访存次数，反而降低性能。正确做法是一次加载整个 BLOCK_T 大小的数据块（包含多个 chunk），在 UB 内通过 reshape 划分后进行批量计算。BLOCK_T 的大小应根据 UB 容量公式计算，尽量填满 85 KB 可用空间。

16 - **2D Grid 到 1D Grid + stride loop 转换**：当原始算子使用与输入形状相关的多维 Grid（如 `grid = (num_blocks, B)`）时，需转换为固定 1D Grid + kernel 内 stride loop。转换方法：将多维工作项线性化为 `total_blocks = B * blocks_per_batch`（非 varlen）或 `total_blocks = len(chunk_indices)`（varlen），在 kernel 内通过 `for work_idx in range(pid, total_blocks, num_programs)` 循环，每次迭代解码出原始的多维索引。**示意如下**：
```python
# Host 侧
grid = (get_vectorcore_num(), )
kernel[grid](..., total_blocks=B * blocks_per_batch, blocks_per_batch=blocks_per_batch)

# Kernel 侧
pid = tl.program_id(0)
num_programs = tl.num_programs(0)
for work_idx in range(pid, total_blocks, num_programs):
    i_b = work_idx // blocks_per_batch
    i_block = work_idx % blocks_per_batch
    # 处理该 block
```

17 - **make_block_ptr + boundary_check 替换为显式指针 + mask**：`tl.make_block_ptr` 配合 `boundary_check` 在昇腾 NPU 上可能内部生成掩码逻辑，影响 MTE 并行。推荐改用 `tl.arange` 构建显式二维指针偏移，配合手动 mask 控制边界。这种方式对 MTE 流水更友好，且 mask 逻辑更透明可控。**示意如下**：
```python
# 优化前
ptr_s = tl.make_block_ptr(base, (T, H), (H, 1), (offset, 0), (BLOCK_T, H), (1, 0))
b_s = tl.load(ptr_s, boundary_check=(0,))

# 优化后
row_offsets = tl.arange(0, BLOCK_T)
col_offsets = tl.arange(0, H)
offsets = (offset + row_offsets[:, None]) * H + col_offsets[None, :]
mask = (offset + row_offsets[:, None]) < T
b_s = tl.load(s_ptr + offsets, mask=mask)
```

18 - **BLOCK_T 动态适配 UB 容量**：BLOCK_T（单次处理的 token 数）应根据 kernel 的 UB 峰值占用动态计算，使其尽量填满 85 KB 可用空间，而非使用固定的经验值。计算公式：
```python
# S_peak_per_token = 每个 token 的 UB 峰值占用（bytes），包含 load + 中间变量 + store
# 例如：load (H * dtype_bytes) + output (H * dtype_bytes) = 2 * H * dtype_bytes
max_tokens = (85 * 1024) // S_peak_per_token
BLOCK_T = prev_power_of_2(max_tokens)  # 向下取 2 的幂
BLOCK_T = max(BLOCK_T, CHUNK_SIZE)     # 至少一个 chunk
```
BLOCK_T 越大，单次 load/store 搬运的数据越多，循环次数越少，访存效率越高。但必须保证不超过 UB 容量限制。

19 - **维度参数必须用 tl.constexpr + 编译期分支控制 load/store 数量**：当 kernel 中 `tl.load`/`tl.store` 的次数依赖某个"维度参数"（如 conv kernel width、sliding window size、MLP 层数等），**禁止硬编码为固定值**。必须将该参数声明为 `tl.constexpr`，高频值用 `if PARAM == 2: ... elif PARAM == 3: ...` 手动展开快速路径，`else` 分支**必须**用 `tl.static_range(0, PARAM)` 循环实现真正的通用路径（因 PARAM 是 constexpr，tl.static_range 会在编译期展开为精确数量的 load/store）。**禁止在 else 分支中硬编码另一个固定数量**（如 else 分支写死 5 个 load），否则当 PARAM=6,7,8,... 时同样越界。越界 `tl.load` 在 Ascend NPU 上不报 segfault，返回未定义值（nan/极大值），导致静默精度错误。详见 E8。

20 - **BLOCK_N 维度必须内存连续（stride=1）**：kernel 中以 `tl.arange(0, BLOCK_N)` 索引的主要数据维度，其在传入 tensor 中的 stride 必须为 1（内存连续）。在 Ascend NPU 上 stride > 1 的访存模式触发 gather/scatter 指令，实测导致 60-85x 性能退化（17us → 1440us）。如果原始 tensor layout 不满足，必须在 wrapper 层预先 transpose + contiguous。详见 E9。

21 - **transpose().contiguous() 创建独立副本时必须 copy 回**：当对 tensor 执行 `t = tensor.transpose(a, b).contiguous()` 且原始 tensor 不满足转置后的 contiguous 条件时，`t` 是独立副本。kernel 修改 `t` 后，原始 tensor 不会被更新。必须在 kernel 执行后显式执行 `tensor.transpose(a, b).copy_(t)` 写回。遗漏此步骤在单步测试中不会暴露（output 正确），仅在多步推理中暴露（state 累积错误）。详见 E13。

## 需遵循的规则和约束

### 单算子模式

单个算子只关注单算子模式下的基础功能和性能，**禁止使用入图方式提升性能**，因为模型侧会以整网入图或分段（Piecewise）方式对多算子进行图优化。

### tl.load 与 mask 使用要求

- 尽量合并相同 load、计算和 store 操作。例如，利用 `tl.load` 与 mask 参数，可一次性加载多个 Tokens 的数据，避免多次独立的 load。减少此类冗余操作有助于提升性能。
- 避免在 `tl.load` 中使用 other 参数，因为其内部会触发 `tl.where`，导致 load 后无法与其他 load 并行。
- 推荐的替代方案：先执行无掩码的 `tl.load`，再通过 `tl.where` 与 mask 组合实现掩码逻辑；当访问内存规则连续时，用 `tl.insert_slice` 代替。

### 分支与编译约束

在 kernel 内部的 `if-else` 分支中，同名变量的 Shape 必须一致，否则会导致编译错误。

### 数据搬运注意事项

- 保证 tl.load 加载的是连续的多行数据；若数据分布离散，需逐行加载。
- 传递给 Triton 算子的 tensor 必须是内存连续的，必要时可通过 `.contiguous()` 方法确保。
- 避免复用 `tl.load` 和 `tl.store` 的变量名，使用不同变量名可提高代码可读性，并减少数据流错误的风险。

## 执行要求

在 Ascend NPU 算子优化中，需自主完成从代码修改（追求“手术级”精准）、测试验证到性能对比的全流程闭环，确保性能提升达到用户要求的 **x 倍** 以上。通过迭代优化，在不引入错误的前提下，持续改进直至达标。

## 结果报告

性能优化目标真正达成后，需准确输出标准化报告：
```
## 优化结果报告

### 算子信息

- 算子名称：<op_name>
- 源文件：<file_path>

### 性能对比

| 基线耗时 (us) | 优化后耗时 (us) | 加速比 |
|-------------|---------------|-------|
| ... | ... | ...x |

### 优化技术清单

1. [已应用] 多个 Token 并行处理：N = ...
2. [已应用] 消除带 other 的 tl.load
3. ...

### 关键修改说明

- 修改点 1：...
- 修改点 2：...
```

## msprof 实测经验（cumsum 算子优化迭代总结）

以下经验均来自 chunk_local_cumsum_scalar_kernel 在昇腾 910B 上的 msprof 实测数据，对其他 Vector 算子同样适用。

### E1 - 性能度量：端到端耗时与 device 侧耗时需同时达标

优化目标是**端到端耗时和 device 侧耗时都缩减**，不能只看其中一个。端到端用 `time.perf_counter()`（含 host 开销、NPU 排队延迟），device 侧用 msprof CSV 中的 `Task Duration(us)` 列。两者不一致时说明瓶颈在 host 侧或调度侧，需分别分析。

端到端耗时不劣化但 device 侧耗时增加 → 说明 host 侧开销减少掩盖了 device 侧退化，仍需修复 device 侧问题。device 侧耗时降低但端到端劣化 → 说明引入了额外 host 侧开销（如辅助算子的 device 侧 Sub+Add+FloorDiv）。

测量方法：
```bash
rm -rf ~/.triton/ __pycache__ .pytest_cache extra-info/
msprof --application='python test_perf.py' --output=./result
# 解析 mindstudio_profiler_output/op_summary_*.csv
# 跳过前 2 次 warmup，取后续调用的 Task Duration 平均值
```

CSV 解析必须用 `csv.reader`，不能用 `line.split(',')` —— Input Shapes 列内含逗号和引号。

### E2 - 消除冗余 transpose 是高收益优化

实测数据：消除 transpose 使 Task Duration 从 4.231us 降至 3.806us（-10.1%）。

当 kernel 内存在 `reshape → tl.trans → cumsum/sum(axis=0) → tl.trans → reshape` 模式时，应分析规约操作是否支持通过 axis 参数直接在目标维度执行。若支持，消除全部 transpose：
```python
# 优化前：4.231us
b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
b_s = tl.trans(b_s, (1, 0, 2))
b_o = tl.cumsum(b_s, axis=0)
b_o = tl.trans(b_o, (1, 0, 2))
b_o = tl.reshape(b_o, (BLOCK_T, H))

# 优化后：3.806us
b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
b_o = tl.cumsum(b_s, axis=1)
b_o = tl.reshape(b_o, (BLOCK_T, H))
```

transpose 在 UB 内产生数据搬移，消耗 Vector 周期且增加 UB 峰值占用。

### E3 - 固定 Grid 的调度开销与核心数强正相关

msprof 实测数据（同一 kernel，不同 Block Dim）：

| Block Dim | Task Duration(us) | 调度开销(us) = task_dur - aiv_time |
|-----------|-------------------|-----------------------------------|
| 7         | 4.439             | 0.871                             |
| 14        | 3.806             | 1.227                             |
| 40        | 5.672             | 3.318                             |

**调度开销随核心数近似线性增长**。40 核调度开销 3.3us 是 14 核的 2.7 倍。Task Duration 取所有核心中最慢核心的完成时间。

**关键结论**：当算子数据量小（total_blocks << 核心数）时，固定 40 核 Grid 的调度开销可能超过计算时间本身，导致 Task Duration 反而增加。优化应在满足固定 Grid 规范的前提下，尽量减少 kernel 内部的 aiv_time 和 host 侧辅助开销，使端到端和 device 侧双重达标。

### E4 - BLOCK_T 调整无法弥补调度开销

BLOCK_T 只影响 aiv_time（核内计算/访存），不影响调度开销。调度开销在核心开始执行第一条指令之前就已产生。

实测数据（均为 40 核固定 Grid）：

| BLOCK_T | total_blocks | Task Duration(us) | aiv_scalar(us) |
|---------|-------------|-------------------|----------------|
| 128     | 55          | 6.643             | 2.489          |
| 512     | 14          | 5.672             | 1.747          |

- 减小 BLOCK_T（512→128）：total_blocks 增至 55，每核需 2 次 stride loop 迭代，aiv_scalar 线性增加
- 增大 BLOCK_T（512→1024）：total_blocks 降至 7，仅 7 核有工作，其余 33 核空转

**每次 stride loop 迭代有固定 Scalar 成本**（循环条件判断、地址计算等），增加迭代次数线性增加 Scalar 时间。

### E5 - make_block_ptr vs 显式指针对 Task Duration 影响可忽略

实测对比（其他条件完全相同，40核，BLOCK_T=512）：

| 方式 | Task Duration(us) |
|------|-------------------|
| make_block_ptr + boundary_check | 5.672 |
| tl.arange 显式指针 + mask        | 5.615 |

差异仅 ~1%，在测量噪声范围内。**对于此类小规模 kernel，make_block_ptr 与显式指针的选择不是性能瓶颈**。

### E6 - 小数据量算子的优化瓶颈分析

当总工作量极小时（如 T×H = 6986×8 ≈ 218KB），Task Duration 的组成：
- 调度开销占 ~58%（40 核时 3.3us / 5.67us）
- Scalar 占 ~31%
- Vector 计算仅占 ~7%

此时计算优化的收益上限很低。优化应集中在：
1. 减少 Scalar 开销（减少 tl.load 次数、简化地址计算）
2. 减少不必要的数据搬移（消除 transpose）
3. 减少 host 侧辅助算子的开销

对于大数据量算子（T=100K+），调度开销可被计算时间摊薄，固定 40 核 Grid 的并行收益才能体现。

### E7 - 2D Grid 到 1D Grid 转换的代价

2D Grid `(num_blocks, B)` 转换为 1D Grid `(get_vectorcore_num(), )` + stride loop 后：
- 新增 stride loop 的 Scalar 开销（循环控制、work_idx 解码）
- 核心数从 `num_blocks` 变为固定 40，当 `num_blocks < 40` 时大量核心空转

实测代价：Task Duration 从 3.806us（14核）增至 5.672us（40核），增加 49%。其中调度开销增加 2.1us，aiv_time 反而降低 0.2us。

**结论**：对于 total_blocks << 40 的小规模算子，2D→1D 转换的 device 侧 Task Duration 代价显著。需确保 kernel 内 aiv_time 优化 + host 侧辅助开销缩减能使端到端总耗时也同步降低。

## causal_conv1d 算子优化实测经验

以下经验来自 `_causal_conv1d_update_kernel_optimized` 在昇腾 910B4（40 vector cores）上的 msprof 实测数据。涵盖精度问题定位、strided 访存性能陷阱、Wrapper 层 transpose/copy 开销、`tl.constexpr` 编译期分支等。

### E8 - Kernel 硬编码参数维度导致越界访存精度错误

**问题现象**：kernel 在特定输入参数组合下输出 `nan` / `inf` / 极大值（如 `3.86e+19`），但在其他参数组合下正常。state 更新正确，仅 output 异常。

**实测场景**：`batch=64, dim=4096, width=3, seqlen=3, bias=True, silu=True, dtype=bf16` 下 output 包含 nan；`width=4` 下同配置正常。

**根因**：kernel 内部硬编码了 `width=4` 的逻辑——始终加载 4 个权重（w0-w3）和 3 个历史状态（h0-h2）。当实际 `width=3`（`state_len=2`）时：
- `tl.load(w_ptr + 3 * stride_w_width + c_idx)` 越界读取了权重 tensor 之外的内存
- `tl.load(cs_base + 2 * stride_cs_state)` 越界读取了 conv_state 之外的内存
- 越界数据参与 MAC 计算，产生 nan/inf 输出
- state 写回只写 `state_len` 行，恰好未越界，所以 state_match 通过

**定位方法**：
1. 对比不同 width 值的测试结果，发现 width=4 通过、width≠4 失败
2. 检查 kernel 内 weight 和 history state 的 load 数量，发现硬编码为固定 4/3 个
3. 计算 `w_ptr + 3 * stride_w_width` 在 width=3 时的实际偏移，确认越界

**修复方案**：将 `width` 作为 `WIDTH: tl.constexpr` 传入 kernel。高频 WIDTH 值（2,3,4）用手动展开的专用分支（性能最优），`else` 分支用 `tl.static_range(0, WIDTH - 1)` / `tl.static_range(0, WIDTH)` 循环实现**真正的通用路径**，支持任意 WIDTH。由于 WIDTH 是 constexpr，`tl.static_range` 在编译期展开为精确数量的指令，无运行时循环开销。**禁止在 else 分支中再硬编码另一个固定数量**（如写死 5 个 load），否则 WIDTH=6,7,8,32 时同样越界。

```python
# 错误做法 1：全局硬编码 width=4
h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
h1 = tl.load(cs_base + 1 * stride_cs_state, mask=c_mask)
h2 = tl.load(cs_base + 2 * stride_cs_state, mask=c_mask)  # width=3 时越界
w3 = tl.load(w_ptr + 3 * stride_w_width + c_idx, mask=c_mask)  # width=3 时越界

# 错误做法 2：有编译期分支但 else 硬编码 width=5
if WIDTH == 2: ...
elif WIDTH == 3: ...
elif WIDTH == 4: ...
else:
    # 硬编码 5 个 load → WIDTH=6,7,8,32 时同样越界！
    h0 = ...; h1 = ...; h2 = ...; h3 = ...
    w0 = ...; w1 = ...; w2 = ...; w3 = ...; w4 = ...

# 正确做法：高频路径手动展开 + else 用 tl.static_range 通用路径
if WIDTH == 2:
    # 手动展开：1 个 history state, 2 个 weight
    ...
elif WIDTH == 3:
    # 手动展开：2 个 history states, 3 个 weights
    ...
elif WIDTH == 4:
    # 手动展开：3 个 history states, 4 个 weights
    ...
else:
    # 通用路径：任意 WIDTH >= 5，tl.static_range 编译期展开
    for t in tl.static_range(0, seqlen):
        x_val = tl.load(x_base + t * stride_x_token, mask=c_mask)
        acc = b_val
        for k in tl.static_range(0, WIDTH - 1):  # 精确 WIDTH-1 个 history loads
            h_k = tl.load(cs_base + k * stride_cs_state, mask=c_mask)
            w_k = tl.load(w_ptr + k * stride_w_width + c_idx, mask=c_mask)
            acc += h_k.to(tl.float32) * w_k.to(tl.float32)
        w_last = tl.load(w_ptr + (WIDTH-1) * stride_w_width + c_idx, mask=c_mask)
        acc += x_val.to(tl.float32) * w_last.to(tl.float32)
        # 滑窗更新 history state（直接在 GM 上操作）
        for k in tl.static_range(0, WIDTH - 2):
            h_next = tl.load(cs_base + (k+1) * stride_cs_state, mask=c_mask)
            tl.store(cs_base + k * stride_cs_state, h_next, mask=c_mask)
        tl.store(cs_base + (WIDTH-2) * stride_cs_state, x_val, mask=c_mask)
        if SILU_ACTIVATION:
            acc = acc / (1.0 + tl.exp(-acc))
        tl.store(o_base + t * stride_o_token, acc, mask=c_mask)
```

**通用教训**：
- 当 kernel 的 load/store 数量依赖于某个"维度参数"（如 conv width、attention head 数、MLP 层数）时，该参数**必须**作为 `tl.constexpr` 传入并用编译期分支控制，**禁止硬编码**。
- **else fallback 分支同样禁止硬编码固定数量**——必须用 `tl.static_range(0, PARAM)` 实现真正通用的循环路径。这是本次优化中犯过两次的错误：第一次全局硬编码 width=4，修复后 else 又硬编码 width=5，导致 width=6,7,8,32 仍然越界。
- 越界 `tl.load` 在 Ascend NPU 上不会报 segfault，而是返回未定义值（通常是极大值或 nan），导致计算结果静默错误。
- 精度测试必须覆盖所有可能的维度参数值（包括极端值如 width=32），不能只测试最常用的值。

### E9 - Ascend NPU strided 访存性能灾难：stride_dim>1 导致 60-85x 减速

**实测数据**（causal_conv1d_update, batch=32, dim=2048, width=4, BLOCK_N=256, 40 核）：

| 访存模式 | stride_dim | Task Duration(us) | 相对性能 |
|---------|-----------|-------------------|---------|
| dim-contiguous（transpose 后） | 1 | 17.30 | 1.0x（基线） |
| strided（直接访问原始 layout） | 3 | 1440 | 0.012x（83x 减速） |

**根因**：conv_state 原始布局 `(N, dim, state_len)` 的 stride 为 `(dim*state_len, state_len, 1)`，即 `stride_dim = state_len = 3`。kernel 以 `c_idx` 索引 dim 维度时，每个元素间隔 3 个位置。NPU vector 单元的 gather/scatter 指令处理这种非连续访问效率极低：
- 无法使用 burst 模式的连续 DMA 搬运
- 每个元素需单独寻址，MTE 流水完全打断
- 编译器可能对 stride>1 的 pattern 生成 scalar 循环而非 vector 指令

**验证方法**：
1. 用相同 kernel 分别传入 `stride_dim=1`（contiguous）和 `stride_dim=3`（strided）的 tensor
2. msprof 对比 Task Duration，确认差异来自访存而非计算

**结论**：在 Ascend NPU 上，**kernel 的主要数据访问维度（BLOCK_N 维度）必须是内存连续的（stride=1）**。如果原始 tensor 布局不满足，宁可在 wrapper 层做一次 host 侧 transpose+contiguous（~13us），也不要在 kernel 内做 strided 访问（~1440us）。这个代价差异是 100x 量级的。

**推论**：
- 尝试在 kernel 内用双缓冲（读 contiguous、写 strided）方式避免 wrapper transpose 也会失败——写侧的 strided store 同样导致 UB overflow 或性能灾难
- 尝试用 `c_offs * stride_dim` 构建显式 strided 指针、用 2D offset 矩阵、用 `tl.arange` 生成离散地址——都无法绕过底层硬件的 gather/scatter 限制
- 这是 Ascend NPU 的**硬件级限制**，不是编译器或 Triton 前端的问题

### E10 - Wrapper 层 transpose/contiguous/copy 开销分析与权衡

**实测数据**（msprof op_summary, batch=32, dim=2048, width=4, fp16）：

| 操作 | 次数/每调用 | 平均耗时(us) | 总耗时(us)/每调用 | 占比 |
|------|-----------|------------|----------------|-----|
| Kernel | 1 | 17.30 | 17.30 | 27.5% |
| Transpose | 3 | 12.91 | 38.74 | 61.7% |
| Copy（state 写回） | 1 | 6.79 | 6.79 | 10.8% |
| **总计** | - | - | **62.83** | 100% |

**3 个 Transpose 的来源**：
1. `weight.transpose(0,1).contiguous()` — 将 `(dim, width)` → `(width, dim)` 使 dim 维度连续
2. `conv_state.transpose(1,2).contiguous()` — 将 `(N, dim, state_len)` → `(N, state_len, dim)` 使 dim 维度连续
3. `conv_state.transpose(1,2).copy_(conv_state_t)` — 将更新后的 state 写回原始 layout

**原始基线的 Bug**：baseline 代码中 `conv_state_t = conv_state.transpose(1,2).contiguous()` 创建了一个**独立副本**。kernel 修改 `conv_state_t` 后，原始 `conv_state` 未被更新。因此 baseline 的 Transpose 只有 2 次（26.26us）且无 Copy，但**结果是错误的**——conv_state 状态丢失。

修复后增加了第 3 个 Transpose（copy-back），是正确性所必需的代价。

**尝试消除 Transpose 的失败路径**：
1. **直接 strided 访问**：kernel 直接读写原始 layout → 83x 减速（E9）
2. **双缓冲**：kernel 读 contiguous、写 strided → BLOCK_N=512 时 UB overflow（1835008 > 1572864 bits）；BLOCK_N=256 时仍 1437us
3. **显式 strided 指针**：用 `c_offs * state_len` 构建地址 → 编译器仍生成 gather/scatter，性能与方案 1 相同
4. **仅写回修改的行**：sparse write-back kernel → strided store 同样极慢

**结论**：在 Ascend NPU 上，当 kernel 的主要访问维度在原始 tensor 中不连续时，wrapper 层的 transpose+contiguous 是不可避免的成本。优化方向应转向：
- 模型层面：在模型初始化时预转置 weight（weight 不变，可一次性转置）
- 框架层面：修改 conv_state 的存储 layout 为 `(N, state_len, dim)` 避免运行时 transpose

### E11 - tl.constexpr 编译期分支的性能影响

**实测数据**（添加 `WIDTH: tl.constexpr` 前后对比，width=4, batch=32, dim=2048）：

| 版本 | Kernel Duration(us) | 说明 |
|------|--------------------:|------|
| 无 WIDTH 分支（硬编码 width=4） | 15.02 | 仅支持 width=4 |
| 有 WIDTH 编译期分支 | 17.30 | 支持 width=2,3,4,5+ |

增加约 2.3us（+15%）。虽然编译器应消除死分支，但 `tl.constexpr` 参数增加会影响编译器优化空间。

**使用 `tl.constexpr` 分支的规则**：
- 当 load/store 数量依赖某参数时，该参数**必须**为 `tl.constexpr`，否则产生越界访存（E8）
- 每个 `tl.constexpr` 的不同值会触发独立编译，首次使用有编译开销（秒级）
- 在同一推理服务中，`width` 通常固定（取决于模型架构），所以编译只发生一次
- 分支内同名变量的 shape 必须一致（Triton 编译约束）；不同分支可用不同变量名

### E12 - Triton kernel 中 tl.slice 不支持及替代方案

**问题**：尝试用 `tensor[:, 0:state_len]` 或 `tensor[:, k:k+1]` 在 kernel 内做 slice 操作时，Triton 报错 `ValueError('unsupported tensor index: slice(...)')`。

**原因**：Triton 的 tensor 索引不支持 Python slice 语法。

**替代方案**：
```python
# 错误：tl.slice 不支持
h_row = conv_state_2d[:, k]  # ValueError

# 正确：用显式 offset + stride 逐元素访问
cs_base = conv_state_ptr + cs_line * stride_cs_batch + c_idx * stride_cs_dim
h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
h1 = tl.load(cs_base + 1 * stride_cs_state, mask=c_mask)
```

### E13 - conv_state 写回的正确性陷阱

**Bug 模式**：
```python
# 创建 contiguous 副本用于 kernel
conv_state_t = conv_state.transpose(1, 2).contiguous()
# ... kernel 修改 conv_state_t ...
# BUG：conv_state 未被更新！conv_state_t 是独立副本
```

**正确写法**：
```python
conv_state_t = conv_state.transpose(1, 2).contiguous()
# ... kernel 修改 conv_state_t ...
# 写回原始 tensor
conv_state.transpose(1, 2).copy_(conv_state_t)
```

**关键认知**：`tensor.transpose().contiguous()` 在 stride 不满足 contiguous 条件时会创建**新的独立 tensor**。后续对新 tensor 的修改不会反映到原始 tensor。必须显式 copy 回去。

这个 bug 在性能测试中不会暴露（单次调用 output 正确），只在**多步推理**（decode 阶段逐 token 生成）中暴露——因为 conv_state 的历史状态未被正确更新，后续 token 的计算基于错误的历史状态。

### E14 - 精度测试设计原则（从 causal_conv1d 精度 bug 总结）

**必须覆盖的测试维度**：
1. **所有可能的"结构参数"值**：如 conv width（2,3,4,5）、attention head 数、group 数等决定 kernel 内 load/store 数量的参数
2. **多种 dtype**：fp16 和 bf16 的精度行为不同，越界访存在不同 dtype 下产生不同的错误值
3. **有/无 padding**：padding 引入 `PAD_SLOT_ID` 路径，需单独验证
4. **seqlen > 1**：多 token 场景下历史状态的滑窗更新逻辑更容易出错
5. **conv_state 的 in-place 更新**：对比 kernel 执行后的 conv_state 与 reference 实现的 conv_state

**Reference 实现的正确构造**：
```python
# 必须传入 initial_states，否则 reference 从零状态开始，与 kernel 行为不一致
for i in range(batch):
    out_i, final_state = causal_conv1d_ref(
        x[i:i+1].transpose(1,2), weight, bias,
        activation=act,
        initial_states=conv_state[i:i+1],  # 关键：使用相同的初始状态
        return_final_states=True
    )
```

**测试判断阈值**：bf16 精度较低，`atol=1e-2, rtol=1e-2` 是合理阈值。fp16 可用 `atol=5e-3, rtol=5e-3`。

### E15 - 性能测试与正确性测试的 conv_state layout 差异

**实测发现**：正确性测试和性能测试中 conv_state 的创建方式不同，导致 stride 不同：

```python
# 正确性测试：transpose 后的 view（dim-contiguous）
cs = torch.randn(N, state_len, dim).transpose(1, 2)
# strides = (dim*state_len, 1, dim) → stride_dim=1（已经 dim-contiguous）

# 性能测试：直接创建（state-contiguous）
cs = torch.randn(N, dim, state_len)
# strides = (dim*state_len, state_len, 1) → stride_dim=state_len（需要 transpose）
```

**影响**：wrapper 中的 `conv_state.transpose(1,2).contiguous()` 在两种情况下行为不同：
- 第一种：transpose 后恢复为 `(N, state_len, dim)` 原始连续布局，`.contiguous()` 是 no-op
- 第二种：transpose 产生非连续 view，`.contiguous()` 触发实际内存拷贝

**教训**：性能测试的 tensor 创建方式必须与**实际推理时**的 layout 一致。在 vLLM/模型推理中，conv_state 通常以 `(N, dim, state_len)` 布局存储（state-contiguous），所以性能测试用 `torch.randn(N, dim, state_len)` 是正确的。
