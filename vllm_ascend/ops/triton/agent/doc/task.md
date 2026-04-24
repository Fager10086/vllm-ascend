# 利用 AI Skill 高效改造昇腾 Triton 算子：流程指南与实战复盘

## 目录

- [1. 本文目标](#1-本文目标)
- [2. 核心方法论：专家模板 + AI 仿照](#2-核心方法论专家模板--ai-仿照)
- [3. 标准工作流（5 步法）](#3-标准工作流5-步法)
  - [Step 1: 分类——确定算子类型与对应模板](#step-1-分类确定算子类型与对应模板)
  - [Step 2: 准备——建立专家模板](#step-2-准备建立专家模板)
  - [Step 3: 编写 Prompt——让 AI 精准工作](#step-3-编写-prompt让-ai-精准工作)
  - [Step 4: 执行——调用 Skill 并审视输出](#step-4-执行调用-skill-并审视输出)
  - [Step 5: 验证——确认正确性与性能](#step-5-验证确认正确性与性能)
- [4. 可用的 Skill 工具](#4-可用的-skill-工具)
- [5. 实战复盘](#5-实战复盘)
  - [5.1 案例一（CV 算子）：solve_tril 的 merge kernel](#51-案例一cv-算子solve_tril-的-merge-kernel)
  - [5.2 案例二（CV 算子）：wy_fast 的 recompute kernel](#52-案例二cv-算子wy_fast-的-recompute-kernel)
  - [5.3 案例三（Vector 算子）：split_qkv_rmsnorm_rope](#53-案例三vector-算子split_qkv_rmsnorm_rope)
- [6. Prompt 编写进阶](#6-prompt-编写进阶)
- [7. 常见问题与经验教训](#7-常见问题与经验教训)
- [8. 附录：快速参考](#8-附录快速参考)

---

## 1. 本文目标

本文档**不是**一份算子优化技术手册，而是一份**流程指南**。

我们在昇腾 NPU 上用 AI Skill 成功改造了多个 VLLM-Ascend 的 Triton 算子，积累了一套可复用的工作流。本文将这套工作流提炼为标准步骤，让其他开发者能够：

1. **快速上手**：即使不熟悉具体优化技术，也能按流程调用 AI Skill 完成算子改造
2. **稳定产出**：通过"专家模板约束"而非"自由发挥"，获得可控、可审视的 AI 输出
3. **规模化复用**：同一套流程适用于不同类型的算子，一个模板可覆盖一批同类算子

> **核心理念**：让 AI 做"模仿 + 适配"的工作，人类做"判断 + 验证"的工作——这是当前 AI 辅助编程最高效的协作模式。

---

## 2. 核心方法论：专家模板 + AI 仿照

### 2.1 为什么不能直接让 AI 自由优化

我们最初尝试过直接让 Skill 优化原生算子，遇到了以下问题：

| 问题 | 表现 | 后果 |
|------|------|------|
| **结果发散** | AI 生成多种优化方向（访存、tiling、mask……），每次不同 | 人工审视成本高，难以建立信任 |
| **泛化性差** | 生成的方案针对单个算子定制 | 无法复用到同类算子，每个都要重新投入 |
| **修改范围失控** | AI 同时修改调度逻辑和计算逻辑 | 引入难以排查的正确性问题 |

### 2.2 "专家模板 + AI 仿照"模式

解决思路很简单——**给 AI 一个范例去模仿，而不是让它从零创造**：

```
                    ┌─────────────┐
                    │  算子专家    │
                    │  手工优化    │
                    │  一个算子    │
                    └──────┬──────┘
                           │ 产出
                    ┌──────▼──────┐
                    │  专家模板    │
                    │ (改造前/后)  │
                    └──────┬──────┘
                           │ 输入
                    ┌──────▼──────┐
         ┌────────►│  AI Skill    │◄────────┐
         │         │  仿照改造    │         │
         │         └──────┬──────┘         │
         │                │                │
   ┌─────┴─────┐   ┌─────▼─────┐   ┌─────┴─────┐
   │ 同类算子 A │   │ 同类算子 B │   │ 同类算子 C │
   └───────────┘   └───────────┘   └───────────┘
```

**关键收益**：

- **可控**：AI 的修改被约束在"模仿模板改动"上，不会发散
- **可审**：改动范围固定，审视者知道该看什么
- **可复用**：一个模板覆盖一批同类算子，边际成本递减
- **低门槛**：执行者无需深入理解优化原理，按流程操作即可

### 2.3 投入产出分析

| 阶段 | 投入 | 产出 |
|------|------|------|
| 专家手工优化第 1 个算子 | 高（需要专家经验） | 1 个优化后的算子 + 1 个可复用模板 |
| AI 仿照第 2~N 个算子 | 低（编写 Prompt + 审视输出） | N-1 个优化后的算子 |

> **第 1 个算子的专家投入是一次性成本，之后每个同类算子的改造成本大幅降低。**

---

## 3. 标准工作流（5 步法）

以下是将一个新算子用 AI Skill 完成改造的标准流程：

### Step 1: 分类——确定算子类型与对应模板

拿到待改造算子后，首先判断它属于哪一类，从而选择正确的专家模板：

| 判断依据 | CV 算子 | Vector 算子 |
|----------|---------|-------------|
| 核心计算 | 矩阵乘（`tl.dot`）、分块矩阵运算 | 逐元素运算、归约（RMSNorm）、旋转（RoPE） |
| 数据模式 | 分块 tile 访问（`BT × BK`） | 按 token 行加载 |
| 使用 `make_block_ptr` | 常见 | 少见 |
| 硬件单元 | AI Core（Cube） | Vector Core |
| **使用模板** | `chunk_scaled_dot_kkt` 优化对 | `split_qkv_rmsnorm_mrope` 优化对 |

**动作**：阅读目标算子代码 → 判断类型 → 找到对应模板文件。

### Step 2: 准备——建立专家模板

> 如果你的算子类型已有现成模板（见附录），可跳过此步。

对于一种新类型的算子，需要由算子专家先手工优化一个代表性算子作为模板：

1. **选择代表性算子**：选择该类型中结构最典型的一个
2. **手工完成优化**：专家按照硬件特性完成改造
3. **保留改造前后两个版本**：这两个文件共同构成"专家模板"
   - 改造前文件：让 AI 理解"原始代码长什么样"
   - 改造后文件：让 AI 理解"应该改成什么样"
4. **确保模板文件可访问**：放在 AI Skill 能读取到的路径下

> **关键原则**：专家模板的价值不在于优化了多少性能，而在于它定义了一种"改造模式"——哪些该改、哪些不该动、改的边界在哪里。

#### 专家模板示例：GDN CV 算子模板（chunk_scaled_dot_kkt）

以下展示 CV 算子专家模板的核心改动，让读者理解"模板"具体长什么样：

**改造前**（专家模板的原始版本）：

```python
# ===== 调用侧：Grid 与问题规模挂钩 =====
chunk_scaled_dot_kkt_fwd_kernel[(NT, 1)](
    ..., BK=128, num_warps=8, num_stages=3, multibuffer=True
)

# ===== Kernel 侧：每个 program_id 处理一个工作项 =====
@triton.jit(do_not_specialize=["T"])
def chunk_scaled_dot_kkt_fwd_kernel(...):
    i_t_i, _ = tl.program_id(0), tl.program_id(1)
    for i_bh in range(B * H):
        # ... 内部计算逻辑 ...
```

**改造后**（专家手工优化后的版本）：

```python
# ===== 硬件拓扑查询 =====
import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]

# ===== 调用侧：Grid 绑定硬件核心数 =====
core_num = AICORE_NUM
grid = (core_num,)
chunk_scaled_dot_kkt_fwd_kernel[grid](
    ..., grid0=NT, BK=128, num_warps=8, num_stages=3, multibuffer=True
)

# ===== Kernel 侧：持久化循环，每核处理多个工作项 =====
@triton.jit(do_not_specialize=["T", "B"])       # ← 新增 "B"，减少重编译
def chunk_scaled_dot_kkt_fwd_kernel(..., grid0=0):  # ← 新增 grid0 参数
    core_num = tl.num_programs(0)
    pid = tl.program_id(0)
    T_orig = T                                    # ← 保存 T 原始值（VARLEN 模式需要）
    for i_t_i in range(pid, grid0, core_num):     # ← 持久化循环
        T = T_orig                                # ← 每轮恢复 T
        bt_stride = B * T                         # ← 依赖 T 的变量移入循环
        for i_bh in range(B * H):
            # ... 内部计算逻辑完全不变 ...
```

**模板提炼的改造模式**（这是 AI 仿照时实际遵循的模式）：

| # | 改动 | 说明 |
|---|------|------|
| 1 | Grid 从 `(NT, 1)` 改为 `(AICORE_NUM,)` | 绑定硬件核心数，消除超额调度 |
| 2 | 新增 `grid0` 参数传递逻辑工作量 | 解耦物理 grid 与逻辑工作量 |
| 3 | Kernel 内用 `for i in range(pid, grid0, core_num)` 包裹 | 持久化循环，每核处理多个工作项 |
| 4 | `do_not_specialize` 扩展加入 `"B"` | 避免 batch size 变化触发重编译 |
| 5 | `T_orig = T` + 循环内 `T = T_orig` | VARLEN 模式下 T 会被修改，需每轮恢复 |
| 6 | 依赖 T 的变量（如 `bt_stride`）移入循环 | 必须在 T 恢复后重新计算 |
| 7 | **内部计算逻辑完全不变** | 纯调度层面优化，不触碰计算正确性 |

#### 专家模板示例：Vector 算子模板（split_qkv_rmsnorm_mrope）

Vector 算子模板涉及更深层的计算改造，核心改造模式：

| # | 改动 | 说明 |
|---|------|------|
| 1 | Grid 从多维（如 `(n_rows, n_cols, 1)`）改为一维 `(core_num,)` | 每个核处理完整 token 行 |
| 2 | 多个独立的 Q/K/V 循环合并为单一循环 | 消除冗余循环开销和冗余输入加载 |
| 3 | 分块 mask 加载改为整行连续无 mask 加载 | 一次连续 DMA 传输优于多次小 mask 传输 |
| 4 | Q、K 的 RMSNorm 合并计算 | 合并相同操作提升向量单元利用率 |
| 5 | V 提前写回（early-store） | MTE 与 Vector 单元独立运行，V 写回可与后续计算并行 |
| 6 | 循环不变量外提 | cos/sin 偏移、weight/bias 在循环外加载一次 |
| 7 | 新增 `do_not_specialize=["batch_size"]` | 避免 batch size 变化触发重编译 |

### Step 3: 编写 Prompt——让 AI 精准工作

Prompt 的质量直接决定 AI 输出的可用性。核心原则是**四要素齐备**：

```
┌──────────────────────────────────────────────────┐
│  ① 触发 Skill        → 指定使用哪个 Skill        │
│  ② 目标文件/函数      → 告诉 AI 改什么            │
│  ③ 参考模板（前+后）  → 告诉 AI 怎么改            │
│  ④ 验证方式（可选）   → 告诉 AI 怎么验证          │
└──────────────────────────────────────────────────┘
```

**CV 算子 Prompt 模板**：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
<目标文件绝对路径>的<目标kernel函数名>。
优化必须参考<改造后参考文件路径>
对<改造前参考文件路径>使用的性能优化手段。
```

**Vector 算子 Prompt 模板**：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子性能优化：
<目标文件绝对路径>。
精度与功能正确性验证使用<验证命令>，
性能测试用<性能测试命令>。
优化必须参考<改造后参考文件路径>
对<改造前参考文件路径>使用的性能优化手段。
```

**编写要点**：

| 要点 | 说明 | 为什么重要 |
|------|------|-----------|
| 明确触发 Skill | 以 `/triton-ascend-ops-optimizer` 开头 | 确保激活正确的 Skill |
| 使用绝对路径 | 文件路径用绝对路径 | 避免 AI 找错文件 |
| 指定 kernel 函数名 | 一个文件可能有多个 kernel | 避免 AI 改错函数 |
| 同时给出改造前+改造后 | 两个文件形成对比 | AI 通过 diff 学习改造模式 |
| 使用"必须参考" | 而非"可以参考" | 强约束，防止 AI 发散 |
| 提供验证命令 | 精度验证 + 性能测试 | AI 可自行闭环验证 |

### Step 4: 执行——调用 Skill 并审视输出

1. **发送 Prompt**：在 Claude Code 中输入编写好的 Prompt
2. **观察 AI 执行过程**：注意 AI 是否在按模板模式做改造
3. **审视输出代码**：重点检查以下内容

| 算子类型 | 审视重点 |
|----------|---------|
| CV 算子 | Grid 是否改为硬件核心数？持久化循环是否正确？多维 grid 展开逻辑是否正确？T 的保存/恢复是否到位？**内部计算逻辑是否保持不变？** |
| Vector 算子 | 循环是否正确融合？`extract_slice` 边界是否正确？early-store 时序是否合理？mask 是否已消除？ |

4. **如有问题，补充 Prompt 修正**：例如"T 需要在循环内恢复原始值"

### Step 5: 验证——确认正确性与性能

1. **数值正确性**：运行精度验证测试，确认输出与改造前一致
2. **性能提升**：运行性能测试，确认有预期的性能收益
3. **回归测试**：确认不影响其他功能

> **建议**：在 Prompt 中直接提供验证命令（见 Step 3），AI 会自动执行验证。

---

## 4. 可用的 Skill 工具

### `/triton-ascend-ops-optimizer`

- **用途**：昇腾 NPU 上 Triton 算子的深度性能优化
- **触发方式**：Prompt 中提及"昇腾 NPU 上 Triton 算子性能优化"时自动触发
- **在本流程中的角色**：接收专家模板 + 目标算子 → 仿照模板完成改造

### `/triton-operator-performance-eval`（基础软件平台开发）

- **用途**：评估 Ascend NPU 上 Triton 算子的性能表现
- **典型用法**：在改造完成后，用于性能瓶颈分析和硬件利用率评估

---

## 5. 实战复盘

以下三个案例展示了标准工作流在不同算子上的实际应用过程。重点不在于技术细节，而在于**如何操作每一步**。

### 5.1 案例一（CV 算子）：solve_tril 的 merge kernel

| 流程步骤 | 实际操作 |
|----------|---------|
| **Step 1: 分类** | 阅读 `solve_tril.py` 的 `merge_16x16_to_64x64_inverse_kernel`，发现包含矩阵分块运算 → 判定为 CV 算子 |
| **Step 2: 准备** | 已有现成模板：`chunk_scaled_dot_kkt.py` → `chunk_scaled_dot_kkt_opt.py` |
| **Step 3: Prompt** | 见下方 |
| **Step 4: 执行** | AI 正确仿照模板：Grid 从 `[NT, B*H]` 改为 `[AICORE_NUM]`，添加持久化循环，二维 grid 展开为一维，内部计算不变 |
| **Step 5: 验证** | 数值正确性通过 |

**使用的 Prompt**：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/solve_tril.py
的 merge_16x16_to_64x64_inverse_kernel。
优化必须参考 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt_opt.py
对 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py
使用的性能优化手段
```

**改造前代码**（目标算子原始状态）：

```python
# 调用侧：二维 Grid，与问题规模挂钩
merge_16x16_to_64x64_inverse_kernel[NT, B * H](...)

# Kernel 侧：每个 program_id 对应一个 (时间块, batch-head) 工作项
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    Ad, A, offsets, T, H, BT: tl.constexpr
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    # ... 使用原始指针算术 + 手工 mask 进行矩阵块合并计算 ...
```

**AI 改造后代码**（AI 仿照 CV 模板自动生成）：

```python
# ===== 新增：硬件拓扑查询 =====
import triton.runtime.driver as driver

device = torch.npu.current_device()
properties = driver.active.utils.get_device_properties(device)
AICORE_NUM = properties["num_aicore"]

# 调用侧：Grid 绑定硬件核心数
core_num = AICORE_NUM
grid = (core_num,)
merge_16x16_to_64x64_inverse_kernel[grid](
    ..., grid0=NT * B * H    # ← 二维工作量展开为一维
)

# Kernel 侧：持久化循环 + 二维展开
@triton.jit(do_not_specialize=["T", "B"])             # ← 新增 "B"
def merge_16x16_to_64x64_inverse_kernel(
    Ad, A, offsets, T, H, BT: tl.constexpr, grid0=0   # ← 新增 grid0
):
    core_num = tl.num_programs(0)
    pid = tl.program_id(0)
    T_orig = T
    for i_work in range(pid, grid0, core_num):         # ← 持久化循环
        T = T_orig
        i_t = i_work // (B * H)                        # ← 从一维反算原始二维索引
        i_bh = i_work % (B * H)
        # ... 内部计算逻辑完全不变 ...
```

**注意到的要点**：该算子原本用二维 grid `[NT, B*H]`，AI 正确地将二维 `(i_t, i_bh)` 展开为一维工作项，并在循环内反算回原始索引。

### 5.2 案例二（CV 算子）：wy_fast 的 recompute kernel

| 流程步骤 | 实际操作 |
|----------|---------|
| **Step 1: 分类** | 阅读 `wy_fast.py` 的 `recompute_w_u_fwd_kernel`，包含矩阵运算 → CV 算子 |
| **Step 2: 准备** | 复用同一个 CV 模板 |
| **Step 3: Prompt** | 与案例一结构相同，仅替换目标文件路径 |
| **Step 4: 执行** | AI 正确完成改造：Grid `(NT, B)` → `(AICORE_NUM,)`，添加 `grid0` 参数 |
| **Step 5: 验证** | 数值正确性通过 |

**使用的 Prompt**：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/wy_fast.py。
优化必须参考 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt_opt.py
对 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/fla/chunk_scaled_dot_kkt.py
使用的性能优化手段。
```

**改造前代码**（目标算子原始状态）：

```python
# 调用侧：二维 Grid，与 (NT, B) 挂钩
recompute_w_u_fwd_kernel[(NT, B)](
    ..., num_warps=4, num_stages=3
)

# Kernel 侧：内层只遍历 H（head 数）
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(...):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    for i_h in range(H):
        # ... 包含两个子循环分别处理 V 维度和 K 维度 ...
        # ... 使用原始指针 + mask（非 make_block_ptr）...
```

**AI 改造后代码**（AI 仿照 CV 模板自动生成）：

```python
# 调用侧：Grid 绑定硬件核心数
core_num = AICORE_NUM
grid = (core_num,)
recompute_w_u_fwd_kernel[grid](
    ..., grid0=NT * B, num_warps=4, num_stages=3    # ← grid0 = NT * B
)

# Kernel 侧：持久化循环
@triton.jit(do_not_specialize=["T", "B"])            # ← 新增 "B"
def recompute_w_u_fwd_kernel(..., grid0=0):           # ← 新增 grid0
    core_num = tl.num_programs(0)
    pid = tl.program_id(0)
    T_orig = T
    for i_work in range(pid, grid0, core_num):        # ← 持久化循环
        T = T_orig
        i_t = i_work // B                             # ← 原始 grid 是 (NT, B)
        i_bh = i_work % B
        for i_h in range(H):
            # ... 内部计算逻辑完全不变 ...
```

**体现的复用价值**：这个算子与案例一结构不同（grid 是 `(NT, B)` 而非 `(NT, B*H)`，内层只遍历 H），但 AI 正确泛化了模板模式——grid 展开公式自动适配为 `i_work // B` 和 `i_work % B`，无需额外指导。

### 5.3 案例三（Vector 算子）：split_qkv_rmsnorm_rope

| 流程步骤 | 实际操作 |
|----------|---------|
| **Step 1: 分类** | 阅读 `split_qkv_rmsnorm_rope.py`，包含 RMSNorm + RoPE 逐元素运算 → Vector 算子 |
| **Step 2: 准备** | 使用 Vector 模板：`split_qkv_rmsnorm_mrope.py` → `split_qkv_rmsnorm_mrope_opt.py` |
| **Step 3: Prompt** | 使用 Vector 模板，**额外提供了验证和性能测试命令** |
| **Step 4: 执行** | AI 完成了多项改造：Grid 一维化、三循环融合、消除 mask、Q+K RMSNorm 合并、V early-store 等 |
| **Step 5: 验证** | AI 自动执行了精度验证和性能测试 |

**使用的 Prompt**：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子性能优化：
/vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_rope.py。
精度与功能正确性验证使用 python -m pytest /vllm-workspace/test_rope/test_split_qkv_rmsnorm_rope_raw.py，
性能测试用 python3 /vllm-workspace/test_rope/test_split_qkv_rmsnorm_rope_raw_perf.py。
优化必须参考 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope_opt.py
对 /vllm-workspace/vllm-ascend/vllm_ascend/ops/triton/linearnorm/split_qkv_rmsnorm_mrope.py
使用的性能优化手段。
```

**改造前代码**（目标算子原始状态）：

```python
# 调用侧：二维 Grid，按行和列分块
n_cols = kv_hidden_size // KV_BLOCK_SIZE
n_rows = num_vectorcore // n_cols
grid = (n_rows, n_cols, 1)
split_qkv_rmsnorm_rope_kernel[grid](...)

# Kernel 侧：三个独立循环分别处理 Q、K、V
@triton.jit   # 无 do_not_specialize
def split_qkv_rmsnorm_rope_kernel(...):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)

    # 循环1：加载 Q 分块 → RMSNorm → RoPE → 写回
    for row_idx in tl.range(row_pid, batch_size, row_step):
        q_block = tl.load(input_ptr + offset + col_indices, mask=valid_mask)
        # ... RMSNorm + RoPE ...
        tl.store(q_out_ptr + ..., q_block, mask=valid_mask)

    # 循环2：加载 K 分块 → RMSNorm → RoPE → 写回
    for row_idx in tl.range(row_pid, batch_size, row_step):
        k_block = tl.load(input_ptr + offset + col_indices, mask=valid_mask)
        # ...

    # 循环3：加载 V 分块 → 写回
    for row_idx in tl.range(row_pid, batch_size, row_step):
        v_block = tl.load(input_ptr + offset + col_indices, mask=valid_mask)
        # ...
```

> **原始代码问题**：同一个 token 的输入数据被从 HBM 加载了 3 次；每次 load/store 都带 mask；列方向分块对 RMSNorm（需归约整个 HEAD_DIM）无意义。

**AI 改造后代码**（AI 仿照 Vector 模板自动生成）：

```python
# 调用侧：一维 Grid，绑定硬件核心数
core_num = min(AIVECCORE_NUM, batch_size)
grid = (core_num,)
split_qkv_rmsnorm_rope_kernel[grid](...)

# Kernel 侧：单一融合循环 + 整行无 mask 加载
@triton.jit(do_not_specialize=["batch_size"])           # ← 新增
def split_qkv_rmsnorm_rope_kernel(...):
    pid = tl.program_id(0)
    core_num = tl.num_programs(0)

    # 循环不变量外提：weight/bias/cos/sin 在循环外加载一次
    weight = tl.load(weight_ptr + tl.arange(0, HIDDEN_SIZE))
    cos = tl.load(cos_ptr + cos_indices)
    sin = tl.load(sin_ptr + sin_indices)

    # 单一融合循环：一次加载整行 QKV
    for row_idx in tl.range(pid, batch_size, core_num):
        # 一次加载整行，无 mask（维度为 constexpr）
        all_data = tl.load(input_ptr + row_idx * stride + all_col_indices)

        # 用 extract_slice 拆分 Q/K/V
        q_data = tl.extract_slice(all_data, [0],                          [q_hidden_size])
        k_data = tl.extract_slice(all_data, [q_hidden_size],              [k_hidden_size])
        v_data = tl.extract_slice(all_data, [q_hidden_size + k_hidden_size], [v_hidden_size])

        # V 提前写回（early-store）：MTE 与 Vector 单元并行
        tl.store(v_out_ptr + ..., v_data)

        # Q+K 合并做 RMSNorm + RoPE
        qk = tl.insert_slice(...)   # 拼接为 (num_qk_heads, HEAD_DIM)
        # ... 统一 RMSNorm + RoPE ...
        tl.store(q_out_ptr + ..., q_result)
        tl.store(k_out_ptr + ..., k_result)
```

> **改造效果**：访存次数 3 → 1，消除全部 mask，V 写回与 Q/K 计算流水并行。

**与 CV 案例的流程差异**：

| 对比项 | CV 算子（案例一、二） | Vector 算子（本案例） |
|--------|---------------------|---------------------|
| Prompt 中的优化类型 | "grid改造" | "性能优化" |
| 是否提供验证命令 | 否 | 是（精度 + 性能） |
| AI 改动范围 | 仅调度层面 | 调度 + 计算 + 访存 |
| 审视重点 | grid 展开、T 保存/恢复 | 循环融合、extract_slice 边界 |

---

## 6. Prompt 编写进阶

### 6.1 同时优化多个 kernel

一个文件中有多个需要改造的 kernel 时，在 Prompt 中列出所有函数名：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
<文件路径>的 kernel_a 和 kernel_b。
优化必须参考...
```

### 6.2 为特殊情况添加约束提示

如果目标算子有特殊逻辑（如 VARLEN 模式），在 Prompt 末尾追加约束：

```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
<文件路径>的<kernel名>。
优化必须参考...
注意：该算子有 VARLEN 模式，需要正确处理 T 的保存/恢复。
```

### 6.3 提升 AI 输出质量的技巧

| 技巧 | 做法 | 效果 |
|------|------|------|
| **缩小改动范围** | CV 算子用"grid改造"而非"性能优化" | 约束 AI 只做调度改造 |
| **提供验证闭环** | 附带 pytest 和性能测试命令 | AI 可自行验证并迭代 |
| **强制参考** | 使用"必须参考"而非"参考" | 防止 AI 自行发挥 |
| **指定函数名** | "文件X的kernel_Y" | 避免 AI 改错函数 |

---

## 7. 常见问题与经验教训

### Q1: 我没有专家模板怎么办？

需要先由具备算子优化经验的开发者手工完成一个同类算子的优化，作为模板。没有模板的情况下直接让 AI 优化，结果不可控。

### Q2: 我的算子不属于 GDN 的 CV 或 Rope 两类怎么办？

现有流程在GDN 的 CV 或 Rope 上进行验证，提供了一个 Agent 优化范式。如果你的算子属于新类型：
1. 按照 Step 2 为新类型建立专家模板
2. 总结该类型的改造模式
3. 为新类型编写 Prompt 模板
4. 将新模板补充到本文档中

### Q3: AI 的输出有问题怎么修？

- **小问题**（如变量名错、边界 off-by-one）：直接手动修复
- **模式性错误**（如 grid 展开逻辑整体错误）：在 Prompt 中追加具体约束，重新执行
- **根本性偏离**（AI 没有按模板改造）：检查 Prompt 是否四要素齐备，"必须参考"是否明确

### Q4: 如何判断算子类型？

快速判断方法：

| 看什么 | CV 算子 | Vector 算子 |
|--------|---------|-------------|
| 是否有 `tl.dot` | 是 | 否 |

### Q5: 一个模板能覆盖多少算子？

经验数据：
- CV 模板（`chunk_scaled_dot_kkt`）：已成功覆盖 `solve_tril`、`wy_fast` 等 FLA 目录下的同类算子
- Vector 模板（`split_qkv_rmsnorm_mrope`）：已成功覆盖 `split_qkv_rmsnorm_rope` 等算子

一般来说，同一目录下结构相似的算子可以共用一个模板。

### Q6: CV 算子改造时，AI 可能犯哪些错？

| 常见错误 | 表现 | 检查方法 |
|----------|------|---------|
| 多维 grid 展开错误 | `i_t` 和 `i_bh` 的计算公式不对 | 核对原始 grid 维度和展开公式 |
| 忘记保存/恢复 T | VARLEN 模式下第二轮循环 T 值错误 | 检查是否有 `T_orig = T` + `T = T_orig` |
| 循环依赖变量未移入循环 | 依赖 T 的变量（如 `bt_stride = B * T`）在循环外只算了一次 | 检查所有依赖 T 的表达式 |
| 修改了内部计算逻辑 | 不应改动的 block 大小、计算公式被修改 | 对比原始代码的计算部分 |

---

## 8. 附录：快速参考

### A. 已有专家模板

| 算子类型 | 改造前 | 改造后 | 路径前缀 |
|----------|--------|--------|---------|
| CV | `chunk_scaled_dot_kkt.py` | `chunk_scaled_dot_kkt_opt.py` | `vllm_ascend/ops/triton/fla/` |
| Vector | `split_qkv_rmsnorm_mrope.py` | `split_qkv_rmsnorm_mrope_opt.py` | `vllm_ascend/ops/triton/linearnorm/` |

### B. 标准工作流速查

```
1. 分类   → 读代码，判断 CV / Vector
2. 准备   → 找到对应的专家模板（改造前 + 改造后两个文件）
3. Prompt → 四要素：触发 Skill + 目标文件 + 参考模板 + 验证命令
4. 执行   → 发送 Prompt，审视 AI 输出是否遵循模板模式
5. 验证   → 跑精度测试 + 性能测试
```

### C. Prompt 速查模板

**CV 算子**：
```
使用 Skill /triton-ascend-ops-optimizer 实现算子grid改造：
<目标文件绝对路径>的<目标kernel函数名>。
精度与功能正确性验证使用<验证命令>，
性能测试用<性能测试命令>。
优化必须参考<改造后参考文件路径>
对<改造前参考文件路径>使用的性能优化手段。
```

**Vector 算子**：
```
使用 Skill /triton-ascend-ops-optimizer 实现算子性能优化：
<目标文件绝对路径>。
精度与功能正确性验证使用<验证命令>，
性能测试用<性能测试命令>。
优化必须参考<改造后参考文件路径>
对<改造前参考文件路径>使用的性能优化手段。
```

---

## 版本信息

- **文档版本**: 2.0
- **适用环境**: 昇腾 NPU + Triton Ascend
- **CV 算子模板路径**: `vllm-ascend/vllm_ascend/ops/triton/fla/`
- **Vector 算子模板路径**: `vllm-ascend/vllm_ascend/ops/triton/linearnorm/`
