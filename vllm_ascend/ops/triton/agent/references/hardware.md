# 硬件限制

对于 Vector 类算子在昇腾 NPU 上的开发，需要考虑两个核心硬件约束：vector 核数量和 UB（Unified Buffer）大小。

## Vector 核

vector 核的数量有限，它决定了 program id 的数量以及每个 id 的 kernel 处理的数据大小。

当 grid 对应的 pid 数量远大于核数时，多出来的 pid 只能排队分波次执行。如果每个 pid 的工作量不够大（或搬运开销占比很高），会出现“grid 很大但像串行一样跑”的情况。

**分核原则：**
- 将输入尽可能均匀地分给每个 vector 核，实现负载均衡。
- 让每个 program id 的 kernel 拥有相同的处理逻辑。
- 约束 grid 大小，避免任务过多。
- 当分核数大于可用核心时，建议在 kernel 函数内处理循环，每个核心均匀分配计算量。

## UB（Unified Buffer）

UB 是 vector 核计算时的动态存储单元。kernel 处理数据的流程：
1. 通过 `tl.load` 将数据从 GM（Global Memory）读到 UB
2. 在 UB 中进行计算（所有中间变量也存储在 UB 上）
3. 通过 `tl.store` 将结果从 UB 写回 GM

### UB 容量约束

由于 UB 大小受限，kernel 同时处理的数据是有限的。需要将 kernel 内的数据切分，以循环形式分批处理。

**切分策略：**
- 单次循环所占用的 UB 尽可能大，但必须小于 UB 大小的一半，以利用 double buffer 特性实现流水并行。
- 例如 910B 的 UB 大小为 192KB，单个 kernel 单次循环的 UB 占用量应控制在 ~85KB（192KB // 2，留余量存小变量）。
- 计算最大处理量时用 `//`（整除）而非 `tl.cdiv`，否则容易 UB overflow。

### 需要计入的变量

计算 UB 占用时，需要考虑：
- kernel 循环体内外 `tl.load` 进来的变量
- kernel 的中间计算变量
- 不同数据类型占用的 UB 大小不同，需分别计算
- 检查是否对二维 tensor 使用了一维的索引和 mask（一维 mask 会占用大量 UB）

### 变量名复用

kernel 内的变量名复用可以节省 UB，但需注意：
- 复用的 tensor 变量名之间可能存在同步关系，导致流水中同名 tensor 等待上一个计算完成（这由流水执行速度决定，无法从代码看出）。
- **建议**：对同一 store 变量的计算过程中涉及的 tensor 复用，不建议跨 load、store、计算流水的变量复用。
- 复用时计算循环占用 UB，按该变量所申请的最大内存计算即可。
