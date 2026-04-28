# Triton 算子设计开发模板

## 1. 算子基本信息 (Operator Metadata)
本节定义算子的身份，用于生成文件名及类名。
- 算子名称 (OpType)：AddCustom(示例)
- 功能描述：简述算子实现的数学逻辑或业务逻辑。
- 计算公式：使用标准数学符号表示，例如：$z = x + y$
- 支持的数据类型：（如：float16，float32，int32）
- 支持的格式：（如：ND，NCHW，NHWC）

## 2. 输入输出定义 (Interface Specification)
供 Agent 定义 Tiling 结构体和核心类的输入输出。

| 端口名称 | 角色 (Input/Output) | 数据类型 | 数据布局 (Format) | 形状 (Shape) 约束 |
| --- | --- | --- | --- | --- |
|x |Input|float16|ND|任意维度| 存在对齐和不对齐的情况|
|y|Input|float16|ND|与 x 一致|
|z|Output|float16|ND|与 x 一致|

## 3. 核心算法逻辑 (Algorithm & Implementation)

本节直接影响 Compute 阶段的代码生成。

### 3.1 内存空间布局

- Global Memory: 输入输出数据存放位置。
- Local Memory (UB): Vector 核的片上存储，所有计算都在 UB 中进行
- 一个完整的数据流为：
    ```python
    Global Memory (GM) → [tl.load] → UB → [Vector 计算] → UB → [tl.store] → GM
    ```

### 3.2 搬运与计算流程

- Stage：CopyIn: 如何从 Global Memory 搬运到 Local Memory (使用tl.load)。
- Stage：Compute：调用的矢量指令(如 *, +, tl.extract_slice 等)。
- Stage：CopyOut：如何从 Local Memory 搬出到 Global Memory。

NOTE：CopyIn、Compute、CopyOut 可能不是一次性完成的，可能存在多次 CopyIn、Compute、CopyOut 的情况

## 4. Tiling 设计策略 (Tiling Strategy)

这是 Triton 开发最关键的部分。Agent 需根据此逻辑生成辅助函数。

- 切分维度：如何切分输入，每个Kernel处理一个 tile。
- 多核并行(grid)：建议使用固定大小，对于纯 Vector 算子，grid大小为 vector 核的数量。
- Double Buffer：默认启用，用于掩盖内存搬运和计算指令的延迟。

## 5. 约束与边界条件 (Constraints)

防止 Agent 生成无法编译或溢出的代码。

- 最大数据量限制：