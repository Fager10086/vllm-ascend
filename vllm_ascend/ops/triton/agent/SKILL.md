---
name: triton-ascend
description: 指导 Triton Ascend 算子在昇腾 NPU 上的开发与性能优化。当用户提到 Triton Ascend、triton 算子开发、NPU triton、昇腾 triton、triton kernel 编写、triton 性能优化、UB overflow、triton 调试时触发。即使用户只是在昇腾环境下讨论 triton 相关问题，也应触发此 skill。
---

# Triton Ascend 算子开发指南

本 skill 指导在昇腾 NPU 上使用 Triton Ascend 进行算子开发、性能优化和问题排查。

参考文档：https://triton-ascend-test.readthedocs.io/zh-cn/latest/

## 开发流程

拿到算子开发或迁移任务后，按以下步骤执行：

1. **分析需求**：明确算子的输入输出、数据类型、shape 范围，确认是 Vector 类还是其他类型算子。
2. **设计分核策略**：根据 vector 核数量设计 grid，将输入均匀分配给每个核，保证负载均衡。详见 `references/hardware.md` 中的 vector 核章节。
3. **计算 UB 用量**：统计 kernel 单次循环内所有 load 变量、中间变量的内存占用，确保总量不超过 UB 容量的一半（以启用 double buffer）。详见 `references/hardware.md` 中的 UB 章节。
4. **编写 kernel**：遵循下方「开发规则约束」编写 kernel 函数和辅助函数。
5. **调试与优化**：遇到问题时查阅 `references/troubleshooting.md`，性能优化参考 `references/tips.md`。

## 开发规则约束

以下规则在编写 kernel 时必须遵守：

### 单算子模式
不能使用入图方式提升单算子性能。模型侧会整网入图或 Piecewise 方式多算子入图，单个算子只关注单算子模式下的基础功能和性能。

### tl.load 与 mask
- tl.load 使用 mask 时可能导致 MTE 搬运单元和 vector 计算单元无法并行。
- 避免在 tl.load 中使用 other 参数（会触发 tl.where，导致 load 后无法与其他 load 并行）。
- 替代方案：load 后用 tl.where + mask 组合做掩码；当访问内存规则连续时，用 tl.insert_slice 替代。

### 分支与编译
- kernel 内 if-else 分支中，同名变量的 shape 必须相同，否则编译报错。

### 数据搬运
- 保证 load 多行连续数据，不能 load 多行离散数据（离散数据需逐行 load）。
- 传给 triton 算子的 tensor 必须连续，必要时用 `.contiguous()` 处理。
- tl.load 和 tl.store 的变量名尽可能不要复用，使用不同的变量名以提高可读性并避免潜在的数据流错误。

### 性能相关
- 减少 kernel 内 scalar 运算：与 pid 和循环变量无关的提到辅助函数，与循环变量无关的提到循环外，能合并则合并。
- 少用 tl.where（主要处理离散数据，性能差）。
- 避免对同一 tensor 多次 insert_slice。

## 参考文档

- `references/hardware.md`：vector 核与 UB 的硬件限制和计算方法
- `references/troubleshooting.md`：常见报错与问题定位
- `references/tips.md`：开发心得与性能优化技巧
