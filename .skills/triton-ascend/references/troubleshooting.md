# 问题定位与解决

## 流水异常

可能原因：
- kernel 函数内使用了 `range()` 作为循环计算。
- load 和 vector 未出现并行：可能是循环迭代之间不能并行。比如每次循环通过偏移获取地址而非通过当前迭代次数计算地址，最好把 load 逻辑统一写在一起。
- 编译器原因：编译器未开 pingpong。

## memref.alloc align 报错

报错示例：
```
oc("/tmp/.../kernel.ttadapter.mlir":106:11): error: cannot align 0 axis for %83 = "memref.alloc"()...
```

可能原因：当 load 了带有多个不同 mask 的 tensor，需要对这些 tensor 进行计算（如加法），在计算之前 reshape 可能会出现此问题。

**解决方案**：在计算完成后再 reshape。

## UB Overflow

可能原因及解决方案：
- 计算 kernel 每次循环的最大处理量有误：重新计算每次迭代能吃满 UB 的最大 token 数。
- kernel 尽量复用 tensor 变量名。
- 优化计算流程，减少同时存活的变量。
- 对高精度 tensor 降低精度：例如 BF16 升 float32 是无损的，`A(bf16) * B(float32)` 中如果 A 是从 GM 中读取的原始 bf16 tensor，A 可不升精度。
- 检查是否对二维 tensor 使用了一维的索引和 mask（一维 mask 会占用大量 UB）。

## if 分支报错

if 分支报错信息可能指向 if 判断条件本身，但实际报错位置不一定准确，需要同时排查 if 分支内部的代码。

## constexpr 参数

有些函数只能使用常量作为参数，需要在 kernel 入参处声明变量类型为 `tl.constexpr`。

涉及的函数包括：`reshape`、`arange`、`make_block_ptr` 等。
