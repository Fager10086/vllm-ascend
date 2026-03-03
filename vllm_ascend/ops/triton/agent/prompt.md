一次循环中可处理的最大 token 数 ( N ) 由 UB（Unified Buffer）容量限制决定。

假设：

* 单核 UB 总大小为 170KB
* 为安全起见，只使用其中一半，即 85KB
* 每个 token 在 kernel 内部同时占用的 UB 空间为 ( $S_{\text{token}}$ )（包括 load、store 以及所有中间变量）

则必须满足约束：
$$
[
N \cdot S_{\text{token}} \le 85 \times 1024
]
$$
因此：
$$
[
N \le \left\lfloor \frac{85 \times 1024}{S_{\text{token}}} \right\rfloor
]
$$

举一个简单例子：

如果 kernel 仅执行一次 load 和一次 store，
加载一个大小为 `(batch_size, hidden_size)` 的 `bf16` Tensor，
每个元素占 2 Bytes，且没有其他中间变量，

则单个 token 占用的 UB 空间为：
$$
[
S_{\text{token}} = hidden_size \times 2
]
$$

因此有：
$$
[
N \cdot hidden_size \cdot 2 \le 85 \times 1024
]
$$
从而可以计算出单次循环允许的最大 token 数 ( N )。