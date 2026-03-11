# 从零开始写算子-Vector篇
## 算子的数据流
Vector核的基本运行流程为（以达芬奇架构的昇腾910B架构为例）：

GM（HBM）——> UB ——> VEC核 ——> UB ——> GM（HBM）

对于上述流程进行详细解释：

其中GM（Gloab Memory）可以理解为显卡的静态全局存储，也就是常说的HBM（我们通常说一张显卡的显存是32G、64G等等，都是指的HBM大小），这里需要解释一下，CUBE和VEC计算单元在操作数据时，并非直接操作GM中的数据，而是分成一小块一小块的数据进行处理，具体来说，Vec核先通过MTE2（搬运单元）从GM读取一小块的数据到UB（Undified Block）中，然后Vec核对UB中的数据再进行处理，并将处理后的结果通过MTE3（搬运单元）保存到GM中。该流程中的数据搬入、计算、数据搬出的操作均在kernel内部开发实现。此处的UB仅限于VEC核，CUBE核类似但不是通过UB，详细流程于CUBE篇介绍。

## 算子的硬件约束
- 算子的计算是由AI Core执行的。910B中，一张卡上有20/24个AI Core，一个AI Core中包含一个CUBE Core和两个VEC Core，因此一张卡上有40/48个Vec核。将GM上的整个输入切分为若干块（切分方式由tiling策略决定，推荐将分核的数量直接固定为硬件的物理核数），每个Vec核各自独立加载GM input_ptr上的一小块数据，计算并保存输出到GM的output_ptr该块对应的内存地址上。
- 与带宽大、线程资源丰富的的GPU相比，NPU的访存受限。基于该限制，在切分输入时，我们一方面要保证各个核分到的数据大小尽可能相近，使核与核之间的数据处理速度相近；另一方面我们要减少核的访存次数，访存次数的减少是由增大每个核每次访存的数据量来实现的，即每次load/store的数据量尽可能大，尽可能合并load/store操作，减少load/store指令的次数，但是要注意，同一次load/store指令的数据内部尽可能保持整段或者多段连续以提升访存效率。
- 上述Tips提到的增大每个核每次访存的数据量大小，是有上限限制的，该限制由UB决定。以910B为例，其AI Core的UB大小为192KB，所以每个kernel每次处理的数据量不能超过192KB。此外，为了利用Double Buffer特性（也叫ping-pong缓冲技术，是一种搬运单元和计算单元的流水并行技术，提前搬运下一轮循环的数据的同时，计算本轮数据，实现数据搬运和计算的掩盖，节省搬运时间，该技术一般由编译器底层实现），需要预留出一半UB用于下一轮循环数据的搬运，因此，每轮循环中加载以及中间变量同时占用的数据量（除了Load的变量外，中间变量也会占用UB，同名变量以最大内存占用为准），不超过UB大小的一半，还要预留出一部分空间用于存放一些小的变量。每个Vec kernel每轮循环占用的数据量有：$$UB_{per\_iter\_per\_vec} < UB_{total} // 2.$$
一张910B卡的vec kernel每轮循环预留的大小为 $192 KB // 2 = 96KB$，预留一部分存放小变量的空间，每轮循环同时占用UB大小不超过85KB（该值不固定，可根据预留空间大小动态修改）。
- 由于上述UB大小限制，每个核可能无法一次处理切分给当前核的所有数据，需要在kernel内进行循环分批处理。因此，算子完整的任务处理流程为：先固定核数，再通过内部循环分批处理任务分块。

## Vector的优化目标
一个好的Vector算子，不包含CUBE计算，不包含或仅包含少量的Scalar计算，计算其性能流水呈现搬运Bound或者Vec计算Bound，且搬运核Vec计算能够实现很好的相互掩盖/并行。
- 搬运Bound：算子一直在进行搬运，从性能流水上看，MTE2的流水能够连起来，没有较为明显的空泡。
- Vec计算Bound：算子一直在进行Vec计算，从性能流水上看，Vector的流水能够连起来，没有较为明显的空泡。Vec核在Cube计算或者Scalar计算上性能较差，因此需要减少kernel内的这两类计算，使计算流水尽可能处于Vec计算上。

## Triton中Vector算子的构成
Triton算子的实现由辅助函数和Kernel两部分组成。辅助函数为Host侧计算，负责tiling（切分）策略、调用Kernel，以及一些通用的Scalar计算；Kernel是Device侧计算，负责算子的核心计算逻辑。

### 辅助函数
以下代码是一个triton算子的示例，对于模型侧来说，辅助函数就是这个算子的入口，通过调用该函数实现对算子的调用。我们通常会将与kernel内部无关的scalar计算放到辅助函数中以避免对device侧性能的影响。

与普通的pytorch函数相比，我们注意到最大的不同在于`split_qkv_rmsnorm_mrope_kernel[(block_dim,)]`这句调用kernel的命令，它除了传入kernel入参外，还有一个`[(block_dim,)]`的参数，这个参数我们称之为grid，它决定了输入tensor以怎样的切分方式分给每个Vec核。此处的`(block_dim,)`就是指将数据分为`block_dim`份，它可以是多维的。

```python
def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
    has_gate: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]

    gate_size = q_size if has_gate else 0

    if rope_dim is None:
        rope_dim = head_size
    IS_PARTIAL_ROPE = rope_dim != head_size

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num

    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = num_tokens // core_num

    q_output = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    gate_output = torch.empty(num_tokens, gate_size, device=qkv.device, dtype=qkv.dtype)

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    has_bias = q_bias is not None

    split_qkv_rmsnorm_mrope_kernel[(block_dim,)](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin,
        q_output,
        k_output,
        v_output,
        gate_output,
        num_tokens,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        num_q_heads,
        num_kv_heads,
        head_size,
        q_size,
        kv_size,
        eps,
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        has_bias,
        is_interleaved,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
        gate_size,
    )

    return q_output, k_output, v_output, gate_output
```
