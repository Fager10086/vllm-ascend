# Triton Ascend算子开发指南
https://triton-ascend-test.readthedocs.io/zh-cn/latest/ ，该网页下的各个子目录对Triton Ascend算子的基础开发指南进行了解释说明，包括但不限于triton函数的用法、示例，以及算子调试与调优的方法。

# 硬件限制
对于Vector类算子在昇腾NPU上的迁移，需要考虑到的硬件限制有：vector核的数量以及UB（Unified Buffer）大小。

## vector核
vector核的数量有限，它决定了program id的数量以及每个id的kernel处理的数据大小，当grid对应的pid数量远大于核数时，多出来的pid只能排队分波次执行，如果每个pid的工作量不够大（或者搬运开销占比很高），就会出现“grid很大但像串行一样跑”，因此，在NPU上把输入尽可能均匀地分给每个vector核来实现负载均衡的并行策略，尽可能让每个program id的kernel拥有相同的处理逻辑，约束Grid的大小，避免任务过多。

## UB：
UB是vector核计算时的动态存储单元，每个kernel处理数据时，需要将数据从GB（Global Buffer）通过tl.load方式读取到UB中，kernel操作UB中的数据，包括涉及到的中间变量统统都是在UB上存储的，当kernel计算完成，通过tl.store将输出从UB存回GB中。
- 由于UB大小受限，因此kernel同时处理的数据是有限的，我们需要将kernel内的数据进行切分，以循环的形式分批处理数据，对于循环切分的策略（即每次循环处理的数据大小切分策略），一般是尽可能地让单次循环所占用（即同时占用）的UB大小尽可能大，但必须小于UB大小的一半，以利用double buffer特性实现流水并行，举个例子，910B的UB大小为192KB，因此我们将单个kernel单次循环的UB占用量控制在85KB左右（192KB // 2，并且还要给出一部分余量用来存放一些小的变量），依据这个占用量，我们能够在辅助函数中计算出每次循环分块的大小（例如单次循环几个token）。
- 计算需要考虑到的变量包括：kernel内的循环体内外的load进来的变量、以及kernel的中间变量，这些均会占用UB，并且需要考虑数据类型，不同数据类型占用的UB大小不一。循环在上述约束的基础上尽可能多的处理数据，如果UB计算不合理，会导致ub overflow或者ub address out of bounds等报错。
- kernel内的变量名复用可以节省UB，但是需要注意复用的tensor变量名之间的同步关系，可能会导致流水中同名的tensor在等待上一个tensor计算的完成（这个是由流水的执行速度决定，无法通过代码看出）。建议对同一store变量的计算过程中涉及的tensor复用，不建议跨load、store、计算流水的变量复用。变量复用时计算循环占用UB，按该变量所申请的最大内存计算即可。

# 开发规则约束
- 在开发算子或者优化算子性能时，不能使用入图的方式来提升单算子性能，因为模型侧会整网入图或者Picewise方式多个算子入图。对于单个算子，我们应该关注的是单算子模式下的基础功能和性能，而不能将入图操作写在辅助函数里。
- tl.load在使用mask参数时，可能会导致MTE搬运单元和vector计算单元无法并行，是由于mask对输入数据的筛选操作导致，一种解决方案是：在tl.load时不使用other参数（这是因为other参数会触发tl.where操作，该操作与load绑定会导致load该tensor后与后面的其他load无法形成并行操作），在load之后再使用tl.where和mask组合进行掩码，但会引入额外的scalar计算，因此，当要访问的内存规则连续时，可以使用tl.insert_slice来作为性能优化方案进行替代。
- triton对于kernel内的if-else分支，在编译时会要求两个分支的同名变量的shape相同，当在if-else分支报错时，可排查此项规则。
- 保证在不超过ub buffer的情况和使用了double buffer的特性下，triton需要load多行连续数据。
- scalar运算会拖慢vector算子的性能，需要减少kernel内的scalar运算。把与kernel pid和循环变量无关的scalar运算提取到辅助函数中，把与循环变量无关的scalar运算提取到循环外，能合并的scalar运算就合并。
- 不能load多行离散数据，需要一行一行load。
- 少用tl.where语句，因为tl.where主要处理离散数据，性能较差。
- 需要保证传给triton算子的tensor连续，比如cos,sin=cos_sin_cache.index_select(0,positions).view(num_tokens,2,rope_size//2).repeat(1,1,2).chunk(2,dim=2)，此时cos，sin指向cos_sin_cache，并不是单独的tensor，需要contiguous操作将cos，sin变成单独的tensor。

# 开发心得
    - 与ascend C的分核方式不同，每次核函数load和save tensor的时候都需要使用mask，通过mask来处理不需要计算的尾块。

# 问题定位解决
- 流水异常的可能原因：
    - 1. kernel函数内使用了range()作为循环计算；
    - 2. load和vector未出现并行的现象，可能在与循环迭代之间不能并行，比如每次循环通过便宜的方式获取地址，而不是通过当前迭代的次数计算地址，最好load逻辑统一写在一起；
    - 3. 编译器原因，编译器未开pingpong
- 报错oc("/tmp/tmp08xlftr9/kernel.ttadapter.mlir":106:11):error:cannot align 0 axis for %83 = "memref.alloc"() <{alignment = 64 : i64, operandSegmentSizes = array<i32:0,0>}> {hivm.stride_align_dims = array<i32:0>,hivm.stride_align_value_in_byte = array<i32:32>}:() -> memref<2x2xi32,#hivm.address_space<ub>>的可能原因：当load了带有多个不同mask的tensor时，需要对这些tensor进行计算，比如加法，在计算之前reshape可能会出现上述问题，可以在计算完成后再reshape。
- uboverflow可能原因：计算kernel的每次循环时的最大处理量出现了问题。解决方案：
    - kernel尽量复用tensor变量名；
    - 是否可以优化计算流程；
    - 对于高精度tensor能否降低精度，比如Bf16升float32是无损的，计算的过程中，A（bf16）*B（float32），A是从gm中将原bf16的tensor读取出来，此时A就可不用升精度；
    - 查看是否对二维tensor使用了一维的索引和mask，一维的索引和mask会占用大量的UB；
    - 计算核函数每次迭代能够吃满ub的最大token数有误。
- if分支报错信息可能指向的是if判断条件本身，实际上这个报错信息并不准确，需要排查的还包括if分支内部的代码。
- 有些函数只能使用常量作为参数，需要在kernel入参处声明该变量的类型为tl.constexpr，如reshape、arange、make_block_ptr，都会有该限制。
