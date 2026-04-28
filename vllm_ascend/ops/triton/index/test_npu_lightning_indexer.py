import pytest
import torch
import torch_npu

from vllm_ascend.utils import enable_custom_op

enable_custom_op()

# ============ Constants ============
WARMUP_ITERS = 5
DEVICE = "npu"

# Input shapes from user specification
QUERY_SHAPE = (4096, 64, 128)       # TND: T=4096, N=64, D=128
KEY_SHAPE = (1058, 128, 1, 128)     # PA_BSND: num_blocks=1058, block_size=128, N=1, D=128
WEIGHTS_SHAPE = (4096, 64)          # (T, N)
BLOCK_TABLE_SHAPE = (1, 1040)       # (batch=1, max_blocks=1040)
BATCH_SIZE = 1
NUM_QUERY_TOKENS = QUERY_SHAPE[0]   # 4096
NUM_BLOCKS_IN_TABLE = BLOCK_TABLE_SHAPE[1]  # 1040
NUM_BLOCKS_IN_POOL = KEY_SHAPE[0]   # 1058
BLOCK_SIZE = KEY_SHAPE[1]           # 128
SPARSE_COUNT = 2048
SPARSE_MODE = 3
LAYOUT_QUERY = "TND"
LAYOUT_KEY = "PA_BSND"


def _create_inputs(dtype=torch.bfloat16):
    """Create input tensors with the specified shapes and move to NPU."""
    query = torch.randn(QUERY_SHAPE, dtype=dtype, device=DEVICE)
    key = torch.randn(KEY_SHAPE, dtype=dtype, device=DEVICE)
    weights = torch.randn(WEIGHTS_SHAPE, dtype=dtype, device=DEVICE)

    # actual_seq_lengths_query: cumulative query token count per batch, shape [1]
    actual_seq_lengths_query = torch.tensor(
        [NUM_QUERY_TOKENS], dtype=torch.int32, device=DEVICE
    )

    # actual_seq_lengths_key: KV sequence length (in blocks) per batch, shape [1]
    actual_seq_lengths_key = torch.tensor(
        [NUM_BLOCKS_IN_TABLE], dtype=torch.int32, device=DEVICE
    )

    # block_table: maps logical block index -> physical block index, shape [1, 1040]
    block_table = torch.randint(
        0, NUM_BLOCKS_IN_POOL, BLOCK_TABLE_SHAPE, dtype=torch.int32, device=DEVICE
    )

    return query, key, weights, actual_seq_lengths_query, actual_seq_lengths_key, block_table


# =====================================================================
# Performance profiling entry point (for msprof)
# =====================================================================
def run_perf():
    """Run npu_lightning_indexer with warmup, suitable for msprof profiling.

    Usage:
        msprof --application="python test_npu_lightning_indexer.py" --output=./prof_result
    """
    query, key, weights, seq_q, seq_k, block_table = _create_inputs()

    # # Warmup to exclude JIT compilation overhead
    # for _ in range(WARMUP_ITERS):
    #     torch_npu.npu_lightning_indexer(
    #         query=query,
    #         key=key,
    #         weights=weights,
    #         actual_seq_lengths_query=seq_q,
    #         actual_seq_lengths_key=seq_k,
    #         block_table=block_table,
    #         layout_query=LAYOUT_QUERY,
    #         layout_key=LAYOUT_KEY,
    #         sparse_count=SPARSE_COUNT,
    #         sparse_mode=SPARSE_MODE,
    #     )
    # torch.npu.synchronize()

    # Profiling region
    for _ in range(2):
        torch_npu.npu_lightning_indexer(
            query=query,
            key=key,
            weights=weights,
            actual_seq_lengths_query=seq_q,
            actual_seq_lengths_key=seq_k,
            block_table=block_table,
            layout_query=LAYOUT_QUERY,
            layout_key=LAYOUT_KEY,
            sparse_count=SPARSE_COUNT,
            sparse_mode=SPARSE_MODE,
        )
    torch.npu.synchronize()
    print("=======pass=========")


if __name__ == "__main__":
    import sys

    if "--perf" in sys.argv:
        run_perf()
    else:
        pytest.main([__file__, "-sv"])
