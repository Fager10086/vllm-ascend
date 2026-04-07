import torch
import triton
import triton.language as tl
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

init_device_properties_triton()

_CORE_NUM = get_vectorcore_num()
_GRID = (_CORE_NUM, )

# UB capacity planning:
# Per-block UB peak = x(BLOCK*4) + y(BLOCK*4) + output(BLOCK*4) = 12*BLOCK bytes
# BLOCK_SIZE=1024: 12*1024 = 12288 bytes per block
# N_BLOCKS = 85*1024 // 12288 = 7 blocks per outer loop iteration
# Elements per outer iteration: 7 * 1024 = 7168
_BLOCK_SIZE = 1024
_N_BLOCKS = 85 * 1024 // (12 * _BLOCK_SIZE)  # = 7
_ELEMS_PER_ITER = _N_BLOCKS * _BLOCK_SIZE  # = 7168


@triton.jit(do_not_specialize=["n_elements"])
def add_kernel(x_ptr,
               y_ptr,
               output_ptr,
               n_elements,
               BLOCK_SIZE: tl.constexpr,
               N_BLOCKS: tl.constexpr,
               ELEMS_PER_ITER: tl.constexpr,
               ):
    pid = tl.program_id(axis=0)
    num_cores = tl.num_programs(0)

    # Reduced number of outer loop iterations
    reduced_loops = tl.cdiv(n_elements, ELEMS_PER_ITER)

    for outer_idx in range(pid, reduced_loops, num_cores):
        base = outer_idx * ELEMS_PER_ITER

        # Inner loop: process N_BLOCKS blocks per iteration
        for b in range(N_BLOCKS):
            start = base + b * BLOCK_SIZE
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            y = tl.load(y_ptr + offsets, mask=mask)
            out = x + y
            tl.store(output_ptr + offsets, out, mask=mask)


def add(x: torch.Tensor, y: torch.Tensor):
    output = torch.empty_like(x)
    n_elements = x.numel()
    add_kernel[_GRID](x, y, output, n_elements,
                      BLOCK_SIZE=_BLOCK_SIZE,
                      N_BLOCKS=_N_BLOCKS,
                      ELEMS_PER_ITER=_ELEMS_PER_ITER)
    return output
