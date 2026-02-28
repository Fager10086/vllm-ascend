import gc
import time

import pytest
import torch
import torch_npu

import vllm_ascend.ops.triton.linearnorm.split_qkv_rmsnorm_mrope_optimized  # noqa
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

NUM_TOKENS = [1, 4, 8, 16, 1024, 4096]
NUM_QKV_HEADS = [(8, 2), (2, 1), (16, 2)]
HEAD_SIZES = [128, 256]
EPS = [1e-6]
MROPE_SECTION = [[11, 11, 10], [24, 20, 20]]
IS_INTERLEAVED = [True, False]
DTYPES = [torch.bfloat16]
DEVICES = [f"npu:{0}"]


@pytest.mark.parametrize("num_tokens", NUM_TOKENS)
@pytest.mark.parametrize("num_q_heads, num_kv_heads", NUM_QKV_HEADS)
@pytest.mark.parametrize("head_size", HEAD_SIZES)
@pytest.mark.parametrize("eps", EPS)
@pytest.mark.parametrize("mrope_section", MROPE_SECTION)
@pytest.mark.parametrize("is_interleaved", IS_INTERLEAVED)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_split_qkv_rmsnorm_mrope(
    num_tokens, num_q_heads, num_kv_heads, head_size,
    mrope_section, eps, dtype, device, is_interleaved,
):
    torch.set_default_device(device)
    init_device_properties_triton()
    rope_dim = 2 * sum(mrope_section)
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    qkv = torch.randn(num_tokens, q_size + kv_size * 2, dtype=dtype, device=device)
    q_weight = torch.randn(head_size, dtype=dtype, device=device)
    k_weight = torch.randn(head_size, dtype=dtype, device=device)
    q_bias = None
    k_bias = None
    cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=dtype, device=device)

    common_kwargs = dict(
        qkv=qkv, q_weight=q_weight, k_weight=k_weight, cos_sin=cos_sin,
        num_q_heads=num_q_heads, num_kv_heads=num_kv_heads, head_size=head_size,
        eps=eps, mrope_section=mrope_section, is_interleaved=is_interleaved,
        rope_dim=rope_dim,
    )

    num_warm_up = 20
    num_runs = 20

    for _ in range(num_warm_up):
        torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(**common_kwargs)
    torch.npu.synchronize()
    gc.collect()
    torch.npu.empty_cache()

    start = time.perf_counter()
    for _ in range(num_runs):
        torch.ops.vllm.triton_split_qkv_rmsnorm_mrope(**common_kwargs)
    torch.npu.synchronize()
    end = time.perf_counter()
    avg_ms = (end - start) / num_runs * 1000

    print(f"Average latency over {num_runs} runs: {avg_ms:.3f} ms")

    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


if __name__ == "__main__":
    test_split_qkv_rmsnorm_mrope(
        num_tokens=4096,
        num_q_heads=2,
        num_kv_heads=1,
        head_size=256,
        mrope_section=[11, 11, 10],
        is_interleaved=True,
        eps=1e-6,
        dtype=torch.bfloat16,
        device="npu",
    )
