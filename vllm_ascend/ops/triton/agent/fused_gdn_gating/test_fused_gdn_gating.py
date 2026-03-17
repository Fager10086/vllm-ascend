import gc
import torch
import pytest
from vllm.model_executor.models.qwen3_next import fused_gdn_gating
import vllm_ascend.patch.worker.patch_triton
from vllm_ascend.ops.triton.fused_gdn_gating import fused_gdn_gating_patch
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

BTACH = [1, 7168, 64*4096, 10]
NUM_V_HEADS = [8, 32, 128, 16]
SEEDS = [0]
DEVICES = [f"npu:{0}"]
DTYPES = [torch.bfloat16, torch.float32]
DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3

# -----------------------------
# PyTorch Reference 实现
# -----------------------------
def fused_gdn_gating_ref(A_log, a, b, dt_bias, beta=1.0, threshold=20.0):
    """
    A_log: (num_heads,)
    a: (batch, num_heads)
    b: (batch, num_heads)
    dt_bias: (num_heads,)
    """
    x = a.float() + dt_bias.float().unsqueeze(0)

    softplus_x = torch.where(
        beta * x <= threshold,
        (1.0 / beta) * torch.log1p(torch.exp(beta * x)),
        x,
    )

    g = -torch.exp(A_log.float()).unsqueeze(0) * softplus_x
    beta_output = torch.sigmoid(b.float())

    return g, beta_output


# -----------------------------
# 单测函数
# -----------------------------
@pytest.mark.parametrize("batch", BTACH)
@pytest.mark.parametrize("num_heads", NUM_V_HEADS)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("device", DEVICES)
@torch.inference_mode()
def test_fused_gdn_gating(batch, seed, num_heads, dtype, device):

    torch.manual_seed(seed)
    torch.set_default_device(device)
    init_device_properties_triton()

    A_log = torch.randn(num_heads, dtype=dtype, device=device)
    a = torch.randn(batch, num_heads, dtype=dtype, device=device)
    b = torch.randn(batch, num_heads, dtype=dtype, device=device)
    dt_bias = torch.randn(num_heads, dtype=dtype, device=device)

    # 被测算子
    g_out, beta_out = fused_gdn_gating_patch(
        A_log=A_log,
        a=a,
        b=b,
        dt_bias=dt_bias,
    )

    # 兼容 shape: (1, B, H) or (B, H)
    if g_out.dim() == 3:
        g_out = g_out.squeeze(0)
    if beta_out.dim() == 3:
        beta_out = beta_out.squeeze(0)

    # reference
    g_ref, beta_ref = fused_gdn_gating_ref(A_log, a, b, dt_bias)

    # dtype 对齐
    g_ref = g_ref.to(g_out.dtype)
    beta_ref = beta_ref.to(beta_out.dtype)

    torch.testing.assert_close(g_out,
                               g_ref,
                               rtol=DEFAULT_RTOL,
                               atol=DEFAULT_ATOL,
                               equal_nan=True)
    
    torch.testing.assert_close(beta_out,
                               beta_ref,
                               rtol=DEFAULT_RTOL,
                               atol=DEFAULT_ATOL,
                               equal_nan=True)
    gc.collect()
    torch.npu.empty_cache()
    torch.npu.reset_peak_memory_stats()


# # -----------------------------
# # 主入口
# # -----------------------------
if __name__ == "__main__":
    test_fused_gdn_gating(
        num_heads=16,
        batch=10,
        seed=0,
        dtype=torch.float32,
        device=f"npu:{0}",
    )
