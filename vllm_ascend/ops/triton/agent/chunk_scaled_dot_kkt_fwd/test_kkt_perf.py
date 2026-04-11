# Performance test for chunk_scaled_dot_kkt (msprof compatible)
import sys
sys.path.insert(0, '/vllm-workspace/vllm-ascend')

import torch
import torch_npu

from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

# Test configuration - production-scale
B = 2
T = 1024
H = 32
Hg = 32
K = 128
BT = 64
dtype = torch.float16
device = 'npu'

torch.manual_seed(42)
k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
beta = torch.randn(B, T, H, device=device, dtype=dtype)
g_cumsum = -torch.abs(torch.randn(B, T, H, device=device, dtype=dtype))

# Warmup (10 iterations)
for _ in range(10):
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum,
        cu_seqlens=None, chunk_indices=None,
        chunk_size=BT, output_dtype=torch.float32,
    )
torch.npu.synchronize()

# Measure (20 iterations for msprof to capture)
torch.npu.synchronize()
for _ in range(20):
    A = chunk_scaled_dot_kkt_fwd(
        k=k, beta=beta, g_cumsum=g_cumsum,
        cu_seqlens=None, chunk_indices=None,
        chunk_size=BT, output_dtype=torch.float32,
    )
torch.npu.synchronize()
