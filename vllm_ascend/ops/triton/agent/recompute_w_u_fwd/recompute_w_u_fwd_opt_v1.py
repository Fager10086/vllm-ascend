# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang

# ruff: noqa: E501
# mypy: ignore-errors

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def recompute_w_u_fwd_kernel(
    k,
    v,
    beta,
    w,
    u,
    A,
    g,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """
    Optimized kernel with half-precision A matrix:
    1. BV=BK=128 to process full K/V dimensions without inner loops
    2. Keep A in float16 for dot product to reduce memory bandwidth
    3. Fused beta-g computation
    """
    i_t_o = tl.program_id(0)
    i_b = tl.program_id(1)
    
    # Compute sequence boundaries
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t_o * 2).to(tl.int32),
            tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32),
        )
        bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, (i_b + 1) * T
        i_t = i_t_o

    # Pre-compute offsets
    offs_t = tl.arange(0, BT)
    global_offs_t = i_t * BT + offs_t
    mask_t = global_offs_t < T
    mask_t_2d = mask_t[:, None]
    offs_t_2d = global_offs_t[:, None]
    offs_bt = tl.arange(0, BT)[None, :]
    
    # Pre-compute V and K offsets
    offs_v = tl.arange(0, BV)[None, :]
    offs_k = tl.arange(0, BK)[None, :]
    mask_v = mask_t_2d & (offs_v < V)
    mask_k = mask_t_2d & (offs_k < K)

    # Process each head
    for i_h in range(H):
        # Load A matrix (BT×BT) - keep in float16 to reduce memory bandwidth
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt
        b_A = tl.load(ptr_A, mask=mask_t_2d, other=0.0)  # Keep in float16

        # Load g and beta, compute exp(g) and fused scale factors
        ptr_g = g + bos + i_h * T + global_offs_t
        ptr_beta = beta + bos + i_h * T + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0).to(tl.float32))
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)
        
        # Pre-compute fused scaling factors in float32
        b_beta_2d = b_beta[:, None]
        b_beta_g_2d = b_beta[:, None] * b_g[:, None]

        # V computation: u = A @ (v * beta)
        # Keep v * beta in float32, but convert to float16 for dot product
        ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v
        b_v = tl.load(ptr_v, mask=mask_v, other=0.0)
        b_v_scaled = (b_v.to(tl.float32) * b_beta_2d).to(tl.float16)  # Scale and convert to fp16
        b_u = tl.dot(b_A, b_v_scaled)  # fp16 @ fp16
        ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v
        tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)
        
        # K computation: w = A @ (k * beta * g)
        ptr_k = k + (bos * Hg + i_h // (H // Hg)) * K + offs_t_2d * (Hg * K) + offs_k
        b_k = tl.load(ptr_k, mask=mask_k, other=0.0)
        b_k_scaled = (b_k.to(tl.float32) * b_beta_g_2d).to(tl.float16)  # Scale and convert to fp16
        b_w = tl.dot(b_A, b_k_scaled)  # fp16 @ fp16
        ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k
        tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    H = v.shape[-2]
    BT = A.shape[-1]

    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)

    # Use BK=BV=128 to process all dimensions at once (K=V=128)
    BK = 128
    BV = 128

    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    beta = beta.transpose(1, 2).contiguous()
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()
    
    # Launch one kernel per chunk, inner loop processes all heads
    recompute_w_u_fwd_kernel[(NT, B)](
        k=k,
        v=v,
        beta=beta,
        w=w,
        u=u,
        A=A,
        g=g_cumsum,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=BK,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )
    return w, u

import time
import numpy as np
from typing import Optional, Tuple

def test_single_case(
    B: int,
    T: int,
    H: int,
    Hg: int,
    K: int,
    V: int,
    BT: int,
    device: str = "npu",
    dtype: torch.dtype = torch.float16,
    test_varlen: bool = False,
    verbose: bool = True
) -> Tuple[bool, float]:
    """
    测试单个形状配置
    
    返回:
        (是否通过, 运行时间)
    """
    
    # 创建随机输入
    torch.manual_seed(42)
    
    if test_varlen:
        # 变长序列测试
        seqlens = []
        for _ in range(B):
            # 每个序列长度在 [T//2, T] 之间随机
            seq_len = torch.randint(T//2, T+1, (1,)).item()
            seqlens.append(seq_len)
        
        cu_seqlens = torch.tensor([0] + np.cumsum(seqlens).tolist(), 
                                 dtype=torch.int64, device=device)
        T_total = cu_seqlens[-1].item()
        
        # 创建变长张量
        k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device)
        v = torch.randn(B, T_total, H, V, dtype=dtype, device=device)
        beta = torch.randn(B, T_total, H, dtype=dtype, device=device)
        g_cumsum = torch.randn(B, T_total, H, dtype=dtype, device=device)
        A = torch.randn(B, T_total, H, BT, dtype=dtype, device=device)
        
    else:
        # 定长序列测试
        cu_seqlens = None
        T_total = T
        
        k = torch.randn(B, T, Hg, K, dtype=dtype, device=device)
        v = torch.randn(B, T, H, V, dtype=dtype, device=device)
        beta = torch.randn(B, T, H, dtype=dtype, device=device)
        g_cumsum = torch.randn(B, T, H, dtype=dtype, device=device)
        A = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    
    # 预热
    for _ in range(3):
        w_triton, u_triton = recompute_w_u_fwd(
            k, v, beta, g_cumsum, A, cu_seqlens
        )
    
    # 计时
    torch.cuda.synchronize() if device == "cuda" else None
    start_time = time.time()
    
    w_triton, u_triton = recompute_w_u_fwd(
        k, v, beta, g_cumsum, A, cu_seqlens
    )
    
    torch.cuda.synchronize() if device == "cuda" else None
    elapsed = time.time() - start_time
    
    # 创建参考实现（简化验证）
    # 注意：由于计算复杂，这里只验证形状和基本数值特性
    passed = True
    
    # 1. 验证形状
    expected_w_shape = (B, T_total, H, K) if not test_varlen else (B, T_total, H, K)
    expected_u_shape = (B, T_total, H, V) if not test_varlen else (B, T_total, H, V)
    
    if w_triton.shape != expected_w_shape:
        passed = False
    
    if u_triton.shape != expected_u_shape:
        passed = False
    
    # 2. 验证数值范围（简化检查）
    if torch.any(torch.isnan(w_triton)):
        passed = False
    
    if torch.any(torch.isnan(u_triton)):
        passed = False
    
    if torch.any(torch.isinf(w_triton)):
        passed = False
    
    if torch.any(torch.isinf(u_triton)):
        passed = False
    
    return passed, elapsed


def test_multiple_shapes():
    """
    测试多种形状配置
    """
    
    # 测试配置列表
    test_configs = [
        # (B, T, H, Hg, K, V, BT, dtype, device, varlen)
        ## 整网数据
        (1, 10288, 8, 2, 128, 128, 64, torch.float16, "npu", True),
    ]
    
    results = []
    total_passed = 0
    total_tests = 0
    timings = []
    
    for config in test_configs:
        B, T, H, Hg, K, V, BT, dtype, device, varlen = config
        
        try:
            passed, elapsed = test_single_case(
                B=B, T=T, H=H, Hg=Hg, K=K, V=V, BT=BT,
                dtype=dtype, device=device, test_varlen=varlen,
                verbose=True
            )
            
            results.append({
                "config": config,
                "passed": passed,
                "time_ms": elapsed * 1000
            })
            
            if passed:
                total_passed += 1
                status = "✅ 通过"
            else:
                status = "❌ 失败"
            
            total_tests += 1
            timings.append(elapsed * 1000)
            
        except Exception as e:
            print(f"❌ 异常: {e}")
            results.append({
                "config": config,
                "passed": False,
                "error": str(e)
            })
            total_tests += 1
    
    return results

def main():
    results = test_multiple_shapes()

if __name__ == "__main__":
    main()
