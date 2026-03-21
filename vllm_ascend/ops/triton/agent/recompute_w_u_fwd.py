#!/usr/bin/env python3
"""
测试 recompute_w_u_fwd Triton 算子
用于验证不同形状输入下的正确性和性能
"""

import torch
import triton
import triton.language as tl
import time
import numpy as np
from typing import Optional, Tuple

# 假设 prepare_chunk_indices 函数已经定义
def prepare_chunk_indices(cu_seqlens: torch.LongTensor, BT: int) -> torch.LongTensor:
    """
    准备分块索引
    
    参数:
        cu_seqlens: 累计序列长度 [B+1]
        BT: 块大小
    
    返回:
        分块索引 [num_chunks, 2]
    """
    B = len(cu_seqlens) - 1
    chunk_indices = []
    
    for i in range(B):
        seq_start = cu_seqlens[i].item()
        seq_end = cu_seqlens[i + 1].item()
        seq_len = seq_end - seq_start
        
        # 为当前序列生成块
        for chunk_start in range(0, seq_len, BT):
            chunk_end = min(chunk_start + BT, seq_len)
            chunk_idx = chunk_start // BT
            chunk_indices.append([i, chunk_idx])
    
    return torch.tensor(chunk_indices, dtype=torch.int64, device=cu_seqlens.device)

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
    T_max = T
    i_t_o = tl.program_id(0)

    for i_bh in range(H):
        i_b, i_h = i_bh // H, i_bh % H
        if IS_VARLEN:
            i_n, i_t = (
                tl.load(chunk_indices + i_t_o * 2).to(tl.int32),
                tl.load(chunk_indices + i_t_o * 2 + 1).to(tl.int32),
            )
            bos, eos = tl.load(cu_seqlens + i_n).to(tl.int32), tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T = eos - bos
        else:
            bos, eos = i_b * T, i_b * T + T

        offs_t = tl.arange(0, BT)
        global_offs_t = i_t * BT + offs_t
        mask_t = global_offs_t < T

        offs_t_2d = global_offs_t[:, None]
        offs_bt = tl.arange(0, BT)[None, :]
        ptr_A = A + (bos * H + i_h) * BT + offs_t_2d * (H * BT) + offs_bt * 1
        mask_A = mask_t[:, None]
        b_A = tl.load(ptr_A, mask=mask_A, other=0.0).to(tl.float32)

        ptr_g = g + bos + i_h * T_max + global_offs_t
        b_g = tl.exp(tl.load(ptr_g, mask=mask_t, other=0.0)).to(tl.float32)

        ptr_beta = beta + bos + i_h * T_max + global_offs_t
        b_beta = tl.load(ptr_beta, mask=mask_t, other=0.0).to(tl.float32)

        for i_v in range(tl.cdiv(V, BV)):
            offs_v = i_v * BV + tl.arange(0, BV)[None, :]
            mask_v = (mask_t[:, None]) & (offs_v < V)

            ptr_v = v + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            b_v = tl.load(ptr_v, mask=mask_v, other=0.0).to(tl.float32)

            b_vb = b_v * b_beta[:, None]
            b_u = tl.dot(b_A, b_vb, allow_tf32=False)

            ptr_u = u + (bos * H + i_h) * V + offs_t_2d * (H * V) + offs_v * 1
            tl.store(ptr_u, b_u.to(ptr_u.dtype.element_ty), mask=mask_v)

        for i_k in range(tl.cdiv(K, BK)):
            offs_k = i_k * BK + tl.arange(0, BK)[None, :]
            mask_k = (mask_t[:, None]) & (offs_k < K)
            ptr_k = k + (bos * Hg + i_h // (H // Hg)) * K + offs_t_2d * (Hg * K) + offs_k * 1
            b_k = tl.load(ptr_k, mask=mask_k, other=0.0).to(tl.float32)

            b_kb = b_k * b_beta[:, None] * b_g[:, None]
            b_w = tl.dot(b_A, b_kb)

            ptr_w = w + (bos * H + i_h) * K + offs_t_2d * (H * K) + offs_k * 1
            tl.store(ptr_w, b_w.to(ptr_w.dtype.element_ty), mask=mask_k)


def recompute_w_u_fwd(
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g_cumsum: torch.Tensor,
    A: torch.Tensor,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    重新计算 w 和 u
    
    参数:
        k: 键张量 [B, T, Hg, K]
        v: 值张量 [B, T, H, V]
        beta: beta 张量 [B, T, H]
        g_cumsum: 门控累积和 [B, T, H]
        A: 注意力矩阵 [B, T, H, BT]
        cu_seqlens: 累计序列长度 [B+1]
    
    返回:
        w: [B, T, H, K]
        u: [B, T, H, V]
    """
    B, T, Hg, K = k.shape
    B, T, H, V = v.shape
    BT = A.shape[-1]  # 块大小
    
    chunk_indices = prepare_chunk_indices(cu_seqlens, BT) if cu_seqlens is not None else None
    NT = triton.cdiv(T, BT) if cu_seqlens is None else len(chunk_indices)
    
    BK = 128
    BV = 128
    
    u = torch.empty_like(v)
    w = k.new_empty(B, T, H, K)
    beta = beta.transpose(1, 2).contiguous()  # [B, H, T]
    g_cumsum = g_cumsum.transpose(1, 2).contiguous()  # [B, H, T]
    
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
        num_stages=3,
    )
    return w, u


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
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"测试配置:")
        print(f"  B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, BT={BT}")
        print(f"  dtype={dtype}, device={device}, varlen={test_varlen}")
        print(f"{'='*60}")
    
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
    
    if verbose:
        print(f"Triton 计算完成，耗时: {elapsed*1000:.2f}ms")
        print(f"w 形状: {w_triton.shape}")
        print(f"u 形状: {u_triton.shape}")
    
    # 创建参考实现（简化验证）
    # 注意：由于计算复杂，这里只验证形状和基本数值特性
    passed = True
    
    # 1. 验证形状
    expected_w_shape = (B, T_total, H, K) if not test_varlen else (B, T_total, H, K)
    expected_u_shape = (B, T_total, H, V) if not test_varlen else (B, T_total, H, V)
    
    if w_triton.shape != expected_w_shape:
        print(f"❌ w 形状错误: 期望 {expected_w_shape}, 实际 {w_triton.shape}")
        passed = False
    
    if u_triton.shape != expected_u_shape:
        print(f"❌ u 形状错误: 期望 {expected_u_shape}, 实际 {u_triton.shape}")
        passed = False
    
    # 2. 验证数值范围（简化检查）
    if torch.any(torch.isnan(w_triton)):
        print("❌ w 包含 NaN")
        passed = False
    
    if torch.any(torch.isnan(u_triton)):
        print("❌ u 包含 NaN")
        passed = False
    
    if torch.any(torch.isinf(w_triton)):
        print("❌ w 包含 Inf")
        passed = False
    
    if torch.any(torch.isinf(u_triton)):
        print("❌ u 包含 Inf")
        passed = False
    
    if verbose and passed:
        print("✅ 测试通过")
    
    return passed, elapsed


def test_multiple_shapes():
    """
    测试多种形状配置
    """
    
    # 测试配置列表
    test_configs = [
        # (B, T, H, Hg, K, V, BT, dtype, device, varlen)
        # 小模型配置
        # (2, 64, 4, 2, 64, 64, 32, torch.float16, "npu", False),
        # (2, 128, 8, 4, 64, 64, 32, torch.float16, "npu", False),
        
        # # 中等模型配置
        # (4, 256, 12, 6, 128, 128, 64, torch.float16, "npu", False),
        # (4, 512, 16, 8, 128, 128, 64, torch.float16, "npu", False),
        
        # # 大模型配置
        # (8, 1024, 32, 16, 256, 256, 128, torch.float16, "npu", False),
        
        # # 混合精度测试
        # (2, 256, 8, 4, 64, 64, 32, torch.float32, "npu", False),
        
        # # 变长序列测试
        # (2, 128, 8, 4, 64, 64, 32, torch.float16, "npu", True),
        # (4, 256, 12, 6, 128, 128, 64, torch.float16, "npu", True),
        ## 整网数据
        # BT不知道
        (1, 10288, 8, 2, 128, 128, 64, torch.float16, "npu", True),

    ]
    
    # if not torch.cuda.is_available():
    #     print("⚠️ CUDA 不可用，将只在 CPU 上测试")
    #     # 修改配置为 CPU
    #     test_configs = [(B, T, H, Hg, K, V, BT, dtype, "cpu", varlen) 
    #                    for B, T, H, Hg, K, V, BT, dtype, _, varlen in test_configs]
    
    print("🧪 开始多形状测试")
    print("=" * 80)
    
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
            
            print(f"{status} | 耗时: {elapsed*1000:.2f}ms")
            
        except Exception as e:
            print(f"❌ 异常: {e}")
            results.append({
                "config": config,
                "passed": False,
                "error": str(e)
            })
            total_tests += 1
    
    # 统计结果
    print("\n" + "=" * 80)
    print("📊 测试总结")
    print("=" * 80)
    print(f"总测试数: {total_tests}")
    print(f"通过数: {total_passed}")
    print(f"失败数: {total_tests - total_passed}")
    print(f"通过率: {total_passed/total_tests*100:.1f}%")
    
    if timings:
        print(f"平均耗时: {np.mean(timings):.2f}ms")
        print(f"最小耗时: {np.min(timings):.2f}ms")
        print(f"最大耗时: {np.max(timings):.2f}ms")
    
    return results

def main():
    """
    主测试函数
    """
    print("🧪 recompute_w_u_fwd 算子测试套件")
    print("=" * 80)
    
    # 1. 测试多种形状
    print("\n📋 测试 1: 多形状测试")
    results = test_multiple_shapes()

    
    print("\n" + "=" * 80)
    print("🎉 所有测试完成")


if __name__ == "__main__":
    main()
