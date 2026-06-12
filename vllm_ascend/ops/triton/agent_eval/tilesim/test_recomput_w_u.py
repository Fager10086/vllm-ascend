import time
import numpy as np
from typing import Optional, Tuple

import torch
from vllm_ascend.ops.triton.fla.wy_fast import recompute_w_u_fwd
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

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
    
    # if verbose:
    #     print(f"\n{'='*60}")
    #     print(f"测试配置:")
    #     print(f"  B={B}, T={T}, H={H}, Hg={Hg}, K={K}, V={V}, BT={BT}")
    #     print(f"  dtype={dtype}, device={device}, varlen={test_varlen}")
    #     print(f"{'='*60}")
    
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
    
    # if verbose:
    #     print(f"Triton 计算完成，耗时: {elapsed*1000:.2f}ms")
    #     print(f"w 形状: {w_triton.shape}")
    #     print(f"u 形状: {u_triton.shape}")
    
    # 创建参考实现（简化验证）
    # 注意：由于计算复杂，这里只验证形状和基本数值特性
    passed = True
    
    # 1. 验证形状
    expected_w_shape = (B, T_total, H, K) if not test_varlen else (B, T_total, H, K)
    expected_u_shape = (B, T_total, H, V) if not test_varlen else (B, T_total, H, V)
    
    if w_triton.shape != expected_w_shape:
        # print(f"❌ w 形状错误: 期望 {expected_w_shape}, 实际 {w_triton.shape}")
        passed = False
    
    if u_triton.shape != expected_u_shape:
        # print(f"❌ u 形状错误: 期望 {expected_u_shape}, 实际 {u_triton.shape}")
        passed = False
    
    # 2. 验证数值范围（简化检查）
    if torch.any(torch.isnan(w_triton)):
        # print("❌ w 包含 NaN")
        passed = False
    
    if torch.any(torch.isnan(u_triton)):
        # print("❌ u 包含 NaN")
        passed = False
    
    if torch.any(torch.isinf(w_triton)):
        # print("❌ w 包含 Inf")
        passed = False
    
    if torch.any(torch.isinf(u_triton)):
        # print("❌ u 包含 Inf")
        passed = False
    
    # if verbose and passed:
        # print("✅ 测试通过")
    
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
        (1, 512, 16, 8, 128, 128, 64, torch.bfloat16, "npu", True),

    ]
    
    # if not torch.cuda.is_available():
    #     print("⚠️ CUDA 不可用，将只在 CPU 上测试")
    #     # 修改配置为 CPU
    #     test_configs = [(B, T, H, Hg, K, V, BT, dtype, "cpu", varlen) 
    #                    for B, T, H, Hg, K, V, BT, dtype, _, varlen in test_configs]
    
    # print("🧪 开始多形状测试")
    # print("=" * 80)
    
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
            
            # print(f"{status} | 耗时: {elapsed*1000:.2f}ms")
            
        except Exception as e:
            print(f"❌ 异常: {e}")
            results.append({
                "config": config,
                "passed": False,
                "error": str(e)
            })
            total_tests += 1
    
    # # 统计结果
    # print("\n" + "=" * 80)
    # print("📊 测试总结")
    # print("=" * 80)
    # print(f"总测试数: {total_tests}")
    # print(f"通过数: {total_passed}")
    # print(f"失败数: {total_tests - total_passed}")
    # print(f"通过率: {total_passed/total_tests*100:.1f}%")
    
    # if timings:
    #     print(f"平均耗时: {np.mean(timings):.2f}ms")
    #     print(f"最小耗时: {np.min(timings):.2f}ms")
    #     print(f"最大耗时: {np.max(timings):.2f}ms")
    
    return results

def main():
    # """
    # 主测试函数
    # """
    # print("🧪 recompute_w_u_fwd 算子测试套件")
    # print("=" * 80)
    
    # # 1. 测试多种形状
    # print("\n📋 测试 1: 多形状测试")
    init_device_properties_triton()
    results = test_multiple_shapes()

    
    # print("\n" + "=" * 80)
    # print("🎉 所有测试完成")

if __name__ == "__main__":
    main()
