import time
import torch
import numpy as np
from typing import Optional, Tuple
from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import (
    chunk_scaled_dot_kkt_fwd,
)


# ==================== 执行调用 ====================
if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, H, Hg, K, V, BT, dtype, device, varle = (1, 10288, 8, 2, 128, 128, 64, torch.float16, "npu", True)
    
    if varle:
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
    
    # 预热
    for _ in range(20):
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    
    # 计时
    torch.npu.synchronize() if device == "npu" else None
    start_time = time.time()
    for _ in range(20):
        A = chunk_scaled_dot_kkt_fwd(k=k, beta=beta, g_cumsum=g_cumsum, cu_seqlens=cu_seqlens, output_dtype=torch.float32)
    
    torch.npu.synchronize() if device == "npu" else None
    elapsed = time.time() - start_time

    print("pass")
