import time
import numpy as np
import torch
from typing import Optional, Tuple
from vllm_ascend.ops.triton.fla.chunk_delta_h import chunk_gated_delta_rule_fwd_h

def test_performance_with_given_shape():

    # shape_str = "1,10288,2,128;1,10288,8,128;1,10288,8,128;1,8,10288"
    # shape_str = "1,10012,4,128;1,10012,16,128;1,10012,16,128;1,16,10012"
    shape_str = "1,10016,2,128;1,10016,8,128;1,10012,8,128;1,8,10016"
    shapes = [list(map(int, s.split(','))) for s in shape_str.split(';')]

    B, T, Hg, K = shapes[0]
    _, _, H, _ = shapes[1]   # w 的 H
    _, _, _, V = shapes[2]   # u 的 V
    assert shapes[1][2] == H and shapes[2][2] == H
    assert shapes[1][3] == K and shapes[2][3] == V

    chunk_size = 128 
    dtype = torch.bfloat16
    device = 'npu'

    torch.manual_seed(42)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)
    
    g_raw = torch.randn(B, H, T, device=device, dtype=torch.float32)
    g = g_raw.transpose(1, 2) 

    initial_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    for _ in range(20):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            save_new_value=True,
            cu_seqlens=None
        )
    torch.npu.synchronize()
    
    start_time = time.time()
    for _ in range(20):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            save_new_value=True,
            cu_seqlens=None
        )
    
    torch.npu.synchronize() if device == "npu" else None
    elapsed = (time.time() - start_time) / 20
    print(f"Task Duration: {elapsed*1000:.2f}ms")


if __name__ == "__main__":
    test_performance_with_given_shape()
