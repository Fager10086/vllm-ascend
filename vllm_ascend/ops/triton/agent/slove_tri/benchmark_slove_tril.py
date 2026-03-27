import torch
import numpy as np
from dataclasses import dataclass
import triton.testing
from vllm_ascend.ops.triton.fla.solve_tril import (
    solve_tril
)
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton

@dataclass
class BenchConfig:
    B: int
    T: int
    H: int
    Hg: int
    K: int
    V: int
    BT: int          # chunk size
    dtype: torch.dtype
    device: str
    varlen: bool
    cu_seqlens: list[int] = None
    
configs = [
    BenchConfig(
        1, 10012, 4, 16, 128, 128,
        64,
        torch.float16,
        "npu",
        True,
        None,
    ),

    BenchConfig(
        1, 10012, 4, 16, 128, 128,
        64,
        torch.bfloat16,
        "npu",
        True,
        None,
    ),
    BenchConfig(
        1, 5, 16, 16, 128, 128,
        64,
        torch.bfloat16,
        "npu",
        True,
        [0,5]
    ),
]

chunk_size = 64
device = "npu"

def build_inputs(cfg: BenchConfig):

    B, T, H, Hg, K, V, BT, dtype, device, varle,cu_seqlens = (
        cfg.B, cfg.T, cfg.H, cfg.Hg, cfg.K, cfg.V, cfg.BT, cfg.dtype, cfg.device, cfg.varlen, cfg.cu_seqlens)
    
    init_device_properties_triton()
    if varle:
        seqlens = []
        for _ in range(B):
            seq_len = torch.randint(T//2, T+1, (1,)).item()
            seqlens.append(seq_len)
        if cu_seqlens is None:
            cu_seqlens = torch.tensor([0] + np.cumsum(seqlens).tolist(), 
                                    dtype=torch.int64, device=device)
        else:
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)
        T_total = cu_seqlens[-1].item()
        
        k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device)
        v = torch.randn(B, T_total, H, V, dtype=dtype, device=device)
        beta = torch.randn(B, T_total, H, dtype=dtype, device=device)
        g_cumsum = torch.randn(B, T_total, H, dtype=dtype, device=device)
        A = torch.randn(B, T_total, H, BT, dtype=dtype, device=device)
    return A, cu_seqlens, dtype

def benchmark_one(cfg: BenchConfig):

    A, cu_seqlens, dtype = build_inputs(cfg)

    def run():
        Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=dtype)

    torch.npu.synchronize()
    torch.npu.empty_cache()

    ms, min_ms, max_ms = triton.testing.do_bench(
        run,
        warmup=50,
        rep=200,
        quantiles=[0.5, 0.2, 0.8],
    )

    return {
        "cfg": cfg,
        "dtype": str(dtype).split(".")[-1],
        "Task Duration": ms,
    }


def benchmark_all():

    results = []

    for cfg in configs:
        result = benchmark_one(cfg)
        results.append(result)

    for r in results:
        c = r["cfg"]
        print(
            f"B={c.B} T={c.T} "
            f"H={c.H} Hg={c.Hg} "
            f"dtype={c.dtype} "
            f"varlen={c.varlen} | "
            f"Task Duration:{r['Task Duration']:.2f} ms | "
        )


if __name__ == "__main__":
    benchmark_all()
