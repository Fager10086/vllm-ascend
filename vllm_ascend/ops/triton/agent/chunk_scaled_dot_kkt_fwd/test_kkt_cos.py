# Correctness verification for chunk_scaled_dot_kkt
import sys
sys.path.insert(0, '/vllm-workspace/vllm-ascend')

import torch
import torch_npu

from vllm_ascend.ops.triton.fla.chunk_scaled_dot_kkt import chunk_scaled_dot_kkt_fwd

def reference_impl(k, beta, g_cumsum, chunk_size=64):
    """Pure PyTorch reference implementation for verification."""
    B, T, Hg, K = k.shape
    H = beta.shape[-1]
    BT = chunk_size
    NT = (T + BT - 1) // BT

    # permute beta and g_cumsum to [H, B, T]
    beta_p = beta.permute(2, 0, 1).contiguous()
    g_p = g_cumsum.permute(2, 0, 1).contiguous()

    A = torch.zeros(B, T, H, BT, device=k.device, dtype=torch.float32)

    for i_b in range(B):
        for i_h in range(H):
            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min(t_start + BT, T)
                actual_bt = t_end - t_start

                # Load k block [actual_bt, K]
                hg_idx = i_h // (H // Hg)
                k_block = k[i_b, t_start:t_end, hg_idx, :].float()

                # Compute K * K^T
                a_block = torch.mm(k_block, k_block.t())

                # Apply gating
                g_block = g_p[i_h, i_b, t_start:t_end].float()
                g_diff = g_block[:, None] - g_block[None, :]
                g_diff = torch.where(g_diff <= 0, g_diff, torch.tensor(float('-inf'), device=k.device))
                a_block = a_block * torch.exp(g_diff)

                # Apply beta
                b_beta = beta_p[i_h, i_b, t_start:t_end].float()
                a_block = a_block * b_beta[:, None]

                # Apply causal mask
                idx = torch.arange(actual_bt, device=k.device).float()
                mask = idx[:, None] > idx[None, :]
                a_block = torch.where(mask, a_block, torch.zeros_like(a_block))

                A[i_b, t_start:t_end, i_h, :actual_bt] = a_block

    return A


def test_correctness():
    torch.manual_seed(42)
    test_configs = [
        # (B, T, H, Hg, K, BT)
        (1, 64, 4, 4, 64, 64),
        (2, 128, 8, 8, 128, 64),
        (2, 1024, 32, 32, 128, 64),
        (1, 256, 16, 4, 64, 64),   # GQA: H != Hg
    ]

    for B, T, H, Hg, K, BT in test_configs:
        print(f"Testing B={B}, T={T}, H={H}, Hg={Hg}, K={K}, BT={BT}...", end=" ")

        k = torch.randn(B, T, Hg, K, device='npu', dtype=torch.float16)
        beta = torch.randn(B, T, H, device='npu', dtype=torch.float16)
        g_cumsum = -torch.abs(torch.randn(B, T, H, device='npu', dtype=torch.float16))

        # Reference
        ref = reference_impl(k, beta, g_cumsum, chunk_size=BT)

        # Triton
        out = chunk_scaled_dot_kkt_fwd(
            k=k, beta=beta, g_cumsum=g_cumsum,
            cu_seqlens=None, chunk_indices=None,
            chunk_size=BT, output_dtype=torch.float32,
        )

        # Compare
        max_diff = (ref - out).abs().max().item()
        mean_diff = (ref - out).abs().mean().item()
        if max_diff < 1e-1:  # FP16 tolerance
            print(f"PASSED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
        else:
            print(f"FAILED (max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f})")
            # Show where differences are
            diff = (ref - out).abs()
            idx = diff.argmax()
            print(f"  Max diff at flat index {idx.item()}")
            return False

    print("\nAll correctness tests PASSED!")
    return True


if __name__ == "__main__":
    test_correctness()
