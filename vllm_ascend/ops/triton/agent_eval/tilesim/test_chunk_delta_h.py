"""
chunk_delta_h operator — 函数格式（无 nn.Module）。

本文件适配了 vllm_ascend/ops/triton/fla/chunk_delta_h.py 中的
chunk_gated_delta_rule_fwd_h 函数的 PyTorch 参考实现，
用于 tilesim 评估理论性能极限。

对应测试用例：test_accuracy_chunk_varlen(8, 128, 0, [0, 15], torch.bfloat16)

使用方式：
    python -m examples.api.operator_api.pytorch_examples.main \
        --script examples/api/operator_api/pytorch_examples/test_chunk_delta_h.py
"""

import torch
import torch.nn.functional as F


def chunk_delta_h_ref(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    B, T, Hg, K = k.shape
    H = u.shape[-2]
    V = u.shape[-1]
    BT = chunk_size

    if cu_seqlens is not None:
        N = len(cu_seqlens) - 1
    else:
        N = B

    h = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device)
    if initial_state is not None:
        h = initial_state.to(torch.float32)

    v_new = torch.zeros_like(u, dtype=torch.float32) if save_new_value else None
    final_state = torch.zeros(N, H, K, V, dtype=torch.float32, device=k.device) if output_final_state else None

    g = g.transpose(1, 2).contiguous() if g is not None else None

    for n in range(N):
        if cu_seqlens is not None:
            bos = int(cu_seqlens[n].item())
            eos = int(cu_seqlens[n + 1].item())
            seq_len = eos - bos
        else:
            bos = n * T
            eos = (n + 1) * T
            seq_len = T

        NT = (seq_len + BT - 1) // BT

        for i_h in range(H):
            h_i = h[n, i_h].clone()

            for i_t in range(NT):
                t_start = i_t * BT
                t_end = min(t_start + BT, seq_len)

                b_w = w[0, bos + t_start : bos + t_end, i_h].to(torch.float32)

                k_head_idx = i_h // (H // Hg) if Hg != H else i_h
                b_k = k[0, bos + t_start : bos + t_end, k_head_idx].transpose(0, 1).to(torch.float32)

                if g is not None:
                    last_idx = t_end - 1
                    b_g_last = g[0, i_h, bos + last_idx]
                    b_g_slice = g[0, i_h, bos + t_start : bos + t_end]
                    b_g = torch.exp(b_g_last - b_g_slice)
                    b_g_last = torch.exp(b_g_last)

                b_u = u[0, bos + t_start : bos + t_end, i_h].to(torch.float32)

                b_v_new = b_u - torch.matmul(b_w, h_i)

                if g is not None:
                    b_v_new = b_v_new * b_g.unsqueeze(-1)
                    h_i = h_i * b_g_last

                h_i = h_i + torch.matmul(b_k, b_v_new)

                if save_new_value:
                    v_new[0, bos + t_start : bos + t_end, i_h] = b_v_new

            h[n, i_h] = h_i

        if output_final_state:
            final_state[n] = h[n]

    num_chunks = (T + BT - 1) // BT
    h_out = torch.zeros(B, num_chunks, H, K, V, dtype=torch.float32, device=k.device)
    return h_out, v_new, final_state


def model(k, w, u, g=None, initial_state=None):
    return chunk_delta_h_ref(
        k=k,
        w=w,
        u=u,
        g=g,
        initial_state=initial_state,
        output_final_state=True,
        chunk_size=64,
        save_new_value=True,
        cu_seqlens=None,
    )


def get_inputs():
    torch.manual_seed(42)
    B, T, H, D = 1, 15, 8, 128
    k = torch.randn((B, T, H, D), dtype=torch.bfloat16)
    w = torch.randn((B, T, H, D), dtype=torch.bfloat16)
    u = torch.randn((B, T, H, D), dtype=torch.bfloat16)
    g = F.logsigmoid(torch.rand(B, T, H, dtype=torch.float32)).to(torch.bfloat16)
    initial_state = torch.randn((B, H, D, D), dtype=torch.bfloat16)
    return [k, w, u, g, initial_state]


if __name__ == "__main__":
    inputs = get_inputs()
    h, v_new, final_state = model(*inputs)
    print(f"k: {inputs[0].shape}")
    print(f"w: {inputs[1].shape}")
    print(f"u: {inputs[2].shape}")
    print(f"g: {inputs[3].shape}")
    print(f"initial_state: {inputs[4].shape}")
    print(f"h: {h.shape}")
    print(f"v_new: {v_new.shape}")
    print(f"final_state: {final_state.shape}")
    print("Model runs OK")
