# Copyright © 2025 Huawei Technologies Co., Ltd.
# Based on vLLM: https://github.com/vllm-project/vllm
# Based on flash-linear-attention: https://github.com/fla-org/flash-linear-attention
#
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# SPDX-FileCopyrightText: Songlin Yang, Yu Zhang
#
# This file contains code copied from the flash-linear-attention project.
# The original source code was licensed under the MIT license and included
# the following copyright notice:
# Copyright (c) 2023-2025, Songlin Yang, Yu Zhang
# ruff: noqa: E501

import os
import numpy as np
from typing import Optional

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule_fwd

device = "npu"


def get_abs_err(x, y):
    return (x.detach() - y.detach()).flatten().abs().max().item()


def get_err_ratio(x, y):
    err = (x.detach() - y.detach()).flatten().square().mean().sqrt().item()
    base = (x.detach()).flatten().square().mean().sqrt().item()
    return err / (base + 1e-8)


def assert_close(prefix, ref, tri, ratio, warning=False, err_atol=1e-6):
    abs_atol = get_abs_err(ref, tri)
    msg = f"{prefix:>16} diff: {abs_atol:.6f} ratio: {get_err_ratio(ref, tri):.6f}"
    error_rate = get_err_ratio(ref, tri)
    if abs_atol <= err_atol:
        return
    else:
        assert error_rate < ratio, msg


def print_diff(name, ref, tri, atol=0.005):
    abs_diff = torch.abs(ref - tri)
    max_abs_diff = abs_diff.max().item()
    print(f"[{name}] Max absolute difference: {max_abs_diff:.6f}")
    if max_abs_diff > atol:
        print(f"Exceeds tolerance ({atol})!")


def recurrent_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )
    B, H, T, K, V = *k.shape, v.shape[-1]
    o = torch.zeros(B, H, T, V).to(v)
    h = torch.zeros(B, H, K, V).to(v)
    if initial_state is not None:
        h = initial_state
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    q = q * scale
    for i in range(T):
        b_q = q[:, :, i]
        b_k = k[:, :, i]
        b_v = v[:, :, i].clone()
        h = h.clone() * g[:, :, i].exp()[..., None, None]
        b_beta = beta[:, :, i]
        b_v = b_v - (h.clone() * b_k[..., None]).sum(-2)
        b_v = b_v * b_beta[..., None]
        h = h.clone() + b_k.unsqueeze(-1) * b_v.unsqueeze(-2)
        o[:, :, i] = torch.einsum("bhd,bhdm->bhm", b_q, h)
    if not output_final_state:
        h = None
    o = o.transpose(1, 2).contiguous()
    return o, h


def chunk_gated_delta_rule_ref(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    chunk_size: int = 64,
    scale: float = None,
    initial_state: torch.Tensor = None,
    output_final_state: bool = False,
):
    BT = chunk_size
    if scale is None:
        scale = 1 / (q.shape[-1] ** 0.5)
    # Calculate padding needed to make T a multiple of BT
    q, k, v, beta, g = map(
        lambda x: x.transpose(1, 2).contiguous().to(torch.float32), [q, k, v, beta, g]
    )

    T = q.shape[-2]
    pad_len = (BT - (T % BT)) % BT
    if pad_len > 0:
        # Pad all tensors
        q = F.pad(q, (0, 0, 0, pad_len))
        k = F.pad(k, (0, 0, 0, pad_len))
        v = F.pad(v, (0, 0, 0, pad_len))
        beta = F.pad(beta, (0, pad_len))
        g = F.pad(g, (0, pad_len))
    q, k, v, beta, g = map(lambda x: x.to(torch.float32), [q, k, v, beta, g])
    decay = g
    chunk_size = BT
    b, h, l, d_k = q.shape
    d_v = v.shape[-1]
    q = q * scale
    v = v * beta[..., None]
    k_beta = k * beta[..., None]
    assert l % chunk_size == 0
    # note that diagonal is masked.
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=0,
    )
    q, k, v, k_beta, decay = map(
        lambda x: rearrange(x, "b h (n c) d -> b h n c d", c=chunk_size),
        [q, k, v, k_beta, decay.unsqueeze(-1)],
    )
    decay = decay.squeeze(-1).cumsum(-1)
    decay_exp = decay.exp()[..., None]
    L_mask = ((decay.unsqueeze(-1) - decay.unsqueeze(-2)).tril().exp().float()).tril()
    attn = -((k_beta @ k.transpose(-1, -2)) * L_mask).masked_fill(mask, 0)
    for i in range(1, chunk_size):
        attn[..., i, :i] = attn[..., i, :i].clone() + (
            attn[..., i, :i, None].clone() * attn[..., :i, :i].clone()
        ).sum(-2)
    attn = attn + torch.eye(chunk_size, dtype=torch.float, device=q.device)
    attn = attn
    k_cumsum = attn @ v
    k_cumdecay = attn @ (k_beta * decay_exp)
    v = k_cumsum
    S = k.new_zeros(b, h, d_k, d_v)
    if initial_state is not None:
        S = initial_state
    o = torch.zeros_like(v)
    mask = torch.triu(
        torch.ones(chunk_size, chunk_size, dtype=torch.bool, device=q.device),
        diagonal=1,
    )
    for i in range(0, l // chunk_size):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn = (q_i @ k_i.transpose(-1, -2) * L_mask[:, :, i]).masked_fill_(mask, 0)
        v_prime = (k_cumdecay[:, :, i]) @ S
        v_new = v_i - v_prime
        o_inter = (q_i * decay[:, :, i, :, None].exp()) @ S
        o[:, :, i] = o_inter + attn @ v_new
        S = (
            S * decay[:, :, i, -1, None, None].exp()
            + (
                k_i * (decay[:, :, i, -1, None] - decay[:, :, i]).exp()[..., None]
            ).transpose(-1, -2)
            @ v_new
        )
    if not output_final_state:
        S = None
    # unpad
    o = rearrange(o, "b h n c d -> b h (n c) d")
    o = o[:, :, :T]
    o = o.transpose(1, 2)
    return o, S


def build_random_cu_seqlens(T, min_len=16, max_len=512, device="cuda"):
    """
    构造真实 varlen batch:
    BS=1
    多个 request 拼接
    总长度 = T
    """
    seqlens = []
    remain = T

    while remain > 0:
        if remain <= max_len:
            seqlens.append(remain)
            break

        l = torch.randint(min_len, max_len + 1, (1,)).item()
        l = min(l, remain)
        seqlens.append(l)
        remain -= l

    cu_seqlens = torch.tensor(
        [0] + np.cumsum(seqlens).tolist(),
        dtype=torch.int64,
        device=device,
    )

    return cu_seqlens, seqlens

@pytest.mark.parametrize(
    ("H", "D", "mask_p", "cu_seqlens", "dtype"),
    [
        pytest.param(*test, id="H{}-D{}-mask_p{}-cu_seqlens{}-{}".format(*test))
        for test in [
            (8, 128, 0, [0, 6], torch.float16),
            (8, 128, 0, [0, 15], torch.float16),
            (8, 128, 0, [0, 24], torch.float16),
            (8, 128, 0, [0, 31], torch.float16),
            (8, 128, 0, [0, 32], torch.float16),
            (8, 128, 0, [0, 33], torch.float16),
            (8, 128, 0, [0, 48], torch.float16),
            (8, 128, 0, [0, 63], torch.float16),
            (8, 128, 0, [0, 64], torch.float16),
            (8, 128, 0, [0, 64], torch.float16),
            (8, 128, 0, [0, 76], torch.float16),
            (8, 128, 0, [0, 84], torch.float16),
            (8, 128, 0, [0, 96], torch.float16),
            (8, 128, 0, [0, 100], torch.float16),
            (8, 128, 0, [0, 115], torch.float16),
            (8, 128, 0, [0, 127], torch.float16),
            (8, 128, 0, [0, 128], torch.float16),
        ]
    ],
)
def test_accuracy_chunk_varlen(
    H: int,
    D: int,
    mask_p: float,
    cu_seqlens: list[int],
    dtype: torch.dtype,
):  
    # # case1 need to dump
    # B, H, Hg, D, mask_p, T, dtype = (8, 16, 4, 128, 0, 10012, torch.float16) 
    # ======================================================
    # ALL CASES
    # ======================================================
    cases = [
        # case1
        dict(B=1, H=16, Hg=4, D=128, mask_p=0, T=10012, dtype=torch.float16),
        # case2
        dict(B=1, H=8, Hg=2, D=128, mask_p=0, T=10012, dtype=torch.float16),
        # case3
        dict(B=1, H=16, Hg=4, D=128, mask_p=0, T=10016, dtype=torch.float16),
        # case4
        dict(B=1, H=8, Hg=2, D=128, mask_p=0, T=10012, dtype=torch.float16),
        # case5
        dict(B=1, H=16, Hg=4, D=128, mask_p=0, T=10012, dtype=torch.bfloat16),
        # case6
        dict(B=1, H=8, Hg=2, D=128, mask_p=0, T=10012, dtype=torch.bfloat16),
        # case7
        dict(B=1, H=16, Hg=4, D=128, mask_p=0, T=10016, dtype=torch.bfloat16),
        # case8
        dict(B=1, H=8, Hg=2, D=128, mask_p=0, T=10012, dtype=torch.bfloat16),
        # case9
        dict(B=1, H=16, Hg=16, D=128, mask_p=0, T=5, dtype=torch.bfloat16, cu_seqlens=[0,5]),
    ]
    
    base_dump_dir = "/vllm-workspace/golden"
    os.makedirs(base_dump_dir, exist_ok=True)
    
    seed = 10
    torch.manual_seed(seed)
    os.environ["TRITON_F32_DEFAULT"] = "ieee"
    
    for case in cases:
        B = case["B"]
        H = case["H"]
        Hg = case["Hg"]
        D = case["D"]
        mask_p = case["mask_p"]
        T = case["T"]
        dtype = case["dtype"]
        cu_seqlens = case.get("cu_seqlens", None)
        
        if D != 128:
            pytest.skip(
                reason="chunk_gated_delta_rule is not supported on alchemist for D!=128"
            )
        
        # randomly split the sequence into N segments
        seqlens = []
        for _ in range(B):
            seq_len = torch.randint(T//2, T+1, (1,)).item()
            seqlens.append(seq_len)
        
        if cu_seqlens is None:
            cu_seqlens, seqlens = build_random_cu_seqlens(
                T,
                min_len=32,
                max_len=512,
                device=device,
            )
        else:
            cu_seqlens = torch.tensor(cu_seqlens, dtype=torch.int64, device=device)
        N = len(cu_seqlens) - 1
        print(N,cu_seqlens)

        # seq-first required for inputs with variable lengths
        q = torch.randn((1, T, Hg, D), dtype=dtype)
        k = F.normalize(torch.randn(1, T, Hg, D, dtype=torch.float32), p=2, dim=-1).to(dtype)
        v = torch.randn((1, T, H, D), dtype=dtype)
        g = F.logsigmoid(torch.rand(1, T, H, dtype=torch.float32))
        g = g * (torch.rand_like(g) > mask_p)
        beta = torch.rand(1, T, H, dtype=dtype).sigmoid()
        h0 = torch.randn((N, H, D, D), dtype=dtype)

        q, k, v, beta, g, h0 = map(
            lambda x: x.to(device).requires_grad_(), (q, k, v, beta, g, h0)
        )
        
        case_name = (
            f"rand{seed}_B{B}_H{H}_Hg{Hg}_D{D}_T{T}_mask{mask_p}_"
            f"{str(dtype).split('.')[-1]}"
        )

        case_dir = os.path.join(base_dump_dir, case_name)
        os.makedirs(case_dir, exist_ok=True)
        
        input_path = os.path.join(case_dir, "input.pt")
        output_path = os.path.join(case_dir, "output.pt")

        torch.save(
            {
                # ===== inputs =====
                "q": q.detach().cpu(),
                "k": k.detach().cpu(),
                "v": v.detach().cpu(),
                "beta": beta.detach().cpu(),
                "g": g.detach().cpu(),
                "h0": h0.detach().cpu(),
                "cu_seqlens": cu_seqlens.detach().cpu(),

            },
            input_path
        )
        g_out, o, A, final_state, w, h, v_new = chunk_gated_delta_rule_fwd(
            q=q.clone(),
            k=k.clone(),
            v=v.clone(),
            beta=beta.clone(),
            g=g.clone(),
            scale=None,
            initial_state=h0.clone(),
            output_final_state=True,
            cu_seqlens=cu_seqlens,
        )
        torch.save(
            {
                # ===== outputs =====
                "g_out": g_out.detach().cpu(),
                "o": o.detach().cpu(),
                "A": A.detach().cpu() if A is not None else None,
                "final_state": final_state.detach().cpu(),
                "w": w.detach().cpu() if w is not None else None,
                "h": h.detach().cpu() if h is not None else None,
                "v_new": v_new.detach().cpu() if v_new is not None else None,
            },
            output_path
        )


if __name__ == "__main__":
    test_accuracy_chunk_varlen(8, 128, 0, [0, 128], torch.float16)
