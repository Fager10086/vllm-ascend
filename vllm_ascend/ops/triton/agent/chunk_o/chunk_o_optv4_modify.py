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

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_offsets, safe_exp
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "NT_total", "N", "NH"])
def chunk_fwd_kernel_o(
    q,
    k,
    v,
    h,
    g,
    o,
    cu_seqlens,
    chunk_offsets,
    scale,
    T,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BK: tl.constexpr,
    BV: tl.constexpr,
    NT_total,
    N,
    NH,
    USE_G: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)
    HperHg: tl.constexpr = H // Hg
    NV: tl.constexpr = V // BV
    total_outer = N * Hg * NT_total

    for outer_idx in range(pid, total_outer, num_programs):
        i_hg = outer_idx % Hg
        tmp = outer_idx // Hg
        i_t = tmp % NT_total
        i_n = tmp // NT_total

        T_max = T
        if IS_VARLEN:
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            T_cur = eos - bos
            NT = tl.cdiv(T_cur, BT)
            boh = tl.load(chunk_offsets + i_n).to(tl.int64)
        else:
            bos = i_n * T
            T_cur = T
            NT = tl.cdiv(T_cur, BT)
            boh = i_n * NT

        if i_t < NT:
            i_tg = boh + i_t
            offs_t = i_t * BT + tl.arange(0, BT)
            mask_t = offs_t < T_cur

            q_base = q + (bos * Hg + i_hg) * K
            k_base = k + (bos * Hg + i_hg) * K

            b_q = tl.load(
                tl.make_block_ptr(q_base, (T_cur, K), (Hg * K, 1), (i_t * BT, 0), (BT, BK), (1, 0)),
                boundary_check=(0, 1)
            )
            b_k = tl.load(
                tl.make_block_ptr(k_base, (K, T_cur), (1, Hg * K), (0, i_t * BT), (BK, BT), (0, 1)),
                boundary_check=(0, 1)
            )
            b_A_base = tl.dot(b_q, b_k)

            o_i = tl.arange(0, BT).to(tl.float32)
            m_A = o_i[:, None] >= o_i[None, :]

            for i_h_local in range(HperHg):
                i_h = i_hg * HperHg + i_h_local
                h_ptr = h + (i_tg * H + i_h).to(tl.int64) * K * V

                b_A = b_A_base

                if USE_G:
                    g_base = g + bos + i_h * T_max
                    b_g_raw = tl.load(g_base + offs_t, mask=mask_t)
                    b_g = tl.where(mask_t, b_g_raw, 0.0)
                    b_A = b_A * safe_exp(b_g[:, None] - b_g[None, :])
                    b_g_exp = tl.exp(b_g)

                b_A = tl.where(m_A, b_A, 0)

                v_base = v + (bos * H + i_h) * V
                o_base = o + (bos * H + i_h) * V

                for i_v in range(NV):
                    b_h = tl.load(
                        tl.make_block_ptr(h_ptr, (K, V), (V, 1), (0, i_v * BV), (BK, BV), (1, 0)),
                        boundary_check=(0, 1)
                    )
                    b_o = tl.dot(b_q, b_h)
                    if USE_G:
                        b_o = b_o * b_g_exp[:, None]

                    p_v = tl.make_block_ptr(v_base, (T_cur, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))
                    p_o = tl.make_block_ptr(o_base, (T_cur, V), (H * V, 1), (i_t * BT, i_v * BV), (BT, BV), (1, 0))

                    b_v = tl.load(p_v, boundary_check=(0, 1))
                    b_o = b_o * scale + tl.dot(b_A.to(b_v.dtype), b_v) * scale
                    tl.store(p_o, b_o.to(p_o.dtype.element_ty), boundary_check=(0, 1))


_chunk_offsets_cache: dict = {}


def chunk_fwd_o(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    h: torch.Tensor,
    g: torch.Tensor | None = None,
    scale: float | None = None,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_size: int = 64,
) -> torch.Tensor:
    B, T, Hg, K, V = *q.shape, v.shape[-1]
    H = v.shape[-2]
    BT = chunk_size

    if scale is None:
        scale = k.shape[-1] ** -0.5

    o = torch.empty_like(v)
    # if cu_seqlens is None:
    #     N, chunk_offsets = B, None
    # else:
    #     N = len(cu_seqlens) - 1
    #     cache_key = (cu_seqlens.data_ptr(), cu_seqlens.shape[0], BT)
    #     if cache_key not in _chunk_offsets_cache:
    #         _chunk_offsets_cache[cache_key] = prepare_chunk_offsets(cu_seqlens, BT)
    #     chunk_offsets = _chunk_offsets_cache[cache_key]
    if cu_seqlens is None:
        N, chunk_offsets = B, None
    else:
        N, chunk_offsets = (
            len(cu_seqlens) - 1,
            prepare_chunk_offsets(cu_seqlens, BT),
        )

    
    NH = N * H
    NT_total = triton.cdiv(T, BT)

    init_device_properties_triton()
    grid = (get_vectorcore_num() // 2,)

    if g is not None:
        g = g.transpose(1, 2).contiguous()
    chunk_fwd_kernel_o[grid](
        q=q,
        k=k,
        v=v,
        h=h,
        g=g,
        o=o,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        scale=scale,
        T=T,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BK=128,
        BV=128,
        NT_total=NT_total,
        N=N,
        NH=NH,
        num_warps=4,
        num_stages=2,
    )
    return o

if __name__ == "__main__":
    torch.manual_seed(42)
    device = "npu"
    dtype = torch.float16  

    T = 10012
    H = 4
    D = 128
    Hg = 16
    dtype = torch.bfloat16
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    import time
    init_device_properties_triton()
    
    q = torch.randn((1, T, H, D), device=device, dtype=dtype)
    k = torch.randn(1, T, H, D, device=device, dtype=dtype)
    v = torch.randn((1, T, Hg, D), device=device, dtype=dtype)
    B, T, Hg, K, V = *k.shape, v.shape[-1]
    chunk_size = 64
    BT = chunk_size
    NT = triton.cdiv(T, BT)
    h = k.new_empty(B, NT, Hg, K, V)
    g = torch.randn((1, T, Hg), device=device, dtype=torch.float32)
    scale = k.shape[-1] ** -0.5
    cu_seqlens:list[int] = [0, T]
    cu_seqlens = torch.LongTensor(cu_seqlens).to(device)

    for _ in range(20):
        chunk_fwd_o(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )

    torch.npu.synchronize() if device == "npu" else None
    start_time = time.time()
    
    for _ in range(20):
        chunk_fwd_o(
            q=q,
            k=k,
            v=v,
            h=h,
            g=g,
            scale=scale,
            cu_seqlens=cu_seqlens,
            chunk_size=chunk_size
        )
  
    torch.npu.synchronize() if device == "npu" else None
    elapsed = (time.time() - start_time) / 20
    print(f"Task Duration: {elapsed*1000:.2f}ms")
