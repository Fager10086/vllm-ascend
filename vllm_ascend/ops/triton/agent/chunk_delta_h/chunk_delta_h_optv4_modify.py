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

from .utils import prepare_chunk_indices, prepare_chunk_offsets, safe_exp
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, get_vectorcore_num, extract_slice

_CONDITIONS = ("seq7168",)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T", "NH", "NV"])
def chunk_gated_delta_rule_fwd_kernel_h_opt(
    k,
    v,
    w,
    v_new,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    h_update,
    T,
    NH,
    NV,
    H: tl.constexpr,
    Hg: tl.constexpr,
    K: tl.constexpr,
    V: tl.constexpr,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= NH * NV:
        return
    i_nh = pid // NV
    i_v = pid % NV
    i_n = i_nh // H
    i_h = i_nh % H
    v_start = i_v * BV

    T_cur = T
    T_max = 1 * T_cur
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T_cur = eos - bos
        NT = tl.cdiv(T_cur, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T_cur, i_n * T_cur + T_cur
        NT = tl.cdiv(T_cur, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K

    w_base = w + bos * H * K + i_h * K
    k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
    v_base = v + bos * H * V + i_h * V
    g_ptr = g + bos + i_h * T_max
    v_new_base = v_new + bos * H * V + i_h * V if SAVE_NEW_VALUE else v_new

    b_h = tl.zeros([128, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        p_h0 = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        b_h += tl.load(p_h0, boundary_check=(0, 1)).to(tl.float32)

    h_stride = H * K * V
    h_base = h + boh * h_stride + i_h * K * V

    for i_t in range(NT):
        p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_h, b_h.to(p_h.dtype.element_ty), boundary_check=(0, 1))

        p_w = tl.make_block_ptr(w_base, (T_cur, K), (stride_w, 1), (i_t * BT, 0), (BT, 128), (1, 0))
        b_w = tl.load(p_w, boundary_check=(0, 1))

        p_k = tl.make_block_ptr(k_base, (K, T_cur), (1, stride_k), (0, i_t * BT), (128, BT), (0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        last_idx = min((i_t + 1) * BT, T_cur) - 1
        b_g_last = tl.load(g_ptr + last_idx)

        p_g = tl.make_block_ptr(g_ptr, (T_cur,), (1,), (i_t * BT,), (BT,), (0,))
        b_g = tl.load(p_g, boundary_check=(0,))

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        p_v = tl.make_block_ptr(v_base, (T_cur, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
        b_v_new = (tl.load(p_v, boundary_check=(0, 1)).to(tl.float32) - tl.dot(b_w, b_h.to(b_w.dtype))).to(k.dtype.element_ty)

        if SAVE_NEW_VALUE:
            p_v_new = tl.make_block_ptr(v_new_base, (T_cur, V), (stride_v, 1), (i_t * BT, v_start), (BT, BV), (1, 0))
            tl.store(p_v_new, b_v_new, boundary_check=(0, 1))

        if USE_G:
            b_h = b_h * b_g_last
            b_v_new = (b_v_new.to(tl.float32) * b_g[:, None]).to(k.dtype.element_ty)

        b_h += tl.dot(b_k, b_v_new)
        h_base = h_base + h_stride

    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_ht, b_h.to(p_ht.dtype.element_ty), boundary_check=(0, 1))



def chunk_gated_delta_rule_fwd_h(
    k: torch.Tensor,
    w: torch.Tensor,
    u: torch.Tensor,
    g: torch.Tensor | None = None,
    initial_state: torch.Tensor | None = None,
    output_final_state: bool = False,
    chunk_size: int = 64,
    save_new_value: bool = True,
    cu_seqlens: torch.LongTensor | None = None,
    chunk_indices: torch.Tensor | None = None,
    chunk_offsets: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    if cu_seqlens is not None and chunk_indices is None:
        chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size)
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        if chunk_offsets is None:
            chunk_offsets = prepare_chunk_offsets(cu_seqlens, BT)
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            chunk_offsets,
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g = g.transpose(1, 2).contiguous()

    init_device_properties_triton()

    BV = 64
    NV = triton.cdiv(V, BV)
    NH = N * H

    grid = (get_vectorcore_num() // 2,)
    chunk_gated_delta_rule_fwd_kernel_h_opt[grid](
        k=k,
        v=u,
        w=w,
        v_new=v_new,
        g=g,
        h=h,
        h0=initial_state,
        ht=final_state,
        cu_seqlens=cu_seqlens,
        chunk_offsets=chunk_offsets,
        h_update=None,
        T=T,
        NH=NH,
        NV=NV,
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        num_warps=4,
        num_stages=2,
        multibuffer=True,
    )
    return h, v_new, final_state
