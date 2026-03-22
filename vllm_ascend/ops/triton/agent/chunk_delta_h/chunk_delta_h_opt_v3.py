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

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices, prepare_chunk_offsets, safe_exp

_CONDITIONS = ("seq7168",)


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_recurrence(
    k,
    v,
    w,
    g,
    h,
    h0,
    ht,
    cu_seqlens,
    chunk_offsets,
    T,
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Phase 1: Sequential recurrence to compute h states. No v_new storage."""
    i_bv = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K
    v_start = i_bv * BV

    b_h_bv = tl.zeros([128, BV], dtype=tl.float32)

    offs_k_col = tl.arange(0, 128)[:, None]
    offs_v = v_start + tl.arange(0, BV)[None, :]
    mask_kv = (offs_k_col < K) & (offs_v < V)

    if USE_INITIAL_STATE:
        h0_ptr = h0 + i_nh * K * V
        ptr_h0_bv = h0_ptr + offs_k_col * V + offs_v * 1
        b_h_bv += tl.load(ptr_h0_bv, mask=mask_kv, other=0.0).to(tl.float32)

    k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
    g_ptr_base = g + bos + i_h * T_max
    v_base = v + bos * H * V + i_h * V
    w_base = w + bos * H * K + i_h * K
    offs_t_inner = tl.arange(0, BT)

    # Create block_ptrs once, advance with tl.advance each iteration
    p_w = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (0, 0), (BT, 128), (1, 0))
    p_k = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, 0), (128, BT), (0, 1))
    p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (0, v_start), (BT, BV), (1, 0))

    # Incremental scalars to reduce per-iteration Scalar engine work
    h_stride = H * K * V
    h_base = h + boh * H * K * V + i_h * K * V
    t_start = 0
    last_idx = BT - 1  # incremental last_idx = min((i_t+1)*BT, T) - 1

    for i_t in range(NT):
        # Store current h state
        p_h_bv = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_h_bv, b_h_bv.to(p_h_bv.dtype.element_ty), boundary_check=(0, 1))

        b_w = tl.load(p_w, boundary_check=(0, 1))
        b_k = tl.load(p_k, boundary_check=(0, 1))

        # Clamp last_idx to T-1 for the tail chunk
        actual_last_idx = tl.minimum(last_idx, T - 1)
        b_g_last = tl.load(g_ptr_base + actual_last_idx)

        offs_t = t_start + offs_t_inner
        mask_t = offs_t < T
        b_g = tl.load(g_ptr_base + offs_t, mask=mask_t, other=0.0)

        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        b_v = tl.load(p_v, boundary_check=(0, 1))
        b_v_new = b_v.to(tl.float32)
        b_v_new -= tl.dot(b_w, b_h_bv.to(b_w.dtype))

        if USE_G:
            b_v_new = b_v_new * b_g[:, None]
            b_h_bv = b_h_bv * b_g_last

        b_v_new = b_v_new.to(k.dtype.element_ty)
        b_h_bv += tl.dot(b_k, b_v_new)

        # Advance pointers and scalars for next iteration
        p_w = tl.advance(p_w, (BT, 0))
        p_k = tl.advance(p_k, (0, BT))
        p_v = tl.advance(p_v, (BT, 0))
        h_base += h_stride
        t_start += BT
        last_idx += BT

    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht_bv = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_ht_bv, b_h_bv.to(p_ht_bv.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_vnew(
    v,
    w,
    h,
    v_new,
    cu_seqlens,
    chunk_offsets,
    T,
    H,
    K,
    V,
    BT: tl.constexpr,
    BV: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    """Phase 2: Fully parallel v_new computation using precomputed h states.
    Grid: (NV, NT, N*H) - fully parallel over all chunks!
    v_new = v - dot(w, h)  (no gate - gate is only applied in h-update recurrence)
    """
    i_bv = tl.program_id(0)
    i_t = tl.program_id(1)
    i_nh = tl.program_id(2)
    i_n, i_h = i_nh // H, i_nh % H
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        boh = i_n * tl.cdiv(T, BT)

    stride_v = H * V
    stride_w = H * K
    v_start = i_bv * BV
    t_start = i_t * BT
    NT = tl.cdiv(T, BT)

    # Skip out-of-range chunks (when grid NT > actual NT)
    if i_t >= NT:
        return

    # Load precomputed h[i_t] state (BF16, stored from Phase 1)
    h_base = h + (boh + i_t) * H * K * V + i_h * K * V
    p_h = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
    b_h = tl.load(p_h, boundary_check=(0, 1)).to(tl.float32)

    # Load w for this chunk
    w_base = w + bos * H * K + i_h * K
    p_w = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (t_start, 0), (BT, 128), (1, 0))
    b_w = tl.load(p_w, boundary_check=(0, 1))

    # Load v and compute v_new = v - dot(w, h)
    v_base = v + bos * H * V + i_h * V
    p_v = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (t_start, v_start), (BT, BV), (1, 0))
    b_v = tl.load(p_v, boundary_check=(0, 1))
    b_v_new = b_v.to(tl.float32) - tl.dot(b_w, b_h.to(b_w.dtype))

    # Store v_new (without gate - gate is only for h-update)
    v_new_base = v_new + bos * H * V + i_h * V
    p_v_new = tl.make_block_ptr(v_new_base, (T, V), (stride_v, 1), (t_start, v_start), (BT, BV), (1, 0))
    tl.store(p_v_new, b_v_new.to(p_v_new.dtype.element_ty), boundary_check=(0, 1))


@triton.heuristics(
    {
        "USE_G": lambda args: args["g"] is not None,
        "USE_INITIAL_STATE": lambda args: args["h0"] is not None,
        "STORE_FINAL_STATE": lambda args: args["ht"] is not None,
        "SAVE_NEW_VALUE": lambda args: args["v_new"] is not None,
        "IS_VARLEN": lambda args: args["cu_seqlens"] is not None,
    }
)
@triton.jit(do_not_specialize=["T"])
def chunk_gated_delta_rule_fwd_kernel_h_blockdim64(
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
    H,
    Hg,
    K,
    V,
    BT: tl.constexpr,
    BV: tl.constexpr,
    USE_G: tl.constexpr,
    USE_INITIAL_STATE: tl.constexpr,
    STORE_FINAL_STATE: tl.constexpr,
    SAVE_NEW_VALUE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_bv = tl.program_id(0)
    i_nh = tl.program_id(1)
    i_n, i_h = i_nh // H, i_nh % H
    T_max = 1 * T
    if IS_VARLEN:
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        NT = tl.cdiv(T, BT)
        boh = tl.load(chunk_offsets + i_n).to(tl.int32)
    else:
        bos, eos = i_n * T, i_n * T + T
        NT = tl.cdiv(T, BT)
        boh = i_n * NT

    stride_v = H * V
    stride_k = Hg * K
    stride_w = H * K
    v_start = i_bv * BV

    b_h_bv = tl.zeros([128, BV], dtype=tl.float32)

    if USE_INITIAL_STATE:
        # K=128 and V=128 are exact, use make_block_ptr for efficient load
        h0_ptr = h0 + i_nh * K * V
        p_h0_bv = tl.make_block_ptr(h0_ptr, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        b_h_bv += tl.load(p_h0_bv, boundary_check=(0, 1)).to(tl.float32)

    k_base = k + bos * Hg * K + (i_h // (H // Hg)) * K
    g_ptr_base = g + bos + i_h * T_max
    v_base = v + bos * H * V + i_h * V
    w_base = w + bos * H * K + i_h * K

    # Use tl.advance for w, k, v, g, v_new to avoid re-creating block_ptrs each iteration
    p_w_init = tl.make_block_ptr(w_base, (T, K), (stride_w, 1), (0, 0), (BT, 128), (1, 0))
    p_k_init = tl.make_block_ptr(k_base, (K, T), (1, stride_k), (0, 0), (128, BT), (0, 1))
    p_v_init = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (0, v_start), (BT, BV), (1, 0))
    p_g_init = tl.make_block_ptr(g_ptr_base, (T,), (1,), (0,), (BT,), (0,))
    if SAVE_NEW_VALUE:
        p_v_new_init = tl.make_block_ptr(v_new + bos * H * V + i_h * V, (T, V), (stride_v, 1), (0, v_start), (BT, BV), (1, 0))
    else:
        p_v_new_init = tl.make_block_ptr(v_base, (T, V), (stride_v, 1), (0, v_start), (BT, BV), (1, 0))  # dummy

    h_stride = H * K * V
    h_base = h + boh * H * K * V + i_h * K * V
    last_idx_inc = BT - 1  # incremental last_idx

    # Main loop: process NT-1 full chunks (no boundary_check needed for full chunks)
    # Last chunk may be partial and is handled separately
    NT_full = NT - tl.where(T % BT == 0, 0, 1)

    for i_t in range(NT_full):
        # Load all data first (no dependency on b_h_bv for loads)
        b_w = tl.load(p_w_init)  # full tile, no boundary check
        b_k = tl.load(p_k_init)  # full tile, no boundary check

        b_g_last = tl.load(g_ptr_base + last_idx_inc)  # last_idx_inc = (i_t+1)*BT-1 for full chunks

        b_g = tl.load(p_g_init)  # full tile, no boundary check
        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        b_v = tl.load(p_v_init)  # full tile, no boundary check

        p_h_bv = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_h_bv, b_h_bv.to(p_h_bv.dtype.element_ty))  # K=128 and V=128 are exact, no check needed

        b_v_new = b_v.to(tl.float32)
        b_v_new -= tl.dot(b_w, b_h_bv.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            tl.store(p_v_new_init, b_v_new.to(p_v_new_init.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new = b_v_new * b_g[:, None]
            b_h_bv = b_h_bv * b_g_last

        b_v_new = b_v_new.to(k.dtype.element_ty)
        b_h_bv += tl.dot(b_k, b_v_new)

        # Advance for next iteration
        p_w_init = tl.advance(p_w_init, (BT, 0))
        p_k_init = tl.advance(p_k_init, (0, BT))
        p_v_init = tl.advance(p_v_init, (BT, 0))
        p_g_init = tl.advance(p_g_init, (BT,))
        p_v_new_init = tl.advance(p_v_new_init, (BT, 0))
        h_base += h_stride
        last_idx_inc += BT

    # Handle the last (potentially partial) chunk
    if NT_full < NT:
        b_w = tl.load(p_w_init, boundary_check=(0, 1))
        b_k = tl.load(p_k_init, boundary_check=(0, 1))

        actual_last_idx = tl.minimum(last_idx_inc, T - 1)
        b_g_last = tl.load(g_ptr_base + actual_last_idx)

        b_g = tl.load(p_g_init, boundary_check=(0,))
        b_g = safe_exp(b_g_last - b_g)
        b_g_last = tl.exp(b_g_last)

        b_v = tl.load(p_v_init, boundary_check=(0, 1))

        p_h_bv = tl.make_block_ptr(h_base, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_h_bv, b_h_bv.to(p_h_bv.dtype.element_ty), boundary_check=(0, 1))

        b_v_new = b_v.to(tl.float32)
        b_v_new -= tl.dot(b_w, b_h_bv.to(b_w.dtype))

        if SAVE_NEW_VALUE:
            tl.store(p_v_new_init, b_v_new.to(p_v_new_init.dtype.element_ty), boundary_check=(0, 1))

        if USE_G:
            b_v_new = b_v_new * b_g[:, None]
            b_h_bv = b_h_bv * b_g_last

        b_v_new = b_v_new.to(k.dtype.element_ty)
        b_h_bv += tl.dot(b_k, b_v_new)

    if STORE_FINAL_STATE:
        ht_ptr = ht + i_nh * K * V
        p_ht_bv = tl.make_block_ptr(ht_ptr, (K, V), (V, 1), (0, v_start), (128, BV), (1, 0))
        tl.store(p_ht_bv, b_h_bv.to(p_ht_bv.dtype.element_ty), boundary_check=(0, 1))


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
) -> tuple[torch.Tensor, torch.Tensor]:
    B, T, Hg, K, V = *k.shape, u.shape[-1]
    H = u.shape[-2]
    BT = chunk_size

    chunk_indices = prepare_chunk_indices(cu_seqlens, chunk_size) if cu_seqlens is not None else None
    if cu_seqlens is None:
        N, NT, chunk_offsets = B, triton.cdiv(T, BT), None
    else:
        N, NT, chunk_offsets = (
            len(cu_seqlens) - 1,
            len(chunk_indices),
            prepare_chunk_offsets(cu_seqlens, BT),
        )
    assert K <= 256, "current kernel does not support head dimension larger than 256."

    h = k.new_empty(B, NT, H, K, V)
    final_state = k.new_empty(N, H, K, V, dtype=torch.float32) if output_final_state else None

    v_new = torch.empty_like(u) if save_new_value else None
    g = g.transpose(1, 2).contiguous()

    BV = 64
    NV = triton.cdiv(V, BV)

    # Single integrated kernel: V-split parallelism (NV=2, Grid = (NV, N*H))
    # This is the optimal approach for this recurrent kernel
    def grid(meta):
        return (NV, N * H)

    chunk_gated_delta_rule_fwd_kernel_h_blockdim64[grid](
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
        H=H,
        Hg=Hg,
        K=K,
        V=V,
        BT=BT,
        BV=BV,
        num_warps=4,
        num_stages=2,
    )

    return h, v_new, final_state


def test_performance_with_given_shape():
    shape_str = "1,10288,2,128;1,10288,8,128;1,10288,8,128;1,8,10288"
    shapes = [list(map(int, s.split(','))) for s in shape_str.split(';')]

    B, T, Hg, K = shapes[0]
    _, _, H, _ = shapes[1]
    _, _, _, V = shapes[2]
    assert shapes[1][2] == H and shapes[2][2] == H
    assert shapes[1][3] == K and shapes[2][3] == V

    print(f"Batch size: {B}, Sequence length: {T}, Heads: {H}, Key heads: {Hg}, K dim: {K}, V dim: {V}")
    chunk_size = 128  # BT
    print(f"Chunk size will be {chunk_size}, number of chunks: {triton.cdiv(T, chunk_size)}")
    dtype = torch.bfloat16
    device = 'npu'

    torch.manual_seed(42)
    k = torch.randn(B, T, Hg, K, device=device, dtype=dtype)
    w = torch.randn(B, T, H, K, device=device, dtype=dtype)
    u = torch.randn(B, T, H, V, device=device, dtype=dtype)

    g_raw = torch.randn(B, H, T, device=device, dtype=torch.float32)
    g = g_raw.transpose(1, 2)
    initial_state = torch.randn(B, H, K, V, device=device, dtype=torch.float32)

    for _ in range(5):
        h, v_new, final_state = chunk_gated_delta_rule_fwd_h(
            k=k, w=w, u=u, g=g,
            initial_state=initial_state,
            output_final_state=True,
            chunk_size=chunk_size,
            save_new_value=True,
            cu_seqlens=None
        )
    torch.npu.synchronize()

    print(f"h shape: {h.shape}")
    if v_new is not None:
        print(f"v_new shape: {v_new.shape}, mean={v_new.mean():.4f}, std={v_new.std():.4f}")
    if final_state is not None:
        print(f"final_state shape: {final_state.shape}, mean={final_state.mean():.4f}, std={final_state.std():.4f}")

if __name__ == "__main__":
    test_performance_with_given_shape()
