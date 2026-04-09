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

from .utils import prepare_chunk_indices
from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["total_T", "total_chunks", "bos_offset"])
def chunk_local_cumsum_simple_kernel(
    s,
    o,
    scale,
    total_T,
    total_chunks,
    bos_offset,
    H: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    """Optimized kernel for single-sequence cases (no sequence search needed)."""
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for ci in range(pid, total_chunks, NUM_CORE):
        chunk_start = ci * CHUNK_SIZE

        if HEAD_FIRST:
            ptr_s = tl.make_block_ptr(s + bos_offset * H, (H, total_T), (total_T, 1), (0, chunk_start), (H, CHUNK_SIZE), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos_offset * H, (H, total_T), (total_T, 1), (0, chunk_start), (H, CHUNK_SIZE), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=1, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
        else:
            ptr_s = tl.make_block_ptr(s + bos_offset * H, (total_T, H), (H, 1), (chunk_start, 0), (CHUNK_SIZE, H), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos_offset * H, (total_T, H), (H, 1), (chunk_start, 0), (CHUNK_SIZE, H), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale

        tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))


@triton.jit(do_not_specialize=["T", "total_chunks", "seq_count"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    T,
    total_chunks,
    seq_count,
    H: tl.constexpr,
    CHUNK_SIZE: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
):
    pid = tl.program_id(0)
    NUM_CORE = tl.num_programs(0)

    for ci in range(pid, total_chunks, NUM_CORE):
        # Find which sequence this chunk belongs to
        bos = 0
        cur_T = 0
        i_chunk = ci
        remaining = ci
        for s_idx in range(seq_count):
            s_bos = tl.load(cu_seqlens + s_idx).to(tl.int32)
            s_eos = tl.load(cu_seqlens + s_idx + 1).to(tl.int32)
            s_len = s_eos - s_bos
            s_nchunks = tl.cdiv(s_len, CHUNK_SIZE)
            if remaining >= 0 and remaining < s_nchunks:
                bos = s_bos
                cur_T = s_len
                i_chunk = remaining
                remaining = -1
            elif remaining >= 0:
                remaining = remaining - s_nchunks

        chunk_start = i_chunk * CHUNK_SIZE

        if HEAD_FIRST:
            ptr_s = tl.make_block_ptr(s + bos * H, (H, cur_T), (cur_T, 1), (0, chunk_start), (H, CHUNK_SIZE), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos * H, (H, cur_T), (cur_T, 1), (0, chunk_start), (H, CHUNK_SIZE), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=1, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
        else:
            ptr_s = tl.make_block_ptr(s + bos * H, (cur_T, H), (H, 1), (chunk_start, 0), (CHUNK_SIZE, H), (1, 0))
            ptr_o = tl.make_block_ptr(o + bos * H, (cur_T, H), (H, 1), (chunk_start, 0), (CHUNK_SIZE, H), (1, 0))
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
            b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale

        tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    block_indices: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.Tensor | None = torch.float,
):
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    init_device_properties_triton()
    grid = (get_vectorcore_num(), )

    has_scale = scale is not None

    if cu_seqlens is not None:
        seq_count = len(cu_seqlens) - 1

        if seq_count == 1:
            # Single sequence: use T from shape directly (avoids GPU sync from .tolist())
            total_T = T  # T is already extracted from g.shape
            total_chunks = (total_T + chunk_size - 1) // chunk_size
            bos_offset = 0  # B=1 assertion ensures bos=0
            chunk_local_cumsum_simple_kernel[grid](
                s=g_org,
                o=g,
                scale=scale,
                total_T=total_T,
                total_chunks=total_chunks,
                bos_offset=bos_offset,
                H=H,
                CHUNK_SIZE=chunk_size,
                HEAD_FIRST=head_first,
                REVERSE=reverse,
                HAS_SCALE=has_scale,
            )
        else:
            cu_list = cu_seqlens.tolist()
            total_chunks = sum((cu_list[i+1] - cu_list[i] + chunk_size - 1) // chunk_size for i in range(seq_count))
            chunk_local_cumsum_scalar_kernel[grid](
                s=g_org,
                o=g,
                scale=scale,
                cu_seqlens=cu_seqlens,
                T=T,
                total_chunks=total_chunks,
                seq_count=seq_count,
                H=H,
                CHUNK_SIZE=chunk_size,
                HEAD_FIRST=head_first,
                REVERSE=reverse,
                HAS_SCALE=has_scale,
            )
    else:
        total_chunks = triton.cdiv(T, chunk_size) * B
        chunk_local_cumsum_simple_kernel[grid](
            s=g_org,
            o=g,
            scale=scale,
            total_T=T * B,
            total_chunks=total_chunks,
            bos_offset=0,
            H=H,
            CHUNK_SIZE=chunk_size,
            HEAD_FIRST=head_first,
            REVERSE=reverse,
            HAS_SCALE=has_scale,
        )

    return g


def chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.dtype | None = torch.float,
    **kwargs,
) -> torch.Tensor:
    if cu_seqlens is not None:
        assert g.shape[0] == 1, "Only batch size 1 is supported when cu_seqlens are provided"
    if len(g.shape) == 3:
        return chunk_local_cumsum_scalar(
            g=g,
            chunk_size=chunk_size,
            reverse=reverse,
            scale=scale,
            cu_seqlens=cu_seqlens,
            block_indices=kwargs.get("block_indices"),
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
