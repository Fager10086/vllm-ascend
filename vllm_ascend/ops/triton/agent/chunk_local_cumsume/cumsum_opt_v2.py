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


def _next_power_of_2(n):
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _cdiv(a, b):
    return (a + b - 1) // b


@triton.heuristics(
    {"HAS_SCALE": lambda args: args["scale"] is not None, "IS_VARLEN": lambda args: args["cu_seqlens"] is not None}
)
@triton.jit(do_not_specialize=["T"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    B: tl.constexpr,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
):
    i_block, i_b = tl.program_id(0), tl.program_id(1)
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE

    if IS_VARLEN:
        i_s, i_block = (
            tl.load(chunk_indices + i_block * 2).to(tl.int32),
            tl.load(chunk_indices + i_block * 2 + 1).to(tl.int32),
        )
        bos, eos = tl.load(cu_seqlens + i_s).to(tl.int32), tl.load(cu_seqlens + i_s + 1).to(tl.int32)
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    if HEAD_FIRST:
        ptr_s = tl.make_block_ptr(s + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (H, T), (T, 1), (0, i_block * BLOCK_T), (H, BLOCK_T), (1, 0))
        if IS_VARLEN:
            b_s = tl.load(ptr_s, boundary_check=(1,)).to(tl.float32)
        else:
            b_s = tl.load(ptr_s).to(tl.float32)
        b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
        b_o = tl.cumsum(b_s, axis=2, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.reshape(b_o, (H, BLOCK_T))
        if IS_VARLEN:
            tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(1,))
        else:
            tl.store(ptr_o, b_o.to(s.dtype.element_ty))
    else:
        ptr_s = tl.make_block_ptr(s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        if IS_VARLEN:
            b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        else:
            b_s = tl.load(ptr_s).to(tl.float32)
        b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
        b_o = tl.cumsum(b_s, axis=1, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.reshape(b_o, (BLOCK_T, H))
        if IS_VARLEN:
            tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))
        else:
            tl.store(ptr_o, b_o.to(s.dtype.element_ty))


def chunk_local_cumsum_scalar(
    g,
    chunk_size,
    reverse: bool = False,
    scale: float = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
    output_dtype: torch.Tensor | None = torch.float,
):
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    assert chunk_size == 2 ** (chunk_size.bit_length() - 1), "chunk_size must be a power of 2"
    BLOCK_T = _next_power_of_2((2**18) // (H * chunk_size))

    T_padded = T
    if cu_seqlens is not None:
        block_indices = prepare_chunk_indices(cu_seqlens, chunk_size=BLOCK_T)
        num_blocks = len(block_indices)
        g_input = g
        T_kernel = T
    else:
        block_indices = None
        T_padded = _cdiv(T, BLOCK_T) * BLOCK_T
        if T_padded != T:
            pad_t = T_padded - T
            if head_first:
                g_input = torch.nn.functional.pad(g, (0, pad_t))
            else:
                g_input = torch.nn.functional.pad(g, (0, 0, 0, pad_t))
            T_kernel = T_padded
        else:
            g_input = g
            T_kernel = T
        num_blocks = _cdiv(T_kernel, BLOCK_T)

    g_org, g_out = g_input, torch.empty_like(g_input, dtype=output_dtype or g_input.dtype)
    grid = (num_blocks, B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g_out,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=block_indices,
        T=T_kernel,
        B=B,
        H=H,
        BLOCK_T=BLOCK_T,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
    )

    if cu_seqlens is None and T_padded != T:
        if head_first:
            return g_out[:, :, :T]
        else:
            return g_out[:, :T, :]
    return g_out


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
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
