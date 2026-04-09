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

from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton, get_vectorcore_num


def _prepare_chunk_indices_fast(cu_seqlens: torch.Tensor, chunk_size: int) -> torch.Tensor:
    """Vectorized version of prepare_chunk_indices on CPU, avoids .tolist() and Python loops."""
    cu_cpu = cu_seqlens.cpu()
    lens = cu_cpu[1:] - cu_cpu[:-1]
    n_chunks = (lens + chunk_size - 1) // chunk_size
    total = int(n_chunks.sum().item())
    seq_ids = torch.repeat_interleave(torch.arange(len(n_chunks)), n_chunks)
    offsets = torch.arange(total)
    cum = torch.nn.functional.pad(n_chunks.cumsum(0), (1, 0), value=0)
    seq_starts = torch.repeat_interleave(cum[:-1], n_chunks)
    chunk_ids = offsets - seq_starts
    return torch.stack([seq_ids, chunk_ids], dim=1).to(cu_seqlens)


@triton.heuristics(
    {"HAS_SCALE": lambda args: args["scale"] is not None, "IS_VARLEN": lambda args: args["cu_seqlens"] is not None}
)
@triton.jit(do_not_specialize=["T", "total_blocks", "blocks_per_batch"])
def chunk_local_cumsum_scalar_kernel(
    s,
    o,
    scale,
    cu_seqlens,
    chunk_indices,
    T,
    total_blocks,
    blocks_per_batch,
    H: tl.constexpr,
    BLOCK_T: tl.constexpr,
    REVERSE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    HEAD_FIRST: tl.constexpr,
    CHUNK_SIZE: tl.constexpr = 64,
):
    N_CHUNKS: tl.constexpr = BLOCK_T // CHUNK_SIZE
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    for work_idx in range(pid, total_blocks, num_programs):
        if IS_VARLEN:
            i_s = tl.load(chunk_indices + work_idx * 2).to(tl.int32)
            i_block = tl.load(chunk_indices + work_idx * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_s).to(tl.int32)
            eos = tl.load(cu_seqlens + i_s + 1).to(tl.int32)
            seq_T = eos - bos
        else:
            i_b = work_idx // blocks_per_batch
            i_block = work_idx % blocks_per_batch
            bos = i_b * T
            seq_T = T

        block_start = i_block * BLOCK_T

        if HEAD_FIRST:
            base_s = s + bos * H
            base_o = o + bos * H
            row_offsets = tl.arange(0, BLOCK_T)
            col_offsets = tl.arange(0, H)
            offsets = col_offsets[:, None] * seq_T + (block_start + row_offsets[None, :])
            mask = (block_start + row_offsets[None, :]) < seq_T
            b_s = tl.load(base_s + offsets, mask=mask)
            b_s = tl.where(mask, b_s, 0.0).to(tl.float32)
            b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
            b_o = tl.cumsum(b_s, axis=2, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            b_o = tl.reshape(b_o, (H, BLOCK_T))
            tl.store(base_o + offsets, b_o.to(s.dtype.element_ty), mask=mask)
        else:
            base_s = s + bos * H
            base_o = o + bos * H
            row_offsets = tl.arange(0, BLOCK_T)
            col_offsets = tl.arange(0, H)
            offsets = (block_start + row_offsets[:, None]) * H + col_offsets[None, :]
            mask = (block_start + row_offsets[:, None]) < seq_T
            b_s = tl.load(base_s + offsets, mask=mask)
            b_s = tl.where(mask, b_s, 0.0).to(tl.float32)
            b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
            b_o = tl.cumsum(b_s, axis=1, reverse=REVERSE)
            if HAS_SCALE:
                b_o *= scale
            b_o = tl.reshape(b_o, (BLOCK_T, H))
            tl.store(base_o + offsets, b_o.to(s.dtype.element_ty), mask=mask)


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

    S_peak_per_token = 2 * H * 4
    max_tokens = (85 * 1024) // S_peak_per_token
    OPTIM_BLOCK_SIZE = triton.next_power_of_2(max_tokens) // 2
    OPTIM_BLOCK_SIZE = max(OPTIM_BLOCK_SIZE, chunk_size)

    block_indices = _prepare_chunk_indices_fast(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE) if cu_seqlens is not None else None
    num_blocks = len(block_indices) if cu_seqlens is not None else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    total_blocks = num_blocks if cu_seqlens is not None else B * num_blocks
    blocks_per_batch = num_blocks if cu_seqlens is None else 0

    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)

    init_device_properties_triton()
    grid = (get_vectorcore_num(), )

    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=block_indices,
        T=T,
        total_blocks=total_blocks,
        blocks_per_batch=blocks_per_batch,
        H=H,
        BLOCK_T=OPTIM_BLOCK_SIZE,
        CHUNK_SIZE=chunk_size,
        HEAD_FIRST=head_first,
        REVERSE=reverse,
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
            head_first=head_first,
            output_dtype=output_dtype,
        )
    else:
        raise ValueError(
            f"Unsupported input shape {g.shape}, "
            f"which should be (B, T, H, D) if `head_first=False` "
            f"or (B, H, T, D) otherwise"
        )
