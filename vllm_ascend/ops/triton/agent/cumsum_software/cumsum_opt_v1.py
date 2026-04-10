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

_VECTORCORE_NUM = -1


def _get_vectorcore_num() -> int:
    global _VECTORCORE_NUM
    if _VECTORCORE_NUM < 0:
        try:
            from triton.runtime import driver
            _VECTORCORE_NUM = driver.active.utils.get_device_properties("npu")["num_vectorcore"]
        except Exception:
            _VECTORCORE_NUM = 40
    return _VECTORCORE_NUM


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
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (H, N_CHUNKS, CHUNK_SIZE))
        b_s = tl.trans(b_s, (2, 0, 1))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (2, 0, 1))
        b_o = tl.reshape(b_o, (H, BLOCK_T))
    else:
        ptr_s = tl.make_block_ptr(s + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        ptr_o = tl.make_block_ptr(o + bos * H, (T, H), (H, 1), (i_block * BLOCK_T, 0), (BLOCK_T, H), (1, 0))
        b_s = tl.load(ptr_s, boundary_check=(0,)).to(tl.float32)
        b_s = tl.reshape(b_s, (N_CHUNKS, CHUNK_SIZE, H))
        b_s = tl.trans(b_s, (1, 0, 2))
        b_o = tl.cumsum(b_s, axis=0, reverse=REVERSE)
        if HAS_SCALE:
            b_o *= scale
        b_o = tl.trans(b_o, (1, 0, 2))
        b_o = tl.reshape(b_o, (BLOCK_T, H))

    tl.store(ptr_o, b_o.to(s.dtype.element_ty), boundary_check=(0,))
    return


def _compute_optimal_block_t(H: int, chunk_size: int, T: int, B: int) -> int:
    """Compute optimal BLOCK_T that maximizes hardware core utilization while
    respecting UB capacity constraints.

    Strategy:
    1. Ensure BLOCK_T is a power of 2 and a multiple of chunk_size.
    2. Choose the largest BLOCK_T such that total blocks >= num_vectorcore.
    3. Respect UB capacity constraint (192KB).
    """
    UB_CAPACITY = 192 * 1024  # 192KB
    core_num = _get_vectorcore_num()

    # Maximum BLOCK_T that fits in UB:
    # Two buffers (input FP32 + output FP32): 2 * BLOCK_T * H * 4 bytes
    max_bt_ub = UB_CAPACITY // (H * 4 * 2) if H > 0 else chunk_size
    max_bt_ub = 1 << (max_bt_ub.bit_length() - 1) if max_bt_ub > 0 else chunk_size

    # To fill all cores: ceil(T * B / BLOCK_T) >= core_num
    # => BLOCK_T <= T * B / core_num
    if core_num > 0 and T > 0:
        max_bt_cores = (T * B) // core_num
        if max_bt_cores >= chunk_size:
            max_bt_cores = 1 << (max_bt_cores.bit_length() - 1)
        else:
            max_bt_cores = chunk_size
    else:
        max_bt_cores = chunk_size

    optimal_bt = max(chunk_size, min(max_bt_ub, max_bt_cores))
    optimal_bt = 1 << (optimal_bt.bit_length() - 1)
    if optimal_bt < chunk_size:
        optimal_bt = chunk_size

    return optimal_bt


def _prepare_chunk_indices_fast(cu_seqlens: torch.Tensor, chunk_size: int) -> tuple:
    """Compute chunk indices with minimal CPU-NPU sync.

    Compared to the original prepare_chunk_indices which calls .tolist() on
    a device tensor (triggering sync), this version pulls cu_seqlens to CPU
    once (single sync) and does all index computation on CPU before transferring
    the result back to device in one shot.

    Returns (chunk_indices, num_blocks).
    """
    cu_seqlens_cpu = cu_seqlens.tolist()
    n_seqs = len(cu_seqlens_cpu) - 1
    seq_ids = []
    block_ids = []
    for i in range(n_seqs):
        seq_len = cu_seqlens_cpu[i + 1] - cu_seqlens_cpu[i]
        n_blocks = (seq_len + chunk_size - 1) // chunk_size
        seq_ids.extend([i] * n_blocks)
        block_ids.extend(range(n_blocks))
    total_blocks = len(seq_ids)
    chunk_indices = torch.tensor(
        list(zip(seq_ids, block_ids)),
        dtype=cu_seqlens.dtype,
        device=cu_seqlens.device,
    )
    return chunk_indices, total_blocks


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
    OPTIM_BLOCK_SIZE = _compute_optimal_block_t(H, chunk_size, T, B)
    if cu_seqlens is not None and block_indices is None:
        block_indices, num_blocks = _prepare_chunk_indices_fast(cu_seqlens, chunk_size=OPTIM_BLOCK_SIZE)
    else:
        num_blocks = len(block_indices) if cu_seqlens is not None else triton.cdiv(T, OPTIM_BLOCK_SIZE)
    g_org, g = g, torch.empty_like(g, dtype=output_dtype or g.dtype)
    grid = (num_blocks, B)
    chunk_local_cumsum_scalar_kernel[grid](
        s=g_org,
        o=g,
        scale=scale,
        cu_seqlens=cu_seqlens,
        chunk_indices=block_indices,
        T=T,
        B=B,
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
