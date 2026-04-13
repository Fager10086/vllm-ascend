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

from vllm_ascend.ops.triton.triton_utils import (
    get_vectorcore_num,
    init_device_properties_triton,
    insert_slice,
    extract_slice,
)

from .utils import prepare_chunk_indices


@triton.jit(do_not_specialize=["T", "total_blocks"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    chunk_indices,
    T,
    total_blocks,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16

    for work_idx in range(pid, total_blocks, num_programs):
        if IS_VARLEN:
            i_t_raw = work_idx // H
            i_h = work_idx % H
            i_n = tl.load(chunk_indices + i_t_raw * 2).to(tl.int32)
            i_t = tl.load(chunk_indices + i_t_raw * 2 + 1).to(tl.int32)
            bos = tl.load(cu_seqlens + i_n).to(tl.int32)
            eos = tl.load(cu_seqlens + i_n + 1).to(tl.int32)
            cur_T = eos - bos
        else:
            blocks_per_batch = tl.cdiv(T, LARGE_BLOCK_T)
            i_bh = work_idx // blocks_per_batch
            i_t = work_idx % blocks_per_batch
            i_b = i_bh // H
            i_h = i_bh % H
            bos = i_b * T
            cur_T = T

        A_base = A + (bos * H + i_h) * BT
        Ad_base = Ad + (bos * H + i_h) * 16

        base_t = i_t * LARGE_BLOCK_T

        # Load all N_BLOCKS 16x16 sub-blocks
        b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)
        for blkid in range(0, N_BLOCKS):
            row_start_o = base_t + blkid * 16
            col_start_o = row_start_o % BT

            offs_rows_in_block = tl.arange(0, 16)
            offs_cols_in_block = tl.arange(0, 16)

            ptr_A_subrec16 = (
                A_base
                + row_start_o * H * BT
                + col_start_o
                + offs_rows_in_block[:, None] * H * BT
                + offs_cols_in_block[None, :]
            )

            global_rows = row_start_o + offs_rows_in_block[:, None]
            global_cols = col_start_o + offs_cols_in_block[None, :]
            load_mask = (global_rows < cur_T) & (global_cols < BT)

            b_A_subrec16 = tl.load(ptr_A_subrec16, mask=load_mask).to(tl.float32)
            b_A = insert_slice(
                ful=b_A,
                sub=b_A_subrec16[None, :, :],
                offsets=[blkid, 0, 0],
                sizes=[1, 16, 16],
                strides=[1, 1, 1],
            )

        local_ori_A = tl.trans(b_A, (1, 0, 2))
        local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))

        # Convert mask into matrix multiplication to avoid for loops ub oom
        tmp = tl.arange(0, 16).to(tl.float32)
        rows = tmp[:, None]
        cols = tmp[None, :]
        is_lower = (rows > cols).to(b_A.dtype)
        b_A = -b_A * is_lower

        # for loop to update N_BLOCKS row vector
        for i in range(1, 16):
            nblks_vec16 = -extract_slice(local_ori_A, (i, 0), (1, 16 * N_BLOCKS), (16 * N_BLOCKS, 1))
            b_a = tl.reshape(nblks_vec16, (N_BLOCKS, 16))

            dot_tmp = tl.trans(b_a[:, :, None] * b_A, (1, 0, 2))
            dot_product = tl.sum(dot_tmp, 0)
            b_a = b_a + dot_product

            b_a_new_expanded = b_a[:, None, :]
            b_A = insert_slice(
                ful=b_A, sub=b_a_new_expanded, offsets=[0, i, 0], sizes=[N_BLOCKS, 1, 16], strides=[1, 1, 1]
            )

        on_diagonal = rows == cols
        b_A = tl.where(on_diagonal, b_A + 1.0, b_A)

        b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))

        # Store using explicit pointer + mask
        offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
        offs_cols_to_store = tl.arange(0, 16)
        p_Ai = Ad_base + base_t * H * 16 + offs_rows_to_store[:, None] * H * 16 + offs_cols_to_store[None, :]
        global_store_rows = base_t + offs_rows_to_store[:, None]
        store_mask = global_store_rows < cur_T
        tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=store_mask)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * 32
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * 32

    p_A_21 = tl.make_block_ptr(A, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ad_11 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ad_22 = tl.make_block_ptr(Ad, (T, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
    p_Ai_11 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0))
    p_Ai_22 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0))
    p_Ai_21 = tl.make_block_ptr(Ai, (T, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

    A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
    Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
    Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
    Ai_21 = -tl.dot(
        tl.dot(Ai_22, A_21, input_precision="ieee"),
        Ai_11,
        input_precision="ieee",
    )
    tl.store(
        p_Ai_11,
        Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_22,
        Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )
    tl.store(
        p_Ai_21,
        Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"),
        boundary_check=(0, 1),
    )


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    chunk_indices,
    T,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    i_t, i_bh = tl.program_id(0), tl.program_id(1)
    i_b, i_h = i_bh // H, i_bh % H
    if IS_VARLEN:
        i_n, i_t_val = (
            tl.load(chunk_indices + i_t * 2).to(tl.int32),
            tl.load(chunk_indices + i_t * 2 + 1).to(tl.int32),
        )
        bos, eos = (
            tl.load(cu_seqlens + i_n).to(tl.int32),
            tl.load(cu_seqlens + i_n + 1).to(tl.int32),
        )
        T = eos - bos
        i_t = i_t_val
    else:
        bos, eos = i_b * T, i_b * T + T

    A += (bos * H + i_h) * BT
    Ad += (bos * H + i_h) * 16
    Ai += (bos * H + i_h) * BT

    offs_16 = tl.arange(0, 16)

    # Load Ai_22 and A_21, compute dot immediately (interleaved load-compute)
    offs_m1 = i_t * 64 + 16 + offs_16
    mask_m1 = offs_m1[:, None] < T
    Ai_22 = tl.load(Ad + offs_m1[:, None] * (H * 16) + offs_16[None, :], mask=mask_m1).to(tl.float32)
    A_21 = tl.load(A + offs_m1[:, None] * (H * BT) + offs_16[None, :], mask=mask_m1).to(tl.float32)
    tmp_21 = tl.dot(Ai_22, A_21, input_precision="ieee")

    # Load Ai_11, complete Ai_21
    offs_m0 = i_t * 64 + offs_16
    mask_m0 = offs_m0[:, None] < T
    Ai_11 = tl.load(Ad + offs_m0[:, None] * (H * 16) + offs_16[None, :], mask=mask_m0).to(tl.float32)
    Ai_21 = -tl.dot(tmp_21, Ai_11, input_precision="ieee")

    # Load Ai_44 and A_43, compute dot immediately (interleaved load-compute)
    offs_m3 = i_t * 64 + 48 + offs_16
    offs_n32 = 32 + offs_16
    mask_m3 = offs_m3[:, None] < T
    Ai_44 = tl.load(Ad + offs_m3[:, None] * (H * 16) + offs_16[None, :], mask=mask_m3).to(tl.float32)
    A_43 = tl.load(A + offs_m3[:, None] * (H * BT) + offs_n32[None, :], mask=mask_m3).to(tl.float32)
    tmp_43 = tl.dot(Ai_44, A_43, input_precision="ieee")

    # Load Ai_33, complete Ai_43
    offs_m2 = i_t * 64 + 32 + offs_16
    mask_m2 = offs_m2[:, None] < T
    Ai_33 = tl.load(Ad + offs_m2[:, None] * (H * 16) + offs_16[None, :], mask=mask_m2).to(tl.float32)
    Ai_43 = -tl.dot(tmp_43, Ai_33, input_precision="ieee")

    # Build 32x32 blocks
    Ai_22_32 = tl.zeros((32, 32), tl.float32)
    Ai_22_32 = insert_slice(Ai_22_32, Ai_33, (0, 0), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_44, (16, 16), (16, 16), (1, 1))
    Ai_22_32 = insert_slice(Ai_22_32, Ai_43, (16, 0), (16, 16), (1, 1))

    Ai_11_32 = tl.zeros((32, 32), tl.float32)
    Ai_11_32 = insert_slice(Ai_11_32, Ai_11, (0, 0), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_22, (16, 16), (16, 16), (1, 1))
    Ai_11_32 = insert_slice(Ai_11_32, Ai_21, (16, 0), (16, 16), (1, 1))

    # 32x32 level merge
    offs_32 = tl.arange(0, 32)
    offs_m_lo = i_t * 64 + 32 + offs_32
    mask_lo = offs_m_lo[:, None] < T
    A_21_32 = tl.load(A + offs_m_lo[:, None] * (H * BT) + offs_32[None, :], mask=mask_lo).to(tl.float32)
    Ai_21_32 = -tl.dot(tl.dot(Ai_22_32, A_21_32, input_precision="ieee"), Ai_11_32, input_precision="ieee")

    # Store all results
    offs_m_s0 = i_t * 64 + offs_32
    mask_s0 = offs_m_s0[:, None] < T
    offs_n_32 = 32 + offs_32

    # Store Ai_11_32 at (0, 0)
    tl.store(
        Ai + offs_m_s0[:, None] * (H * BT) + offs_32[None, :],
        Ai_11_32.to(Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_s0,
    )

    # Store Ai_22_32 at (32, 32)
    tl.store(
        Ai + offs_m_lo[:, None] * (H * BT) + offs_n_32[None, :],
        Ai_22_32.to(Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_lo,
    )

    # Store Ai_21_32 at (32, 0)
    tl.store(
        Ai + offs_m_lo[:, None] * (H * BT) + offs_32[None, :],
        Ai_21_32.to(Ai.dtype.element_ty, fp_downcast_rounding="rtne"),
        mask=mask_lo,
    )

    # Zero out upper-right 32x32 block (rows 0~31, cols 32~63)
    mask_z = (offs_m_s0[:, None] < T) & (offs_n_32[None, :] < BT)
    zero_block = tl.zeros((32, 32), dtype=tl.float32)
    tl.store(
        Ai + offs_m_s0[:, None] * (H * BT) + offs_n_32[None, :],
        zero_block.to(Ai.dtype.element_ty),
        mask=mask_z,
    )


def solve_tril(
    A: torch.Tensor,
    cu_seqlens: torch.Tensor | None = None,
    chunk_indices_large_block: torch.Tensor | None = None,
    chunk_indices_bt: torch.Tensor | None = None,
    output_dtype: torch.dtype = torch.float,
) -> torch.Tensor:
    """
    Compute the inverse of the matrix I + A
    A should be strictly lower triangular, i.e., A.triu() == 0.

    Args:
        A (torch.Tensor):
            [B, T, H, BT], where BT should only be 16, 32, or 64.
        cu_seqlens (torch.Tensor):
            The cumulative sequence lengths of the input tensor. Default: `None`.
        output_dtype (torch.dtype):
            The dtype of the output tensor. Default: `torch.float`.
            If `None`, the output dtype will be the same as the input dtype.

    Returns:
        (I + A)^-1 with the same shape as A
    """
    assert A.shape[-1] in [16, 32, 64]

    B, T, H, BT = A.shape
    Ad = torch.empty(B, T, H, 16, device=A.device, dtype=torch.float if BT != 16 else output_dtype)

    # Reduce LARGE_BLOCK_T to increase parallelism
    # Original: 1216, NT=1 for T=1216 -> only 16 blocks total
    # Optimized: 304, NT=4 for T=1216 -> 64 blocks total
    # UB budget: N_BLOCKS = 304/16 = 19
    # b_A: 19*16*16*4 = 19456 bytes = 19 KB
    # local_ori_A: 16*(16*19)*4 = 19456 bytes = 19 KB
    # Total: ~38 KB << 85 KB limit, well within Double Buffering range
    LARGE_BLOCK_T = 304

    init_device_properties_triton()
    IS_VARLEN = cu_seqlens is not None

    # Always recompute chunk_indices for the new LARGE_BLOCK_T
    # (prebuilt chunk_indices_large_block may use a different block size)
    if IS_VARLEN:
        chunk_indices = prepare_chunk_indices(cu_seqlens, LARGE_BLOCK_T)
    else:
        chunk_indices = chunk_indices_large_block

    if IS_VARLEN:
        NT = len(chunk_indices)
        total_blocks_16 = NT * H
    else:
        NT = triton.cdiv(T, LARGE_BLOCK_T)
        total_blocks_16 = B * H * NT

    grid_16 = (get_vectorcore_num(), )
    solve_tril_16x16_kernel[grid_16](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        total_blocks=total_blocks_16,
        H=H,
        BT=BT,
        IS_VARLEN=IS_VARLEN,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        num_warps=1,
        num_stages=4,
    )

    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    if IS_VARLEN and chunk_indices_bt is None:
        chunk_indices_bt = prepare_chunk_indices(cu_seqlens, BT)
    chunk_indices = chunk_indices_bt
    NT = len(chunk_indices) if IS_VARLEN else triton.cdiv(T, BT)

    merge_fn[NT, B * H](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        chunk_indices=chunk_indices,
        T=T,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=3,
    )
    return Ai
