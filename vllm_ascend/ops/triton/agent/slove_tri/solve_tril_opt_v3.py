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
    extract_slice, insert_slice,
    get_vectorcore_num, init_device_properties_triton,
)

from vllm_ascend.ops.triton.fla.utils import prepare_chunk_indices


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "total_tasks"])
def solve_tril_16x16_kernel(
    A,
    Ad,
    cu_seqlens,
    i_t_arr,
    bos_arr,
    T_eff_arr,
    i_h_arr,
    T,
    total_tasks,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    LARGE_BLOCK_T: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    NTASKS: tl.constexpr = 2
    N_BLOCKS: tl.constexpr = LARGE_BLOCK_T // 16 // NTASKS
    BLOCK_STEP: tl.constexpr = LARGE_BLOCK_T // NTASKS

    tmp = tl.arange(0, 16).to(tl.float32)
    rows_mask = tmp[:, None]
    cols_mask = tmp[None, :]
    is_lower = (rows_mask > cols_mask).to(tl.float32)
    on_diagonal = rows_mask == cols_mask

    for task_id in range(pid, total_tasks, num_progs):
        i_t = tl.load(i_t_arr + task_id).to(tl.int32)
        bos = tl.load(bos_arr + task_id).to(tl.int32)
        T_eff = tl.load(T_eff_arr + task_id).to(tl.int32)
        i_h = tl.load(i_h_arr + task_id).to(tl.int32)

        A_base = A + (bos * H + i_h) * BT
        Ad_base = Ad + (bos * H + i_h) * 16
        base_t = i_t * LARGE_BLOCK_T

        for taskid in range(0, NTASKS):
            base_t_inner = base_t + taskid * BLOCK_STEP

            b_A = tl.zeros((N_BLOCKS, 16, 16), dtype=tl.float32)
            for blkid in range(0, N_BLOCKS):
                row_start_o = base_t_inner + blkid * 16
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
                load_mask = global_rows < T_eff

                b_A_raw = tl.load(ptr_A_subrec16).to(tl.float32)
                b_A_subrec16 = tl.where(load_mask, b_A_raw, 0.0)

                b_A = insert_slice(
                    ful=b_A,
                    sub=b_A_subrec16[None, :, :],
                    offsets=[blkid, 0, 0],
                    sizes=[1, 16, 16],
                    strides=[1, 1, 1],
                )

            local_ori_A = tl.trans(b_A, (1, 0, 2))
            local_ori_A = tl.reshape(local_ori_A, (16, 16 * N_BLOCKS))

            b_A = -b_A * is_lower

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

            b_A = tl.where(on_diagonal, b_A + 1.0, b_A)
            b_A = tl.reshape(b_A, (N_BLOCKS * 16, 16))

            offs_rows_to_store = tl.arange(0, N_BLOCKS * 16)
            offs_cols_to_store = tl.arange(0, 16)

            p_Ai = Ad_base + base_t_inner * H * 16 + offs_rows_to_store[:, None] * H * 16 + offs_cols_to_store[None, :]
            global_store_rows = base_t_inner + offs_rows_to_store[:, None]
            store_mask = global_store_rows < T_eff
            tl.store(p_Ai, b_A.to(p_Ai.dtype.element_ty, fp_downcast_rounding="rtne"), mask=store_mask)


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "total_tasks"])
def merge_16x16_to_32x32_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    i_t_arr,
    bos_arr,
    T_eff_arr,
    i_h_arr,
    T,
    total_tasks,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)

    for task_id in range(pid, total_tasks, num_progs):
        i_t = tl.load(i_t_arr + task_id).to(tl.int32)
        bos = tl.load(bos_arr + task_id).to(tl.int32)
        T_eff = tl.load(T_eff_arr + task_id).to(tl.int32)
        i_h = tl.load(i_h_arr + task_id).to(tl.int32)

        A_base = A + (bos * H + i_h) * 32
        Ad_base = Ad + (bos * H + i_h) * 16
        Ai_base = Ai + (bos * H + i_h) * 32

        p_A_21 = tl.make_block_ptr(A_base, (T_eff, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
        p_Ad_11 = tl.make_block_ptr(Ad_base, (T_eff, 16), (H * 16, 1), (i_t * 32, 0), (16, 16), (1, 0))
        p_Ad_22 = tl.make_block_ptr(Ad_base, (T_eff, 16), (H * 16, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))
        p_Ai_11 = tl.make_block_ptr(Ai_base, (T_eff, 32), (H * 32, 1), (i_t * 32, 0), (16, 16), (1, 0))
        p_Ai_22 = tl.make_block_ptr(Ai_base, (T_eff, 32), (H * 32, 1), (i_t * 32 + 16, 16), (16, 16), (1, 0))
        p_Ai_21 = tl.make_block_ptr(Ai_base, (T_eff, 32), (H * 32, 1), (i_t * 32 + 16, 0), (16, 16), (1, 0))

        A_21 = tl.load(p_A_21, boundary_check=(0, 1)).to(tl.float32)
        Ai_11 = tl.load(p_Ad_11, boundary_check=(0, 1)).to(tl.float32)
        Ai_22 = tl.load(p_Ad_22, boundary_check=(0, 1)).to(tl.float32)
        Ai_21 = -tl.dot(
            tl.dot(Ai_22, A_21, input_precision="ieee"),
            Ai_11,
            input_precision="ieee",
        )
        tl.store(p_Ai_11, Ai_11.to(p_Ai_11.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_22, Ai_22.to(p_Ai_22.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))
        tl.store(p_Ai_21, Ai_21.to(p_Ai_21.dtype.element_ty, fp_downcast_rounding="rtne"), boundary_check=(0, 1))


@triton.heuristics({"IS_VARLEN": lambda args: args["cu_seqlens"] is not None})
@triton.jit(do_not_specialize=["T", "total_tasks"])
def merge_16x16_to_64x64_inverse_kernel(
    A,
    Ad,
    Ai,
    cu_seqlens,
    i_t_arr,
    bos_arr,
    T_eff_arr,
    i_h_arr,
    T,
    total_tasks,
    H: tl.constexpr,
    BT: tl.constexpr,
    IS_VARLEN: tl.constexpr,
):
    pid = tl.program_id(0)
    num_progs = tl.num_programs(0)
    offs_n16 = tl.arange(0, 16)
    offs_n32 = tl.arange(0, 32)
    offs_n32s = 32 + tl.arange(0, 16)

    for task_a in range(pid, total_tasks, num_progs * 2):
        task_b = task_a + num_progs
        valid_b = task_b < total_tasks

        i_t_a = tl.load(i_t_arr + task_a).to(tl.int32)
        bos_a = tl.load(bos_arr + task_a).to(tl.int32)
        T_eff_a = tl.load(T_eff_arr + task_a).to(tl.int32)
        i_h_a = tl.load(i_h_arr + task_a).to(tl.int32)
        i_t_b = tl.load(i_t_arr + task_b).to(tl.int32)
        bos_b = tl.load(bos_arr + task_b).to(tl.int32)
        T_eff_b = tl.load(T_eff_arr + task_b).to(tl.int32)
        i_h_b = tl.load(i_h_arr + task_b).to(tl.int32)

        A_base_a = A + (bos_a * H + i_h_a) * 64
        Ad_base_a = Ad + (bos_a * H + i_h_a) * 16
        Ai_base_a = Ai + (bos_a * H + i_h_a) * 64
        A_base_b = A + (bos_b * H + i_h_b) * 64
        Ad_base_b = Ad + (bos_b * H + i_h_b) * 16
        Ai_base_b = Ai + (bos_b * H + i_h_b) * 64

        offs_m_22_a = i_t_a * 64 + 16 + tl.arange(0, 16)
        offs_m_11_a = i_t_a * 64 + tl.arange(0, 16)
        offs_m_44_a = i_t_a * 64 + 48 + tl.arange(0, 16)
        offs_m_33_a = i_t_a * 64 + 32 + tl.arange(0, 16)
        offs_m_32_a = i_t_a * 64 + 32 + tl.arange(0, 32)
        offs_m_22_b = i_t_b * 64 + 16 + tl.arange(0, 16)
        offs_m_11_b = i_t_b * 64 + tl.arange(0, 16)
        offs_m_44_b = i_t_b * 64 + 48 + tl.arange(0, 16)
        offs_m_33_b = i_t_b * 64 + 32 + tl.arange(0, 16)
        offs_m_32_b = i_t_b * 64 + 32 + tl.arange(0, 32)

        Ai_22_a_raw = tl.load(Ad_base_a + offs_m_22_a[:, None] * (H * 16) + offs_n16[None, :])
        A_21_a_raw = tl.load(A_base_a + offs_m_22_a[:, None] * (H * 64) + offs_n16[None, :])
        Ai_11_a_raw = tl.load(Ad_base_a + offs_m_11_a[:, None] * (H * 16) + offs_n16[None, :])
        Ai_44_a_raw = tl.load(Ad_base_a + offs_m_44_a[:, None] * (H * 16) + offs_n16[None, :])
        A_43_a_raw = tl.load(A_base_a + offs_m_44_a[:, None] * (H * 64) + offs_n32s[None, :])
        Ai_33_a_raw = tl.load(Ad_base_a + offs_m_33_a[:, None] * (H * 16) + offs_n16[None, :])
        A_21_32_a_raw = tl.load(A_base_a + offs_m_32_a[:, None] * (H * 64) + offs_n32[None, :])
        Ai_22_b_raw = tl.load(Ad_base_b + offs_m_22_b[:, None] * (H * 16) + offs_n16[None, :])
        A_21_b_raw = tl.load(A_base_b + offs_m_22_b[:, None] * (H * 64) + offs_n16[None, :])
        Ai_11_b_raw = tl.load(Ad_base_b + offs_m_11_b[:, None] * (H * 16) + offs_n16[None, :])
        Ai_44_b_raw = tl.load(Ad_base_b + offs_m_44_b[:, None] * (H * 16) + offs_n16[None, :])
        A_43_b_raw = tl.load(A_base_b + offs_m_44_b[:, None] * (H * 64) + offs_n32s[None, :])
        Ai_33_b_raw = tl.load(Ad_base_b + offs_m_33_b[:, None] * (H * 16) + offs_n16[None, :])
        A_21_32_b_raw = tl.load(A_base_b + offs_m_32_b[:, None] * (H * 64) + offs_n32[None, :])

        mask_22_a = offs_m_22_a[:, None] < T_eff_a
        mask_11_a = offs_m_11_a[:, None] < T_eff_a
        mask_44_a = offs_m_44_a[:, None] < T_eff_a
        mask_33_a = offs_m_33_a[:, None] < T_eff_a
        mask_32_a = offs_m_32_a[:, None] < T_eff_a
        mask_22_b = offs_m_22_b[:, None] < T_eff_b
        mask_11_b = offs_m_11_b[:, None] < T_eff_b
        mask_44_b = offs_m_44_b[:, None] < T_eff_b
        mask_33_b = offs_m_33_b[:, None] < T_eff_b
        mask_32_b = offs_m_32_b[:, None] < T_eff_b

        Ai_22_a = tl.where(mask_22_a, Ai_22_a_raw.to(tl.float32), 0.0)
        A_21_a = tl.where(mask_22_a, A_21_a_raw.to(tl.float32), 0.0)
        Ai_11_a = tl.where(mask_11_a, Ai_11_a_raw.to(tl.float32), 0.0)
        Ai_44_a = tl.where(mask_44_a, Ai_44_a_raw.to(tl.float32), 0.0)
        A_43_a = tl.where(mask_44_a, A_43_a_raw.to(tl.float32), 0.0)
        Ai_33_a = tl.where(mask_33_a, Ai_33_a_raw.to(tl.float32), 0.0)
        A_21_32_a = tl.where(mask_32_a, A_21_32_a_raw.to(tl.float32), 0.0)
        Ai_22_b = tl.where(mask_22_b, Ai_22_b_raw.to(tl.float32), 0.0)
        A_21_b = tl.where(mask_22_b, A_21_b_raw.to(tl.float32), 0.0)
        Ai_11_b = tl.where(mask_11_b, Ai_11_b_raw.to(tl.float32), 0.0)
        Ai_44_b = tl.where(mask_44_b, Ai_44_b_raw.to(tl.float32), 0.0)
        A_43_b = tl.where(mask_44_b, A_43_b_raw.to(tl.float32), 0.0)
        Ai_33_b = tl.where(mask_33_b, Ai_33_b_raw.to(tl.float32), 0.0)
        A_21_32_b = tl.where(mask_32_b, A_21_32_b_raw.to(tl.float32), 0.0)

        tmp_a = tl.dot(Ai_22_a, A_21_a, input_precision="ieee")
        Ai_21_a = -tl.dot(tmp_a, Ai_11_a, input_precision="ieee")
        tmp_a = tl.dot(Ai_44_a, A_43_a, input_precision="ieee")
        Ai_43_a = -tl.dot(tmp_a, Ai_33_a, input_precision="ieee")

        tmp_b = tl.dot(Ai_22_b, A_21_b, input_precision="ieee")
        Ai_21_b = -tl.dot(tmp_b, Ai_11_b, input_precision="ieee")
        tmp_b = tl.dot(Ai_44_b, A_43_b, input_precision="ieee")
        Ai_43_b = -tl.dot(tmp_b, Ai_33_b, input_precision="ieee")

        Ai_22_32_a = tl.zeros((32, 32), tl.float32)
        Ai_22_32_a = insert_slice(Ai_22_32_a, Ai_33_a, (0, 0), (16, 16), (1, 1))
        Ai_22_32_a = insert_slice(Ai_22_32_a, Ai_44_a, (16, 16), (16, 16), (1, 1))
        Ai_22_32_a = insert_slice(Ai_22_32_a, Ai_43_a, (16, 0), (16, 16), (1, 1))
        tmp_a = tl.dot(Ai_22_32_a, A_21_32_a, input_precision="ieee")
        Ai_11_32_a = tl.zeros((32, 32), tl.float32)
        Ai_11_32_a = insert_slice(Ai_11_32_a, Ai_11_a, (0, 0), (16, 16), (1, 1))
        Ai_11_32_a = insert_slice(Ai_11_32_a, Ai_22_a, (16, 16), (16, 16), (1, 1))
        Ai_11_32_a = insert_slice(Ai_11_32_a, Ai_21_a, (16, 0), (16, 16), (1, 1))
        Ai_21_32_a = -tl.dot(tmp_a, Ai_11_32_a, input_precision="ieee")

        Ai_22_32_b = tl.zeros((32, 32), tl.float32)
        Ai_22_32_b = insert_slice(Ai_22_32_b, Ai_33_b, (0, 0), (16, 16), (1, 1))
        Ai_22_32_b = insert_slice(Ai_22_32_b, Ai_44_b, (16, 16), (16, 16), (1, 1))
        Ai_22_32_b = insert_slice(Ai_22_32_b, Ai_43_b, (16, 0), (16, 16), (1, 1))
        tmp_b = tl.dot(Ai_22_32_b, A_21_32_b, input_precision="ieee")
        Ai_11_32_b = tl.zeros((32, 32), tl.float32)
        Ai_11_32_b = insert_slice(Ai_11_32_b, Ai_11_b, (0, 0), (16, 16), (1, 1))
        Ai_11_32_b = insert_slice(Ai_11_32_b, Ai_22_b, (16, 16), (16, 16), (1, 1))
        Ai_11_32_b = insert_slice(Ai_11_32_b, Ai_21_b, (16, 0), (16, 16), (1, 1))
        Ai_21_32_b = -tl.dot(tmp_b, Ai_11_32_b, input_precision="ieee")

        offs_m_s1_a = i_t_a * 64 + tl.arange(0, 32)
        offs_n_s_a = tl.arange(0, 32)
        mask_s1_a = (offs_m_s1_a[:, None] < T_eff_a) & (offs_n_s_a[None, :] < 64)
        tl.store(Ai_base_a + offs_m_s1_a[:, None] * (H * 64) + offs_n_s_a[None, :], Ai_11_32_a.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s1_a)
        offs_m_s2_a = i_t_a * 64 + 32 + tl.arange(0, 32)
        offs_n_s2_a = 32 + tl.arange(0, 32)
        mask_s2_a = (offs_m_s2_a[:, None] < T_eff_a) & (offs_n_s2_a[None, :] < 64)
        tl.store(Ai_base_a + offs_m_s2_a[:, None] * (H * 64) + offs_n_s2_a[None, :], Ai_22_32_a.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s2_a)
        offs_n_s3_a = tl.arange(0, 32)
        mask_s3_a = (offs_m_s2_a[:, None] < T_eff_a) & (offs_n_s3_a[None, :] < 64)
        tl.store(Ai_base_a + offs_m_s2_a[:, None] * (H * 64) + offs_n_s3_a[None, :], Ai_21_32_a.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s3_a)
        offs_m_z_a = i_t_a * 64 + tl.arange(0, 32)
        offs_n_z_a = 32 + tl.arange(0, 32)
        mask_z_a = (offs_m_z_a[:, None] < T_eff_a) & (offs_n_z_a[None, :] < BT)
        tl.store(Ai_base_a + offs_m_z_a[:, None] * (H * BT) + offs_n_z_a[None, :], tl.zeros((32, 32), dtype=A.dtype.element_ty), mask=mask_z_a)

        offs_m_s1_b = i_t_b * 64 + tl.arange(0, 32)
        offs_n_s_b = tl.arange(0, 32)
        mask_s1_b = ((offs_m_s1_b[:, None] < T_eff_b) & (offs_n_s_b[None, :] < 64)) & valid_b
        tl.store(Ai_base_b + offs_m_s1_b[:, None] * (H * 64) + offs_n_s_b[None, :], Ai_11_32_b.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s1_b)
        offs_m_s2_b = i_t_b * 64 + 32 + tl.arange(0, 32)
        offs_n_s2_b = 32 + tl.arange(0, 32)
        mask_s2_b = ((offs_m_s2_b[:, None] < T_eff_b) & (offs_n_s2_b[None, :] < 64)) & valid_b
        tl.store(Ai_base_b + offs_m_s2_b[:, None] * (H * 64) + offs_n_s2_b[None, :], Ai_22_32_b.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s2_b)
        offs_n_s3_b = tl.arange(0, 32)
        mask_s3_b = ((offs_m_s2_b[:, None] < T_eff_b) & (offs_n_s3_b[None, :] < 64)) & valid_b
        tl.store(Ai_base_b + offs_m_s2_b[:, None] * (H * 64) + offs_n_s3_b[None, :], Ai_21_32_b.to(A.dtype.element_ty, fp_downcast_rounding="rtne"), mask=mask_s3_b)
        offs_m_z_b = i_t_b * 64 + tl.arange(0, 32)
        offs_n_z_b = 32 + tl.arange(0, 32)
        mask_z_b = ((offs_m_z_b[:, None] < T_eff_b) & (offs_n_z_b[None, :] < BT)) & valid_b
        tl.store(Ai_base_b + offs_m_z_b[:, None] * (H * BT) + offs_n_z_b[None, :], tl.zeros((32, 32), dtype=A.dtype.element_ty), mask=mask_z_b)


def _compute_task_flat_arrays(ci, cu_seqlens, task_chunk_idx, task_i_bh, H, T, is_varlen, device):
    if is_varlen:
        ci_rows = ci[task_chunk_idx]
        i_n = ci_rows[:, 0].to(torch.int64)
        i_t_arr = ci_rows[:, 1].to(torch.int32)
        bos_arr = cu_seqlens[i_n].to(torch.int32)
        T_eff_arr = (cu_seqlens[i_n + 1] - cu_seqlens[i_n]).to(torch.int32)
    else:
        i_t_arr = task_chunk_idx.to(torch.int32)
        i_b_arr = (task_i_bh // H).to(torch.int32)
        bos_arr = (i_b_arr * T).to(torch.int32)
        T_eff_arr = torch.full((len(task_chunk_idx),), T, dtype=torch.int32, device=device)
    i_h_arr = (task_i_bh % H).to(torch.int32)
    return (
        i_t_arr.contiguous(),
        bos_arr.contiguous(),
        T_eff_arr.contiguous(),
        i_h_arr.contiguous(),
    )


_solve_tril_cache: dict = {}


def _get_cached(cu_seqlens, chunk_size: int, BH: int, H: int, T: int, device, pad: int = 0):
    key = (id(cu_seqlens), chunk_size, BH, pad)
    # if key in _solve_tril_cache:
    #     entry = _solve_tril_cache[key]
    #     if entry[-1] is cu_seqlens:
    #         return entry[:-1]
    #     del _solve_tril_cache[key]
    ci = prepare_chunk_indices(cu_seqlens, chunk_size)
    NT = len(ci)
    total = NT * BH
    task_ids = torch.arange(total, device=device, dtype=torch.int32)
    tci = (task_ids // BH).contiguous()
    tib = (task_ids % BH).contiguous()
    i_t_arr, bos_arr, T_eff_arr, i_h_arr = _compute_task_flat_arrays(
        ci, cu_seqlens, tci, tib, H, T, True, device
    )
    if pad > 0 and total > 0:
        i_t_arr = torch.cat([i_t_arr, i_t_arr[-1:].expand(pad)]).contiguous()
        bos_arr = torch.cat([bos_arr, bos_arr[-1:].expand(pad)]).contiguous()
        T_eff_arr = torch.cat([T_eff_arr, T_eff_arr[-1:].expand(pad)]).contiguous()
        i_h_arr = torch.cat([i_h_arr, i_h_arr[-1:].expand(pad)]).contiguous()
    result = (ci, NT, i_t_arr, bos_arr, T_eff_arr, i_h_arr)
    # _solve_tril_cache[key] = result + (cu_seqlens,)
    return result


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

    LARGE_BLOCK_T = 608 * 2

    init_device_properties_triton()
    BH = B * H

    if cu_seqlens is not None:
        ci_large, NT, i_t_l, bos_l, T_eff_l, i_h_l = _get_cached(
            cu_seqlens, LARGE_BLOCK_T, BH, H, T, A.device
        )
        if chunk_indices_large_block is not None:
            ci_large = chunk_indices_large_block
            NT = len(ci_large)
    else:
        NT = triton.cdiv(T, LARGE_BLOCK_T)
        if chunk_indices_large_block is None:
            ci_large = None
        else:
            ci_large = chunk_indices_large_block
            NT = len(ci_large)
        total_large = NT * BH
        task_ids = torch.arange(total_large, device=A.device, dtype=torch.int32)
        tci_l = (task_ids // BH).contiguous()
        tib_l = (task_ids % BH).contiguous()
        i_t_l, bos_l, T_eff_l, i_h_l = _compute_task_flat_arrays(
            ci_large, cu_seqlens, tci_l, tib_l, H, T, False, A.device
        )

    total_tasks_large = NT * BH
    grid_vec = (get_vectorcore_num(), )

    solve_tril_16x16_kernel[grid_vec](
        A=A,
        Ad=Ad,
        cu_seqlens=cu_seqlens,
        i_t_arr=i_t_l,
        bos_arr=bos_l,
        T_eff_arr=T_eff_l,
        i_h_arr=i_h_l,
        T=T,
        total_tasks=total_tasks_large,
        H=H,
        BT=BT,
        LARGE_BLOCK_T=LARGE_BLOCK_T,
        num_warps=1,
        num_stages=1,
    )

    if BT == 16:
        return Ad

    Ai = torch.empty(B, T, H, BT, device=A.device, dtype=output_dtype)
    merge_fn = merge_16x16_to_32x32_inverse_kernel if BT == 32 else merge_16x16_to_64x64_inverse_kernel
    pad_m = get_vectorcore_num() // 2 if BT == 64 else 0

    if cu_seqlens is not None:
        ci_bt, NT_bt, i_t_m, bos_m, T_eff_m, i_h_m = _get_cached(
            cu_seqlens, BT, BH, H, T, A.device, pad=pad_m
        )
        if chunk_indices_bt is not None:
            ci_bt = chunk_indices_bt
            NT_bt = len(ci_bt)
    else:
        NT_bt = triton.cdiv(T, BT)
        if chunk_indices_bt is None:
            ci_bt = None
        else:
            ci_bt = chunk_indices_bt
            NT_bt = len(ci_bt)
        total_bt = NT_bt * BH
        task_ids = torch.arange(total_bt, device=A.device, dtype=torch.int32)
        tci_m = (task_ids // BH).contiguous()
        tib_m = (task_ids % BH).contiguous()
        i_t_m, bos_m, T_eff_m, i_h_m = _compute_task_flat_arrays(
            ci_bt, cu_seqlens, tci_m, tib_m, H, T, False, A.device
        )

    total_tasks_merge = NT_bt * BH
    grid_cube = (get_vectorcore_num() // 2, )

    merge_fn[grid_cube](
        A=A,
        Ad=Ad,
        Ai=Ai,
        cu_seqlens=cu_seqlens,
        i_t_arr=i_t_m,
        bos_arr=bos_m,
        T_eff_arr=T_eff_m,
        i_h_arr=i_h_m,
        T=T,
        total_tasks=total_tasks_merge,
        H=H,
        BT=BT,
        num_warps=4,
        num_stages=4,
    )
    return Ai



import time
import numpy as np
from typing import Optional, Tuple

if __name__ == "__main__":
    torch.manual_seed(42)
    B, T, H, Hg, K, V, BT, dtype, device, varle = (1, 10012, 4, 16, 128, 128, 64, torch.float16, "npu", True)
    from vllm_ascend.ops.triton.triton_utils import init_device_properties_triton
    init_device_properties_triton()
    if varle:
        seqlens = []
        for _ in range(B):
            seq_len = torch.randint(T//2, T+1, (1,)).item()
            seqlens.append(seq_len)
        
        cu_seqlens = torch.tensor([0] + np.cumsum(seqlens).tolist(), 
                                 dtype=torch.int64, device=device)
        T_total = cu_seqlens[-1].item()
        
        k = torch.randn(B, T_total, Hg, K, dtype=dtype, device=device)
        v = torch.randn(B, T_total, H, V, dtype=dtype, device=device)
        beta = torch.randn(B, T_total, H, dtype=dtype, device=device)
        g_cumsum = torch.randn(B, T_total, H, dtype=dtype, device=device)
        A = torch.randn(B, T_total, H, BT, dtype=dtype, device=device)
    
    for _ in range(20):
        Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    
    torch.npu.synchronize() if device == "npu" else None
    
    start_time = time.time()
    for _ in range(20):
        Ai = solve_tril(A=A, cu_seqlens=cu_seqlens, output_dtype=k.dtype)
    
    torch.npu.synchronize() if device == "npu" else None
    elapsed = (time.time() - start_time) / 20
    print(f"Task Duration: {elapsed*1000:.2f}ms")
