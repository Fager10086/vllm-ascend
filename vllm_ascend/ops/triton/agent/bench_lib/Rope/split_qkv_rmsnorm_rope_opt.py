#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# This file is a part of the vllm-ascend project.
#

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(do_not_specialize=["batch_size"])
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    cos_sin_ptr,
    pos_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    qk_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    BIAS: tl.constexpr,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_qk_heads: tl.constexpr,
    batch_size_per_vec: tl.constexpr,
    batch_size_per_iter_per_vec: tl.constexpr,
    iter_num_per_vec: tl.constexpr,
    q_output_step_per_vec: tl.constexpr,
    q_output_step_per_iter_per_vec: tl.constexpr,
    kv_output_step_per_vec: tl.constexpr,
    kv_output_step_per_iter_per_vec: tl.constexpr,
    q_total_ele_num: tl.constexpr,
    kv_total_ele_num: tl.constexpr,
    v_batch_size_per_iter_per_vec: tl.constexpr,
    v_iter_num_per_vec: tl.constexpr,
):
    row_pid = tl.program_id(0)

    q_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    k_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
    if BIAS:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)
        k_bias = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM)).to(tl.float32)

    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)
    output_q_offset = row_pid * q_output_step_per_vec
    output_kv_offset = row_pid * kv_output_step_per_vec
    output_q_offset_end = min(output_q_offset + q_output_step_per_vec, q_total_ele_num)
    output_kv_offset_end = min(output_kv_offset + kv_output_step_per_vec, kv_total_ele_num)

    mblk_idx = tl.arange(0, batch_size_per_iter_per_vec) + input_batch_offset
    nblk_idx = tl.arange(0, qk_hidden_size)
    output_q_indices = output_q_offset + tl.arange(0, q_output_step_per_iter_per_vec)
    output_kv_indices = output_kv_offset + tl.arange(0, kv_output_step_per_iter_per_vec)

    for index in range(iter_num_per_vec):
        cur_mblk_idx = mblk_idx + index * batch_size_per_iter_per_vec
        mmask = cur_mblk_idx < input_batch_offset_end
        mask = (mmask[:, None]) & (nblk_idx[None, :] < qk_hidden_size)
        idx = cur_mblk_idx[:, None] * total_hidden_size + nblk_idx[None, :]
        # load q+k together, keep float32 throughout for extract/insert_slice compatibility
        qk_values = tl.load(input_ptr + idx, mask=mask, other=0.0).to(tl.float32).reshape(
            batch_size_per_iter_per_vec * num_qk_heads, HEAD_DIM
        )

        # rmsnorm
        squares = qk_values * qk_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(batch_size_per_iter_per_vec * num_qk_heads, 1)
        normalized = qk_values * reciprocal_std  # float32

        # split q and k
        normalized_3d = normalized.reshape(batch_size_per_iter_per_vec, num_qk_heads, HEAD_DIM)

        q_norm = tl.extract_slice(
            normalized_3d,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, num_q_heads, HEAD_DIM),
            strides=(1, 1, 1),
        ) * q_weight
        if BIAS:
            q_norm = q_norm + q_bias

        k_norm = tl.extract_slice(
            normalized_3d,
            offsets=(0, num_q_heads, 0),
            sizes=(batch_size_per_iter_per_vec, num_kv_heads, HEAD_DIM),
            strides=(1, 1, 1),
        ) * k_weight
        if BIAS:
            k_norm = k_norm + k_bias

        # load cos/sin for this batch via positions
        pos_vals = tl.load(pos_ptr + cur_mblk_idx, mask=mmask, other=0).to(tl.int64)
        cos_offsets = pos_vals[:, None] * HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)[None, :]
        sin_offsets = pos_vals[:, None] * HEAD_DIM + tl.arange(HALF_HEAD_DIM, HEAD_DIM)[None, :]
        cos = tl.load(cos_sin_ptr + cos_offsets, mask=mmask[:, None], other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_ptr + sin_offsets, mask=mmask[:, None], other=0.0).to(tl.float32)
        # (batch, 1, half_dim) for broadcasting over heads
        cos = cos.reshape(batch_size_per_iter_per_vec, 1, HALF_HEAD_DIM)
        sin = sin.reshape(batch_size_per_iter_per_vec, 1, HALF_HEAD_DIM)

        # q rope: all float32 for extract/insert_slice
        q_x1 = tl.extract_slice(
            q_norm,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, num_q_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        q_x2 = tl.extract_slice(
            q_norm,
            offsets=(0, 0, HALF_HEAD_DIM),
            sizes=(batch_size_per_iter_per_vec, num_q_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        roped_q = tl.zeros((batch_size_per_iter_per_vec, num_q_heads, HEAD_DIM), dtype=tl.float32)
        roped_q = tl.insert_slice(
            roped_q, q_x1 * cos - q_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, num_q_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        roped_q = tl.insert_slice(
            roped_q, q_x2 * cos + q_x1 * sin,
            offsets=(0, 0, HALF_HEAD_DIM),
            sizes=(batch_size_per_iter_per_vec, num_q_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        q_output_offset = index * q_output_step_per_iter_per_vec
        tl.store(
            q_ptr + output_q_indices + q_output_offset,
            roped_q.to(tl.bfloat16).reshape(q_output_step_per_iter_per_vec),
            mask=output_q_indices + q_output_offset < output_q_offset_end,
        )

        # k rope: all float32
        k_x1 = tl.extract_slice(
            k_norm,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, num_kv_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        k_x2 = tl.extract_slice(
            k_norm,
            offsets=(0, 0, HALF_HEAD_DIM),
            sizes=(batch_size_per_iter_per_vec, num_kv_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        roped_k = tl.zeros((batch_size_per_iter_per_vec, num_kv_heads, HEAD_DIM), dtype=tl.float32)
        roped_k = tl.insert_slice(
            roped_k, k_x1 * cos - k_x2 * sin,
            offsets=(0, 0, 0),
            sizes=(batch_size_per_iter_per_vec, num_kv_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        roped_k = tl.insert_slice(
            roped_k, k_x2 * cos + k_x1 * sin,
            offsets=(0, 0, HALF_HEAD_DIM),
            sizes=(batch_size_per_iter_per_vec, num_kv_heads, HALF_HEAD_DIM),
            strides=(1, 1, 1),
        )
        kv_output_offset = index * kv_output_step_per_iter_per_vec
        tl.store(
            k_ptr + output_kv_indices + kv_output_offset,
            roped_k.to(tl.bfloat16).reshape(kv_output_step_per_iter_per_vec),
            mask=output_kv_indices + kv_output_offset < output_kv_offset_end,
        )

    # v: independent loop to avoid UB contention with qk
    v_mblk_idx = tl.arange(0, v_batch_size_per_iter_per_vec) + input_batch_offset
    v_nblk_idx = tl.arange(0, kv_hidden_size)

    for _ in tl.range(v_iter_num_per_vec):
        v_mmask = v_mblk_idx < input_batch_offset_end
        v_mask = (v_mmask[:, None]) & (v_nblk_idx[None, :] < kv_hidden_size)
        v_in_idx = v_mblk_idx[:, None] * total_hidden_size + (q_hidden_size + kv_hidden_size) + v_nblk_idx[None, :]
        v_values = tl.load(input_ptr + v_in_idx, mask=v_mask, other=0.0)
        out_v_idx = v_mblk_idx[:, None] * kv_hidden_size + v_nblk_idx[None, :]
        tl.store(v_ptr + out_v_idx, v_values, mask=v_mask)
        v_mblk_idx += v_batch_size_per_iter_per_vec


def split_qkv_rmsnorm_rope_impl(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_vectorcore = get_vectorcore_num()
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    qk_hidden_size = q_hidden_size + kv_hidden_size
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim
    num_qk_heads = num_q_heads + num_kv_heads
    HALF_HEAD_DIM = head_dim // 2
    BIAS = q_bias is not None

    batch_size_per_vec = triton.cdiv(batch_size, num_vectorcore)

    # UB capacity planning (85KB): account for all live tensors per token
    # qk_values, normalized: num_qk_heads * head_dim * 2
    # q_norm, k_norm: num_qk_heads * head_dim
    # cos, sin: HALF_HEAD_DIM * 2
    # roped_q, roped_k: num_qk_heads * head_dim
    ub_elements = 85 * 1024 // 4  # float32 elements
    per_token_elements = (
        num_qk_heads * head_dim * 4  # qk_values, normalized, q_norm+k_norm, roped_q+roped_k
        + HALF_HEAD_DIM * 2          # cos, sin
    )
    batch_size_per_iter_per_vec = max(1, ub_elements // per_token_elements)
    batch_size_per_iter_per_vec = min(batch_size_per_iter_per_vec, batch_size_per_vec)
    iter_num_per_vec = triton.cdiv(batch_size_per_vec, batch_size_per_iter_per_vec)

    q_output_step_per_vec = batch_size_per_vec * q_hidden_size
    q_output_step_per_iter_per_vec = batch_size_per_iter_per_vec * q_hidden_size
    kv_output_step_per_vec = batch_size_per_vec * kv_hidden_size
    kv_output_step_per_iter_per_vec = batch_size_per_iter_per_vec * kv_hidden_size
    q_total_ele_num = batch_size * q_hidden_size
    kv_total_ele_num = batch_size * kv_hidden_size

    v_batch_size_per_iter_per_vec = max(1, 85 * 1024 // input.element_size() // (kv_hidden_size + 1))
    v_batch_size_per_iter_per_vec = min(v_batch_size_per_iter_per_vec, batch_size_per_vec)
    v_iter_num_per_vec = triton.cdiv(batch_size_per_vec, v_batch_size_per_iter_per_vec)

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    grid = (min(num_vectorcore, batch_size),)
    split_qkv_rmsnorm_rope_kernel[grid](
        input,
        cos_sin_cache,
        positions,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        int(batch_size),
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        qk_hidden_size,
        eps,
        head_dim,
        HALF_HEAD_DIM,
        BIAS,
        num_q_heads,
        num_kv_heads,
        num_qk_heads,
        int(batch_size_per_vec),
        int(batch_size_per_iter_per_vec),
        int(iter_num_per_vec),
        int(q_output_step_per_vec),
        int(q_output_step_per_iter_per_vec),
        int(kv_output_step_per_vec),
        int(kv_output_step_per_iter_per_vec),
        int(q_total_ele_num),
        int(kv_total_ele_num),
        int(v_batch_size_per_iter_per_vec),
        int(v_iter_num_per_vec),
    )
    return q_output, k_output, v_output


def split_qkv_rmsnorm_rope_impl_fake(
    input: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    q_hidden_size: int,
    kv_hidden_size: int,
    head_dim: int,
    eps: float,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    batch_size = input.shape[0]
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
