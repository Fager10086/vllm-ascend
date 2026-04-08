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
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_qk_heads: tl.constexpr,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    qk_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    batch_size_per_vec: tl.constexpr,
):
    row_pid = tl.program_id(0)

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    if BIAS:
        q_bias_val = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
        k_bias_val = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    input_batch_offset = row_pid * batch_size_per_vec
    input_batch_offset_end = min(input_batch_offset + batch_size_per_vec, batch_size)

    all_col_indices = tl.arange(0, total_hidden_size)
    cos_offsets = tl.arange(0, HALF_HEAD_DIM)
    sin_offsets = tl.arange(HALF_HEAD_DIM, HEAD_DIM)

    input_offset = input_batch_offset * total_hidden_size
    q_output_offset = input_batch_offset * q_hidden_size
    k_output_offset = input_batch_offset * kv_hidden_size
    v_output_offset = input_batch_offset * kv_hidden_size

    for row_idx in tl.range(input_batch_offset, input_batch_offset_end):
        # Load entire row (Q+K+V) at once — single contiguous load
        all_values = tl.load(input_ptr + input_offset + all_col_indices)

        # Extract V and store immediately (write interleaving)
        v_values = tl.extract_slice(all_values.reshape(1, total_hidden_size),
                                     offsets=(0, qk_size), sizes=(1, kv_hidden_size), strides=(1, 1))
        tl.store(v_ptr + v_output_offset + tl.arange(0, kv_hidden_size), v_values.reshape(kv_hidden_size))

        # Extract QK part and reshape for RMSNorm
        qk_values = tl.extract_slice(all_values.reshape(1, total_hidden_size),
                                      offsets=(0, 0), sizes=(1, qk_size), strides=(1, 1))
        qk_values = qk_values.reshape(num_qk_heads, HEAD_DIM)

        # RMSNorm Q+K
        normalized_values = qk_values.to(tl.float32)
        squares = normalized_values * normalized_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_qk_heads, 1)
        normalized_values = qk_values * reciprocal_std

        # Load cos/sin
        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_base = pos_idx * HEAD_DIM
        cos = tl.load(cos_sin_ptr + cos_base + cos_offsets).to(tl.float32).reshape(1, HALF_HEAD_DIM)
        sin = tl.load(cos_sin_ptr + cos_base + sin_offsets).to(tl.float32).reshape(1, HALF_HEAD_DIM)

        # Q rope
        q_norm = tl.extract_slice(normalized_values, offsets=(0, 0), sizes=(num_q_heads, HEAD_DIM), strides=(1, 1))
        q_norm = q_norm * q_rmsnorm_weight
        if BIAS:
            q_norm = q_norm + q_bias_val

        x1 = tl.extract_slice(q_norm, offsets=(0, 0), sizes=(num_q_heads, HALF_HEAD_DIM), strides=(1, 1))
        x2 = tl.extract_slice(q_norm, offsets=(0, HALF_HEAD_DIM), sizes=(num_q_heads, HALF_HEAD_DIM), strides=(1, 1))
        roped_q1 = x1 * cos - x2 * sin
        roped_q2 = x2 * cos + x1 * sin
        roped_q = tl.zeros((num_q_heads, HEAD_DIM), dtype=tl.bfloat16)
        roped_q = tl.insert_slice(roped_q, roped_q1.to(tl.bfloat16), offsets=(0, 0), sizes=(num_q_heads, HALF_HEAD_DIM), strides=(1, 1))
        roped_q = tl.insert_slice(roped_q, roped_q2.to(tl.bfloat16), offsets=(0, HALF_HEAD_DIM), sizes=(num_q_heads, HALF_HEAD_DIM), strides=(1, 1))
        tl.store(q_ptr + q_output_offset + tl.arange(0, q_hidden_size), roped_q.reshape(q_hidden_size))

        # K rope
        k_norm = tl.extract_slice(normalized_values, offsets=(num_q_heads, 0), sizes=(num_kv_heads, HEAD_DIM), strides=(1, 1))
        k_norm = k_norm * k_rmsnorm_weight
        if BIAS:
            k_norm = k_norm + k_bias_val

        y1 = tl.extract_slice(k_norm, offsets=(0, 0), sizes=(num_kv_heads, HALF_HEAD_DIM), strides=(1, 1))
        y2 = tl.extract_slice(k_norm, offsets=(0, HALF_HEAD_DIM), sizes=(num_kv_heads, HALF_HEAD_DIM), strides=(1, 1))
        roped_k1 = y1 * cos - y2 * sin
        roped_k2 = y2 * cos + y1 * sin
        roped_k = tl.zeros((num_kv_heads, HEAD_DIM), dtype=tl.bfloat16)
        roped_k = tl.insert_slice(roped_k, roped_k1.to(tl.bfloat16), offsets=(0, 0), sizes=(num_kv_heads, HALF_HEAD_DIM), strides=(1, 1))
        roped_k = tl.insert_slice(roped_k, roped_k2.to(tl.bfloat16), offsets=(0, HALF_HEAD_DIM), sizes=(num_kv_heads, HALF_HEAD_DIM), strides=(1, 1))
        tl.store(k_ptr + k_output_offset + tl.arange(0, kv_hidden_size), roped_k.reshape(kv_hidden_size))

        input_offset += total_hidden_size
        q_output_offset += q_hidden_size
        k_output_offset += kv_hidden_size
        v_output_offset += kv_hidden_size


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
    core_num = get_vectorcore_num()
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim
    num_qk_heads = num_q_heads + num_kv_heads
    qk_size = q_hidden_size + kv_hidden_size
    BIAS = q_bias is not None

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    batch_size_per_vec = triton.cdiv(batch_size, core_num)

    grid = (core_num, )
    split_qkv_rmsnorm_rope_kernel[grid](
        input, cos_sin_cache, positions,
        q_output, k_output, v_output,
        q_weight, q_bias, k_weight, k_bias,
        int(batch_size),
        num_q_heads, num_kv_heads, num_qk_heads,
        q_hidden_size, kv_hidden_size, qk_size, total_hidden_size,
        eps, BIAS, head_dim, head_dim // 2,
        int(batch_size_per_vec),
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
    return (
        torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype),
        torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype),
        torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype),
    )


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
