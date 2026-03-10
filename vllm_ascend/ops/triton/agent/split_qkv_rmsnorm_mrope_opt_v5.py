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


@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num",
                       "num_tokens_each_front_core",
                       "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    combined_cos_ptr: torch.Tensor,
    combined_sin_ptr: torch.Tensor,
    out_q_ptr: torch.Tensor,
    out_k_ptr: torch.Tensor,
    out_v_ptr: torch.Tensor,
    num_tokens,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    head_size: tl.constexpr,
    q_size: tl.constexpr,
    kv_size: tl.constexpr,
    eps: tl.constexpr,
    has_bias: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    BATCH_N: tl.constexpr,
    qkv_stride: tl.constexpr,
):
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num
            + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    reduced_loops = loop_num // BATCH_N

    for batch_idx in range(reduced_loops):
        batch_start = block_offset + batch_idx * BATCH_N

        qkv_data = tl.load(
            in_qkv_ptr + batch_start * qkv_stride
            + tl.arange(0, BATCH_N * qkv_stride)
        ).reshape(BATCH_N, qkv_stride)

        in_v_data = tl.extract_slice(
            qkv_data, (0, q_size + kv_size), (BATCH_N, kv_size), (1, 1),
        )
        tl.store(
            out_v_ptr + batch_start * kv_size
            + tl.arange(0, BATCH_N * kv_size),
            in_v_data.reshape(BATCH_N * kv_size),
        )

        in_q_data = tl.extract_slice(
            qkv_data, (0, 0), (BATCH_N, q_size), (1, 1),
        ).to(tl.float32).reshape(BATCH_N * num_q_heads, head_size)

        in_k_data = tl.extract_slice(
            qkv_data, (0, q_size), (BATCH_N, kv_size), (1, 1),
        ).to(tl.float32).reshape(BATCH_N * num_kv_heads, head_size)

        # Q RMSNorm
        squares_q = in_q_data * in_q_data
        var_q = tl.sum(squares_q, axis=1) / head_size
        rstd_q = (1 / tl.sqrt(var_q + eps)).reshape(BATCH_N * num_q_heads, 1)
        q_norm = in_q_data * rstd_q * q_rmsnorm_weight
        if has_bias:
            q_norm = q_norm + q_bias

        # K RMSNorm
        squares_k = in_k_data * in_k_data
        var_k = tl.sum(squares_k, axis=1) / head_size
        rstd_k = (1 / tl.sqrt(var_k + eps)).reshape(BATCH_N * num_kv_heads, 1)
        k_norm = in_k_data * rstd_k * k_rmsnorm_weight
        if has_bias:
            k_norm = k_norm + k_bias

        # Single contiguous load for precomputed cos/sin
        cos_half = tl.load(
            combined_cos_ptr + batch_start * half_rope_dim
            + tl.arange(0, BATCH_N * half_rope_dim)
        ).to(tl.float32).reshape(BATCH_N, half_rope_dim)
        sin_half = tl.load(
            combined_sin_ptr + batch_start * half_rope_dim
            + tl.arange(0, BATCH_N * half_rope_dim)
        ).to(tl.float32).reshape(BATCH_N, half_rope_dim)

        # Q MRoPE — in-place
        q_norm_3d = q_norm.reshape(BATCH_N, num_q_heads, head_size)
        cos_q = cos_half.reshape(BATCH_N, 1, half_rope_dim)
        sin_q = sin_half.reshape(BATCH_N, 1, half_rope_dim)

        x1 = tl.extract_slice(
            q_norm_3d, (0, 0, 0),
            (BATCH_N, num_q_heads, half_rope_dim), (1, 1, 1),
        )
        x2 = tl.extract_slice(
            q_norm_3d, (0, 0, half_rope_dim),
            (BATCH_N, num_q_heads, half_rope_dim), (1, 1, 1),
        )
        res1_q = x1 * cos_q - x2 * sin_q
        res2_q = x2 * cos_q + x1 * sin_q

        q_norm_3d = tl.insert_slice(
            q_norm_3d, res1_q, (0, 0, 0),
            (BATCH_N, num_q_heads, half_rope_dim), (1, 1, 1),
        )
        q_norm_3d = tl.insert_slice(
            q_norm_3d, res2_q, (0, 0, half_rope_dim),
            (BATCH_N, num_q_heads, half_rope_dim), (1, 1, 1),
        )

        tl.store(
            out_q_ptr + batch_start * q_size
            + tl.arange(0, BATCH_N * q_size),
            q_norm_3d.to(tl.bfloat16).reshape(BATCH_N * q_size),
        )

        # K MRoPE — in-place
        k_norm_3d = k_norm.reshape(BATCH_N, num_kv_heads, head_size)
        cos_k = cos_half.reshape(BATCH_N, 1, half_rope_dim)
        sin_k = sin_half.reshape(BATCH_N, 1, half_rope_dim)

        y1 = tl.extract_slice(
            k_norm_3d, (0, 0, 0),
            (BATCH_N, num_kv_heads, half_rope_dim), (1, 1, 1),
        )
        y2 = tl.extract_slice(
            k_norm_3d, (0, 0, half_rope_dim),
            (BATCH_N, num_kv_heads, half_rope_dim), (1, 1, 1),
        )
        res1_k = y1 * cos_k - y2 * sin_k
        res2_k = y2 * cos_k + y1 * sin_k

        k_norm_3d = tl.insert_slice(
            k_norm_3d, res1_k, (0, 0, 0),
            (BATCH_N, num_kv_heads, half_rope_dim), (1, 1, 1),
        )
        k_norm_3d = tl.insert_slice(
            k_norm_3d, res2_k, (0, 0, half_rope_dim),
            (BATCH_N, num_kv_heads, half_rope_dim), (1, 1, 1),
        )

        tl.store(
            out_k_ptr + batch_start * kv_size
            + tl.arange(0, BATCH_N * kv_size),
            k_norm_3d.to(tl.bfloat16).reshape(BATCH_N * kv_size),
        )

    # Remainder: single-token
    remainder_start = reduced_loops * BATCH_N
    for index in range(remainder_start, loop_num):
        token_idx = block_offset + index

        in_q_offset = in_qkv_ptr + token_idx * qkv_stride
        in_q_tensor = tl.load(
            in_q_offset + tl.arange(0, q_size)
        ).to(tl.float32).reshape(num_q_heads, head_size)

        in_k_offset = in_qkv_ptr + token_idx * qkv_stride + q_size
        in_k_tensor = tl.load(
            in_k_offset + tl.arange(0, kv_size)
        ).to(tl.float32).reshape(num_kv_heads, head_size)

        in_v_offset = in_qkv_ptr + token_idx * qkv_stride + q_size + kv_size
        in_v_tensor = tl.load(in_v_offset + tl.arange(0, kv_size))
        tl.store(
            out_v_ptr + token_idx * kv_size + tl.arange(0, kv_size),
            in_v_tensor,
        )

        cos_tensor = tl.load(
            combined_cos_ptr + token_idx * half_rope_dim
            + tl.arange(0, half_rope_dim)
        ).to(tl.float32).reshape(1, half_rope_dim)
        sin_tensor = tl.load(
            combined_sin_ptr + token_idx * half_rope_dim
            + tl.arange(0, half_rope_dim)
        ).to(tl.float32).reshape(1, half_rope_dim)

        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * reciprocal_std * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias

        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * reciprocal_std * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias

        x1 = tl.extract_slice(
            q_normalized, (0, 0), (num_q_heads, half_rope_dim), (1, 1),
        )
        x2 = tl.extract_slice(
            q_normalized, (0, half_rope_dim),
            (num_q_heads, half_rope_dim), (1, 1),
        )
        res1 = x1 * cos_tensor - x2 * sin_tensor
        res2 = x2 * cos_tensor + x1 * sin_tensor
        q_normalized = tl.insert_slice(
            q_normalized, res1, (0, 0),
            (num_q_heads, half_rope_dim), (1, 1),
        )
        q_normalized = tl.insert_slice(
            q_normalized, res2, (0, half_rope_dim),
            (num_q_heads, half_rope_dim), (1, 1),
        )

        y1 = tl.extract_slice(
            k_normalized, (0, 0), (num_kv_heads, half_rope_dim), (1, 1),
        )
        y2 = tl.extract_slice(
            k_normalized, (0, half_rope_dim),
            (num_kv_heads, half_rope_dim), (1, 1),
        )
        res1 = y1 * cos_tensor - y2 * sin_tensor
        res2 = y2 * cos_tensor + y1 * sin_tensor
        k_normalized = tl.insert_slice(
            k_normalized, res1, (0, 0),
            (num_kv_heads, half_rope_dim), (1, 1),
        )
        k_normalized = tl.insert_slice(
            k_normalized, res2, (0, half_rope_dim),
            (num_kv_heads, half_rope_dim), (1, 1),
        )

        tl.store(
            out_q_ptr + token_idx * q_size + tl.arange(0, q_size),
            q_normalized.to(tl.bfloat16).reshape(q_size),
        )
        tl.store(
            out_k_ptr + token_idx * kv_size + tl.arange(0, kv_size),
            k_normalized.to(tl.bfloat16).reshape(kv_size),
        )


def _combine_mrope_cos_sin(cos_sin, mrope_section, is_interleaved, rope_dim):
    half_rd = rope_dim // 2
    cos = cos_sin[:, :, :half_rd]
    sin = cos_sin[:, :, half_rd:]

    offsets = torch.arange(half_rd, device=cos_sin.device)
    if is_interleaved:
        h_mask = ((offsets % 3) == 1) & (offsets <= 3 * mrope_section[1])
        w_mask = ((offsets % 3) == 2) & (offsets <= 3 * mrope_section[2])
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = offsets < mrope_section[0]
        h_mask = (mrope_section[0] - 1 < offsets) & (
            offsets < mrope_section[0] + mrope_section[1]
        )
        w_mask = (mrope_section[0] + mrope_section[1] - 1 < offsets) & (
            offsets < mrope_section[0] + mrope_section[1] + mrope_section[2]
        )

    t_f = t_mask.to(cos_sin.dtype)
    h_f = h_mask.to(cos_sin.dtype)
    w_f = w_mask.to(cos_sin.dtype)

    combined_cos = (cos[0] * t_f + cos[1] * h_f + cos[2] * w_f).contiguous()
    combined_sin = (sin[0] * t_f + sin[1] * h_f + sin[2] * w_f).contiguous()
    return combined_cos, combined_sin


def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    is_interleaved: bool,
    rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()

    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]

    if rope_dim is None:
        rope_dim = head_size
    IS_PARTIAL_ROPE = rope_dim != head_size
    half_rope_dim = rope_dim // 2

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num

    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = num_tokens // core_num

    q_output = torch.empty(num_tokens, q_size, device=qkv.device,
                           dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device,
                           dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device,
                           dtype=qkv.dtype)

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    has_bias = q_bias is not None

    qkv_stride = q_size + 2 * kv_size

    combined_cos, combined_sin = _combine_mrope_cos_sin(
        cos_sin, mrope_section, is_interleaved, rope_dim
    )

    # UB peak per token: max(load, norm, rope) phases
    s_token_load = qkv_stride * 2 + q_size * 4 + kv_size * 4
    s_token_norm = q_size * 4 + kv_size * 4 + q_size * 4
    s_token_rope = (q_size * 4 + kv_size * 4
                    + half_rope_dim * 8
                    + (num_q_heads + num_kv_heads) * rope_dim * 4)
    s_token = max(s_token_load, s_token_norm, s_token_rope)
    BATCH_N = max(1, (85 * 1024) // s_token)
    max_tokens_per_core = max(num_tokens_each_front_core,
                              num_tokens_each_tail_core
                              if num_tokens_each_tail_core > 0 else 0,
                              1)
    BATCH_N = min(BATCH_N, max_tokens_per_core)

    split_qkv_rmsnorm_mrope_kernel[(block_dim,)](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        combined_cos,
        combined_sin,
        q_output,
        k_output,
        v_output,
        num_tokens,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        num_q_heads,
        num_kv_heads,
        head_size,
        q_size,
        kv_size,
        eps,
        has_bias,
        rope_dim,
        half_rope_dim,
        IS_PARTIAL_ROPE,
        BATCH_N,
        qkv_stride,
    )

    return q_output, k_output, v_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor,
    q_weight: torch.Tensor,
    k_weight: torch.Tensor,
    cos_sin: torch.Tensor,
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    eps: float,
    mrope_section: list[int],
    q_bias: torch.Tensor | None = None,
    k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = qkv.shape[0]
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    q_output = torch.empty(
        num_tokens, q_size, device=qkv.device, dtype=qkv.dtype,
    )
    k_output = torch.empty(
        num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype,
    )
    v_output = torch.empty(
        num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype,
    )

    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
