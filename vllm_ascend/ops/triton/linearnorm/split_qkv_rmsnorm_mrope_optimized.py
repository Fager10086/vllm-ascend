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
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr: torch.Tensor,
    q_weight_ptr: torch.Tensor,
    q_bias_ptr: torch.Tensor,
    k_weight_ptr: torch.Tensor,
    k_bias_ptr: torch.Tensor,
    cos_sin_ptr: torch.Tensor,
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
    mrope_section_t: tl.constexpr,
    mrope_section_h: tl.constexpr,
    mrope_section_w: tl.constexpr,
    has_bias: tl.constexpr,
    is_interleaved: tl.constexpr,
    rope_dim: tl.constexpr,
    half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
):
    block_idx = tl.program_id(0)

    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core

    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (
            num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core
        )

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))

    if has_bias:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, head_size))

    # [Opt-5] Move loop-invariant mask computation outside the loop
    cos_offsets = tl.arange(0, half_rope_dim)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets < mrope_section_t
        h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
        w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (
            cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w
        )

    for index in range(loop_num):
        token_idx = block_offset + index
        qkv_base = in_qkv_ptr + token_idx * (q_size + 2 * kv_size)

        # [Opt-3] Load V and store immediately to free UB
        in_v_tensor = tl.load(qkv_base + q_size + kv_size + tl.arange(0, kv_size))
        tl.store(out_v_ptr + token_idx * kv_size + tl.arange(0, kv_size), in_v_tensor)

        # [Opt-1] Load cos/sin without mask, then apply mask via tl.where
        # [Opt-6] Interleave load and add for better pipeline overlap
        t_cos_offset = cos_sin_ptr + token_idx * rope_dim
        h_cos_offset = t_cos_offset + num_tokens * rope_dim
        w_cos_offset = h_cos_offset + num_tokens * rope_dim

        t_sin_offset = cos_sin_ptr + token_idx * rope_dim + half_rope_dim
        h_sin_offset = t_sin_offset + num_tokens * rope_dim
        w_sin_offset = h_sin_offset + num_tokens * rope_dim

        t_cos_raw = tl.load(t_cos_offset + cos_offsets)
        cos_tensor = tl.where(t_mask, t_cos_raw, 0.0).to(tl.float32)

        h_cos_raw = tl.load(h_cos_offset + cos_offsets)
        cos_tensor = cos_tensor + tl.where(h_mask, h_cos_raw, 0.0).to(tl.float32)

        w_cos_raw = tl.load(w_cos_offset + cos_offsets)
        cos_tensor = (cos_tensor + tl.where(w_mask, w_cos_raw, 0.0).to(tl.float32)).reshape(1, half_rope_dim)
        cos_tensor = tl.broadcast_to(cos_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        t_sin_raw = tl.load(t_sin_offset + cos_offsets)
        sin_tensor = tl.where(t_mask, t_sin_raw, 0.0).to(tl.float32)

        h_sin_raw = tl.load(h_sin_offset + cos_offsets)
        sin_tensor = sin_tensor + tl.where(h_mask, h_sin_raw, 0.0).to(tl.float32)

        w_sin_raw = tl.load(w_sin_offset + cos_offsets)
        sin_tensor = (sin_tensor + tl.where(w_mask, w_sin_raw, 0.0).to(tl.float32)).reshape(1, half_rope_dim)
        sin_tensor = tl.broadcast_to(sin_tensor, (2, half_rope_dim)).reshape(1, rope_dim)

        # --- Q: load, rmsnorm, rope, store ---
        # [Opt-4] Reuse variable names to reduce UB peak usage
        in_q_tensor = tl.load(qkv_base + tl.arange(0, q_size)).to(tl.float32).reshape(num_q_heads, head_size)

        # Q-RMSNorm
        squares = in_q_tensor * in_q_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_q_heads, 1)
        in_q_tensor = in_q_tensor * reciprocal_std
        in_q_tensor = in_q_tensor * q_rmsnorm_weight
        if has_bias:
            in_q_tensor = in_q_tensor + q_bias

        # Q-MRoPE
        x1 = tl.extract_slice(
            in_q_tensor,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            in_q_tensor,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
        cat_x = tl.insert_slice(
            cat_x,
            -x2,
            offsets=(0, 0),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_x = tl.insert_slice(
            cat_x,
            x1,
            offsets=(0, half_rope_dim),
            sizes=(num_q_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(
                in_q_tensor,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = in_q_tensor
        roped_q = cat_x * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            in_q_tensor = tl.insert_slice(
                in_q_tensor,
                roped_q,
                offsets=(0, 0),
                sizes=(num_q_heads, rope_dim),
                strides=(1, 1),
            ).to(tl.bfloat16)
        else:
            in_q_tensor = roped_q.to(tl.bfloat16)

        # [Opt-3] Store Q immediately after computation
        tl.store(out_q_ptr + token_idx * q_size + tl.arange(0, q_size), in_q_tensor.reshape(q_size))

        # --- K: load, rmsnorm, rope, store ---
        in_k_tensor = tl.load(qkv_base + q_size + tl.arange(0, kv_size)).to(tl.float32).reshape(num_kv_heads, head_size)

        # K-RMSNorm (reuse squares, variances, reciprocal_std variable names)
        squares = in_k_tensor * in_k_tensor
        variances = tl.sum(squares, axis=1) / head_size
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(num_kv_heads, 1)
        in_k_tensor = in_k_tensor * reciprocal_std
        in_k_tensor = in_k_tensor * k_rmsnorm_weight
        if has_bias:
            in_k_tensor = in_k_tensor + k_bias

        # K-MRoPE
        y1 = tl.extract_slice(
            in_k_tensor,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        y2 = tl.extract_slice(
            in_k_tensor,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
        cat_y = tl.insert_slice(
            cat_y,
            -y2,
            offsets=(0, 0),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        cat_y = tl.insert_slice(
            cat_y,
            y1,
            offsets=(0, half_rope_dim),
            sizes=(num_kv_heads, half_rope_dim),
            strides=(1, 1),
        )
        if IS_PARTIAL_ROPE:
            orig_qk = tl.extract_slice(
                in_k_tensor,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            )
        else:
            orig_qk = in_k_tensor
        roped_k = cat_y * sin_tensor + orig_qk * cos_tensor

        if IS_PARTIAL_ROPE:
            in_k_tensor = tl.insert_slice(
                in_k_tensor,
                roped_k,
                offsets=(0, 0),
                sizes=(num_kv_heads, rope_dim),
                strides=(1, 1),
            ).to(tl.bfloat16)
        else:
            in_k_tensor = roped_k.to(tl.bfloat16)

        # [Opt-3] Store K immediately after computation
        tl.store(out_k_ptr + token_idx * kv_size + tl.arange(0, kv_size), in_k_tensor.reshape(kv_size))


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

    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num

    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num

    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = num_tokens // core_num

    q_output = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    k_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    v_output = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    has_bias = q_bias is not None

    split_qkv_rmsnorm_mrope_kernel[(block_dim,)](
        qkv,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        cos_sin,
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
        mrope_section[0],
        mrope_section[1],
        mrope_section[2],
        has_bias,
        is_interleaved,
        rope_dim,
        rope_dim // 2,
        IS_PARTIAL_ROPE,
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
        num_tokens,
        q_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    k_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    v_output = torch.empty(
        num_tokens,
        kv_size,
        device=qkv.device,
        dtype=qkv.dtype,
    )

    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
