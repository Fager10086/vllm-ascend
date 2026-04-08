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
    do_not_specialize=["batch_size", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_rope_kernel(
    input_ptr,
    cos_sin_ptr,
    q_ptr,
    k_ptr,
    v_ptr,
    q_weight_ptr,
    q_bias_ptr,
    k_weight_ptr,
    k_bias_ptr,
    batch_size,
    front_core_num,
    num_tokens_each_front_core,
    num_tokens_each_tail_core,
    q_hidden_size: tl.constexpr,
    kv_hidden_size: tl.constexpr,
    total_hidden_size: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
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

    # Load weights once outside the loop
    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))

    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    for index in range(loop_num):
        token_idx = block_offset + index

        ## Load Q, K, V from fused input
        in_base = input_ptr + token_idx * total_hidden_size

        # Load Q
        in_q = tl.load(in_base + tl.arange(0, q_hidden_size)).to(tl.float32).reshape(NUM_Q_HEADS, HEAD_DIM)

        # Load K
        in_k = tl.load(in_base + q_hidden_size + tl.arange(0, kv_hidden_size)).to(tl.float32).reshape(NUM_KV_HEADS, HEAD_DIM)

        # Load V
        in_v = tl.load(in_base + q_hidden_size + kv_hidden_size + tl.arange(0, kv_hidden_size))

        # Load cos/sin - use tl.arange-based indexing for pos
        # cos_sin_cache layout: [max_position, HEAD_DIM] where first HALF_HEAD_DIM is cos, next HALF_HEAD_DIM is sin
        # positions are embedded in cos_sin_ptr already via caller preprocessing
        cos_sin_offset = cos_sin_ptr + token_idx * HEAD_DIM
        cos = tl.load(cos_sin_offset + tl.arange(0, HALF_HEAD_DIM)).reshape(1, HALF_HEAD_DIM)
        sin = tl.load(cos_sin_offset + tl.arange(HALF_HEAD_DIM, HEAD_DIM)).reshape(1, HALF_HEAD_DIM)

        ## Q RMSNorm
        q_squares = in_q * in_q
        q_var = tl.sum(q_squares, axis=1) / HEAD_DIM
        q_rstd = (1 / tl.sqrt(q_var + eps)).reshape(NUM_Q_HEADS, 1)
        q_norm = in_q * q_rstd * q_weight_values
        if BIAS:
            q_norm = q_norm + q_bias_values

        ## K RMSNorm
        k_squares = in_k * in_k
        k_var = tl.sum(k_squares, axis=1) / HEAD_DIM
        k_rstd = (1 / tl.sqrt(k_var + eps)).reshape(NUM_KV_HEADS, 1)
        k_norm = in_k * k_rstd * k_weight_values
        if BIAS:
            k_norm = k_norm + k_bias_values

        ## Q RoPE
        q_x1 = tl.extract_slice(q_norm, offsets=(0, 0), sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        q_x2 = tl.extract_slice(q_norm, offsets=(0, HALF_HEAD_DIM), sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_q1 = q_x1 * cos - q_x2 * sin
        roped_q2 = q_x2 * cos + q_x1 * sin
        roped_q = tl.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=tl.float32)
        roped_q = tl.insert_slice(roped_q, roped_q1, offsets=(0, 0), sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_q = tl.insert_slice(roped_q, roped_q2, offsets=(0, HALF_HEAD_DIM), sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))

        ## K RoPE
        k_x1 = tl.extract_slice(k_norm, offsets=(0, 0), sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        k_x2 = tl.extract_slice(k_norm, offsets=(0, HALF_HEAD_DIM), sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_k1 = k_x1 * cos - k_x2 * sin
        roped_k2 = k_x2 * cos + k_x1 * sin
        roped_k = tl.zeros((NUM_KV_HEADS, HEAD_DIM), dtype=tl.float32)
        roped_k = tl.insert_slice(roped_k, roped_k1, offsets=(0, 0), sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_k = tl.insert_slice(roped_k, roped_k2, offsets=(0, HALF_HEAD_DIM), sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))

        ## Store Q
        out_q_offset = q_ptr + token_idx * q_hidden_size
        tl.store(out_q_offset + tl.arange(0, q_hidden_size), roped_q.to(tl.bfloat16).reshape(q_hidden_size))

        ## Store K
        out_k_offset = k_ptr + token_idx * kv_hidden_size
        tl.store(out_k_offset + tl.arange(0, kv_hidden_size), roped_k.to(tl.bfloat16).reshape(kv_hidden_size))

        ## Store V
        out_v_offset = v_ptr + token_idx * kv_hidden_size
        tl.store(out_v_offset + tl.arange(0, kv_hidden_size), in_v)


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
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    # Preprocess: gather cos_sin using positions (index_select outside kernel)
    cos_sin_gathered = cos_sin_cache.index_select(0, positions)  # [batch_size, head_dim]

    core_num = get_vectorcore_num()
    BIAS = q_bias is not None

    front_core_num = core_num
    if batch_size % core_num != 0:
        front_core_num = batch_size % core_num

    num_tokens_each_front_core = (batch_size + core_num - 1) // core_num

    tail_core_num = 0
    if batch_size > core_num:
        tail_core_num = core_num - front_core_num

    num_tokens_each_tail_core = batch_size // core_num

    total_core = front_core_num + tail_core_num
    block_dim = core_num
    if total_core < core_num:
        block_dim = total_core

    split_qkv_rmsnorm_rope_kernel[(block_dim,)](
        input,
        cos_sin_gathered,
        q_output,
        k_output,
        v_output,
        q_weight,
        q_bias,
        k_weight,
        k_bias,
        batch_size,
        front_core_num,
        num_tokens_each_front_core,
        num_tokens_each_tail_core,
        q_hidden_size,
        kv_hidden_size,
        total_hidden_size,
        eps,
        BIAS,
        head_dim,
        head_dim // 2,
        num_q_heads,
        num_kv_heads,
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
    # Fake implementation for shape inference during Dynamo/AOT tracing.
    # Note: sin and cos are not used in shape computation, but must be present in signature.
    batch_size = input.shape[0]
    q_output = torch.empty(
        batch_size,
        q_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    k_output = torch.empty(
        batch_size,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    v_output = torch.empty(
        batch_size,
        kv_hidden_size,
        device=input.device,
        dtype=input.dtype,
    )
    return q_output, k_output, v_output


direct_register_custom_op(
    op_name="qkv_rmsnorm_rope",
    op_func=split_qkv_rmsnorm_rope_impl,
    fake_impl=split_qkv_rmsnorm_rope_impl_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
