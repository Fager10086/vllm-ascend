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

from vllm_ascend.ops.triton.triton_utils import extract_slice, get_vectorcore_num, insert_slice


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
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    NUM_Q_HEADS: tl.constexpr,
    NUM_KV_HEADS: tl.constexpr,
):
    pid = tl.program_id(0)
    num_cores = tl.num_programs(0)

    # Pre-load weights once
    q_weight_vals = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_vals = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        q_bias_vals = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
        k_bias_vals = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    total_arange = tl.arange(0, total_hidden_size)
    q_arange = tl.arange(0, q_hidden_size)
    kv_arange = tl.arange(0, kv_hidden_size)

    for row_idx in tl.range(pid, batch_size, num_cores):
        input_row_base = row_idx * total_hidden_size

        # Load entire QKV row at once (single large MTE operation)
        qkv_data = tl.load(input_ptr + input_row_base + total_arange)

        # Split using extract_slice (zero-copy slicing in UB)
        q_data = extract_slice(qkv_data, offsets=(0,),
                               sizes=(q_hidden_size,), strides=(1,))
        k_data = extract_slice(qkv_data, offsets=(q_hidden_size,),
                               sizes=(kv_hidden_size,), strides=(1,))
        v_data = extract_slice(qkv_data, offsets=(q_hidden_size + kv_hidden_size,),
                               sizes=(kv_hidden_size,), strides=(1,))

        # Store V immediately (no computation) - let MTE start writing
        tl.store(v_ptr + row_idx * kv_hidden_size + kv_arange, v_data)

        # Load pos, cos, sin once per token
        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_base = pos_idx * HEAD_DIM
        cos_val = tl.load(cos_sin_ptr + cos_base + tl.arange(0, HALF_HEAD_DIM)).reshape(1, HALF_HEAD_DIM)
        sin_val = tl.load(cos_sin_ptr + cos_base + tl.arange(HALF_HEAD_DIM, HEAD_DIM)).reshape(1, HALF_HEAD_DIM)

        # ---- Q: RMSNorm + RoPE ----
        q_f32 = q_data.to(tl.float32).reshape(NUM_Q_HEADS, HEAD_DIM)
        q_sq = q_f32 * q_f32
        q_var = tl.sum(q_sq, axis=1) / HEAD_DIM
        q_rstd = (1 / tl.sqrt(q_var + eps)).reshape(NUM_Q_HEADS, 1)
        q_norm = q_f32 * q_rstd

        if BIAS:
            q_norm_bf16 = (q_norm * q_weight_vals + q_bias_vals).to(tl.bfloat16)
        else:
            q_norm_bf16 = (q_norm * q_weight_vals).to(tl.bfloat16)

        q_x1 = extract_slice(q_norm_bf16, offsets=(0, 0),
                              sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        q_x2 = extract_slice(q_norm_bf16, offsets=(0, HALF_HEAD_DIM),
                              sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        rq1 = q_x1 * cos_val - q_x2 * sin_val
        rq2 = q_x2 * cos_val + q_x1 * sin_val

        roped_q = tl.zeros((NUM_Q_HEADS, HEAD_DIM), dtype=tl.bfloat16)
        roped_q = insert_slice(roped_q, rq1, offsets=(0, 0),
                               sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_q = insert_slice(roped_q, rq2, offsets=(0, HALF_HEAD_DIM),
                               sizes=(NUM_Q_HEADS, HALF_HEAD_DIM), strides=(1, 1))

        # Store Q
        tl.store(q_ptr + row_idx * q_hidden_size + q_arange,
                 roped_q.reshape(q_hidden_size).to(q_ptr.dtype.element_ty))

        # ---- K: RMSNorm + RoPE ----
        k_f32 = k_data.to(tl.float32).reshape(NUM_KV_HEADS, HEAD_DIM)
        k_sq = k_f32 * k_f32
        k_var = tl.sum(k_sq, axis=1) / HEAD_DIM
        k_rstd = (1 / tl.sqrt(k_var + eps)).reshape(NUM_KV_HEADS, 1)
        k_norm = k_f32 * k_rstd

        if BIAS:
            k_norm_bf16 = (k_norm * k_weight_vals + k_bias_vals).to(tl.bfloat16)
        else:
            k_norm_bf16 = (k_norm * k_weight_vals).to(tl.bfloat16)

        k_x1 = extract_slice(k_norm_bf16, offsets=(0, 0),
                              sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        k_x2 = extract_slice(k_norm_bf16, offsets=(0, HALF_HEAD_DIM),
                              sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        rk1 = k_x1 * cos_val - k_x2 * sin_val
        rk2 = k_x2 * cos_val + k_x1 * sin_val

        roped_k = tl.zeros((NUM_KV_HEADS, HEAD_DIM), dtype=tl.bfloat16)
        roped_k = insert_slice(roped_k, rk1, offsets=(0, 0),
                               sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))
        roped_k = insert_slice(roped_k, rk2, offsets=(0, HALF_HEAD_DIM),
                               sizes=(NUM_KV_HEADS, HALF_HEAD_DIM), strides=(1, 1))

        # Store K
        tl.store(k_ptr + row_idx * kv_hidden_size + kv_arange,
                 roped_k.to(tl.bfloat16).reshape(kv_hidden_size))


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
    assert head_dim == triton.next_power_of_2(head_dim)
    assert q_hidden_size % head_dim == 0
    assert kv_hidden_size % head_dim == 0
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    BIAS = q_bias is not None
    num_vectorcore = get_vectorcore_num()
    num_q_heads = q_hidden_size // head_dim
    num_kv_heads = kv_hidden_size // head_dim

    grid = (num_vectorcore, )

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
        batch_size,
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
