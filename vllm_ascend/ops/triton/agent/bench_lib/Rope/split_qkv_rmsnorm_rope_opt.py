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


@triton.jit
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
    num_q_heads: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_qk_heads: tl.constexpr,
    eps: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
    batch_size_per_vec: tl.constexpr,
    batch_size_per_iter: tl.constexpr,
    iter_num: tl.constexpr,
    q_total_elems: tl.constexpr,
    kv_total_elems: tl.constexpr,
    q_step_per_vec: tl.constexpr,
    q_step_per_iter: tl.constexpr,
    kv_step_per_vec: tl.constexpr,
    kv_step_per_iter: tl.constexpr,
    v_batch_per_iter: tl.constexpr,
    v_iter_num: tl.constexpr,
):
    row_pid = tl.program_id(0)

    q_weight = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        q_bias = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
        k_bias = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    input_batch_base = row_pid * batch_size_per_vec
    input_batch_end = min(input_batch_base + batch_size_per_vec, batch_size)
    q_out_base = row_pid * q_step_per_vec
    kv_out_base = row_pid * kv_step_per_vec
    q_out_end = min(q_out_base + q_step_per_vec, q_total_elems)
    kv_out_end = min(kv_out_base + kv_step_per_vec, kv_total_elems)

    # indices for qk load: [batch_per_iter, qk_hidden_size]
    feat_idx = tl.arange(0, qk_hidden_size)
    feat_mask = feat_idx < qk_hidden_size
    q_out_idx = tl.arange(0, q_step_per_iter)
    kv_out_idx = tl.arange(0, kv_step_per_iter)

    for i in range(iter_num):
        batch_idx = input_batch_base + i * batch_size_per_iter + tl.arange(0, batch_size_per_iter)
        bmask = batch_idx < input_batch_end
        mask2d = bmask[:, None] & feat_mask[None, :]

        # load Q+K together: [batch_per_iter, qk_hidden_size]
        idx2d = batch_idx[:, None] * total_hidden_size + feat_idx[None, :]
        qk_vals = tl.load(input_ptr + idx2d, mask=mask2d, other=0.0).to(tl.float32)
        qk_vals = qk_vals.reshape(batch_size_per_iter * num_qk_heads, HEAD_DIM)

        # RMSNorm for all Q+K heads together
        sq = qk_vals * qk_vals
        var = tl.sum(sq, axis=1) / HEAD_DIM
        rstd = (1.0 / tl.sqrt(var + eps)).reshape(batch_size_per_iter * num_qk_heads, 1)
        normed = qk_vals * rstd  # [batch*num_qk_heads, HEAD_DIM]

        # split Q and K
        normed_3d = normed.reshape(batch_size_per_iter, num_qk_heads, HEAD_DIM)
        q_normed = tl.extract_slice(normed_3d,
                                    offsets=(0, 0, 0),
                                    sizes=(batch_size_per_iter, num_q_heads, HEAD_DIM),
                                    strides=(1, 1, 1))
        k_normed = tl.extract_slice(normed_3d,
                                    offsets=(0, num_q_heads, 0),
                                    sizes=(batch_size_per_iter, num_kv_heads, HEAD_DIM),
                                    strides=(1, 1, 1))

        q_normed = q_normed * q_weight
        k_normed = k_normed * k_weight
        if BIAS:
            q_normed = q_normed + q_bias
            k_normed = k_normed + k_bias

        # load cos/sin for this batch: positions -> [batch_per_iter, HEAD_DIM]
        pos_idx = tl.load(pos_ptr + batch_idx, mask=bmask, other=0).to(tl.int64)
        cos_offsets = pos_idx[:, None] * HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)[None, :]
        sin_offsets = pos_idx[:, None] * HEAD_DIM + tl.arange(HALF_HEAD_DIM, HEAD_DIM)[None, :]
        cos = tl.load(cos_sin_ptr + cos_offsets, mask=bmask[:, None], other=0.0).to(tl.float32)
        sin = tl.load(cos_sin_ptr + sin_offsets, mask=bmask[:, None], other=0.0).to(tl.float32)
        # broadcast to [batch, 1, HALF_HEAD_DIM] for head dimension
        cos = cos.reshape(batch_size_per_iter, 1, HALF_HEAD_DIM)
        sin = sin.reshape(batch_size_per_iter, 1, HALF_HEAD_DIM)

        # RoPE for Q
        q_x1 = tl.extract_slice(q_normed,
                                 offsets=(0, 0, 0),
                                 sizes=(batch_size_per_iter, num_q_heads, HALF_HEAD_DIM),
                                 strides=(1, 1, 1))
        q_x2 = tl.extract_slice(q_normed,
                                 offsets=(0, 0, HALF_HEAD_DIM),
                                 sizes=(batch_size_per_iter, num_q_heads, HALF_HEAD_DIM),
                                 strides=(1, 1, 1))
        roped_q = tl.zeros((batch_size_per_iter, num_q_heads, HEAD_DIM), dtype=tl.bfloat16)
        roped_q = tl.insert_slice(roped_q,
                                  (q_x1 * cos - q_x2 * sin).to(tl.bfloat16),
                                  offsets=(0, 0, 0),
                                  sizes=(batch_size_per_iter, num_q_heads, HALF_HEAD_DIM),
                                  strides=(1, 1, 1))
        roped_q = tl.insert_slice(roped_q,
                                  (q_x2 * cos + q_x1 * sin).to(tl.bfloat16),
                                  offsets=(0, 0, HALF_HEAD_DIM),
                                  sizes=(batch_size_per_iter, num_q_heads, HALF_HEAD_DIM),
                                  strides=(1, 1, 1))

        q_out_offset = i * q_step_per_iter
        tl.store(q_ptr + q_out_idx + q_out_base + q_out_offset,
                 roped_q.reshape(q_step_per_iter),
                 mask=q_out_idx + q_out_base + q_out_offset < q_out_end)

        # RoPE for K
        k_x1 = tl.extract_slice(k_normed,
                                 offsets=(0, 0, 0),
                                 sizes=(batch_size_per_iter, num_kv_heads, HALF_HEAD_DIM),
                                 strides=(1, 1, 1))
        k_x2 = tl.extract_slice(k_normed,
                                 offsets=(0, 0, HALF_HEAD_DIM),
                                 sizes=(batch_size_per_iter, num_kv_heads, HALF_HEAD_DIM),
                                 strides=(1, 1, 1))
        roped_k = tl.zeros((batch_size_per_iter, num_kv_heads, HEAD_DIM), dtype=tl.bfloat16)
        roped_k = tl.insert_slice(roped_k,
                                  (k_x1 * cos - k_x2 * sin).to(tl.bfloat16),
                                  offsets=(0, 0, 0),
                                  sizes=(batch_size_per_iter, num_kv_heads, HALF_HEAD_DIM),
                                  strides=(1, 1, 1))
        roped_k = tl.insert_slice(roped_k,
                                  (k_x2 * cos + k_x1 * sin).to(tl.bfloat16),
                                  offsets=(0, 0, HALF_HEAD_DIM),
                                  sizes=(batch_size_per_iter, num_kv_heads, HALF_HEAD_DIM),
                                  strides=(1, 1, 1))

        kv_out_offset = i * kv_step_per_iter
        tl.store(k_ptr + kv_out_idx + kv_out_base + kv_out_offset,
                 roped_k.reshape(kv_step_per_iter),
                 mask=kv_out_idx + kv_out_base + kv_out_offset < kv_out_end)

    # V: independent loop with its own UB budget
    v_feat_idx = tl.arange(q_hidden_size + kv_hidden_size, total_hidden_size)
    v_feat_mask = v_feat_idx < total_hidden_size
    v_out_idx = tl.arange(0, kv_hidden_size)
    v_batch_idx = input_batch_base + tl.arange(0, v_batch_per_iter)

    for _ in tl.range(v_iter_num):
        v_bmask = v_batch_idx < input_batch_end
        v_mask2d = v_bmask[:, None] & v_feat_mask[None, :]
        v_idx2d = v_batch_idx[:, None] * total_hidden_size + v_feat_idx[None, :]
        v_vals = tl.load(input_ptr + v_idx2d, mask=v_mask2d, other=0.0)
        v_out_mask = v_bmask[:, None] & (v_out_idx[None, :] < kv_hidden_size)
        v_out_2d = v_batch_idx[:, None] * kv_hidden_size + v_out_idx[None, :]
        tl.store(v_ptr + v_out_2d, v_vals, mask=v_out_mask)
        v_batch_idx += v_batch_per_iter


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
    num_qk_heads = num_q_heads + num_kv_heads
    qk_hidden_size = q_hidden_size + kv_hidden_size
    half_head_dim = head_dim // 2

    core_num = get_vectorcore_num()
    batch_size_per_vec = triton.cdiv(batch_size, core_num)

    # UB capacity planning (85KB): account for all live buffers per iteration
    # qk_vals: batch*num_qk_heads*head_dim*4B, normed: same
    # q_normed/k_normed: batch*(num_q+num_kv)*head_dim*4B
    # roped_q/roped_k: batch*(num_q+num_kv)*head_dim*2B
    # cos/sin: batch*half_head_dim*4B*2
    elem_size = input.element_size()
    ub_bytes = 85 * 1024
    bytes_per_token = (
        num_qk_heads * head_dim * 4 * 2  # qk_vals + normed (fp32)
        + num_qk_heads * head_dim * 4    # q_normed + k_normed (fp32)
        + num_qk_heads * head_dim * elem_size  # roped_q + roped_k (bf16)
        + half_head_dim * 4 * 2          # cos + sin (fp32)
        + qk_hidden_size * elem_size     # input load
    )
    batch_size_per_iter = max(1, ub_bytes // bytes_per_token)
    batch_size_per_iter = min(batch_size_per_iter, batch_size_per_vec)
    iter_num = triton.cdiv(batch_size_per_vec, batch_size_per_iter)

    q_total_elems = batch_size * q_hidden_size
    kv_total_elems = batch_size * kv_hidden_size
    q_step_per_vec = batch_size_per_vec * q_hidden_size
    q_step_per_iter = batch_size_per_iter * q_hidden_size
    kv_step_per_vec = batch_size_per_vec * kv_hidden_size
    kv_step_per_iter = batch_size_per_iter * kv_hidden_size

    # V UB planning: only kv_hidden_size per token
    v_bytes_per_token = kv_hidden_size * elem_size * 2  # load + store
    v_batch_per_iter = max(1, ub_bytes // (v_bytes_per_token + 1))
    v_batch_per_iter = min(v_batch_per_iter, batch_size_per_vec)
    v_iter_num = triton.cdiv(batch_size_per_vec, v_batch_per_iter)

    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)

    BIAS = q_bias is not None
    grid = (min(core_num, batch_size), 1)

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
        qk_hidden_size,
        num_q_heads,
        num_kv_heads,
        num_qk_heads,
        eps,
        BIAS,
        head_dim,
        half_head_dim,
        int(batch_size_per_vec),
        int(batch_size_per_iter),
        int(iter_num),
        int(q_total_elems),
        int(kv_total_elems),
        int(q_step_per_vec),
        int(q_step_per_iter),
        int(kv_step_per_vec),
        int(kv_step_per_iter),
        int(v_batch_per_iter),
        int(v_iter_num),
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
