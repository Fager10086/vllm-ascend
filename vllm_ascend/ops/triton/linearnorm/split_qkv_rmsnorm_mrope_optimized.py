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


import math

import torch
import triton  # type: ignore
import triton.language as tl  # type: ignore
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

ASCEND_UB_BYTES = 256 * 1024
UB_SAFETY_FACTOR = 0.8


def compute_tile_tokens(
    num_q_heads: int,
    num_kv_heads: int,
    head_size: int,
    rope_dim: int,
    ub_bytes: int = ASCEND_UB_BYTES,
) -> int:
    """Dynamically compute how many tokens one vector-core can process
    per iteration based on kernel UB footprint and hardware UB capacity.

    The estimation enumerates all tensors that are simultaneously live in
    UB at peak usage (Q rmsnorm + rope phase while K data is waiting):

    Per-token live tensors (bytes):
      - Q input fp32:               q_size * 4
      - Q normalized fp32:          q_size * 4
      - K input fp32 (waiting):     kv_size * 4
      - V passthrough bf16:         kv_size * 2
      - cos/sin fp32:               rope_dim * 4 * 2
      - rope scratch (cat_x) fp32:  num_q_heads * rope_dim * 4
      - output Q bf16:              q_size * 2
      - output K bf16:              kv_size * 2
      - output V bf16:              kv_size * 2

    Shared (loop-invariant, loaded once):
      - rmsnorm weights fp32:       head_size * 4 * 2
      - biases fp32 (worst case):   head_size * 4 * 2
    """
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    shared_bytes = head_size * 4 * 4

    per_token_bytes = (
        q_size * 4                      # Q input fp32
        + q_size * 4                    # Q normalized fp32
        + kv_size * 4                   # K input fp32 (loaded, waiting)
        + kv_size * 2                   # V passthrough bf16
        + rope_dim * 4 * 2              # cos + sin fp32
        + num_q_heads * rope_dim * 4    # rope scratch (cat_x) fp32
        + q_size * 2                    # output Q bf16
        + kv_size * 2                   # output K bf16
        + kv_size * 2                   # output V bf16
    )

    available = int((ub_bytes - shared_bytes) * UB_SAFETY_FACTOR)
    tile = max(1, available // per_token_bytes)
    tile = 1 << int(math.log2(tile))
    return max(1, tile)


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
    TILE_TOKENS: tl.constexpr,
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

    cos_offsets = tl.arange(0, half_rope_dim)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets < mrope_section_t
        h_mask = (
            (mrope_section_t - 1 < cos_offsets)
            & (cos_offsets < mrope_section_t + mrope_section_h)
        )
        w_mask = (
            (mrope_section_t + mrope_section_h - 1 < cos_offsets)
            & (cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w)
        )

    TILE_Q_HEADS: tl.constexpr = TILE_TOKENS * num_q_heads
    TILE_KV_HEADS: tl.constexpr = TILE_TOKENS * num_kv_heads
    qkv_stride: tl.constexpr = q_size + 2 * kv_size

    tile_offsets = tl.arange(0, TILE_TOKENS)
    q_elem = tl.arange(0, q_size)
    kv_elem = tl.arange(0, kv_size)

    for tile_start in range(0, loop_num, TILE_TOKENS):
        valid = (tile_start + tile_offsets) < loop_num
        token_ids = block_offset + tile_start + tile_offsets

        # ========== Load Q (TILE_TOKENS, q_size) -> (TILE_Q_HEADS, head_size) ==========
        q_ptrs = in_qkv_ptr + token_ids[:, None] * qkv_stride + q_elem[None, :]
        in_q = tl.load(q_ptrs, mask=valid[:, None], other=0).to(tl.float32)
        in_q = in_q.reshape(TILE_Q_HEADS, head_size)

        # ========== Load K (TILE_TOKENS, kv_size) -> (TILE_KV_HEADS, head_size) ==========
        k_ptrs = in_qkv_ptr + token_ids[:, None] * qkv_stride + q_size + kv_elem[None, :]
        in_k = tl.load(k_ptrs, mask=valid[:, None], other=0).to(tl.float32)
        in_k = in_k.reshape(TILE_KV_HEADS, head_size)

        # ========== Load & Store V (passthrough) ==========
        v_ptrs = in_qkv_ptr + token_ids[:, None] * qkv_stride + q_size + kv_size + kv_elem[None, :]
        in_v = tl.load(v_ptrs, mask=valid[:, None], other=0)
        v_out_ptrs = out_v_ptr + token_ids[:, None] * kv_size + kv_elem[None, :]
        tl.store(v_out_ptrs, in_v, mask=valid[:, None])

        # ========== Load cos/sin for tile (TILE_TOKENS, half_rope_dim) ==========
        cos_bases = token_ids * rope_dim
        t_cos_2d = cos_sin_ptr + cos_bases[:, None] + cos_offsets[None, :]
        h_cos_2d = t_cos_2d + num_tokens * rope_dim
        w_cos_2d = h_cos_2d + num_tokens * rope_dim
        t_sin_2d = cos_sin_ptr + cos_bases[:, None] + half_rope_dim + cos_offsets[None, :]
        h_sin_2d = t_sin_2d + num_tokens * rope_dim
        w_sin_2d = h_sin_2d + num_tokens * rope_dim

        vm_t = valid[:, None] & t_mask[None, :]
        vm_h = valid[:, None] & h_mask[None, :]
        vm_w = valid[:, None] & w_mask[None, :]

        cos_half = (
            tl.load(t_cos_2d, mask=vm_t, other=0)
            + tl.load(h_cos_2d, mask=vm_h, other=0)
            + tl.load(w_cos_2d, mask=vm_w, other=0)
        ).to(tl.float32)
        sin_half = (
            tl.load(t_sin_2d, mask=vm_t, other=0)
            + tl.load(h_sin_2d, mask=vm_h, other=0)
            + tl.load(w_sin_2d, mask=vm_w, other=0)
        ).to(tl.float32)

        cos_tile = tl.broadcast_to(
            cos_half.reshape(TILE_TOKENS, 1, half_rope_dim),
            (TILE_TOKENS, 2, half_rope_dim),
        ).reshape(TILE_TOKENS, rope_dim)
        sin_tile = tl.broadcast_to(
            sin_half.reshape(TILE_TOKENS, 1, half_rope_dim),
            (TILE_TOKENS, 2, half_rope_dim),
        ).reshape(TILE_TOKENS, rope_dim)

        q_cos = tl.broadcast_to(
            cos_tile.reshape(TILE_TOKENS, 1, rope_dim),
            (TILE_TOKENS, num_q_heads, rope_dim),
        ).reshape(TILE_Q_HEADS, rope_dim)
        q_sin = tl.broadcast_to(
            sin_tile.reshape(TILE_TOKENS, 1, rope_dim),
            (TILE_TOKENS, num_q_heads, rope_dim),
        ).reshape(TILE_Q_HEADS, rope_dim)
        k_cos = tl.broadcast_to(
            cos_tile.reshape(TILE_TOKENS, 1, rope_dim),
            (TILE_TOKENS, num_kv_heads, rope_dim),
        ).reshape(TILE_KV_HEADS, rope_dim)
        k_sin = tl.broadcast_to(
            sin_tile.reshape(TILE_TOKENS, 1, rope_dim),
            (TILE_TOKENS, num_kv_heads, rope_dim),
        ).reshape(TILE_KV_HEADS, rope_dim)

        # ========== Q RMSNorm ==========
        sq = in_q * in_q
        var_q = tl.sum(sq, axis=1) / head_size
        rstd_q = (1 / tl.sqrt(var_q + eps)).reshape(TILE_Q_HEADS, 1)
        q_norm = in_q * rstd_q * q_rmsnorm_weight
        if has_bias:
            q_norm = q_norm + q_bias

        # ========== K RMSNorm ==========
        sk = in_k * in_k
        var_k = tl.sum(sk, axis=1) / head_size
        rstd_k = (1 / tl.sqrt(var_k + eps)).reshape(TILE_KV_HEADS, 1)
        k_norm = in_k * rstd_k * k_rmsnorm_weight
        if has_bias:
            k_norm = k_norm + k_bias

        # ========== Q MRoPE ==========
        qx1 = tl.extract_slice(q_norm, (0, 0), (TILE_Q_HEADS, half_rope_dim), (1, 1))
        qx2 = tl.extract_slice(q_norm, (0, half_rope_dim), (TILE_Q_HEADS, half_rope_dim), (1, 1))
        cat_q = tl.zeros((TILE_Q_HEADS, rope_dim), dtype=tl.float32)
        cat_q = tl.insert_slice(cat_q, -qx2, (0, 0), (TILE_Q_HEADS, half_rope_dim), (1, 1))
        cat_q = tl.insert_slice(cat_q, qx1, (0, half_rope_dim), (TILE_Q_HEADS, half_rope_dim), (1, 1))
        if IS_PARTIAL_ROPE:
            orig_q = tl.extract_slice(q_norm, (0, 0), (TILE_Q_HEADS, rope_dim), (1, 1))
        else:
            orig_q = q_norm
        roped_q = cat_q * q_sin + orig_q * q_cos

        # ========== K MRoPE ==========
        ky1 = tl.extract_slice(k_norm, (0, 0), (TILE_KV_HEADS, half_rope_dim), (1, 1))
        ky2 = tl.extract_slice(k_norm, (0, half_rope_dim), (TILE_KV_HEADS, half_rope_dim), (1, 1))
        cat_k = tl.zeros((TILE_KV_HEADS, rope_dim), dtype=tl.float32)
        cat_k = tl.insert_slice(cat_k, -ky2, (0, 0), (TILE_KV_HEADS, half_rope_dim), (1, 1))
        cat_k = tl.insert_slice(cat_k, ky1, (0, half_rope_dim), (TILE_KV_HEADS, half_rope_dim), (1, 1))
        if IS_PARTIAL_ROPE:
            orig_k = tl.extract_slice(k_norm, (0, 0), (TILE_KV_HEADS, rope_dim), (1, 1))
        else:
            orig_k = k_norm
        roped_k = cat_k * k_sin + orig_k * k_cos

        # ========== Apply partial rope & dtype cast ==========
        if IS_PARTIAL_ROPE:
            q_norm = tl.insert_slice(
                q_norm, roped_q, (0, 0), (TILE_Q_HEADS, rope_dim), (1, 1),
            ).to(tl.bfloat16)
            k_norm = tl.insert_slice(
                k_norm, roped_k, (0, 0), (TILE_KV_HEADS, rope_dim), (1, 1),
            ).to(tl.bfloat16)
        else:
            q_norm = roped_q.to(tl.bfloat16)
            k_norm = roped_k.to(tl.bfloat16)

        # ========== Store Q ==========
        q_out = q_norm.reshape(TILE_TOKENS, q_size)
        q_out_ptrs = out_q_ptr + token_ids[:, None] * q_size + q_elem[None, :]
        tl.store(q_out_ptrs, q_out, mask=valid[:, None])

        # ========== Store K ==========
        k_out = k_norm.reshape(TILE_TOKENS, kv_size)
        k_out_ptrs = out_k_ptr + token_ids[:, None] * kv_size + kv_elem[None, :]
        tl.store(k_out_ptrs, k_out, mask=valid[:, None])


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

    tile_tokens = compute_tile_tokens(num_q_heads, num_kv_heads, head_size, rope_dim)

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
        tile_tokens,
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
