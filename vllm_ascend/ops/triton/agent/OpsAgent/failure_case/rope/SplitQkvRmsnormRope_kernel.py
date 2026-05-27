import json
import os
import torch
import torch.nn as nn
import triton
import triton.language as tl


def _get_vectorcore_num():
    device_properties = triton.runtime.driver.active.utils.get_device_properties(
        torch.npu.current_device()
    )
    num_vectorcore = device_properties.get("num_vectorcore", -1)
    assert num_vectorcore > 0, "Failed to detect vectorcore count."
    return num_vectorcore


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
    eps: tl.constexpr,
    Q_BLOCK_SIZE: tl.constexpr,
    KV_BLOCK_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    HALF_HEAD_DIM: tl.constexpr,
):
    row_pid = tl.program_id(0)
    col_pid = tl.program_id(1)
    row_step = tl.num_programs(0)
    out_dtype = q_ptr.dtype.element_ty
    # q
    weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size
    output_offset = row_pid * q_hidden_size
    input_offset_step = row_step * total_hidden_size
    output_offset_step = row_step * q_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
        valid_mask = col_indices < q_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std  # (Q_BLOCK_SIZE//HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = normalized_values * weight_values + bias_values
        else:
            normalized_values = normalized_values * weight_values

        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_offsets = pos_idx * HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)
        sin_offsets = pos_idx * HEAD_DIM + tl.arange(HALF_HEAD_DIM, HEAD_DIM)
        cos = (tl.load(cos_sin_ptr + cos_offsets)).to(tl.float32).reshape(1, HALF_HEAD_DIM)
        sin = (tl.load(cos_sin_ptr + sin_offsets)).to(tl.float32).reshape(1, HALF_HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_q1 = x1 * cos - x2 * sin
        roped_q2 = x2 * cos + x1 * sin

        roped_q = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.float32)
        roped_q = tl.insert_slice(
            roped_q,
            roped_q1,
            offsets=(0, 0),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_q = tl.insert_slice(
            roped_q,
            roped_q2,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        tl.store(
            q_ptr + output_offset + col_indices,
            roped_q.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))
    input_offset = row_pid * total_hidden_size + q_hidden_size
    output_offset = row_pid * kv_hidden_size
    output_offset_step = row_step * kv_hidden_size
    for row_idx in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = (
            tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        squares = input_values * input_values
        variances = tl.sum(squares, axis=1) / HEAD_DIM
        reciprocal_std = (1 / tl.sqrt(variances + eps)).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        normalized_values = input_values * reciprocal_std  # (KV_BLOCK_SIZE/HEAD_DIM, HEAD_DIM)
        if BIAS:
            normalized_values = normalized_values * weight_values + bias_values
        else:
            normalized_values = normalized_values * weight_values

        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_offsets = pos_idx * HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)
        sin_offsets = pos_idx * HEAD_DIM + tl.arange(HALF_HEAD_DIM, HEAD_DIM)
        cos = (tl.load(cos_sin_ptr + cos_offsets)).to(tl.float32).reshape(1, HALF_HEAD_DIM)
        sin = (tl.load(cos_sin_ptr + sin_offsets)).to(tl.float32).reshape(1, HALF_HEAD_DIM)
        x1 = tl.extract_slice(
            normalized_values,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        x2 = tl.extract_slice(
            normalized_values,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_k1 = x1 * cos - x2 * sin
        roped_k2 = x2 * cos + x1 * sin

        roped_k = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.float32)
        roped_k = tl.insert_slice(
            roped_k,
            roped_k1,
            offsets=(0, 0),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        roped_k = tl.insert_slice(
            roped_k,
            roped_k2,
            offsets=(0, HALF_HEAD_DIM),
            sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM),
            strides=(1, 1),
        )
        tl.store(
            k_ptr + output_offset + col_indices,
            roped_k.reshape(KV_BLOCK_SIZE).to(k_ptr.dtype.element_ty),
            mask=valid_mask,
        )
        input_offset += input_offset_step
        output_offset += output_offset_step

    input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    output_offset = row_pid * kv_hidden_size
    for _ in tl.range(row_pid, batch_size, row_step):
        col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
        valid_mask = col_indices < kv_hidden_size
        input_values = tl.load(input_ptr + input_offset + col_indices, mask=valid_mask, other=0.0)
        tl.store(v_ptr + output_offset + col_indices, input_values, mask=valid_mask)
        input_offset += input_offset_step
        output_offset += output_offset_step


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
    KV_BLOCK_SIZE = triton.next_power_of_2(head_dim)
    assert head_dim == KV_BLOCK_SIZE
    assert q_hidden_size % kv_hidden_size == 0
    Q_BLOCK_SIZE = q_hidden_size // kv_hidden_size * head_dim
    batch_size = input.shape[0]
    total_hidden_size = q_hidden_size + kv_hidden_size * 2
    q_output = torch.empty(batch_size, q_hidden_size, device=input.device, dtype=input.dtype)
    k_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    v_output = torch.empty(batch_size, kv_hidden_size, device=input.device, dtype=input.dtype)
    n_cols = kv_hidden_size // KV_BLOCK_SIZE
    num_vectorcore = _get_vectorcore_num()
    assert num_vectorcore % n_cols == 0
    n_rows = num_vectorcore // n_cols
    BIAS = q_bias is not None

    split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
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
        Q_BLOCK_SIZE,
        KV_BLOCK_SIZE,
        BIAS,
        head_dim,
        head_dim // 2,
    )
    return q_output, k_output, v_output

class ModelNew(nn.Module):
    """
    Triton kernel implementation of fused split-QKV + RMSNorm + RoPE.

    Replaces the pure PyTorch reference with an Ascend Triton kernel that
    processes Q (RMSNorm + RoPE), K (RMSNorm + RoPE), and V (passthrough)
    in a single fused kernel launch.
    """
    def __init__(self):
        super(ModelNew, self).__init__()

    def forward(
        self,
        input: torch.Tensor,
        cos_sin_cache: torch.Tensor,
        positions: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        q_hidden_size: int,
        kv_hidden_size: int,
        head_dim: int,
        eps: float,
        q_bias: torch.Tensor = None,
        k_bias: torch.Tensor = None,
    ) -> tuple:
        return split_qkv_rmsnorm_rope_impl(
            input, cos_sin_cache, positions,
            q_weight, k_weight,
            q_hidden_size, kv_hidden_size, head_dim, eps,
            q_bias, k_bias,
        )
