import torch
import torch.nn as nn
import triton
import triton.language as tl
from vllm_ascend.ops.triton.triton_utils import (
    extract_slice, get_vectorcore_num, init_device_properties_triton, insert_slice
)


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

    q_weight_values = tl.load(q_weight_ptr + tl.arange(0, HEAD_DIM))
    k_weight_values = tl.load(k_weight_ptr + tl.arange(0, HEAD_DIM))
    if BIAS:
        q_bias_values = tl.load(q_bias_ptr + tl.arange(0, HEAD_DIM))
        k_bias_values = tl.load(k_bias_ptr + tl.arange(0, HEAD_DIM))

    q_col_indices = col_pid * Q_BLOCK_SIZE + tl.arange(0, Q_BLOCK_SIZE)
    q_valid_mask = q_col_indices < q_hidden_size
    kv_col_indices = col_pid * KV_BLOCK_SIZE + tl.arange(0, KV_BLOCK_SIZE)
    kv_valid_mask = kv_col_indices < kv_hidden_size

    q_input_offset = row_pid * total_hidden_size
    q_output_offset = row_pid * q_hidden_size
    k_input_offset = row_pid * total_hidden_size + q_hidden_size
    kv_output_offset = row_pid * kv_hidden_size
    v_input_offset = row_pid * total_hidden_size + q_hidden_size + kv_hidden_size
    row_stride_total = row_step * total_hidden_size
    row_stride_q = row_step * q_hidden_size
    row_stride_kv = row_step * kv_hidden_size

    for row_idx in tl.range(row_pid, batch_size, row_step):
        pos_idx = tl.load(pos_ptr + row_idx).to(tl.int64)
        cos_offsets = pos_idx * HEAD_DIM + tl.arange(0, HALF_HEAD_DIM)
        sin_offsets = pos_idx * HEAD_DIM + tl.arange(HALF_HEAD_DIM, HEAD_DIM)
        cos = tl.load(cos_sin_ptr + cos_offsets).reshape(1, HALF_HEAD_DIM)
        sin = tl.load(cos_sin_ptr + sin_offsets).reshape(1, HALF_HEAD_DIM)

        # q
        q_in = (
            tl.load(input_ptr + q_input_offset + q_col_indices, mask=q_valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        q_var = tl.sum(q_in * q_in, axis=1) / HEAD_DIM
        q_rstd = (1 / tl.sqrt(q_var + eps)).reshape(Q_BLOCK_SIZE // HEAD_DIM, 1)
        q_norm = q_in * q_rstd
        if BIAS:
            q_norm = (q_norm * q_weight_values + q_bias_values).to(tl.bfloat16)
        else:
            q_norm = (q_norm * q_weight_values).to(tl.bfloat16)
        q_x1 = extract_slice(q_norm, offsets=(0, 0),
                              sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        q_x2 = extract_slice(q_norm, offsets=(0, HALF_HEAD_DIM),
                              sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        roped_q = tl.zeros((Q_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.bfloat16)
        roped_q = insert_slice(roped_q, q_x1 * cos - q_x2 * sin, offsets=(0, 0),
                               sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        roped_q = insert_slice(roped_q, q_x2 * cos + q_x1 * sin, offsets=(0, HALF_HEAD_DIM),
                               sizes=(Q_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        tl.store(q_ptr + q_output_offset + q_col_indices,
                 roped_q.reshape(Q_BLOCK_SIZE).to(q_ptr.dtype.element_ty), mask=q_valid_mask)

        # k
        k_in = (
            tl.load(input_ptr + k_input_offset + kv_col_indices, mask=kv_valid_mask, other=0.0)
            .to(tl.float32)
            .reshape(KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM)
        )
        k_var = tl.sum(k_in * k_in, axis=1) / HEAD_DIM
        k_rstd = (1 / tl.sqrt(k_var + eps)).reshape(KV_BLOCK_SIZE // HEAD_DIM, 1)
        k_norm = k_in * k_rstd
        if BIAS:
            k_norm = (k_norm * k_weight_values + k_bias_values).to(tl.bfloat16)
        else:
            k_norm = (k_norm * k_weight_values).to(tl.bfloat16)
        k_x1 = extract_slice(k_norm, offsets=(0, 0),
                              sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        k_x2 = extract_slice(k_norm, offsets=(0, HALF_HEAD_DIM),
                              sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        roped_k = tl.zeros((KV_BLOCK_SIZE // HEAD_DIM, HEAD_DIM), dtype=tl.bfloat16)
        roped_k = insert_slice(roped_k, k_x1 * cos - k_x2 * sin, offsets=(0, 0),
                               sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        roped_k = insert_slice(roped_k, k_x2 * cos + k_x1 * sin, offsets=(0, HALF_HEAD_DIM),
                               sizes=(KV_BLOCK_SIZE // HEAD_DIM, HALF_HEAD_DIM), strides=(1, 1))
        tl.store(k_ptr + kv_output_offset + kv_col_indices,
                 roped_k.reshape(KV_BLOCK_SIZE).to(tl.bfloat16), mask=kv_valid_mask)

        # v
        tl.store(v_ptr + kv_output_offset + kv_col_indices,
                 tl.load(input_ptr + v_input_offset + kv_col_indices, mask=kv_valid_mask, other=0.0),
                 mask=kv_valid_mask)

        q_input_offset += row_stride_total
        q_output_offset += row_stride_q
        k_input_offset += row_stride_total
        kv_output_offset += row_stride_kv
        v_input_offset += row_stride_total


class ModelNew(nn.Module):
    def __init__(self, num_q_heads, num_kv_heads, head_size, eps, max_pos):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.eps = eps

    def forward(self, qkv, q_weight, k_weight, cos_sin_cache, positions):
        init_device_properties_triton()
        T = qkv.shape[0]
        H = self.head_size
        nq, nkv = self.num_q_heads, self.num_kv_heads
        q_size, kv_size = nq * H, nkv * H
        total = q_size + kv_size * 2

        q_out = torch.empty(T, q_size, device=qkv.device, dtype=qkv.dtype)
        k_out = torch.empty(T, kv_size, device=qkv.device, dtype=qkv.dtype)
        v_out = torch.empty(T, kv_size, device=qkv.device, dtype=qkv.dtype)

        KV_BLOCK_SIZE = triton.next_power_of_2(H)
        Q_BLOCK_SIZE = q_size // kv_size * H
        n_cols = kv_size // KV_BLOCK_SIZE
        num_vectorcore = get_vectorcore_num()
        n_rows = num_vectorcore // n_cols

        split_qkv_rmsnorm_rope_kernel[(n_rows, n_cols, 1)](
            qkv, cos_sin_cache, positions,
            q_out, k_out, v_out,
            q_weight, None, k_weight, None,
            T, q_size, kv_size, total, self.eps,
            Q_BLOCK_SIZE, KV_BLOCK_SIZE, False, H, H // 2,
            multibuffer=True,
        )
        return q_out, k_out, v_out
