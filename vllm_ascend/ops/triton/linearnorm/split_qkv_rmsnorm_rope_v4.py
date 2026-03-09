import math
import torch
import torch_npu
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


_MERGE_MASK_CACHE = {}


def _calc_tokens_per_iter(num_q_heads, num_kv_heads, head_size, rope_dim):
    UB_CAPACITY = 192 * 1024
    half_rope_dim = rope_dim // 2
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    per_token_input = (q_size + kv_size + kv_size) * 2
    per_token_f32 = (q_size + kv_size) * 4
    per_token_cos_sin = half_rope_dim * 4 * 2
    per_token_rope = (num_q_heads + num_kv_heads) * half_rope_dim * 4 * 4
    per_token_output = (q_size + kv_size) * 2
    per_token_total = int(1.5 * (per_token_input + per_token_f32 + per_token_cos_sin
                       + per_token_rope + per_token_output))
    shared = head_size * 2 * 4
    available = UB_CAPACITY - shared
    max_tokens = available // per_token_total
    if max_tokens >= 4:
        return 4
    return 2


def _get_merge_mask(mrope_section, is_interleaved, rope_dim, device):
    key = (tuple(mrope_section), is_interleaved, rope_dim, device)
    if key not in _MERGE_MASK_CACHE:
        hrd = rope_dim // 2
        idx = torch.arange(hrd, device=device)
        if is_interleaved:
            hm = ((idx % 3) == 1) & (idx <= 3 * mrope_section[1])
            wm = ((idx % 3) == 2) & (idx <= 3 * mrope_section[2])
            tm = ~(hm | wm)
        else:
            tm = idx < mrope_section[0]
            hm = (mrope_section[0] - 1 < idx) & (idx < mrope_section[0] + mrope_section[1])
            wm = (mrope_section[0] + mrope_section[1] - 1 < idx) & (idx < sum(mrope_section))
        _MERGE_MASK_CACHE[key] = torch.stack([tm, hm, wm]).float().unsqueeze(1)
    return _MERGE_MASK_CACHE[key]


def _merge_cos_sin(cos_sin, mrope_section, is_interleaved, num_tokens, rope_dim):
    hrd = rope_dim // 2
    mask = _get_merge_mask(mrope_section, is_interleaved, rope_dim, cos_sin.device)
    cos_all = cos_sin[:, :, :hrd].float()
    sin_all = cos_sin[:, :, hrd:].float()
    mc = (cos_all * mask).sum(dim=0).contiguous()
    ms = (sin_all * mask).sum(dim=0).contiguous()
    return mc, ms


@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num",
                       "num_tokens_each_front_core",
                       "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel_v4(
    in_qkv_ptr, q_weight_ptr, q_bias_ptr, k_weight_ptr, k_bias_ptr,
    merged_cos_ptr, merged_sin_ptr,
    out_q_ptr, out_k_ptr, out_v_ptr,
    num_tokens, front_core_num,
    num_tokens_each_front_core, num_tokens_each_tail_core,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    head_size: tl.constexpr, q_size: tl.constexpr, kv_size: tl.constexpr,
    qkv_size: tl.constexpr, eps: tl.constexpr, has_bias: tl.constexpr,
    rope_dim: tl.constexpr, half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
    TOKENS_PER_ITER: tl.constexpr,
):
    block_idx = tl.program_id(0)
    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core
    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = (num_tokens_each_front_core * front_core_num
                        + (block_idx - front_core_num)
                        * num_tokens_each_tail_core)

    q_w = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_w = tl.load(k_weight_ptr + tl.arange(0, head_size))
    if has_bias:
        q_b = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_b = tl.load(k_bias_ptr + tl.arange(0, head_size))
    ca = tl.arange(0, half_rope_dim)
    qa = tl.arange(0, q_size)
    ka = tl.arange(0, kv_size)

    n4 = loop_num // TOKENS_PER_ITER
    rem = loop_num - n4 * TOKENS_PER_ITER

    for quad in range(n4):
        t0 = block_offset + quad * TOKENS_PER_ITER

        # LOAD PHASE
        q_0 = tl.load(in_qkv_ptr + t0 * qkv_size + qa).reshape(num_q_heads, head_size)
        q_1 = tl.load(in_qkv_ptr + (t0 + 1) * qkv_size + qa).reshape(num_q_heads, head_size)
        k_0 = tl.load(in_qkv_ptr + t0 * qkv_size + q_size + ka).reshape(num_kv_heads, head_size)
        k_1 = tl.load(in_qkv_ptr + (t0 + 1) * qkv_size + q_size + ka).reshape(num_kv_heads, head_size)
        v_0 = tl.load(in_qkv_ptr + t0 * qkv_size + q_size + kv_size + ka)
        tl.store(out_v_ptr + t0 * kv_size + ka, v_0)
        v_1 = tl.load(in_qkv_ptr + (t0 + 1) * qkv_size + q_size + kv_size + ka)
        tl.store(out_v_ptr + (t0 + 1) * kv_size + ka, v_1)
        c_0 = tl.load(merged_cos_ptr + t0 * half_rope_dim + ca).reshape(1, half_rope_dim)
        s_0 = tl.load(merged_sin_ptr + t0 * half_rope_dim + ca).reshape(1, half_rope_dim)
        c_1 = tl.load(merged_cos_ptr + (t0 + 1) * half_rope_dim + ca).reshape(1, half_rope_dim)
        s_1 = tl.load(merged_sin_ptr + (t0 + 1) * half_rope_dim + ca).reshape(1, half_rope_dim)

        if TOKENS_PER_ITER >= 4:
            q_2 = tl.load(in_qkv_ptr + (t0 + 2) * qkv_size + qa).reshape(num_q_heads, head_size)
            q_3 = tl.load(in_qkv_ptr + (t0 + 3) * qkv_size + qa).reshape(num_q_heads, head_size)
            k_2 = tl.load(in_qkv_ptr + (t0 + 2) * qkv_size + q_size + ka).reshape(num_kv_heads, head_size)
            k_3 = tl.load(in_qkv_ptr + (t0 + 3) * qkv_size + q_size + ka).reshape(num_kv_heads, head_size)
            v_2 = tl.load(in_qkv_ptr + (t0 + 2) * qkv_size + q_size + kv_size + ka)
            tl.store(out_v_ptr + (t0 + 2) * kv_size + ka, v_2)
            v_3 = tl.load(in_qkv_ptr + (t0 + 3) * qkv_size + q_size + kv_size + ka)
            tl.store(out_v_ptr + (t0 + 3) * kv_size + ka, v_3)
            c_2 = tl.load(merged_cos_ptr + (t0 + 2) * half_rope_dim + ca).reshape(1, half_rope_dim)
            s_2 = tl.load(merged_sin_ptr + (t0 + 2) * half_rope_dim + ca).reshape(1, half_rope_dim)
            c_3 = tl.load(merged_cos_ptr + (t0 + 3) * half_rope_dim + ca).reshape(1, half_rope_dim)
            s_3 = tl.load(merged_sin_ptr + (t0 + 3) * half_rope_dim + ca).reshape(1, half_rope_dim)

        # RMSNORM
        sq = q_0.to(tl.float32) * q_0
        var = tl.sum(sq, axis=1) / head_size
        q_0 = q_0.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
        if has_bias:
            q_0 = q_0 + q_b
        sq = q_1.to(tl.float32) * q_1
        var = tl.sum(sq, axis=1) / head_size
        q_1 = q_1.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
        if has_bias:
            q_1 = q_1 + q_b
        sq = k_0.to(tl.float32) * k_0
        var = tl.sum(sq, axis=1) / head_size
        k_0 = k_0.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
        if has_bias:
            k_0 = k_0 + k_b
        sq = k_1.to(tl.float32) * k_1
        var = tl.sum(sq, axis=1) / head_size
        k_1 = k_1.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
        if has_bias:
            k_1 = k_1 + k_b

        if TOKENS_PER_ITER >= 4:
            sq = q_2.to(tl.float32) * q_2
            var = tl.sum(sq, axis=1) / head_size
            q_2 = q_2.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
            if has_bias:
                q_2 = q_2 + q_b
            sq = q_3.to(tl.float32) * q_3
            var = tl.sum(sq, axis=1) / head_size
            q_3 = q_3.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
            if has_bias:
                q_3 = q_3 + q_b
            sq = k_2.to(tl.float32) * k_2
            var = tl.sum(sq, axis=1) / head_size
            k_2 = k_2.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
            if has_bias:
                k_2 = k_2 + k_b
            sq = k_3.to(tl.float32) * k_3
            var = tl.sum(sq, axis=1) / head_size
            k_3 = k_3.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
            if has_bias:
                k_3 = k_3 + k_b

        # ROPE Q + STORE
        x1 = tl.extract_slice(q_0, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_0, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        r1 = x1 * c_0 - x2 * s_0
        r2 = x2 * c_0 + x1 * s_0
        if IS_PARTIAL_ROPE:
            q_0 = tl.insert_slice(q_0, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_0 = tl.insert_slice(q_0, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            q_0 = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            q_0 = tl.insert_slice(q_0, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_0 = tl.insert_slice(q_0, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_q_ptr + t0 * q_size + qa, q_0.reshape(q_size))

        x1 = tl.extract_slice(q_1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_1, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        r1 = x1 * c_1 - x2 * s_1
        r2 = x2 * c_1 + x1 * s_1
        if IS_PARTIAL_ROPE:
            q_1 = tl.insert_slice(q_1, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_1 = tl.insert_slice(q_1, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            q_1 = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            q_1 = tl.insert_slice(q_1, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_1 = tl.insert_slice(q_1, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_q_ptr + (t0 + 1) * q_size + qa, q_1.reshape(q_size))

        if TOKENS_PER_ITER >= 4:
            x1 = tl.extract_slice(q_2, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            x2 = tl.extract_slice(q_2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            r1 = x1 * c_2 - x2 * s_2
            r2 = x2 * c_2 + x1 * s_2
            if IS_PARTIAL_ROPE:
                q_2 = tl.insert_slice(q_2, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
                q_2 = tl.insert_slice(q_2, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            else:
                q_2 = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
                q_2 = tl.insert_slice(q_2, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
                q_2 = tl.insert_slice(q_2, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            tl.store(out_q_ptr + (t0 + 2) * q_size + qa, q_2.reshape(q_size))

            x1 = tl.extract_slice(q_3, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            x2 = tl.extract_slice(q_3, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            r1 = x1 * c_3 - x2 * s_3
            r2 = x2 * c_3 + x1 * s_3
            if IS_PARTIAL_ROPE:
                q_3 = tl.insert_slice(q_3, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
                q_3 = tl.insert_slice(q_3, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            else:
                q_3 = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
                q_3 = tl.insert_slice(q_3, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
                q_3 = tl.insert_slice(q_3, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            tl.store(out_q_ptr + (t0 + 3) * q_size + qa, q_3.reshape(q_size))

        # ROPE K + STORE
        y1 = tl.extract_slice(k_0, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_0, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        r1 = y1 * c_0 - y2 * s_0
        r2 = y2 * c_0 + y1 * s_0
        if IS_PARTIAL_ROPE:
            k_0 = tl.insert_slice(k_0, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_0 = tl.insert_slice(k_0, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            k_0 = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            k_0 = tl.insert_slice(k_0, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_0 = tl.insert_slice(k_0, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_k_ptr + t0 * kv_size + ka, k_0.reshape(kv_size))

        y1 = tl.extract_slice(k_1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_1, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        r1 = y1 * c_1 - y2 * s_1
        r2 = y2 * c_1 + y1 * s_1
        if IS_PARTIAL_ROPE:
            k_1 = tl.insert_slice(k_1, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_1 = tl.insert_slice(k_1, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            k_1 = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            k_1 = tl.insert_slice(k_1, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_1 = tl.insert_slice(k_1, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_k_ptr + (t0 + 1) * kv_size + ka, k_1.reshape(kv_size))

        if TOKENS_PER_ITER >= 4:
            y1 = tl.extract_slice(k_2, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            y2 = tl.extract_slice(k_2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            r1 = y1 * c_2 - y2 * s_2
            r2 = y2 * c_2 + y1 * s_2
            if IS_PARTIAL_ROPE:
                k_2 = tl.insert_slice(k_2, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
                k_2 = tl.insert_slice(k_2, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            else:
                k_2 = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
                k_2 = tl.insert_slice(k_2, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
                k_2 = tl.insert_slice(k_2, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            tl.store(out_k_ptr + (t0 + 2) * kv_size + ka, k_2.reshape(kv_size))

            y1 = tl.extract_slice(k_3, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            y2 = tl.extract_slice(k_3, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            r1 = y1 * c_3 - y2 * s_3
            r2 = y2 * c_3 + y1 * s_3
            if IS_PARTIAL_ROPE:
                k_3 = tl.insert_slice(k_3, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
                k_3 = tl.insert_slice(k_3, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            else:
                k_3 = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
                k_3 = tl.insert_slice(k_3, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
                k_3 = tl.insert_slice(k_3, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
            tl.store(out_k_ptr + (t0 + 3) * kv_size + ka, k_3.reshape(kv_size))

    for r_idx in range(rem):
        tidx = block_offset + n4 * TOKENS_PER_ITER + r_idx
        base_in = in_qkv_ptr + tidx * qkv_size
        qt = tl.load(base_in + qa).reshape(num_q_heads, head_size)
        kt = tl.load(base_in + q_size + ka).reshape(num_kv_heads, head_size)
        vt = tl.load(base_in + q_size + kv_size + ka)
        tl.store(out_v_ptr + tidx * kv_size + ka, vt)
        ch = tl.load(merged_cos_ptr + tidx * half_rope_dim + ca).reshape(1, half_rope_dim)
        sh = tl.load(merged_sin_ptr + tidx * half_rope_dim + ca).reshape(1, half_rope_dim)
        sq = qt.to(tl.float32) * qt
        var = tl.sum(sq, axis=1) / head_size
        qt = qt.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
        if has_bias:
            qt = qt + q_b
        sq = kt.to(tl.float32) * kt
        var = tl.sum(sq, axis=1) / head_size
        kt = kt.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
        if has_bias:
            kt = kt + k_b
        x1 = tl.extract_slice(qt, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(qt, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        r1 = x1 * ch - x2 * sh
        r2 = x2 * ch + x1 * sh
        if IS_PARTIAL_ROPE:
            qt = tl.insert_slice(qt, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            qt = tl.insert_slice(qt, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            qt = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            qt = tl.insert_slice(qt, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            qt = tl.insert_slice(qt, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_q_ptr + tidx * q_size + qa, qt.reshape(q_size))
        y1 = tl.extract_slice(kt, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(kt, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        r1 = y1 * ch - y2 * sh
        r2 = y2 * ch + y1 * sh
        if IS_PARTIAL_ROPE:
            kt = tl.insert_slice(kt, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            kt = tl.insert_slice(kt, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            kt = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            kt = tl.insert_slice(kt, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            kt = tl.insert_slice(kt, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_k_ptr + tidx * kv_size + ka, kt.reshape(kv_size))


def triton_split_qkv_rmsnorm_mrope(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin: torch.Tensor, num_q_heads: int, num_kv_heads: int,
    head_size: int, eps: float, mrope_section: list[int],
    is_interleaved: bool, rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None, k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    core_num = get_vectorcore_num()
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]
    if rope_dim is None:
        rope_dim = head_size
    IS_PARTIAL_ROPE = rope_dim != head_size
    TOKENS_PER_ITER = _calc_tokens_per_iter(num_q_heads, num_kv_heads, head_size, rope_dim)
    front_core_num = core_num
    if num_tokens % core_num != 0:
        front_core_num = num_tokens % core_num
    nf = (num_tokens + core_num - 1) // core_num
    tail_core_num = 0
    if num_tokens > core_num:
        tail_core_num = core_num - front_core_num
    nt = num_tokens // core_num
    qo = torch.empty(num_tokens, q_size, device=qkv.device, dtype=qkv.dtype)
    ko = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    vo = torch.empty(num_tokens, kv_size, device=qkv.device, dtype=qkv.dtype)
    bd = min(core_num, front_core_num + tail_core_num)
    mc, ms = _merge_cos_sin(cos_sin, mrope_section, is_interleaved, num_tokens, rope_dim)
    split_qkv_rmsnorm_mrope_kernel_v4[(bd,)](
        qkv, q_weight, q_bias, k_weight, k_bias, mc, ms, qo, ko, vo,
        num_tokens, front_core_num, nf, nt,
        num_q_heads, num_kv_heads, head_size, q_size, kv_size,
        q_size + 2 * kv_size,
        eps, q_bias is not None, rope_dim, rope_dim // 2, IS_PARTIAL_ROPE,
        TOKENS_PER_ITER,
    )
    return qo, ko, vo


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor,
    cos_sin: torch.Tensor, num_q_heads: int, num_kv_heads: int,
    head_size: int, eps: float, mrope_section: list[int],
    is_interleaved: bool = False, rope_dim: int | None = None,
    q_bias: torch.Tensor | None = None, k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = qkv.shape[0]
    qs = num_q_heads * head_size
    ks = num_kv_heads * head_size
    return (torch.empty(n, qs, device=qkv.device, dtype=qkv.dtype),
            torch.empty(n, ks, device=qkv.device, dtype=qkv.dtype),
            torch.empty(n, ks, device=qkv.device, dtype=qkv.dtype))


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
