import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit(
    do_not_specialize=["num_tokens", "front_core_num", "num_tokens_each_front_core", "num_tokens_each_tail_core"]
)
def split_qkv_rmsnorm_mrope_kernel(
    in_qkv_ptr, q_weight_ptr, q_bias_ptr, k_weight_ptr, k_bias_ptr,
    merged_cos_ptr, merged_sin_ptr,
    out_q_ptr, out_k_ptr, out_v_ptr,
    num_tokens, front_core_num, num_tokens_each_front_core, num_tokens_each_tail_core,
    num_q_heads: tl.constexpr, num_kv_heads: tl.constexpr,
    head_size: tl.constexpr, q_size: tl.constexpr, kv_size: tl.constexpr,
    qkv_size: tl.constexpr, eps: tl.constexpr, has_bias: tl.constexpr,
    rope_dim: tl.constexpr, half_rope_dim: tl.constexpr,
    IS_PARTIAL_ROPE: tl.constexpr,
):
    block_idx = tl.program_id(0)
    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core
    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core

    q_w = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_w = tl.load(k_weight_ptr + tl.arange(0, head_size))
    if has_bias:
        q_b = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_b = tl.load(k_bias_ptr + tl.arange(0, head_size))
    ca = tl.arange(0, half_rope_dim)
    qa = tl.arange(0, q_size)
    ka = tl.arange(0, kv_size)

    n2 = loop_num // 2
    rem = loop_num - n2 * 2

    for pair in range(n2):
        ta = block_offset + pair * 2
        tb = ta + 1
        ba = in_qkv_ptr + ta * qkv_size
        bb = in_qkv_ptr + tb * qkv_size

        q_a = tl.load(ba + qa).reshape(num_q_heads, head_size)
        q_b_t = tl.load(bb + qa).reshape(num_q_heads, head_size)
        k_a = tl.load(ba + q_size + ka).reshape(num_kv_heads, head_size)
        k_b_t = tl.load(bb + q_size + ka).reshape(num_kv_heads, head_size)
        v_a = tl.load(ba + q_size + kv_size + ka)
        tl.store(out_v_ptr + ta * kv_size + ka, v_a)
        v_b = tl.load(bb + q_size + kv_size + ka)
        tl.store(out_v_ptr + tb * kv_size + ka, v_b)

        c_a = tl.load(merged_cos_ptr + ta * half_rope_dim + ca).reshape(1, half_rope_dim)
        s_a = tl.load(merged_sin_ptr + ta * half_rope_dim + ca).reshape(1, half_rope_dim)
        c_b = tl.load(merged_cos_ptr + tb * half_rope_dim + ca).reshape(1, half_rope_dim)
        s_b = tl.load(merged_sin_ptr + tb * half_rope_dim + ca).reshape(1, half_rope_dim)

        # rmsnorm all 4
        sq = q_a.to(tl.float32) * q_a
        var = tl.sum(sq, axis=1) / head_size
        q_a = q_a.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
        if has_bias:
            q_a = q_a + q_b

        sq = q_b_t.to(tl.float32) * q_b_t
        var = tl.sum(sq, axis=1) / head_size
        q_b_t = q_b_t.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_q_heads, 1) * q_w
        if has_bias:
            q_b_t = q_b_t + q_b

        sq = k_a.to(tl.float32) * k_a
        var = tl.sum(sq, axis=1) / head_size
        k_a = k_a.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
        if has_bias:
            k_a = k_a + k_b

        sq = k_b_t.to(tl.float32) * k_b_t
        var = tl.sum(sq, axis=1) / head_size
        k_b_t = k_b_t.to(tl.float32) * (1 / tl.sqrt(var + eps)).reshape(num_kv_heads, 1) * k_w
        if has_bias:
            k_b_t = k_b_t + k_b

        # rope q_a + store
        x1 = tl.extract_slice(q_a, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_a, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        r1 = x1 * c_a - x2 * s_a
        r2 = x2 * c_a + x1 * s_a
        if IS_PARTIAL_ROPE:
            q_a = tl.insert_slice(q_a, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_a = tl.insert_slice(q_a, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            q_a = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            q_a = tl.insert_slice(q_a, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_a = tl.insert_slice(q_a, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_q_ptr + ta * q_size + qa, q_a.reshape(q_size))

        # rope q_b + store
        x1 = tl.extract_slice(q_b_t, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_b_t, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        r1 = x1 * c_b - x2 * s_b
        r2 = x2 * c_b + x1 * s_b
        if IS_PARTIAL_ROPE:
            q_b_t = tl.insert_slice(q_b_t, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_b_t = tl.insert_slice(q_b_t, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            q_b_t = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            q_b_t = tl.insert_slice(q_b_t, r1, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_b_t = tl.insert_slice(q_b_t, r2, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_q_ptr + tb * q_size + qa, q_b_t.reshape(q_size))

        # rope k_a + store
        y1 = tl.extract_slice(k_a, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_a, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        r1 = y1 * c_a - y2 * s_a
        r2 = y2 * c_a + y1 * s_a
        if IS_PARTIAL_ROPE:
            k_a = tl.insert_slice(k_a, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_a = tl.insert_slice(k_a, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            k_a = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            k_a = tl.insert_slice(k_a, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_a = tl.insert_slice(k_a, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_k_ptr + ta * kv_size + ka, k_a.reshape(kv_size))

        # rope k_b + store
        y1 = tl.extract_slice(k_b_t, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_b_t, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        r1 = y1 * c_b - y2 * s_b
        r2 = y2 * c_b + y1 * s_b
        if IS_PARTIAL_ROPE:
            k_b_t = tl.insert_slice(k_b_t, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_b_t = tl.insert_slice(k_b_t, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        else:
            k_b_t = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            k_b_t = tl.insert_slice(k_b_t, r1, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_b_t = tl.insert_slice(k_b_t, r2, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1)).to(tl.bfloat16)
        tl.store(out_k_ptr + tb * kv_size + ka, k_b_t.reshape(kv_size))

    for r_idx in range(rem):
        tidx = block_offset + n2 * 2 + r_idx
        base = in_qkv_ptr + tidx * qkv_size
        qt = tl.load(base + qa).reshape(num_q_heads, head_size)
        kt = tl.load(base + q_size + ka).reshape(num_kv_heads, head_size)
        vt = tl.load(base + q_size + kv_size + ka)
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


def _merge_cos_sin(cos_sin, mrope_section, is_interleaved, num_tokens, rope_dim):
    hrd = rope_dim // 2
    t_cs, h_cs, w_cs = cos_sin[0], cos_sin[1], cos_sin[2]
    tc, hc, wc = t_cs[:, :hrd], h_cs[:, :hrd], w_cs[:, :hrd]
    ts, hs, ws = t_cs[:, hrd:], h_cs[:, hrd:], w_cs[:, hrd:]
    idx = torch.arange(hrd, device=cos_sin.device)
    if is_interleaved:
        hm = ((idx % 3) == 1) & (idx <= 3 * mrope_section[1])
        wm = ((idx % 3) == 2) & (idx <= 3 * mrope_section[2])
        tm = ~(hm | wm)
    else:
        tm = idx < mrope_section[0]
        hm = (mrope_section[0] - 1 < idx) & (idx < mrope_section[0] + mrope_section[1])
        wm = (mrope_section[0] + mrope_section[1] - 1 < idx) & (idx < sum(mrope_section))
    tm, hm, wm = tm.unsqueeze(0), hm.unsqueeze(0), wm.unsqueeze(0)
    z = torch.zeros(1, device=cos_sin.device, dtype=cos_sin.dtype)
    return (torch.where(tm, tc, z) + torch.where(hm, hc, z) + torch.where(wm, wc, z)).to(torch.float32).contiguous(), \
           (torch.where(tm, ts, z) + torch.where(hm, hs, z) + torch.where(wm, ws, z)).to(torch.float32).contiguous()


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
    split_qkv_rmsnorm_mrope_kernel[(bd,)](
        qkv, q_weight, q_bias, k_weight, k_bias, mc, ms, qo, ko, vo,
        num_tokens, front_core_num, nf, nt,
        num_q_heads, num_kv_heads, head_size, q_size, kv_size, q_size + 2 * kv_size,
        eps, q_bias is not None, rope_dim, rope_dim // 2, IS_PARTIAL_ROPE,
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
