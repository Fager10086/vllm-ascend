import torch
import triton
import triton.language as tl
from vllm.utils.torch_utils import direct_register_custom_op
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


_output_cache = {}
_core_num = None
_direct_launch_cache = {}


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
    qkv_stride: tl.constexpr,
):
    block_idx = tl.program_id(0)
    loop_num = num_tokens_each_front_core
    if block_idx >= front_core_num:
        loop_num = num_tokens_each_tail_core
    block_offset = num_tokens_each_front_core * block_idx
    if block_idx >= front_core_num:
        block_offset = num_tokens_each_front_core * front_core_num + (block_idx - front_core_num) * num_tokens_each_tail_core

    q_rmsnorm_weight = tl.load(q_weight_ptr + tl.arange(0, head_size))
    k_rmsnorm_weight = tl.load(k_weight_ptr + tl.arange(0, head_size))
    if has_bias:
        q_bias_val = tl.load(q_bias_ptr + tl.arange(0, head_size))
        k_bias_val = tl.load(k_bias_ptr + tl.arange(0, head_size))

    cos_offsets = tl.arange(0, half_rope_dim)
    if is_interleaved:
        h_mask = ((cos_offsets % 3) == 1) & (cos_offsets <= 3 * mrope_section_h)
        w_mask = ((cos_offsets % 3) == 2) & (cos_offsets <= 3 * mrope_section_w)
        t_mask = ~(h_mask | w_mask)
    else:
        t_mask = cos_offsets < mrope_section_t
        h_mask = (mrope_section_t - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h)
        w_mask = (mrope_section_t + mrope_section_h - 1 < cos_offsets) & (cos_offsets < mrope_section_t + mrope_section_h + mrope_section_w)

    q_arange = tl.arange(0, q_size)
    kv_arange = tl.arange(0, kv_size)

    for index in range(loop_num):
        token_idx = block_offset + index
        in_base = in_qkv_ptr + token_idx * qkv_stride
        in_q_tensor = tl.load(in_base + q_arange).to(tl.float32).reshape(num_q_heads, head_size)
        in_k_tensor = tl.load(in_base + q_size + kv_arange).to(tl.float32).reshape(num_kv_heads, head_size)
        in_v_tensor = tl.load(in_base + q_size + kv_size + kv_arange)
        tl.store(out_v_ptr + token_idx * kv_size + kv_arange, in_v_tensor)

        cos_base = cos_sin_ptr + token_idx * rope_dim
        t_cos_tensor = tl.load(cos_base + cos_offsets, mask=t_mask, other=0)
        h_cos_tensor = tl.load(cos_base + num_tokens * rope_dim + cos_offsets, mask=h_mask, other=0)
        w_cos_tensor = tl.load(cos_base + 2 * num_tokens * rope_dim + cos_offsets, mask=w_mask, other=0)
        cos_half = (t_cos_tensor + h_cos_tensor + w_cos_tensor).to(tl.float32).reshape(1, half_rope_dim)
        sin_base = cos_base + half_rope_dim
        t_sin_tensor = tl.load(sin_base + cos_offsets, mask=t_mask, other=0)
        h_sin_tensor = tl.load(sin_base + num_tokens * rope_dim + cos_offsets, mask=h_mask, other=0)
        w_sin_tensor = tl.load(sin_base + 2 * num_tokens * rope_dim + cos_offsets, mask=w_mask, other=0)
        sin_half = (t_sin_tensor + h_sin_tensor + w_sin_tensor).to(tl.float32).reshape(1, half_rope_dim)

        q_rstd = tl.rsqrt(tl.sum(in_q_tensor * in_q_tensor, axis=1) / head_size + eps).reshape(num_q_heads, 1)
        q_normalized = in_q_tensor * q_rstd * q_rmsnorm_weight
        if has_bias:
            q_normalized = q_normalized + q_bias_val
        k_rstd = tl.rsqrt(tl.sum(in_k_tensor * in_k_tensor, axis=1) / head_size + eps).reshape(num_kv_heads, 1)
        k_normalized = in_k_tensor * k_rstd * k_rmsnorm_weight
        if has_bias:
            k_normalized = k_normalized + k_bias_val

        x1 = tl.extract_slice(q_normalized, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        x2 = tl.extract_slice(q_normalized, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
        q_rope_first = x1 * cos_half - x2 * sin_half
        q_rope_second = x2 * cos_half + x1 * sin_half
        y1 = tl.extract_slice(k_normalized, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        y2 = tl.extract_slice(k_normalized, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        k_rope_first = y1 * cos_half - y2 * sin_half
        k_rope_second = y2 * cos_half + y1 * sin_half

        if IS_PARTIAL_ROPE:
            q_result = tl.insert_slice(q_normalized, q_rope_first, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_result = tl.insert_slice(q_result, q_rope_second, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            k_result = tl.insert_slice(k_normalized, k_rope_first, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_result = tl.insert_slice(k_result, k_rope_second, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
        else:
            q_result = tl.zeros((num_q_heads, rope_dim), dtype=tl.float32)
            q_result = tl.insert_slice(q_result, q_rope_first, offsets=(0, 0), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            q_result = tl.insert_slice(q_result, q_rope_second, offsets=(0, half_rope_dim), sizes=(num_q_heads, half_rope_dim), strides=(1, 1))
            k_result = tl.zeros((num_kv_heads, rope_dim), dtype=tl.float32)
            k_result = tl.insert_slice(k_result, k_rope_first, offsets=(0, 0), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))
            k_result = tl.insert_slice(k_result, k_rope_second, offsets=(0, half_rope_dim), sizes=(num_kv_heads, half_rope_dim), strides=(1, 1))

        tl.store(out_q_ptr + token_idx * q_size + q_arange, q_result.to(tl.bfloat16).reshape(q_size))
        tl.store(out_k_ptr + token_idx * kv_size + kv_arange, k_result.to(tl.bfloat16).reshape(kv_size))


def _get_core_num():
    global _core_num
    if _core_num is None:
        _core_num = get_vectorcore_num()
    return _core_num


def _get_cached_outputs(num_tokens, q_size, kv_size, device, dtype):
    global _output_cache
    cache_key = (num_tokens, q_size, kv_size)
    cached = _output_cache.get(cache_key)
    if cached is not None:
        return cached
    q_output = torch.empty(num_tokens, q_size, device=device, dtype=dtype)
    k_output = torch.empty(num_tokens, kv_size, device=device, dtype=dtype)
    v_output = torch.empty(num_tokens, kv_size, device=device, dtype=dtype)
    _output_cache[cache_key] = (q_output, k_output, v_output)
    return q_output, k_output, v_output


def _warmup_and_cache(block_dim, constexpr_key, all_args):
    global _direct_launch_cache
    kern = split_qkv_rmsnorm_mrope_kernel
    device = 0
    existing_keys = set(kern.cache.get(device, {}).keys())
    launcher = kern[(block_dim,)]
    launcher(*all_args)
    cache = kern.cache[device]
    new_keys = set(cache.keys()) - existing_keys
    if new_keys:
        compiled_key = new_keys.pop()
    else:
        compiled_key = list(cache.keys())[-1]
    compiled = cache[compiled_key]
    npu_launcher = compiled.run
    c_launch = npu_launcher.launch
    fn_ptr = compiled.function
    pm = compiled.packed_metadata
    lm = compiled.launch_metadata
    eh = compiled.launch_enter_hook
    xh = compiled.launch_exit_hook
    stream = torch.npu.current_stream().npu_stream
    result = (c_launch, stream, fn_ptr, pm, lm, eh, xh, block_dim)
    _direct_launch_cache[constexpr_key] = result
    return result


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
    core_num = _get_core_num()
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    num_tokens = qkv.shape[0]
    if rope_dim is None:
        rope_dim = head_size
    remainder = num_tokens % core_num
    front_core_num = remainder if remainder != 0 else core_num
    num_tokens_each_front_core = (num_tokens + core_num - 1) // core_num
    num_tokens_each_tail_core = num_tokens // core_num
    block_dim = core_num if num_tokens > core_num else num_tokens
    has_bias = q_bias is not None
    half_rope_dim = rope_dim // 2
    IS_PARTIAL_ROPE = rope_dim != head_size
    qkv_stride = q_size + 2 * kv_size

    q_output, k_output, v_output = _get_cached_outputs(num_tokens, q_size, kv_size, qkv.device, qkv.dtype)

    constexpr_key = (num_q_heads, num_kv_heads, head_size,
                     mrope_section[0], mrope_section[1], mrope_section[2],
                     has_bias, is_interleaved, rope_dim, block_dim)

    cached = _direct_launch_cache.get(constexpr_key)
    if cached is None:
        all_args = (
            qkv, q_weight, q_bias, k_weight, k_bias, cos_sin,
            q_output, k_output, v_output,
            num_tokens, front_core_num, num_tokens_each_front_core, num_tokens_each_tail_core,
            num_q_heads, num_kv_heads, head_size, q_size, kv_size, eps,
            mrope_section[0], mrope_section[1], mrope_section[2],
            has_bias, is_interleaved, rope_dim, half_rope_dim,
            IS_PARTIAL_ROPE, qkv_stride,
        )
        cached = _warmup_and_cache(block_dim, constexpr_key, all_args)

    c_launch, stream, fn_ptr, pm, lm, eh, xh, bd = cached
    c_launch(
        bd, 1, 1, stream, fn_ptr,
        pm, lm, eh, xh,
        qkv, q_weight, q_bias, k_weight, k_bias, cos_sin,
        q_output, k_output, v_output,
        num_tokens, front_core_num, num_tokens_each_front_core, num_tokens_each_tail_core,
    )
    return q_output, k_output, v_output


def triton_split_qkv_rmsnorm_mrope_fake(
    qkv: torch.Tensor, q_weight: torch.Tensor, k_weight: torch.Tensor, cos_sin: torch.Tensor,
    num_q_heads: int, num_kv_heads: int, head_size: int, eps: float, mrope_section: list[int],
    q_bias: torch.Tensor | None = None, k_bias: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    nt = qkv.shape[0]
    qs = num_q_heads * head_size
    ks = num_kv_heads * head_size
    return torch.empty(nt, qs, device=qkv.device, dtype=qkv.dtype), torch.empty(nt, ks, device=qkv.device, dtype=qkv.dtype), torch.empty(nt, ks, device=qkv.device, dtype=qkv.dtype)


direct_register_custom_op(
    op_name="triton_split_qkv_rmsnorm_mrope",
    op_func=triton_split_qkv_rmsnorm_mrope,
    fake_impl=triton_split_qkv_rmsnorm_mrope_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
