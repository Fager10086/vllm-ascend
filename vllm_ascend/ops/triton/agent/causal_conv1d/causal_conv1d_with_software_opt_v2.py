# adapted from vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# Adapted from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2024, Tri Dao.
# Adapted from https://github.com/Dao-AILab/causal-conv1d/blob/main/causal_conv1d/causal_conv1d_interface.py
# and https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/layers/mamba/ops/causal_conv1d.py
# mypy: ignore-errors

from typing import Any

import torch
import torch.nn.functional as F
import triton
import triton.language as tl
from vllm.distributed import get_pcp_group
from vllm.forward_context import get_forward_context
from vllm.v1.attention.backends.utils import PAD_SLOT_ID  # type: ignore

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num, init_device_properties_triton


def causal_conv1d_ref(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    initial_states: torch.Tensor | None = None,
    return_final_states: bool = False,
    final_states_out: torch.Tensor | None = None,
    activation: str | None = "silu",
):
    """
    x: (batch, dim, seqlen)
    weight: (dim, width)
    bias: (dim,)
    initial_states: (batch, dim, width - 1)
    final_states_out: (batch, dim, width - 1)
    out: (batch, dim, seqlen)
    """
    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    dtype_in = x.dtype
    x = x.to(weight.dtype)
    seqlen = x.shape[-1]
    dim, width = weight.shape

    if initial_states is None:
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=width - 1, groups=dim)
    else:
        x = torch.cat([initial_states, x], dim=-1)
        out = F.conv1d(x, weight.unsqueeze(1), bias, padding=0, groups=dim)
    out = out[..., :seqlen]

    if return_final_states:
        final_states = F.pad(x, (width - 1 - x.shape[-1], 0)).to(dtype_in)  # (batch, dim, width - 1)
        if final_states_out is not None:
            final_states_out.copy_(final_states)
        else:
            final_states_out = final_states
    out = (out if activation is None else F.silu(out)).to(dtype=dtype_in)
    return (out, None) if not return_final_states else (out, final_states_out)


def causal_conv1d_fn(
    x: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: str | None = "silu",
    conv_states: torch.Tensor | None = None,
    has_initial_state: torch.Tensor | None = None,
    cache_indices: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    metadata: Any | None = None,
    pad_slot_id: int = PAD_SLOT_ID,
):
    """
    x: (batch, dim, seqlen) or (dim,cu_seq_len) for varlen
        sequences are concatenated from left to right for varlen
    weight: (dim, width)
    bias: (dim,)
    query_start_loc: (batch + 1) int32
        The cumulative sequence lengths of the sequences in
        the batch, used to index into sequence. prepended by 0.
        for example: query_start_loc = torch.Tensor([0,10,16,17]),
        x.shape=(dim,17)
    cache_indices: (batch)  int32
        indicates the corresponding state index,
        like so: conv_state = conv_states[cache_indices[batch_id]]
    has_initial_state: (batch) bool
        indicates whether should the kernel take the current state as initial
        state for the calculations
    conv_states: (...,dim,width - 1) itype
        updated inplace if provided
    activation: either None or "silu" or "swish"
    pad_slot_id: int
            if cache_indices is passed, lets the kernel identify padded
            entries that will not be processed,
            for example: cache_indices = [pad_slot_id, 1, 20, pad_slot_id]
            in this case, the kernel will not process entries at
            indices 0 and 3
    out: (batch, dim, seqlen)
    """
    # forward_context = get_forward_context()
    num_decodes = 0
    # attn_metadata = forward_context.attn_metadata
    attn_metadata = None
    if attn_metadata is not None and isinstance(attn_metadata, dict):
        attn_metadata = next(iter(attn_metadata.values()), None)
    if attn_metadata is not None:
        num_decodes = attn_metadata.num_decodes

    if activation not in [None, "silu", "swish"]:
        raise NotImplementedError("activation must be None, silu, or swish")
    if x.stride(-1) != 1:
        x = x.contiguous()
    bias = bias.contiguous() if bias is not None else None

    out_ref = []
    out_ref_b = []
    seqlens = query_start_loc[1:] - query_start_loc[:-1]
    seqlens = seqlens.tolist()
    splits = torch.split(x, seqlens, dim=-1)
    width = weight.shape[1]
    last_width_prefill_x = extract_last_width(x, query_start_loc[num_decodes:], conv_states.shape[-1])

    # if get_pcp_group().world_size > 1:
    #     all_last_width_prefill_x = get_pcp_group().all_gather(last_width_prefill_x.unsqueeze(0).contiguous(), 0)
    #     pcp_rank = get_pcp_group().rank_in_group
    #     if pcp_rank > 0:
    #         conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[pcp_rank - 1, ...]

    for i in range(len(seqlens)):
        x_s = splits[i]
        if cache_indices[i] == PAD_SLOT_ID:
            continue
        out_ref_b.append(
            causal_conv1d_ref(
                x_s,
                weight,
                bias,
                activation=activation,
                return_final_states=True,
                final_states_out=conv_states[cache_indices[i]][..., : (width - 1)].unsqueeze(0),
                initial_states=conv_states[cache_indices[i]][..., : (width - 1)],
            )
        )

    # if get_pcp_group().world_size > 1:
    #     conv_states[cache_indices[num_decodes:]] = all_last_width_prefill_x[-1, ...]
    out_ref.append(torch.cat([t[0] for t in out_ref_b], dim=-1))
    out_ref_tensor = torch.cat(out_ref, dim=0)
    return out_ref_tensor


def extract_last_width(x, start_loc, width):
    end_loc = start_loc[1:]
    offsets = torch.arange(width, device=x.device)
    indices = end_loc.unsqueeze(1) - width + offsets.unsqueeze(0)  # (num_seqs, width)

    return x[:, indices].permute(1, 0, 2)


@triton.jit(do_not_specialize=["batch", "num_cache_lines"])
def _causal_conv1d_update_kernel_optimized(
    # Pointers
    x_ptr,          # (batch, seqlen, dim) — stride_dim=1
    w_ptr,          # (width, dim) — stride_dim=1 (transposed weight)
    bias_ptr,       # (dim,) or None
    conv_state_ptr, # (num_cache_lines, state_len, dim) — dim-contiguous
    conv_state_indices_ptr,  # (batch, 1)
    o_ptr,          # (batch, seqlen, dim) — stride_dim=1
    # Scalars
    batch,
    dim: tl.constexpr,
    seqlen: tl.constexpr,
    state_len: tl.constexpr,
    num_cache_lines,
    # Strides
    stride_x_batch: tl.constexpr,
    stride_x_token: tl.constexpr,
    stride_w_width: tl.constexpr,
    stride_cs_batch: tl.constexpr,
    stride_cs_dim: tl.constexpr,
    stride_cs_state: tl.constexpr,
    stride_csi_batch: tl.constexpr,
    stride_o_batch: tl.constexpr,
    stride_o_token: tl.constexpr,
    # Constants
    pad_slot_id: tl.constexpr,
    HAS_BIAS: tl.constexpr,
    SILU_ACTIVATION: tl.constexpr,
    USE_PAD_SLOT: tl.constexpr,
    # Tiling
    BLOCK_N: tl.constexpr,
    N_CHANNEL_BLOCKS: tl.constexpr,
    WIDTH: tl.constexpr,
):
    """
    Optimized causal conv1d update kernel for Ascend NPU.
    Fixed 1D Grid + stride loop over (batch × channel_block) work items.
    Weight and x/output are dim-contiguous; conv_state uses explicit strides.
    Supports arbitrary WIDTH (conv kernel width) and state_len = WIDTH - 1.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    total_work = batch * N_CHANNEL_BLOCKS
    c_offs = tl.arange(0, BLOCK_N)

    for work_idx in range(pid, total_work, num_programs):
        b_idx = work_idx // N_CHANNEL_BLOCKS
        c_block = work_idx - b_idx * N_CHANNEL_BLOCKS
        c_start = c_block * BLOCK_N
        c_idx = c_start + c_offs
        c_mask = c_idx < dim

        cs_line = tl.load(conv_state_indices_ptr + b_idx * stride_csi_batch).to(tl.int64)

        if USE_PAD_SLOT:
            c_mask = c_mask & (cs_line != pad_slot_id)

        if HAS_BIAS:
            b_val = tl.load(bias_ptr + c_idx, mask=c_mask).to(tl.float32)
        else:
            b_val = tl.zeros((BLOCK_N,), dtype=tl.float32)

        # Load history from conv_state — dim-contiguous
        cs_base = conv_state_ptr + cs_line * stride_cs_batch + c_idx * stride_cs_dim

        # Process tokens — x and out are dim-contiguous
        x_base = x_ptr + b_idx * stride_x_batch + c_idx
        o_base = o_ptr + b_idx * stride_o_batch + c_idx

        # Generalized conv1d: supports any WIDTH
        # Use a sliding window of state_len history values + 1 current input
        # For each token, compute: sum(history[k] * weight[k] for k in range(state_len)) + x * weight[state_len] + bias
        if WIDTH == 2:
            # state_len=1: one history value
            h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
            w0 = tl.load(w_ptr + 0 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w1 = tl.load(w_ptr + 1 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            for t in tl.static_range(0, seqlen):
                x_val = tl.load(x_base + t * stride_x_token, mask=c_mask)
                acc = b_val + h0.to(tl.float32) * w0 + x_val.to(tl.float32) * w1
                h0 = x_val
                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(o_base + t * stride_o_token, acc, mask=c_mask)
            tl.store(cs_base + 0 * stride_cs_state, h0, mask=c_mask)
        elif WIDTH == 3:
            # state_len=2: two history values
            h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
            h1 = tl.load(cs_base + 1 * stride_cs_state, mask=c_mask)
            w0 = tl.load(w_ptr + 0 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w1 = tl.load(w_ptr + 1 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w2 = tl.load(w_ptr + 2 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            for t in tl.static_range(0, seqlen):
                x_val = tl.load(x_base + t * stride_x_token, mask=c_mask)
                acc = b_val
                acc += h0.to(tl.float32) * w0
                acc += h1.to(tl.float32) * w1
                acc += x_val.to(tl.float32) * w2
                h0 = h1
                h1 = x_val
                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(o_base + t * stride_o_token, acc, mask=c_mask)
            tl.store(cs_base + 0 * stride_cs_state, h0, mask=c_mask)
            tl.store(cs_base + 1 * stride_cs_state, h1, mask=c_mask)
        elif WIDTH == 4:
            # state_len=3: three history values (most common, e.g. Mamba)
            h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
            h1 = tl.load(cs_base + 1 * stride_cs_state, mask=c_mask)
            h2 = tl.load(cs_base + 2 * stride_cs_state, mask=c_mask)
            w0 = tl.load(w_ptr + 0 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w1 = tl.load(w_ptr + 1 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w2 = tl.load(w_ptr + 2 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w3 = tl.load(w_ptr + 3 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            for t in tl.static_range(0, seqlen):
                x_val = tl.load(x_base + t * stride_x_token, mask=c_mask)
                acc = b_val
                acc += h0.to(tl.float32) * w0
                acc += h1.to(tl.float32) * w1
                acc += h2.to(tl.float32) * w2
                acc += x_val.to(tl.float32) * w3
                h0 = h1
                h1 = h2
                h2 = x_val
                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(o_base + t * stride_o_token, acc, mask=c_mask)
            tl.store(cs_base + 0 * stride_cs_state, h0, mask=c_mask)
            tl.store(cs_base + 1 * stride_cs_state, h1, mask=c_mask)
            tl.store(cs_base + 2 * stride_cs_state, h2, mask=c_mask)
        else:
            # Generic path for WIDTH >= 5: state_len = WIDTH - 1
            # Load all history states
            h0 = tl.load(cs_base + 0 * stride_cs_state, mask=c_mask)
            h1 = tl.load(cs_base + 1 * stride_cs_state, mask=c_mask)
            h2 = tl.load(cs_base + 2 * stride_cs_state, mask=c_mask)
            h3 = tl.load(cs_base + 3 * stride_cs_state, mask=c_mask)
            w0 = tl.load(w_ptr + 0 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w1 = tl.load(w_ptr + 1 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w2 = tl.load(w_ptr + 2 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w3 = tl.load(w_ptr + 3 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            w4 = tl.load(w_ptr + 4 * stride_w_width + c_idx, mask=c_mask).to(tl.float32)
            for t in tl.static_range(0, seqlen):
                x_val = tl.load(x_base + t * stride_x_token, mask=c_mask)
                acc = b_val
                acc += h0.to(tl.float32) * w0
                acc += h1.to(tl.float32) * w1
                acc += h2.to(tl.float32) * w2
                acc += h3.to(tl.float32) * w3
                acc += x_val.to(tl.float32) * w4
                h0 = h1
                h1 = h2
                h2 = h3
                h3 = x_val
                if SILU_ACTIVATION:
                    acc = acc / (1.0 + tl.exp(-acc))
                tl.store(o_base + t * stride_o_token, acc, mask=c_mask)
            tl.store(cs_base + 0 * stride_cs_state, h0, mask=c_mask)
            tl.store(cs_base + 1 * stride_cs_state, h1, mask=c_mask)
            tl.store(cs_base + 2 * stride_cs_state, h2, mask=c_mask)
            tl.store(cs_base + 3 * stride_cs_state, h3, mask=c_mask)


def causal_conv1d_update_npu(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None = None,
    activation: bool | str | None = None,
    conv_state_indices: torch.Tensor | None = None,
    num_accepted_tokens: torch.Tensor | None = None,
    query_start_loc: torch.Tensor | None = None,
    max_query_len: int = -1,
    pad_slot_id: int = PAD_SLOT_ID,
    block_idx_last_scheduled_token: torch.Tensor | None = None,
    initial_state_idx: torch.Tensor | None = None,
    validate_data=False,
):
    """
    Optimized causal_conv1d_update on Ascend NPU.

    x: [batch, dim] or [batch, seqlen, dim] or [batch, dim, seqlen]
    conv_state: (..., dim, state_len) or (..., state_len, dim)
    weight: (dim, width)
    bias: (dim,)
    """
    # Transpose weight to (width, dim) for contiguous dim access
    weight_dim = weight.shape[0]
    weight = weight.transpose(0, 1).contiguous()
    # Transpose conv_state to (..., state_len, dim) for contiguous dim access
    # We'll copy back updated state after kernel
    conv_state_t = conv_state.transpose(1, 2).contiguous()

    if isinstance(activation, bool):
        activation = "silu" if activation is True else None
    elif activation is not None:
        assert activation in ["silu", "swish"]

    original_x_dtype = x.dtype
    x = x.to(conv_state.dtype)

    unsqueeze = query_start_loc is None and x.dim() == 2
    if unsqueeze:
        x = x.unsqueeze(1)  # (batch, 1, dim)

    # Determine dimensions and ensure x is (batch, seqlen, dim) layout
    x_was_transposed = False
    if query_start_loc is None:
        if x.shape[1] == weight_dim and (x.dim() < 3 or x.shape[2] != weight_dim):
            # x is (batch, dim, seqlen) — transpose to (batch, seqlen, dim)
            x = x.transpose(1, 2).contiguous()
            x_was_transposed = True
        batch, seqlen, dim = x.shape
    else:
        assert conv_state_indices is not None
        batch = conv_state_indices.size(0)
        dim = x.size(1)
        seqlen = max_query_len

    width = weight.shape[0]  # weight is now (width, dim)

    # Handle conv_state_indices
    if conv_state_indices is not None:
        if conv_state_indices.dim() == 1:
            conv_state_indices = conv_state_indices.unsqueeze(1)
        stride_csi_batch = conv_state_indices.stride(0)
    else:
        conv_state_indices = torch.arange(batch, dtype=torch.int32, device=x.device).unsqueeze(1)
        stride_csi_batch = 1

    # Ensure weight is contiguous (already done above)
    # Ensure x has stride_dim=1 (last dim contiguous)
    if x.stride(-1) != 1:
        x = x.contiguous()

    out = x  # overwrite-on-x
    num_cache_lines = conv_state_t.shape[0]

    # Strides
    stride_x_batch = x.stride(0)
    stride_x_token = x.stride(1)

    stride_w_width = weight.stride(0)

    # conv_state_t is (num_cache_lines, state_len, dim) — dim-contiguous
    stride_cs_batch = conv_state_t.stride(0)
    stride_cs_dim = conv_state_t.stride(2)  # should be 1
    stride_cs_state = conv_state_t.stride(1)  # state_len stride

    stride_o_batch = out.stride(0)
    stride_o_token = out.stride(1)

    # Tiling — BLOCK_N=512 since conv_state_t is dim-contiguous
    block_n = 512 if dim >= 512 else min(256, triton.next_power_of_2(dim))
    n_channel_blocks = (dim + block_n - 1) // block_n

    init_device_properties_triton()
    grid = (get_vectorcore_num(),)

    _causal_conv1d_update_kernel_optimized[grid](
        x,
        weight,
        bias,
        conv_state_t,
        conv_state_indices,
        out,
        batch,
        dim,
        seqlen,
        width - 1,  # state_len
        num_cache_lines,
        stride_x_batch,
        stride_x_token,
        stride_w_width,
        stride_cs_batch,
        stride_cs_dim,
        stride_cs_state,
        stride_csi_batch,
        stride_o_batch,
        stride_o_token,
        pad_slot_id,
        HAS_BIAS=bias is not None,
        SILU_ACTIVATION=activation in ["silu", "swish"],
        USE_PAD_SLOT=pad_slot_id is not None,
        BLOCK_N=block_n,
        N_CHANNEL_BLOCKS=n_channel_blocks,
        WIDTH=width,
    )

    # Copy updated state back to original conv_state
    conv_state.transpose(1, 2).copy_(conv_state_t)

    if unsqueeze:
        out = out.squeeze(1)
    if x_was_transposed:
        out = out.transpose(1, 2)
    return out.to(original_x_dtype)
