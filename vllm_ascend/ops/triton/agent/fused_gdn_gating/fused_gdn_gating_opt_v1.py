# Adapt from https://github.com/vllm-project/vllm/blob/main/vllm/model_executor/models/qwen3_next.py
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num

UNIFIED_BUFFER_SIZE = 1572864


@triton.jit
def fused_gdn_gating_kernel(
    g,
    beta_output,
    A_log,
    a,
    b,
    dt_bias,
    seq_len,
    NUM_HEADS: tl.constexpr,
    NUM_BATCHES: tl.constexpr,
    beta: tl.constexpr,
    threshold: tl.constexpr,
    BLK_BATCHES: tl.constexpr,
):
    pid = tl.program_id(0)
    batch_off = pid * BLK_BATCHES + tl.arange(0, BLK_BATCHES)
    head_off = tl.arange(0, NUM_HEADS)
    
    batch_mask = batch_off < NUM_BATCHES
    mask = batch_mask[:, None]
    off = batch_off[:, None] * NUM_HEADS + head_off[None, :]
    
    blk_A_log = tl.load(A_log + head_off)
    blk_bias = tl.load(dt_bias + head_off)
    blk_a = tl.load(a + off, mask=mask)
    blk_b = tl.load(b + off, mask=mask)
    
    exp_A_log_f32 = tl.exp(blk_A_log.to(tl.float32))
    x = blk_a.to(tl.float32) + blk_bias.to(tl.float32)[None, :]
    beta_x = beta * x
    softplus_x = tl.where(beta_x <= threshold, (1 / beta) * tl.log(1 + tl.exp(beta_x)), x)
    blk_g = -exp_A_log_f32[None, :] * softplus_x
    
    tl.store(g + off, blk_g.to(g.dtype.element_ty), mask=mask)
    
    blk_beta_output = tl.sigmoid(blk_b.to(tl.float32))
    tl.store(beta_output + off, blk_beta_output.to(beta_output.dtype.element_ty), mask=mask)


def fused_gdn_gating_patch(
    A_log: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: float = 1.0,
    threshold: float = 20.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch, num_heads = a.shape

    UB_CAPACITY = 85 * 1024
    per_token_ub = num_heads * 16
    shared_ub = num_heads * 12
    max_tokens = (UB_CAPACITY - shared_ub) // per_token_ub
    BLK_BATCHES = triton.next_power_of_2(max_tokens) // 2
    BLK_BATCHES = max(1, min(BLK_BATCHES, 512))
    
    num_progs = triton.cdiv(batch, BLK_BATCHES)

    g = torch.empty(1, batch, num_heads, dtype=torch.float32, device=a.device)
    beta_output = torch.empty(1, batch, num_heads, dtype=b.dtype, device=b.device)

    grid = (num_progs,)
    fused_gdn_gating_kernel[grid](
        g,
        beta_output,
        A_log,
        a,
        b,
        dt_bias,
        1,
        NUM_HEADS=num_heads,
        NUM_BATCHES=batch,
        beta=beta,
        threshold=threshold,
        BLK_BATCHES=BLK_BATCHES,
    )
    return g, beta_output
