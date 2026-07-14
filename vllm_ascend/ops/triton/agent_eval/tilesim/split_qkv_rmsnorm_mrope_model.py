"""
Split QKV + RMSNorm + MRoPE - PyTorch Implementation for Tilesim

这是 triton_split_qkv_rmsnorm_mrope 算子的参考实现，用于 tilesim 性能建模。

输入输出规格 (根据 shape_stats.json):
- 输入 0: qkv [num_tokens, q_size + 2 * kv_size] 或 [num_tokens, 2 * q_size + 2 * kv_size] (with gate)
- 输入 1: q_weight [head_size]
- 输入 2: k_weight [head_size]
- 输入 3: cos_sin [3, num_tokens, rope_dim]
- 输出 0: q [num_tokens, q_size]
- 输出 1: k [num_tokens, kv_size]
- 输出 2: v [num_tokens, kv_size]
- 输出 3: gate (optional) [num_tokens, q_size]

核心步骤:
1. Split QKV: 分割 qkv 为 q, k, v (和 gate)
2. RMSNorm: 对 q 和 k 进行 RMS 归一化 (带 bias)
3. MRoPE: 对 q 和 k 应用多维旋转位置编码
4. 输出：q_rope, k_rope, v, gate (optional)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, List


def apply_interleaved_rope(x: torch.Tensor, mrope_section: List[int]) -> torch.Tensor:
    """
    Apply interleaved MRoPE to 3D rotary embeddings.
    
    Reorganizes frequency layout from chunked [TTT...HHH...WWW] to
    interleaved [THTHWHTHW...TT], preserving frequency continuity.
    
    为了支持 torch.fx tracing，使用纯函数式操作，避免条件分支。
    mrope_section 是编译时常量，所有索引预先计算。
    
    Args:
        x: [3, num_tokens, rope_dim]
        mrope_section: [t_section, h_section, w_section]
    
    Returns:
        x_t: [3, num_tokens, rope_dim]
    """
    t_section = mrope_section[1]
    w_section = mrope_section[2]
    
    # 向量化实现，避免动态索引
    x_t = x[0].clone()
    
    # 使用静态切片 - mrope_section 是常量，所以切片也是常量
    # Height dimension: indices 1 : h_section * 3 : 3
    # 直接使用切片，因为 mrope_section 是常量
    h_end = t_section * 3
    if h_end > 1:
        x_t[..., 1:h_end:3] = x[1, ..., 1:h_end:3]
    
    # Width dimension: indices 2 : w_section * 3 : 3
    w_end = w_section * 3
    if w_end > 2:
        x_t[..., 2:w_end:3] = x[2, ..., 2:w_end:3]
    
    return x_t


def rms_norm(
    x: torch.Tensor,
    norm_weight: torch.Tensor,
    eps: float,
    norm_bias: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    RMS Norm implementation.
    
    Args:
        x: [N, head_size]
        norm_weight: [head_size]
        eps: epsilon for numerical stability
        norm_bias: [head_size] (optional)
    
    Returns:
        output: [N, head_size]
    """
    x = x.to(torch.float32)
    norm_weight = norm_weight.to(torch.float32)
    
    # Compute reciprocal standard deviation
    reciprocal_std = 1 / torch.sqrt(
        torch.mean(x ** 2, axis=-1, keepdims=True) + eps
    )
    
    # Apply normalization
    out = x * reciprocal_std * norm_weight
    
    # Add bias if provided
    if norm_bias is not None:
        norm_bias = norm_bias.to(torch.float32)
        out = out + norm_bias
    
    return out


def apply_mrope(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    rope_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply MRoPE (Multi-dimensional Rotary Positional Embedding).
    
    为了支持 torch.fx tracing，使用纯函数式操作，避免动态索引和赋值。
    使用 torch.gather 和 precomputed indices。
    
    Args:
        q: [num_tokens, num_q_heads, head_size]
        k: [num_tokens, num_kv_heads, head_size]
        cos: [3, num_tokens, rope_dim]
        sin: [3, num_tokens, rope_dim]
        mrope_section: [t_section, h_section, w_section]
        rope_dim: rotary dimension
    
    Returns:
        q_out: [num_tokens, num_q_heads, head_size]
        k_out: [num_tokens, num_kv_heads, head_size]
    """
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    half_rd = rope_dim // 2
    
    t_section, h_section, w_section = mrope_section
    
    # Reshape cos/sin: [3, num_tokens, rope_dim] -> [num_tokens, rope_dim, 3]
    cos_reshaped = cos.permute(1, 2, 0)
    sin_reshaped = sin.permute(1, 2, 0)
    
    # 构建 source_dim 索引张量 - 使用纯函数式方法，避免 item assignment
    # 创建三个部分然后拼接
    t_indices = torch.zeros(t_section, dtype=torch.long, device=q.device)
    h_indices = torch.ones(h_section, dtype=torch.long, device=q.device)
    w_indices = torch.full((w_section,), 2, dtype=torch.long, device=q.device)
    source_dim = torch.cat([t_indices, h_indices, w_indices])  # [half_rd]
    
    # 使用 gather 从 cos_reshaped 中提取
    # cos_reshaped: [num_tokens, rope_dim, 3]
    # 需要 gather along the last dimension
    cos_expanded = cos_reshaped.gather(2, source_dim.view(1, 1, -1).expand(num_tokens, half_rd, -1))
    sin_expanded = sin_reshaped.gather(2, source_dim.view(1, 1, -1).expand(num_tokens, half_rd, -1))
    
    # cos_row: [num_tokens, half_rd], 取对角线
    cos_row = cos_expanded.diagonal(dim1=0, dim2=2).t()  # [num_tokens, half_rd]
    sin_row = sin_expanded.diagonal(dim1=0, dim2=2).t()  # [num_tokens, half_rd]
    
    # 扩展到 [num_tokens, num_heads, half_rd] 用于广播
    cos_half = cos_row.unsqueeze(1)  # [num_tokens, 1, half_rd]
    sin_half = sin_row.unsqueeze(1)  # [num_tokens, 1, half_rd]
    
    # 提取 q1, q2, k1, k2
    q1 = q[:, :, :half_rd]
    q2 = q[:, :, half_rd:rope_dim]
    k1 = k[:, :, :half_rd]
    k2 = k[:, :, half_rd:rope_dim]
    
    # RoPE formula (向量化)
    new_q1 = q1 * cos_half - q2 * sin_half
    new_q2 = q2 * cos_half + q1 * sin_half
    
    new_k1 = k1 * cos_half - k2 * sin_half
    new_k2 = k2 * cos_half + k1 * sin_half
    
    # 拼接结果
    q_out = torch.cat([new_q1, new_q2, q[:, :, rope_dim:]], dim=-1)
    k_out = torch.cat([new_k1, new_k2, k[:, :, rope_dim:]], dim=-1)
    
    return q_out, k_out


def apply_mrope_interleaved(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    mrope_section: List[int],
    rope_dim: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply interleaved MRoPE.
    
    为了支持 torch.fx tracing，使用纯函数式操作，避免动态索引和赋值。
    使用 torch.gather 和 precomputed indices。
    
    Args:
        q: [num_tokens, num_q_heads, head_size]
        k: [num_tokens, num_kv_heads, head_size]
        cos: [3, num_tokens, rope_dim]
        sin: [3, num_tokens, rope_dim]
        mrope_section: [t_section, h_section, w_section]
        rope_dim: rotary dimension
    
    Returns:
        q_out: [num_tokens, num_q_heads, head_size]
        k_out: [num_tokens, num_kv_heads, head_size]
    """
    num_tokens = q.shape[0]
    num_q_heads = q.shape[1]
    num_kv_heads = k.shape[1]
    half_rd = rope_dim // 2
    
    t_section, h_section, w_section = mrope_section
    
    # Apply interleaved reorganization (向量化)
    cos_reshaped = apply_interleaved_rope(cos, mrope_section)
    sin_reshaped = apply_interleaved_rope(sin, mrope_section)
    
    # Reshape to [num_tokens, rope_dim, 3]
    cos_reshaped = cos_reshaped.permute(1, 2, 0)
    sin_reshaped = sin_reshaped.permute(1, 2, 0)
    
    # 构建 source_dim 索引张量 - 使用纯函数式方法，避免 item assignment
    # 创建三个部分然后拼接
    t_indices = torch.zeros(t_section, dtype=torch.long, device=q.device)
    h_indices = torch.ones(h_section, dtype=torch.long, device=q.device)
    w_indices = torch.full((w_section,), 2, dtype=torch.long, device=q.device)
    source_dim = torch.cat([t_indices, h_indices, w_indices])  # [half_rd]
    
    # 使用 gather 从 cos_reshaped 中提取
    # cos_reshaped: [num_tokens, rope_dim, 3]
    cos_expanded = cos_reshaped.gather(2, source_dim.view(1, 1, -1).expand(num_tokens, half_rd, -1))
    sin_expanded = sin_reshaped.gather(2, source_dim.view(1, 1, -1).expand(num_tokens, half_rd, -1))
    
    # cos_row: [num_tokens, half_rd], 取对角线
    cos_row = cos_expanded.diagonal(dim1=0, dim2=2).t()  # [num_tokens, half_rd]
    sin_row = sin_expanded.diagonal(dim1=0, dim2=2).t()  # [num_tokens, half_rd]
    
    # 扩展到 [num_tokens, num_heads, half_rd] 用于广播
    cos_half = cos_row.unsqueeze(1)  # [num_tokens, 1, half_rd]
    sin_half = sin_row.unsqueeze(1)  # [num_tokens, 1, half_rd]
    
    # 提取 q1, q2, k1, k2
    q1 = q[:, :, :half_rd]
    q2 = q[:, :, half_rd:rope_dim]
    k1 = k[:, :, :half_rd]
    k2 = k[:, :, half_rd:rope_dim]
    
    # RoPE formula (向量化)
    new_q1 = q1 * cos_half - q2 * sin_half
    new_q2 = q2 * cos_half + q1 * sin_half
    
    new_k1 = k1 * cos_half - k2 * sin_half
    new_k2 = k2 * cos_half + k1 * sin_half
    
    # 拼接结果
    q_out = torch.cat([new_q1, new_q2, q[:, :, rope_dim:]], dim=-1)
    k_out = torch.cat([new_k1, new_k2, k[:, :, rope_dim:]], dim=-1)
    
    return q_out, k_out


class SplitQKVRMSNormMRoPE(nn.Module):
    """
    Split QKV + RMSNorm + MRoPE - Pure PyTorch Implementation
    
    这是 triton_split_qkv_rmsnorm_mrope 算子的参考实现，用于 tilesim 性能建模。
    
    输入输出规格 (根据 shape_stats.json):
    - 输入 0: qkv [num_tokens, q_size + 2 * kv_size] 或 [num_tokens, 2 * q_size + 2 * kv_size] (with gate)
    - 输入 1: q_weight [head_size]
    - 输入 2: k_weight [head_size]
    - 输入 3: cos_sin [3, num_tokens, rope_dim]
    - 输出 0: q [num_tokens, q_size]
    - 输出 1: k [num_tokens, kv_size]
    - 输出 2: v [num_tokens, kv_size]
    - 输出 3: gate (optional) [num_tokens, q_size]
    """
    
    def __init__(
        self,
        num_q_heads: int = 14,
        num_kv_heads: int = 7,
        head_size: int = 256,
        eps: float = 1e-6,
        mrope_section: List[int] = None,
        is_interleaved: bool = False,
        has_gate: bool = False,
    ):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.eps = eps
        self.mrope_section = mrope_section or [11, 11, 10]
        self.is_interleaved = is_interleaved
        self.has_gate = has_gate
        
        self.q_size = num_q_heads * head_size
        self.kv_size = num_kv_heads * head_size
        self.rope_dim = 2 * sum(self.mrope_section)
        
        # 注意：为了支持 meta device tracing，不使用 nn.Parameter
        # 所有参数都通过输入传递
    
    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
        q_bias: Optional[torch.Tensor] = None,
        k_bias: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            qkv: [num_tokens, q_size + 2 * kv_size] 或 [num_tokens, 2 * q_size + 2 * kv_size] (with gate)
            q_weight: [head_size]
            k_weight: [head_size]
            cos_sin: [3, num_tokens, rope_dim]
            q_bias: [head_size] (optional)
            k_bias: [head_size] (optional)
        
        Returns:
            q: [num_tokens, q_size]
            k: [num_tokens, kv_size]
            v: [num_tokens, kv_size]
            gate: [num_tokens, q_size] (if has_gate)
        """
        num_tokens = qkv.shape[0]
        
        # 1. Split QKV (and gate if present)
        if self.has_gate:
            # qkv: [num_tokens, 2 * q_size + 2 * kv_size]
            q_gate_data = qkv[:, :self.q_size * 2].view(-1, self.num_q_heads, self.head_size * 2)
            q_data, gate = torch.chunk(q_gate_data, 2, dim=-1)
            gate = gate.reshape(-1, self.q_size)
            q_data = q_data.reshape(-1, self.q_size)
            
            k_data = qkv[:, 2 * self.q_size:2 * self.q_size + self.kv_size]
            v_data = qkv[:, 2 * self.q_size + self.kv_size:]
            
            q = q_data
            k = k_data
            v = v_data
        else:
            # qkv: [num_tokens, q_size + 2 * kv_size]
            q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
            gate = None
        
        # 2. RMSNorm for Q and K
        q = q.reshape(-1, self.head_size)
        k = k.reshape(-1, self.head_size)
        
        q_norm = rms_norm(q, q_weight, self.eps, norm_bias=q_bias)
        k_norm = rms_norm(k, k_weight, self.eps, norm_bias=k_bias)
        
        # Reshape to multi-head format
        q_norm = q_norm.reshape(num_tokens, self.num_q_heads, self.head_size)
        k_norm = k_norm.reshape(num_tokens, self.num_kv_heads, self.head_size)
        
        # 3. Extract cos/sin
        cos, sin = cos_sin.chunk(2, dim=-1)
        # cos: [3, num_tokens, rope_dim]
        # sin: [3, num_tokens, rope_dim]
        
        # 4. Apply MRoPE
        if self.is_interleaved:
            q_rope, k_rope = apply_mrope_interleaved(
                q_norm, k_norm, cos, sin,
                self.mrope_section, self.rope_dim
            )
        else:
            q_rope, k_rope = apply_mrope(
                q_norm, k_norm, cos, sin,
                self.mrope_section, self.rope_dim
            )
        
        # Reshape back to hidden size
        q_rope = q_rope.reshape(num_tokens, self.q_size)
        k_rope = k_rope.reshape(num_tokens, self.kv_size)
        
        # v 保持不变
        v_out = v
        
        if self.has_gate:
            return q_rope, k_rope, v_out, gate
        else:
            return q_rope, k_rope, v_out


class Model(nn.Module):
    """Wrapper model for tilesim compatibility."""
    
    def __init__(self):
        super().__init__()
        # 使用 shape_stats.json 中的典型配置
        self.model = SplitQKVRMSNormMRoPE(
            num_q_heads=14,
            num_kv_heads=7,
            head_size=256,
            eps=1e-6,
            mrope_section=[11, 11, 10],
            is_interleaved=False,
            has_gate=False,
        )
    
    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.model(qkv, q_weight, k_weight, cos_sin)


def get_inputs():
    """
    Create sample inputs for the model.
    
    Returns:
        list of input tensors
    """
    # Configuration (matching shape_stats.json)
    num_tokens = 1
    num_q_heads = 14
    num_kv_heads = 7
    head_size = 256
    q_size = num_q_heads * head_size  # 3584
    kv_size = num_kv_heads * head_size  # 1792
    rope_dim = 2 * sum([11, 11, 10])  # 64
    
    # Create inputs
    qkv = torch.randn(
        num_tokens,
        q_size + 2 * kv_size,  # 3584 + 2 * 1792 = 7168
        dtype=torch.bfloat16
    )
    q_weight = torch.randn(head_size, dtype=torch.bfloat16)
    k_weight = torch.randn(head_size, dtype=torch.bfloat16)
    cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=torch.bfloat16)
    
    return [qkv, q_weight, k_weight, cos_sin]


if __name__ == "__main__":
    # Create model and inputs
    model = Model()
    inputs = get_inputs()
    
    # Run forward pass
    with torch.no_grad():
        outputs = model(*inputs)
    
    # Print output shapes
    print(f"✅ Output shapes:")
    for i, out in enumerate(outputs):
        print(f"   output[{i}]: {out.shape}, dtype: {out.dtype}")
    
    # Verify shapes
    num_tokens = inputs[0].shape[0]
    num_q_heads = 14
    num_kv_heads = 7
    head_size = 256
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size
    
    assert outputs[0].shape == (num_tokens, q_size), f"Expected q shape {(num_tokens, q_size)}, got {outputs[0].shape}"
    assert outputs[1].shape == (num_tokens, kv_size), f"Expected k shape {(num_tokens, kv_size)}, got {outputs[1].shape}"
    assert outputs[2].shape == (num_tokens, kv_size), f"Expected v shape {(num_tokens, kv_size)}, got {outputs[2].shape}"
    
    print("✅ All output shapes are correct!")
