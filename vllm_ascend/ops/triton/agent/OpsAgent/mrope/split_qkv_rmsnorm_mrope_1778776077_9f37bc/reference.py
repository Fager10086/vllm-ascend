"""
Reference (pure PyTorch) implementation for split_qkv_rmsnorm_mrope.
Adapted from test_split_qkv_rmsnorm_mrope.py for AutoResearch evaluation.
"""
import torch
import torch.nn as nn


class Model(nn.Module):
    """Pure PyTorch reference for split_qkv + RMSNorm + MRoPE."""

    def __init__(
        self,
        num_q_heads: int = 8,
        num_kv_heads: int = 2,
        head_size: int = 128,
        eps: float = 1e-6,
        mrope_section: list | None = None,
        is_interleaved: bool = False,
        rope_dim: int | None = None,
        has_gate: bool = False,
    ) -> None:
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.eps = eps
        self.mrope_section = mrope_section or [11, 11, 10]
        self.is_interleaved = is_interleaved
        self.rope_dim = rope_dim or 2 * sum(self.mrope_section)
        self.has_gate = has_gate

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
        x_fp32 = x.to(torch.float32)
        w_fp32 = weight.to(torch.float32)
        reciprocal_std = 1 / torch.sqrt(
            torch.mean(x_fp32 ** 2, dim=-1, keepdim=True) + self.eps
        )
        return x_fp32 * reciprocal_std * w_fp32

    def _apply_interleaved_rope(self, x: torch.Tensor) -> torch.Tensor:
        """Apply interleaved MRoPE: reorganize frequency layout."""
        s = self.mrope_section
        x_t = x[0].clone()
        x_t[..., 1:s[1] * 3:3] = x[1, ..., 1:s[1] * 3:3]
        x_t[..., 2:s[2] * 3:3] = x[2, ..., 2:s[2] * 3:3]
        return x_t

    def forward(
        self,
        qkv: torch.Tensor,
        q_weight: torch.Tensor,
        k_weight: torch.Tensor,
        cos_sin: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        num_q_heads = self.num_q_heads
        num_kv_heads = self.num_kv_heads
        head_size = self.head_size
        mrope_section = self.mrope_section
        rope_dim = self.rope_dim
        half_rd = rope_dim // 2

        q_size = num_q_heads * head_size
        kv_size = num_kv_heads * head_size
        num_tokens = qkv.shape[0]

        # ---- Split QKV (with optional gate) ----
        if self.has_gate:
            q_gate_data = qkv[:, :q_size * 2].view(-1, num_q_heads, head_size * 2)
            q_data, gate = torch.chunk(q_gate_data, 2, dim=-1)
            gate = gate.reshape(-1, q_size)
            q_data = q_data.reshape(-1, q_size)
            k_data = qkv[:, 2 * q_size:2 * q_size + kv_size]
            v = qkv[:, 2 * q_size + kv_size:]
        else:
            q_data, k_data, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
            gate = torch.empty(num_tokens, 0, device=qkv.device, dtype=qkv.dtype)

        # ---- RMSNorm ----
        q = self._rms_norm(q_data.reshape(-1, head_size), q_weight)
        k = self._rms_norm(k_data.reshape(-1, head_size), k_weight)

        # ---- MRoPE ----
        q_reshaped = q.view(num_tokens, num_q_heads, head_size)
        k_reshaped = k.view(num_tokens, num_kv_heads, head_size)

        cos, sin = cos_sin.chunk(2, dim=-1)  # each: (3, num_tokens, half_rope_dim)

        if self.is_interleaved:
            cos_combined = self._apply_interleaved_rope(cos)
            sin_combined = self._apply_interleaved_rope(sin)

            for token_idx in range(num_tokens):
                cos_row = cos_combined[token_idx].unsqueeze(0)
                sin_row = sin_combined[token_idx].unsqueeze(0)

                q_t = q_reshaped[token_idx]
                k_t = k_reshaped[token_idx]
                q1, q2 = q_t[:, :half_rd], q_t[:, half_rd:rope_dim]
                k1, k2 = k_t[:, :half_rd], k_t[:, half_rd:rope_dim]

                q_reshaped[token_idx, :, :half_rd] = q1 * cos_row - q2 * sin_row
                q_reshaped[token_idx, :, half_rd:rope_dim] = q2 * cos_row + q1 * sin_row
                k_reshaped[token_idx, :, :half_rd] = k1 * cos_row - k2 * sin_row
                k_reshaped[token_idx, :, half_rd:rope_dim] = k2 * cos_row + k1 * sin_row
        else:
            cos_perm = cos.permute(1, 2, 0)  # (num_tokens, half_rope_dim, 3)
            sin_perm = sin.permute(1, 2, 0)

            for token_idx in range(num_tokens):
                token_cos = cos_perm[token_idx]
                token_sin = sin_perm[token_idx]

                cos_row = torch.zeros(half_rd, device=qkv.device, dtype=q.dtype)
                sin_row = torch.zeros(half_rd, device=qkv.device, dtype=q.dtype)

                t_end = mrope_section[0]
                h_end = t_end + mrope_section[1]

                cos_row[:t_end] = token_cos[:t_end, 0]
                sin_row[:t_end] = token_sin[:t_end, 0]
                cos_row[t_end:h_end] = token_cos[t_end:h_end, 1]
                sin_row[t_end:h_end] = token_sin[t_end:h_end, 1]
                cos_row[h_end:half_rd] = token_cos[h_end:half_rd, 2]
                sin_row[h_end:half_rd] = token_sin[h_end:half_rd, 2]

                cos_half = cos_row.unsqueeze(0)
                sin_half = sin_row.unsqueeze(0)

                q_t = q_reshaped[token_idx]
                k_t = k_reshaped[token_idx]
                q1, q2 = q_t[:, :half_rd], q_t[:, half_rd:rope_dim]
                k1, k2 = k_t[:, :half_rd], k_t[:, half_rd:rope_dim]

                q_reshaped[token_idx, :, :rope_dim] = torch.cat(
                    [q1 * cos_half - q2 * sin_half, q2 * cos_half + q1 * sin_half], dim=1
                )
                k_reshaped[token_idx, :, :rope_dim] = torch.cat(
                    [k1 * cos_half - k2 * sin_half, k2 * cos_half + k1 * sin_half], dim=1
                )

        q_out = q_reshaped.reshape(num_tokens, -1).to(qkv.dtype)
        k_out = k_reshaped.reshape(num_tokens, -1).to(qkv.dtype)
        v_out = v.to(qkv.dtype)

        return q_out, k_out, v_out, gate


# ---------------------------------------------------------------------------
# AutoResearch required exports
# ---------------------------------------------------------------------------

def get_init_inputs():
    """Constructor args for Model(). Matches a typical non-interleaved, no-gate config."""
    #         num_q_heads, num_kv_heads, head_size, eps,
    #         mrope_section,    is_interleaved, rope_dim, has_gate
    return [8, 2, 128, 1e-6, [11, 11, 10], False, 64, False]


def get_input_groups():
    """Multiple input cases with varying num_tokens for performance profiling."""
    num_q_heads, num_kv_heads = 8, 2
    head_size = 128
    rope_dim = 64  # 2 * sum([11, 11, 10])
    q_size = num_q_heads * head_size
    kv_size = num_kv_heads * head_size

    input_groups = []
    for num_tokens in [1, 4, 8, 16, 256, 1024, 4096]:
        g = torch.Generator().manual_seed(42)
        qkv = torch.randn(num_tokens, q_size + kv_size * 2, dtype=torch.bfloat16, generator=g)
        q_weight = torch.randn(head_size, dtype=torch.bfloat16, generator=g)
        k_weight = torch.randn(head_size, dtype=torch.bfloat16, generator=g)
        cos_sin = torch.randn(3, num_tokens, rope_dim, dtype=torch.bfloat16, generator=g)
        input_groups.append([qkv, q_weight, k_weight, cos_sin])

    return input_groups
