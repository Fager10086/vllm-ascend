import numpy as np
import torch
import torch.nn as nn


def _rms_norm(x, w, eps, bias=None):
    x = x.to(torch.float32)
    w = w.to(torch.float32)
    out = x * (1 / torch.sqrt(x.pow(2).mean(-1, keepdim=True) + eps)) * w
    if bias is not None:
        out = out + bias.to(torch.float32)
    return out


def _rope(q, k, sin, cos):
    def rotate(x):
        h = x.shape[-1] // 2
        return torch.cat([-x[..., h:], x[..., :h]], dim=-1) * sin + x * cos
    return rotate(q), rotate(k)


class Model(nn.Module):
    def __init__(self, num_q_heads, num_kv_heads, head_size, eps, max_pos):
        super().__init__()
        self.num_q_heads = num_q_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.eps = eps
        self.max_pos = max_pos

    def forward(self, qkv, q_weight, k_weight, cos_sin_cache, positions):
        T = qkv.shape[0]
        H = self.head_size
        nq, nkv = self.num_q_heads, self.num_kv_heads
        q_size, kv_size = nq * H, nkv * H

        _q, _k, v = qkv.split([q_size, kv_size, kv_size], dim=-1)
        _q = _rms_norm(_q.reshape(-1, H), q_weight, self.eps).reshape(T, 1, nq, H)
        _k = _rms_norm(_k.reshape(-1, H), k_weight, self.eps).reshape(T, 1, nkv, H)

        cos, sin = (
            cos_sin_cache.index_select(0, positions)
            .view(T, 2, -1).repeat(1, 1, 2).chunk(2, dim=-2)
        )
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)

        q_out, k_out = _rope(_q, _k, sin, cos)
        return q_out.reshape(T, q_size).to(qkv.dtype), k_out.reshape(T, kv_size).to(qkv.dtype), v


def get_init_inputs():
    return [32, 4, 128, 1e-6, 262144]


def get_input_groups():
    configs = [
        (1,    32, 4,  128),
        (16,   32, 4,  128),
        (1024, 32, 4,  128),
        (256,  32, 4,  128),
        (4096, 32, 4,  128),
    ]
    max_pos = 262144
    groups = []
    for (T, nq, nkv, H) in configs:
        q_size, kv_size = nq * H, nkv * H
        qkv = torch.randn(T, q_size + kv_size * 2, dtype=torch.bfloat16)
        q_w = torch.randn(H, dtype=torch.bfloat16)
        k_w = torch.randn(H, dtype=torch.bfloat16)
        cos_sin = torch.from_numpy(
            np.random.uniform(0, 1, [max_pos, H]).astype(np.float32)
        ).to(torch.bfloat16)
        pos = torch.randint(0, max_pos, (T,), dtype=torch.int64)
        groups.append((qkv, q_w, k_w, cos_sin, pos))
    return groups
