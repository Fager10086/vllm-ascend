import json
import os
import torch
import torch.nn as nn


class Model(nn.Module):
    """
    Fused operator that splits a concatenated QKV tensor, applies RMSNorm
    to Q and K heads independently, then applies Rotary Position Embedding
    (RoPE) to the normalized Q and K, and passes V through unchanged.

    The input tensor has shape [batch_size, q_hidden_size + 2 * kv_hidden_size]
    where the first q_hidden_size columns are Q, the next kv_hidden_size columns
    are K, and the last kv_hidden_size columns are V.

    RMSNorm is applied per-head (each head has head_dim elements), then RoPE
    rotation is applied using a cos_sin_cache indexed by position ids.

    This is a common fused kernel in LLM inference for models like Qwen2.

    Pure PyTorch reference implementation of:
        split_qkv_rmsnorm_rope_impl(input, cos_sin_cache, positions,
                                     q_weight, k_weight,
                                     q_hidden_size, kv_hidden_size, head_dim, eps,
                                     q_bias=None, k_bias=None)
        -> (q_output, k_output, v_output)
    """
    def __init__(self):
        super(Model, self).__init__()

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
        """
        Args:
            input (torch.Tensor): Concatenated QKV tensor of shape
                [batch_size, q_hidden_size + 2 * kv_hidden_size], dtype bfloat16/float16.
            cos_sin_cache (torch.Tensor): Precomputed cos/sin cache of shape
                [max_position, head_dim], where the first head_dim//2 elements per row
                are cos values and the last head_dim//2 are sin values.
                dtype bfloat16/float16.
            positions (torch.Tensor): Position ids of shape [batch_size], dtype int64.
            q_weight (torch.Tensor): RMSNorm weight for Q heads, shape [head_dim],
                dtype bfloat16/float16.
            k_weight (torch.Tensor): RMSNorm weight for K heads, shape [head_dim],
                dtype bfloat16/float16.
            q_hidden_size (int): Total hidden size for Q (num_q_heads * head_dim).
            kv_hidden_size (int): Total hidden size for each of K and V
                (num_kv_heads * head_dim).
            head_dim (int): Dimension of each attention head.
            eps (float): Epsilon for RMSNorm numerical stability.
            q_bias (torch.Tensor, optional): Bias for Q RMSNorm, shape [head_dim].
            k_bias (torch.Tensor, optional): Bias for K RMSNorm, shape [head_dim].

        Returns:
            tuple: (q_output, k_output, v_output)
                - q_output: [batch_size, q_hidden_size], RMSNorm + RoPE applied
                - k_output: [batch_size, kv_hidden_size], RMSNorm + RoPE applied
                - v_output: [batch_size, kv_hidden_size], passthrough copy
        """
        batch_size = input.shape[0]
        half_head_dim = head_dim // 2
        orig_dtype = input.dtype

        # --- Split into Q, K, V ---
        q_input = input[:, :q_hidden_size]
        k_input = input[:, q_hidden_size:q_hidden_size + kv_hidden_size]
        v_input = input[:, q_hidden_size + kv_hidden_size:]

        # --- V passthrough ---
        v_output = v_input.clone()

        # --- Process Q: RMSNorm per head + RoPE ---
        num_q_heads = q_hidden_size // head_dim
        q_reshaped = q_input.to(torch.float32).reshape(batch_size, num_q_heads, head_dim)

        # RMSNorm per head (all in float32)
        q_var = (q_reshaped * q_reshaped).sum(dim=-1, keepdim=True) / head_dim
        q_normed = q_reshaped * (1.0 / torch.sqrt(q_var + eps))
        q_weight_f = q_weight.to(torch.float32)
        if q_bias is not None:
            q_normed = q_normed * q_weight_f + q_bias.to(torch.float32)
        else:
            q_normed = q_normed * q_weight_f

        # RoPE for Q (all in float32 for precision)
        cos_vals = cos_sin_cache[positions, :half_head_dim].to(torch.float32)
        sin_vals = cos_sin_cache[positions, half_head_dim:].to(torch.float32)

        q_x1 = q_normed[:, :, :half_head_dim]
        q_x2 = q_normed[:, :, half_head_dim:]
        cos_q = cos_vals.unsqueeze(1)
        sin_q = sin_vals.unsqueeze(1)
        q_roped1 = q_x1 * cos_q - q_x2 * sin_q
        q_roped2 = q_x2 * cos_q + q_x1 * sin_q
        q_roped = torch.cat([q_roped1, q_roped2], dim=-1)
        q_output = q_roped.reshape(batch_size, q_hidden_size).to(orig_dtype)

        # --- Process K: RMSNorm per head + RoPE ---
        num_kv_heads = kv_hidden_size // head_dim
        k_reshaped = k_input.to(torch.float32).reshape(batch_size, num_kv_heads, head_dim)

        # RMSNorm per head (all in float32)
        k_var = (k_reshaped * k_reshaped).sum(dim=-1, keepdim=True) / head_dim
        k_normed = k_reshaped * (1.0 / torch.sqrt(k_var + eps))
        k_weight_f = k_weight.to(torch.float32)
        if k_bias is not None:
            k_normed = k_normed * k_weight_f + k_bias.to(torch.float32)
        else:
            k_normed = k_normed * k_weight_f

        # RoPE for K (all in float32 for precision)
        k_x1 = k_normed[:, :, :half_head_dim]
        k_x2 = k_normed[:, :, half_head_dim:]
        cos_k = cos_vals.unsqueeze(1)
        sin_k = sin_vals.unsqueeze(1)
        k_roped1 = k_x1 * cos_k - k_x2 * sin_k
        k_roped2 = k_x2 * cos_k + k_x1 * sin_k
        k_roped = torch.cat([k_roped1, k_roped2], dim=-1)
        k_output = k_roped.reshape(batch_size, kv_hidden_size).to(orig_dtype)

        return q_output, k_output, v_output


def get_input_groups():
    """Generate input groups from JSON test cases."""
    # Look for any .json file in the same directory as this script.
    # The scaffold copies sidecar JSONs preserving their original names,
    # while renaming this .py to reference.py — so we can't rely on
    # __file__ basename matching the JSON name.
    _dir = os.path.dirname(os.path.abspath(__file__))
    _json_candidates = [f for f in os.listdir(_dir) if f.endswith(".json")
                        and not f.startswith(".")]
    if not _json_candidates:
        raise FileNotFoundError(f"No .json sidecar found in {_dir}")
    json_path = os.path.join(_dir, _json_candidates[0])
    input_groups = []
    with open(json_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            case = json.loads(line)
            inputs = case["inputs"]

            dtype_map = {
                "float32": torch.float32,
                "float16": torch.float16,
                "bfloat16": torch.bfloat16,
            }

            tensors = {}
            attrs = {}
            for inp in inputs:
                name = inp["name"]
                if inp["type"] == "tensor":
                    shape = inp.get("shape")
                    dtype_str = inp.get("dtype", "float32")
                    dtype = dtype_map.get(dtype_str, torch.float32)
                    if shape is None:
                        tensors[name] = None
                    else:
                        tensors[name] = torch.randn(shape, dtype=dtype)
                elif inp["type"] == "attr":
                    attrs[name] = inp["value"]

            # Build positions: random valid indices into cos_sin_cache
            max_position = tensors["cos_sin_cache"].shape[0]
            batch_size = tensors["input"].shape[0]
            positions = torch.randint(0, max_position, (batch_size,), dtype=torch.int64)

            group = [
                tensors["input"],
                tensors["cos_sin_cache"],
                positions,
                tensors["q_weight"],
                tensors["k_weight"],
                attrs["q_hidden_size"],
                attrs["kv_hidden_size"],
                attrs["head_dim"],
                attrs["eps"],
                tensors.get("q_bias"),
                tensors.get("k_bias"),
            ]
            input_groups.append(group)
    return input_groups


def get_init_inputs():
    return []
