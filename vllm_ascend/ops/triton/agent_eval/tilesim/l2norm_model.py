"""
l2norm_fwd — tilesim PyTorch adaptor 示例模型。

基于 test_l2norm.py 的 PyTorch Golden 实现:
    ref = F.normalize(x, dim=-1, p=2)

数学公式:
    y = x * rsqrt(sum(x^2, dim=-1) + eps)

等价于:
    x_sq = x * x                         # 逐元素平方
    sum_sq = x_sq.sum(dim=-1, keepdim=True)  # 沿最后一维求和
    inv_norm = torch.rsqrt(sum_sq + eps)  # 逆平方根
    output = x * inv_norm                 # 逐元素乘

运行方式:
    cd D:/算子/算子/vllm-ascend-main/tilesim-master/tilesim-master
    python -m examples.api.operator_api.pytorch_examples.main --script examples/api/operator_api/pytorch_examples/l2norm_model.py
    python -m examples.api.operator_api.pytorch_examples.main --script examples/api/operator_api/pytorch_examples/l2norm_model.py --accelerator core/config/arc_config/910B1/910B1.yaml
"""

import torch
import torch.nn as nn


class L2Norm(nn.Module):
    """L2 归一化，与 F.normalize(x, dim=-1, p=2) 等价。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x):
        x_sq = x * x
        sum_sq = x_sq.sum(dim=-1, keepdim=True)
        inv_norm = 1 / torch.sqrt(sum_sq + self.eps)
        output = x * inv_norm
        return output


class Model(nn.Module):
    """tilesim adaptor 要求的 Model 类。"""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.norm = L2Norm(eps=eps)

    def forward(self, x):
        return self.norm(x)


def get_inputs():
    """adaptor 要求的 get_inputs 函数。

    使用 shape_stats.json 中的 shape: [1, 1514, 4, 128] bfloat16
    """
    torch.manual_seed(42)
    return [torch.randn(1, 1514, 4, 128, dtype=torch.bfloat16)]


def get_inputs_small():
    """小 shape 用于快速验证。"""
    torch.manual_seed(42)
    return [torch.randn(2, 128, 4, 64, dtype=torch.bfloat16)]


def get_inputs_float32():
    """float32 输入。"""
    torch.manual_seed(42)
    return [torch.randn(3, 1024, 4, 128, dtype=torch.float32)]


if __name__ == "__main__":
    model = Model()
    inputs = get_inputs()
    y = model(*inputs)
    print(f"Input:  {inputs[0].shape} {inputs[0].dtype}")
    print(f"Output: {y.shape} {y.dtype}")

    ref = torch.nn.functional.normalize(inputs[0].to(torch.float32), dim=-1, p=2)
    diff = (y.to(torch.float32) - ref).abs().max().item()
    print(f"Max diff vs F.normalize: {diff:.2e}")
    print("Model runs OK")
