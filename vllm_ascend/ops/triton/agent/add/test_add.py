import torch
from add_kernel import add

DEFAULT_ATOL = 5e-2
DEFAULT_RTOL = 5e-3

@torch.inference_mode()
def test_add():
    torch.manual_seed(0)
    size = 98432
    x = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
    y = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
    output_torch = x + y
    output_triton = add(x, y)

    torch.testing.assert_close(output_torch.cpu(),
                                output_triton.cpu(),
                                atol=DEFAULT_ATOL,
                                rtol=DEFAULT_RTOL)

    # print(output_torch)
    # print(output_triton)
    # print(f'The maximum difference between torch and triton is '
    # f'{torch.max(torch.abs(output_torch - output_triton))}')
