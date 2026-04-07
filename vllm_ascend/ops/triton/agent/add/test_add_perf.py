import torch
import gc
from add_kernel import add
import time

torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
y = torch.rand(size, device='npu')  # 【修改】指定为昇腾NPU设备
output_torch = x + y
num_warm_up = 20
for _ in range(num_warm_up):
    output_triton = add(x, y)
torch.npu.synchronize()
gc.collect()
torch.npu.empty_cache()
torch.npu.reset_peak_memory_stats()

num_runs = 20
start = time.perf_counter()

for _ in range(num_runs):
    output_triton = add(x, y)

torch.npu.synchronize()
end = time.perf_counter()
avg_ms_host = (end - start) / num_runs * 1000

print(f"Task Duration: {avg_ms_host:.3f} ms")

gc.collect()
torch.npu.empty_cache()
torch.npu.reset_peak_memory_stats()
