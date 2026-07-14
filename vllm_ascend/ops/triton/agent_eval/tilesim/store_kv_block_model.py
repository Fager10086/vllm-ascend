"""
store_kv_block — tilesim PyTorch adaptor 示例模型。

基于 test_store_kv_block.py 中的 store_kv_block 算子实现:
    输入：keylist, cache_table, slot_mapping
    输出：更新后的 cache_table

数学公式:
    for token_idx, slot in enumerate(slotmap):
        if slot < 0:
            continue
        block_idx = slot // block_size
        block_offset = slot % block_size
        expected_cache[block_idx, block_offset, :, :] = keylist[token_idx, :, :]

运行方式:
    cd D:/算子/算子/vllm-ascend-main/tilesim-master/tilesim-master
    python -m examples.api.operator_api.pytorch_examples.main --script examples/api/operator_api/pytorch_examples/store_kv_block_model.py
    python -m examples.api.operator_api.pytorch_examples.main --script examples/api/operator_api/pytorch_examples/store_kv_block_model.py --accelerator core/config/arc_config/910B1/910B1.yaml
"""

import torch
import torch.nn as nn


class StoreKVBlock(nn.Module):
    """Store KV Block 操作实现。"""

    def __init__(self, block_size: int = 128):
        super().__init__()
        self.block_size = block_size

    def forward(self, keylist, cache_table, slot_mapping):
        """
        Args:
            keylist: [num_tokens, num_head, head_size] tensor
            cache_table: [num_blocks, block_size, num_head, head_size] tensor
            slot_mapping: [num_tokens] tensor with slot indices

        Returns:
            updated cache_table
        """
        expected_cache = cache_table.clone()
        num_tokens = keylist.shape[0]

        for token_idx in range(num_tokens):
            slot = slot_mapping[token_idx].item()
            if slot < 0:
                continue

            block_idx = slot // self.block_size
            block_offset = slot % self.block_size

            expected_cache[block_idx, block_offset, :, :] = keylist[token_idx, :, :]

        return expected_cache


class Model(nn.Module):
    """tilesim adaptor 要求的 Model 类。"""

    def __init__(self, block_size: int = 128):
        super().__init__()
        self.store_kv_block = StoreKVBlock(block_size=block_size)

    def forward(self, keylist, cache_table, slot_mapping):
        return self.store_kv_block(keylist, cache_table, slot_mapping)


def get_inputs():
    """adaptor 要求的 get_inputs 函数。

    使用 test_store_kv_block.py 中的 shape:
        keylist: [32768, 1, 128] float16
        cache_table: [1773, 128, 1, 128] float16
        slot_mapping: [32768] int32 (连续映射)
    """
    torch.manual_seed(42)
    num_tokens = 32 * 1024
    num_head = 1
    num_blocks = 1773
    head_size = 128
    block_size = 128

    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = list(range(num_tokens))
    slot_mapping = torch.tensor(slotmap, dtype=torch.int32)

    return [keylist, cache_table, slot_mapping]


def get_inputs_small():
    """小 shape 用于快速验证。"""
    torch.manual_seed(42)
    num_tokens = 256
    num_head = 1
    num_blocks = 4
    head_size = 64
    block_size = 128

    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    slotmap = list(range(num_tokens))
    slot_mapping = torch.tensor(slotmap, dtype=torch.int32)

    return [keylist, cache_table, slot_mapping]


def get_inputs_discontinuous():
    """非连续 slot mapping 用于测试。"""
    torch.manual_seed(42)
    num_tokens = 1024
    num_head = 1
    num_blocks = 256
    head_size = 128
    block_size = 128

    keylist = torch.rand(
        size=(num_tokens, num_head, head_size),
        dtype=torch.float16,
    )

    cache_table = torch.rand(
        size=(num_blocks, block_size, num_head, head_size),
        dtype=torch.float16,
    )

    import random
    slotmap = []
    r = 0
    for i in range(num_tokens):
        r = r + random.randint(0, 5)
        slotmap.append(i + r)

    slot_mapping = torch.tensor(slotmap, dtype=torch.int32)

    return [keylist, cache_table, slot_mapping]


def golden_store_kv_block(keylist, cache_table, slot_mapping, block_size):
    """Golden 实现用于验证。"""
    expected_cache = cache_table.clone()
    num_tokens = keylist.shape[0]

    for token_idx in range(num_tokens):
        slot = slot_mapping[token_idx].item()
        if slot < 0:
            continue

        block_idx = slot // block_size
        block_offset = slot % block_size

        expected_cache[block_idx, block_offset, :, :] = keylist[token_idx, :, :]

    return expected_cache


if __name__ == "__main__":
    model = Model(block_size=128)
    inputs = get_inputs_small()
    y = model(*inputs)
    print(f"keylist:      {inputs[0].shape} {inputs[0].dtype}")
    print(f"cache_table:  {inputs[1].shape} {inputs[1].dtype}")
    print(f"slot_mapping: {inputs[2].shape} {inputs[2].dtype}")
    print(f"output:       {y.shape} {y.dtype}")

    ref = golden_store_kv_block(inputs[0], inputs[1], inputs[2], block_size=128)
    diff = (y - ref).abs().max().item()
    print(f"Max diff vs golden: {diff:.2e}")
    print("Model runs OK")
