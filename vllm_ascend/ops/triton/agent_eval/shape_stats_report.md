# vLLM Ascend 算子 Shape 统计报告

**统计时间**: 2026-04-20 18:54:39 - 2026-04-20 18:56:21

**总记录数**: 23891

**唯一算子数**: 17

## 算子调用统计

| 算子名称 | 调用次数 | 参数 Shape 分布 |
|:---|:---:|:---|
| custom_op.gdn_attention_core | 10656 | arg0: [1, 2048]:torch.bfloat16(9792), [16, 2048]:torch.bfloat16(192), [8, 2048]:torch.bfloat16(192), [4, 2048]:torch.bfloat16(192), [2, 2048]:torch.bfloat16(192), [8192, 2048]:torch.bfloat16(96); arg1: [1, 8]:torch.bfloat16(9792), [16, 8]:torch.bfloat16(192), [8, 8]:torch.bfloat16(192), [4, 8]:torch.bfloat16(192), [2, 8]:torch.bfloat16(192), [8192, 8]:torch.bfloat16(96); arg2: [1, 8]:torch.bfloat16(9792), [16, 8]:torch.bfloat16(192), [8, 8]:torch.bfloat16(192), [4, 8]:torch.bfloat16(192), [2, 8]:torch.bfloat16(192), [8192, 8]:torch.bfloat16(96); arg3: [1, 8, 128]:torch.bfloat16(9792), [16, 8, 128]:torch.bfloat16(192), [8, 8, 128]:torch.bfloat16(192), [4, 8, 128]:torch.bfloat16(192), [2, 8, 128]:torch.bfloat16(192), [8192, 8, 128]:torch.bfloat16(96) |
| custom_op.unquantized_gemm | 7880 | arg0: [1, 2560]:torch.bfloat16(1108), [16, 2560]:torch.bfloat16(708), [8, 2560]:torch.bfloat16(704), [4, 2560]:torch.bfloat16(704), [2, 2560]:torch.bfloat16(704), [65536, 1, 1024]:torch.bfloat16(576), [8192, 2560]:torch.bfloat16(352), [16, 1024]:torch.bfloat16(256), [16, 2304]:torch.bfloat16(256), [8, 1024]:torch.bfloat16(256), [8, 2304]:torch.bfloat16(256), [4, 1024]:torch.bfloat16(256), [4, 2304]:torch.bfloat16(256), [2, 1024]:torch.bfloat16(256), [2, 2304]:torch.bfloat16(256), [1, 1024]:torch.bfloat16(256), [1, 2304]:torch.bfloat16(256), [65536, 1, 256]:torch.bfloat16(192), [8192, 1024]:torch.bfloat16(128), [8192, 2304]:torch.bfloat16(128), [16384, 4096]:torch.bfloat16(8), [16384, 1024]:torch.bfloat16(8); arg1: [2560, 1024]:torch.bfloat16(1416), [4608, 2560]:torch.bfloat16(1408), [2560, 2304]:torch.bfloat16(1408), [3072, 2560]:torch.bfloat16(1056), [16, 2560]:torch.bfloat16(1056), [62080, 2560]:torch.bfloat16(408), [1024, 1024]:torch.bfloat16(384), [2560, 2560]:torch.bfloat16(352), [768, 1024]:torch.bfloat16(192), [1024, 256]:torch.bfloat16(192), [1024, 4096]:torch.bfloat16(8); arg2: [1024]:torch.bfloat16(296), [768]:torch.bfloat16(192), [2560]:torch.bfloat16(2) |
| custom_op.maybe_chunk_residual | 2816 | arg0: [16, 2560]:torch.bfloat16(512), [8, 2560]:torch.bfloat16(512), [4, 2560]:torch.bfloat16(512), [2, 2560]:torch.bfloat16(512), [1, 2560]:torch.bfloat16(512), [8192, 2560]:torch.bfloat16(256); arg1: [16, 2560]:torch.bfloat16(512), [8, 2560]:torch.bfloat16(512), [4, 2560]:torch.bfloat16(512), [2, 2560]:torch.bfloat16(512), [1, 2560]:torch.bfloat16(512), [8192, 2560]:torch.bfloat16(256) |
| all_reduce | 452 | arg0: [1, 2560]:torch.bfloat16(256), [65536, 1, 1024]:torch.bfloat16(192), [16384, 2560]:torch.bfloat16(4) |
| custom_op.maybe_pad_and_reduce | 400 | arg0: [1, 2560]:torch.bfloat16(400) |
| linear | 396 | arg0: [65536, 1, 1024]:torch.bfloat16(288), [65536, 1, 256]:torch.bfloat16(96), [16384, 4096]:torch.bfloat16(4), [16384, 1024]:torch.bfloat16(4), [1, 2560]:torch.bfloat16(4); arg1: [1024, 1024]:torch.bfloat16(192), [768, 1024]:torch.bfloat16(96), [1024, 256]:torch.bfloat16(96), [1024, 4096]:torch.bfloat16(4), [2560, 1024]:torch.bfloat16(4), [62080, 2560]:torch.bfloat16(4); arg2: [1024]:torch.bfloat16(148), [768]:torch.bfloat16(96), [2560]:torch.bfloat16(1) |
| custom_op.triton_split_qkv_rmsnorm_mrope | 352 | arg0: [16, 2560]:torch.bfloat16(64), [8, 2560]:torch.bfloat16(64), [4, 2560]:torch.bfloat16(64), [2, 2560]:torch.bfloat16(64), [1, 2560]:torch.bfloat16(64), [8192, 2560]:torch.bfloat16(32); arg1: [256]:torch.bfloat16(352); arg2: [256]:torch.bfloat16(352); arg3: [3, 16, 64]:torch.bfloat16(64), [3, 8, 64]:torch.bfloat16(64), [3, 4, 64]:torch.bfloat16(64), [3, 2, 64]:torch.bfloat16(64), [3, 1, 64]:torch.bfloat16(64), [3, 8192, 64]:torch.bfloat16(32) |
| npu_add_rms_norm_bias | 256 | arg0: [1, 2560]:torch.bfloat16(256); arg1: [1, 2560]:torch.bfloat16(256); arg2: [2560]:torch.bfloat16(256) |
| cat | 223 | arg0: [65536, 32]:torch.bfloat16(384), [8192, 16]:torch.float32(8), [1048576, 32]:torch.float32(8), [1, 1, 3, 4096, 4096]:torch.float32(8), [65536, 1536]:torch.float32(4), [65536, 1024]:torch.bfloat16(4), [65536, 2]:torch.int64(4), [16384, 2560]:torch.bfloat16(4), [1]:torch.int64(3) |
| npu_swiglu | 128 | arg0: [1, 4608]:torch.bfloat16(128) |
| npu_rotary_mul | 96 | arg0: [2, 65536, 4, 64]:torch.bfloat16(96); arg1: [1, 65536, 1, 64]:torch.bfloat16(96); arg2: [1, 65536, 1, 64]:torch.bfloat16(96) |
| layer_norm_fwd_npu | 96 | arg0: [8, 128]:torch.bfloat16(96); arg1: [128]:torch.bfloat16(96) |
| custom_op.maybe_all_gather_and_maybe_unpad | 64 | arg0: [2, 4]:torch.bfloat16(64) |
| custom_op.quantize | 64 | arg0: [2, 4]:torch.bfloat16(64); arg1: [4]:torch.bfloat16(64); arg2: [4]:torch.bfloat16(64); arg3: [4]:torch.bfloat16(64) |
| npu_gemma_rms_norm | 4 | arg0: [1, 2560]:torch.bfloat16(4); arg1: [2560]:torch.bfloat16(4) |
| all_gather_into_tensor | 4 | arg0: [4, 62080]:torch.bfloat16(4); arg1: [1, 62080]:torch.bfloat16(4) |
| all_gather | 4 | arg0: [1, 62080]:torch.bfloat16(4) |

## 详细 Shape 分布

### custom_op.gdn_attention_core

- **调用次数**: 10656
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2048] | torch.bfloat16 | 9792 |
| [16, 2048] | torch.bfloat16 | 192 |
| [8, 2048] | torch.bfloat16 | 192 |
| [4, 2048] | torch.bfloat16 | 192 |
| [2, 2048] | torch.bfloat16 | 192 |
| [8192, 2048] | torch.bfloat16 | 96 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 8] | torch.bfloat16 | 9792 |
| [16, 8] | torch.bfloat16 | 192 |
| [8, 8] | torch.bfloat16 | 192 |
| [4, 8] | torch.bfloat16 | 192 |
| [2, 8] | torch.bfloat16 | 192 |
| [8192, 8] | torch.bfloat16 | 96 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 8] | torch.bfloat16 | 9792 |
| [16, 8] | torch.bfloat16 | 192 |
| [8, 8] | torch.bfloat16 | 192 |
| [4, 8] | torch.bfloat16 | 192 |
| [2, 8] | torch.bfloat16 | 192 |
| [8192, 8] | torch.bfloat16 | 96 |

| arg3 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 8, 128] | torch.bfloat16 | 9792 |
| [16, 8, 128] | torch.bfloat16 | 192 |
| [8, 8, 128] | torch.bfloat16 | 192 |
| [4, 8, 128] | torch.bfloat16 | 192 |
| [2, 8, 128] | torch.bfloat16 | 192 |
| [8192, 8, 128] | torch.bfloat16 | 96 |

### custom_op.unquantized_gemm

- **调用次数**: 7880
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 1108 |
| [16, 2560] | torch.bfloat16 | 708 |
| [8, 2560] | torch.bfloat16 | 704 |
| [4, 2560] | torch.bfloat16 | 704 |
| [2, 2560] | torch.bfloat16 | 704 |
| [65536, 1, 1024] | torch.bfloat16 | 576 |
| [8192, 2560] | torch.bfloat16 | 352 |
| [16, 1024] | torch.bfloat16 | 256 |
| [16, 2304] | torch.bfloat16 | 256 |
| [8, 1024] | torch.bfloat16 | 256 |
| [8, 2304] | torch.bfloat16 | 256 |
| [4, 1024] | torch.bfloat16 | 256 |
| [4, 2304] | torch.bfloat16 | 256 |
| [2, 1024] | torch.bfloat16 | 256 |
| [2, 2304] | torch.bfloat16 | 256 |
| [1, 1024] | torch.bfloat16 | 256 |
| [1, 2304] | torch.bfloat16 | 256 |
| [65536, 1, 256] | torch.bfloat16 | 192 |
| [8192, 1024] | torch.bfloat16 | 128 |
| [8192, 2304] | torch.bfloat16 | 128 |
| [16384, 4096] | torch.bfloat16 | 8 |
| [16384, 1024] | torch.bfloat16 | 8 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [2560, 1024] | torch.bfloat16 | 1416 |
| [4608, 2560] | torch.bfloat16 | 1408 |
| [2560, 2304] | torch.bfloat16 | 1408 |
| [3072, 2560] | torch.bfloat16 | 1056 |
| [16, 2560] | torch.bfloat16 | 1056 |
| [62080, 2560] | torch.bfloat16 | 408 |
| [1024, 1024] | torch.bfloat16 | 384 |
| [2560, 2560] | torch.bfloat16 | 352 |
| [768, 1024] | torch.bfloat16 | 192 |
| [1024, 256] | torch.bfloat16 | 192 |
| [1024, 4096] | torch.bfloat16 | 8 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [1024] | torch.bfloat16 | 296 |
| [768] | torch.bfloat16 | 192 |
| [2560] | torch.bfloat16 | 2 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [65536, 1, 1024] | torch.bfloat16 | 576 |
| [16, 2560] | torch.bfloat16 | 576 |
| [8, 2560] | torch.bfloat16 | 576 |
| [4, 2560] | torch.bfloat16 | 576 |
| [2, 2560] | torch.bfloat16 | 576 |
| [1, 2560] | torch.bfloat16 | 576 |
| [1, 62080] | torch.bfloat16 | 404 |
| [8192, 2560] | torch.bfloat16 | 288 |
| [16, 4608] | torch.bfloat16 | 256 |
| [8, 4608] | torch.bfloat16 | 256 |
| [4, 4608] | torch.bfloat16 | 256 |
| [2, 4608] | torch.bfloat16 | 256 |
| [1, 4608] | torch.bfloat16 | 256 |
| [65536, 1, 768] | torch.bfloat16 | 192 |
| [16, 3072] | torch.bfloat16 | 192 |
| [16, 16] | torch.bfloat16 | 192 |
| [8, 3072] | torch.bfloat16 | 192 |
| [8, 16] | torch.bfloat16 | 192 |
| [4, 3072] | torch.bfloat16 | 192 |
| [4, 16] | torch.bfloat16 | 192 |
| [2, 3072] | torch.bfloat16 | 192 |
| [2, 16] | torch.bfloat16 | 192 |
| [1, 3072] | torch.bfloat16 | 192 |
| [1, 16] | torch.bfloat16 | 192 |
| [8192, 4608] | torch.bfloat16 | 128 |
| [8192, 3072] | torch.bfloat16 | 96 |
| [8192, 16] | torch.bfloat16 | 96 |
| [16384, 1024] | torch.bfloat16 | 8 |
| [16384, 2560] | torch.bfloat16 | 8 |
| [16, 62080] | torch.bfloat16 | 4 |

### custom_op.maybe_chunk_residual

- **调用次数**: 2816
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [16, 2560] | torch.bfloat16 | 512 |
| [8, 2560] | torch.bfloat16 | 512 |
| [4, 2560] | torch.bfloat16 | 512 |
| [2, 2560] | torch.bfloat16 | 512 |
| [1, 2560] | torch.bfloat16 | 512 |
| [8192, 2560] | torch.bfloat16 | 256 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [16, 2560] | torch.bfloat16 | 512 |
| [8, 2560] | torch.bfloat16 | 512 |
| [4, 2560] | torch.bfloat16 | 512 |
| [2, 2560] | torch.bfloat16 | 512 |
| [1, 2560] | torch.bfloat16 | 512 |
| [8192, 2560] | torch.bfloat16 | 256 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [16, 2560] | torch.bfloat16 | 512 |
| [8, 2560] | torch.bfloat16 | 512 |
| [4, 2560] | torch.bfloat16 | 512 |
| [2, 2560] | torch.bfloat16 | 512 |
| [1, 2560] | torch.bfloat16 | 512 |
| [8192, 2560] | torch.bfloat16 | 256 |

### all_reduce

- **调用次数**: 452
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 256 |
| [65536, 1, 1024] | torch.bfloat16 | 192 |
| [16384, 2560] | torch.bfloat16 | 4 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 256 |
| [65536, 1, 1024] | torch.bfloat16 | 192 |
| [16384, 2560] | torch.bfloat16 | 4 |

### custom_op.maybe_pad_and_reduce

- **调用次数**: 400
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 400 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 400 |

### linear

- **调用次数**: 396
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [65536, 1, 1024] | torch.bfloat16 | 288 |
| [65536, 1, 256] | torch.bfloat16 | 96 |
| [16384, 4096] | torch.bfloat16 | 4 |
| [16384, 1024] | torch.bfloat16 | 4 |
| [1, 2560] | torch.bfloat16 | 4 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [1024, 1024] | torch.bfloat16 | 192 |
| [768, 1024] | torch.bfloat16 | 96 |
| [1024, 256] | torch.bfloat16 | 96 |
| [1024, 4096] | torch.bfloat16 | 4 |
| [2560, 1024] | torch.bfloat16 | 4 |
| [62080, 2560] | torch.bfloat16 | 4 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [1024] | torch.bfloat16 | 148 |
| [768] | torch.bfloat16 | 96 |
| [2560] | torch.bfloat16 | 1 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [65536, 1, 1024] | torch.bfloat16 | 288 |
| [65536, 1, 768] | torch.bfloat16 | 96 |
| [16384, 1024] | torch.bfloat16 | 4 |
| [16384, 2560] | torch.bfloat16 | 4 |
| [1, 62080] | torch.bfloat16 | 4 |

### custom_op.triton_split_qkv_rmsnorm_mrope

- **调用次数**: 352
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [16, 2560] | torch.bfloat16 | 64 |
| [8, 2560] | torch.bfloat16 | 64 |
| [4, 2560] | torch.bfloat16 | 64 |
| [2, 2560] | torch.bfloat16 | 64 |
| [1, 2560] | torch.bfloat16 | 64 |
| [8192, 2560] | torch.bfloat16 | 32 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [256] | torch.bfloat16 | 352 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [256] | torch.bfloat16 | 352 |

| arg3 Shape | DType | 次数 |
|:---|:---|:---:|
| [3, 16, 64] | torch.bfloat16 | 64 |
| [3, 8, 64] | torch.bfloat16 | 64 |
| [3, 4, 64] | torch.bfloat16 | 64 |
| [3, 2, 64] | torch.bfloat16 | 64 |
| [3, 1, 64] | torch.bfloat16 | 64 |
| [3, 8192, 64] | torch.bfloat16 | 32 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [16, 1024] | torch.bfloat16 | 128 |
| [16, 256] | torch.bfloat16 | 128 |
| [8, 1024] | torch.bfloat16 | 128 |
| [8, 256] | torch.bfloat16 | 128 |
| [4, 1024] | torch.bfloat16 | 128 |
| [4, 256] | torch.bfloat16 | 128 |
| [2, 1024] | torch.bfloat16 | 128 |
| [2, 256] | torch.bfloat16 | 128 |
| [1, 1024] | torch.bfloat16 | 128 |
| [1, 256] | torch.bfloat16 | 128 |
| [8192, 1024] | torch.bfloat16 | 64 |
| [8192, 256] | torch.bfloat16 | 64 |

### npu_add_rms_norm_bias

- **调用次数**: 256
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 256 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 256 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [2560] | torch.bfloat16 | 256 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 512 |
| [1, 1] | torch.float32 | 256 |

### cat

- **调用次数**: 223
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [65536, 32] | torch.bfloat16 | 384 |
| [8192, 16] | torch.float32 | 8 |
| [1048576, 32] | torch.float32 | 8 |
| [1, 1, 3, 4096, 4096] | torch.float32 | 8 |
| [65536, 1536] | torch.float32 | 4 |
| [65536, 1024] | torch.bfloat16 | 4 |
| [65536, 2] | torch.int64 | 4 |
| [16384, 2560] | torch.bfloat16 | 4 |
| [1] | torch.int64 | 3 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [65536, 64] | torch.bfloat16 | 192 |
| [8192, 32] | torch.float32 | 4 |
| [1048576, 64] | torch.float32 | 4 |
| [1, 2, 3, 4096, 4096] | torch.float32 | 4 |
| [65536, 1536] | torch.float32 | 4 |
| [65536, 1024] | torch.bfloat16 | 4 |
| [65536, 2] | torch.int64 | 4 |
| [16384, 2560] | torch.bfloat16 | 4 |
| [1] | torch.int64 | 3 |

### npu_swiglu

- **调用次数**: 128
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 4608] | torch.bfloat16 | 128 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2304] | torch.bfloat16 | 128 |

### npu_rotary_mul

- **调用次数**: 96
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 65536, 4, 64] | torch.bfloat16 | 96 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 65536, 1, 64] | torch.bfloat16 | 96 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 65536, 1, 64] | torch.bfloat16 | 96 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 65536, 4, 64] | torch.bfloat16 | 96 |

### layer_norm_fwd_npu

- **调用次数**: 96
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [8, 128] | torch.bfloat16 | 96 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [128] | torch.bfloat16 | 96 |

#### 关键字参数 Shape 分布

| z Shape | DType | 次数 |
|:---|:---|:---:|
| [8, 128] | torch.bfloat16 | 96 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [8, 128] | torch.bfloat16 | 96 |
| [8] | torch.float32 | 96 |

### custom_op.maybe_all_gather_and_maybe_unpad

- **调用次数**: 64
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 4] | torch.bfloat16 | 64 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 4] | torch.bfloat16 | 64 |

### custom_op.quantize

- **调用次数**: 64
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 4] | torch.bfloat16 | 64 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [4] | torch.bfloat16 | 64 |

| arg2 Shape | DType | 次数 |
|:---|:---|:---:|
| [4] | torch.bfloat16 | 64 |

| arg3 Shape | DType | 次数 |
|:---|:---|:---:|
| [4] | torch.bfloat16 | 64 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [2, 4] | torch.int8 | 64 |

### npu_gemma_rms_norm

- **调用次数**: 4
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 4 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [2560] | torch.bfloat16 | 4 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 2560] | torch.bfloat16 | 4 |
| [1, 1] | torch.float32 | 4 |

### all_gather_into_tensor

- **调用次数**: 4
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [4, 62080] | torch.bfloat16 | 4 |

| arg1 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 62080] | torch.bfloat16 | 4 |

### all_gather

- **调用次数**: 4
#### 参数 Shape 分布

| arg0 Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 62080] | torch.bfloat16 | 4 |

#### 输出 Shape 分布

| Shape | DType | 次数 |
|:---|:---|:---:|
| [1, 248320] | torch.bfloat16 | 4 |

