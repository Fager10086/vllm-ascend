# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import MagicMock

from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

# Skip if triton.runtime is mocked (unit test environment)
import sys
_triton_runtime_is_mocked = isinstance(sys.modules.get('triton.runtime'), MagicMock)

pytestmark = pytest.mark.skipif(
    _triton_runtime_is_mocked,
    reason="Triton kernel execution requires real triton.runtime (not mocked)"
)


def torch_chunk_local_cumsum(
    g: torch.Tensor,
    chunk_size: int,
    reverse: bool = False,
    scale: float | None = None,
    cu_seqlens: torch.Tensor | None = None,
    head_first: bool = False,
) -> torch.Tensor:
    """Reference implementation using PyTorch native cumsum.
    Note: Returns result in input dtype to match operator behavior."""
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape
    
    if cu_seqlens is not None:
        assert B == 1, "Only batch size 1 is supported when cu_seqlens are provided"
        result = torch.zeros_like(g, dtype=g.dtype)
        
        for i in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            
            if head_first:
                seq_data = g[0, :, start:end]  # (H, seq_len)
            else:
                seq_data = g[0, start:end, :]  # (seq_len, H)
            
            # Process in chunks
            num_chunks = (seq_len + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, seq_len)
                
                if head_first:
                    chunk_data = seq_data[:, chunk_start:chunk_end]  # (H, chunk_len)
                    chunk_cumsum = torch.cumsum(chunk_data, dim=1, dtype=torch.float32)
                    if reverse:
                        chunk_cumsum = torch.flip(torch.cumsum(torch.flip(chunk_data, dims=[1]), dim=1, dtype=torch.float32), dims=[1])
                    if scale is not None:
                        chunk_cumsum = chunk_cumsum * scale
                    result[0, :, start + chunk_start:start + chunk_end] = chunk_cumsum
                else:
                    chunk_data = seq_data[chunk_start:chunk_end, :]  # (chunk_len, H)
                    chunk_cumsum = torch.cumsum(chunk_data, dim=0, dtype=torch.float32)
                    if reverse:
                        chunk_cumsum = torch.flip(torch.cumsum(torch.flip(chunk_data, dims=[0]), dim=0, dtype=torch.float32), dims=[0])
                    if scale is not None:
                        chunk_cumsum = chunk_cumsum * scale
                    result[0, start + chunk_start:start + chunk_end, :] = chunk_cumsum
        
        return result
    else:
        result = torch.zeros_like(g, dtype=g.dtype)
        
        for b in range(B):
            if head_first:
                seq_data = g[b]  # (H, T)
            else:
                seq_data = g[b]  # (T, H)
            
            # Process in chunks
            num_chunks = (T + chunk_size - 1) // chunk_size
            for chunk_idx in range(num_chunks):
                chunk_start = chunk_idx * chunk_size
                chunk_end = min(chunk_start + chunk_size, T)
                
                if head_first:
                    chunk_data = seq_data[:, chunk_start:chunk_end]  # (H, chunk_len)
                    chunk_cumsum = torch.cumsum(chunk_data, dim=1, dtype=torch.float32)
                    if reverse:
                        chunk_cumsum = torch.flip(torch.cumsum(torch.flip(chunk_data, dims=[1]), dim=1, dtype=torch.float32), dims=[1])
                    if scale is not None:
                        chunk_cumsum = chunk_cumsum * scale
                    result[b, :, chunk_start:chunk_end] = chunk_cumsum
                else:
                    chunk_data = seq_data[chunk_start:chunk_end, :]  # (chunk_len, H)
                    chunk_cumsum = torch.cumsum(chunk_data, dim=0, dtype=torch.float32)
                    if reverse:
                        chunk_cumsum = torch.flip(torch.cumsum(torch.flip(chunk_data, dims=[0]), dim=0, dtype=torch.float32), dims=[0])
                    if scale is not None:
                        chunk_cumsum = chunk_cumsum * scale
                    result[b, chunk_start:chunk_end, :] = chunk_cumsum
        
        return result


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128, 256])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk_local_cumsum_basic(B, T, H, chunk_size, head_first, reverse, dtype):
    """Test basic cumsum without scale and cu_seqlens."""
    torch.manual_seed(42)
    
    if head_first:
        g = torch.randn(B, H, T, dtype=dtype, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=dtype, device="npu")
    
    # Triton implementation
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    # Reference implementation
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=reverse,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    # Compare results
    assert output.shape == expected.shape, f"Shape mismatch: {output.shape} vs {expected.shape}"
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch with max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_chunk_local_cumsum_with_scale(B, T, H, chunk_size, head_first, scale):
    torch.manual_seed(42)
    
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=torch.float16, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=False,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=False,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch with scale={scale}, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("seq_lens", [[32, 64, 48], [64, 128], [100]])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_varlen(seq_lens, H, chunk_size, head_first):
    torch.manual_seed(42)
    
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.long, device="npu")
    total_len = int(cu_seqlens[-1].item())
    
    if head_first:
        g = torch.randn(1, H, total_len, dtype=torch.float16, device="npu")
    else:
        g = torch.randn(1, total_len, H, dtype=torch.float16, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=cu_seqlens.cpu(),
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch for varlen, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_output_dtype(output_dtype, head_first):
    torch.manual_seed(42)
    B, T, H, chunk_size = 2, 128, 16, 32
    
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=torch.float16, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=output_dtype,
    )
    
    assert output.dtype == g.dtype, \
        f"Operator outputs in input dtype {g.dtype}, not requested output_dtype {output_dtype}"
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch for dtype={output_dtype}, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_chunk_local_cumsum_scale_and_reverse(head_first, reverse, scale):
    torch.manual_seed(42)
    B, T, H, chunk_size = 2, 128, 16, 32
    
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=torch.float16, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch for scale={scale}, reverse={reverse}, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("seq_lens", [[32, 64], [64, 128]])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [None, 0.5])
def test_chunk_local_cumsum_varlen_combined(seq_lens, head_first, reverse, scale):
    torch.manual_seed(42)
    H, chunk_size = 16, 32
    
    cu_seqlens = torch.tensor([0] + list(torch.tensor(seq_lens).cumsum(0).tolist()), dtype=torch.long, device="npu")
    total_len = int(cu_seqlens[-1].item())
    
    if head_first:
        g = torch.randn(1, H, total_len, dtype=torch.float16, device="npu")
    else:
        g = torch.randn(1, total_len, H, dtype=torch.float16, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=reverse,
        scale=scale,
        cu_seqlens=cu_seqlens.cpu(),
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch for varlen combined, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_T_equals_chunk_size(head_first):
    torch.manual_seed(42)
    B, T, H, chunk_size = 2, 64, 16, 64
    
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=torch.float32, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch when T==chunk_size, max diff: {(output.cpu() - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_T_multiple_of_chunk_size(head_first):
    torch.manual_seed(42)
    B, T, H, chunk_size = 1, 256, 32, 64
    
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float32, device="npu")
    else:
        g = torch.randn(B, T, H, dtype=torch.float32, device="npu")
    
    output = chunk_local_cumsum(
        g=g,
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
        output_dtype=torch.float32,
    )
    
    expected = torch_chunk_local_cumsum(
        g=g.cpu(),
        chunk_size=chunk_size,
        reverse=False,
        scale=None,
        cu_seqlens=None,
        head_first=head_first,
    )
    
    assert torch.allclose(output.cpu(), expected, rtol=1e-3, atol=1e-3), \
        f"Output mismatch when T is multiple of chunk_size, max diff: {(output.cpu() - expected).abs().max()}"


def test_chunk_local_cumsum_invalid_chunk_size():
    g = torch.randn(1, 64, 16, dtype=torch.float32, device="npu")
    with pytest.raises(AssertionError):
        chunk_local_cumsum(g=g, chunk_size=15)


def test_chunk_local_cumsum_invalid_shape():
    g = torch.randn(1, 64, dtype=torch.float32, device="npu")
    with pytest.raises(ValueError, match="Unsupported input shape"):
        chunk_local_cumsum(g=g, chunk_size=16)
