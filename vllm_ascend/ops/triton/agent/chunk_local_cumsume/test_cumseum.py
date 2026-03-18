# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from unittest.mock import patch


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n |= n >> 32
    return n + 1


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


def _pytorch_kernel_impl(
    s, o, scale, cu_seqlens, chunk_indices,
    T, B, H, BLOCK_T, CHUNK_SIZE, HEAD_FIRST, REVERSE, **kwargs
):
    """Pure PyTorch reimplementation of chunk_local_cumsum_scalar_kernel."""
    if cu_seqlens is not None:
        num_blocks = len(chunk_indices)
        for blk in range(num_blocks):
            i_s = int(chunk_indices[blk, 0].item())
            i_block = int(chunk_indices[blk, 1].item())
            bos = int(cu_seqlens[i_s].item())
            eos = int(cu_seqlens[i_s + 1].item())
            T_seq = eos - bos

            for i_b in range(B):
                start_t = i_block * BLOCK_T
                end_t = min(start_t + BLOCK_T, T_seq)
                if start_t >= T_seq:
                    continue
                actual_len = end_t - start_t
                N_CHUNKS = BLOCK_T // CHUNK_SIZE

                if HEAD_FIRST:
                    block_data = s[i_b, :, bos + start_t:bos + end_t].float()
                    pad_len = N_CHUNKS * CHUNK_SIZE - actual_len
                    if pad_len > 0:
                        block_data = torch.cat(
                            [block_data, torch.zeros(H, pad_len)], dim=1
                        )
                    block_data = block_data.reshape(H, N_CHUNKS, CHUNK_SIZE)
                    block_data = block_data.permute(2, 0, 1)
                    if REVERSE:
                        block_data = torch.flip(block_data, dims=[0])
                        cs = torch.cumsum(block_data, dim=0)
                        cs = torch.flip(cs, dims=[0])
                    else:
                        cs = torch.cumsum(block_data, dim=0)
                    if scale is not None:
                        cs = cs * scale
                    cs = cs.permute(1, 2, 0).reshape(H, N_CHUNKS * CHUNK_SIZE)
                    o[i_b, :, bos + start_t:bos + end_t] = cs[:, :actual_len].to(o.dtype)
                else:
                    block_data = s[i_b, bos + start_t:bos + end_t, :].float()
                    pad_len = N_CHUNKS * CHUNK_SIZE - actual_len
                    if pad_len > 0:
                        block_data = torch.cat(
                            [block_data, torch.zeros(pad_len, H)], dim=0
                        )
                    block_data = block_data.reshape(N_CHUNKS, CHUNK_SIZE, H)
                    block_data = block_data.permute(1, 0, 2)
                    if REVERSE:
                        block_data = torch.flip(block_data, dims=[0])
                        cs = torch.cumsum(block_data, dim=0)
                        cs = torch.flip(cs, dims=[0])
                    else:
                        cs = torch.cumsum(block_data, dim=0)
                    if scale is not None:
                        cs = cs * scale
                    cs = cs.permute(1, 0, 2).reshape(N_CHUNKS * CHUNK_SIZE, H)
                    o[i_b, bos + start_t:bos + end_t, :] = cs[:actual_len, :].to(o.dtype)
    else:
        num_blocks = _cdiv(T, BLOCK_T)
        N_CHUNKS = BLOCK_T // CHUNK_SIZE
        for i_b in range(B):
            for block_idx in range(num_blocks):
                start_t = block_idx * BLOCK_T
                end_t = min(start_t + BLOCK_T, T)
                if start_t >= T:
                    continue
                actual_len = end_t - start_t

                if HEAD_FIRST:
                    block_data = s[i_b, :, start_t:end_t].float()
                    pad_len = N_CHUNKS * CHUNK_SIZE - actual_len
                    if pad_len > 0:
                        block_data = torch.cat(
                            [block_data, torch.zeros(H, pad_len)], dim=1
                        )
                    block_data = block_data.reshape(H, N_CHUNKS, CHUNK_SIZE)
                    block_data = block_data.permute(2, 0, 1)
                    if REVERSE:
                        block_data = torch.flip(block_data, dims=[0])
                        cs = torch.cumsum(block_data, dim=0)
                        cs = torch.flip(cs, dims=[0])
                    else:
                        cs = torch.cumsum(block_data, dim=0)
                    if scale is not None:
                        cs = cs * scale
                    cs = cs.permute(1, 2, 0).reshape(H, N_CHUNKS * CHUNK_SIZE)
                    o[i_b, :, start_t:end_t] = cs[:, :actual_len].to(o.dtype)
                else:
                    block_data = s[i_b, start_t:end_t, :].float()
                    pad_len = N_CHUNKS * CHUNK_SIZE - actual_len
                    if pad_len > 0:
                        block_data = torch.cat(
                            [block_data, torch.zeros(pad_len, H)], dim=0
                        )
                    block_data = block_data.reshape(N_CHUNKS, CHUNK_SIZE, H)
                    block_data = block_data.permute(1, 0, 2)
                    if REVERSE:
                        block_data = torch.flip(block_data, dims=[0])
                        cs = torch.cumsum(block_data, dim=0)
                        cs = torch.flip(cs, dims=[0])
                    else:
                        cs = torch.cumsum(block_data, dim=0)
                    if scale is not None:
                        cs = cs * scale
                    cs = cs.permute(1, 0, 2).reshape(N_CHUNKS * CHUNK_SIZE, H)
                    o[i_b, start_t:end_t, :] = cs[:actual_len, :].to(o.dtype)


class _MockKernel:
    """Mock that supports grid[grid](...) launch syntax."""
    def __getitem__(self, grid):
        def launcher(**kwargs):
            return _pytorch_kernel_impl(**kwargs)
        return launcher


@pytest.fixture(autouse=True)
def _patch_triton():
    """Patch triton utilities and kernel for CPU-based testing."""
    import vllm.triton_utils as _tu
    import vllm_ascend.ops.triton.fla.cumsum as _cum
    import vllm_ascend.ops.triton.fla.utils as _utils

    old_kernel = _cum.chunk_local_cumsum_scalar_kernel
    old_np2 = getattr(_tu.triton, 'next_power_of_2', None)
    old_cdiv = getattr(_tu.triton, 'cdiv', None)

    _cum.chunk_local_cumsum_scalar_kernel = _MockKernel()
    _tu.triton.next_power_of_2 = _next_power_of_2
    _tu.triton.cdiv = _cdiv

    # Also patch triton ref used in utils.py (prepare_chunk_indices)
    from vllm.triton_utils import triton as _triton_ref
    _triton_ref.next_power_of_2 = _next_power_of_2
    _triton_ref.cdiv = _cdiv

    yield

    _cum.chunk_local_cumsum_scalar_kernel = old_kernel
    if old_np2 is not None:
        _tu.triton.next_power_of_2 = old_np2
    if old_cdiv is not None:
        _tu.triton.cdiv = old_cdiv


def _torch_reference_cumsum(g, chunk_size, reverse=False, scale=None,
                            cu_seqlens=None, head_first=False):
    """Simple PyTorch native reference: cumsum within each chunk."""
    if head_first:
        B, H, T = g.shape
    else:
        B, T, H = g.shape

    result = torch.zeros_like(g, dtype=torch.float32)

    if cu_seqlens is not None:
        for i in range(len(cu_seqlens) - 1):
            start = int(cu_seqlens[i].item())
            end = int(cu_seqlens[i + 1].item())
            seq_len = end - start
            num_chunks = _cdiv(seq_len, chunk_size)
            for ci in range(num_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, seq_len)
                if head_first:
                    cd = g[0, :, start + cs:start + ce].float()
                    if reverse:
                        cv = torch.flip(
                            torch.cumsum(torch.flip(cd, [1]), dim=1), [1]
                        )
                    else:
                        cv = torch.cumsum(cd, dim=1)
                    if scale is not None:
                        cv = cv * scale
                    result[0, :, start + cs:start + ce] = cv
                else:
                    cd = g[0, start + cs:start + ce, :].float()
                    if reverse:
                        cv = torch.flip(
                            torch.cumsum(torch.flip(cd, [0]), dim=0), [0]
                        )
                    else:
                        cv = torch.cumsum(cd, dim=0)
                    if scale is not None:
                        cv = cv * scale
                    result[0, start + cs:start + ce, :] = cv
    else:
        for b in range(B):
            num_chunks = _cdiv(T, chunk_size)
            for ci in range(num_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, T)
                if head_first:
                    cd = g[b, :, cs:ce].float()
                    if reverse:
                        cv = torch.flip(
                            torch.cumsum(torch.flip(cd, [1]), dim=1), [1]
                        )
                    else:
                        cv = torch.cumsum(cd, dim=1)
                    if scale is not None:
                        cv = cv * scale
                    result[b, :, cs:ce] = cv
                else:
                    cd = g[b, cs:ce, :].float()
                    if reverse:
                        cv = torch.flip(
                            torch.cumsum(torch.flip(cd, [0]), dim=0), [0]
                        )
                    else:
                        cv = torch.cumsum(cd, dim=0)
                    if scale is not None:
                        cv = cv * scale
                    result[b, cs:ce, :] = cv

    return result


# ---------------------------------------------------------------------------
# Test cases
# ---------------------------------------------------------------------------

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
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    if head_first:
        g = torch.randn(B, H, T, dtype=dtype)
    else:
        g = torch.randn(B, T, H, dtype=dtype)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=reverse,
        scale=None, cu_seqlens=None, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, reverse=reverse, head_first=head_first,
    )

    assert output.shape == expected.shape
    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_chunk_local_cumsum_with_scale(B, T, H, chunk_size, head_first, scale):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16)
    else:
        g = torch.randn(B, T, H, dtype=torch.float16)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=False,
        scale=scale, cu_seqlens=None, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, scale=scale, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("seq_lens", [[32, 64, 48], [64, 128], [100]])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_varlen(seq_lens, H, chunk_size, head_first):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.long,
    )
    total_len = int(cu_seqlens[-1].item())

    if head_first:
        g = torch.randn(1, H, total_len, dtype=torch.float16)
    else:
        g = torch.randn(1, total_len, H, dtype=torch.float16)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=False,
        scale=None, cu_seqlens=cu_seqlens, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, cu_seqlens=cu_seqlens, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("output_dtype", [torch.float16, torch.float32])
@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_output_dtype(output_dtype, head_first):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    B, T, H, chunk_size = 2, 128, 16, 32
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16)
    else:
        g = torch.randn(B, T, H, dtype=torch.float16)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=False,
        scale=None, cu_seqlens=None, head_first=head_first,
        output_dtype=output_dtype,
    )

    # The operator stores with o.dtype which is output_dtype or g.dtype
    expected = _torch_reference_cumsum(
        g, chunk_size, head_first=head_first,
    )

    assert torch.allclose(output.float(), expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output.float() - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_chunk_local_cumsum_scale_and_reverse(head_first, reverse, scale):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    B, T, H, chunk_size = 2, 128, 16, 32
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float16)
    else:
        g = torch.randn(B, T, H, dtype=torch.float16)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=reverse,
        scale=scale, cu_seqlens=None, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, reverse=reverse, scale=scale, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("seq_lens", [[32, 64], [64, 128]])
@pytest.mark.parametrize("head_first", [False, True])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [None, 0.5])
def test_chunk_local_cumsum_varlen_combined(seq_lens, head_first, reverse, scale):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    H, chunk_size = 16, 32
    cu_seqlens = torch.tensor(
        [0] + list(torch.tensor(seq_lens).cumsum(0).tolist()),
        dtype=torch.long,
    )
    total_len = int(cu_seqlens[-1].item())

    if head_first:
        g = torch.randn(1, H, total_len, dtype=torch.float16)
    else:
        g = torch.randn(1, total_len, H, dtype=torch.float16)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=reverse,
        scale=scale, cu_seqlens=cu_seqlens, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, reverse=reverse, scale=scale,
        cu_seqlens=cu_seqlens, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_T_equals_chunk_size(head_first):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    B, T, H, chunk_size = 2, 64, 16, 64
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float32)
    else:
        g = torch.randn(B, T, H, dtype=torch.float32)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=False,
        scale=None, cu_seqlens=None, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


@pytest.mark.parametrize("head_first", [False, True])
def test_chunk_local_cumsum_T_multiple_of_chunk_size(head_first):
    torch.manual_seed(42)
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

    B, T, H, chunk_size = 1, 256, 32, 64
    if head_first:
        g = torch.randn(B, H, T, dtype=torch.float32)
    else:
        g = torch.randn(B, T, H, dtype=torch.float32)

    output = chunk_local_cumsum(
        g=g, chunk_size=chunk_size, reverse=False,
        scale=None, cu_seqlens=None, head_first=head_first,
        output_dtype=torch.float32,
    )
    expected = _torch_reference_cumsum(
        g, chunk_size, head_first=head_first,
    )

    assert torch.allclose(output, expected, rtol=1e-3, atol=1e-3), \
        f"Max diff: {(output - expected).abs().max()}"


def test_chunk_local_cumsum_invalid_chunk_size():
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    g = torch.randn(1, 64, 16, dtype=torch.float32)
    with pytest.raises(AssertionError):
        chunk_local_cumsum(g=g, chunk_size=15)


def test_chunk_local_cumsum_invalid_shape():
    from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum
    g = torch.randn(1, 64, dtype=torch.float32)
    with pytest.raises(ValueError, match="Unsupported input shape"):
        chunk_local_cumsum(g=g, chunk_size=16)
