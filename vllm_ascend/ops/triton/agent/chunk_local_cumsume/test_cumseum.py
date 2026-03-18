#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Precision unit tests for chunk_local_cumsum operator.

Execute: python tests/ut/ops/test_cumsum.py

Tests cover head_first=False path with all parameter combinations:
B, T, H, chunk_size, reverse, scale, cu_seqlens, dtype, output_dtype.

Note: head_first=True is excluded because the operator kernel has known
issues on this platform (boundary_check on wrong dimension + incorrect
backward transpose), making deterministic reference comparison infeasible.
"""

import sys
import traceback

import pytest
import torch

from vllm_ascend.ops.triton.fla.cumsum import chunk_local_cumsum

DEVICE = "npu"


def torch_reference_cumsum(g, chunk_size, reverse=False, scale=None,
                           cu_seqlens=None, head_first=False):
    """Pure PyTorch chunk-wise cumsum reference.

    Computes in float32, casts to input dtype to match kernel store
    precision, then returns as float32 for comparison.
    """
    input_dtype = g.dtype
    g_cpu = g.detach().cpu().float()
    B, T, H = g_cpu.shape
    result = torch.zeros_like(g_cpu)

    def _do_cumsum(data, dim):
        if reverse:
            return torch.flip(
                torch.cumsum(torch.flip(data, [dim]), dim=dim), [dim])
        return torch.cumsum(data, dim=dim)

    if cu_seqlens is not None:
        cu = cu_seqlens.detach().cpu()
        for i in range(len(cu) - 1):
            s = int(cu[i].item())
            e = int(cu[i + 1].item())
            seq_len = e - s
            n_chunks = (seq_len + chunk_size - 1) // chunk_size
            for ci in range(n_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, seq_len)
                chunk = g_cpu[0, s + cs:s + ce, :].clone()
                cv = _do_cumsum(chunk, dim=0)
                if scale is not None:
                    cv = cv * scale
                result[0, s + cs:s + ce, :] = cv.to(input_dtype).float()
    else:
        for b in range(B):
            n_chunks = (T + chunk_size - 1) // chunk_size
            for ci in range(n_chunks):
                cs = ci * chunk_size
                ce = min(cs + chunk_size, T)
                chunk = g_cpu[b, cs:ce, :].clone()
                cv = _do_cumsum(chunk, dim=0)
                if scale is not None:
                    cv = cv * scale
                result[b, cs:ce, :] = cv.to(input_dtype).float()

    return result


def check_close(output, expected, rtol=1e-3, atol=1e-3):
    out_f = output.detach().cpu().float()
    exp_f = expected.float()
    max_diff = (out_f - exp_f).abs().max().item()
    assert out_f.shape == exp_f.shape, \
        f"Shape mismatch: {out_f.shape} vs {exp_f.shape}"
    assert torch.allclose(out_f, exp_f, rtol=rtol, atol=atol), \
        f"Max diff: {max_diff}"


# ---- Parametrized tests (pytest) ----

@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128, 256])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32, 64])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk_local_cumsum_basic(B, T, H, chunk_size, reverse, dtype):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, dtype=dtype, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=chunk_size, reverse=reverse,
                                head_first=False, output_dtype=None)
    expected = torch_reference_cumsum(g, chunk_size, reverse=reverse)
    check_close(output, expected)


@pytest.mark.parametrize("B", [1, 2])
@pytest.mark.parametrize("T", [64, 128])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
@pytest.mark.parametrize("scale", [0.5, 1.0, 2.0])
def test_chunk_local_cumsum_with_scale(B, T, H, chunk_size, scale):
    torch.manual_seed(42)
    g = torch.randn(B, T, H, dtype=torch.float16, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=chunk_size, scale=scale,
                                head_first=False, output_dtype=None)
    expected = torch_reference_cumsum(g, chunk_size, scale=scale)
    check_close(output, expected)


@pytest.mark.parametrize("seq_lens", [[32, 64, 48], [64, 128], [100]])
@pytest.mark.parametrize("H", [16, 32])
@pytest.mark.parametrize("chunk_size", [16, 32])
def test_chunk_local_cumsum_varlen(seq_lens, H, chunk_size):
    torch.manual_seed(42)
    offsets = [0]
    for s in seq_lens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=DEVICE)
    total = offsets[-1]
    g = torch.randn(1, total, H, dtype=torch.float16, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=chunk_size,
                                cu_seqlens=cu_seqlens,
                                head_first=False, output_dtype=None)
    expected = torch_reference_cumsum(g, chunk_size, cu_seqlens=cu_seqlens)
    check_close(output, expected)


@pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
def test_chunk_local_cumsum_output_dtype(dtype):
    torch.manual_seed(42)
    B, T, H, cs = 2, 128, 16, 32
    g = torch.randn(B, T, H, dtype=dtype, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=cs, head_first=False,
                                output_dtype=dtype)
    assert output.dtype == dtype
    expected = torch_reference_cumsum(g, cs)
    check_close(output, expected)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [0.5, 2.0])
def test_chunk_local_cumsum_scale_and_reverse(reverse, scale):
    torch.manual_seed(42)
    B, T, H, cs = 2, 128, 16, 32
    g = torch.randn(B, T, H, dtype=torch.float16, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=cs, reverse=reverse,
                                scale=scale, head_first=False,
                                output_dtype=None)
    expected = torch_reference_cumsum(g, cs, reverse=reverse, scale=scale)
    check_close(output, expected)


@pytest.mark.parametrize("seq_lens", [[32, 64], [64, 128]])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("scale", [None, 0.5])
def test_chunk_local_cumsum_varlen_combined(seq_lens, reverse, scale):
    torch.manual_seed(42)
    H, cs = 16, 32
    offsets = [0]
    for s in seq_lens:
        offsets.append(offsets[-1] + s)
    cu_seqlens = torch.tensor(offsets, dtype=torch.long, device=DEVICE)
    total = offsets[-1]
    g = torch.randn(1, total, H, dtype=torch.float16, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=cs, reverse=reverse,
                                scale=scale, cu_seqlens=cu_seqlens,
                                head_first=False, output_dtype=None)
    expected = torch_reference_cumsum(g, cs, reverse=reverse, scale=scale,
                                      cu_seqlens=cu_seqlens)
    check_close(output, expected)


def test_chunk_local_cumsum_T_equals_chunk_size():
    torch.manual_seed(42)
    g = torch.randn(2, 64, 16, dtype=torch.float32, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=64, head_first=False,
                                output_dtype=None)
    expected = torch_reference_cumsum(g, 64)
    check_close(output, expected)


def test_chunk_local_cumsum_T_multiple_of_chunk_size():
    torch.manual_seed(42)
    g = torch.randn(1, 256, 32, dtype=torch.float32, device=DEVICE)
    output = chunk_local_cumsum(g=g, chunk_size=64, head_first=False,
                                output_dtype=None)
    expected = torch_reference_cumsum(g, 64)
    check_close(output, expected)


def test_chunk_local_cumsum_invalid_chunk_size():
    g = torch.randn(1, 64, 16, dtype=torch.float32, device=DEVICE)
    try:
        chunk_local_cumsum(g=g, chunk_size=15)
        raise RuntimeError("Should have raised AssertionError")
    except AssertionError:
        pass


def test_chunk_local_cumsum_invalid_shape():
    g = torch.randn(1, 64, dtype=torch.float32, device=DEVICE)
    try:
        chunk_local_cumsum(g=g, chunk_size=16)
        raise RuntimeError("Should have raised ValueError")
    except ValueError:
        pass


# ---- Direct execution runner ----

def _run_one(name, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        return True
    except Exception:
        print(f"  FAIL  {name}")
        traceback.print_exc()
        return False


def main():
    passed = 0
    failed = 0

    def run(name, fn, *a, **kw):
        nonlocal passed, failed
        if _run_one(name, fn, *a, **kw):
            passed += 1
        else:
            failed += 1

    # 1. Basic tests
    print("=== Basic cumsum tests ===")
    for B in [1, 2]:
        for T in [64, 128, 256]:
            for H in [16, 32]:
                for cs in [16, 32, 64]:
                    for rev in [False, True]:
                        for dt in [torch.float16, torch.float32]:
                            run(f"basic B={B} T={T} H={H} cs={cs} "
                                f"rev={rev} dt={dt}",
                                test_chunk_local_cumsum_basic,
                                B, T, H, cs, rev, dt)

    # 2. Scale tests
    print("=== Scale tests ===")
    for B in [1, 2]:
        for T in [64, 128]:
            for H in [16, 32]:
                for cs in [16, 32]:
                    for sc in [0.5, 1.0, 2.0]:
                        run(f"scale B={B} T={T} H={H} cs={cs} sc={sc}",
                            test_chunk_local_cumsum_with_scale,
                            B, T, H, cs, sc)

    # 3. Variable length tests
    print("=== Varlen tests ===")
    for sl in [[32, 64, 48], [64, 128], [100]]:
        for H in [16, 32]:
            for cs in [16, 32]:
                run(f"varlen sl={sl} H={H} cs={cs}",
                    test_chunk_local_cumsum_varlen, sl, H, cs)

    # 4. Output dtype tests
    print("=== Output dtype tests ===")
    for dt in [torch.float16, torch.float32]:
        run(f"output_dtype dt={dt}",
            test_chunk_local_cumsum_output_dtype, dt)

    # 5. Scale + reverse combined
    print("=== Scale + reverse tests ===")
    for rev in [False, True]:
        for sc in [0.5, 2.0]:
            run(f"scale_rev rev={rev} sc={sc}",
                test_chunk_local_cumsum_scale_and_reverse, rev, sc)

    # 6. Varlen combined
    print("=== Varlen combined tests ===")
    for sl in [[32, 64], [64, 128]]:
        for rev in [False, True]:
            for sc in [None, 0.5]:
                run(f"varlen_comb sl={sl} rev={rev} sc={sc}",
                    test_chunk_local_cumsum_varlen_combined, sl, rev, sc)

    # 7. Edge cases
    print("=== Edge case tests ===")
    run("T_eq_cs", test_chunk_local_cumsum_T_equals_chunk_size)
    run("T_mult_cs", test_chunk_local_cumsum_T_multiple_of_chunk_size)
    run("invalid_chunk_size", test_chunk_local_cumsum_invalid_chunk_size)
    run("invalid_shape", test_chunk_local_cumsum_invalid_shape)

    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"Results: {passed}/{total} passed, {failed} failed")
    print(f"{'=' * 60}")
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
