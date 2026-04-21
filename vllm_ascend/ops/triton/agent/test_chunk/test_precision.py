"""
Precision test script for chunk_gated_delta_rule_fwd (Ascend NPU Triton kernel).

Covers the user-specified scenario:
    q:             torch.Size([1, 5, 16, 128])   # [B, T, H, K]
    k:             torch.Size([1, 5, 16, 128])   # [B, T, H, K]
    v:             torch.Size([1, 5, 16, 128])   # [B, T, H, V]
    g:             torch.Size([1, 5, 16])         # [B, T, H]
    beta:          torch.Size([1, 5, 16])         # [B, T, H]
    initial_state: torch.Size([1, 16, 128, 128])  # [N, H, K, V]
    cu_seqlens:    torch.Size([2])                 # [N+1], values=[0, 5]

Run:
    pytest test_precision_chunk_gated_delta_rule.py -v
    # or directly:
    python test_precision_chunk_gated_delta_rule.py
"""
import sys
sys.path.insert(0, '/vllm-workspace/vllm-ascend')

import pytest
import torch
import torch_npu
import torch.nn.functional as F

from vllm_ascend.ops.triton.fla.chunk import chunk_gated_delta_rule_fwd

DEVICE = 'npu'

# ---- Shape constants from user specification ----
B, T, H, K, V = 1, 5, 16, 128, 128
CU_SEQLENS_LIST = [0, 5]  # single sequence of length 5


# ---------------------------------------------------------------------------
# Naive reference implementation (float64 on CPU for maximum precision)
# ---------------------------------------------------------------------------
def naive_gated_delta_rule_fwd(q, k, v, g, beta, scale, initial_state,
                                cu_seqlens=None):
    """
    Recurrent gated delta rule, token-by-token in float64 on CPU.

    Recurrence per head:
        S_t = exp(g_t) * S_{t-1} + beta_t * k_t ⊗ (v_t - S_{t-1}^T @ k_t)
        o_t = S_t^T @ q_t * scale
    """
    q  = q.cpu().to(torch.float64)
    k  = k.cpu().to(torch.float64)
    v  = v.cpu().to(torch.float64)
    g  = g.cpu().to(torch.float64)
    beta = beta.cpu().to(torch.float64)
    initial_state = initial_state.cpu().to(torch.float64)

    B_loc, T_total, H_loc, K_loc = q.shape
    V_loc = v.shape[-1]
    o = torch.zeros(B_loc, T_total, H_loc, V_loc, dtype=torch.float64)
    final_state = initial_state.clone()

    if cu_seqlens is not None:
        cu = cu_seqlens.cpu().tolist()
        N = len(cu) - 1
        for n in range(N):
            start, end = int(cu[n]), int(cu[n + 1])
            h = initial_state[n].clone()           # [H, K, V]
            for t in range(start, end):
                # Gating
                h = h * torch.exp(g[0, t])[:, None, None]
                k_t = k[0, t]                       # [H, K]
                v_t = v[0, t]                       # [H, V]
                beta_t = beta[0, t]                 # [H]
                # Delta correction: retrieve current estimate for k_t
                correction = torch.einsum('hkv,hk->hv', h, k_t)
                delta = v_t - correction
                # State update
                h = h + beta_t[:, None, None] * torch.einsum(
                    'hk,hv->hkv', k_t, delta)
                # Output
                o[0, t] = torch.einsum('hkv,hk->hv', h, q[0, t]) * scale
            final_state[n] = h
    else:
        for b_idx in range(B_loc):
            h = initial_state[b_idx].clone()
            for t in range(T_total):
                h = h * torch.exp(g[b_idx, t])[:, None, None]
                k_t = k[b_idx, t]
                v_t = v[b_idx, t]
                beta_t = beta[b_idx, t]
                correction = torch.einsum('hkv,hk->hv', h, k_t)
                delta = v_t - correction
                h = h + beta_t[:, None, None] * torch.einsum(
                    'hk,hv->hkv', k_t, delta)
                o[b_idx, t] = torch.einsum('hkv,hk->hv', h, q[b_idx, t]) * scale
            final_state[b_idx] = h

    return o, final_state


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def make_inputs(dtype=torch.bfloat16, zero_state=False, seed=42,
                state_scale=0.01):
    """
    Generate inputs matching the user-specified shapes exactly.
    Returns: q, k, v, g, beta, scale, initial_state, cu_seqlens
    """
    torch.manual_seed(seed)
    q = torch.randn(B, T, H, K, dtype=dtype, device=DEVICE)
    k = F.normalize(
        torch.randn(B, T, H, K, dtype=dtype, device=DEVICE), p=2, dim=-1)
    v = torch.randn(B, T, H, V, dtype=dtype, device=DEVICE)
    g = F.logsigmoid(torch.randn(B, T, H, dtype=dtype, device=DEVICE))
    beta = torch.rand(B, T, H, dtype=dtype, device=DEVICE).sigmoid()
    scale = K ** -0.5

    cu_seqlens = torch.tensor(CU_SEQLENS_LIST, dtype=torch.long, device=DEVICE)

    N = len(CU_SEQLENS_LIST) - 1  # number of sequences = 1
    if zero_state:
        initial_state = torch.zeros(N, H, K, V, dtype=dtype, device=DEVICE)
    else:
        initial_state = torch.randn(N, H, K, V, dtype=dtype,
                                    device=DEVICE) * state_scale
    return q, k, v, g, beta, scale, initial_state, cu_seqlens


def run_triton(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    """Run the Triton chunk kernel and return (output, final_state)."""
    torch.npu.synchronize()
    result = chunk_gated_delta_rule_fwd(
        q=q, k=k, v=v,
        g=g.clone(),   # clone because cumsum modifies g in-place
        beta=beta,
        scale=scale,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu_seqlens,
    )
    torch.npu.synchronize()
    # result = (g_cumsum, o, A, final_state, w|None, h|None, v_new|None)
    return result[1], result[3]


def compute_metrics(ref, act, label=""):
    """Compute and return a dict of precision metrics."""
    ref_f = ref.flatten().to(torch.float64)
    act_f = act.flatten().to(torch.float64)

    diff = (ref_f - act_f).abs()
    max_abs_err = diff.max().item()
    mean_abs_err = diff.mean().item()

    # Relative error (avoid division by zero)
    denom = ref_f.abs().clamp(min=1e-12)
    max_rel_err = (diff / denom).max().item()
    mean_rel_err = (diff / denom).mean().item()

    # Cosine similarity
    cos_sim = F.cosine_similarity(
        ref_f.unsqueeze(0).float(),
        act_f.unsqueeze(0).float()
    ).item()

    # Signal-to-noise ratio (dB)
    signal_power = (ref_f ** 2).mean().item()
    noise_power = (diff ** 2).mean().item()
    snr_db = 10 * torch.log10(
        torch.tensor(signal_power / max(noise_power, 1e-30))
    ).item()

    return {
        "label": label,
        "max_abs_err": max_abs_err,
        "mean_abs_err": mean_abs_err,
        "max_rel_err": max_rel_err,
        "mean_rel_err": mean_rel_err,
        "cosine_sim": cos_sim,
        "snr_db": snr_db,
    }


def print_metrics(metrics):
    """Pretty-print precision metrics."""
    label = metrics["label"]
    print(f"\n{'='*60}")
    print(f"  Precision Report: {label}")
    print(f"{'='*60}")
    print(f"  Max  Absolute Error : {metrics['max_abs_err']:.6e}")
    print(f"  Mean Absolute Error : {metrics['mean_abs_err']:.6e}")
    print(f"  Max  Relative Error : {metrics['max_rel_err']:.6e}")
    print(f"  Mean Relative Error : {metrics['mean_rel_err']:.6e}")
    print(f"  Cosine Similarity   : {metrics['cosine_sim']:.10f}")
    print(f"  SNR (dB)            : {metrics['snr_db']:.2f}")
    print(f"{'='*60}")


def assert_precision(metrics, atol, cos_thresh, rtol=None):
    """Assert precision metrics are within acceptable thresholds."""
    label = metrics["label"]
    assert metrics["max_abs_err"] < atol, (
        f"FAIL [{label}] max_abs_err={metrics['max_abs_err']:.4e} > atol={atol}")
    assert metrics["cosine_sim"] > cos_thresh, (
        f"FAIL [{label}] cosine_sim={metrics['cosine_sim']:.8f} < thresh={cos_thresh}")
    if rtol is not None:
        assert metrics["mean_rel_err"] < rtol, (
            f"FAIL [{label}] mean_rel_err={metrics['mean_rel_err']:.4e} > rtol={rtol}")


# ---------------------------------------------------------------------------
# Tolerance constants
# ---------------------------------------------------------------------------
ATOL_BF16 = 5e-2      # bfloat16: 7-bit mantissa
ATOL_FP16 = 5e-2      # float16:  10-bit mantissa (tighter possible but kernel uses fp32 accum)
COS_THRESH = 0.99
RTOL = 1.0             # mean relative error bound


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(autouse=True)
def sync_npu():
    """Synchronize NPU before and after each test."""
    torch.npu.synchronize()
    yield
    torch.npu.synchronize()


# ---------------------------------------------------------------------------
# Shape validation test
# ---------------------------------------------------------------------------
class TestInputShapes:
    """Verify inputs match the user-specified shapes exactly."""

    def test_shape_q(self):
        q, *_ = make_inputs()
        assert q.shape == torch.Size([1, 5, 16, 128]), f"q shape: {q.shape}"

    def test_shape_k(self):
        _, k, *_ = make_inputs()
        assert k.shape == torch.Size([1, 5, 16, 128]), f"k shape: {k.shape}"

    def test_shape_v(self):
        _, _, v, *_ = make_inputs()
        assert v.shape == torch.Size([1, 5, 16, 128]), f"v shape: {v.shape}"

    def test_shape_g(self):
        _, _, _, g, *_ = make_inputs()
        assert g.shape == torch.Size([1, 5, 16]), f"g shape: {g.shape}"

    def test_shape_beta(self):
        _, _, _, _, beta, *_ = make_inputs()
        assert beta.shape == torch.Size([1, 5, 16]), f"beta shape: {beta.shape}"

    def test_shape_initial_state(self):
        *_, initial_state, _ = make_inputs()
        assert initial_state.shape == torch.Size([1, 16, 128, 128]), \
            f"initial_state shape: {initial_state.shape}"

    def test_shape_cu_seqlens(self):
        *_, cu_seqlens = make_inputs()
        assert cu_seqlens.shape == torch.Size([2]), \
            f"cu_seqlens shape: {cu_seqlens.shape}"


# ---------------------------------------------------------------------------
# Core precision tests
# ---------------------------------------------------------------------------
class TestPrecisionChunkGatedDeltaRule:
    """
    Precision tests for chunk_gated_delta_rule_fwd against naive reference.
    All tests use the user-specified shapes:
        q/k/v: [1,5,16,128], g/beta: [1,5,16],
        initial_state: [1,16,128,128], cu_seqlens: [2]
    """

    # ---- Scenario 1: bfloat16, varlen, random initial state ----
    def test_bf16_varlen_random_state(self):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.bfloat16, zero_state=False, seed=42)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), "bf16_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), "bf16_state")
        print_metrics(m_o)
        print_metrics(m_s)

        assert_precision(m_o, ATOL_BF16, COS_THRESH)
        assert_precision(m_s, ATOL_BF16, COS_THRESH)

    # ---- Scenario 2: bfloat16, varlen, zero initial state ----
    def test_bf16_varlen_zero_state(self):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.bfloat16, zero_state=True, seed=123)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), "bf16_zero_state_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), "bf16_zero_state_state")
        print_metrics(m_o)
        print_metrics(m_s)

        assert_precision(m_o, ATOL_BF16, COS_THRESH)
        assert_precision(m_s, ATOL_BF16, COS_THRESH)

    # ---- Scenario 3: float16, varlen, random initial state ----
    def test_fp16_varlen_random_state(self):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.float16, zero_state=False, seed=42)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), "fp16_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), "fp16_state")
        print_metrics(m_o)
        print_metrics(m_s)

        assert_precision(m_o, ATOL_FP16, COS_THRESH)
        assert_precision(m_s, ATOL_FP16, COS_THRESH)

    # ---- Scenario 4: float16, varlen, zero initial state ----
    def test_fp16_varlen_zero_state(self):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.float16, zero_state=True, seed=999)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), "fp16_zero_state_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), "fp16_zero_state_state")
        print_metrics(m_o)
        print_metrics(m_s)

        assert_precision(m_o, ATOL_FP16, COS_THRESH)
        assert_precision(m_s, ATOL_FP16, COS_THRESH)

    # ---- Scenario 5: bfloat16, larger initial state magnitude ----
    def test_bf16_large_initial_state(self):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.bfloat16, zero_state=False, seed=77,
            state_scale=0.1)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), "bf16_large_state_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), "bf16_large_state_state")
        print_metrics(m_o)
        print_metrics(m_s)

        assert_precision(m_o, ATOL_BF16, COS_THRESH)
        assert_precision(m_s, ATOL_BF16, COS_THRESH)

    # ---- Scenario 6: reproducibility (same seed => same output) ----
    def test_reproducibility(self):
        def _run(seed=42):
            q, k, v, g, beta, scale, state, cu = make_inputs(seed=seed)
            return run_triton(q, k, v, g, beta, scale, state, cu)

        o1, s1 = _run()
        o2, s2 = _run()

        torch.testing.assert_close(o1, o2, atol=1e-3, rtol=1e-2)
        torch.testing.assert_close(s1, s2, atol=1e-3, rtol=1e-2)

    # ---- Scenario 7: multiple random seeds (statistical robustness) ----
    @pytest.mark.parametrize("seed", [1, 42, 100, 256, 999])
    def test_bf16_multiple_seeds(self, seed):
        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=torch.bfloat16, seed=seed)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), f"bf16_seed{seed}_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), f"bf16_seed{seed}_state")

        assert_precision(m_o, ATOL_BF16, COS_THRESH)
        assert_precision(m_s, ATOL_BF16, COS_THRESH)


# ---------------------------------------------------------------------------
# Standalone runner with detailed report
# ---------------------------------------------------------------------------
def main():
    """Run all precision checks and print a detailed report."""
    print("=" * 70)
    print("  chunk_gated_delta_rule_fwd Precision Test")
    print("  Shapes: q/k/v=[1,5,16,128], g/beta=[1,5,16],")
    print("          initial_state=[1,16,128,128], cu_seqlens=[2]")
    print("=" * 70)

    all_passed = True
    results = []

    scenarios = [
        ("bf16_random_state",  torch.bfloat16, False, 42,   0.01, ATOL_BF16),
        ("bf16_zero_state",    torch.bfloat16, True,  123,  0.01, ATOL_BF16),
        ("fp16_random_state",  torch.float16,  False, 42,   0.01, ATOL_FP16),
        ("fp16_zero_state",    torch.float16,  True,  999,  0.01, ATOL_FP16),
        ("bf16_large_state",   torch.bfloat16, False, 77,   0.1,  ATOL_BF16),
    ]

    for name, dtype, zero_state, seed, state_scale, atol in scenarios:
        print(f"\n>>> Scenario: {name} (dtype={dtype}, seed={seed})")

        q, k, v, g, beta, scale, state, cu = make_inputs(
            dtype=dtype, zero_state=zero_state, seed=seed,
            state_scale=state_scale)

        ref_o, ref_s = naive_gated_delta_rule_fwd(
            q, k, v, g, beta, scale, state, cu)
        tri_o, tri_s = run_triton(q, k, v, g, beta, scale, state, cu)

        m_o = compute_metrics(ref_o.cpu(), tri_o.cpu(), f"{name}_output")
        m_s = compute_metrics(ref_s.cpu(), tri_s.cpu(), f"{name}_state")
        print_metrics(m_o)
        print_metrics(m_s)

        passed_o = (m_o["max_abs_err"] < atol and m_o["cosine_sim"] > COS_THRESH)
        passed_s = (m_s["max_abs_err"] < atol and m_s["cosine_sim"] > COS_THRESH)

        status_o = "PASS" if passed_o else "FAIL"
        status_s = "PASS" if passed_s else "FAIL"
        print(f"  Output: [{status_o}]  State: [{status_s}]")

        if not (passed_o and passed_s):
            all_passed = False

        results.append((name, m_o, m_s, passed_o and passed_s))

    # Summary table
    print(f"\n{'='*70}")
    print("  SUMMARY")
    print(f"{'='*70}")
    print(f"  {'Scenario':<25} {'Output MaxErr':<16} {'State MaxErr':<16} {'CosSim(O)':<12} {'Result'}")
    print(f"  {'-'*25} {'-'*16} {'-'*16} {'-'*12} {'-'*6}")
    for name, m_o, m_s, passed in results:
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<25} {m_o['max_abs_err']:<16.4e} {m_s['max_abs_err']:<16.4e} {m_o['cosine_sim']:<12.8f} {status}")

    print(f"\n  Overall: {'ALL PASSED' if all_passed else 'SOME FAILED'}")
    print(f"{'='*70}")
    return 0 if all_passed else 1


if __name__ == "__main__":
    import sys as _sys
    _sys.exit(main())
