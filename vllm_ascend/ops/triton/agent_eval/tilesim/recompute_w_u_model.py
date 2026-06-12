import torch
import torch.nn as nn


# Shape config: B=1, T=512, H=16, Hg=8, K=128, V=128, BT=64, varlen=True
B, T, H, Hg, K, V, BT = 1, 512, 16, 8, 128, 128, 64
ng = H // Hg  # heads per kv-head group


class Model(nn.Module):
    def forward(self, k, v, beta, g_cumsum, A):
        """
        k:       [B, T, Hg, K]
        v:       [B, T, H,  V]
        beta:    [B, T, H]
        g_cumsum:[B, T, H]
        A:       [B, T, H, BT]  lower-triangular chunk factor

        u[b,t,h,:] = A[b,t,h,:L] @ (v[b,t,h,:] * beta[b,t,h])   per chunk
        w[b,t,h,:] = A[b,t,h,:L] @ (k[b,t,hg,:] * beta[b,t,h] * exp(g[b,t,h]))
        """
        k_f = k.float()
        v_f = v.float()
        beta_f = beta.float()
        g_f = g_cumsum.float()
        A_f = A.float()

        NT = (T + BT - 1) // BT
        w_out = torch.zeros(B, T, H, K, dtype=torch.float32, device=k.device)
        u_out = torch.zeros(B, T, H, V, dtype=torch.float32, device=v.device)

        for b in range(B):
            for i_t in range(NT):
                ts, te = i_t * BT, min(i_t * BT + BT, T)
                L = te - ts
                for h in range(H):
                    hg = h // ng
                    b_A = A_f[b, ts:te, h, :L]          # [L, L]
                    b_beta = beta_f[b, ts:te, h]         # [L]

                    b_vb = v_f[b, ts:te, h, :] * b_beta[:, None]
                    u_out[b, ts:te, h, :] = b_A @ b_vb

                    b_g = torch.exp(g_f[b, ts:te, h])
                    b_kb = k_f[b, ts:te, hg, :] * b_beta[:, None] * b_g[:, None]
                    w_out[b, ts:te, h, :] = b_A @ b_kb

        return w_out.to(v.dtype), u_out.to(v.dtype)


def get_inputs():
    torch.manual_seed(0)
    device = "cpu"
    dtype = torch.bfloat16
    k        = torch.randn(B, T, Hg, K, dtype=dtype, device=device)
    v        = torch.randn(B, T, H,  V, dtype=dtype, device=device)
    beta     = torch.randn(B, T, H,     dtype=dtype, device=device)
    g_cumsum = torch.randn(B, T, H,     dtype=dtype, device=device)
    A        = torch.randn(B, T, H, BT, dtype=dtype, device=device)
    return [k, v, beta, g_cumsum, A]
