"""Factor-model Cholesky head.

Reparameterizes the predicted covariance with a structural prior:

    Sigma = B F B^T + diag(d^2)

where:
  - B (N, k)   : per-asset factor loadings (predicted from each asset's embedding)
  - F (k, k)   : factor covariance, PSD, predicted from the [CTX] token embedding
  - d (N,)     : idiosyncratic vol per asset (predicted from asset embedding)

Cuts the head's free degrees of freedom from N(N+1)/2 (~722 for N=38) to
N*k + k*(k+1)/2 + N (~200 for N=38, k=4). This is the canonical decomposition
of risk in modern portfolio theory (Fama-French factor models, etc.) and gives
the network a much stronger inductive bias — instead of having to learn a
full N(N+1)/2 entry covariance from scratch each fold, the model only needs to
learn:
  1. Each asset's loadings on a small number of latent factors
  2. The (regime-dependent) factor covariance
  3. Per-asset idiosyncratic vol

This should help with the data-scarcity problem observed in v5 (38 assets,
~500 training snapshots per fold).

The head returns the lower-triangular Cholesky factor L of Sigma for
compatibility with the existing downstream NLL / Sigma utilities. Cholesky is
re-derived per-snapshot via `torch.linalg.cholesky` which is differentiable.
"""
from __future__ import annotations

import torch
from torch import Tensor, nn


class FactorCholeskyHead(nn.Module):
    """Rank-k factor structure + diagonal idiosyncratic head."""

    def __init__(
        self,
        n_assets: int,
        d_model: int,
        n_factors: int = 4,
        min_diag: float = 0.01,
        jitter: float = 1e-6,
    ):
        super().__init__()
        self.n = n_assets
        self.k = n_factors
        self.min_diag = min_diag
        self.jitter = jitter

        # Per-asset heads (operate on asset embeddings (B, N, d_model))
        self.B_head = nn.Linear(d_model, n_factors)         # loadings (N x k)
        self.d_head = nn.Linear(d_model, 1)                 # log-idio-vol scalar

        # Context head — uses the [CTX] token's embedding to predict the
        # regime-dependent factor covariance. The output is the lower-triangular
        # entries of a k x k Cholesky factor.
        n_flat = n_factors * (n_factors + 1) // 2
        self.F_head = nn.Linear(d_model, n_flat)

        # Buffer of (row, col) indices for the lower-triangular k x k matrix.
        rows, cols = torch.tril_indices(n_factors, n_factors).unbind(0)
        self.register_buffer("f_tril_rows", rows)
        self.register_buffer("f_tril_cols", cols)

    def forward(self, asset_emb: Tensor, ctx_emb: Tensor) -> Tensor:
        """asset_emb: (B, N, d_model)
        ctx_emb:   (B, d_model)        (embedding of the [CTX] token output)
        Returns L: (B, N, N) lower-triangular Cholesky of Sigma = B F B^T + D.
        """
        Bsize = asset_emb.size(0)

        # Loadings
        B = self.B_head(asset_emb)                                    # (Bsize, N, k)

        # Idiosyncratic vols (softplus shift -> ~0.018 init, plus floor)
        log_d_raw = self.d_head(asset_emb).squeeze(-1)                # (Bsize, N)
        d = torch.nn.functional.softplus(log_d_raw - 4.0) + self.min_diag
        D_diag = d ** 2                                                # variance

        # Factor covariance — predict the (k*(k+1)/2) lower-tri entries from CTX
        f_raw = self.F_head(ctx_emb)                                  # (Bsize, n_flat)
        F_chol = asset_emb.new_zeros(Bsize, self.k, self.k)
        F_chol[:, self.f_tril_rows, self.f_tril_cols] = f_raw
        # Force positive diagonal via softplus on the diagonal entries
        f_diag = torch.diagonal(F_chol, dim1=-2, dim2=-1)
        f_diag_pos = torch.nn.functional.softplus(f_diag) + self.min_diag
        F_chol = F_chol - torch.diag_embed(f_diag) + torch.diag_embed(f_diag_pos)
        F_cov = F_chol @ F_chol.transpose(-1, -2)                     # (Bsize, k, k)

        # Assemble Sigma = B F B^T + diag(D)
        BFB = B @ F_cov @ B.transpose(-1, -2)                         # (Bsize, N, N)
        Sigma = BFB + torch.diag_embed(D_diag)
        # Jitter for numerical stability of the Cholesky.
        eye = torch.eye(self.n, device=Sigma.device, dtype=Sigma.dtype).expand(Bsize, self.n, self.n)
        Sigma = Sigma + self.jitter * eye

        # Cholesky of the structured Sigma. PyTorch's torch.linalg.cholesky is
        # differentiable. If it ever fails (shouldn't, given the floor + jitter),
        # we add more jitter and retry once.
        try:
            L = torch.linalg.cholesky(Sigma)
        except RuntimeError:
            L = torch.linalg.cholesky(Sigma + 1e-3 * eye)
        return L
