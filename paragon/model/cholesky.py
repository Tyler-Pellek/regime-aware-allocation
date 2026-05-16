"""Cholesky -> covariance utilities + Gaussian NLL loss.

Sigma = L L^T, with L from the transformer's head (lower-triangular, positive
diagonal enforced). Masked rows/cols (assets not tradable on date t) are
handled by zeroing the corresponding rows of L before forming Sigma and then
substituting an identity block on the masked indices so the matrix stays
strictly positive-definite (the loss restricts to the unmasked sub-block, so
this substitution has no effect on gradients of the active assets).
"""
from __future__ import annotations

import torch
from torch import Tensor


def sigma_from_L(L: Tensor, mask: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """Sigma = L L^T plus eps*I jitter for numerical PSD.

    L    : (B, N, N) lower-triangular
    mask : (B, N) bool — if given, rows of L corresponding to masked-out assets
           are zeroed and identity blocks are inserted on those diag entries.
    """
    B, N, _ = L.shape
    if mask is not None:
        m = mask.unsqueeze(-1).float()         # (B, N, 1)
        L = L * m                              # zero rows of inactive assets
    Sigma = L @ L.transpose(-1, -2)
    I = torch.eye(N, device=L.device, dtype=L.dtype).expand(B, N, N)
    Sigma = Sigma + eps * I
    if mask is not None:
        # Add a unit on the diagonal of inactive assets so Sigma stays PSD.
        inactive = (~mask).float().unsqueeze(-1) * torch.eye(N, device=L.device, dtype=L.dtype)
        Sigma = Sigma + inactive
    return Sigma


def gaussian_nll(
    mu: Tensor, L: Tensor, target: Tensor, mask: Tensor, eps: float = 1e-6
) -> Tensor:
    """Negative log-likelihood of `target` under N(mu, Sigma=L L^T), restricted
    to the active sub-block defined by `mask`.

    For a single sample with active set A of size k_A:
        NLL = 0.5 * (k_A * log 2pi + log det Sigma_A + r_A^T Sigma_A^{-1} r_A)

    All inputs are batched:
      mu     : (B, N)
      L      : (B, N, N) lower-triangular
      target : (B, N)
      mask   : (B, N) bool — True where active. Masked entries of target may be
               anything (they are ignored).

    Returns scalar mean NLL across the batch (per-element averaged over active
    dimension count to keep the magnitude stable as N varies).
    """
    B, N = mu.shape
    losses = []
    log2pi = torch.log(torch.tensor(2.0 * torch.pi, device=mu.device, dtype=mu.dtype))
    for b in range(B):
        m = mask[b]
        k = int(m.sum().item())
        if k == 0:
            continue
        idx = m.nonzero(as_tuple=True)[0]
        Lb = L[b][idx][:, idx]                   # (k, k) lower-tri
        # Build Sigma_A directly from sub-Cholesky to avoid PSD issues.
        # NOTE: the sub-block of a lower-triangular matrix is itself lower-triangular
        # *only* if the index set is a prefix; for an arbitrary subset of rows/cols
        # it generally is NOT. So we form Sigma_full first and then sub-select.
        Sigma_full_b = L[b] @ L[b].transpose(-1, -2)
        Sigma_A = Sigma_full_b[idx][:, idx]
        Sigma_A = Sigma_A + eps * torch.eye(k, device=mu.device, dtype=mu.dtype)
        r = (target[b] - mu[b])[idx]
        # Cholesky of Sigma_A (jittered) -> stable solve + logdet.
        # We avoid `torch.cholesky_solve` because MPS doesn't implement it;
        # instead we do two triangular solves explicitly (Lz=r, then L^T x=z).
        try:
            chol = torch.linalg.cholesky(Sigma_A)
        except RuntimeError:
            chol = torch.linalg.cholesky(
                Sigma_A + (1e-3) * torch.eye(k, device=mu.device, dtype=mu.dtype)
            )
        z = torch.linalg.solve_triangular(chol, r.unsqueeze(-1), upper=False)
        x = torch.linalg.solve_triangular(chol.transpose(-1, -2), z, upper=True)
        solve = x.squeeze(-1)
        quad = (r * solve).sum()
        logdet = 2.0 * torch.log(torch.diagonal(chol)).sum()
        nll_b = 0.5 * (k * log2pi + logdet + quad)
        losses.append(nll_b / max(k, 1))         # per-active-asset normalization
    if not losses:
        return torch.zeros((), device=mu.device, dtype=mu.dtype, requires_grad=True)
    return torch.stack(losses).mean()
