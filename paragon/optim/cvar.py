"""Convex CVaR optimizer with L1 turnover penalty.

Under a Gaussian assumption r ~ N(mu, Sigma), the closed form for portfolio
CVaR at level alpha is:

    CVaR_alpha(w) = -w^T mu + k_alpha * sqrt(w^T Sigma w)

where k_alpha = phi(z_alpha) / (1 - alpha) and z_alpha = Phi^{-1}(alpha).
This is convex in w (a positive multiple of an SOC norm minus a linear term),
so the full problem

    minimize    -w^T mu + k_alpha * sqrt(w^T Sigma w) + lambda * ||w - w_prev||_1
    subject to  1^T w = 1
                w_min <= w_i <= w_max
                (optionally) per-name and gross caps

is a small SOCP that solves in milliseconds with ECOS / Clarabel.

For a non-Gaussian variant (sample-based CVaR using Monte-Carlo scenarios from
the predicted distribution) see `cvar_scenario_based` below.
"""
from __future__ import annotations

from dataclasses import dataclass

import cvxpy as cp
import numpy as np
from scipy.stats import norm


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #

@dataclass
class CVaRConfig:
    alpha: float = 0.95            # CVaR level (e.g., 0.95 = expected loss in worst 5%)
    turnover_lambda: float = 5e-4  # L1 turnover penalty coefficient
    long_only: bool = True
    w_min: float = 0.0             # per-asset min weight (when long-only)
    w_max: float = 0.30            # per-asset max weight (concentration cap)
    gross_max: float = 1.0         # ||w||_1 cap (1.0 = fully invested, no leverage)
    cash_allowed: bool = True      # if True, sum(w) <= 1 (rest is cash); else == 1
    solver: str = "CLARABEL"       # CVXPY solver name; CLARABEL, ECOS, SCS all work


def _k_alpha(alpha: float) -> float:
    """CVaR-of-normal scaling: k = phi(z) / (1 - alpha)."""
    z = norm.ppf(alpha)
    return float(norm.pdf(z) / (1.0 - alpha))


# --------------------------------------------------------------------------- #
# Closed-form Gaussian CVaR optimizer
# --------------------------------------------------------------------------- #

def optimize_cvar(
    mu: np.ndarray,
    sigma: np.ndarray,
    prev_w: np.ndarray,
    mask: np.ndarray,
    cfg: CVaRConfig,
) -> tuple[np.ndarray, dict]:
    """Solve the Gaussian-CVaR + L1-turnover SOCP.

    Inputs:
      mu      : (N,) expected returns
      sigma   : (N, N) covariance (PSD)
      prev_w  : (N,) previous weights (full-length; masked indices contribute
                turnover cost if a held asset becomes inactive)
      mask    : (N,) bool — True = tradable now
      cfg     : CVaRConfig
    Returns:
      w_full  : (N,) optimal weights (zero on masked-out indices)
      info    : dict with solver status, expected return, expected vol, CVaR, turnover
    """
    N = mu.shape[0]
    active = np.where(mask)[0]
    k = active.size
    if k == 0:
        return np.zeros(N), {"status": "no_active_assets"}

    mu_a = mu[active]
    sig_a = sigma[np.ix_(active, active)]
    prev_a = prev_w[active]
    # Forced unwind cost: previous weight in inactive assets pays full turnover
    # (we have to sell them, so the L1 cost on the inactive side is fixed).
    inactive = np.where(~mask)[0]
    forced_unwind = float(np.abs(prev_w[inactive]).sum())

    w = cp.Variable(k)
    if cfg.long_only:
        constraints = [w >= cfg.w_min, w <= cfg.w_max]
    else:
        constraints = [w >= -cfg.w_max, w <= cfg.w_max]
    if cfg.cash_allowed:
        constraints += [cp.sum(w) <= 1.0, cp.sum(w) >= 0.0]
    else:
        constraints += [cp.sum(w) == 1.0]
    if cfg.gross_max is not None:
        constraints += [cp.norm1(w) <= cfg.gross_max]

    k_a = _k_alpha(cfg.alpha)
    # Symmetrize + jitter, then take Cholesky so portfolio vol = ||L^T w||_2,
    # which is a DCP-compliant SOC term (cp.sqrt(cp.quad_form(...)) is not).
    sig_sym = 0.5 * (sig_a + sig_a.T) + 1e-8 * np.eye(k)
    try:
        chol_a = np.linalg.cholesky(sig_sym)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sig_sym)
        eigvals = np.clip(eigvals, 1e-8, None)
        sig_sym = (eigvecs * eigvals) @ eigvecs.T
        chol_a = np.linalg.cholesky(sig_sym)
    portfolio_vol = cp.norm(chol_a.T @ w, 2)
    cvar_term = -mu_a @ w + k_a * portfolio_vol
    turnover = cp.norm1(w - prev_a)
    objective = cp.Minimize(cvar_term + cfg.turnover_lambda * turnover)

    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cfg.solver, verbose=False)
    except (cp.SolverError, Exception):
        # Fallback chain.
        solved = False
        for backup in ("ECOS", "SCS", "CLARABEL"):
            if backup == cfg.solver:
                continue
            try:
                problem.solve(solver=backup, verbose=False)
                solved = True
                break
            except Exception:
                continue
        if not solved:
            return prev_w * 0.0, {"status": "solver_failed"}

    if w.value is None:
        return prev_w * 0.0, {"status": problem.status}

    w_active = np.asarray(w.value).flatten()
    # Defensive cleanup: tiny numerical negatives -> 0 under long-only.
    if cfg.long_only:
        w_active = np.clip(w_active, 0.0, None)
    w_full = np.zeros(N)
    w_full[active] = w_active

    info = {
        "status": problem.status,
        "expected_return": float(mu_a @ w_active),
        "expected_vol": float(np.sqrt(max(w_active @ sig_a @ w_active, 0.0))),
        "cvar_alpha": float(-(mu_a @ w_active) + k_a * np.sqrt(max(w_active @ sig_a @ w_active, 0.0))),
        "turnover_active": float(np.abs(w_active - prev_a).sum()),
        "turnover_total": float(np.abs(w_full - prev_w).sum()),
        "forced_unwind": forced_unwind,
        "n_active": int(k),
        "k_alpha": k_a,
    }
    return w_full, info


# --------------------------------------------------------------------------- #
# Sample-based CVaR (asymmetric / non-Gaussian)
# --------------------------------------------------------------------------- #

def optimize_cvar_scenarios(
    scenarios: np.ndarray,           # (M, N) sampled returns
    prev_w: np.ndarray,
    mask: np.ndarray,
    cfg: CVaRConfig,
) -> tuple[np.ndarray, dict]:
    """Rockafellar-Uryasev CVaR LP.

    Useful when the predicted distribution is non-Gaussian (e.g., samples from
    a generative model or bootstrap from residuals). Slower than the closed-form
    Gaussian case but handles arbitrary scenario sets.
    """
    M, N = scenarios.shape
    active = np.where(mask)[0]
    k = active.size
    if k == 0:
        return np.zeros(N), {"status": "no_active_assets"}

    R = scenarios[:, active]                          # (M, k)
    prev_a = prev_w[active]
    w = cp.Variable(k)
    eta = cp.Variable()
    u = cp.Variable(M)

    constraints = [u >= 0, u >= -R @ w - eta]
    if cfg.long_only:
        constraints += [w >= cfg.w_min, w <= cfg.w_max]
    else:
        constraints += [w >= -cfg.w_max, w <= cfg.w_max]
    if cfg.cash_allowed:
        constraints += [cp.sum(w) <= 1.0, cp.sum(w) >= 0.0]
    else:
        constraints += [cp.sum(w) == 1.0]
    if cfg.gross_max is not None:
        constraints += [cp.norm1(w) <= cfg.gross_max]

    cvar_term = eta + (1.0 / ((1.0 - cfg.alpha) * M)) * cp.sum(u)
    turnover = cp.norm1(w - prev_a)
    problem = cp.Problem(cp.Minimize(cvar_term + cfg.turnover_lambda * turnover), constraints)

    try:
        problem.solve(solver=cfg.solver, verbose=False)
    except Exception:
        for backup in ("ECOS", "SCS", "CLARABEL"):
            if backup == cfg.solver:
                continue
            try:
                problem.solve(solver=backup, verbose=False)
                break
            except Exception:
                continue

    if w.value is None:
        return prev_w * 0.0, {"status": problem.status}
    w_active = np.asarray(w.value).flatten()
    if cfg.long_only:
        w_active = np.clip(w_active, 0.0, None)
    w_full = np.zeros(N)
    w_full[active] = w_active
    info = {
        "status": problem.status,
        "cvar_alpha": float(problem.value - cfg.turnover_lambda * np.abs(w_active - prev_a).sum()),
        "turnover_total": float(np.abs(w_full - prev_w).sum()),
        "n_active": int(k),
    }
    return w_full, info
