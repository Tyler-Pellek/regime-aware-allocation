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
    mode: str = "cvar"             # "cvar" (minimize CVaR) or "mv" (mean-variance:
                                   # max mu'w - gamma/2 * w'Sigma w with vol cap).
                                   # CVaR makes sense for single-asset-class
                                   # universes; mean-variance is the right tool
                                   # when bonds/cash-equivalents are in the
                                   # universe (otherwise CVaR degenerates to
                                   # parking in the lowest-vol asset).
    alpha: float = 0.95            # CVaR level (e.g., 0.95 = expected loss in worst 5%)
    risk_aversion: float = 20.0    # gamma in mean-variance. Higher = more
                                   # risk-averse. ~10-50 for weekly horizon.
    gamma_cv_grid: list[float] | None = None
                                   # If set, the walkforward picks the gamma
                                   # from this grid that maximizes Sharpe on
                                   # a held-out validation window at the end
                                   # of each fold's training data. Different
                                   # regimes want different risk aversion;
                                   # data-driven gamma adapts per fold.
    demean_mu: bool = False        # if True, subtract cross-sectional mean of mu
                                   # over active assets before optimization.
                                   # Converts absolute-return predictions into
                                   # relative-rank signals; useful when the
                                   # model's mu predictions have weak absolute
                                   # signal but real relative signal.
    turnover_lambda: float = 5e-4  # L1 turnover penalty coefficient
    long_only: bool = True
    w_min: float = 0.0             # per-asset min weight (when long-only)
    w_max: float = 0.30            # per-asset max weight (concentration cap)
    gross_max: float = 1.0         # ||w||_1 cap (1.0 = fully invested, no leverage)
    cash_allowed: bool = True      # if True, sum(w) <= 1 (rest is cash); else == 1
    min_invested: float = 0.0      # only honored when cash_allowed=True; floors
                                   # the gross book at this fraction (e.g. 0.5 =
                                   # keep at least 50% in equities). Prevents the
                                   # degenerate w=0 solution seen when CVaR cost
                                   # dominates expected return.
    vol_cap_weekly: float | None = None  # optional explicit weekly vol cap on the
                                         # predicted portfolio: ||L^T w||_2 <= cap.
                                         # Defensive against over-confident Sigma.
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
        constraints += [cp.sum(w) <= 1.0, cp.sum(w) >= max(0.0, cfg.min_invested)]
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
    if cfg.vol_cap_weekly is not None:
        constraints += [portfolio_vol <= cfg.vol_cap_weekly]
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
# Mean-variance optimizer with vol cap (the right tool for multi-asset)
# --------------------------------------------------------------------------- #

def optimize_mean_variance(
    mu: np.ndarray,
    sigma: np.ndarray,
    prev_w: np.ndarray,
    mask: np.ndarray,
    cfg: CVaRConfig,
) -> tuple[np.ndarray, dict]:
    """Solve the constrained mean-variance problem:

        minimize    -mu^T w  +  (gamma/2) w^T Sigma w  +  lambda |w - w_prev|_1
        s.t.        sum(w) = 1                 (fully invested by default)
                    0 <= w_i <= w_max
                    sqrt(w^T Sigma w) <= vol_cap_weekly    (optional)

    Equivalent to "maximize expected return minus quadratic vol penalty". The
    right objective for multi-asset universes where bonds-equity correlations
    matter; CVaR minimization alone would degenerate to bonds.

    If `cfg.demean_mu`, mu is centered cross-sectionally over active assets
    before optimization — converts absolute-return predictions into relative
    rank signals, more robust when mu has weak absolute scale.
    """
    N = mu.shape[0]
    active = np.where(mask)[0]
    k = active.size
    if k == 0:
        return np.zeros(N), {"status": "no_active_assets"}

    mu_a = mu[active].copy()
    if cfg.demean_mu and k > 1:
        mu_a = mu_a - mu_a.mean()
    sig_a = sigma[np.ix_(active, active)]
    prev_a = prev_w[active]

    sig_sym = 0.5 * (sig_a + sig_a.T) + 1e-8 * np.eye(k)
    try:
        chol_a = np.linalg.cholesky(sig_sym)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sig_sym)
        eigvals = np.clip(eigvals, 1e-8, None)
        sig_sym = (eigvecs * eigvals) @ eigvecs.T
        chol_a = np.linalg.cholesky(sig_sym)

    w = cp.Variable(k)
    if cfg.long_only:
        constraints = [w >= cfg.w_min, w <= cfg.w_max]
    else:
        constraints = [w >= -cfg.w_max, w <= cfg.w_max]
    if cfg.cash_allowed:
        constraints += [cp.sum(w) <= 1.0, cp.sum(w) >= max(0.0, cfg.min_invested)]
    else:
        constraints += [cp.sum(w) == 1.0]
    if cfg.gross_max is not None:
        constraints += [cp.norm1(w) <= cfg.gross_max]
    portfolio_vol = cp.norm(chol_a.T @ w, 2)
    if cfg.vol_cap_weekly is not None:
        constraints += [portfolio_vol <= cfg.vol_cap_weekly]

    # variance term as ||L^T w||_2^2 (DCP-friendly, no need for psd_wrap)
    variance_term = cp.sum_squares(chol_a.T @ w)
    turnover = cp.norm1(w - prev_a)
    objective = cp.Minimize(
        -mu_a @ w
        + 0.5 * cfg.risk_aversion * variance_term
        + cfg.turnover_lambda * turnover
    )
    problem = cp.Problem(objective, constraints)
    try:
        problem.solve(solver=cfg.solver, verbose=False)
    except Exception:
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
        if not solved or w.value is None:
            return prev_w * 0.0, {"status": "solver_failed"}

    if w.value is None:
        return prev_w * 0.0, {"status": problem.status}

    w_active = np.asarray(w.value).flatten()
    if cfg.long_only:
        w_active = np.clip(w_active, 0.0, None)
    w_full = np.zeros(N)
    w_full[active] = w_active

    realized_vol = float(np.sqrt(max(w_active @ sig_a @ w_active, 0.0)))
    info = {
        "status": problem.status,
        "expected_return": float(mu_a @ w_active),
        "expected_vol": realized_vol,
        "sharpe_like": float((mu_a @ w_active) / (realized_vol + 1e-12)),
        "turnover_active": float(np.abs(w_active - prev_a).sum()),
        "turnover_total": float(np.abs(w_full - prev_w).sum()),
        "n_active": int(k),
        "vol_cap_binding": bool(
            cfg.vol_cap_weekly is not None and realized_vol >= cfg.vol_cap_weekly - 1e-5
        ),
    }
    return w_full, info


# --------------------------------------------------------------------------- #
# Unified dispatch
# --------------------------------------------------------------------------- #

def optimize_portfolio(
    mu: np.ndarray,
    sigma: np.ndarray,
    prev_w: np.ndarray,
    mask: np.ndarray,
    cfg: CVaRConfig,
) -> tuple[np.ndarray, dict]:
    """Dispatch on cfg.mode -> the right optimizer."""
    if cfg.mode == "mv":
        return optimize_mean_variance(mu, sigma, prev_w, mask, cfg)
    return optimize_cvar(mu, sigma, prev_w, mask, cfg)


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
