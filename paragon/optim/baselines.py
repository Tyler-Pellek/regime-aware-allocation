"""Baseline allocation policies used as benchmarks in the backtest report.

Each baseline implements the same interface:
    weights = policy(prev_w, mask, history) -> (N,) np.ndarray

`history` is a (T, N) DataFrame of trailing log returns (so baselines can
estimate their own covariance).

The baselines cover the canonical portfolio-construction reference points:
  - equal_weight     : 1/N
  - inverse_vol      : 1/sigma_i (degenerate risk parity assuming diagonal Sigma)
  - min_variance     : argmin w'Sigma w  s.t. fully-invested, long-only
  - max_sharpe       : argmax (mu'w) / sqrt(w'Sigma w)  (tangency portfolio)
  - risk_parity      : iterative equal risk-contribution allocation
  - sample_cov_cvar  : full mean-variance (or CVaR) optimizer with sample stats
"""
from __future__ import annotations

import cvxpy as cp
import numpy as np
import pandas as pd

from .cvar import CVaRConfig, optimize_cvar, optimize_portfolio


# --------------------------------------------------------------------------- #
# Trivial baselines
# --------------------------------------------------------------------------- #

def equal_weight(prev_w: np.ndarray, mask: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    """1/k across active names, 0 in cash."""
    N = mask.size
    k = int(mask.sum())
    if k == 0:
        return np.zeros(N)
    w = np.zeros(N)
    w[mask] = 1.0 / k
    return w


def inverse_vol(prev_w: np.ndarray, mask: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    """w_i ~ 1 / sigma_i, normalized to sum to 1 across active assets."""
    N = mask.size
    if not mask.any():
        return np.zeros(N)
    vols = history.iloc[-60:].std(ddof=0).values
    vols = np.where(np.isfinite(vols) & (vols > 1e-8), vols, np.nan)
    w = np.zeros(N)
    inv = 1.0 / vols
    inv[~mask] = 0.0
    inv[~np.isfinite(inv)] = 0.0
    s = inv.sum()
    if s > 0:
        w = inv / s
    else:
        w[mask] = 1.0 / mask.sum()
    return w


def buy_and_hold(prev_w: np.ndarray, mask: np.ndarray, history: pd.DataFrame) -> np.ndarray:
    """If prev_w is empty, equal-weight; else hold prev_w."""
    if np.allclose(prev_w, 0):
        return equal_weight(prev_w, mask, history)
    return prev_w


# --------------------------------------------------------------------------- #
# Sample-covariance CVaR (the apples-to-apples baseline for our model)
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Classical efficient-frontier reference points
# --------------------------------------------------------------------------- #

def _sample_stats(history: pd.DataFrame, mask: np.ndarray, lookback: int = 252):
    """Return (mu_active, Sigma_active, active_idx). Both scaled to 5-day horizon."""
    active = np.where(mask)[0]
    k = active.size
    if k == 0:
        return None, None, active
    win = history.iloc[-lookback:]
    mu_full = win.mean().values * 5.0
    sig_full = win.cov().values * 5.0
    mu_a = np.nan_to_num(mu_full[active], nan=0.0)
    sig_a = sig_full[np.ix_(active, active)]
    sig_a = np.nan_to_num(sig_a, nan=0.0)
    sig_a = 0.5 * (sig_a + sig_a.T) + 1e-6 * np.eye(k)
    return mu_a, sig_a, active


def min_variance(
    prev_w: np.ndarray,
    mask: np.ndarray,
    history: pd.DataFrame,
    w_max: float = 0.30,
    lookback: int = 252,
) -> np.ndarray:
    """argmin w' Sigma w  s.t. sum(w)=1, 0 <= w_i <= w_max.

    The classical min-variance portfolio. Ignores expected returns entirely
    — purely risk-driven. On a diversified universe this typically overweights
    bonds / low-vol names.
    """
    N = mask.size
    mu_a, sig_a, active = _sample_stats(history, mask, lookback)
    if mu_a is None:
        return np.zeros(N)
    k = active.size
    try:
        chol = np.linalg.cholesky(sig_a)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sig_a)
        chol = np.linalg.cholesky((eigvecs * np.clip(eigvals, 1e-8, None)) @ eigvecs.T)
    w = cp.Variable(k)
    obj = cp.Minimize(cp.sum_squares(chol.T @ w))
    cons = [cp.sum(w) == 1.0, w >= 0.0, w <= w_max]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        prob.solve(solver="ECOS", verbose=False)
    if w.value is None:
        return prev_w * 0.0
    w_full = np.zeros(N)
    w_full[active] = np.clip(np.asarray(w.value).flatten(), 0.0, None)
    return w_full


def max_sharpe(
    prev_w: np.ndarray,
    mask: np.ndarray,
    history: pd.DataFrame,
    w_max: float = 0.30,
    lookback: int = 252,
) -> np.ndarray:
    """Tangency portfolio: argmax (mu' w) / sqrt(w' Sigma w), long-only, fully
    invested.

    Solved via the standard rotation: substitute y = w / (mu' w) (assuming
    mu' w > 0), the problem becomes min y' Sigma y s.t. mu' y = 1, y >= 0;
    then renormalize w = y / sum(y) to enforce sum(w) = 1. This handles the
    long-only constraint cleanly.

    Falls back to inverse-vol if mu is mostly non-positive (no positive-return
    portfolio achievable).
    """
    N = mask.size
    mu_a, sig_a, active = _sample_stats(history, mask, lookback)
    if mu_a is None:
        return np.zeros(N)
    k = active.size
    if (mu_a <= 0).all():
        return inverse_vol(prev_w, mask, history)
    try:
        chol = np.linalg.cholesky(sig_a)
    except np.linalg.LinAlgError:
        eigvals, eigvecs = np.linalg.eigh(sig_a)
        chol = np.linalg.cholesky((eigvecs * np.clip(eigvals, 1e-8, None)) @ eigvecs.T)
    y = cp.Variable(k)
    obj = cp.Minimize(cp.sum_squares(chol.T @ y))
    cons = [mu_a @ y == 1.0, y >= 0.0]
    prob = cp.Problem(obj, cons)
    try:
        prob.solve(solver="CLARABEL", verbose=False)
    except Exception:
        prob.solve(solver="ECOS", verbose=False)
    if y.value is None or np.asarray(y.value).sum() <= 1e-9:
        return inverse_vol(prev_w, mask, history)
    y_arr = np.clip(np.asarray(y.value).flatten(), 0.0, None)
    w_a = y_arr / max(y_arr.sum(), 1e-12)
    # Enforce per-asset cap by clipping + renormalizing (simple projection)
    w_a = np.minimum(w_a, w_max)
    if w_a.sum() < 1.0:
        # add residual proportionally to uncapped names
        slack = 1.0 - w_a.sum()
        room = w_max - w_a
        room = np.where(room > 1e-12, room, 0.0)
        if room.sum() > 1e-12:
            w_a = w_a + slack * (room / room.sum())
    w_a = w_a / max(w_a.sum(), 1e-12)
    w_full = np.zeros(N)
    w_full[active] = w_a
    return w_full


def risk_parity(
    prev_w: np.ndarray,
    mask: np.ndarray,
    history: pd.DataFrame,
    n_iter: int = 50,
    tol: float = 1e-6,
    lookback: int = 252,
) -> np.ndarray:
    """Equal Risk Contribution (ERC) portfolio.

    Each asset's marginal contribution to portfolio variance is equalized:
        w_i * (Sigma w)_i = (1/N) * w' Sigma w  for all i

    Iteratively solved via the canonical fixed-point update (Maillard, Roncalli,
    Teiletche 2010):
        w_i <- w_i * (1 / (Sigma w)_i)
        w_i <- w_i / sum(w)
    Converges in <20 iterations for typical covariances.
    """
    N = mask.size
    mu_a, sig_a, active = _sample_stats(history, mask, lookback)
    if mu_a is None:
        return np.zeros(N)
    k = active.size
    w = np.ones(k) / k
    for _ in range(n_iter):
        marginal_risk = sig_a @ w
        marginal_risk = np.where(marginal_risk > 1e-12, marginal_risk, 1e-12)
        new_w = 1.0 / marginal_risk
        new_w = new_w / new_w.sum()
        w_blend = 0.5 * (w + new_w)            # damped update for stability
        if np.abs(w_blend - w).max() < tol:
            w = w_blend
            break
        w = w_blend
    w_full = np.zeros(N)
    w_full[active] = w
    return w_full


def sample_cov_cvar(
    prev_w: np.ndarray,
    mask: np.ndarray,
    history: pd.DataFrame,
    cfg: CVaRConfig | None = None,
    lookback: int = 252,
) -> np.ndarray:
    """Use the rolling sample mean & covariance of returns as the optimizer
    inputs. This is the classical static covariance baseline that the memo
    targets — useful to show our learned Sigma actually beats it."""
    cfg = cfg or CVaRConfig()
    N = mask.size
    if not mask.any():
        return np.zeros(N)
    win = history.iloc[-lookback:]
    mu = win.mean().values * 5.0  # weekly horizon = 5 trading days
    sig = win.cov().values * 5.0
    # Replace NaN rows/cols (assets with insufficient history) with identity-ish.
    bad = ~np.isfinite(mu)
    if bad.any():
        mu[bad] = 0.0
        sig[bad, :] = 0.0
        sig[:, bad] = 0.0
        sig[bad, bad] = 1e-2
    bad_sig = ~np.isfinite(sig)
    sig[bad_sig] = 0.0
    sig = 0.5 * (sig + sig.T) + 1e-6 * np.eye(N)
    w_full, _ = optimize_portfolio(mu, sig, prev_w, mask, cfg)
    return w_full


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

BASELINES = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "buy_and_hold": buy_and_hold,
    "min_variance": min_variance,
    "max_sharpe": max_sharpe,
    "risk_parity": risk_parity,
    "sample_cov_cvar": sample_cov_cvar,
}
