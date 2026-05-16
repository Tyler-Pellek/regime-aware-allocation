"""Baseline allocation policies used as benchmarks in the backtest report.

Each baseline implements the same interface:
    weights = policy(prev_w, mask, history) -> (N,) np.ndarray

`history` is a (T, N) DataFrame of trailing log returns (so baselines can
estimate their own covariance).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .cvar import CVaRConfig, optimize_cvar


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
    w_full, _ = optimize_cvar(mu, sig, prev_w, mask, cfg)
    return w_full


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

BASELINES = {
    "equal_weight": equal_weight,
    "inverse_vol": inverse_vol,
    "buy_and_hold": buy_and_hold,
    "sample_cov_cvar": sample_cov_cvar,
}
