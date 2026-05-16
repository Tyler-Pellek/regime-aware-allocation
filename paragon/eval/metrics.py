"""Performance and risk metrics for daily-return series."""
from __future__ import annotations

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def cum_return(rets: pd.Series) -> pd.Series:
    return (1.0 + rets).cumprod()


def total_return(rets: pd.Series) -> float:
    return float((1.0 + rets).prod() - 1.0)


def cagr(rets: pd.Series) -> float:
    if rets.empty:
        return float("nan")
    n_years = max(len(rets) / TRADING_DAYS, 1e-9)
    return float((1.0 + rets).prod() ** (1.0 / n_years) - 1.0)


def ann_vol(rets: pd.Series) -> float:
    return float(rets.std(ddof=0) * np.sqrt(TRADING_DAYS))


def sharpe(rets: pd.Series, rf: float = 0.0) -> float:
    excess = rets - rf / TRADING_DAYS
    sd = excess.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(excess.mean() / sd * np.sqrt(TRADING_DAYS))


def sortino(rets: pd.Series, rf: float = 0.0) -> float:
    excess = rets - rf / TRADING_DAYS
    downside = excess.clip(upper=0.0)
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0 or np.isnan(dd):
        return float("nan")
    return float(excess.mean() / dd * np.sqrt(TRADING_DAYS))


def max_drawdown(rets: pd.Series) -> float:
    eq = cum_return(rets)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min()) if not dd.empty else float("nan")


def calmar(rets: pd.Series) -> float:
    mdd = max_drawdown(rets)
    if mdd == 0 or np.isnan(mdd):
        return float("nan")
    return float(cagr(rets) / abs(mdd))


def hist_var(rets: pd.Series, alpha: float = 0.95) -> float:
    """Empirical 1-day VaR (positive number = loss magnitude)."""
    if rets.empty:
        return float("nan")
    return float(-np.quantile(rets, 1.0 - alpha))


def hist_cvar(rets: pd.Series, alpha: float = 0.95) -> float:
    """Empirical 1-day CVaR (mean loss in the worst (1-alpha) tail)."""
    if rets.empty:
        return float("nan")
    threshold = np.quantile(rets, 1.0 - alpha)
    tail = rets[rets <= threshold]
    if len(tail) == 0:
        return float("nan")
    return float(-tail.mean())


def turnover(weights: pd.DataFrame) -> pd.Series:
    """L1 turnover between consecutive rebalance dates."""
    if weights.empty:
        return pd.Series(dtype=float)
    return weights.diff().abs().sum(axis=1)


def annualized_turnover(weights: pd.DataFrame, freq_days: int) -> float:
    """Sum of per-rebalance L1 turnover, annualized by rebalances/year."""
    t = turnover(weights).sum()
    n_rebals = len(weights)
    if n_rebals < 2:
        return float("nan")
    rebals_per_year = TRADING_DAYS / freq_days
    return float(t / n_rebals * rebals_per_year)


def alpha_beta(rets: pd.Series, bench: pd.Series) -> tuple[float, float]:
    """Annualized alpha + beta from OLS of strategy on benchmark daily returns."""
    common = rets.index.intersection(bench.index)
    r = rets.loc[common]
    b = bench.loc[common]
    if len(common) < 30:
        return float("nan"), float("nan")
    cov = np.cov(r, b, ddof=0)
    beta = float(cov[0, 1] / cov[1, 1])
    alpha_daily = float(r.mean() - beta * b.mean())
    return alpha_daily * TRADING_DAYS, beta


def information_ratio(rets: pd.Series, bench: pd.Series) -> float:
    common = rets.index.intersection(bench.index)
    diff = (rets.loc[common] - bench.loc[common])
    sd = diff.std(ddof=0)
    if sd == 0 or np.isnan(sd):
        return float("nan")
    return float(diff.mean() / sd * np.sqrt(TRADING_DAYS))


def hit_rate(rets: pd.Series) -> float:
    if rets.empty:
        return float("nan")
    return float((rets > 0).mean())


def summarize(
    rets: pd.Series,
    bench: pd.Series | None = None,
    weights: pd.DataFrame | None = None,
    rebal_freq_days: int = 5,
    rf: float = 0.0,
) -> dict:
    out = {
        "total_return": total_return(rets),
        "cagr": cagr(rets),
        "ann_vol": ann_vol(rets),
        "sharpe": sharpe(rets, rf=rf),
        "sortino": sortino(rets, rf=rf),
        "max_drawdown": max_drawdown(rets),
        "calmar": calmar(rets),
        "hist_var_95": hist_var(rets, 0.95),
        "hist_cvar_95": hist_cvar(rets, 0.95),
        "hit_rate_daily": hit_rate(rets),
        "n_days": int(len(rets)),
    }
    if bench is not None and len(bench) > 0:
        a, b = alpha_beta(rets, bench)
        out.update({"alpha_ann": a, "beta": b, "info_ratio": information_ratio(rets, bench)})
    if weights is not None and len(weights) > 0:
        out["ann_turnover"] = annualized_turnover(weights, rebal_freq_days)
        out["avg_n_active"] = float((weights.abs() > 1e-6).sum(axis=1).mean())
    return out
