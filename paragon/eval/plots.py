"""Matplotlib reporting plots. Saves PNGs to a directory."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .metrics import cum_return, max_drawdown


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def equity_curves(returns_dict: dict[str, pd.Series], outpath: str | Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, r in returns_dict.items():
        if r.empty:
            continue
        eq = cum_return(r)
        ax.plot(eq.index, eq.values, label=f"{name} (CAGR-ish={(eq.iloc[-1] - 1):.1%})", lw=1.4)
    ax.set_title("Equity curves (growth of $1)")
    ax.set_ylabel("Cumulative wealth")
    ax.set_yscale("log")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    _save(fig, Path(outpath))


def drawdown(returns: pd.Series, outpath: str | Path, label: str = "Strategy") -> None:
    eq = cum_return(returns)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.fill_between(dd.index, dd.values, 0, color="firebrick", alpha=0.4)
    ax.set_title(f"{label} — drawdown (max = {dd.min():.1%})")
    ax.grid(alpha=0.3)
    _save(fig, Path(outpath))


def regime_overlay(
    returns: pd.Series, regime_probs: pd.DataFrame, outpath: str | Path
) -> None:
    if regime_probs.empty:
        return
    eq = cum_return(returns)
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(eq.index, eq.values, color="black", lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_title("Strategy equity + inferred regime")
    axes[0].grid(alpha=0.3)
    # Stacked area of regime probs (clipped to returns range)
    rp = regime_probs.reindex(returns.index).ffill()
    axes[1].stackplot(rp.index, rp.T.values, labels=rp.columns, alpha=0.7)
    axes[1].set_ylim(0, 1)
    axes[1].legend(loc="upper left", fontsize=8, ncol=len(rp.columns))
    axes[1].set_title("HMM regime posteriors")
    axes[1].grid(alpha=0.3)
    _save(fig, Path(outpath))


def weights_heatmap(weights: pd.DataFrame, outpath: str | Path) -> None:
    if weights.empty:
        return
    fig, ax = plt.subplots(figsize=(11, max(3, 0.3 * weights.shape[1])))
    data = weights.T  # tickers x time
    im = ax.imshow(
        data.values, aspect="auto", interpolation="nearest", origin="lower",
        cmap="viridis", extent=[0, data.shape[1], 0, data.shape[0]],
    )
    ax.set_yticks(np.arange(data.shape[0]) + 0.5)
    ax.set_yticklabels(data.index, fontsize=8)
    n = data.shape[1]
    step = max(n // 12, 1)
    ax.set_xticks(np.arange(0, n, step) + 0.5)
    ax.set_xticklabels([str(d.date()) for d in data.columns[::step]], rotation=45, ha="right", fontsize=8)
    fig.colorbar(im, ax=ax, label="weight")
    ax.set_title("Portfolio weights through time")
    _save(fig, Path(outpath))


def rolling_sharpe(returns: pd.Series, outpath: str | Path, window: int = 126) -> None:
    if len(returns) < window:
        return
    mu = returns.rolling(window).mean()
    sd = returns.rolling(window).std(ddof=0)
    rs = (mu / sd) * np.sqrt(252)
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.plot(rs.index, rs.values, color="navy", lw=1.2)
    ax.axhline(0, color="grey", lw=0.5)
    ax.set_title(f"Rolling {window}-day annualized Sharpe")
    ax.grid(alpha=0.3)
    _save(fig, Path(outpath))


def turnover_plot(decisions: pd.DataFrame, outpath: str | Path) -> None:
    if decisions.empty or "turnover_total" not in decisions:
        return
    fig, ax = plt.subplots(figsize=(11, 3.5))
    ax.bar(decisions.index, decisions["turnover_total"].values, width=4.0, color="teal", alpha=0.7)
    ax.set_title("L1 turnover at each rebalance")
    ax.set_ylabel("|Δw|_1")
    ax.grid(alpha=0.3)
    _save(fig, Path(outpath))
