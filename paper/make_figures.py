"""Generate all paper figures with a single consistent style.

Outputs (in /Users/tylerpellek/ParagonProject/paper/):
  fig_equity.pdf       — out-of-sample equity curves, strategy vs baselines + SPY
  fig_drawdown.pdf     — strategy drawdown profile vs SPY drawdown
  fig_metrics.pdf      — Sharpe + Calmar bar chart across all baselines
  fig_regime.pdf       — equity overlay against HMM regime probabilities
  fig_weights.pdf      — strategy weights heatmap through time
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# ---------- style ----------
mpl.rcParams.update({
    "font.family":        "serif",
    "font.serif":         ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size":          9,
    "axes.titlesize":     10,
    "axes.labelsize":     9,
    "xtick.labelsize":    8,
    "ytick.labelsize":    8,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.45,
    "grid.linestyle":     "-",
    "axes.axisbelow":     True,
    "lines.linewidth":    1.2,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
    "pdf.fonttype":       42,
    "ps.fonttype":        42,
})

# Consistent palette across all figures.
COLORS = {
    "strategy":        "#111111",   # black, principal series
    "max_sharpe":      "#0072B2",   # blue
    "sample_cov_cvar": "#D55E00",   # orange (Markowitz MV)
    "equal_weight":    "#009E73",   # green
    "risk_parity":     "#8B4513",   # brown
    "inverse_vol":     "#7B3F99",   # purple
    "min_variance":    "#757575",   # grey
    "benchmark":       "#56B4E9",   # sky blue (SPY)
}

LABELS = {
    "strategy":        "Strategy (production)",
    "max_sharpe":      "Long-only tangency",
    "sample_cov_cvar": "Markowitz MV (sample stats)",
    "equal_weight":    "Equal weight (1/N)",
    "risk_parity":     "Risk parity (ERC)",
    "inverse_vol":     "Inverse volatility",
    "min_variance":    "Minimum variance",
    "benchmark":       "SPY benchmark",
}

REPORT = Path("/Users/tylerpellek/ParagonProject/artifacts/reports/v7b_extended")
OUT = Path("/Users/tylerpellek/ParagonProject/paper")
OUT.mkdir(parents=True, exist_ok=True)


def load(name: str) -> pd.Series:
    if name == "strategy":
        p = REPORT / "strategy_returns.csv"
    elif name == "benchmark":
        p = REPORT / "benchmark_returns.csv"
    else:
        p = REPORT / f"baseline_{name}_returns.csv"
    df = pd.read_csv(p, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def summarize(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    n_years = max(len(r) / 252.0, 1e-9)
    cagr = eq.iloc[-1] ** (1.0 / n_years) - 1.0
    vol = r.std(ddof=0) * np.sqrt(252)
    sh = (r.mean() * 252) / vol if vol > 0 else np.nan
    dd = (eq - eq.cummax()) / eq.cummax()
    return dict(cagr=cagr, vol=vol, sh=sh, mdd=dd.min(),
                calmar=cagr / abs(dd.min()) if dd.min() < 0 else np.nan)


# ============================================================
# Figure 1 — equity curves
# ============================================================
def fig_equity():
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    series_order = [
        "strategy", "max_sharpe", "equal_weight", "sample_cov_cvar",
        "risk_parity", "inverse_vol", "min_variance", "benchmark",
    ]
    for name in series_order:
        r = load(name)
        eq = (1 + r).cumprod()
        ax.plot(eq.index, eq.values,
                color=COLORS[name],
                lw=1.6 if name == "strategy" else 1.0,
                label=LABELS[name])
    ax.set_yscale("log")
    ax.set_ylabel("Cumulative wealth (log scale)")
    ax.set_xlabel("Date")
    ax.legend(loc="upper left", framealpha=0.9, edgecolor="0.7", fancybox=False)
    fig.savefig(OUT / "fig_equity.pdf")
    plt.close(fig)
    print("wrote fig_equity.pdf")


# ============================================================
# Figure 2 — drawdown
# ============================================================
def fig_drawdown():
    fig, ax = plt.subplots(figsize=(6.5, 3.0))
    for name, lw in [("strategy", 1.6), ("benchmark", 1.0)]:
        r = load(name)
        eq = (1 + r).cumprod()
        dd = (eq - eq.cummax()) / eq.cummax()
        ax.fill_between(dd.index, dd.values, 0,
                        color=COLORS[name],
                        alpha=0.30 if name == "strategy" else 0.18,
                        linewidth=0)
        ax.plot(dd.index, dd.values, color=COLORS[name],
                lw=lw, label=LABELS[name])
    ax.set_ylabel("Drawdown")
    ax.set_xlabel("Date")
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.7", fancybox=False)
    fig.savefig(OUT / "fig_drawdown.pdf")
    plt.close(fig)
    print("wrote fig_drawdown.pdf")


# ============================================================
# Figure 3 — Sharpe + Calmar bars
# ============================================================
def fig_metrics():
    names = ["strategy", "max_sharpe", "equal_weight", "inverse_vol",
             "risk_parity", "min_variance", "sample_cov_cvar", "benchmark"]
    rows = []
    for n in names:
        m = summarize(load(n))
        rows.append((LABELS[n], COLORS[n], m["sh"], m["calmar"]))
    sortr = sorted(rows, key=lambda x: x[2], reverse=True)
    labels = [r[0] for r in sortr]
    colors = [r[1] for r in sortr]
    sharpes = [r[2] for r in sortr]
    calmars = [r[3] for r in sortr]

    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.2))
    y = np.arange(len(labels))
    axes[0].barh(y, sharpes, color=colors, edgecolor="black", lw=0.4)
    axes[0].set_yticks(y); axes[0].set_yticklabels(labels)
    axes[0].invert_yaxis()
    axes[0].set_title("Sharpe ratio")
    axes[0].grid(True, axis="x", linewidth=0.4, alpha=0.45)
    axes[0].grid(False, axis="y")
    for i, v in enumerate(sharpes):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    axes[1].barh(y, calmars, color=colors, edgecolor="black", lw=0.4)
    axes[1].set_yticks(y); axes[1].set_yticklabels([""] * len(labels))
    axes[1].invert_yaxis()
    axes[1].set_title("Calmar ratio")
    axes[1].grid(True, axis="x", linewidth=0.4, alpha=0.45)
    axes[1].grid(False, axis="y")
    for i, v in enumerate(calmars):
        axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=8)

    fig.savefig(OUT / "fig_metrics.pdf")
    plt.close(fig)
    print("wrote fig_metrics.pdf")


# ============================================================
# Figure 4 — equity + regime overlay
# ============================================================
def fig_regime():
    strat = load("strategy")
    eq = (1 + strat).cumprod()
    # Load regime probs from the report's CSV (decisions has them per decision date)
    # Easier: reload from the parquet/summary if available — here use the
    # already-rendered regime overlay underlying data from summary.json metadata
    # is not directly there, so reconstruct by reading regime_probs.csv if it
    # exists. If it doesn't we fall back to plotting just the equity curve.
    reg_path = REPORT / "regime_probs.csv"
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 4.0),
                              sharex=True,
                              gridspec_kw={"height_ratios": [2, 1]})
    axes[0].plot(eq.index, eq.values, color=COLORS["strategy"], lw=1.4)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Cumulative wealth (log)")

    if reg_path.exists():
        rp = pd.read_csv(reg_path, index_col=0, parse_dates=True)
        rp = rp.reindex(eq.index).ffill()
        regime_colors = ["#A8E6CF", "#FFD3B6", "#FFAAA5"]   # calm / mid / stress
        labels = ["Regime 0 (low vol)", "Regime 1 (mid)", "Regime 2 (stress)"]
        cols = list(rp.columns)
        axes[1].stackplot(rp.index, rp[cols].T.values,
                           colors=regime_colors[: len(cols)],
                           labels=labels[: len(cols)],
                           alpha=0.85, linewidth=0)
        axes[1].set_ylim(0, 1)
        axes[1].set_ylabel("Posterior prob.")
        axes[1].legend(loc="upper left", ncol=len(cols), framealpha=0.9,
                       edgecolor="0.7", fancybox=False)
    else:
        axes[1].text(0.5, 0.5, "Regime probabilities CSV not found",
                     ha="center", va="center", transform=axes[1].transAxes,
                     fontsize=8, color="gray")
        axes[1].set_ylim(0, 1)

    axes[1].set_xlabel("Date")
    fig.savefig(OUT / "fig_regime.pdf")
    plt.close(fig)
    print("wrote fig_regime.pdf")


# ============================================================
# Figure 5 — weights heatmap
# ============================================================
def fig_weights():
    w_path = REPORT / "weights.csv"
    if not w_path.exists():
        print("no weights.csv; skipping")
        return
    W = pd.read_csv(w_path, index_col=0, parse_dates=True)
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    arr = W.T.values   # (N, T)
    extent = [0, arr.shape[1], 0, arr.shape[0]]
    im = ax.imshow(arr, aspect="auto", origin="lower", cmap="cividis",
                   extent=extent, interpolation="nearest", vmin=0, vmax=arr.max())
    ax.set_yticks(np.arange(arr.shape[0]) + 0.5)
    ax.set_yticklabels(W.columns, fontsize=7)
    n = arr.shape[1]
    step = max(n // 10, 1)
    ax.set_xticks(np.arange(0, n, step) + 0.5)
    ax.set_xticklabels([str(d.date()) for d in W.index[::step]],
                       rotation=45, ha="right", fontsize=7)
    cbar = fig.colorbar(im, ax=ax, label="Weight", fraction=0.025, pad=0.02)
    cbar.ax.tick_params(labelsize=7)
    ax.grid(False)
    fig.savefig(OUT / "fig_weights.pdf")
    plt.close(fig)
    print("wrote fig_weights.pdf")


if __name__ == "__main__":
    fig_equity()
    fig_drawdown()
    fig_metrics()
    fig_regime()
    fig_weights()
