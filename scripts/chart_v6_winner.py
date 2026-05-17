"""Generate a single comparison chart: v6_03 strategy vs all classical baselines.

Reads the run's CSVs for strategy + each baseline return series and produces:
  - artifacts/reports/v6_03_winner_full_baselines/figures/comparison.png
    (equity curves on log scale)
  - artifacts/reports/v6_03_winner_full_baselines/figures/metrics_bar.png
    (bar chart of Sharpe and Calmar across all strategies)
  - prints a clean summary table

Run after the full-baseline v6_03 backtest completes.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPORT_DIR = Path("artifacts/reports/v6_03_winner_full_baselines")


def _load_returns(name: str) -> pd.Series:
    """Strategy/benchmark live at fixed paths; baselines follow a prefix pattern."""
    if name == "strategy":
        path = REPORT_DIR / "strategy_returns.csv"
    elif name == "benchmark":
        path = REPORT_DIR / "benchmark_returns.csv"
    else:
        path = REPORT_DIR / f"baseline_{name}_returns.csv"
    if not path.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


# Display order + labels + colors
SERIES = [
    ("strategy",        "Paragon (warm-start + factor head)", "#d62728", 2.5),
    ("max_sharpe",      "Max Sharpe (tangency)",              "#1f77b4", 1.2),
    ("sample_cov_cvar", "Markowitz MV (sample stats)",        "#ff7f0e", 1.2),
    ("equal_weight",    "Equal weight (1/N)",                 "#2ca02c", 1.2),
    ("inverse_vol",     "Inverse vol",                        "#9467bd", 1.0),
    ("risk_parity",     "Risk parity (ERC)",                  "#8c564b", 1.0),
    ("min_variance",    "Minimum variance",                   "#7f7f7f", 1.0),
    ("benchmark",       "SPY benchmark",                      "#17becf", 1.0),
]


def equity_curve(rets: pd.Series) -> pd.Series:
    return (1.0 + rets).cumprod()


def summarize(rets: pd.Series) -> dict:
    if rets.empty:
        return {}
    eq = equity_curve(rets)
    peak = eq.cummax()
    dd = (eq - peak) / peak
    n_years = max(len(rets) / 252.0, 1e-9)
    cagr = eq.iloc[-1] ** (1.0 / n_years) - 1.0
    ann_vol = rets.std(ddof=0) * np.sqrt(252)
    sharpe = (rets.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return {"cagr": cagr, "vol": ann_vol, "sharpe": sharpe, "max_dd": max_dd, "calmar": calmar}


def main() -> None:
    fig_dir = REPORT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    data: dict[str, dict] = {}
    for key, label, color, lw in SERIES:
        r = _load_returns(key)
        if r.empty:
            print(f"  (skipping {key}: no returns file)")
            continue
        data[key] = {"label": label, "color": color, "lw": lw, "rets": r, "metrics": summarize(r)}

    # ---- Equity curves (log scale) ----
    fig, ax = plt.subplots(figsize=(13, 6.5))
    for key, label, color, lw in SERIES:
        if key not in data:
            continue
        rets = data[key]["rets"]
        eq = equity_curve(rets)
        m = data[key]["metrics"]
        sharpe_label = f"Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']*100:.1f}%, MaxDD={m['max_dd']*100:.1f}%"
        ax.plot(eq.index, eq.values, label=f"{label}  ({sharpe_label})", color=color, lw=lw)
    ax.set_yscale("log")
    ax.set_title("Paragon (winning config) vs classical portfolio-construction baselines\n"
                 "Walk-forward backtest, 16 mega-cap tech, 2008-01 to 2020-04",
                 fontsize=12)
    ax.set_ylabel("Cumulative wealth (log scale, start = $1)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    out_eq = fig_dir / "comparison.png"
    fig.savefig(out_eq, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_eq}")

    # ---- Metrics bar chart (Sharpe + Calmar) ----
    names = [data[k]["label"] for k, *_ in SERIES if k in data]
    sharpes = [data[k]["metrics"]["sharpe"] for k, *_ in SERIES if k in data]
    calmars = [data[k]["metrics"]["calmar"] for k, *_ in SERIES if k in data]
    colors = [data[k]["color"] for k, *_ in SERIES if k in data]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    y = np.arange(len(names))
    axes[0].barh(y, sharpes, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=9)
    axes[0].invert_yaxis()
    axes[0].set_title("Sharpe ratio (higher = better risk-adjusted return)")
    axes[0].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(sharpes):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)

    axes[1].barh(y, calmars, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y); axes[1].set_yticklabels(names, fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_title("Calmar ratio (CAGR / |MaxDD|, higher = better tail protection)")
    axes[1].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(calmars):
        axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_bar = fig_dir / "metrics_bar.png"
    fig.savefig(out_bar, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_bar}")

    # ---- Printed summary ----
    print()
    print(f"  {'Strategy':<40s}  {'CAGR':>7s}  {'Vol':>7s}  {'Sharpe':>7s}  {'MaxDD':>8s}  {'Calmar':>7s}")
    print("  " + "-" * 95)
    for key, label, *_ in SERIES:
        if key not in data:
            continue
        m = data[key]["metrics"]
        print(f"  {label:<40s}  "
              f"{m['cagr']*100:6.2f}%  {m['vol']*100:6.2f}%  "
              f"{m['sharpe']:7.2f}  {m['max_dd']*100:7.2f}%  {m['calmar']:7.2f}")


if __name__ == "__main__":
    main()
