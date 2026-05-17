"""Chart v7 (apples-to-apples + extended) vs v6_03 winner + all baselines.

Loads return series from the existing v6_03 winner report AND the new v7 runs,
overlays equity curves and prints a comparison table.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (label, report_dir, series_name_in_dir, color, linewidth)
# series_name_in_dir = "strategy" or "benchmark" or a baseline policy name
SERIES = [
    ("Paragon v7b NORMALIZED (CURRENT WINNER)",
     "v7b_normalized",              "strategy",        "#9d0208", 2.8),
    ("Paragon v6_03 (prior winner)",
     "v6_03_winner_full_baselines", "strategy",        "#d62728", 2.0),
    ("Paragon v7 apples (unnormalized features - FAILED)",
     "v7_apples_to_apples",        "strategy",        "#fb8500", 1.4),
    ("Max Sharpe (tangency)",
     "v6_03_winner_full_baselines", "max_sharpe",      "#1f77b4", 1.2),
    ("Markowitz MV (sample stats)",
     "v6_03_winner_full_baselines", "sample_cov_cvar", "#ff7f0e", 1.2),
    ("Equal weight (1/N)",
     "v6_03_winner_full_baselines", "equal_weight",    "#2ca02c", 1.2),
    ("Risk parity (ERC)",
     "v6_03_winner_full_baselines", "risk_parity",     "#8c564b", 1.0),
    ("Inverse vol",
     "v6_03_winner_full_baselines", "inverse_vol",     "#9467bd", 1.0),
    ("Minimum variance",
     "v6_03_winner_full_baselines", "min_variance",    "#7f7f7f", 1.0),
    ("SPY benchmark",
     "v6_03_winner_full_baselines", "benchmark",       "#17becf", 1.0),
]


def load_returns(report_dir: str, series: str) -> pd.Series:
    base = Path("artifacts/reports") / report_dir
    if series == "strategy":
        path = base / "strategy_returns.csv"
    elif series == "benchmark":
        path = base / "benchmark_returns.csv"
    else:
        path = base / f"baseline_{series}_returns.csv"
    if not path.is_file():
        return pd.Series(dtype=float)
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return df.iloc[:, 0]


def summarize(rets: pd.Series) -> dict:
    if rets.empty:
        return {}
    eq = (1.0 + rets).cumprod()
    peak = eq.cummax()
    dd = (eq - peak) / peak
    n_years = max(len(rets) / 252.0, 1e-9)
    cagr = eq.iloc[-1] ** (1.0 / n_years) - 1.0
    ann_vol = rets.std(ddof=0) * np.sqrt(252)
    sharpe = (rets.mean() * 252) / ann_vol if ann_vol > 0 else np.nan
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else np.nan
    return dict(cagr=cagr, vol=ann_vol, sharpe=sharpe, max_dd=max_dd, calmar=calmar)


def main():
    out_dir = Path("artifacts/reports/v7_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = []
    for label, report, series, color, lw in SERIES:
        r = load_returns(report, series)
        if r.empty:
            print(f"  (skipping {label}: not found in {report})")
            continue
        data.append({"label": label, "color": color, "lw": lw, "rets": r,
                     "metrics": summarize(r)})

    if not data:
        print("No series found!")
        return

    # ---- Equity curves chart ----
    fig, ax = plt.subplots(figsize=(14, 7))
    for d in data:
        eq = (1.0 + d["rets"]).cumprod()
        m = d["metrics"]
        label = (f"{d['label']}  "
                 f"(Sh={m['sharpe']:.2f}, CAGR={m['cagr']*100:.1f}%, "
                 f"MaxDD={m['max_dd']*100:.1f}%)")
        ax.plot(eq.index, eq.values, label=label, color=d["color"], lw=d["lw"])
    ax.set_yscale("log")
    ax.set_title("Paragon v7 vs v6_03 vs classical baselines\n"
                 "Walk-forward backtest on 16 mega-cap tech",
                 fontsize=12)
    ax.set_ylabel("Cumulative wealth (log scale, start = $1)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=8, framealpha=0.92)
    fig.tight_layout()
    out_eq = out_dir / "equity_curves.png"
    fig.savefig(out_eq, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_eq}")

    # ---- Sharpe / Calmar bars ----
    names = [d["label"] for d in data]
    sharpes = [d["metrics"]["sharpe"] for d in data]
    calmars = [d["metrics"]["calmar"] for d in data]
    colors = [d["color"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    y = np.arange(len(names))
    axes[0].barh(y, sharpes, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_yticks(y); axes[0].set_yticklabels(names, fontsize=8)
    axes[0].invert_yaxis()
    axes[0].set_title("Sharpe ratio")
    axes[0].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(sharpes):
        axes[0].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    axes[1].barh(y, calmars, color=colors, edgecolor="black", linewidth=0.5)
    axes[1].set_yticks(y); axes[1].set_yticklabels(names, fontsize=8)
    axes[1].invert_yaxis()
    axes[1].set_title("Calmar ratio")
    axes[1].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(calmars):
        axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_bar = out_dir / "sharpe_calmar.png"
    fig.savefig(out_bar, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_bar}")

    # ---- Table ----
    print()
    print(f"  {'Strategy':<46s}  {'CAGR':>7s}  {'Vol':>7s}  {'Sharpe':>7s}  {'MaxDD':>8s}  {'Calmar':>7s}")
    print("  " + "-" * 100)
    for d in data:
        m = d["metrics"]
        print(f"  {d['label']:<46s}  "
              f"{m['cagr']*100:6.2f}%  {m['vol']*100:6.2f}%  "
              f"{m['sharpe']:7.2f}  {m['max_dd']*100:7.2f}%  {m['calmar']:7.2f}")


if __name__ == "__main__":
    main()
