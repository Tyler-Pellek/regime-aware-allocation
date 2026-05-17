"""Final comprehensive chart: every Paragon version + every classical baseline.

Sets up the cleanest possible side-by-side comparison for the writeup.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# (label, report_dir, series_name, color, lw)
SERIES = [
    # Extended-window strategies (2008-2024)
    ("Paragon v7b EXTENDED (WINNER, 2008-2024)",
     "v7b_extended",                "strategy",        "#9d0208", 3.0),
    ("Paragon v9 realized-cov + ensemble EXTENDED (regressed)",
     "v9_extended",                 "strategy",        "#6a040f", 2.0),
    ("Paragon v8 pretrained EXTENDED",
     "v8_pretrained_extended",      "strategy",        "#370617", 2.0),
    # Apples-to-apples winners (2008-2020)
    ("Paragon v7b (2008-2020)",
     "v7b_normalized",              "strategy",        "#e85d04", 1.6),
    ("Paragon v6_03 (2008-2020)",
     "v6_03_winner_full_baselines", "strategy",        "#dc2f02", 1.2),
    # Baselines from the EXTENDED period
    ("Max Sharpe (tangency) [2008-2024]",
     "v7b_extended",                "max_sharpe",      "#1f77b4", 1.3),
    ("Markowitz MV [2008-2024]",
     "v7b_extended",                "sample_cov_cvar", "#ff7f0e", 1.3),
    ("Equal weight (1/N) [2008-2024]",
     "v7b_extended",                "equal_weight",    "#2ca02c", 1.3),
    ("Risk parity (ERC) [2008-2024]",
     "v7b_extended",                "risk_parity",     "#8c564b", 1.1),
    ("Inverse vol [2008-2024]",
     "v7b_extended",                "inverse_vol",     "#9467bd", 1.0),
    ("Minimum variance [2008-2024]",
     "v7b_extended",                "min_variance",    "#7f7f7f", 1.0),
    ("SPY benchmark [2008-2024]",
     "v7b_extended",                "benchmark",       "#17becf", 1.0),
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
    out = Path("artifacts/reports/final_chart"); out.mkdir(parents=True, exist_ok=True)

    data = []
    for label, report, series, color, lw in SERIES:
        r = load_returns(report, series)
        if r.empty:
            print(f"  (skipping {label}: not found)")
            continue
        data.append({"label": label, "color": color, "lw": lw, "rets": r, "metrics": summarize(r)})

    # ---- Equity curves ----
    fig, ax = plt.subplots(figsize=(15, 8))
    for d in data:
        eq = (1.0 + d["rets"]).cumprod()
        m = d["metrics"]
        label = (f"{d['label']}  "
                 f"(Sh={m['sharpe']:.2f}, CAGR={m['cagr']*100:.1f}%, "
                 f"MaxDD={m['max_dd']*100:.1f}%)")
        ax.plot(eq.index, eq.values, label=label, color=d["color"], lw=d["lw"])
    ax.set_yscale("log")
    ax.set_title("Paragon vs classical baselines — full lineage\n"
                 "16 mega-cap tech, walk-forward backtest, 2008-2020 and 2008-2024",
                 fontsize=13)
    ax.set_ylabel("Cumulative wealth (log scale, start = $1)")
    ax.grid(True, alpha=0.3, which="both")
    ax.legend(loc="upper left", fontsize=7.5, framealpha=0.92)
    fig.tight_layout()
    out_eq = out / "equity_curves.png"
    fig.savefig(out_eq, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_eq}")

    # ---- Sharpe / Calmar bars ----
    names = [d["label"] for d in data]
    sharpes = [d["metrics"]["sharpe"] for d in data]
    calmars = [d["metrics"]["calmar"] for d in data]
    colors = [d["color"] for d in data]

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
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
    axes[1].set_title("Calmar ratio (CAGR / |MaxDD|)")
    axes[1].grid(True, alpha=0.3, axis="x")
    for i, v in enumerate(calmars):
        axes[1].text(v + 0.01, i, f"{v:.2f}", va="center", fontsize=9)
    fig.tight_layout()
    out_bar = out / "sharpe_calmar.png"
    fig.savefig(out_bar, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_bar}")

    # ---- Table ----
    print()
    print(f"  {'Strategy':<52s}  {'CAGR':>7s}  {'Vol':>7s}  {'Sharpe':>7s}  {'MaxDD':>8s}  {'Calmar':>7s}")
    print("  " + "-" * 110)
    for d in data:
        m = d["metrics"]
        print(f"  {d['label']:<52s}  "
              f"{m['cagr']*100:6.2f}%  {m['vol']*100:6.2f}%  "
              f"{m['sharpe']:7.2f}  {m['max_dd']*100:7.2f}%  {m['calmar']:7.2f}")


if __name__ == "__main__":
    main()
