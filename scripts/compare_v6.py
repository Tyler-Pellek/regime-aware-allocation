"""Aggregate the 8 v6 experiment summaries into one comparison table.

For each experiment, prints headline metrics for the strategy AND its
sample_cov_cvar baseline (apples-to-apples). At the bottom, prints a
cross-experiment ranking on multiple metrics.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

V6_NAMES = [
    "01_tech_baseline",
    "02_tech_warm",
    "03_tech_warm_factor",
    "04_tech_warm_factor_daily",
    "05_multi_baseline",
    "06_multi_warm",
    "07_multi_warm_factor",
    "08_multi_warm_factor_daily",
]


def load_summary(name: str) -> dict | None:
    path = Path("artifacts/reports") / f"v6_{name}" / "summary.json"
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def fmt_pct(x: float, w: int = 6) -> str:
    if x is None or pd.isna(x):
        return " " * w
    return f"{x*100:{w}.2f}%"


def fmt_num(x: float, w: int = 5, d: int = 2) -> str:
    if x is None or pd.isna(x):
        return " " * w
    return f"{x:{w}.{d}f}"


def main() -> None:
    rows = []
    for name in V6_NAMES:
        s = load_summary(name)
        if s is None:
            print(f"  (no summary for {name})")
            continue
        strat = s.get("strategy", {})
        base = s.get("sample_cov_cvar", {})
        ew = s.get("equal_weight", {})
        bench = s.get("benchmark", {})

        rows.append({
            "exp":            name,
            "S_CAGR":         strat.get("cagr"),
            "S_Vol":          strat.get("ann_vol"),
            "S_Sharpe":       strat.get("sharpe"),
            "S_MaxDD":        strat.get("max_drawdown"),
            "S_Calmar":       strat.get("calmar"),
            "Base_CAGR":      base.get("cagr"),
            "Base_Sharpe":    base.get("sharpe"),
            "Base_MaxDD":     base.get("max_drawdown"),
            "Edge_CAGR_pp":   (strat.get("cagr", 0) - base.get("cagr", 0)) * 100 if base else None,
            "Edge_Sharpe":    (strat.get("sharpe", 0) - base.get("sharpe", 0)) if base else None,
            "EW_CAGR":        ew.get("cagr"),
            "EW_Sharpe":      ew.get("sharpe"),
            "SPY_CAGR":       bench.get("cagr"),
            "SPY_Sharpe":     bench.get("sharpe"),
        })

    if not rows:
        print("No v6 results found.")
        return

    df = pd.DataFrame(rows).set_index("exp")

    print()
    print("=" * 110)
    print("  v6 SWEEP RESULTS — strategy vs sample_cov_cvar baseline (apples-to-apples)")
    print("=" * 110)
    print(f"  {'experiment':<28s}  {'CAGR':>7s}  {'Vol':>7s}  {'Sharpe':>6s}  {'MaxDD':>7s}  {'Calmar':>6s}  "
          f"{'BaseSh':>6s}  {'EdgeSh':>7s}  {'EW Sh':>6s}")
    print("-" * 110)
    for name, r in df.iterrows():
        print(f"  {name:<28s}  "
              f"{fmt_pct(r['S_CAGR'])}  {fmt_pct(r['S_Vol'])}  {fmt_num(r['S_Sharpe'])}  "
              f"{fmt_pct(r['S_MaxDD'])}  {fmt_num(r['S_Calmar'])}  "
              f"{fmt_num(r['Base_Sharpe'])}  {fmt_num(r['Edge_Sharpe'], w=6, d=3)}  "
              f"{fmt_num(r['EW_Sharpe'])}")
    print("-" * 110)

    # Rankings on the key metrics
    print()
    print("Rankings (top-3 each metric):")
    for metric, label, higher_better in [
        ("S_Sharpe", "Strategy Sharpe", True),
        ("S_CAGR",   "Strategy CAGR",   True),
        ("S_Calmar", "Strategy Calmar", True),
        ("S_MaxDD",  "Strategy MaxDD (least negative)", True),
        ("Edge_Sharpe", "Edge vs sample_cov baseline (Sharpe)", True),
        ("Edge_CAGR_pp", "Edge vs sample_cov baseline (CAGR pp)", True),
    ]:
        ranked = df[[metric]].dropna().sort_values(metric, ascending=not higher_better)
        print(f"  {label}:")
        for i, (idx, row) in enumerate(ranked.head(3).iterrows(), 1):
            v = row[metric]
            print(f"    {i}. {idx:<28s}  {v:+.4f}")

    # Save the full DataFrame as CSV
    out_csv = Path("artifacts/reports/v6_comparison.csv")
    df.to_csv(out_csv)
    print()
    print(f"Full comparison CSV: {out_csv}")


if __name__ == "__main__":
    main()
