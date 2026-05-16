"""Compose a human-readable report from a backtest's artifacts."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .metrics import summarize
from .plots import (
    drawdown,
    equity_curves,
    regime_overlay,
    rolling_sharpe,
    turnover_plot,
    weights_heatmap,
)


def write_report(
    out_dir: str | Path,
    strategy_returns: pd.Series,
    weights: pd.DataFrame,
    decisions: pd.DataFrame,
    regime_probs: pd.DataFrame,
    benchmark_returns: pd.Series | None = None,
    baselines: dict[str, pd.Series] | None = None,
    rebal_freq_days: int = 5,
) -> dict:
    """Write all artifacts (CSVs + PNGs + summary.json) to `out_dir`."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(exist_ok=True)

    # Summary metrics
    summary = {
        "strategy": summarize(
            strategy_returns, bench=benchmark_returns, weights=weights, rebal_freq_days=rebal_freq_days,
        ),
    }
    if baselines:
        for name, r in baselines.items():
            summary[name] = summarize(r, bench=benchmark_returns, rebal_freq_days=rebal_freq_days)
    if benchmark_returns is not None:
        summary["benchmark"] = summarize(benchmark_returns, rebal_freq_days=rebal_freq_days)

    # Plots
    curves = {"strategy": strategy_returns}
    if baselines:
        curves.update(baselines)
    if benchmark_returns is not None:
        curves["benchmark (SPY)"] = benchmark_returns
    equity_curves(curves, fig_dir / "equity_curves.png")
    drawdown(strategy_returns, fig_dir / "drawdown.png", label="Paragon")
    regime_overlay(strategy_returns, regime_probs, fig_dir / "regime_overlay.png")
    weights_heatmap(weights, fig_dir / "weights_heatmap.png")
    rolling_sharpe(strategy_returns, fig_dir / "rolling_sharpe.png")
    turnover_plot(decisions, fig_dir / "turnover.png")

    # CSVs
    strategy_returns.to_csv(out_dir / "strategy_returns.csv", header=["ret"])
    weights.to_csv(out_dir / "weights.csv")
    decisions.to_csv(out_dir / "decisions.csv")
    if benchmark_returns is not None:
        benchmark_returns.to_csv(out_dir / "benchmark_returns.csv", header=["ret"])
    if baselines:
        for name, r in baselines.items():
            r.to_csv(out_dir / f"baseline_{name}_returns.csv", header=["ret"])

    with (out_dir / "summary.json").open("w") as fh:
        json.dump(summary, fh, indent=2, default=float)

    # Pretty-printed text summary
    lines = ["# Paragon backtest summary\n"]
    for name, m in summary.items():
        lines.append(f"\n## {name}\n")
        for k, v in m.items():
            lines.append(f"- {k}: {v:.4f}\n" if isinstance(v, float) else f"- {k}: {v}\n")
    (out_dir / "summary.md").write_text("".join(lines))

    return summary
