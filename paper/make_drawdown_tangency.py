"""Generate a drawdown figure: strategy vs long-only tangency baseline.

Output:
  fig_drawdown_tangency.pdf   — for the slides
  fig_drawdown_tangency.png   — preview
"""
from __future__ import annotations
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
    "axes.axisbelow":     True,
    "lines.linewidth":    1.2,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
    "pdf.fonttype":       42,
})

COLORS = {
    "strategy":   "#111111",
    "max_sharpe": "#0072B2",
}
LABELS = {
    "strategy":   "Strategy (production)",
    "max_sharpe": "Long-only tangency baseline",
}

REPORT = Path("/Users/tylerpellek/ParagonProject/artifacts/reports/v7b_extended")
OUT = Path("/Users/tylerpellek/ParagonProject/paper")

def load(name: str) -> pd.Series:
    p = REPORT / ("strategy_returns.csv" if name == "strategy"
                  else f"baseline_{name}_returns.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True).iloc[:, 0]

fig, ax = plt.subplots(figsize=(6.5, 3.0))
for name, lw in [("strategy", 1.6), ("max_sharpe", 1.0)]:
    r = load(name)
    eq = (1 + r).cumprod()
    dd = (eq - eq.cummax()) / eq.cummax()
    ax.fill_between(dd.index, dd.values, 0,
                    color=COLORS[name],
                    alpha=0.30 if name == "strategy" else 0.18,
                    linewidth=0)
    ax.plot(dd.index, dd.values, color=COLORS[name], lw=lw, label=LABELS[name])
    print(f"{name:12s}  MaxDD={dd.min()*100:6.2f}%  on {dd.idxmin().date()}")
ax.set_ylabel("Drawdown")
ax.set_xlabel("Date")
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(loc="lower right", framealpha=0.9, edgecolor="0.7", fancybox=False)
fig.savefig(OUT / "fig_drawdown_tangency.pdf")
fig.savefig(OUT / "fig_drawdown_tangency.png", dpi=200)
plt.close(fig)
print("wrote fig_drawdown_tangency.pdf and .png")
