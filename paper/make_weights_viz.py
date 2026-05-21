"""Generate portfolio-weights-over-time visualizations.

Outputs in /Users/tylerpellek/ParagonProject/paper/:
  fig_weights_area.pdf   — stacked-area plot (best for talks)
  fig_weights_area.png   — same, raster version for slides
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
    "legend.fontsize":    7,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.grid":          True,
    "grid.linewidth":     0.4,
    "grid.alpha":         0.35,
    "axes.axisbelow":     True,
    "savefig.bbox":       "tight",
    "savefig.dpi":        300,
    "pdf.fonttype":       42,
})

REPORT = Path("/Users/tylerpellek/ParagonProject/artifacts/reports/v7b_extended")
OUT = Path("/Users/tylerpellek/ParagonProject/paper")
OUT.mkdir(parents=True, exist_ok=True)

W_raw = pd.read_csv(REPORT / "weights.csv", index_col=0, parse_dates=True)

# Smooth with a 13-rebalance (~quarterly) rolling mean to suppress weekly
# rotation and surface only regime-level shifts in allocation.
W = W_raw.rolling(window=13, min_periods=1).mean()

# Order assets by total cumulative weight (largest at the bottom of the stack).
order = W.sum(axis=0).sort_values(ascending=False).index.tolist()
W = W[order]

# tab20 colors are designed for categorical legibility at >10 categories.
cmap = mpl.colormaps["tab20"]
colors = [cmap(i / 20) for i in range(len(W.columns))]

# --------- Figure 1: smoothed stacked area ---------
fig, ax = plt.subplots(figsize=(10.0, 5.5))
ax.stackplot(
    W.index, W.T.values,
    labels=W.columns, colors=colors,
    alpha=0.95, edgecolor="white", linewidth=0.0,
)
ax.set_ylabel("Portfolio weight")
ax.set_xlabel("Date")
ax.set_ylim(0, 1)
ax.set_xlim(W.index.min(), W.index.max())
ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax.legend(
    loc="center left", bbox_to_anchor=(1.01, 0.5),
    ncol=1, frameon=False, fontsize=9, handlelength=1.8,
)
ax.set_title("Portfolio Weights Through Time (13-week smoothed)")
fig.savefig(OUT / "fig_weights_area.pdf")
fig.savefig(OUT / "fig_weights_area.png", dpi=200)
plt.close(fig)
print("wrote fig_weights_area.pdf and .png")

# --------- Figure 3: annual-average heatmap ---------
W_ann = W_raw.resample("YE").mean()
W_ann.index = W_ann.index.year

# Order assets by overall average weight (largest at top).
overall_order = W_raw.mean(axis=0).sort_values(ascending=False).index.tolist()
W_ann = W_ann[overall_order]

fig3, ax3 = plt.subplots(figsize=(8.5, 5.5))
im = ax3.imshow(
    W_ann.T.values, aspect="auto", cmap="viridis",
    vmin=0, vmax=W_ann.values.max(),
    interpolation="nearest",
)
ax3.set_yticks(np.arange(len(overall_order)))
ax3.set_yticklabels(overall_order, fontsize=9)
ax3.set_xticks(np.arange(len(W_ann.index)))
ax3.set_xticklabels(W_ann.index, rotation=45, ha="right", fontsize=8)
ax3.set_xlabel("Year")
ax3.set_title("Average Annual Portfolio Weights")
cbar = fig3.colorbar(im, ax=ax3, fraction=0.04, pad=0.02)
cbar.ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
cbar.set_label("Average weight", fontsize=9)
# annotate each cell
for i in range(W_ann.shape[1]):
    for j in range(W_ann.shape[0]):
        v = W_ann.values[j, i]
        if v > 0.005:
            ax3.text(j, i, f"{v*100:.0f}", ha="center", va="center",
                     fontsize=7, color="white" if v > 0.08 else "black")
ax3.grid(False)
fig3.savefig(OUT / "fig_weights_heatmap.pdf")
fig3.savefig(OUT / "fig_weights_heatmap.png", dpi=200)
plt.close(fig3)
print("wrote fig_weights_heatmap.pdf and .png")

# --------- Figure 2: top-6 per-asset lines ---------
# Cleaner alternative — show the 6 largest-average-weight names as lines.
top6 = W_raw.mean(axis=0).sort_values(ascending=False).head(6).index.tolist()
W6 = W_raw[top6].rolling(window=13, min_periods=1).mean()

fig2, ax2 = plt.subplots(figsize=(9.0, 4.5))
for i, t in enumerate(top6):
    ax2.plot(W6.index, W6[t].values, label=t, color=cmap(i / 20), lw=1.6)
ax2.set_ylabel("Portfolio weight")
ax2.set_xlabel("Date")
ax2.set_xlim(W6.index.min(), W6.index.max())
ax2.set_ylim(0, ax2.get_ylim()[1])
ax2.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
ax2.legend(loc="upper right", ncol=3, frameon=True, framealpha=0.9,
           edgecolor="0.8", fancybox=False, fontsize=9)
ax2.set_title("Top 6 Holdings by Average Weight (13-week smoothed)")
fig2.savefig(OUT / "fig_weights_lines.pdf")
fig2.savefig(OUT / "fig_weights_lines.png", dpi=200)
plt.close(fig2)
print("wrote fig_weights_lines.pdf and .png")

# Also report some descriptive statistics for the talk
n_active = (W > 1e-4).sum(axis=1)
print(f"average active positions: {n_active.mean():.2f}")
print(f"max active positions:     {n_active.max()}")
print(f"min active positions:     {n_active.min()}")
print(f"avg max single weight:    {W.max(axis=1).mean():.3f}")
print()
print("average weight per ticker (sorted):")
avg = W.mean(axis=0).sort_values(ascending=False)
for tkr, w in avg.items():
    print(f"  {tkr:6s} {w*100:5.2f}%")
