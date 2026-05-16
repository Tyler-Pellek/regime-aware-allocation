"""Universe definition and dynamic availability masking.

The mega-cap tech universe is hard-coded here (defined by the user spec). Some
names IPO mid-backtest (META=2012, TSLA=2010); the universe handler emits a
boolean availability matrix that the Transformer's key-padding mask consumes,
so an asset is invisible to attention until it has enough history.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# 16 mega-cap tech / FAANG-adjacent. Order is canonical and used everywhere.
# Note: the Kaggle dataset is dated April 2020, so Facebook is "FB" (the rename
# to META happened in 2021).
TECH_UNIVERSE: list[str] = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB",   "NFLX", "NVDA", "TSLA",
    "ADBE", "CRM",  "ORCL",  "INTC", "CSCO", "QCOM", "IBM",  "TXN",
]

# SPY = benchmark; not part of the trading universe, used in eval.
BENCHMARK = "SPY"


@dataclass
class UniverseState:
    """Wraps a price frame plus a parallel boolean mask of the same shape.

    `mask[t, i] = True` means asset i is tradable on date t. This is True iff
    asset i has at least `min_history` valid prices ending at (and including)
    date t.
    """
    prices: pd.DataFrame                # T x N adjusted close
    mask: pd.DataFrame                  # T x N bool

    @property
    def tickers(self) -> list[str]:
        return list(self.prices.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.prices.index

    def n_assets_on(self, date: pd.Timestamp) -> int:
        return int(self.mask.loc[date].sum())


def build_universe(
    prices: pd.DataFrame,
    tickers: list[str] = TECH_UNIVERSE,
    min_history: int = 60,
) -> UniverseState:
    """Filter `prices` to `tickers` (preserving order) and build availability mask.

    Asset i is marked tradable on date t iff prices.iloc[:t+1, i] has at least
    `min_history` non-NaN observations.
    """
    # Reorder + select; missing tickers raise (we want to fail loudly).
    missing = [tk for tk in tickers if tk not in prices.columns]
    if missing:
        raise KeyError(f"Tickers not in price frame: {missing}")
    px = prices[tickers].copy()

    # Cumulative count of non-NaN per column.
    valid = px.notna().astype(int)
    cum = valid.cumsum(axis=0)
    mask = cum >= min_history
    return UniverseState(prices=px, mask=mask)
