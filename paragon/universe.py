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

# v5 multi-asset+ universe (38 names across 7 buckets). Adds full sector
# rotation (XLU/XLP/XLV/XLE/XLF/XLK/XLY/XLI/XLB) and factor ETFs (USMV/MTUM/
# QUAL/VLUE) on top of the v3 multi-asset base. Sector rotation is the
# textbook regime story (utilities/staples lead late cycle, cyclicals lead
# early cycle) and factor ETFs encode the systematic style premia that
# regime-aware models should be able to time. Factor ETFs IPO late
# (USMV 2011, MTUM/QUAL/VLUE 2013) — universe mask handles staggered entry.
MULTI_ASSET_PLUS_UNIVERSE: list[str] = [
    # Mega-cap tech single names (8)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ORCL", "CSCO", "INTC",
    # Diverse single names (5)
    "JPM", "JNJ", "XOM", "PG", "WMT",
    # Sector SPDR ETFs (9) — covers the full GICS sector wheel except real
    # estate (handled separately by VNQ) and communications (handled by tech
    # singles + QQQ overlap).
    "XLU",   # utilities
    "XLP",   # consumer staples
    "XLV",   # health care
    "XLE",   # energy
    "XLF",   # financials
    "XLK",   # technology
    "XLY",   # consumer discretionary
    "XLI",   # industrials
    "XLB",   # materials
    # Factor / style ETFs (4) — regime-rotating style premia
    "USMV",  # MSCI USA min vol
    "MTUM",  # USA momentum
    "QUAL",  # USA quality
    "VLUE",  # USA value
    # Broad equity ETFs (4)
    "QQQ", "IWM", "EFA", "EEM",
    # Bond ETFs (5)
    "TLT", "IEF", "LQD", "HYG", "SHY",
    # Real assets (3)
    "GLD", "USO", "VNQ",
]


# v3 multi-asset universe (24 names) — kept for reproducing earlier runs.
MULTI_ASSET_UNIVERSE: list[str] = [
    # Mega-cap tech (8)
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ORCL", "CSCO", "INTC",
    # Diversifying single names across sectors (6)
    "JPM",   # financials
    "JNJ",   # healthcare
    "XOM",   # energy
    "PG",    # consumer staples
    "KO",    # consumer staples
    "WMT",   # consumer staples / retail
    # Equity ETFs (4) — non-overlapping with the tech basket
    "QQQ",   # Nasdaq 100
    "IWM",   # Russell 2000 (small-cap)
    "EFA",   # MSCI EAFE (intl developed)
    "EEM",   # MSCI emerging markets
    # Bond ETFs (4) — the key diversifier; TLT-equity correlation goes deeply
    # negative in stress (flight-to-quality), positive in normal regimes.
    "TLT",   # 20+ year Treasury
    "IEF",   # 7-10 year Treasury
    "LQD",   # IG corporate bond
    "HYG",   # High yield bond
    # Real assets (2)
    "GLD",   # Gold
    "USO",   # WTI crude oil
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

    Tickers that are entirely absent from `prices` (e.g. FB before 2012,
    factor ETFs before 2011-2013) are inserted as all-NaN columns — their
    mask will be permanently False, so the Transformer's key-padding mask
    just ignores them. This keeps the universe spec stable across runs even
    when the underlying date range excludes some assets.
    """
    missing = [tk for tk in tickers if tk not in prices.columns]
    px = prices.reindex(columns=tickers).copy()  # missing -> NaN columns
    if missing:
        from .utils.logging import get_logger
        get_logger(__name__).info(
            "Universe: %d tickers absent from price frame -> NaN-filled (never tradable): %s",
            len(missing), missing,
        )

    # Cumulative count of non-NaN per column.
    valid = px.notna().astype(int)
    cum = valid.cumsum(axis=0)
    mask = cum >= min_history
    return UniverseState(prices=px, mask=mask)
