"""Data layer.

Two sources are stitched together:

1. **Kaggle stock-market-dataset** (Jackson Crow). Per-ticker OHLCV CSVs in
   `data/raw/kaggle_stock_market/stocks/<TICKER>.csv` and `etfs/<TICKER>.csv`.
   Columns: Date, Open, High, Low, Close, Adj Close, Volume.
   Dataset snapshot is dated April 2020.

2. **yfinance macro indicators** (VIX, ^TNX, ^GSPC, DXY, etc.) — fetched once
   and cached to parquet.

The loader produces an aligned, business-day-indexed wide frame of adjusted
close prices for the requested universe + a parallel macro frame.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

from .utils.logging import get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Kaggle per-ticker CSVs
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class KagglePaths:
    root: Path

    @property
    def stocks(self) -> Path:
        return self.root / "stocks"

    @property
    def etfs(self) -> Path:
        return self.root / "etfs"

    def ticker_csv(self, ticker: str) -> Path | None:
        """Return the CSV path for a ticker, searching stocks/ then etfs/.
        Returns None if not present."""
        candidates = [self.stocks / f"{ticker}.csv", self.etfs / f"{ticker}.csv"]
        for p in candidates:
            if p.is_file():
                return p
        return None


def _read_kaggle_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    df = df.sort_values("Date").drop_duplicates(subset="Date").set_index("Date")
    # Some rows in the Kaggle dump have NaN Adj Close; drop them.
    df = df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]].dropna(subset=["Adj Close"])
    return df


def load_kaggle_prices(
    tickers: Sequence[str],
    kaggle_root: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
    field: str = "Adj Close",
) -> pd.DataFrame:
    """Load a wide DataFrame: index=DatetimeIndex (business days), columns=tickers.

    Missing tickers are skipped with a warning. Date alignment uses an outer
    union of all tickers' available dates, then forward-fills only inside each
    column's natural range (so a ticker that didn't exist yet stays NaN — this
    is what the universe mask in `paragon.universe` consumes).
    """
    paths = KagglePaths(Path(kaggle_root))
    series: dict[str, pd.Series] = {}
    missing: list[str] = []

    for tk in tickers:
        p = paths.ticker_csv(tk)
        if p is None:
            missing.append(tk)
            continue
        df = _read_kaggle_csv(p)
        s = df[field].rename(tk)
        if start is not None:
            s = s.loc[s.index >= pd.Timestamp(start)]
        if end is not None:
            s = s.loc[s.index <= pd.Timestamp(end)]
        if s.empty:
            missing.append(tk)
            continue
        series[tk] = s

    if missing:
        LOG.warning("Kaggle CSV missing or empty for: %s", missing)
    if not series:
        raise FileNotFoundError(
            f"No Kaggle CSVs found under {paths.root}. Run scripts/download_kaggle.sh first."
        )

    wide = pd.concat(series.values(), axis=1).sort_index()
    # Restrict to business days.
    bdays = pd.bdate_range(wide.index.min(), wide.index.max())
    wide = wide.reindex(bdays)
    # Inside each column's lifespan, ffill small gaps (holidays, missing print
    # days). Outside the lifespan, leave NaN so universe masking works.
    wide = wide.apply(_ffill_within_lifespan, axis=0)
    wide.index.name = "Date"
    return wide


def _ffill_within_lifespan(s: pd.Series) -> pd.Series:
    valid = s.dropna()
    if valid.empty:
        return s
    first, last = valid.index.min(), valid.index.max()
    inside = (s.index >= first) & (s.index <= last)
    out = s.copy()
    out.loc[inside] = out.loc[inside].ffill()
    return out


# --------------------------------------------------------------------------- #
# yfinance macro
# --------------------------------------------------------------------------- #

# Symbols we pull. Names must be valid yfinance symbols.
DEFAULT_MACRO_SYMBOLS: dict[str, str] = {
    "VIX": "^VIX",       # implied vol
    "TNX": "^TNX",       # 10y Treasury yield (in %, *10 -> tenths of pct)
    "DXY": "DX-Y.NYB",   # US Dollar Index
    "SPX": "^GSPC",      # S&P 500
    "OIL": "CL=F",       # WTI crude futures
    "GOLD": "GC=F",      # gold futures
    "HYG": "HYG",        # high yield credit ETF (credit spread proxy)
    "TLT": "TLT",        # long-duration Treasury ETF
}


def fetch_macro(
    symbols: dict[str, str] | None = None,
    start: str | pd.Timestamp = "2000-01-01",
    end: str | pd.Timestamp | None = None,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    """Fetch macro series via yfinance, cached as parquet.

    Returns a DataFrame indexed by business days; each column is the close
    price (or yield) for the named macro series.
    """
    symbols = symbols or DEFAULT_MACRO_SYMBOLS
    cache_path = Path(cache_path) if cache_path else None
    start_ts = pd.Timestamp(start)
    end_ts = pd.Timestamp(end) if end else None
    if cache_path and cache_path.is_file() and not refresh:
        cached = pd.read_parquet(cache_path)
        # Only honor the cache if it actually covers the requested range —
        # otherwise a narrow earlier cache silently clips a wider new request.
        covers_start = cached.index.min() <= start_ts
        covers_end = end_ts is None or cached.index.max() >= end_ts
        if covers_start and covers_end:
            LOG.info("Loading macro cache from %s", cache_path)
            return cached
        LOG.info(
            "Macro cache range [%s, %s] does not cover requested [%s, %s] — refetching.",
            cached.index.min().date(), cached.index.max().date(), start_ts.date(),
            end_ts.date() if end_ts is not None else "now",
        )

    import yfinance as yf

    LOG.info("Fetching %d macro symbols from yfinance ...", len(symbols))
    cols: dict[str, pd.Series] = {}
    for name, sym in symbols.items():
        try:
            df = yf.download(
                sym, start=str(start), end=str(end) if end else None,
                progress=False, auto_adjust=False, threads=False,
            )
        except Exception as exc:  # pragma: no cover - network
            LOG.warning("yfinance failed for %s (%s): %s", name, sym, exc)
            continue
        if df is None or df.empty:
            LOG.warning("Empty yfinance response for %s (%s)", name, sym)
            continue
        # yfinance returns MultiIndex columns when multiple tickers. We pass one.
        col = df["Close"] if "Close" in df.columns else df.iloc[:, 0]
        if isinstance(col, pd.DataFrame):
            col = col.iloc[:, 0]
        cols[name] = col.rename(name)

    if not cols:
        raise RuntimeError("No macro symbols fetched. Check network / yfinance.")

    macro = pd.concat(cols.values(), axis=1).sort_index()
    bdays = pd.bdate_range(macro.index.min(), macro.index.max())
    macro = macro.reindex(bdays).ffill()
    macro.index.name = "Date"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        macro.to_parquet(cache_path)
        LOG.info("Cached macro to %s", cache_path)
    return macro


# --------------------------------------------------------------------------- #
# Alignment
# --------------------------------------------------------------------------- #

def align_prices_macro(
    prices: pd.DataFrame, macro: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align both frames on the intersection of their business-day indices.

    Macro is forward-filled across price dates (e.g. on a price day where VIX
    has no print, carry the previous value)."""
    common = prices.index.intersection(macro.index)
    p = prices.loc[common]
    m = macro.reindex(common).ffill()
    return p, m
