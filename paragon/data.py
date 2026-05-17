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


def load_kaggle_ohlcv(
    tickers: Sequence[str],
    kaggle_root: str | Path,
    start: str | pd.Timestamp | None = None,
    end: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Load FULL OHLCV per ticker (not just adjusted close).

    Returns a dict mapping ticker -> DataFrame with columns
    [Open, High, Low, Close, Adj Close, Volume], business-day indexed and
    inside-lifespan ffilled (same logic as `load_kaggle_prices`). Missing
    tickers are skipped silently with a warning. Used by v7+ feature
    engineering for OHLCV-derived signals (intraday range, volume z-score,
    overnight gap, etc.).
    """
    paths = KagglePaths(Path(kaggle_root))
    out: dict[str, pd.DataFrame] = {}
    missing: list[str] = []
    for tk in tickers:
        p = paths.ticker_csv(tk)
        if p is None:
            missing.append(tk)
            continue
        df = _read_kaggle_csv(p)
        if start is not None:
            df = df.loc[df.index >= pd.Timestamp(start)]
        if end is not None:
            df = df.loc[df.index <= pd.Timestamp(end)]
        if df.empty:
            missing.append(tk)
            continue
        out[tk] = df
    if missing:
        LOG.warning("Kaggle OHLCV missing or empty for: %s", missing)
    if not out:
        raise FileNotFoundError(f"No Kaggle CSVs found under {paths.root}.")

    # Reindex each frame to a common business-day index covering all tickers
    # and ffill within each ticker's lifespan.
    all_idx = sorted({d for df in out.values() for d in df.index})
    if not all_idx:
        return out
    bdays = pd.bdate_range(all_idx[0], all_idx[-1])
    fixed = {}
    for tk, df in out.items():
        df2 = df.reindex(bdays)
        # ffill inside lifespan only (columns)
        for col in df2.columns:
            valid = df2[col].dropna()
            if valid.empty:
                continue
            first, last = valid.index.min(), valid.index.max()
            inside = (df2.index >= first) & (df2.index <= last)
            df2.loc[inside, col] = df2.loc[inside, col].ffill()
        df2.index.name = "Date"
        fixed[tk] = df2
    return fixed


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

# Extended macro set adding the yield curve. Used by v7+ configs. The yield
# curve (esp. 10y-3m spread) has preceded every US recession since 1955 —
# powerful regime input the original DEFAULT set was missing.
EXTENDED_MACRO_SYMBOLS: dict[str, str] = {
    **DEFAULT_MACRO_SYMBOLS,
    "IRX": "^IRX",       # 13-week T-bill yield
    "TYX": "^TYX",       # 30-year Treasury yield
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
        # Only honor the cache if it actually covers the requested range AND
        # contains every requested symbol — otherwise refetch (a narrow earlier
        # cache silently clips a wider new request, and a cache missing
        # newly-added symbols silently drops them).
        covers_start = cached.index.min() <= start_ts
        covers_end = end_ts is None or cached.index.max() >= end_ts
        covers_symbols = set(symbols.keys()).issubset(set(cached.columns))
        if covers_start and covers_end and covers_symbols:
            LOG.info("Loading macro cache from %s", cache_path)
            return cached
        if not covers_symbols:
            missing = set(symbols.keys()) - set(cached.columns)
            LOG.info("Macro cache missing symbols %s — refetching.", sorted(missing))
        else:
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
# yfinance extension (for dates after Kaggle's April-2020 cutoff)
# --------------------------------------------------------------------------- #

KAGGLE_CUTOFF = pd.Timestamp("2020-04-01")


def load_yfinance_ohlcv(
    tickers: Sequence[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    cache_path: str | Path | None = None,
    refresh: bool = False,
) -> dict[str, pd.DataFrame]:
    """Fetch full OHLCV from yfinance for tickers (slow but only run for the
    post-Kaggle extension window). Cached as a single parquet file with a
    MultiIndex (ticker, date)."""
    import yfinance as yf

    cache_path = Path(cache_path) if cache_path else None
    if cache_path and cache_path.is_file() and not refresh:
        LOG.info("Loading yfinance extension cache from %s", cache_path)
        big = pd.read_parquet(cache_path)
        out: dict[str, pd.DataFrame] = {}
        for tk in tickers:
            if tk in big.columns.get_level_values(0):
                out[tk] = big[tk]
        return out

    LOG.info("Fetching yfinance OHLCV for %d tickers (post-Kaggle extension) ...", len(tickers))
    # yfinance wants DATE-only strings; passing 'YYYY-MM-DD 00:00:00' raises
    # ValueError: unconverted data remains. Format defensively.
    start_str = pd.Timestamp(start).strftime("%Y-%m-%d")
    end_str = pd.Timestamp(end).strftime("%Y-%m-%d") if end is not None else None
    out = {}
    for tk in tickers:
        try:
            df = yf.download(
                tk, start=start_str, end=end_str,
                progress=False, auto_adjust=False, threads=False,
            )
        except Exception as exc:                       # pragma: no cover
            LOG.warning("yfinance failed for %s: %s", tk, exc)
            continue
        if df is None or df.empty:
            continue
        # yfinance returns MultiIndex columns when one ticker; flatten.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        keep = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        df = df[keep].dropna(subset=["Adj Close"])
        df.index = pd.DatetimeIndex(df.index).tz_localize(None)
        out[tk] = df

    if cache_path and out:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        big = pd.concat(out, axis=1)
        big.to_parquet(cache_path)
        LOG.info("Cached yfinance extension to %s", cache_path)
    return out


def splice_ohlcv(
    kaggle_dict: dict[str, pd.DataFrame],
    yf_dict: dict[str, pd.DataFrame],
    splice_date: pd.Timestamp = KAGGLE_CUTOFF,
) -> dict[str, pd.DataFrame]:
    """Stitch the Kaggle (pre-splice) and yfinance (post-splice) OHLCV frames
    per ticker. Applies a ratio adjustment to the post-splice Adj Close so the
    series stays continuous across the boundary (Kaggle and yfinance use
    slightly different dividend/split anchoring).
    """
    out: dict[str, pd.DataFrame] = {}
    for tk in set(kaggle_dict.keys()) | set(yf_dict.keys()):
        k = kaggle_dict.get(tk)
        y = yf_dict.get(tk)
        if k is None:
            out[tk] = y
            continue
        if y is None:
            out[tk] = k
            continue
        # Find the splice anchor date in both
        k_pre = k.loc[k.index < splice_date]
        y_post = y.loc[y.index >= splice_date]
        if k_pre.empty or y_post.empty:
            out[tk] = k_pre if not k_pre.empty else y_post
            continue
        # Anchor on the last common-or-nearest Kaggle date <= splice
        k_anchor_idx = k_pre.index.max()
        # Find yfinance row near k_anchor_idx for ratio
        y_overlap = y.loc[(y.index >= k_anchor_idx) & (y.index <= splice_date)]
        if y_overlap.empty:
            ratio = 1.0
        else:
            y_anchor = y_overlap.iloc[0]
            k_anchor = k_pre.loc[k_anchor_idx]
            if y_anchor["Adj Close"] > 0:
                ratio = float(k_anchor["Adj Close"]) / float(y_anchor["Adj Close"])
            else:
                ratio = 1.0
        # Apply ratio to all price columns in y_post (volume unscaled).
        y_scaled = y_post.copy()
        for col in ["Open", "High", "Low", "Close", "Adj Close"]:
            if col in y_scaled.columns:
                y_scaled[col] = y_scaled[col] * ratio
        out[tk] = pd.concat([k_pre, y_scaled], axis=0).sort_index()
        out[tk] = out[tk][~out[tk].index.duplicated(keep="last")]
    return out


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
