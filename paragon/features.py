"""Feature engineering.

Two flavors of features:

1. **Asset-level** (per-token features that populate token positions 1..N):
   - sliding-window log returns of length W,
   - rolling-W realized volatility,
   - placeholder scalar for w_{t-1} (filled by the backtester at inference time).

2. **Context** (the [CTX] token at position 0):
   - HMM regime posterior probabilities (computed in `paragon.regime.hmm`),
   - macro indicators over the same lookback window:
       * VIX last value + Δlog VIX over W,
       * S&P 500 W-day return,
       * 10Y yield level + W-day Δ,
       * DXY W-day return,
       * HYG/TLT ratio W-day Δlog (credit-stress proxy).

This module produces aligned tensors but does NOT build the final input tensor —
that's the `paragon.model.input_builder` module. Here we just compute the raw
columns and serve them.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Returns / volatility
# --------------------------------------------------------------------------- #

def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """T x N log returns. First row is NaN."""
    return np.log(prices).diff()


def rolling_vol(returns: pd.DataFrame, window: int) -> pd.DataFrame:
    """T x N rolling std of log returns (sample std, ddof=0 for stability)."""
    return returns.rolling(window=window, min_periods=window).std(ddof=0)


def winsorize(df: pd.DataFrame, q: float = 0.005) -> pd.DataFrame:
    """Clip per-column at the (q, 1-q) quantiles. Used to tame extreme prints."""
    lo = df.quantile(q)
    hi = df.quantile(1.0 - q)
    return df.clip(lower=lo, upper=hi, axis=1)


# --------------------------------------------------------------------------- #
# Macro context features
# --------------------------------------------------------------------------- #

def build_macro_features(macro: pd.DataFrame, window: int) -> pd.DataFrame:
    """Build a small set of macro features aligned to the macro index.

    Output columns:
      - vix_level            : VIX / 100
      - vix_chg              : log(VIX_t / VIX_{t-W})
      - spx_ret_W            : SPX cumulative return over W days
      - tnx_level            : 10y yield / 100 (TNX is in tenths of %, /100 -> raw %)
      - tnx_chg_W            : (TNX_t - TNX_{t-W}) / 100
      - dxy_ret_W            : DXY W-day log return
      - oil_ret_W            : crude W-day log return
      - credit_spread_proxy  : -log(HYG/TLT), level (lower HYG vs TLT = wider spread)
      - credit_spread_chg_W  : W-day change of the above
    Any missing symbols are silently skipped (so the loader is robust to
    partial macro fetches).
    """
    cols: dict[str, pd.Series] = {}
    m = macro.copy()

    if "VIX" in m:
        cols["vix_level"] = m["VIX"] / 100.0
        cols["vix_chg"] = np.log(m["VIX"] / m["VIX"].shift(window))
    if "SPX" in m:
        cols["spx_ret_W"] = np.log(m["SPX"] / m["SPX"].shift(window))
    if "TNX" in m:
        cols["tnx_level"] = m["TNX"] / 100.0
        cols["tnx_chg_W"] = (m["TNX"] - m["TNX"].shift(window)) / 100.0
    if "DXY" in m:
        cols["dxy_ret_W"] = np.log(m["DXY"] / m["DXY"].shift(window))
    if "OIL" in m:
        cols["oil_ret_W"] = np.log(m["OIL"] / m["OIL"].shift(window))
    if "HYG" in m and "TLT" in m:
        spread = -np.log(m["HYG"] / m["TLT"])
        cols["credit_spread_proxy"] = spread
        cols["credit_spread_chg_W"] = spread - spread.shift(window)

    # ---- yield-curve features (v7+) ----
    # Note: yfinance returns yields in tenths of a percent (e.g. TNX=42.0 means
    # 4.20%), so divide by 100 to get raw percent. Spreads are then in percent.
    if "TNX" in m and "IRX" in m:
        cols["term_spread_10y_3m"] = (m["TNX"] - m["IRX"]) / 100.0
        cols["term_spread_10y_3m_chg_W"] = (
            (m["TNX"] - m["IRX"]) / 100.0
            - (m["TNX"].shift(window) - m["IRX"].shift(window)) / 100.0
        )
    if "TYX" in m and "TNX" in m:
        cols["term_spread_30y_10y"] = (m["TYX"] - m["TNX"]) / 100.0
        cols["term_spread_30y_10y_chg_W"] = (
            (m["TYX"] - m["TNX"]) / 100.0
            - (m["TYX"].shift(window) - m["TNX"].shift(window)) / 100.0
        )

    feats = pd.DataFrame(cols)
    return feats


# --------------------------------------------------------------------------- #
# Container
# --------------------------------------------------------------------------- #

@dataclass
class FeatureBundle:
    """All the pre-computed pieces a model snapshot needs.

    Indexes are aligned (same DatetimeIndex on `returns`, `vol`, `macro_feats`,
    and `regime_probs`).
    """
    returns: pd.DataFrame              # T x N log returns
    vol: pd.DataFrame                  # T x N rolling vol
    macro_feats: pd.DataFrame          # T x M_macro
    regime_probs: pd.DataFrame         # T x k regime posteriors
    mask: pd.DataFrame                 # T x N tradability bool
    ohlcv_feats: np.ndarray | None = None  # (T, N, F_ohlcv) per-asset OHLCV
                                           # features. None when not requested
                                           # so old configs train identically.
    ohlcv_feature_names: list[str] | None = None

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index

    @property
    def n_ohlcv_feats(self) -> int:
        return 0 if self.ohlcv_feats is None else self.ohlcv_feats.shape[2]


# --------------------------------------------------------------------------- #
# Per-asset OHLCV features (v7+)
# --------------------------------------------------------------------------- #

def build_ohlcv_features(
    ohlcv_dict: dict[str, pd.DataFrame],
    tickers: list[str],
    dates: pd.DatetimeIndex,
    short_window: int = 5,
    vol_z_window: int = 60,
    zscore_window: int = 252,
    normalize: bool = True,
) -> tuple[np.ndarray, list[str]]:
    """Compute per-asset OHLCV-derived features at each date.

    When ``normalize=True`` (v7b+ default), every feature is rolling-z-scored
    over a 252-day window so they all sit in roughly [-3, 3] regardless of
    natural scale. Without normalization (v7 original), vol_z dominated the
    input projection because its raw magnitude was ~10x bigger than the
    price-based features, drowning out the others. This was the cause of the
    v7 apples-to-apples regression vs v6_03.

    Returns
    -------
    feats : np.ndarray of shape (T, N, F_ohlcv)
        For tickers missing from `ohlcv_dict` (or with insufficient data on a
        given date), the corresponding row/column is zero-filled — the
        universe mask in FeatureBundle will gate these out anyway.
    names : list[str]
        Human-readable names of the F_ohlcv features (for diagnostics).
    """
    feature_names = [
        "hl_range_5d",          # mean of (High-Low)/Close over short_window
        "intraday_ret_5d",      # mean of (Close-Open)/Open over short_window
        "gap_ret_5d",           # mean of overnight gap (Open - prev Close)/prev Close
        "vol_z",                # (Volume_t - mean60d) / std60d
    ]
    T = len(dates)
    N = len(tickers)
    F = len(feature_names)
    out = np.zeros((T, N, F), dtype=np.float32)

    for i, tk in enumerate(tickers):
        if tk not in ohlcv_dict:
            continue
        df = ohlcv_dict[tk].reindex(dates)
        close = df["Close"]
        open_ = df["Open"]
        high = df["High"]
        low = df["Low"]
        vol = df["Volume"].astype(np.float64)

        hl_range = ((high - low) / close).rolling(short_window, min_periods=1).mean()
        intraday = ((close - open_) / open_).rolling(short_window, min_periods=1).mean()
        prev_close = close.shift(1)
        gap = ((open_ - prev_close) / prev_close).rolling(short_window, min_periods=1).mean()
        vol_mean = vol.rolling(vol_z_window, min_periods=10).mean()
        vol_std = vol.rolling(vol_z_window, min_periods=10).std(ddof=0).replace(0, np.nan)
        vol_z = (vol - vol_mean) / vol_std

        cols = [hl_range, intraday, gap, vol_z]
        if normalize:
            # Per-feature rolling z-score so all 4 features live on the same
            # numeric scale. This prevents vol_z (natural scale ~3-5) from
            # drowning out hl_range (natural scale ~0.01) in the model's
            # input projection.
            normed = []
            for c in cols:
                m = c.rolling(zscore_window, min_periods=20).mean()
                s = c.rolling(zscore_window, min_periods=20).std(ddof=0).replace(0, np.nan)
                normed.append((c - m) / s)
            cols = normed

        block = np.stack([c.values.astype(np.float32) for c in cols], axis=1)
        block = np.clip(block, -5.0, 5.0)
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        out[:, i, :] = block

    return out, feature_names


def assemble_bundle(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    regime_probs: pd.DataFrame,
    mask: pd.DataFrame,
    window: int,
    ohlcv_dict: dict[str, pd.DataFrame] | None = None,
) -> FeatureBundle:
    rets = log_returns(prices)
    vol = rolling_vol(rets, window=window)
    macro_feats = build_macro_features(macro, window=window)

    # Reindex everything to the price index (already business-day).
    idx = prices.index
    macro_feats = macro_feats.reindex(idx).ffill()
    regime_probs = regime_probs.reindex(idx).ffill()
    mask = mask.reindex(idx).fillna(False)

    ohlcv_feats = None
    ohlcv_names = None
    if ohlcv_dict is not None:
        ohlcv_feats, ohlcv_names = build_ohlcv_features(
            ohlcv_dict, tickers=list(prices.columns), dates=idx,
        )

    return FeatureBundle(
        returns=rets.reindex(idx),
        vol=vol.reindex(idx),
        macro_feats=macro_feats,
        regime_probs=regime_probs,
        mask=mask,
        ohlcv_feats=ohlcv_feats,
        ohlcv_feature_names=ohlcv_names,
    )
