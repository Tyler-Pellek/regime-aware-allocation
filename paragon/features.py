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

    @property
    def tickers(self) -> list[str]:
        return list(self.returns.columns)

    @property
    def dates(self) -> pd.DatetimeIndex:
        return self.returns.index


def assemble_bundle(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    regime_probs: pd.DataFrame,
    mask: pd.DataFrame,
    window: int,
) -> FeatureBundle:
    rets = log_returns(prices)
    vol = rolling_vol(rets, window=window)
    macro_feats = build_macro_features(macro, window=window)

    # Reindex everything to the price index (already business-day).
    idx = prices.index
    macro_feats = macro_feats.reindex(idx).ffill()
    regime_probs = regime_probs.reindex(idx).ffill()
    mask = mask.reindex(idx).fillna(False)

    return FeatureBundle(
        returns=rets.reindex(idx),
        vol=vol.reindex(idx),
        macro_feats=macro_feats,
        regime_probs=regime_probs,
        mask=mask,
    )
