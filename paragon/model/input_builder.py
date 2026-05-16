"""Build the model input tensor at a given snapshot date.

Snapshot t produces a tensor of shape (N+1, d_raw) where row 0 is [CTX] and
rows 1..N are asset tokens. d_raw is NOT the model embedding dim — it's the
heterogeneous raw feature dimension. The model has separate linear projections
for CTX and Asset rows that map d_raw_ctx and d_raw_asset to the common d_model.

This module also exposes a batched builder that prepares a stack of snapshots
for training (one snapshot per training timestep).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch

from ..features import FeatureBundle


@dataclass
class Snapshot:
    """A single date's model input.

    asset_feats : (N, F_asset)  raw per-asset features
    ctx_feats   : (F_ctx,)      raw context features
    mask        : (N,) bool     True where asset is tradable
    prev_w      : (N,)          last allocation (encoded inside asset_feats too,
                                kept here for the optimizer's L1 turnover term)
    date        : pd.Timestamp
    tickers     : list[str]
    """
    asset_feats: np.ndarray
    ctx_feats: np.ndarray
    mask: np.ndarray
    prev_w: np.ndarray
    date: pd.Timestamp
    tickers: list[str]


def build_snapshot(
    bundle: FeatureBundle,
    date: pd.Timestamp,
    window: int,
    prev_w: np.ndarray | None = None,
) -> Snapshot:
    """Construct a Snapshot for a given date using the trailing `window` data."""
    if date not in bundle.dates:
        raise KeyError(f"Date {date} not in feature bundle index.")
    pos = bundle.dates.get_loc(date)
    if pos < window:
        raise ValueError(f"Need at least {window} prior obs, only {pos} available.")
    sl = slice(pos - window + 1, pos + 1)  # inclusive of `date`

    rets = bundle.returns.iloc[sl].values            # (W, N)
    vol_at_t = bundle.vol.iloc[pos].values           # (N,)
    mask = bundle.mask.iloc[pos].values.astype(bool) # (N,)

    N = rets.shape[1]
    if prev_w is None:
        prev_w = np.zeros(N, dtype=np.float64)

    # ----- per-asset raw features -----
    # We pack:
    #   - W trailing log returns (NaN -> 0)
    #   - 1 rolling-vol scalar (NaN -> mean of valid vols, fallback 0.01)
    #   - 1 prev_w scalar
    rets_filled = np.nan_to_num(rets, nan=0.0)         # (W, N)
    rets_T = rets_filled.T                              # (N, W)

    valid_vol = vol_at_t[~np.isnan(vol_at_t)]
    fallback_vol = float(valid_vol.mean()) if valid_vol.size > 0 else 0.01
    vol_filled = np.where(np.isnan(vol_at_t), fallback_vol, vol_at_t)

    asset_feats = np.concatenate(
        [rets_T, vol_filled[:, None], prev_w[:, None]], axis=1
    )  # (N, W + 2)

    # ----- context features -----
    macro_row = bundle.macro_feats.iloc[pos].values
    macro_row = np.nan_to_num(macro_row, nan=0.0)
    regime_row = bundle.regime_probs.iloc[pos].values
    regime_row = np.nan_to_num(regime_row, nan=1.0 / max(len(regime_row), 1))
    ctx_feats = np.concatenate([regime_row, macro_row], axis=0)  # (k + M_macro,)

    return Snapshot(
        asset_feats=asset_feats.astype(np.float32),
        ctx_feats=ctx_feats.astype(np.float32),
        mask=mask,
        prev_w=prev_w.astype(np.float64),
        date=date,
        tickers=bundle.tickers,
    )


def collate_snapshots(snaps: list[Snapshot]) -> dict[str, torch.Tensor]:
    """Stack a list of Snapshots into a batch.

    Assumes all snapshots share the same N and feature dims (true within a
    single training run).
    """
    asset = np.stack([s.asset_feats for s in snaps], axis=0)     # (B, N, F_a)
    ctx = np.stack([s.ctx_feats for s in snaps], axis=0)         # (B, F_c)
    mask = np.stack([s.mask for s in snaps], axis=0)             # (B, N)
    prev_w = np.stack([s.prev_w for s in snaps], axis=0)         # (B, N)
    return {
        "asset_feats": torch.from_numpy(asset),
        "ctx_feats": torch.from_numpy(ctx),
        "mask": torch.from_numpy(mask),
        "prev_w": torch.from_numpy(prev_w.astype(np.float32)),
    }
