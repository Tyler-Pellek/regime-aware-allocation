"""Snapshot dataset for training.

A training example consists of:
  - input snapshot at decision date t  (asset_feats, ctx_feats, mask, prev_w)
  - target = next-period (h-day) cumulative log returns r_{t -> t+h}
  - target_mask = mask & assets that survive through t+h

We don't materialize prev_w during training (the model is trained as a
covariance forecaster, not closed-loop). Instead we set prev_w = 0 in features,
which is consistent: the model learns Sigma conditional on no-position context,
and the optimizer at inference time supplies the actual prev_w through the
asset-feature scalar.

Caveat: this introduces a small train/eval mismatch (the prev_w slot is always
0 at train but populated at inference). In practice the model treats it as a
weak nudge and downstream evaluation shows it doesn't materially harm OOS
performance. If you want exact parity, you can either:
  (a) populate prev_w from a teacher policy (e.g., equal-weight) during train,
  (b) drop the prev_w slot at inference too and route the L1 anchor purely
      through the optimizer (already where the L1 lives — the inside-token
      placement is the memo's design choice).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from ..features import FeatureBundle
from ..model.input_builder import build_snapshot, collate_snapshots, Snapshot


@dataclass
class TrainSample:
    snapshot: Snapshot
    target: np.ndarray            # (N,) forward h-day cumulative log return
    target_mask: np.ndarray       # (N,) bool — asset is active at both t and t+h
    sigma_baseline: np.ndarray    # (N, N) rolling sample covariance, h-scaled.
                                  # Used as a soft prior the model is anchored to.


class SnapshotDataset(Dataset):
    """Yields TrainSamples for every t in `decision_dates`.

    Decision dates should be a subset of the bundle's index that satisfies:
      - position(t) >= window
      - position(t) + horizon < len(bundle.dates)
    """

    def __init__(
        self,
        bundle: FeatureBundle,
        decision_dates: pd.DatetimeIndex,
        window: int,
        horizon: int,
    ):
        self.bundle = bundle
        self.window = window
        self.horizon = horizon
        # Filter to dates with enough history and enough forward.
        n = len(bundle.dates)
        valid = []
        for d in decision_dates:
            if d not in bundle.dates:
                continue
            pos = bundle.dates.get_loc(d)
            if pos < window:
                continue
            if pos + horizon >= n:
                continue
            valid.append(d)
        self.decision_dates = pd.DatetimeIndex(valid)

    def __len__(self) -> int:
        return len(self.decision_dates)

    def __getitem__(self, i: int) -> TrainSample:
        d = self.decision_dates[i]
        snap = build_snapshot(self.bundle, d, window=self.window, prev_w=None)
        pos = self.bundle.dates.get_loc(d)
        # Forward cumulative log return over horizon trading days.
        fwd = self.bundle.returns.iloc[pos + 1 : pos + 1 + self.horizon].values  # (h, N)
        # Where any of the h forward days is NaN, the asset doesn't survive.
        survives = ~np.isnan(fwd).any(axis=0)
        fwd_filled = np.nan_to_num(fwd, nan=0.0)
        target = fwd_filled.sum(axis=0).astype(np.float32)        # (N,)
        target_mask = snap.mask & survives

        # Rolling-window sample covariance, scaled to the forward horizon.
        # Inactive (NaN) asset rows and cols become identity entries so the
        # matrix stays PSD and the aux loss correctly ignores them via the
        # mask at training time.
        rets_window = self.bundle.returns.iloc[pos - self.window + 1 : pos + 1].values
        rets_clean = np.nan_to_num(rets_window, nan=0.0)
        sigma_baseline = (np.cov(rets_clean.T, ddof=0) * self.horizon).astype(np.float32)
        # Replace inactive-asset rows/cols with identity to avoid pulling the
        # predicted Sigma toward zero on assets the model is masking out.
        inactive = ~snap.mask
        if inactive.any():
            sigma_baseline[inactive, :] = 0.0
            sigma_baseline[:, inactive] = 0.0
            sigma_baseline[inactive, inactive] = 1.0
        return TrainSample(snap, target, target_mask, sigma_baseline)


def collate(batch: list[TrainSample]) -> dict[str, torch.Tensor]:
    snaps = [b.snapshot for b in batch]
    out = collate_snapshots(snaps)
    out["target"] = torch.from_numpy(np.stack([b.target for b in batch], axis=0))
    out["target_mask"] = torch.from_numpy(np.stack([b.target_mask for b in batch], axis=0))
    out["sigma_baseline"] = torch.from_numpy(
        np.stack([b.sigma_baseline for b in batch], axis=0)
    )
    return out
