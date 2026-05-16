"""Walk-forward backtester.

Given:
  - a FeatureBundle covering the full study period,
  - a sequence of "fold" boundaries (train_end, test_end),
the backtester:
  1. (re)fits the HMM on data up to train_end and re-emits regime probabilities
     for the entire bundle (the HMM is a low-D model; refit cost is trivial).
  2. (re)trains the Cross-Asset Transformer on snapshots whose decision date
     <= train_end.
  3. Rolls forward week by week from train_end+1 to test_end:
       - builds the snapshot using the bundle "as of" the decision date
         (only past data — no leakage),
       - runs the model -> Sigma, mu,
       - runs the CVaR optimizer with current prev_w,
       - applies the new weights at next-day open (we use t+1 close-to-close
         returns aggregated over the next horizon),
       - records portfolio return, weights, costs, regime probs.
  4. Returns a results bundle with per-step rows and summary stats.

Walk-forward folds are non-overlapping in test windows but cumulative in train.
This is the "expanding-window" walk-forward variant; a "rolling-window" variant
is configurable via `train_lookback_days`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..features import FeatureBundle, assemble_bundle, log_returns, rolling_vol, build_macro_features
from ..model.cholesky import sigma_from_L
from ..model.input_builder import build_snapshot, collate_snapshots
from ..model.transformer import CrossAssetTransformer, TransformerConfig
from ..optim.baselines import BASELINES
from ..optim.cvar import CVaRConfig, optimize_cvar
from ..regime.hmm import HMMConfig, RegimeHMM, make_regime_signal
from ..training.trainer import TrainConfig, train_model
from ..utils.logging import get_logger

LOG = get_logger(__name__)


# --------------------------------------------------------------------------- #
# Configs
# --------------------------------------------------------------------------- #

@dataclass
class WalkForwardConfig:
    start: str = "2005-01-03"
    end: str = "2020-04-01"
    initial_train_end: str = "2010-12-31"
    rebal_freq_days: int = 5            # weekly
    fold_test_days: int = 252            # ~1 year per test fold before retrain
    train_lookback_days: int | None = None   # None = expanding window
    transaction_cost_bps: float = 2.0   # one-way, applied to L1 turnover
    prev_w_in_features: bool = True


@dataclass
class BacktestArtifacts:
    portfolio_returns: pd.Series           # daily returns (after costs)
    weights: pd.DataFrame                  # T x N decision-date weights
    decisions: pd.DataFrame                # decision-step diagnostics
    regime_probs: pd.DataFrame             # T x k inferred regimes
    summary: dict


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

def _generate_fold_boundaries(
    bundle_dates: pd.DatetimeIndex, cfg: WalkForwardConfig
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """Yield (train_end, test_end) pairs.

    Each fold's test window is `fold_test_days` business days long. When the
    next step would not advance (cur is already at the last available date,
    or test_end == cur), we stop — that bug previously caused an infinite
    loop appending duplicate folds.
    """
    start = pd.Timestamp(cfg.initial_train_end)
    finish = pd.Timestamp(cfg.end)
    folds: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    cur = start
    last_idx = len(bundle_dates) - 1
    while cur < finish:
        cur_idx = int(bundle_dates.get_indexer([cur], method="ffill")[0])
        if cur_idx < 0 or cur_idx >= last_idx:
            break
        test_end_idx = min(cur_idx + cfg.fold_test_days, last_idx)
        test_end = bundle_dates[test_end_idx]
        if test_end <= cur:                     # no forward progress -> stop
            break
        folds.append((cur, test_end))
        if test_end_idx >= last_idx or test_end >= finish:
            break
        cur = test_end
    return folds


def _decision_dates_in_window(
    bundle_dates: pd.DatetimeIndex,
    start: pd.Timestamp,
    end: pd.Timestamp,
    freq_days: int,
) -> pd.DatetimeIndex:
    mask = (bundle_dates > start) & (bundle_dates <= end)
    sub = bundle_dates[mask]
    return sub[::freq_days]


def _train_decision_dates(
    bundle_dates: pd.DatetimeIndex, train_end: pd.Timestamp, freq_days: int, window: int,
    train_lookback_days: int | None,
) -> pd.DatetimeIndex:
    if train_lookback_days is None:
        cand = bundle_dates[bundle_dates <= train_end]
    else:
        train_start = train_end - pd.Timedelta(days=train_lookback_days)
        cand = bundle_dates[(bundle_dates > train_start) & (bundle_dates <= train_end)]
    cand = cand[window:]
    return cand[::freq_days]


def _refit_hmm(
    macro: pd.DataFrame, train_end: pd.Timestamp, hmm_cfg: HMMConfig
) -> tuple[RegimeHMM, pd.DataFrame]:
    """Refit HMM on macro data up to train_end, predict over the full series."""
    sig = make_regime_signal(macro)
    sig_train = sig.loc[:train_end].dropna()
    hmm = RegimeHMM(hmm_cfg).fit(sig_train)
    probs = hmm.predict_proba(sig)
    return hmm, probs


# --------------------------------------------------------------------------- #
# Inference
# --------------------------------------------------------------------------- #

@torch.no_grad()
def _infer_one(
    model: CrossAssetTransformer,
    bundle: FeatureBundle,
    date: pd.Timestamp,
    window: int,
    prev_w: np.ndarray,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (mu, sigma, mask) at `date`."""
    snap = build_snapshot(bundle, date, window=window, prev_w=prev_w)
    batch = collate_snapshots([snap])
    batch = {k: v.to(device) for k, v in batch.items()}
    out = model(batch["asset_feats"], batch["ctx_feats"], batch["mask"])
    Sigma = sigma_from_L(out["L"], batch["mask"])
    mu = out["mu"][0].cpu().numpy().astype(np.float64)
    sigma = Sigma[0].cpu().numpy().astype(np.float64)
    mask = snap.mask
    return mu, sigma, mask


# --------------------------------------------------------------------------- #
# Realized return between two decision dates
# --------------------------------------------------------------------------- #

def _realized_returns(
    prices: pd.DataFrame, t0: pd.Timestamp, t1: pd.Timestamp, weights: np.ndarray
) -> pd.Series:
    """Daily portfolio returns from t0 (exclusive) through t1 (inclusive)."""
    sub = prices.loc[(prices.index > t0) & (prices.index <= t1)]
    if sub.empty:
        return pd.Series(dtype=float)
    ret = sub.pct_change().fillna(0.0)
    # Special-case: first return needs a base = price at t0 (close).
    if t0 in prices.index:
        base = prices.loc[t0]
        first = (sub.iloc[0] / base) - 1.0
        ret.iloc[0] = first.fillna(0.0).values
    # Apply (held weights) — assume rebalanced at decision date (no drift within step).
    pf_ret = (ret * weights).sum(axis=1)
    return pf_ret


# --------------------------------------------------------------------------- #
# Public driver
# --------------------------------------------------------------------------- #

def run_walk_forward(
    prices: pd.DataFrame,
    macro: pd.DataFrame,
    universe_mask: pd.DataFrame,
    cfg: WalkForwardConfig,
    model_cfg_factory,                    # callable(F_asset, F_ctx, N) -> TransformerConfig
    train_cfg: TrainConfig,
    cvar_cfg: CVaRConfig,
    hmm_cfg: HMMConfig | None = None,
    artifacts_dir: str | Path | None = None,
) -> BacktestArtifacts:
    hmm_cfg = hmm_cfg or HMMConfig()
    artifacts_dir = Path(artifacts_dir) if artifacts_dir else None
    if artifacts_dir is not None:
        artifacts_dir.mkdir(parents=True, exist_ok=True)

    # Trim everything to the study window.
    p = prices.loc[(prices.index >= cfg.start) & (prices.index <= cfg.end)]
    m = macro.reindex(p.index).ffill()
    um = universe_mask.reindex(p.index).fillna(False)

    folds = _generate_fold_boundaries(p.index, cfg)
    LOG.info("Walk-forward folds (train_end -> test_end):")
    for f in folds:
        LOG.info("  %s -> %s", f[0].date(), f[1].date())

    # Storage
    weights_history: list[pd.Series] = []
    decision_rows: list[dict] = []
    daily_returns: list[pd.Series] = []
    regime_probs_full: pd.DataFrame | None = None

    N = p.shape[1]
    prev_w = np.zeros(N)
    device = None  # set after first model train

    for fold_idx, (train_end, test_end) in enumerate(folds):
        LOG.info("\n=== Fold %d : train<=%s, test<=%s ===", fold_idx, train_end.date(), test_end.date())

        # ----- 1. Refit HMM -----
        hmm, probs = _refit_hmm(m, train_end, hmm_cfg)
        if regime_probs_full is None:
            regime_probs_full = probs.copy()
        else:
            regime_probs_full.update(probs)

        # ----- 2. Build feature bundle (using current regime probs) -----
        bundle = assemble_bundle(p, m, probs, um, window=train_cfg.window)

        # ----- 3. Build training snapshot decision dates (only past) -----
        train_dates = _train_decision_dates(
            p.index, train_end, cfg.rebal_freq_days, train_cfg.window, cfg.train_lookback_days,
        )
        # Need horizon room too.
        n = len(p.index)
        train_dates = train_dates[
            train_dates.map(lambda d: p.index.get_loc(d) + train_cfg.horizon < n)
        ]
        if len(train_dates) < 50:
            LOG.warning("Too few training dates (%d) — skipping fold %d.", len(train_dates), fold_idx)
            continue

        # ----- 4. Build/train model -----
        # Asset token raw feature dim: W trailing log returns + 1 vol + 1 prev_w.
        F_asset = train_cfg.window + 2
        F_ctx = bundle.regime_probs.shape[1] + bundle.macro_feats.shape[1]
        model_cfg = model_cfg_factory(F_asset, F_ctx, N)
        model, _hist = train_model(
            bundle=bundle,
            decision_dates=train_dates,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            save_path=(artifacts_dir / f"model_fold{fold_idx}.pt") if artifacts_dir else None,
        )
        from ..training.trainer import select_device
        device = select_device(train_cfg.device)
        model.eval()

        # ----- 5. Roll the test window with periodic rebalances -----
        test_dates = _decision_dates_in_window(p.index, train_end, test_end, cfg.rebal_freq_days)
        if len(test_dates) == 0:
            continue
        # Need each decision date to have at least one forward day for return accrual.
        test_dates = pd.DatetimeIndex([d for d in test_dates if d != p.index[-1]])

        for i, d in enumerate(test_dates):
            mu, sigma, mask = _infer_one(model, bundle, d, train_cfg.window, prev_w, device)
            # Symmetrize Sigma defensively (matrix from PyTorch may have tiny asym).
            sigma = 0.5 * (sigma + sigma.T)
            new_w, info = optimize_cvar(mu, sigma, prev_w, mask, cvar_cfg)
            # Realized window: from d (exclusive) to next decision date (inclusive).
            d_next = test_dates[i + 1] if i + 1 < len(test_dates) else min(test_end, p.index[-1])
            ret_seg = _realized_returns(p, d, d_next, new_w)
            # Apply transaction costs to first return of segment.
            tcost = cfg.transaction_cost_bps * 1e-4 * info.get("turnover_total", 0.0)
            if not ret_seg.empty:
                ret_seg.iloc[0] = ret_seg.iloc[0] - tcost
                daily_returns.append(ret_seg)

            weights_history.append(pd.Series(new_w, index=p.columns, name=d))
            decision_rows.append({
                "date": d,
                "fold": fold_idx,
                "n_active": info.get("n_active", int(mask.sum())),
                "expected_return": info.get("expected_return", np.nan),
                "expected_vol": info.get("expected_vol", np.nan),
                "cvar_alpha": info.get("cvar_alpha", np.nan),
                "turnover_total": info.get("turnover_total", 0.0),
                "tcost": tcost,
                "status": info.get("status", ""),
            })
            prev_w = new_w

    # ----- Aggregate -----
    pf_returns = pd.concat(daily_returns).sort_index() if daily_returns else pd.Series(dtype=float)
    pf_returns = pf_returns.groupby(pf_returns.index).sum()  # collapse any duplicate days
    weights_df = pd.DataFrame(weights_history)
    decisions_df = pd.DataFrame(decision_rows).set_index("date") if decision_rows else pd.DataFrame()

    summary = {
        "n_decisions": len(decision_rows),
        "n_folds": len(folds),
        "first_date": str(pf_returns.index.min().date()) if not pf_returns.empty else None,
        "last_date": str(pf_returns.index.max().date()) if not pf_returns.empty else None,
    }
    if regime_probs_full is None:
        regime_probs_full = pd.DataFrame(index=p.index)
    return BacktestArtifacts(
        portfolio_returns=pf_returns,
        weights=weights_df,
        decisions=decisions_df,
        regime_probs=regime_probs_full,
        summary=summary,
    )


# --------------------------------------------------------------------------- #
# Baseline runner (no model — just a policy fn)
# --------------------------------------------------------------------------- #

def run_baseline_walk_forward(
    prices: pd.DataFrame,
    universe_mask: pd.DataFrame,
    cfg: WalkForwardConfig,
    policy_name: str,
    cvar_cfg: CVaRConfig | None = None,
) -> BacktestArtifacts:
    """Run a baseline policy on the same calendar as the model backtest."""
    p = prices.loc[(prices.index >= cfg.start) & (prices.index <= cfg.end)]
    um = universe_mask.reindex(p.index).fillna(False)
    rets = log_returns(p)

    # Decision calendar: all decisions across the full study period
    full_dates = _decision_dates_in_window(p.index, p.index.min(), p.index.max(), cfg.rebal_freq_days)
    full_dates = full_dates[full_dates >= pd.Timestamp(cfg.initial_train_end)]
    full_dates = pd.DatetimeIndex([d for d in full_dates if d != p.index[-1]])

    policy = BASELINES[policy_name]
    N = p.shape[1]
    prev_w = np.zeros(N)
    weights_history = []
    daily_returns = []
    rows = []
    for i, d in enumerate(full_dates):
        mask = um.loc[d].values.astype(bool)
        history = rets.loc[:d].iloc[-252:]
        if policy_name == "sample_cov_cvar":
            new_w = policy(prev_w, mask, history, cfg=cvar_cfg)
        else:
            new_w = policy(prev_w, mask, history)
        d_next = full_dates[i + 1] if i + 1 < len(full_dates) else p.index[-1]
        seg = _realized_returns(p, d, d_next, new_w)
        tcost = cfg.transaction_cost_bps * 1e-4 * float(np.abs(new_w - prev_w).sum())
        if not seg.empty:
            seg.iloc[0] = seg.iloc[0] - tcost
            daily_returns.append(seg)
        weights_history.append(pd.Series(new_w, index=p.columns, name=d))
        rows.append({"date": d, "turnover_total": float(np.abs(new_w - prev_w).sum()), "tcost": tcost})
        prev_w = new_w

    pf_returns = pd.concat(daily_returns).sort_index() if daily_returns else pd.Series(dtype=float)
    pf_returns = pf_returns.groupby(pf_returns.index).sum()
    return BacktestArtifacts(
        portfolio_returns=pf_returns,
        weights=pd.DataFrame(weights_history),
        decisions=pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame(),
        regime_probs=pd.DataFrame(index=p.index),
        summary={"policy": policy_name, "n_decisions": len(rows)},
    )
