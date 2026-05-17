"""Universe-wide pretraining for the Cross-Asset Transformer backbone.

Trains the model on randomly-sampled subsets of a 200-ticker pretraining
universe so the backbone learns covariance structure from ~30x more data
than the 16-ticker fine-tune universe provides.

Architecture requirements:
  - use_asset_pos=False        : permutation-invariant over assets
  - head_type='factor'         : universe-agnostic factor head
With these set, every parameter is independent of N, so the saved backbone
transfers cleanly to a fine-tune universe of any size.

The pretrained checkpoint is consumed by run_backtest.py via
`backtest.pretrained_checkpoint: <path>` — fold 0's model warm-starts from
the pretrain weights instead of random init.
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

from paragon.data import (
    EXTENDED_MACRO_SYMBOLS,
    align_prices_macro,
    fetch_macro,
    load_kaggle_ohlcv,
    load_kaggle_prices,
)
from paragon.features import assemble_bundle
from paragon.model.cholesky import gaussian_nll
from paragon.model.input_builder import build_snapshot, collate_snapshots
from paragon.model.transformer import CrossAssetTransformer, TransformerConfig
from paragon.regime.hmm import HMMConfig, RegimeHMM, make_regime_signal
from paragon.universe import build_universe
from paragon.utils.logging import get_logger
from paragon.utils.seed import set_seed

LOG = get_logger("paragon.pretrain")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/pretrain.yaml")
    p.add_argument("--out", default="artifacts/pretrained/backbone.pt")
    args = p.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    set_seed(cfg.get("seed", 42))

    # ---- Load universe + data ----
    tickers = json.loads(Path(cfg["universe_json"]).read_text())
    LOG.info("Pretrain universe: %d tickers", len(tickers))

    LOG.info("Loading Kaggle prices ...")
    prices = load_kaggle_prices(
        tickers, kaggle_root=cfg["data"]["kaggle_root"],
        start=cfg["data"]["start"], end=cfg["data"]["end"],
    )
    LOG.info("Prices shape: %s", prices.shape)
    LOG.info("Loading Kaggle OHLCV ...")
    ohlcv = load_kaggle_ohlcv(
        tickers, kaggle_root=cfg["data"]["kaggle_root"],
        start=cfg["data"]["start"], end=cfg["data"]["end"],
    )

    LOG.info("Loading extended macro (with yield curve) ...")
    macro = fetch_macro(
        symbols=EXTENDED_MACRO_SYMBOLS,
        start=cfg["data"]["start"], end=cfg["data"]["end"],
        cache_path=cfg["data"]["macro_cache"], refresh=False,
    )
    prices, macro = align_prices_macro(prices, macro)

    # Universe mask (some tickers IPO'd mid-window)
    universe = build_universe(prices, tickers=tickers, min_history=60)

    # Fit HMM on full macro (this is technically using future data for the HMM,
    # but the HMM is just providing a soft regime signal — not the prediction
    # target. For pretraining this is acceptable; fine-tuning still does
    # walk-forward HMM refits per fold.)
    LOG.info("Fitting HMM ...")
    hmm = RegimeHMM(HMMConfig(n_states=cfg["regime"]["n_states"])).fit(
        make_regime_signal(macro).dropna()
    )
    probs = hmm.predict_proba(make_regime_signal(macro))

    LOG.info("Assembling big bundle ...")
    bundle = assemble_bundle(
        universe.prices, macro, probs, universe.mask,
        window=cfg["training"]["window"], ohlcv_dict=ohlcv,
    )

    # ---- Pre-compute valid (date, mask) pairs we can train on ----
    W = cfg["training"]["window"]
    H = cfg["training"]["horizon"]
    n_dates = len(bundle.dates)
    # Need W trailing days AND H forward days, and at least N_per_step active assets.
    N_per_step = cfg["training"]["n_assets_per_step"]
    valid_date_indices = []
    for t in range(W, n_dates - H):
        n_active = int(bundle.mask.iloc[t].sum())
        if n_active >= N_per_step:
            valid_date_indices.append(t)
    LOG.info("Valid training dates: %d (need >= %d active assets)",
             len(valid_date_indices), N_per_step)
    if not valid_date_indices:
        raise RuntimeError("No valid pretrain dates with enough active assets.")

    # ---- Build model ----
    F_asset = W + 2 + bundle.n_ohlcv_feats
    F_ctx = bundle.regime_probs.shape[1] + bundle.macro_feats.shape[1]
    mcfg = TransformerConfig(
        n_assets=N_per_step,
        asset_in_dim=F_asset,
        ctx_in_dim=F_ctx,
        d_model=cfg["model"]["d_model"],
        n_heads=cfg["model"]["n_heads"],
        n_layers=cfg["model"]["n_layers"],
        ff_mult=cfg["model"]["ff_mult"],
        dropout=cfg["model"]["dropout"],
        chol_min_diag=cfg["model"]["chol_min_diag"],
        head_type="factor",
        n_factors=cfg["model"]["n_factors"],
        use_asset_pos=False,                # REQUIRED for universe transfer
    )
    device = torch.device(cfg["training"]["device"])
    model = CrossAssetTransformer(mcfg).to(device)
    LOG.info("Model params: %.2fM", sum(p.numel() for p in model.parameters()) / 1e6)

    opt = AdamW(
        model.parameters(),
        lr=cfg["training"]["lr"],
        weight_decay=cfg["training"]["weight_decay"],
    )
    n_steps = cfg["training"]["n_steps"]
    sched = CosineAnnealingLR(opt, T_max=n_steps)

    # ---- Training loop ----
    rng = random.Random(cfg.get("seed", 42))
    np_rng = np.random.default_rng(cfg.get("seed", 42))
    batch_size = cfg["training"]["batch_size"]
    log_every = cfg["training"].get("log_every", 50)
    ckpt_every = cfg["training"].get("ckpt_every", 1000)

    start = time.time()
    losses_window = []
    for step in range(1, n_steps + 1):
        # Sample a different ticker subset per batch (one universe per batch
        # is the simplest setup — all batch elements share the same N tickers).
        chosen_idx = np_rng.choice(len(tickers), size=N_per_step, replace=False)
        chosen_tickers = [tickers[i] for i in chosen_idx]

        # Sample batch_size random valid dates
        date_pos_list = rng.sample(valid_date_indices, k=batch_size) \
            if len(valid_date_indices) >= batch_size \
            else [rng.choice(valid_date_indices) for _ in range(batch_size)]

        # Build snapshots (we leverage the existing builder but with a
        # subsampled bundle slice). For efficiency we just build the snapshot
        # for the FULL bundle then slice the asset dimensions afterwards.
        snaps = []
        targets = []
        target_masks = []
        for pos in date_pos_list:
            d = bundle.dates[pos]
            snap_full = build_snapshot(bundle, d, window=W, prev_w=None)
            # Slice down to the sampled asset subset
            asset_feats = snap_full.asset_feats[chosen_idx]            # (N_per_step, F_a)
            mask_sub = snap_full.mask[chosen_idx]
            prev_w_sub = snap_full.prev_w[chosen_idx]
            # Forward returns target
            fwd = bundle.returns.iloc[pos + 1 : pos + 1 + H].values[:, chosen_idx]  # (H, N)
            survives = ~np.isnan(fwd).any(axis=0)
            target = np.nan_to_num(fwd, nan=0.0).sum(axis=0).astype(np.float32)
            tmask = mask_sub & survives
            snaps.append((asset_feats, snap_full.ctx_feats, mask_sub, prev_w_sub))
            targets.append(target)
            target_masks.append(tmask)

        # Stack
        asset_feats = torch.from_numpy(np.stack([s[0] for s in snaps], axis=0)).to(device)
        ctx_feats = torch.from_numpy(np.stack([s[1] for s in snaps], axis=0)).to(device)
        mask = torch.from_numpy(np.stack([s[2] for s in snaps], axis=0)).to(device)
        target = torch.from_numpy(np.stack(targets, axis=0)).to(device)
        target_mask = torch.from_numpy(np.stack(target_masks, axis=0)).to(device)

        out = model(asset_feats, ctx_feats, mask)
        loss = gaussian_nll(out["mu"], out["L"], target, target_mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["training"]["grad_clip"])
        opt.step()
        sched.step()

        losses_window.append(float(loss.item()))
        if step % log_every == 0:
            avg = float(np.mean(losses_window[-log_every:]))
            elapsed = time.time() - start
            steps_per_sec = step / elapsed
            eta_sec = (n_steps - step) / max(steps_per_sec, 1e-9)
            LOG.info(
                "step %5d/%d  loss=%.4f  lr=%.2e  %.1f steps/s  eta=%dm",
                step, n_steps, avg, opt.param_groups[0]["lr"],
                steps_per_sec, int(eta_sec / 60),
            )
        if step % ckpt_every == 0 or step == n_steps:
            out_path = Path(args.out)
            out_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "state_dict": model.state_dict(),
                "model_cfg": mcfg,
                "step": step,
                "n_pretrain_tickers": len(tickers),
                "n_assets_per_step": N_per_step,
            }, out_path)
            LOG.info("  -> saved checkpoint to %s (step %d)", out_path, step)

    LOG.info("Done. Total time: %.1f min", (time.time() - start) / 60.0)


if __name__ == "__main__":
    main()
