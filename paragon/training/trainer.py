"""Training loop for the Cross-Asset Transformer (covariance forecaster)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from ..features import FeatureBundle
from ..model.cholesky import gaussian_nll, sigma_anchor_loss
from ..model.transformer import CrossAssetTransformer, TransformerConfig
from ..utils.logging import get_logger
from .dataset import SnapshotDataset, collate

LOG = get_logger(__name__)


@dataclass
class TrainConfig:
    window: int = 60
    horizon: int = 5             # weekly forward target
    batch_size: int = 32
    epochs: int = 30
    lr: float = 3e-4
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    val_fraction: float = 0.15
    early_stop_patience: int = 6
    device: str = "cuda"         # "cuda" / "mps" / "cpu" — auto-fallback in code
    anchor_lambda: float = 0.0   # weight on the Sigma-anchor auxiliary loss
                                 # (0 = pure NLL, 1.0+ = strong shrinkage to
                                 # rolling sample covariance). 0.1-0.5 typical.


def select_device(preferred: str) -> torch.device:
    if preferred == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if preferred == "mps" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def split_train_val(dates: pd.DatetimeIndex, val_fraction: float) -> tuple[pd.DatetimeIndex, pd.DatetimeIndex]:
    """Chronological split: last `val_fraction` of dates -> validation."""
    n = len(dates)
    cut = int(n * (1.0 - val_fraction))
    return dates[:cut], dates[cut:]


def train_model(
    bundle: FeatureBundle,
    decision_dates: pd.DatetimeIndex,
    model_cfg: TransformerConfig,
    train_cfg: TrainConfig,
    save_path: str | Path | None = None,
    warm_start_state: dict | None = None,
    warm_start_lr_scale: float = 0.5,
) -> tuple[CrossAssetTransformer, dict]:
    """Train a CrossAssetTransformer on snapshots from `decision_dates`.

    If `warm_start_state` is provided (a state_dict from a previous fold's
    model), the new model is initialized with those weights instead of from
    scratch, and the learning rate is scaled by `warm_start_lr_scale` (default
    0.5x) — this is fine-tuning, not initial training. Per memo §6 this
    "preserves the network's deep memory of historical market crashes".

    Returns the trained model and a history dict (per-epoch losses).
    """
    device = select_device(train_cfg.device)
    LOG.info("Training on device: %s%s", device, " (WARM START)" if warm_start_state else "")

    train_dates, val_dates = split_train_val(decision_dates, train_cfg.val_fraction)
    LOG.info("Train snapshots: %d   Val snapshots: %d", len(train_dates), len(val_dates))

    train_ds = SnapshotDataset(bundle, train_dates, train_cfg.window, train_cfg.horizon)
    val_ds = SnapshotDataset(bundle, val_dates, train_cfg.window, train_cfg.horizon)
    if len(train_ds) == 0:
        raise RuntimeError("Empty training dataset — check decision_dates / window / horizon.")

    train_loader = DataLoader(
        train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate,
        num_workers=0, pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate,
        num_workers=0, pin_memory=(device.type == "cuda"),
    ) if len(val_ds) > 0 else None

    model = CrossAssetTransformer(model_cfg).to(device)
    if warm_start_state is not None:
        # Tolerate small parameter shape mismatches (e.g. universe N changed) by
        # loading what matches and warning about the rest.
        own_state = model.state_dict()
        loadable = {k: v for k, v in warm_start_state.items()
                    if k in own_state and own_state[k].shape == v.shape}
        missing = [k for k in warm_start_state.keys() if k not in loadable]
        if missing:
            LOG.warning("Warm-start: skipping %d/%d params due to shape mismatch (e.g. %s)",
                        len(missing), len(warm_start_state), missing[:3])
        own_state.update(loadable)
        model.load_state_dict(own_state)
    effective_lr = train_cfg.lr * (warm_start_lr_scale if warm_start_state is not None else 1.0)
    opt = torch.optim.AdamW(model.parameters(), lr=effective_lr, weight_decay=train_cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=train_cfg.epochs)

    history = {"train_loss": [], "val_loss": [], "train_nll": [], "train_anchor": []}
    best_val = float("inf")
    best_state = None
    bad_epochs = 0
    use_anchor = train_cfg.anchor_lambda > 0

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        total, total_nll, total_anchor, n_batches = 0.0, 0.0, 0.0, 0
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(batch["asset_feats"], batch["ctx_feats"], batch["mask"])
            nll = gaussian_nll(out["mu"], out["L"], batch["target"], batch["target_mask"])
            if use_anchor:
                aux = sigma_anchor_loss(
                    out["L"], batch["sigma_baseline"], batch["mask"]
                )
                loss = nll + train_cfg.anchor_lambda * aux
                total_anchor += float(aux.item())
            else:
                loss = nll
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            opt.step()
            total += float(loss.item())
            total_nll += float(nll.item())
            n_batches += 1
        train_loss = total / max(n_batches, 1)
        train_nll_val = total_nll / max(n_batches, 1)
        train_anchor_val = total_anchor / max(n_batches, 1)
        history["train_loss"].append(train_loss)
        history["train_nll"].append(train_nll_val)
        history["train_anchor"].append(train_anchor_val)

        if val_loader is not None:
            model.eval()
            vtotal, vn = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = {k: v.to(device) for k, v in batch.items()}
                    out = model(batch["asset_feats"], batch["ctx_feats"], batch["mask"])
                    vnll = gaussian_nll(out["mu"], out["L"], batch["target"], batch["target_mask"])
                    if use_anchor:
                        vaux = sigma_anchor_loss(
                            out["L"], batch["sigma_baseline"], batch["mask"]
                        )
                        vloss = vnll + train_cfg.anchor_lambda * vaux
                    else:
                        vloss = vnll
                    vtotal += float(vloss.item())
                    vn += 1
            val_loss = vtotal / max(vn, 1)
        else:
            val_loss = train_loss
        history["val_loss"].append(val_loss)

        sched.step()
        if use_anchor:
            LOG.info(
                "epoch %02d   train_nll=%.4f anchor=%.4f   val_loss=%.4f   lr=%.2e",
                epoch, train_nll_val, train_anchor_val, val_loss, opt.param_groups[0]["lr"],
            )
        else:
            LOG.info(
                "epoch %02d   train_nll=%.5f   val_nll=%.5f   lr=%.2e",
                epoch, train_loss, val_loss, opt.param_groups[0]["lr"],
            )
        # Early stop on val
        if val_loss + 1e-6 < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
        else:
            bad_epochs += 1
            if bad_epochs >= train_cfg.early_stop_patience:
                LOG.info("Early stop at epoch %d (no val improvement for %d epochs).", epoch, bad_epochs)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"state_dict": model.state_dict(), "config": model_cfg, "history": history}, save_path)
        LOG.info("Saved model to %s", save_path)
    return model, history
