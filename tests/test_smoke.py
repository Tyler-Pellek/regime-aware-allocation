"""Smoke tests using synthetic data — no Kaggle / yfinance required.

Run: pytest -q tests/test_smoke.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from paragon.features import assemble_bundle, build_macro_features, log_returns, rolling_vol
from paragon.model.cholesky import gaussian_nll, sigma_from_L
from paragon.model.input_builder import build_snapshot, collate_snapshots
from paragon.model.transformer import CrossAssetTransformer, TransformerConfig
from paragon.optim.cvar import CVaRConfig, optimize_cvar
from paragon.regime.hmm import HMMConfig, RegimeHMM, make_regime_signal
from paragon.universe import build_universe


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #

@pytest.fixture(scope="module")
def synthetic_panel():
    """Return (prices, macro) with realistic shapes for testing."""
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2005-01-03", "2020-04-01")
    tickers = ["AAA", "BBB", "CCC", "DDD", "EEE", "FFF"]
    # Latent regime: most days low-vol, occasional high-vol bursts.
    n = len(dates)
    sigma_path = np.where(rng.random(n) < 0.07, 0.04, 0.012)
    rets = rng.normal(0.0003, sigma_path[:, None], size=(n, len(tickers)))
    prices = pd.DataFrame(100.0 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=tickers)

    # Stagger one ticker (entering 2010-01).
    prices.loc[prices.index < "2010-01-01", "FFF"] = np.nan

    macro = pd.DataFrame({
        "VIX": 15 + 30 * (sigma_path - sigma_path.min()) / (sigma_path.max() - sigma_path.min() + 1e-9),
        "SPX": 1000.0 * np.exp(np.cumsum(rng.normal(0.0003, 0.01, size=n))),
        "TNX": 30 + 10 * np.cumsum(rng.normal(0, 0.001, size=n)),
        "DXY": 90 + 5 * np.cumsum(rng.normal(0, 0.001, size=n)),
        "OIL": 60 + np.cumsum(rng.normal(0, 0.5, size=n)),
        "HYG": 80 + np.cumsum(rng.normal(0, 0.05, size=n)),
        "TLT": 100 + np.cumsum(rng.normal(0, 0.05, size=n)),
    }, index=dates)
    return prices, macro


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_universe_mask(synthetic_panel):
    prices, _ = synthetic_panel
    u = build_universe(prices, tickers=list(prices.columns), min_history=60)
    assert u.mask.shape == prices.shape
    # FFF only becomes tradable after 2010 + 60-day warmup.
    first_date_fff = u.mask["FFF"].idxmax()
    assert first_date_fff > pd.Timestamp("2010-03-01")


def test_features(synthetic_panel):
    prices, macro = synthetic_panel
    rets = log_returns(prices)
    vol = rolling_vol(rets, window=60)
    assert rets.shape == prices.shape
    # First 5 columns are populated from day 0; FFF (NaN before 2010) is not.
    full_cols = ["AAA", "BBB", "CCC", "DDD", "EEE"]
    assert vol.iloc[60][full_cols].isna().sum() == 0
    macro_feats = build_macro_features(macro, window=60)
    assert "vix_level" in macro_feats.columns


def test_hmm_roundtrip(synthetic_panel):
    _, macro = synthetic_panel
    sig = make_regime_signal(macro)
    hmm = RegimeHMM(HMMConfig(n_states=3, n_iter=50)).fit(sig)
    probs = hmm.predict_proba(sig)
    assert probs.shape[1] == 3
    valid = probs.dropna()
    assert np.allclose(valid.sum(axis=1).values, 1.0, atol=1e-6)


def test_transformer_forward(synthetic_panel):
    import torch
    prices, macro = synthetic_panel
    rets = log_returns(prices)
    vol = rolling_vol(rets, window=60)
    macro_feats = build_macro_features(macro, window=60)

    # Fake regime probs
    probs = pd.DataFrame(
        np.tile([0.3, 0.5, 0.2], (len(prices), 1)),
        index=prices.index, columns=["r0", "r1", "r2"],
    )
    u = build_universe(prices, tickers=list(prices.columns), min_history=60)
    bundle = assemble_bundle(prices, macro, probs, u.mask, window=60)

    # Pick a date well into the series.
    d = bundle.dates[800]
    snap = build_snapshot(bundle, d, window=60)
    batch = collate_snapshots([snap, snap])

    cfg = TransformerConfig(
        n_assets=prices.shape[1],
        asset_in_dim=snap.asset_feats.shape[1],
        ctx_in_dim=snap.ctx_feats.shape[0],
        d_model=64, n_heads=4, n_layers=2, ff_mult=2, dropout=0.0,
    )
    model = CrossAssetTransformer(cfg)
    out = model(batch["asset_feats"], batch["ctx_feats"], batch["mask"])
    L = out["L"]
    Sigma = sigma_from_L(L, batch["mask"])
    assert L.shape == (2, prices.shape[1], prices.shape[1])
    # Lower-triangular check.
    upper = torch.triu(L, diagonal=1)
    assert torch.allclose(upper, torch.zeros_like(upper))
    # PSD check (eigvals of active sub-block all >= 0).
    for b in range(2):
        active = batch["mask"][b].numpy().astype(bool)
        S = Sigma[b].detach().numpy()[np.ix_(active, active)]
        eig = np.linalg.eigvalsh(0.5 * (S + S.T))
        assert eig.min() > -1e-6
    # NLL is finite + scalar.
    target = torch.zeros_like(out["mu"])
    nll = gaussian_nll(out["mu"], L, target, batch["mask"])
    assert torch.isfinite(nll).item()


def test_cvar_optimizer():
    rng = np.random.default_rng(1)
    N = 6
    A = rng.standard_normal((N, N))
    Sigma = A @ A.T / N + 0.01 * np.eye(N)
    mu = rng.normal(0.001, 0.002, size=N)
    prev_w = np.zeros(N)
    mask = np.ones(N, dtype=bool)
    mask[0] = False
    w, info = optimize_cvar(mu, Sigma, prev_w, mask, CVaRConfig())
    assert info["status"] in ("optimal", "optimal_inaccurate")
    assert np.isclose(w[0], 0.0)
    assert w.sum() <= 1.0 + 1e-6
    assert (w >= -1e-9).all()
