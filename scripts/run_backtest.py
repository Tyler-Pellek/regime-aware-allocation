"""End-to-end runner: data -> features -> walk-forward -> report.

Usage:
    python scripts/run_backtest.py --config configs/default.yaml
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf

from paragon.backtest.walkforward import (
    BacktestArtifacts,
    WalkForwardConfig,
    run_baseline_walk_forward,
    run_walk_forward,
)
from paragon.data import (
    DEFAULT_MACRO_SYMBOLS,
    align_prices_macro,
    fetch_macro,
    load_kaggle_prices,
)
from paragon.eval.metrics import cum_return
from paragon.eval.report import write_report
from paragon.features import log_returns
from paragon.model.transformer import TransformerConfig
from paragon.optim.cvar import CVaRConfig
from paragon.regime.hmm import HMMConfig
from paragon.training.trainer import TrainConfig
from paragon.universe import BENCHMARK, TECH_UNIVERSE, build_universe
from paragon.utils.config import load_config
from paragon.utils.logging import get_logger
from paragon.utils.seed import set_seed

LOG = get_logger("paragon.run")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--skip-baselines", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(cfg.seed)

    tickers = cfg.universe.tickers or TECH_UNIVERSE

    # ---------- 1. Load prices + macro ----------
    LOG.info("Loading Kaggle prices for %d tickers ...", len(tickers))
    prices = load_kaggle_prices(
        tickers, kaggle_root=cfg.data.kaggle_root, start=cfg.data.start, end=cfg.data.end,
    )
    LOG.info("Prices shape: %s, range %s -> %s", prices.shape, prices.index.min().date(), prices.index.max().date())

    LOG.info("Fetching macro indicators ...")
    macro = fetch_macro(
        symbols=DEFAULT_MACRO_SYMBOLS, start=cfg.data.start, end=cfg.data.end,
        cache_path=cfg.data.macro_cache, refresh=False,
    )
    prices, macro = align_prices_macro(prices, macro)

    LOG.info("Building universe mask ...")
    universe = build_universe(prices, tickers=tickers, min_history=cfg.universe.min_history)
    LOG.info(
        "Universe activated dates per ticker (first day each becomes tradable):\n%s",
        universe.mask.idxmax().sort_values().to_string(),
    )

    # ---------- 2. Configs ----------
    train_cfg = TrainConfig(
        window=cfg.training.window,
        horizon=cfg.training.horizon,
        batch_size=cfg.training.batch_size,
        epochs=cfg.training.epochs,
        lr=cfg.training.lr,
        weight_decay=cfg.training.weight_decay,
        grad_clip=cfg.training.grad_clip,
        val_fraction=cfg.training.val_fraction,
        early_stop_patience=cfg.training.early_stop_patience,
        device=cfg.training.device,
        anchor_lambda=cfg.training.get("anchor_lambda", 0.0),
    )
    cvar_cfg = CVaRConfig(
        mode=cfg.cvar.get("mode", "cvar"),
        alpha=cfg.cvar.alpha,
        risk_aversion=cfg.cvar.get("risk_aversion", 20.0),
        demean_mu=cfg.cvar.get("demean_mu", False),
        turnover_lambda=cfg.cvar.turnover_lambda,
        long_only=cfg.cvar.long_only,
        w_min=cfg.cvar.w_min,
        w_max=cfg.cvar.w_max,
        gross_max=cfg.cvar.gross_max,
        cash_allowed=cfg.cvar.cash_allowed,
        min_invested=cfg.cvar.get("min_invested", 0.0),
        vol_cap_weekly=cfg.cvar.get("vol_cap_weekly", None),
        solver=cfg.cvar.solver,
    )
    hmm_cfg = HMMConfig(
        n_states=cfg.regime.n_states,
        covariance_type=cfg.regime.covariance_type,
        n_iter=cfg.regime.n_iter,
    )
    wf_cfg = WalkForwardConfig(
        start=cfg.backtest.start,
        end=cfg.backtest.end,
        initial_train_end=cfg.backtest.initial_train_end,
        rebal_freq_days=cfg.backtest.rebal_freq_days,
        train_freq_days=cfg.backtest.get("train_freq_days", None),
        fold_test_days=cfg.backtest.fold_test_days,
        train_lookback_days=cfg.backtest.train_lookback_days,
        transaction_cost_bps=cfg.backtest.transaction_cost_bps,
        shrinkage_alpha=cfg.backtest.get("shrinkage_alpha", 1.0),
        use_model_mu=cfg.backtest.get("use_model_mu", True),
        warm_start=cfg.backtest.get("warm_start", False),
        warm_start_lr_scale=cfg.backtest.get("warm_start_lr_scale", 0.5),
    )

    def make_model_cfg(F_asset: int, F_ctx: int, N: int) -> TransformerConfig:
        return TransformerConfig(
            n_assets=N,
            asset_in_dim=F_asset,
            ctx_in_dim=F_ctx,
            d_model=cfg.model.d_model,
            n_heads=cfg.model.n_heads,
            n_layers=cfg.model.n_layers,
            ff_mult=cfg.model.ff_mult,
            dropout=cfg.model.dropout,
            chol_min_diag=cfg.model.chol_min_diag,
            head_type=cfg.model.get("head_type", "standard"),
            n_factors=cfg.model.get("n_factors", 4),
        )

    # ---------- 3. Walk-forward run ----------
    LOG.info("Starting Paragon walk-forward backtest ...")
    artifacts: BacktestArtifacts = run_walk_forward(
        prices=universe.prices,
        macro=macro,
        universe_mask=universe.mask,
        cfg=wf_cfg,
        model_cfg_factory=make_model_cfg,
        train_cfg=train_cfg,
        cvar_cfg=cvar_cfg,
        hmm_cfg=hmm_cfg,
        artifacts_dir=cfg.artifacts_dir,
    )
    LOG.info("Strategy summary: %s", artifacts.summary)

    # ---------- 4. Baselines ----------
    baseline_returns = {}
    if not args.skip_baselines:
        for name in cfg.backtest.baselines:
            LOG.info("Running baseline: %s ...", name)
            ba = run_baseline_walk_forward(
                prices=universe.prices,
                universe_mask=universe.mask,
                cfg=wf_cfg,
                policy_name=name,
                cvar_cfg=cvar_cfg,
            )
            baseline_returns[name] = ba.portfolio_returns

    # ---------- 5. Benchmark (SPY) ----------
    LOG.info("Fetching benchmark %s ...", BENCHMARK)
    spy = yf.download(BENCHMARK, start=cfg.backtest.start, end=cfg.backtest.end, progress=False, auto_adjust=False)
    bench_close = spy["Adj Close"] if "Adj Close" in spy.columns else spy["Close"]
    if isinstance(bench_close, pd.DataFrame):
        bench_close = bench_close.iloc[:, 0]
    bench_ret = bench_close.pct_change().dropna()

    # ---------- 6. Report ----------
    LOG.info("Writing report to %s ...", cfg.artifacts_dir)
    summary = write_report(
        out_dir=cfg.artifacts_dir,
        strategy_returns=artifacts.portfolio_returns,
        weights=artifacts.weights,
        decisions=artifacts.decisions,
        regime_probs=artifacts.regime_probs,
        benchmark_returns=bench_ret,
        baselines=baseline_returns,
        rebal_freq_days=wf_cfg.rebal_freq_days,
    )
    LOG.info("Done. Headline metrics:")
    for k in ("strategy", "benchmark", *baseline_returns.keys()):
        if k in summary:
            m = summary[k]
            LOG.info(
                "  %-20s  CAGR=%6.2f%%  Vol=%6.2f%%  Sharpe=%5.2f  MaxDD=%6.2f%%",
                k, 100 * m.get("cagr", 0), 100 * m.get("ann_vol", 0),
                m.get("sharpe", 0), 100 * m.get("max_drawdown", 0),
            )


if __name__ == "__main__":
    main()
