# Paragon — Regime-Aware Dynamic Asset Allocation

Implementation of the design described in `quant_memo.pdf`:

- **Cross-Asset Transformer** with `[CTX]` token + N asset tokens (sequence = portfolio).
- **HMM regime detector** populates the `[CTX]` token with posterior probabilities.
- **Cholesky head** projects asset embeddings to `Σ = LLᵀ` (PSD by construction).
- **Mean head** predicts forward returns `μ`.
- **CVaR + L1-turnover convex optimizer** (closed-form Gaussian CVaR → small SOCP, plus a scenario-based RU formulation for non-Gaussian use).
- **Walk-forward backtester** with periodic retrain (default: ~yearly) and weekly rebalance.

## Universe

Mega-cap US tech, 16 names (canonical order in `paragon/universe.py:TECH_UNIVERSE`):

```
AAPL MSFT GOOGL AMZN META NFLX NVDA TSLA
ADBE CRM  ORCL  INTC CSCO QCOM IBM  TXN
```

`META` and `TSLA` enter the universe later in the sample (IPO 2012, 2010). The
backtester emits a per-date availability mask consumed by the Transformer's
`key_padding_mask` — assets are invisible to attention until they have ≥ 60
days of history.

## Data sources

1. **Kaggle `jacksoncrow/stock-market-dataset`** — per-ticker OHLCV CSVs
   (snapshot dated April 2020).
2. **yfinance** — macro indicators (VIX, ^GSPC, ^TNX, DX-Y.NYB, CL=F, GC=F, HYG, TLT) and the SPY benchmark.

## Setup

```bash
# (Recommended) fresh env
python -m venv .venv && source .venv/bin/activate
pip install -e .

# Get Kaggle credentials, place ~/.kaggle/kaggle.json (chmod 600)
# Then:
bash scripts/download_kaggle.sh
```

## Run a backtest

```bash
python scripts/run_backtest.py --config configs/default.yaml
```

Outputs land in `artifacts/reports/default/`:

- `summary.json`, `summary.md` — headline metrics for strategy + baselines + benchmark
- `figures/` — equity curves, drawdown, regime overlay, weights heatmap, rolling Sharpe, turnover
- `strategy_returns.csv`, `weights.csv`, `decisions.csv`, baseline + benchmark return CSVs
- `model_fold{i}.pt` — trained transformer checkpoints (one per walk-forward fold)

## Tests

```bash
pip install pytest
pytest -q tests/
```

The smoke tests use synthetic data — they require neither the Kaggle download
nor network access.

## Configuration

All knobs live in `configs/default.yaml`. The CVaR formulation, transaction cost
in bps, training window `W`, forward horizon `h`, model size, and walk-forward
fold geometry are all there. Save as a new YAML and pass it via `--config`.

## Layout

```
paragon/
├── data.py                 # Kaggle CSV loader + yfinance macro
├── universe.py             # ticker list + dynamic availability mask
├── features.py             # log returns, rolling vol, macro features, FeatureBundle
├── regime/hmm.py           # GaussianHMM wrapper with stable state ordering
├── model/transformer.py    # Cross-Asset Transformer + Cholesky head
├── model/cholesky.py       # Σ = LLᵀ + Gaussian NLL loss
├── model/input_builder.py  # snapshot tensor assembly
├── optim/cvar.py           # Gaussian closed-form CVaR + L1 (SOCP), and scenario CVaR (LP)
├── optim/baselines.py      # equal_weight, inverse_vol, sample_cov_cvar
├── training/dataset.py     # SnapshotDataset
├── training/trainer.py     # AdamW + cosine schedule + early stop
├── backtest/walkforward.py # walk-forward driver (model + baselines)
├── eval/metrics.py         # CAGR, Sharpe, MaxDD, CVaR, alpha/beta, turnover
├── eval/plots.py           # report figures
└── eval/report.py          # write_report() bundles summary + figures + CSVs
```

## Notes / known caveats

- **Train/inference `prev_w`**: training sets `prev_w = 0` in the asset-token slot
  (the model is trained as a covariance forecaster). At inference the actual
  prior weight populates that slot. The architectural rationale (memo §3) is
  sound but the train-time mismatch is a small approximation. The L1 turnover
  anchor still operates exactly through the optimizer regardless.
- **Look-ahead**: HMM is refit per fold using only data ≤ `train_end`; the
  Transformer trains on snapshots whose decision date is ≤ `train_end`; macro
  features at decision date `t` use only data through `t`.
- **Costs**: a single per-side bps charge applied to `||Δw||₁` at each rebalance.
  Slippage / borrow / shorting are not modeled (the default config is long-only).
- **Survivorship**: the Kaggle dataset is a 2020 snapshot, so any tech name that
  delisted before then is missing. The mega-cap tech universe is robust to this
  but be aware if you swap in mid-cap names.
