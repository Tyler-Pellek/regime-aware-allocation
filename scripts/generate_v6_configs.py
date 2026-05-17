"""Generate the 8 v6 experiment YAML configs from a base template.

The experiment matrix tests four independent dimensions:
  - universe       : tech (16) | multi (38)
  - train_freq     : weekly (5) | daily (1)
  - warm_start     : False | True
  - head_type      : standard | factor

We don't run the full 16-way grid (too expensive). Instead the 8 chosen
experiments isolate one knob at a time relative to a baseline so we can
attribute effects cleanly. See `EXPERIMENTS` below for the layout.

Each run writes its artifacts to `artifacts/reports/v6_<name>/`.
"""
from __future__ import annotations

import yaml
from pathlib import Path
from dataclasses import dataclass


@dataclass
class V6Spec:
    name: str
    universe: str       # "tech" or "multi"
    train_freq: int     # 1 (daily) or 5 (weekly)
    warm_start: bool
    head_type: str      # "standard" or "factor"
    notes: str = ""


EXPERIMENTS: list[V6Spec] = [
    # ---- TECH UNIVERSE (16 names) ----
    V6Spec("01_tech_baseline",          "tech",  5, False, "standard",
           "v1-style refresh: tech, weekly train, no warm, standard head. Reference point."),
    V6Spec("02_tech_warm",              "tech",  5, True,  "standard",
           "Isolates warm-start effect on tech."),
    V6Spec("03_tech_warm_factor",       "tech",  5, True,  "factor",
           "Adds factor head on top of warm-start, tech universe."),
    V6Spec("04_tech_warm_factor_daily", "tech",  1, True,  "factor",
           "Pushes training data 5x with daily snapshots."),
    # ---- MULTI-ASSET UNIVERSE (38 names) ----
    V6Spec("05_multi_baseline",         "multi", 5, False, "standard",
           "v5-style refresh as multi baseline."),
    V6Spec("06_multi_warm",             "multi", 5, True,  "standard",
           "Isolates warm-start effect on multi-asset."),
    V6Spec("07_multi_warm_factor",      "multi", 5, True,  "factor",
           "Adds factor head; should help most here (largest N)."),
    V6Spec("08_multi_warm_factor_daily","multi", 1, True,  "factor",
           "Kitchen sink: multi + warm + factor + daily training."),
]


TECH_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "FB",   "NFLX", "NVDA", "TSLA",
    "ADBE", "CRM",  "ORCL",  "INTC", "CSCO", "QCOM", "IBM",  "TXN",
]

MULTI_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "ORCL", "CSCO", "INTC",
    "JPM", "JNJ", "XOM", "PG", "WMT",
    "XLU", "XLP", "XLV", "XLE", "XLF", "XLK", "XLY", "XLI", "XLB",
    "USMV", "MTUM", "QUAL", "VLUE",
    "QQQ", "IWM", "EFA", "EEM",
    "TLT", "IEF", "LQD", "HYG", "SHY",
    "GLD", "USO", "VNQ",
]


def build_config(spec: V6Spec) -> dict:
    tickers = TECH_TICKERS if spec.universe == "tech" else MULTI_TICKERS

    # For the tech universe, sample_cov_cvar + min-CVaR worked well in v1.
    # For multi-asset, mean-variance is the only objective that doesn't degenerate.
    # We standardize on mean-variance across all v6 runs for apples-to-apples,
    # except in tech where we keep MV too (more meaningful comparison).
    cvar_mode = "mv"

    # Tech-only universes can tolerate higher concentration (only 16 names).
    w_max = 0.20 if spec.universe == "tech" else 0.15

    cfg = {
        "seed": 42,
        "artifacts_dir": f"artifacts/reports/v6_{spec.name}",
        "data": {
            "kaggle_root": "data/raw/kaggle_stock_market",
            "macro_cache": "data/cache/macro.parquet",
            "start": "2003-01-01",
            "end":   "2020-04-01",
        },
        "universe": {
            "tickers": tickers,
            "min_history": 60,
        },
        "regime": {
            "n_states": 3,
            "covariance_type": "full",
            "n_iter": 200,
        },
        "model": {
            "d_model": 192,
            "n_heads": 8,
            "n_layers": 5,
            "ff_mult": 4,
            "dropout": 0.15,
            "chol_min_diag": 0.01,
            "head_type": spec.head_type,
            "n_factors": 4,
        },
        "training": {
            "window": 60,
            "horizon": 5,
            "batch_size": 32,
            "epochs": 30,
            "lr": 0.0003,
            "weight_decay": 0.0005,
            "grad_clip": 1.0,
            "val_fraction": 0.15,
            "early_stop_patience": 6,
            "device": "cpu",
            "anchor_lambda": 0.0,
        },
        "cvar": {
            "mode": cvar_mode,
            "risk_aversion": 25.0,
            "demean_mu": False,
            "alpha": 0.95,
            "turnover_lambda": 0.005,
            "long_only": True,
            "w_min": 0.0,
            "w_max": w_max,
            "gross_max": 1.0,
            "cash_allowed": False,
            "min_invested": 0.0,
            "vol_cap_weekly": 0.04,
            "solver": "CLARABEL",
        },
        "backtest": {
            "start": "2005-01-03",
            "end":   "2020-04-01",
            "initial_train_end": "2007-12-31",
            "rebal_freq_days": 5,
            "train_freq_days": spec.train_freq,
            "fold_test_days": 252,
            "train_lookback_days": None,
            "transaction_cost_bps": 2.0,
            "shrinkage_alpha": 0.7,
            "use_model_mu": False,
            "warm_start": spec.warm_start,
            "warm_start_lr_scale": 0.5,
            "baselines": ["equal_weight", "inverse_vol", "sample_cov_cvar"],
        },
    }
    return cfg


def write_all(out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for spec in EXPERIMENTS:
        cfg = build_config(spec)
        path = out_dir / f"{spec.name}.yaml"
        with path.open("w") as fh:
            # Pin notes as a top-of-file comment
            fh.write(f"# v6 experiment: {spec.name}\n#   {spec.notes}\n#\n")
            fh.write(f"# Dimensions: universe={spec.universe}, train_freq={spec.train_freq}d, "
                     f"warm_start={spec.warm_start}, head={spec.head_type}\n\n")
            yaml.safe_dump(cfg, fh, sort_keys=False, default_flow_style=False)
        print(f"  wrote {path}")


if __name__ == "__main__":
    out = Path(__file__).resolve().parent.parent / "configs" / "v6"
    write_all(out)
    print(f"\nGenerated {len(EXPERIMENTS)} v6 configs in {out}")
