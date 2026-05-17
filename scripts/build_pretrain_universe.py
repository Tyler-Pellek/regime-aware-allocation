"""Build the pretraining universe from the Kaggle stock-market dataset.

We want a broad cross-section of US equities + ETFs that:
  - Have history going back to at least 2005-01-01
  - Have data through 2020-04-01 (the Kaggle cutoff)
  - Are reasonably liquid (avg dollar volume > $5M / day)

Outputs a JSON file with the chosen ticker list. Used by `pretrain.py`.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from paragon.data import KagglePaths, _read_kaggle_csv


def main(
    kaggle_root: str = "data/raw/kaggle_stock_market",
    min_start: str = "2005-01-01",
    min_end: str = "2019-12-31",
    min_avg_dollar_vol: float = 5_000_000.0,    # $5M/day avg
    max_tickers: int = 200,
    out_path: str = "data/cache/pretrain_universe.json",
) -> None:
    paths = KagglePaths(Path(kaggle_root))
    candidates = []
    for folder in [paths.stocks, paths.etfs]:
        if not folder.is_dir():
            continue
        for csv in folder.glob("*.csv"):
            candidates.append(csv)
    print(f"Scanning {len(candidates)} CSV files ...")

    selected = []
    min_start_ts = pd.Timestamp(min_start)
    min_end_ts = pd.Timestamp(min_end)

    for i, csv in enumerate(candidates):
        if i % 500 == 0:
            print(f"  scanned {i}/{len(candidates)}, selected {len(selected)} so far")
        tk = csv.stem
        try:
            df = _read_kaggle_csv(csv)
        except Exception:
            continue
        if df.empty:
            continue
        if df.index.min() > min_start_ts:
            continue
        if df.index.max() < min_end_ts:
            continue
        # Liquidity filter: avg dollar volume in the last 5 years of available data
        recent = df.loc[df.index >= "2015-01-01"]
        if recent.empty:
            continue
        adv = (recent["Adj Close"] * recent["Volume"]).mean()
        if not np.isfinite(adv) or adv < min_avg_dollar_vol:
            continue
        selected.append({
            "ticker": tk,
            "first_date": str(df.index.min().date()),
            "last_date": str(df.index.max().date()),
            "n_days": int(len(df)),
            "avg_dollar_vol_5yr": float(adv),
        })

    selected.sort(key=lambda x: -x["avg_dollar_vol_5yr"])
    if max_tickers is not None and len(selected) > max_tickers:
        selected = selected[:max_tickers]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([s["ticker"] for s in selected], indent=2))
    print(f"\nSelected {len(selected)} tickers (top by avg dollar vol):")
    for s in selected[:20]:
        print(f"  {s['ticker']:<8s}  {s['first_date']} -> {s['last_date']}  "
              f"adv=${s['avg_dollar_vol_5yr']/1e6:6.1f}M")
    print(f"  ... and {max(len(selected)-20, 0)} more")
    print(f"\nWrote ticker list to {out}")


if __name__ == "__main__":
    main()
