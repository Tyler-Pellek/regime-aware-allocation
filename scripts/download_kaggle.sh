#!/usr/bin/env bash
# Download the Jackson Crow stock-market-dataset (huge — ~700MB unzipped, ~8000 tickers).
# Requires: kaggle CLI installed and ~/.kaggle/kaggle.json present (chmod 600).
#   pip install kaggle
#   Place kaggle.json from https://www.kaggle.com/settings into ~/.kaggle/

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="${ROOT}/data/raw/kaggle_stock_market"
mkdir -p "${DEST}"

if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found. Run: pip install kaggle" >&2
  exit 1
fi

echo "Downloading jacksoncrow/stock-market-dataset into ${DEST} ..."
kaggle datasets download -d jacksoncrow/stock-market-dataset -p "${DEST}" --unzip

echo "Done. Top-level contents:"
ls -la "${DEST}"
