"""Colab A100 quickstart for Paragon.

Open a fresh Colab notebook (Runtime -> Change runtime type -> A100 GPU),
then paste each "Cell" block below into a separate notebook cell and run
top-to-bottom.

Prereqs you need on hand:
  - kaggle.json  (https://www.kaggle.com/settings -> Create New Token).
    Upload it via Colab's Files panel (left sidebar -> folder icon -> Upload).
"""
# fmt: off

COLAB_CELLS = r"""
# ============================================================================
# Cell 1 — Clone the repo and install dependencies
# ============================================================================
!git clone https://github.com/Tyler-Pellek/regime-aware-allocation.git /content/paragon || (cd /content/paragon && git pull)
%cd /content/paragon
!pip install -q -r requirements.txt

# Verify GPU is wired up
import torch
print("CUDA:", torch.cuda.is_available(), "| device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU only")


# ============================================================================
# Cell 2 — Place your Kaggle API token
# ============================================================================
# Upload kaggle.json via the Files panel first (drag it into the file tree).
import os, shutil, pathlib
src = pathlib.Path('/content/kaggle.json')
assert src.exists(), "Upload kaggle.json to /content/ via the Files panel before running this cell."
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy(src, os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)
!kaggle datasets list -s "stock-market-dataset" --max-size 1 | head -5   # auth check


# ============================================================================
# Cell 3 — Download the Kaggle stock-market-dataset (~700 MB unzipped)
# ============================================================================
!bash scripts/download_kaggle.sh
!ls data/raw/kaggle_stock_market/stocks/ | wc -l    # should print ~5800
!ls data/raw/kaggle_stock_market/etfs/  | wc -l    # should print ~2100


# ============================================================================
# Cell 4 — Smoke test the pipeline on synthetic data (~10 s)
# ============================================================================
!pip install -q pytest && PYTHONPATH=. pytest -q tests/


# ============================================================================
# Cell 5 — Run the full A100 backtest (2005-2020, weekly rebalance)
# ============================================================================
# Streaming output via `-u` so you see fold progress live.
# Expected runtime on A100: ~20-45 min depending on number of folds.
!PYTHONPATH=. python -u scripts/run_backtest.py --config configs/default.yaml


# ============================================================================
# Cell 6 — Inspect headline results
# ============================================================================
import json, pathlib, pandas as pd
summary = json.loads(pathlib.Path('artifacts/reports/default/summary.json').read_text())
rows = []
for name, m in summary.items():
    rows.append({
        'strategy': name,
        'CAGR':   f"{m.get('cagr', 0)*100:6.2f}%",
        'Vol':    f"{m.get('ann_vol', 0)*100:6.2f}%",
        'Sharpe': f"{m.get('sharpe', 0):5.2f}",
        'MaxDD':  f"{m.get('max_drawdown', 0)*100:6.2f}%",
        'Calmar': f"{m.get('calmar', 0):5.2f}",
        'Alpha':  f"{m.get('alpha_ann', 0)*100:6.2f}%" if 'alpha_ann' in m else '',
    })
print(pd.DataFrame(rows).to_string(index=False))


# ============================================================================
# Cell 7 — View the report figures inline
# ============================================================================
from IPython.display import Image, display
for png in [
    'equity_curves.png',
    'drawdown.png',
    'regime_overlay.png',
    'weights_heatmap.png',
    'rolling_sharpe.png',
    'turnover.png',
]:
    p = f'artifacts/reports/default/figures/{png}'
    print(f'\n--- {png} ---')
    display(Image(filename=p))


# ============================================================================
# Cell 8 — Zip + download the full report (figures, CSVs, summary, model ckpts)
# ============================================================================
!cd artifacts/reports && zip -qr /content/paragon_default_report.zip default
from google.colab import files
files.download('/content/paragon_default_report.zip')


# ============================================================================
# Cell 9 (optional) — Iterate: pull latest code changes and re-run
# ============================================================================
# After editing the repo locally and `git push`-ing, run this to pick up changes
# without re-downloading the Kaggle data:
%cd /content/paragon
!git pull
# (Then re-run Cell 5.)


# ============================================================================
# Cell 10 (optional) — Custom config sweep
# ============================================================================
# Drop a new YAML into /content/paragon/configs/ and point the runner at it:
# %%writefile configs/highvol.yaml
# ... yaml contents ...
# !PYTHONPATH=. python -u scripts/run_backtest.py --config configs/highvol.yaml
"""

if __name__ == "__main__":
    print(COLAB_CELLS)
