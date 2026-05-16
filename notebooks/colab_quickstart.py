"""Colab A100 quickstart — paste this into a single cell.

This script handles install + Kaggle credentials + dataset download + run.
Expects:
  - this repo cloned at /content/ParagonProject (or override REPO_DIR)
  - a Kaggle API token uploaded to /content/kaggle.json (download from
    https://www.kaggle.com/settings -> Create New Token)
"""
# fmt: off
COLAB_SETUP = r"""
# === Cell 1: clone + install ===
!git clone https://github.com/Tyler-Pellek/regime-aware-allocation.git /content/paragon || true
%cd /content/paragon
!pip install -q -r requirements.txt

# === Cell 2: Kaggle credentials ===
import os, shutil
os.makedirs(os.path.expanduser('~/.kaggle'), exist_ok=True)
shutil.copy('/content/kaggle.json', os.path.expanduser('~/.kaggle/kaggle.json'))
os.chmod(os.path.expanduser('~/.kaggle/kaggle.json'), 0o600)

# === Cell 3: download Kaggle data ===
!bash scripts/download_kaggle.sh

# === Cell 4: run backtest on the A100 ===
!python scripts/run_backtest.py --config configs/default.yaml
"""
print(COLAB_SETUP)
