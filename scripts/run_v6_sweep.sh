#!/usr/bin/env bash
# Run the full 8-experiment v6 sweep sequentially. Each experiment is a fresh
# walk-forward backtest with its own config + artifacts directory. After all
# runs complete, the comparison script aggregates results into a single table.
#
# Usage:
#   bash scripts/run_v6_sweep.sh
#
# Estimated runtime on Mac CPU: 5-9 hours (designed to run overnight).
# Per-experiment outputs land in artifacts/reports/v6_<name>/ and the
# full per-experiment log is in artifacts/reports/v6_<name>/run.log.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

LOG_DIR="${ROOT}/artifacts/reports"
mkdir -p "${LOG_DIR}"

CONFIGS=(
  configs/v6/01_tech_baseline.yaml
  configs/v6/02_tech_warm.yaml
  configs/v6/03_tech_warm_factor.yaml
  configs/v6/04_tech_warm_factor_daily.yaml
  configs/v6/05_multi_baseline.yaml
  configs/v6/06_multi_warm.yaml
  configs/v6/07_multi_warm_factor.yaml
  configs/v6/08_multi_warm_factor_daily.yaml
)

START_TIME=$(date +%s)
echo "=== v6 sweep starting at $(date) ==="
echo "    ${#CONFIGS[@]} experiments to run sequentially"
echo

for cfg in "${CONFIGS[@]}"; do
  name=$(basename "${cfg}" .yaml)
  out_dir="${LOG_DIR}/v6_${name#*_}"
  log_file="${out_dir}/run.log"

  echo "------------------------------------------------------------"
  echo "[$(date +%H:%M:%S)] Starting: ${name}"
  echo "  config:  ${cfg}"
  echo "  output:  ${out_dir}"

  # Clean prior output for this experiment but preserve siblings
  rm -rf "${out_dir}" 2>/dev/null || true
  mkdir -p "${out_dir}"

  exp_start=$(date +%s)
  if PYTHONPATH=. python -u scripts/run_backtest.py --config "${cfg}" \
        > "${log_file}" 2>&1; then
    exp_end=$(date +%s)
    elapsed=$(( exp_end - exp_start ))
    echo "  OK      (${elapsed}s)"
    if [ -f "${out_dir}/summary.json" ]; then
      echo "  headline:"
      python -c "
import json, pathlib
s = json.loads(pathlib.Path('${out_dir}/summary.json').read_text())
for name in ('strategy', 'sample_cov_cvar', 'equal_weight', 'benchmark'):
    if name in s:
        m = s[name]
        print(f'    {name:18s}  CAGR={m.get(\"cagr\",0)*100:6.2f}%  Vol={m.get(\"ann_vol\",0)*100:6.2f}%  '
              f'Sharpe={m.get(\"sharpe\",0):5.2f}  MaxDD={m.get(\"max_drawdown\",0)*100:6.2f}%')
"
    fi
  else
    exp_end=$(date +%s)
    elapsed=$(( exp_end - exp_start ))
    echo "  FAIL    (${elapsed}s) — see ${log_file}"
  fi
  echo
done

END_TIME=$(date +%s)
TOTAL=$(( END_TIME - START_TIME ))
echo "============================================================"
echo "v6 sweep complete in ${TOTAL}s ($(printf '%dh %dm' $((TOTAL/3600)) $(((TOTAL%3600)/60))))"
echo

echo "Running comparison script ..."
PYTHONPATH=. python scripts/compare_v6.py | tee "${LOG_DIR}/v6_comparison.txt"
echo
echo "Full comparison table written to ${LOG_DIR}/v6_comparison.txt"
