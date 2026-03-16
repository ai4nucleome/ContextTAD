#!/bin/bash
# exp4: right-boundary CTCF ChIP-seq support on GM12878

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"

echo "========================================"
echo "exp4: right-boundary CTCF ChIP-seq"
echo "========================================"
echo "work_dir: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate robustad 2>/dev/null || true
fi

echo "python: $(python --version)"
echo "python_bin: $(which python)"
echo

echo "[run] compute_results.py"
python compute_results.py

echo
echo "[done] exp4 completed"
echo "[output] exp4_csv/exp4_results.csv"
