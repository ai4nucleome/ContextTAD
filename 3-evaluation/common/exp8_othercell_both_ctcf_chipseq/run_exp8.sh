#!/bin/bash
# exp8: cross-cell left/right boundary support against CTCF ChIP-seq

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"

echo "========================================"
echo "exp8: cross-cell CTCF ChIP-seq boundary support"
echo "========================================"
echo "work_dir: $SCRIPT_DIR"

cd "$SCRIPT_DIR"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate robustad 2>/dev/null || true
fi

if command -v python3 >/dev/null 2>&1; then
  PYTHON_BIN="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_BIN="python"
else
  echo "[error] Python interpreter not found" >&2
  exit 1
fi

echo "python: $($PYTHON_BIN --version)"
echo "python_bin: $(command -v "$PYTHON_BIN")"
echo

echo "[run] compute_results.py"
"$PYTHON_BIN" compute_results.py

echo
echo "[done] exp8 completed"
echo "[output] exp8_csv/exp8_boundary_counts.csv"
