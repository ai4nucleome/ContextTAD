#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
EVAL_TOOLS_ROOT="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/exp9_csv}"
PYTHON_BIN="${PYTHON_BIN:-python}"

echo "========================================"
echo "exp9: 250M TAD-size statistics"
echo "========================================"
echo "tools_root: $EVAL_TOOLS_ROOT"
echo "output_dir: $OUTPUT_DIR"

EVAL_DATA_ROOT="$EVAL_DATA_ROOT" \
EVAL_TOOLS_ROOT="$EVAL_TOOLS_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
COVERAGE="${COVERAGE:-250M}" \
MAX_SIZE_BP="${MAX_SIZE_BP:-3000000}" \
  "$PYTHON_BIN" "$SCRIPT_DIR/compute_results.py"

echo "[done] exp9 completed"
