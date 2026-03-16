#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAIN_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$MAIN_DIR/../.." && pwd)"
EXPS_ROOT="${EXPS_ROOT:-$PROJECT_ROOT/3-evaluation/common}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
TOOLS_ROOT="${TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"

RUN_ID="${1:-exp2_main_$(date +%Y%m%d_%H%M%S)}"
DEFAULT_CONTEXTTAD_BED="$PROJECT_ROOT/2-training/step2_infer_decode/outputs/$RUN_ID/beds/GM12878_250M_contexttad.bed"
USER_OVERRIDE_PROVIDED=0
if [[ -n "${2-}" ]]; then
  USER_OVERRIDE_PROVIDED=1
fi
CONTEXTTAD_BED_OVERRIDE="${2:-$DEFAULT_CONTEXTTAD_BED}"
TOOL_FILTER="${3:-}"

RUN_DIR="$MAIN_DIR/outputs/runs/$RUN_ID"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$LOG_DIR"

OUTPUT_DIR="$RUN_DIR/exp2_struct_protein"
FIG_DIR="$RUN_DIR/exp2_csv"
EXP2_SCRIPT="$EXPS_ROOT/exp2_struct_protein/run_exp2.sh"

if [[ ! -x "$EXP2_SCRIPT" ]]; then
  echo "[error] missing exp2 runner: $EXP2_SCRIPT" >&2
  exit 2
fi

if [[ "$USER_OVERRIDE_PROVIDED" == "1" && -z "$TOOL_FILTER" ]]; then
  TOOL_FILTER="ContextTAD"
fi

echo "[run] exp2 run_id=$RUN_ID"
EVAL_DATA_ROOT="$EVAL_DATA_ROOT" \
EVAL_TOOLS_ROOT="$TOOLS_ROOT" \
OUTPUT_DIR="$OUTPUT_DIR" \
FIG_DIR="$FIG_DIR" \
CONTEXTTAD_BED_OVERRIDE="$CONTEXTTAD_BED_OVERRIDE" \
TOOL_FILTER="$TOOL_FILTER" \
CLEAN_OUTPUT=1 \
bash "$EXP2_SCRIPT" > "$LOG_DIR/exp2.log" 2>&1

if [[ -f "$FIG_DIR/exp2_results.csv" && -z "$TOOL_FILTER" ]]; then
  cp "$FIG_DIR/exp2_results.csv" "$MAIN_DIR/outputs/current/exp2_results.csv"
fi

echo "[done]"
echo "  run_dir: $RUN_DIR"
echo "  csv:     $FIG_DIR/exp2_results.csv"
