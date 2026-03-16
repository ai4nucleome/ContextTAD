#!/bin/bash
# exp7: cross-cell (K562/IMR90) TAD counts and CTCF ChIA-PET support
# Stage 1: generate .L0 and .L1+ for each tool
# Stage 2: aggregate metrics via compute_results.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
DATA_DIR="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}/other_celltypes"
CONDA_ENV="robustad"

CELL_LINES=(IMR90 K562)

TOOL_LIST=(
  Arrowhead
  GMAP
  HiTAD
  OnTAD
  RefHiC
  ContextTAD
  RobusTAD
)

declare -A TOOL_DIR
TOOL_DIR[Arrowhead]="Arrowhead"
TOOL_DIR[GMAP]="GMAP"
TOOL_DIR[HiTAD]="HiTAD"
TOOL_DIR[OnTAD]="OnTAD"
TOOL_DIR[RefHiC]="RefHic"
TOOL_DIR[ContextTAD]="ContextTAD"
TOOL_DIR[RobusTAD]="RobusTAD"

declare -A TOOL_BED_IMR90
TOOL_BED_IMR90[Arrowhead]="IMR90_Arrowhead.bed"
TOOL_BED_IMR90[GMAP]="IMR90_rGMAP.bed"
TOOL_BED_IMR90[HiTAD]="IMR90_hitad.bed"
TOOL_BED_IMR90[OnTAD]="IMR90_OnTAD.bed"
TOOL_BED_IMR90[RefHiC]="IMR90_RefHiC.bed"
TOOL_BED_IMR90[ContextTAD]="IMR90_contexttad.bed"
TOOL_BED_IMR90[RobusTAD]="IMR90_robustad_all.good.bed"

declare -A TOOL_BED_K562
TOOL_BED_K562[Arrowhead]="K562_Arrowhead.bed"
TOOL_BED_K562[GMAP]="K562_rGMAP.bed"
TOOL_BED_K562[HiTAD]="K562_hitad.bed"
TOOL_BED_K562[OnTAD]="K562_OnTAD.bed"
TOOL_BED_K562[RefHiC]="K562_RefHiC.bed"
TOOL_BED_K562[ContextTAD]="K562_contexttad.bed"
TOOL_BED_K562[RobusTAD]="K562_robustad_all.good.bed"

EXTRACT_L0="$SCRIPT_DIR/extractL0.py"
EXTRACT_L1="$SCRIPT_DIR/extractL1+.py"

sort_bed_file() {
  local input_file="$1"
  local output_file="$2"
  LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$input_file" > "$output_file"
}

echo "=========================================="
echo "exp7: cross-cell CTCF ChIA-PET support"
echo "=========================================="
echo "[1/2] Generate L0/L1+ BED splits"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

cd "$DATA_DIR"
TOOL_COUNT=0
SUCCESS_COUNT=0
SKIPPED_COUNT=0
FAILED_TOOLS=()

for cell in "${CELL_LINES[@]}"; do
  for tool in "${TOOL_LIST[@]}"; do
    if [[ "$cell" == "IMR90" ]]; then
      base="${TOOL_BED_IMR90[$tool]:-}"
    else
      base="${TOOL_BED_K562[$tool]:-}"
    fi

    if [[ -z "$base" ]]; then
      echo "[skip] $cell/$tool: missing bed mapping"
      continue
    fi

    tool_dir="$DATA_DIR/$cell/${TOOL_DIR[$tool]}"
    [[ -d "$tool_dir" ]] || { echo "[skip] $cell/$tool: missing directory $tool_dir"; continue; }

    input_file="$tool_dir/$base"
    [[ -f "$input_file" ]] || { echo "[skip] $cell/$tool: missing bed $base"; continue; }

    l0_output="$tool_dir/${base}.L0"
    l1_output="$tool_dir/${base}.L1+"

    TOOL_COUNT=$((TOOL_COUNT + 1))
    if [[ -f "$l0_output" && -f "$l1_output" ]]; then
      echo "[skip] $cell/$tool/$base: L0/L1+ already exist"
      SKIPPED_COUNT=$((SKIPPED_COUNT + 1))
      continue
    fi

    tool_success=true

    if [[ ! -f "$l0_output" ]]; then
      l0_tmp=$(mktemp)
      if python "$EXTRACT_L0" --tadfile "$input_file" --output "$l0_tmp" 2>/dev/null; then
        if [[ -s "$l0_tmp" ]]; then
          sort_bed_file "$l0_tmp" "$l0_output"
        else
          : > "$l0_output"
        fi
      else
        tool_success=false
      fi
      rm -f "$l0_tmp"
    fi

    if [[ ! -f "$l1_output" ]]; then
      l1_tmp=$(mktemp)
      if python "$EXTRACT_L1" --tadfile "$input_file" --output "$l1_tmp" 2>/dev/null; then
        if [[ -s "$l1_tmp" ]]; then
          sort_bed_file "$l1_tmp" "$l1_output"
        else
          : > "$l1_output"
        fi
      else
        tool_success=false
      fi
      rm -f "$l1_tmp"
    fi

    if [[ "$tool_success" == true ]]; then
      echo "[ok] $cell/$tool/$base"
      SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
      echo "[fail] $cell/$tool/$base"
      FAILED_TOOLS+=("$cell/$tool/$base")
    fi
  done
done

echo "[summary] processed=$TOOL_COUNT success=$SUCCESS_COUNT skipped=$SKIPPED_COUNT"
if [[ ${#FAILED_TOOLS[@]} -gt 0 ]]; then
  echo "[summary] failed=${#FAILED_TOOLS[@]}"
  for t in "${FAILED_TOOLS[@]}"; do
    echo "  - $t"
  done
fi

echo
echo "[2/2] Aggregate exp7 metrics"
cd "$SCRIPT_DIR"

if python compute_results.py; then
  echo "[done] exp7 completed"
  echo "[output] $SCRIPT_DIR/exp7_csv/exp7_results.csv"
else
  echo "[error] compute_results.py failed"
  exit 1
fi
