#!/bin/bash
# exp1: GM12878 250M TAD counts and CTCF ChIA-PET support
# Stage 1: generate .L0 and .L1+ files for each tool bed
# Stage 2: aggregate evaluation metrics into CSV via compute_results.py

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
DATA_DIR="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}/250M"
CONDA_ENV="robustad"

TOOL_LIST=(
  Armatus
  Arrowhead
  CaTCH
  Domaincall
  EAST
  Grinch
  HiCSeg
  IC-Finder
  OnTAD
  GMAP
  TopDom
  RefHiC
  deDoc
  HiTAD
  ContextTAD
  RobusTAD
)

declare -A TOOL_BED
TOOL_BED[Armatus]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Armatus.bed"
TOOL_BED[Arrowhead]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Arrowhead.bed"
TOOL_BED[CaTCH]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_CaTCH.bed"
TOOL_BED[Domaincall]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_DI.bed"
TOOL_BED[EAST]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_EAST2.bed"
TOOL_BED[Grinch]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_grinch.bed"
TOOL_BED[HiCSeg]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_HiCSeg.bed"
TOOL_BED[IC-Finder]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_icfinder.bed"
TOOL_BED[OnTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_OnTAD.bed"
TOOL_BED[GMAP]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_rGMAP.bed"
TOOL_BED[TopDom]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_TopDom.bed"
TOOL_BED[RefHiC]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_refhic.bed"
TOOL_BED[deDoc]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_deDoc.deDocM.bed"
TOOL_BED[HiTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_hitad.bed"
TOOL_BED[ContextTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_contexttad.bed"
TOOL_BED[RobusTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"

EXTRACT_L0="$SCRIPT_DIR/extractL0.py"
EXTRACT_L1="$SCRIPT_DIR/extractL1+.py"

sort_bed_file() {
  local input_file="$1"
  local output_file="$2"
  LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$input_file" > "$output_file"
}

echo "=========================================="
echo "exp1: GM12878 250M CTCF ChIA-PET support"
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

for tool in "${TOOL_LIST[@]}"; do
  base="${TOOL_BED[$tool]:-}"
  if [[ -z "$base" ]]; then
    echo "[skip] $tool: missing bed mapping"
    continue
  fi

  tool_dir="$DATA_DIR/$tool"
  [[ -d "$tool_dir" ]] || { echo "[skip] $tool: missing directory $tool_dir"; continue; }

  input_file="$tool_dir/$base"
  [[ -f "$input_file" ]] || { echo "[skip] $tool: missing bed $base"; continue; }

  l0_output="$tool_dir/${base}.L0"
  l1_output="$tool_dir/${base}.L1+"
  TOOL_COUNT=$((TOOL_COUNT + 1))

  if [[ -f "$l0_output" && -f "$l1_output" ]]; then
    echo "[skip] $tool/$base: L0/L1+ already exist"
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
    echo "[ok] $tool/$base"
    SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
  else
    echo "[fail] $tool/$base"
    FAILED_TOOLS+=("$tool/$base")
  fi
done

echo "[summary] processed=$TOOL_COUNT success=$SUCCESS_COUNT skipped=$SKIPPED_COUNT"
if [[ ${#FAILED_TOOLS[@]} -gt 0 ]]; then
  echo "[summary] failed=${#FAILED_TOOLS[@]}"
  for t in "${FAILED_TOOLS[@]}"; do
    echo "  - $t"
  done
fi

echo
echo "[2/2] Aggregate exp1 metrics"
cd "$SCRIPT_DIR"

if python compute_results.py; then
  echo "[done] exp1 completed"
  echo "[output] $SCRIPT_DIR/exp1_csv/exp1_results.csv"
else
  echo "[error] compute_results.py failed"
  exit 1
fi
