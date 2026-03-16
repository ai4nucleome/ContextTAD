#!/bin/bash
# exp6: multi-coverage TAD counts and CTCF ChIA-PET support
# Stage 1: generate L0/L1+ files for each tool and coverage
# Stage 2: aggregate per-coverage and combined CSV metrics

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
BASE_DIR="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"
CONDA_ENV="robustad"

declare -A COV_NAME
COV_NAME[4000M]="4000M"
COV_NAME[2000M]="2000M"
COV_NAME[1000M]="1000M"
COV_NAME[500M]="500M"
COV_NAME[250M]="250M"
COV_NAME[125M]="125M"
COV_NAME[62_5M]="62.5M"

COV_LIST=(4000M 2000M 1000M 500M 250M 125M 62_5M)

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
TOOL_BED[Armatus]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_Armatus.bed"
TOOL_BED[Arrowhead]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_Arrowhead.bed"
TOOL_BED[CaTCH]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_CaTCH.bed"
TOOL_BED[Domaincall]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_DI.bed"
TOOL_BED[EAST]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_EAST2.bed"
TOOL_BED[Grinch]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_grinch.bed"
TOOL_BED[HiCSeg]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_HiCSeg.bed"
TOOL_BED[IC-Finder]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_icfinder.bed"
TOOL_BED[OnTAD]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_OnTAD.bed"
TOOL_BED[GMAP]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_rGMAP.bed"
TOOL_BED[TopDom]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_TopDom.bed"
TOOL_BED[RefHiC]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_refhic.bed"
TOOL_BED[deDoc]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_deDoc.deDocM.bed"
TOOL_BED[HiTAD]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_hitad.bed"
TOOL_BED[ContextTAD]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_contexttad.bed"
TOOL_BED[RobusTAD]="4DNFIXP4QG5B_Rao2014_GM12878_%s_5K_robustad.good.bed"

EXTRACT_L0="$SCRIPT_DIR/extractL0.py"
EXTRACT_L1="$SCRIPT_DIR/extractL1+.py"

sort_bed_file() {
  local input_file="$1"
  local output_file="$2"
  LC_ALL=C sort -k1,1 -k2,2n -k3,3n "$input_file" > "$output_file"
}

echo "=========================================="
echo "exp6: multi-coverage CTCF ChIA-PET support"
echo "=========================================="
echo "[1/2] Generate L0/L1+ BED splits"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

TOTAL_TOOLS=0
TOTAL_SUCCESS=0
TOTAL_SKIPPED=0
FAILED_TOOLS=()

cd "$BASE_DIR"
for cov_dir in "${COV_LIST[@]}"; do
  cov_name="${COV_NAME[$cov_dir]}"
  echo
  echo "[coverage] $cov_name"

  for tool in "${TOOL_LIST[@]}"; do
    tpl="${TOOL_BED[$tool]:-}"
    if [[ -z "$tpl" ]]; then
      echo "[skip] $cov_name/$tool: missing bed template"
      continue
    fi

    tool_dir="$BASE_DIR/$cov_dir/$tool"
    if [[ ! -d "$tool_dir" ]]; then
      echo "[skip] $cov_name/$tool: missing directory"
      continue
    fi

    bed_name=$(printf "$tpl" "$cov_name")
    input_file="$tool_dir/$bed_name"
    if [[ ! -f "$input_file" ]]; then
      echo "[skip] $cov_name/$tool: missing bed $bed_name"
      continue
    fi

    l0_output="$tool_dir/${bed_name}.L0"
    l1_output="$tool_dir/${bed_name}.L1+"

    TOTAL_TOOLS=$((TOTAL_TOOLS + 1))
    tool_success=true

    if [[ -f "$l0_output" && -f "$l1_output" ]]; then
      echo "[skip] $cov_name/$tool/$bed_name: L0/L1+ already exist"
      TOTAL_SKIPPED=$((TOTAL_SKIPPED + 1))
      continue
    fi

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
      echo "[ok] $cov_name/$tool/$bed_name"
      TOTAL_SUCCESS=$((TOTAL_SUCCESS + 1))
    else
      echo "[fail] $cov_name/$tool/$bed_name"
      FAILED_TOOLS+=("$cov_name/$tool/$bed_name")
    fi
  done
done

echo "[summary] processed=$TOTAL_TOOLS success=$TOTAL_SUCCESS skipped=$TOTAL_SKIPPED"
if [[ ${#FAILED_TOOLS[@]} -gt 0 ]]; then
  echo "[summary] failed=${#FAILED_TOOLS[@]}"
  for t in "${FAILED_TOOLS[@]}"; do
    echo "  - $t"
  done
fi

echo
echo "[2/2] Aggregate exp6 metrics"
cd "$SCRIPT_DIR"

if python compute_results.py; then
  echo "[done] exp6 completed"
  echo "[output] $SCRIPT_DIR/exp6_csv/exp6_results_all.csv"
else
  echo "[error] compute_results.py failed"
  exit 1
fi
