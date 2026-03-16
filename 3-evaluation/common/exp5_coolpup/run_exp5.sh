#!/bin/bash
# exp5: coolpup benchmark for GM12878 250M across selected tools
# Stage 1: generate pileup text files per tool
# Stage 2: draw combined figure

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
cd "$SCRIPT_DIR"

EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
BED_DIR="${BED_DIR:-${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}/250M}"
MCOOL_FILE="${MCOOL_FILE:-$EVAL_DATA_ROOT/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac1.mcool::/resolutions/5000}"
DATA_DIR="${DATA_DIR:-data}"

TOOL_LIST=(
  RobusTAD
  Arrowhead
  EAST
  HiTAD
  IC-Finder
  OnTAD
  RefHiC
  GMAP
  TopDom
  Armatus
  Domaincall
  CaTCH
  deDoc
  Grinch
  HiCSeg
)

declare -A TOOL_BED
TOOL_BED[RobusTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"
TOOL_BED[Arrowhead]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Arrowhead.bed"
TOOL_BED[EAST]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_EAST2.bed"
TOOL_BED[HiTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_hitad.bed"
TOOL_BED[IC-Finder]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_icfinder.bed"
TOOL_BED[OnTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_OnTAD.bed"
TOOL_BED[RefHiC]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_refhic.bed"
TOOL_BED[GMAP]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_rGMAP.bed"
TOOL_BED[TopDom]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_TopDom.bed"
TOOL_BED[Armatus]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Armatus.bed"
TOOL_BED[Domaincall]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_DI.bed"
TOOL_BED[CaTCH]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_CaTCH.bed"
TOOL_BED[deDoc]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_deDoc.deDocM.bed"
TOOL_BED[Grinch]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_grinch.bed"
TOOL_BED[HiCSeg]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_HiCSeg.bed"

echo "=========================================="
echo "exp5: coolpup benchmark"
echo "=========================================="
echo "BED_DIR: $BED_DIR"
echo "MCOOL_FILE: $MCOOL_FILE"

if [[ ! -d "$BED_DIR" ]]; then
  echo "[error] BED directory not found: $BED_DIR"
  exit 2
fi

success_count=0
skip_count=0
failed_tools=()

for tool_dir in "${TOOL_LIST[@]}"; do
  echo
  echo "=========================================="
  echo "Processing: $tool_dir"
  echo "=========================================="

  bed_filename="${TOOL_BED[$tool_dir]}"
  if [[ -z "$bed_filename" ]]; then
    echo "[error] bed mapping not found for: $tool_dir"
    failed_tools+=("$tool_dir")
    continue
  fi

  bed_file="$BED_DIR/$tool_dir/$bed_filename"
  if [[ ! -f "$bed_file" ]]; then
    echo "[error] bed file not found: $bed_file"
    failed_tools+=("$tool_dir")
    continue
  fi

  tool_output_dir="$SCRIPT_DIR/$DATA_DIR/$tool_dir"
  mkdir -p "$tool_output_dir"

  expected_txt_pattern="${bed_filename%.bed}_10-shifts_local_rescaled.txt"
  existing_txt=$(find "$tool_output_dir" -maxdepth 1 -name "*_10-shifts_local_rescaled.txt" | head -1)
  if [[ -n "$existing_txt" && -f "$existing_txt" ]]; then
    echo "[skip] txt file already exists: $existing_txt"
    skip_count=$((skip_count + 1))
    success_count=$((success_count + 1))
    continue
  fi

  cd "$tool_output_dir"
  echo "[run] coolpup.py"
  if ! coolpup.py --ignore_diags 0 --local --rescale -p 8 --seed 2026 "$MCOOL_FILE" "$bed_file"; then
    echo "[error] coolpup.py failed for $tool_dir"
    failed_tools+=("$tool_dir")
    cd "$SCRIPT_DIR"
    continue
  fi

  clpy_file=$(find "$tool_output_dir" -maxdepth 1 -name "*.clpy" | head -1)
  if [[ -z "$clpy_file" || ! -f "$clpy_file" ]]; then
    echo "[error] no .clpy output found for $tool_dir"
    failed_tools+=("$tool_dir")
    cd "$SCRIPT_DIR"
    continue
  fi

  txt_file="${clpy_file%.clpy}.txt"
  echo "[run] clpy2txt.py"
  if ! python "$SCRIPT_DIR/clpy2txt.py" "$clpy_file" "$txt_file"; then
    echo "[error] clpy2txt.py failed for $tool_dir"
    failed_tools+=("$tool_dir")
    cd "$SCRIPT_DIR"
    continue
  fi

  echo "[ok] completed $tool_dir"
  success_count=$((success_count + 1))
  cd "$SCRIPT_DIR"
done

echo
echo "=========================================="
echo "Stage-1 summary"
echo "=========================================="
echo "success: $success_count / ${#TOOL_LIST[@]}"
if [[ $skip_count -gt 0 ]]; then
  echo "skipped: $skip_count"
fi
if [[ ${#failed_tools[@]} -gt 0 ]]; then
  echo "failed: ${failed_tools[*]}"
fi

echo
echo "=========================================="
echo "Stage-2: draw combined figure"
echo "=========================================="
FIG_DIR="$SCRIPT_DIR/exp5_figs"
mkdir -p "$FIG_DIR"
python plot_pileup.py --all "$DATA_DIR" "$FIG_DIR/fig3_15toolsPU.pdf"

echo
echo "[done] exp5 completed"
