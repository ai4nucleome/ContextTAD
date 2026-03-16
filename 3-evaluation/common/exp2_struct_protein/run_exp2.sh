#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
DATA_DIR="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}/250M"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/struct_protein}"
FIG_DIR="${FIG_DIR:-$SCRIPT_DIR/exp2_csv}"
R_SCRIPT="${R_SCRIPT:-$SCRIPT_DIR/r_scripts/StructProt_EnrichBoundaries_script.R}"
PROTEIN_PEAKS_DIR="${PROTEIN_PEAKS_DIR:-$SCRIPT_DIR/data/protein_peaks/}"
CHR_SIZES="${CHR_SIZES:-$SCRIPT_DIR/GRCh38.chrom.sizes}"
RESOLUTION="${RESOLUTION:-5}"
CONDA_ENV="${CONDA_ENV:-robustad}"
TOOL_FILTER="${TOOL_FILTER:-}"
CONTEXTTAD_BED_OVERRIDE="${CONTEXTTAD_BED_OVERRIDE:-}"
CLEAN_OUTPUT="${CLEAN_OUTPUT:-1}"
PROTEIN_PEAKS_DIR="${PROTEIN_PEAKS_DIR%/}/"

echo "========================================"
echo "exp2: Structural protein enrichment"
echo "========================================"
echo "data_dir:     $DATA_DIR"
echo "output_dir:   $OUTPUT_DIR"
echo "fig_dir:      $FIG_DIR"
echo "r_script:     $R_SCRIPT"
echo "peaks_dir:    $PROTEIN_PEAKS_DIR"
echo "chr_sizes:    $CHR_SIZES"
echo "resolution:   ${RESOLUTION}kb"
if [[ -n "$CONTEXTTAD_BED_OVERRIDE" ]]; then
  echo "contexttad bed:   $CONTEXTTAD_BED_OVERRIDE (override)"
fi
echo

if [[ ! -d "$DATA_DIR" ]]; then
  echo "[error] data directory not found: $DATA_DIR" >&2
  exit 2
fi
if [[ ! -f "$R_SCRIPT" ]]; then
  echo "[error] R script not found: $R_SCRIPT" >&2
  exit 2
fi
if [[ ! -f "$CHR_SIZES" ]]; then
  echo "[error] chromosome size file not found: $CHR_SIZES" >&2
  exit 2
fi
for prot in CTCF RAD21 SMC3; do
  if [[ ! -f "${PROTEIN_PEAKS_DIR}${prot}_peaks.bed" ]]; then
    echo "[error] peak file missing: ${PROTEIN_PEAKS_DIR}${prot}_peaks.bed" >&2
    exit 2
  fi
done
if [[ -n "$CONTEXTTAD_BED_OVERRIDE" && ! -f "$CONTEXTTAD_BED_OVERRIDE" ]]; then
  echo "[error] override bed not found: $CONTEXTTAD_BED_OVERRIDE" >&2
  exit 2
fi

if [[ "$CLEAN_OUTPUT" == "1" ]]; then
  rm -rf "$OUTPUT_DIR" "$FIG_DIR"
fi
mkdir -p "$OUTPUT_DIR" "$FIG_DIR"

CONDA_HOME="${CONDA_HOME:-$HOME/anaconda3}"
if [[ -f "$CONDA_HOME/etc/profile.d/conda.sh" ]]; then
  source "$CONDA_HOME/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV" 2>/dev/null || true
fi

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
  RobusTAD
  ContextTAD
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
TOOL_BED[RobusTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"
TOOL_BED[ContextTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_contexttad.bed"

declare -A TOOL_KEEP
if [[ -n "$TOOL_FILTER" ]]; then
  IFS=',' read -ra _tool_arr <<< "$TOOL_FILTER"
  for t in "${_tool_arr[@]}"; do
    t="${t// /}"
    [[ -n "$t" ]] && TOOL_KEEP["$t"]=1
  done
fi

n_ok=0
n_skip=0
for tool_name in "${TOOL_LIST[@]}"; do
  if [[ -n "$TOOL_FILTER" && -z "${TOOL_KEEP[$tool_name]:-}" ]]; then
    continue
  fi

  base="${TOOL_BED[$tool_name]:-}"
  if [[ -z "$base" ]]; then
    echo "[skip] $tool_name: missing bed mapping"
    n_skip=$((n_skip + 1))
    continue
  fi

  bed_path="$DATA_DIR/$tool_name/$base"
  if [[ "$tool_name" == "ContextTAD" && -n "$CONTEXTTAD_BED_OVERRIDE" ]]; then
    bed_path="$CONTEXTTAD_BED_OVERRIDE"
  fi
  if [[ ! -f "$bed_path" ]]; then
    echo "[skip] $tool_name: missing bed $bed_path"
    n_skip=$((n_skip + 1))
    continue
  fi

  work_tmp="$SCRIPT_DIR/.tmp_exp2_${tool_name}"
  rm -rf "$work_tmp"
  mkdir -p "$work_tmp"

  echo "[run] $tool_name"
  pushd "$work_tmp" >/dev/null
  mkdir -p "$tool_name"
  if Rscript "$R_SCRIPT" \
      "$bed_path" \
      "$RESOLUTION" \
      "$tool_name" \
      "$CHR_SIZES" \
      "$PROTEIN_PEAKS_DIR" >/dev/null 2>&1; then
    if [[ -d "$tool_name" ]] && find "$tool_name" -type f -name "StructProteins_chr*_res${RESOLUTION}kb.txt" | head -1 >/dev/null; then
      rm -rf "$OUTPUT_DIR/$tool_name"
      mv "$tool_name" "$OUTPUT_DIR/"
      n_ok=$((n_ok + 1))
    else
      echo "[skip] $tool_name: empty output"
      n_skip=$((n_skip + 1))
    fi
  else
    echo "[skip] $tool_name: Rscript failed"
    n_skip=$((n_skip + 1))
  fi
  popd >/dev/null
  rm -rf "$work_tmp"
done

echo "[info] finished R stage: ok=$n_ok skip=$n_skip"

OUTPUT_TSV="$OUTPUT_DIR/StructProteins_allTools.tsv"
{
  echo -e "TadsFile\tresolution_kb\tprotein\tdomains_ratio\tfc_over_bg\tpval_of_peak"
  for tool_dir in "$OUTPUT_DIR"/*/; do
    [[ -d "$tool_dir" ]] || continue
    tool_name="$(basename "$tool_dir")"
    result_file="$(find "$tool_dir" -type f -name "StructProteins_chr*_res${RESOLUTION}kb.txt" | head -1 || true)"
    [[ -f "$result_file" ]] || continue
    tail -n +2 "$result_file" | awk -v tool="$tool_name" 'BEGIN{OFS="\t"} {print tool, $2, $3, $4, $5, $6}'
  done
} > "$OUTPUT_TSV"

echo "[info] build summary TSV: $OUTPUT_TSV"
EXP6_DATA_FILE="$OUTPUT_TSV" EXP6_OUTPUT_DIR="$FIG_DIR" python "$SCRIPT_DIR/compute_results.py"

echo "[done]"
echo "  - $OUTPUT_TSV"
echo "  - $FIG_DIR/exp2_results.csv"
