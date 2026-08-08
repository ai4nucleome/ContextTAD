#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
EVAL_TOOLS_ROOT="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"
COVERAGE="${COVERAGE:-250M}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/results}"
CSV_DIR="${CSV_DIR:-$SCRIPT_DIR/exp10_csv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-bedtools}"
REUSE_EXISTING="${REUSE_EXISTING:-0}"

EXAMPLE_DATA="$EVAL_DATA_ROOT/TAD-benchmarking-scripts/Example_Data"
H3K27ME3_BEDGRAPH="${H3K27ME3_BEDGRAPH:-$EXAMPLE_DATA/ENCFF039JOT_H3K27me3_chr15-17_hg38.bedGraph}"
H3K36ME3_BEDGRAPH="${H3K36ME3_BEDGRAPH:-$EXAMPLE_DATA/ENCFF171MDW_H3K36me3_chr15-17_hg38.bedGraph}"

for command in "$PYTHON_BIN" "$RSCRIPT_BIN" "$BEDTOOLS_BIN"; do
  command -v "$command" >/dev/null 2>&1 || {
    echo "[error] required command not found: $command" >&2
    exit 2
  }
done
for track in "$H3K27ME3_BEDGRAPH" "$H3K36ME3_BEDGRAPH"; do
  [[ -f "$track" ]] || { echo "[error] missing histone track: $track" >&2; exit 2; }
done

declare -A BED_NAMES=(
  [Armatus]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Armatus.bed"
  [Arrowhead]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_Arrowhead.bed"
  [CaTCH]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_CaTCH.bed"
  [deDoc]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_deDoc.deDocM.bed"
  [Domaincall]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_DI.bed"
  [EAST]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_EAST2.bed"
  [GMAP]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_rGMAP.bed"
  [Grinch]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_grinch.bed"
  [HiCSeg]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_HiCSeg.bed"
  [HiTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_hitad.bed"
  [IC-Finder]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_icfinder.bed"
  [OnTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_OnTAD.bed"
  [RefHiC]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_refhic.bed"
  [RobusTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_robustad.good.bed"
  [TopDom]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_TopDom.bed"
  [ContextTAD]="4DNFIXP4QG5B_Rao2014_GM12878_250M_5K_contexttad.bed"
)

DEFAULT_TOOLS=(
  Armatus Arrowhead CaTCH deDoc Domaincall EAST GMAP Grinch
  HiCSeg HiTAD IC-Finder OnTAD RefHiC RobusTAD TopDom ContextTAD
)
if [[ "${CONTEXT_ONLY:-0}" == "1" ]]; then
  TOOLS=(ContextTAD)
elif [[ -n "${HISTMOD_TOOLS:-}" ]]; then
  IFS=',' read -r -a TOOLS <<< "$HISTMOD_TOOLS"
else
  TOOLS=("${DEFAULT_TOOLS[@]}")
fi

mkdir -p "$OUTPUT_DIR" "$CSV_DIR"
echo "========================================"
echo "exp10: H3K27me3/H3K36me3 contrast"
echo "========================================"
echo "coverage: $COVERAGE"
echo "tools: ${TOOLS[*]}"
echo "results: $OUTPUT_DIR"

export H3K27ME3_BEDGRAPH H3K36ME3_BEDGRAPH PYTHON_BIN RSCRIPT_BIN BEDTOOLS_BIN
for tool in "${TOOLS[@]}"; do
  if [[ -z "${BED_NAMES[$tool]+x}" ]]; then
    echo "[error] unsupported tool: $tool" >&2
    exit 2
  fi
  bed="$EVAL_TOOLS_ROOT/$COVERAGE/$tool/${BED_NAMES[$tool]}"
  [[ -f "$bed" ]] || { echo "[error] missing BED: $bed" >&2; exit 2; }
  mean_file="$OUTPUT_DIR/${tool}_HistMod/Results_df_shuf10_share0.1_fdrthresh0.1_mean.txt"
  if [[ "$REUSE_EXISTING" == "1" && -s "$mean_file" ]]; then
    echo "[skip] existing result: $tool"
    continue
  fi
  echo "[run] $tool"
  bash "$SCRIPT_DIR/scripts/run_histmod_tool.sh" "$bed" "$tool" "$OUTPUT_DIR"
done

"$PYTHON_BIN" "$SCRIPT_DIR/collect_results.py" \
  --results-dir "$OUTPUT_DIR" \
  --output "$CSV_DIR/exp10_histone_results.csv" \
  --tools "${TOOLS[@]}"

echo "[done] exp10 completed"
