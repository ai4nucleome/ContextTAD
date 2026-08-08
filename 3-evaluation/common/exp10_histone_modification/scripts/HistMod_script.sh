#!/usr/bin/env bash
set -euo pipefail

# Assess the fraction of TADs with a significant H3K27me3/H3K36me3
# log10-ratio under the published permutation-test procedure.

TADS_FILE="${1:?Usage: $0 <chromosome_bed> <output_dir>}"
OUT_DIR="${2:?Usage: $0 <chromosome_bed> <output_dir>}"
H27_FC="${H3K27ME3_BEDGRAPH:?Set H3K27ME3_BEDGRAPH}"
H36_FC="${H3K36ME3_BEDGRAPH:?Set H3K36ME3_BEDGRAPH}"
RSCRIPT_BIN="${RSCRIPT_BIN:-Rscript}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
BEDTOOLS_BIN="${BEDTOOLS_BIN:-bedtools}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SHARE="${HISTMOD_SHARE:-0.1}"
N_SHUFFLES="${HISTMOD_N_SHUFFLES:-10}"
FDR_THRESHOLD="${HISTMOD_FDR_THRESHOLD:-0.10}"
CHROM="$(awk 'NR == 1 {sub(/^chr/, "", $1); print $1}' "$TADS_FILE")"

if [[ -z "$CHROM" ]]; then
  echo "[error] empty chromosome BED: $TADS_FILE" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"
BINSIZES_FILE="$OUT_DIR/binsizedf_chr${CHROM}_share${SHARE}.txt"
"$RSCRIPT_BIN" "$SCRIPT_DIR/HistMod_computeBinsizes_script.R" \
  "$TADS_FILE" "$BINSIZES_FILE" "$SHARE"

H27_BINNED=""
H36_BINNED=""
while read -r binsize; do
  BINNED_FILE="$OUT_DIR/chr${CHROM}_binsize${binsize}.bed"
  "$PYTHON_BIN" "$SCRIPT_DIR/HistMod_binChr.py" \
    -o "$BINNED_FILE" -c "$CHROM" -b "$binsize"

  H27_INTERSECTIONS="$OUT_DIR/h27_fc_intersections.bedGraph"
  H27_BINNED="$OUT_DIR/h27_fc_binsize${binsize}.bedGraph"
  "$BEDTOOLS_BIN" intersect -wao -a "$BINNED_FILE" -b "$H27_FC" \
    > "$H27_INTERSECTIONS"
  "$RSCRIPT_BIN" "$SCRIPT_DIR/HistMod_aggregateBdg.R" \
    "$H27_INTERSECTIONS" "$H27_BINNED" "$binsize"

  H36_INTERSECTIONS="$OUT_DIR/h36_fc_intersections.bedGraph"
  H36_BINNED="$OUT_DIR/h36_fc_binsize${binsize}.bedGraph"
  "$BEDTOOLS_BIN" intersect -wao -a "$BINNED_FILE" -b "$H36_FC" \
    > "$H36_INTERSECTIONS"
  "$RSCRIPT_BIN" "$SCRIPT_DIR/HistMod_aggregateBdg.R" \
    "$H36_INTERSECTIONS" "$H36_BINNED" "$binsize"
done < "$BINSIZES_FILE"

"$RSCRIPT_BIN" "$SCRIPT_DIR/HistMod_FDRclassic_script.R" \
  "$TADS_FILE" "$SHARE" "$N_SHUFFLES" "$FDR_THRESHOLD" "$CHROM" \
  "$OUT_DIR/" "$H27_BINNED" "$H36_BINNED"

rm -f "$OUT_DIR"/binsize* "$OUT_DIR"/chr* "$OUT_DIR"/h27* "$OUT_DIR"/h36*
