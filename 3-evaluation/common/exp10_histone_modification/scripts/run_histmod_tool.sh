#!/usr/bin/env bash
set -euo pipefail

INPUT_BED="${1:?Usage: $0 <input_bed> <tool> <output_root>}"
TOOL="${2:?Usage: $0 <input_bed> <tool> <output_root>}"
OUTPUT_ROOT="${3:?Usage: $0 <input_bed> <tool> <output_root>}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TOOL_OUTPUT="$OUTPUT_ROOT/${TOOL}_HistMod"

mkdir -p "$TOOL_OUTPUT"
for chrom in chr15 chr16 chr17; do
  CHROM_BED="$TOOL_OUTPUT/${TOOL}_${chrom}.bed"
  awk -v chrom="$chrom" '$1 == chrom && $3 - $2 > 20000' "$INPUT_BED" \
    > "$CHROM_BED"
  if [[ ! -s "$CHROM_BED" ]]; then
    echo "[error] no TADs longer than 20 kb for $TOOL $chrom" >&2
    exit 2
  fi
  bash "$SCRIPT_DIR/HistMod_script.sh" "$CHROM_BED" "$TOOL_OUTPUT"
  rm -f "$CHROM_BED"
done

"${PYTHON_BIN:-python3}" "$SCRIPT_DIR/HistMean.py" "$TOOL_OUTPUT"
