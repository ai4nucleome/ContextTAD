#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

usage() {
  cat <<'EOF'
Usage:
  bash run_exp11.sh --final-bed PATH --raw-bed PATH --mcool PATH \
    --chiapet PATH --ctcf PATH --h3k27 PATH --h3k36 PATH [options]

Required:
  --final-bed PATH       Final GM12878 4000M ContextTAD BED
  --raw-bed PATH         Raw GM12878 4000M candidate BED containing pair scores
  --mcool PATH           Balanced GM12878 4000M multi-resolution cooler
  --chiapet PATH         GM12878 CTCF ChIA-PET BEDPE
  --ctcf PATH            GM12878 CTCF bigWig
  --h3k27 PATH           GM12878 H3K27me3 bigWig
  --h3k36 PATH           GM12878 H3K36me3 bigWig

Options:
  --output-dir PATH      Result directory (default: <experiment>/results)
  --cache-dir PATH       Intermediate cache directory (default: <experiment>/intermediate)
  --python PATH          Python executable (default: python)
  --pygenometracks-bin PATH
                         pyGenomeTracks executable (default: pyGenomeTracks)
  --dpi N                Output resolution (default: 350)
  -h, --help             Show this help message
EOF
}

FINAL_BED=""
RAW_BED=""
MCOOL=""
CHIAPET=""
CTCF=""
H3K27=""
H3K36=""
OUTPUT_DIR="$HERE/results"
CACHE_DIR="$HERE/intermediate"
PYTHON_BIN="python"
PYGENOMETRACKS_BIN="pyGenomeTracks"
DPI=350

while [[ $# -gt 0 ]]; do
  case "$1" in
    --final-bed) FINAL_BED="${2:?missing value for --final-bed}"; shift 2 ;;
    --raw-bed) RAW_BED="${2:?missing value for --raw-bed}"; shift 2 ;;
    --mcool) MCOOL="${2:?missing value for --mcool}"; shift 2 ;;
    --chiapet) CHIAPET="${2:?missing value for --chiapet}"; shift 2 ;;
    --ctcf) CTCF="${2:?missing value for --ctcf}"; shift 2 ;;
    --h3k27) H3K27="${2:?missing value for --h3k27}"; shift 2 ;;
    --h3k36) H3K36="${2:?missing value for --h3k36}"; shift 2 ;;
    --output-dir) OUTPUT_DIR="${2:?missing value for --output-dir}"; shift 2 ;;
    --cache-dir) CACHE_DIR="${2:?missing value for --cache-dir}"; shift 2 ;;
    --python) PYTHON_BIN="${2:?missing value for --python}"; shift 2 ;;
    --pygenometracks-bin) PYGENOMETRACKS_BIN="${2:?missing value for --pygenometracks-bin}"; shift 2 ;;
    --dpi) DPI="${2:?missing value for --dpi}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

for required in FINAL_BED RAW_BED MCOOL CHIAPET CTCF H3K27 H3K36; do
  [[ -n "${!required}" ]] || {
    echo "[error] --$(echo "$required" | tr '[:upper:]_' '[:lower:]-') is required" >&2
    usage >&2
    exit 2
  }
done

LOG_DIR="$OUTPUT_DIR/logs"
mkdir -p "$OUTPUT_DIR" "$CACHE_DIR" "$LOG_DIR"

"$PYTHON_BIN" "$HERE/scripts/plot_fig6_pairscore_structure.py" \
  --final-bed "$FINAL_BED" \
  --raw-bed "$RAW_BED" \
  --mcool "$MCOOL" \
  --output-dir "$OUTPUT_DIR" \
  --cache-dir "$CACHE_DIR" \
  --dpi "$DPI" \
  2>&1 | tee "$LOG_DIR/pairscore_structure.log"

"$PYTHON_BIN" "$HERE/scripts/plot_figS_representative_regions.py" \
  --final-bed "$FINAL_BED" \
  --mcool "$MCOOL" \
  --chiapet "$CHIAPET" \
  --ctcf "$CTCF" \
  --h3k27 "$H3K27" \
  --h3k36 "$H3K36" \
  --output-dir "$OUTPUT_DIR" \
  --asset-dir "$CACHE_DIR/representative_region_assets" \
  --pygenometracks-bin "$PYGENOMETRACKS_BIN" \
  --dpi "$DPI" \
  2>&1 | tee "$LOG_DIR/representative_regions.log"

"$PYTHON_BIN" "$HERE/scripts/assemble_final_fig6.py" \
  --main-figure "$OUTPUT_DIR/fig6_pairscore_structure.png" \
  --regions-table "$OUTPUT_DIR/representative_regions.tsv" \
  --asset-dir "$CACHE_DIR/representative_region_assets" \
  --output-prefix "$OUTPUT_DIR/fig6_final" \
  --dpi "$DPI" \
  2>&1 | tee "$LOG_DIR/assemble.log"

echo "[done] results=$OUTPUT_DIR"
