#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MCOOL="${MCOOL:-/home/weicai/projectnvme/TADAnno_final/0-data/2_eval_tads_data/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac1.mcool}"
FASTA="${FASTA:-/home/weicai/projectnvme/TADAnno_final/0-data/hg38.fa}"
OUTPUT_DIR="${OUTPUT_DIR:-$SCRIPT_DIR/../outputs/check_eigen}"

python "$SCRIPT_DIR/plot_chrom_heatmap_suppl_v2.py" \
  --mcool "$MCOOL" \
  --resolution 5000 \
  --fasta "$FASTA" \
  --output-dir "$OUTPUT_DIR"
