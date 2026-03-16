# Data Requirements (exp2)

This experiment reads BEDs from `EVAL_DATA_ROOT` and uses an external enrichment R script.

Required paths:
- `tools_output/250M/<ToolName>/*.bed`

Required assets for enrichment (bundled in this folder):
- `R_SCRIPT` (default: `exp2_struct_protein/r_scripts/StructProt_EnrichBoundaries_script.R`)
- `CHR_SIZES` (default: `exp2_struct_protein/GRCh38.chrom.sizes`)
- `PROTEIN_PEAKS_DIR` containing:
  - `CTCF_peaks.bed`
  - `RAD21_peaks.bed`
  - `SMC3_peaks.bed`

The runner is self-contained and does not use external fallback paths.
You can still override paths via environment variables if needed.
