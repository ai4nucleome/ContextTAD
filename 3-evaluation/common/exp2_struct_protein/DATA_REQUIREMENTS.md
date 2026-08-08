# Data Requirements (exp2)

This experiment reads BEDs from `EVAL_DATA_ROOT` and uses the bundled
TAD-benchmarking enrichment R script.

Required paths:
- `tools_output/250M/<ToolName>/*.bed`

Required assets for enrichment:
- `R_SCRIPT` (default: `exp2_struct_protein/r_scripts/StructProt_EnrichBoundaries_script.R`)
- `CHR_SIZES` (default: `exp2_struct_protein/GRCh38.chrom.sizes`)
- `EVAL_DATA_ROOT/TAD-benchmarking-scripts/Example_Data/` containing:
  - `CTCF_peaks.bed`
  - `RAD21_peaks.bed`
  - `SMC3_peaks.bed`

Set `PROTEIN_PEAKS_DIR` to use a different directory. Set `RSCRIPT_BIN` when
`Rscript` is not available on `PATH`.
