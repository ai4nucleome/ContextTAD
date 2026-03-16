# Data Requirements (exp5)

This experiment computes coolpup pileups for the 250M setting.

Required:
- `BED_DIR` (default: `${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}/250M`)
- `MCOOL_FILE` with resolution selector suffix `::/resolutions/5000`

Default `MCOOL_FILE`:
- `<project_root>/0-data/2_eval_tads_data/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac1.mcool::/resolutions/5000`

Software requirements:
- `coolpup.py` available in PATH
- Python environment with dependencies for `clpy2txt.py` and `plot_pileup.py`
