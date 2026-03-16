# Data Requirements (exp6)

This experiment reads from `EVAL_DATA_ROOT` (default: `<project_root>/0-data/2_eval_tads_data`).

Required paths:
- `tools_output/{4000M,2000M,1000M,500M,250M,125M,62_5M}/<ToolName>/*.bed`
- `ctcf_chiapet/gm12878.tang.ctcf-chiapet.hg38.bedpe`

Notes:
- `run_exp6.sh` auto-generates `.L0` and `.L1+` if missing.
