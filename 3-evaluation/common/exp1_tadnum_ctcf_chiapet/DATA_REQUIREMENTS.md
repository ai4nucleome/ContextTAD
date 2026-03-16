# Data Requirements (exp1)

This experiment reads from `EVAL_DATA_ROOT` (default: `<project_root>/0-data/2_eval_tads_data`).

Required paths:
- `tools_output/250M/<ToolName>/*.bed`
- `ctcf_chiapet/gm12878.tang.ctcf-chiapet.hg38.bedpe`

Notes:
- `run_exp1.sh` will generate `.L0` and `.L1+` beside each BED file when missing.
- Set `EVAL_TOOLS_ROOT` to override the default tools path.
