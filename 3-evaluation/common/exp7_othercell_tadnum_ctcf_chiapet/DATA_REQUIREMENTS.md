# Data Requirements (exp7)

This experiment reads from `EVAL_DATA_ROOT` (default: `<project_root>/0-data/2_eval_tads_data`).

Required paths:
- `tools_output/other_celltypes/K562/<ToolName>/*.bed`
- `tools_output/other_celltypes/IMR90/<ToolName>/*.bed`
- `ctcf_chiapet/k562.encode.ctcf-chiapet.5k.hg38.bedpe`
- `ctcf_chiapet/imr90_ctcf_chiapet_hg38_ENCFF682YFU.bedpe`

Notes:
- `run_exp7.sh` auto-generates `.L0` and `.L1+` if missing.
