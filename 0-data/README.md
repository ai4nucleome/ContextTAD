# 0-data

This folder is the expected data root.

Most scripts in this repo assume:
- `TAD_DATA_DIR=$PROJECT_ROOT/0-data/1_dp_train_infer_data`
- `EVAL_DATA_ROOT=$PROJECT_ROOT/0-data/2_eval_tads_data`

If you store data elsewhere, set environment variables explicitly before running scripts.

---

## 1 Required datasets

You need **two data trees**:

1. `1_dp_train_infer_data` (for training + inference)
2. `2_eval_tads_data` (for eval_tads experiments)

Expected layout:

```text
0-data/
 1_dp_train_infer_data/
    window_list.json
    labels/
       chr*/
           <window>_labels.npy
           <window>_tads.npy
    4000M/2000M/1000M/500M/250M/125M/62_5M/
       chr*/<window>.npy
    other_celltypes/          # optional, for exp16/18
        K562/
        IMR90/

 2_eval_tads_data/
     tools_output/
        4000M/2000M/.../62_5M/
        other_celltypes/
     ctcf_chiapet/
     ctcf_chipseq/
     mcool_data/
```

Notes:
- For this project, each `1_dp_train_infer_data/<cov>/<chr>/<window>.npy` contains two channels (obs and O/E). O/E is used for main experiments, and obs is used for ablation.
- Training/inference scripts read `window_list.json` for split and window IDs.
- `tools_output` in `2_eval_tads_data` is used by evaluation scripts to compare tools, and to temporarily replace `ContextTAD` during our model evaluation.

---

## 2 Prepare data in your own environment

This repository does not ship prebuilt data bundles because the full dataset is too large to distribute directly.
Please prepare data locally by running the preprocessing pipeline in Section 3, or by linking your own existing data trees.

If you already have prepared datasets in another location, copy or symlink them:

```bash
# example
ln -s /path/to/ready/1_dp_train_infer_data 0-data/1_dp_train_infer_data
ln -s /path/to/ready/2_eval_tads_data      0-data/2_eval_tads_data
```

This is only a directory wiring example; data content must be prepared by the user.

---

## 3 Build from raw inputs (scripted path)

Use scripts under `1-prepare_data/`.

### 3.1 Build training/inference arrays

Main entry:
- `1-prepare_data/step2_prepare_labels/scripts/prepare_data.py`

Example:

```bash
export TAD_DATA_DIR=/path/to/project/0-data/1_dp_train_infer_data
export MCOOL_TEMPLATE="/path/to/mcool/4DNFIXP4QG5B_Rao2014_GM12878_frac{frac}.mcool"

python 1-prepare_data/step2_prepare_labels/scripts/prepare_data.py
```

Optional modes:
- `--only-4000M` (process only 4000M)
- `--skip-4000M` (process only low coverages)

### 3.2 Build other-celltype inference data (optional)

Use:
- `1-prepare_data/step1_process_data/scripts/prepare_othercell_inference_data.py`

Output should be written under:
- `0-data/1_dp_train_infer_data/other_celltypes/{K562,IMR90}`

### 3.3 Build GT BED from labels (optional utility)

Use:
- `1-prepare_data/step3_build_gt/scripts/build_ground_truth.py` (test chromosomes)
- `1-prepare_data/step3_build_gt/scripts/build_ground_truth_all.py` (whole genome)

---

## 4 Minimal data checks before running training

Run these checks from project root:

```bash
test -f 0-data/1_dp_train_infer_data/window_list.json
test -d 0-data/1_dp_train_infer_data/labels
test -d 0-data/1_dp_train_infer_data/4000M
test -d 0-data/2_eval_tads_data/tools_output
```

Optional quick sanity check:

```bash
python - <<'PY'
import json, pathlib
p = pathlib.Path('0-data/1_dp_train_infer_data/window_list.json')
with open(p) as f:
    wl = json.load(f)
print({k: len(v) for k, v in wl['windows'].items()})
PY
```

---

## 5 Environment variables used by pipeline scripts

- `TAD_DATA_DIR`: training/inference data root (`.../0-data/1_dp_train_infer_data`)
- `SAM3_PATH`: local SAM3 model directory
- `EVAL_DATA_ROOT`: evaluation data root (`.../0-data/2_eval_tads_data`)
- `TOOLS_ROOT` (optional): tool BED root for eval scripts (defaults to `$EVAL_DATA_ROOT/tools_output`)

---

## 6 First end-to-end smoke run

After data is ready, this one command runs train + infer + exp1/7/8/14:

```bash
bash 2-training/step1_train/scripts/run_temp_base_train_infer_eval.sh <gpu_train> [gpu_infer] [run_tag] [batch_size]
```

If this passes, your data wiring is correct.
