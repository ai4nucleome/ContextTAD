# ContextTAD data layout

This directory is the default data root used by training, inference, and evaluation scripts.

The full processed dataset is not committed to Git because of its size. Most processed resources are available from [Zenodo](https://doi.org/10.5281/zenodo.19062598); raw `.mcool` files must be obtained from the 4DN Data Portal and processed locally.

## Required directories

```text
0-data/
├── 1_dp_train_infer_data/
│   ├── window_list.json
│   ├── labels/
│   │   └── chr*/
│   │       ├── <window>_labels.npy
│   │       └── <window>_tads.npy
│   ├── 4000M/chr*/<window>.npy
│   ├── 2000M/chr*/<window>.npy
│   ├── 1000M/chr*/<window>.npy
│   ├── 500M/chr*/<window>.npy
│   ├── 250M/chr*/<window>.npy
│   ├── 125M/chr*/<window>.npy
│   ├── 62_5M/chr*/<window>.npy
│   └── other_celltypes/           # Optional: K562 and IMR90
└── 2_eval_tads_data/
    ├── tools_output/
    ├── ctcf_chiapet/
    ├── ctcf_chipseq/
    ├── mcool_data/
    ├── TAD-benchmarking-scripts/Example_Data/
    │   ├── ENCFF039JOT_H3K27me3_chr15-17_hg38.bedGraph
    │   ├── ENCFF171MDW_H3K36me3_chr15-17_hg38.bedGraph
    │   ├── ENCFF232FCT_CTCF.bigWig
    │   ├── ENCFF039JOT.bigWig
    │   └── ENCFF171MDW.bigWig
    └── additional experiment-specific assets
```

## Training array format

Each matrix file contains a two-channel NumPy array:

```text
channel 0: observed contact matrix
channel 1: O/E contact matrix
shape:     [2, 400, 400]
```

The base model reads channel 1. Channel 0 is retained for the observed-matrix ablation.

Each `*_tads.npy` file stores TAD endpoints as closed bin intervals `[left, right]`. Invalid padding rows have a negative left endpoint. Both training heads are supervised at their stored endpoint bins without subtracting one from the right endpoint.

`window_list.json` defines the train, validation, and test windows. The released split uses chromosomes 15, 16, and 17 as the held-out test set.

## Using data outside the repository

Set these variables before running the scripts:

```bash
export TAD_DATA_DIR=/path/to/1_dp_train_infer_data
export EVAL_DATA_ROOT=/path/to/2_eval_tads_data
```

Alternatively, create local symbolic links:

```bash
ln -s /path/to/1_dp_train_infer_data 0-data/1_dp_train_infer_data
ln -s /path/to/2_eval_tads_data 0-data/2_eval_tads_data
```

## Data preparation

The preprocessing code is organized under `1-prepare_data/`:

```text
step1_process_data/     raw Hi-C and genomic-track processing
step2_prepare_labels/   window construction and DP-refined labels
step3_build_gt/         merged ground-truth BED utilities
```

The main array-construction entry point is:

```bash
python 1-prepare_data/step2_prepare_labels/scripts/prepare_data.py
```

Build other-cell inference arrays with:

```bash
python 1-prepare_data/step1_process_data/scripts/prepare_othercell_inference_data.py \
  --mcool /path/to/cell_line.mcool::/resolutions/5000 \
  --out_data_dir /path/to/1_dp_train_infer_data/other_celltypes/K562 \
  --coverage_tag K562
```

## Required source datasets

| Category | Cell line | Accession |
|---|---|---|
| Hi-C | GM12878 | `4DNFIXP4QG5B` |
| Hi-C | K562 | `4DNFI4DGNY7J` |
| Hi-C | IMR90 | `4DNFIJTOIGOI` |
| CTCF ChIP-seq | GM12878 | `ENCFF796WRU` |
| CTCF signal bigWig | GM12878 | `ENCFF232FCT` |
| CTCF ChIP-seq | K562 | `ENCFF901CBP` |
| CTCF ChIP-seq | IMR90 | `ENCFF203SRF` |
| CTCF ChIA-PET | GM12878 | `GSE72816` |
| CTCF ChIA-PET | K562 | `ENCFF001THV` |
| CTCF ChIA-PET | IMR90 | `ENCFF682YFU` |
| H3K27me3 ChIP-seq | GM12878 | `ENCFF039JOT` |
| H3K36me3 ChIP-seq | GM12878 | `ENCFF171MDW` |

## Sanity checks

```bash
test -f "$TAD_DATA_DIR/window_list.json"
test -d "$TAD_DATA_DIR/labels"
test -d "$TAD_DATA_DIR/4000M"
test -d "$EVAL_DATA_ROOT/tools_output"
```

The CSV and pileup files under `contexttad_eval_results/` are compact copies
of the released benchmark summaries and do not replace the complete evaluation
data tree. Detailed requirements for every benchmark are documented under
`3-evaluation/common/<experiment>/DATA_REQUIREMENTS.md`.

Figure 6 additionally requires the GM12878 4000M final BED and scored raw BED
from the same ContextTAD inference run. These model outputs are generated under
`2-training/step2_infer_decode/outputs/<run_id>/{beds,raw_preds}` rather than
stored in the evaluation-data tree.
