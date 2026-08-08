# exp10 data requirements

This experiment computes the H3K27me3/H3K36me3 contrast shown in Figure 2f.
All callers are evaluated by the same deterministic permutation-test pipeline.

## Input

- GM12878 250M predictions under `EVAL_DATA_ROOT/tools_output/250M/`.
- H3K27me3 fold-change BedGraph: ENCODE `ENCFF039JOT`.
- H3K36me3 fold-change BedGraph: ENCODE `ENCFF171MDW`.
- The default track locations are:

```text
EVAL_DATA_ROOT/TAD-benchmarking-scripts/Example_Data/
├── ENCFF039JOT_H3K27me3_chr15-17_hg38.bedGraph
└── ENCFF171MDW_H3K36me3_chr15-17_hg38.bedGraph
```

Custom paths can be supplied through `H3K27ME3_BEDGRAPH` and
`H3K36ME3_BEDGRAPH`.

## Definition

- Evaluation chromosomes: chr15, chr16, and chr17.
- Minimum TAD size: greater than 20 kb.
- Histone contrast: TAD-level H3K27me3/H3K36me3 log10-ratio.
- Permutations: 10.
- Random seed: 42.
- Multiple-testing correction: Benjamini-Hochberg.
- Significance threshold: FDR < 0.1.

## Software

The workflow requires Python, `Rscript`, and `bedtools`. It is substantially
slower than the other Figure 2 evaluations and can require several GB of RAM.

## Run

```bash
EVAL_DATA_ROOT=/path/to/2_eval_tads_data \
bash run_exp10.sh
```

Run ContextTAD only with:

```bash
CONTEXT_ONLY=1 \
EVAL_DATA_ROOT=/path/to/2_eval_tads_data \
bash run_exp10.sh
```

The summary is written to `exp10_csv/exp10_histone_results.csv` unless
`CSV_DIR` is set.
