# exp9 data requirements

This experiment summarizes the TAD-size distributions shown in Figure 2a.

## Input

- GM12878 250M predictions under `EVAL_DATA_ROOT/tools_output/250M/`.
- One BED file for each of the 15 reference callers and ContextTAD.
- BED columns 1-3 must contain chromosome, start, and end coordinates.

## Definition

- Evaluation chromosomes: chr15, chr16, and chr17, as provided by the benchmark BED files.
- TAD size: `end - start`.
- TADs at least 3 Mb are excluded, matching the original benchmark.
- Sizes are reported in kb and callers are ranked by median TAD size.

## Run

```bash
EVAL_DATA_ROOT=/path/to/2_eval_tads_data \
bash run_exp9.sh
```

The output is written to `exp9_csv/exp9_tad_size_results.csv` unless
`OUTPUT_DIR` is set.
