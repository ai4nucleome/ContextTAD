# Pair-score structural characterization

This experiment tests whether ContextTAD pair confidence is associated with
TAD-scale contact structure in the matching GM12878 4000M Hi-C map.

Run it from the repository root with explicit input paths:

```bash
bash 3-evaluation/common/exp11_pairscore_structure/run_exp11.sh \
  --final-bed /path/to/GM12878_4000M_contexttad.bed \
  --raw-bed /path/to/ContextTAD_structure_4000M.bed \
  --mcool /path/to/GM12878_4000M.mcool \
  --chiapet /path/to/gm12878_ctcf_chiapet.bedpe \
  --ctcf /path/to/ENCFF232FCT_CTCF.bigWig \
  --h3k27 /path/to/ENCFF039JOT_H3K27me3.bigWig \
  --h3k36 /path/to/ENCFF171MDW_H3K36me3.bigWig \
  --output-dir /path/to/figure6_results \
  --cache-dir /path/to/figure6_cache
```

The script generates result tables under the requested output directory; the required Hi-C and track inputs are listed above.

