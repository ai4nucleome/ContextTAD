# Experiment 11 data requirements

This experiment reproduces the structural characterization in Figure 6 using
GM12878 4000M predictions and the matching 5-kb Hi-C map.

## Required inputs

| Option | Required content |
|---|---|
| `--final-bed` | Final ContextTAD calls for GM12878 4000M, chromosomes 15-17 |
| `--raw-bed` | Pre-postprocessing candidate BED with pair score in column 4 |
| `--mcool` | Balanced GM12878 4000M `.mcool` containing `/resolutions/5000` |
| `--chiapet` | GM12878 CTCF ChIA-PET loops in BEDPE format |
| `--ctcf` | GM12878 CTCF signal in bigWig format (`ENCFF232FCT`) |
| `--h3k27` | GM12878 H3K27me3 signal in bigWig format (`ENCFF039JOT`) |
| `--h3k36` | GM12878 H3K36me3 signal in bigWig format (`ENCFF171MDW`) |

The final BED and raw BED must come from the same inference run. Exact interval
matching is used to transfer each raw pair score to its retained final call.

## Analyses

1. Sort final TADs by pair score and divide them into stable Q1-Q4 rank groups.
2. Compute size-rescaled O/E pileups from the 4000M map.
3. Compute left- and right-boundary insulation profiles within +/-200 kb using
   a 50-kb diamond window.
4. Select and render three reproducible 2-Mb regions on chromosomes 15-17.
5. Assemble the analytical panels and representative regions into Figure 6.

`pyGenomeTracks` is required only for the representative-region panels. The
large O/E and insulation caches are generated under `intermediate/` and are not
distributed in this repository.

