#!/bin/bash
# Two-stage preprocessing pipeline for DP-refined TAD label construction.
# Stage 1: process_data.py (extract window-level matrices and annotations)
# Stage 2: tad_dp_refine.py (DP refinement on per-window TAD candidates)
#
# Usage:
#   bash run_pipeline.sh [--step1-only] [--step2-only] [--debug]

set -e

PYTHON="${PYTHON:-/home/weicai/anaconda3/envs/robustad/bin/python}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Stage-1 inputs
COOLPATH="${COOLPATH:-/home/weicai/projectnvme/TADAnno_final/0-data/2_eval_tads_data/mcool_data/GM12878/Rao2014/4DNFIXP4QG5B_Rao2014_GM12878_frac1.mcool::/resolutions/5000}"
CHROM=""              # Optional CSV chromosome list, e.g. "chr1,chr2,chr3"
SIZE=400               # Window size in bins
STEP=200               # Sliding stride in bins
RESOL=5000             # Resolution in base pairs
SAVEDIR="${SAVEDIR:-$STEP_ROOT/outputs/lwc_gm12878}"

# Stage-2 input root (expects chr*/chr*_<start> subfolders)
ROOT_DIR="${SAVEDIR}"

RUN_STEP1=true
RUN_STEP2=true
DEBUG_FLAG=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --step1-only)
      RUN_STEP2=false
      shift
      ;;
    --step2-only)
      RUN_STEP1=false
      shift
      ;;
    --debug)
      DEBUG_FLAG="--debug"
      shift
      ;;
    -h|--help)
      echo "Usage: bash run_pipeline.sh [--step1-only] [--step2-only] [--debug]"
      echo
      echo "Options:"
      echo "  --step1-only   Run process_data.py only"
      echo "  --step2-only   Run tad_dp_refine.py only"
      echo "  --debug        Enable debug mode in tad_dp_refine.py"
      exit 0
      ;;
    *)
      echo "Unknown argument: $1"
      echo "Use --help for usage."
      exit 1
      ;;
  esac
done

echo "============================================================"
echo "Preprocessing Pipeline"
echo "============================================================"
echo "script_dir : ${SCRIPT_DIR}"
echo "python_bin : ${PYTHON}"
echo "run_step1  : ${RUN_STEP1}"
echo "run_step2  : ${RUN_STEP2}"
echo "debug_mode : ${DEBUG_FLAG:-false}"
echo "============================================================"

if [[ "${RUN_STEP1}" == true ]]; then
  echo
  echo "============================================================"
  echo "Stage 1: process_data.py"
  echo "============================================================"
  echo "cool_path   : ${COOLPATH}"
  echo "chrom_list  : ${CHROM:-all}"
  echo "window_bins : ${SIZE}"
  echo "stride_bins : ${STEP}"
  echo "resolution  : ${RESOL}"
  echo "output_dir  : ${SAVEDIR}"
  echo "------------------------------------------------------------"

  CMD="${PYTHON} ${SCRIPT_DIR}/process_data.py"
  CMD+=" --coolpath '${COOLPATH}'"
  CMD+=" --size ${SIZE}"
  CMD+=" --step ${STEP}"
  CMD+=" --resol ${RESOL}"
  CMD+=" --savedir '${SAVEDIR}'"
  if [[ -n "${CHROM}" ]]; then
    CMD+=" --chrom '${CHROM}'"
  fi

  echo "Command:"
  echo "${CMD}"
  echo
  eval ${CMD}
  echo
  echo "Stage 1 completed."
fi

if [[ "${RUN_STEP2}" == true ]]; then
  echo
  echo "============================================================"
  echo "Stage 2: tad_dp_refine.py"
  echo "============================================================"
  echo "root_dir: ${ROOT_DIR}"
  echo "------------------------------------------------------------"

  CMD="${PYTHON} ${SCRIPT_DIR}/tad_dp_refine.py --root_dir '${ROOT_DIR}'"
  if [[ -n "${DEBUG_FLAG}" ]]; then
    CMD+=" ${DEBUG_FLAG}"
  fi

  echo "Command:"
  echo "${CMD}"
  echo
  eval ${CMD}
  echo
  echo "Stage 2 completed."
fi

echo
echo "============================================================"
echo "Pipeline Finished"
echo "============================================================"
echo "output_root: ${SAVEDIR}"
echo
echo "Expected files include:"
echo "  - obs.txt          : ICE-normalized matrix slices"
echo "  - obsLarge.txt     : larger context matrix for score computation"
echo "  - oe.txt           : O/E matrix slices"
echo "  - TAD.txt          : raw TAD intervals"
echo "  - TAD_dp.txt       : DP-refined TAD intervals"
echo "  - linearAnno.csv   : CTCF/ATAC/eigenvector features"
echo "============================================================"
