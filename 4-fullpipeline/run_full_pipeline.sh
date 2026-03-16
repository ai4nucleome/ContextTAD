#!/usr/bin/env bash
set -euo pipefail

# One-click full pipeline.
# Default evaluation: exp1/exp3/exp4/exp6.
# Use --all-exps to run the full suite (exp1..exp8, and exp5 via RUN_EXP10=1).
#
# Usage:
#   bash run_full_pipeline.sh <gpu_train> [gpu_infer] [run_id] [batch_size] [master_port] [--all-exps]

ALL_EXPS=0
POSITIONAL=()
for arg in "$@"; do
  case "$arg" in
    --all-exps)
      ALL_EXPS=1
      ;;
    *)
      POSITIONAL+=("$arg")
      ;;
  esac
done
set -- "${POSITIONAL[@]}"

GPU_TRAIN="${1:?Usage: $0 <gpu_train> [gpu_infer] [run_id] [batch_size] [master_port] [--all-exps]}"
GPU_INFER="${2:-$GPU_TRAIN}"
RUN_ID="${3:-fullpipeline_$(date +%Y%m%d_%H%M%S)}"
BATCH_SIZE="${4:-2}"
MASTER_PORT="${5:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLISH_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

TRAIN_SCRIPT="$PUBLISH_ROOT/2-training/step1_train/scripts/run_train_base.sh"
INFER_GM_SCRIPT="$PUBLISH_ROOT/2-training/step2_infer_decode/scripts/run_infer_decode_gm12878.sh"
INFER_OTH_SCRIPT="$PUBLISH_ROOT/2-training/step2_infer_decode/scripts/run_infer_decode_othercell.sh"
EVAL_DEFAULT_SCRIPT="$PUBLISH_ROOT/3-evaluation/step2_model_ablation_ours_only/scripts/run_model_ablation_eval.sh"
EVAL_ALL_SCRIPT="$PUBLISH_ROOT/3-evaluation/step1_main_results_vs_tools/scripts/run_main_results.sh"

DATA_ROOT="${DATA_ROOT:-$PUBLISH_ROOT/0-data}"
TRAIN_DATA_DIR="${TAD_DATA_DIR:-$DATA_ROOT/1_dp_train_infer_data}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$DATA_ROOT/2_eval_tads_data}"
OTHERCELL_DATA_ROOT="${OTHERCELL_DATA_ROOT:-$TRAIN_DATA_DIR/other_celltypes}"
SAM3_DIR="${SAM3_PATH:-$PUBLISH_ROOT/2-training/sam3}"
EXPS_ROOT="${EXPS_ROOT:-$PUBLISH_ROOT/3-evaluation/common}"

if [[ ! -d "$TRAIN_DATA_DIR" ]]; then
  echo "[error] missing train data dir: $TRAIN_DATA_DIR" >&2
  exit 2
fi
if [[ ! -d "$EVAL_DATA_ROOT" ]]; then
  echo "[error] missing eval data dir: $EVAL_DATA_ROOT" >&2
  exit 2
fi
if [[ ! -d "$SAM3_DIR" ]]; then
  echo "[error] missing SAM3 dir: $SAM3_DIR" >&2
  exit 2
fi
if [[ ! -d "$EXPS_ROOT" ]]; then
  echo "[error] missing common exp dir: $EXPS_ROOT" >&2
  exit 2
fi
if [[ "$ALL_EXPS" == "1" && ! -d "$OTHERCELL_DATA_ROOT" ]]; then
  echo "[error] --all-exps requires other-cell data dir: $OTHERCELL_DATA_ROOT" >&2
  exit 2
fi

FP_OUT_ROOT="$SCRIPT_DIR/outputs"
FP_RUN_DIR="$FP_OUT_ROOT/$RUN_ID"
mkdir -p "$FP_RUN_DIR" "$FP_RUN_DIR/train" "$FP_RUN_DIR/infer" "$FP_RUN_DIR/eval"

link_path() {
  local src="$1"
  local dst="$2"
  if [[ ! -e "$src" ]]; then
    echo "[warn] skip link (missing): $src"
    return 0
  fi
  mkdir -p "$(dirname "$dst")"
  ln -sfn "$src" "$dst"
}

# -----------------------------------------------------------------------------
# Stage 1: train (weights remain in 2-training; no weight links in fullpipeline)
# -----------------------------------------------------------------------------
TRAIN_RUN_ID="train_base_${RUN_ID}"
TRAIN_RUNS_ROOT="$PUBLISH_ROOT/2-training/step1_train/outputs"

echo "[stage 1/4] train base model"
RUNS_ROOT="$TRAIN_RUNS_ROOT" \
STOP_AFTER_EPOCH=5 \
TAD_DATA_DIR="$TRAIN_DATA_DIR" \
SAM3_PATH="$SAM3_DIR" \
bash "$TRAIN_SCRIPT" \
  "$GPU_TRAIN" \
  "$TRAIN_RUN_ID" \
  none \
  10 \
  "$BATCH_SIZE" \
  "$MASTER_PORT"

TRAIN_RUN_DIR="$TRAIN_RUNS_ROOT/$TRAIN_RUN_ID"
CKPT="$TRAIN_RUN_DIR/train_outputs/checkpoints/epoch_005.pt"
if [[ ! -f "$CKPT" ]]; then
  echo "[error] missing checkpoint: $CKPT" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Stage 2: infer/decode GM12878
# -----------------------------------------------------------------------------
INFER_GM_RUN_ID="infer_gm12878_${RUN_ID}"

echo "[stage 2/4] infer/decode GM12878"
TAD_DATA_DIR="$TRAIN_DATA_DIR" \
SAM3_PATH="$SAM3_DIR" \
bash "$INFER_GM_SCRIPT" \
  "$CKPT" \
  "$GPU_INFER" \
  "$INFER_GM_RUN_ID" \
  auto \
  default

INFER_GM_RUN_DIR="$PUBLISH_ROOT/2-training/step2_infer_decode/outputs/$INFER_GM_RUN_ID"
GM_BEDS_DIR="$INFER_GM_RUN_DIR/beds"
if [[ ! -d "$GM_BEDS_DIR" ]]; then
  echo "[error] missing GM12878 beds dir: $GM_BEDS_DIR" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Stage 3: optional infer/decode other-cell (for full exp suite)
# -----------------------------------------------------------------------------
OTH_BEDS_DIR=""
INFER_OTH_RUN_DIR=""
if [[ "$ALL_EXPS" == "1" ]]; then
  INFER_OTH_RUN_ID="infer_othercell_${RUN_ID}"
  echo "[stage 3/4] infer/decode other-cell (K562, IMR90)"
  OTHERCELL_DATA_ROOT="$OTHERCELL_DATA_ROOT" \
  SAM3_PATH="$SAM3_DIR" \
  bash "$INFER_OTH_SCRIPT" \
    "$CKPT" \
    "$GPU_INFER" \
    "$INFER_OTH_RUN_ID" \
    auto \
    default

  INFER_OTH_RUN_DIR="$PUBLISH_ROOT/2-training/step2_infer_decode/outputs/$INFER_OTH_RUN_ID"
  OTH_BEDS_DIR="$INFER_OTH_RUN_DIR/beds"
  if [[ ! -d "$OTH_BEDS_DIR" ]]; then
    echo "[error] missing other-cell beds dir: $OTH_BEDS_DIR" >&2
    exit 2
  fi
else
  echo "[stage 3/4] skip other-cell inference (use --all-exps to enable)"
fi

# -----------------------------------------------------------------------------
# Stage 4: evaluation
# -----------------------------------------------------------------------------
if [[ "$ALL_EXPS" == "1" ]]; then
  EVAL_RUN_ID="main_results_${RUN_ID}"
  echo "[stage 4/4] evaluate full suite (exp1..exp8 + exp5)"
  EVAL_DATA_ROOT="$EVAL_DATA_ROOT" \
  EXPS_ROOT="$EXPS_ROOT" \
  RUN_EXP10=1 \
  bash "$EVAL_ALL_SCRIPT" \
    "$GM_BEDS_DIR" \
    "$OTH_BEDS_DIR" \
    "$EVAL_RUN_ID"

  EVAL_RUN_DIR="$PUBLISH_ROOT/3-evaluation/step1_main_results_vs_tools/outputs/runs/$EVAL_RUN_ID"
else
  EVAL_RUN_ID="default_eval_${RUN_ID}"
  echo "[stage 4/4] evaluate default suite (exp1/3/4/6)"
  EVAL_DATA_ROOT="$EVAL_DATA_ROOT" \
  EXPS_ROOT="$EXPS_ROOT" \
  bash "$EVAL_DEFAULT_SCRIPT" \
    "$GM_BEDS_DIR" \
    "$EVAL_RUN_ID"

  EVAL_RUN_DIR="$PUBLISH_ROOT/3-evaluation/step2_model_ablation_ours_only/outputs/$EVAL_RUN_ID"
fi

if [[ ! -d "$EVAL_RUN_DIR" ]]; then
  echo "[error] missing evaluation run dir: $EVAL_RUN_DIR" >&2
  exit 2
fi

# -----------------------------------------------------------------------------
# Full-pipeline output index (symlinks only; no file copy)
# -----------------------------------------------------------------------------
link_path "$TRAIN_RUN_DIR/logs" "$FP_RUN_DIR/train/logs"
link_path "$TRAIN_RUN_DIR/train_outputs/args.json" "$FP_RUN_DIR/train/args.json"
link_path "$TRAIN_RUN_DIR/train_outputs/history.json" "$FP_RUN_DIR/train/history.json"

link_path "$INFER_GM_RUN_DIR" "$FP_RUN_DIR/infer/gm12878"
if [[ "$ALL_EXPS" == "1" ]]; then
  link_path "$INFER_OTH_RUN_DIR" "$FP_RUN_DIR/infer/other_celltypes"
fi

link_path "$EVAL_RUN_DIR" "$FP_RUN_DIR/eval/run"
link_path "$EVAL_RUN_DIR/csv" "$FP_RUN_DIR/eval/csv"
link_path "$EVAL_RUN_DIR/logs" "$FP_RUN_DIR/eval/logs"

# -----------------------------------------------------------------------------
# Link common experiment result directories for quick browsing
# -----------------------------------------------------------------------------
COMMON_LINK_DIR="$FP_RUN_DIR/eval/common_results"
mkdir -p "$COMMON_LINK_DIR"

# Always available from default/full evaluation.
link_path "$EXPS_ROOT/exp1_tadnum_ctcf_chiapet/exp1_csv" "$COMMON_LINK_DIR/exp1_tadnum_ctcf_chiapet"
link_path "$EXPS_ROOT/exp3_tadb_left_ctcf_chipseq/exp3_csv" "$COMMON_LINK_DIR/exp3_tadb_left_ctcf_chipseq"
link_path "$EXPS_ROOT/exp4_tadb_right_ctcf_chipseq/exp4_csv" "$COMMON_LINK_DIR/exp4_tadb_right_ctcf_chipseq"
link_path "$EXPS_ROOT/exp6_tadnum_ctcf_chiapet_downsample/exp6_csv" "$COMMON_LINK_DIR/exp6_tadnum_ctcf_chiapet_downsample"

# Full-suite extras.
if [[ "$ALL_EXPS" == "1" ]]; then
  if [[ -d "$EVAL_RUN_DIR/exp2_csv" ]]; then
    link_path "$EVAL_RUN_DIR/exp2_csv" "$COMMON_LINK_DIR/exp2_struct_protein_csv"
  else
    link_path "$EXPS_ROOT/exp2_struct_protein/exp2_csv" "$COMMON_LINK_DIR/exp2_struct_protein_csv"
  fi
  if [[ -d "$EVAL_RUN_DIR/exp2_struct_protein" ]]; then
    link_path "$EVAL_RUN_DIR/exp2_struct_protein" "$COMMON_LINK_DIR/exp2_struct_protein_raw"
  else
    link_path "$EXPS_ROOT/exp2_struct_protein/struct_protein" "$COMMON_LINK_DIR/exp2_struct_protein_raw"
  fi

  link_path "$EXPS_ROOT/exp7_othercell_tadnum_ctcf_chiapet/exp7_csv" "$COMMON_LINK_DIR/exp7_othercell_tadnum_ctcf_chiapet"
  link_path "$EXPS_ROOT/exp8_othercell_both_ctcf_chipseq/exp8_csv" "$COMMON_LINK_DIR/exp8_othercell_both_ctcf_chipseq"
  link_path "$EXPS_ROOT/exp5_coolpup/exp5_figs" "$COMMON_LINK_DIR/exp5_coolpup_figs"
  link_path "$EXPS_ROOT/exp5_coolpup/data" "$COMMON_LINK_DIR/exp5_coolpup_data"
fi

ln -sfn "$FP_RUN_DIR" "$FP_OUT_ROOT/latest"

echo "[done] full pipeline completed"
echo "run_id=$RUN_ID"
echo "fullpipeline_links=$FP_RUN_DIR"
echo "train_run=$TRAIN_RUN_DIR"
echo "gm_infer_run=$INFER_GM_RUN_DIR"
if [[ "$ALL_EXPS" == "1" ]]; then
  echo "othercell_infer_run=$INFER_OTH_RUN_DIR"
fi
echo "eval_run=$EVAL_RUN_DIR"
echo "checkpoint_epoch005=$CKPT"
