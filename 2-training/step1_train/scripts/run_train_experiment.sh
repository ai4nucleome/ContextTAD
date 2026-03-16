#!/usr/bin/env bash
set -euo pipefail

# Train one experiment under model_analysis strategy + val-random-coverage:
#   - train_sampling: random_single (keep model_analysis legacy behavior)
#   - val_sampling  : random_single (new change, same as train)
# Usage:
#   bash run_train_experiment.sh <exp_name> <gpu_ids> [run_id] [init_ckpt_or_none] [epochs] [batch_size] [master_port]
# exp_name:
#   base | no_tofe | no_text | no_pairloss | no_count | obs_input
# gpu_ids examples:
#   3          -> single GPU
#   1,2,3,4    -> DDP on 4 GPUs
# Optional external override:
#   EXTRA_TRAIN_ARGS="--count_weight 0.0" bash run_train_experiment.sh ...
# Notes:
#   - EXTRA_TRAIN_ARGS are appended at the end, so they override defaults above.

EXP_NAME="${1:?Usage: $0 <exp_name> <gpu_ids> [run_id] [init_ckpt_or_none] [epochs] [batch_size] [master_port]}"
GPU_SPEC_RAW="${2:?Usage: $0 <exp_name> <gpu_ids> [run_id] [init_ckpt_or_none] [epochs] [batch_size] [master_port]}"
RUN_ID="${3:-train_${EXP_NAME}_modelanalysis_valrand_$(date +%Y%m%d_%H%M%S)}"
INIT_CKPT="${4:-none}"
EPOCHS="${5:-10}"
BATCH_SIZE="${6:-2}"
MASTER_PORT="${7:-}"
STOP_AFTER_EPOCH="${STOP_AFTER_EPOCH:-0}"
GPU_SPEC="${GPU_SPEC_RAW// /}"

IFS=',' read -r -a GPU_ARR <<< "$GPU_SPEC"
WORLD_SIZE="${#GPU_ARR[@]}"
if [[ "$WORLD_SIZE" -lt 1 ]]; then
  echo "[error] invalid gpu_ids: $GPU_SPEC_RAW"
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PUBLISH_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN_ROOT="$PUBLISH_ROOT/2-training"
CORE_DIR="$TRAIN_ROOT/core"
CONDA_SH="/home/weicai/anaconda3/etc/profile.d/conda.sh"
DATA_DIR="${TAD_DATA_DIR:-$PUBLISH_ROOT/0-data/1_dp_train_infer_data}"
SAM3_DIR="${SAM3_PATH:-$TRAIN_ROOT/sam3}"
RUNS_ROOT="${RUNS_ROOT:-$TRAIN_ROOT/step1_train/outputs/${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"

RUN_DIR="$RUNS_ROOT/$RUN_ID"
OUT_DIR="$RUN_DIR/train_outputs"
LOG_DIR="$RUN_DIR/logs"
mkdir -p "$OUT_DIR" "$LOG_DIR"

source "$CONDA_SH"
conda activate 3dgenome

EXTRA_ARGS=()
METHOD="text_oe"
case "$EXP_NAME" in
  base)
    ;;
  no_tofe)
    EXTRA_ARGS+=(--disable_tofe)
    ;;
  no_text)
    EXTRA_ARGS+=(--disable_text_branch)
    ;;
  no_pairloss)
    EXTRA_ARGS+=(--pair_weight 0.0)
    ;;
  no_count)
    EXTRA_ARGS+=(--count_weight 0.0)
    ;;
  obs_input)
    METHOD="text_obs"
    ;;
  *)
    echo "[error] unsupported exp_name: $EXP_NAME"
    exit 2
    ;;
esac

TRAIN_ARGS=(
  --method "$METHOD"
  --epochs "$EPOCHS"
  --stop_after_epoch "$STOP_AFTER_EPOCH"
  --batch_size "$BATCH_SIZE"
  --lr 3e-4
  --lora_r 16
  --seed 42
  --pair_weight 2.6
  --count_weight 0.3
  --pair_hard_window 10
  --tofe_mix 1.0
  --train_sampling random_single
  --val_sampling random_single
  --save_every_epoch
  --output_dir "$OUT_DIR"
)

TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

if [[ -n "$INIT_CKPT" && "$INIT_CKPT" != "none" ]]; then
  TRAIN_ARGS+=(--init_ckpt "$INIT_CKPT")
fi

USER_EXTRA_ARGS_STR="${EXTRA_TRAIN_ARGS:-}"
if [[ -n "$USER_EXTRA_ARGS_STR" ]]; then
  # shellcheck disable=SC2206
  USER_EXTRA_ARGS=($USER_EXTRA_ARGS_STR)
  TRAIN_ARGS+=("${USER_EXTRA_ARGS[@]}")
fi

if [[ "$WORLD_SIZE" -gt 1 ]]; then
  if [[ -z "$MASTER_PORT" ]]; then
    MASTER_PORT="$((20000 + RANDOM % 20000))"
  fi
  CMD=(
    accelerate launch
    --num_processes "$WORLD_SIZE"
    --mixed_precision fp16
    --main_process_port "$MASTER_PORT"
    "$CORE_DIR/train.py"
    "${TRAIN_ARGS[@]}"
  )
else
  CMD=(
    python "$CORE_DIR/train.py"
    "${TRAIN_ARGS[@]}"
  )
fi

{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "CUDA_VISIBLE_DEVICES=%q TAD_DATA_DIR=%q SAM3_PATH=%q " \
    "$GPU_SPEC" "$DATA_DIR" "$SAM3_DIR"
  printf "%q " "${CMD[@]}"
  echo
} > "$RUN_DIR/run_command.sh"
chmod +x "$RUN_DIR/run_command.sh"

{
  echo "[info] exp_name=$EXP_NAME"
  echo "[info] method=$METHOD"
  echo "[info] run_id=$RUN_ID"
  echo "[info] gpu_ids=$GPU_SPEC"
  echo "[info] world_size=$WORLD_SIZE"
  if [[ "$WORLD_SIZE" -gt 1 ]]; then
    echo "[info] ddp=on master_port=$MASTER_PORT"
  else
    echo "[info] ddp=off"
  fi
  echo "[info] epochs=$EPOCHS batch_size=$BATCH_SIZE"
  echo "[info] stop_after_epoch=$STOP_AFTER_EPOCH"
  echo "[info] init_ckpt=$INIT_CKPT"
  echo "[info] out_dir=$OUT_DIR"
  echo "[info] extra_args=${EXTRA_ARGS[*]:-(none)}"
  echo "[info] extra_train_args=${USER_EXTRA_ARGS_STR:-(none)}"
} | tee "$LOG_DIR/meta.log"

CUDA_VISIBLE_DEVICES="$GPU_SPEC" \
TAD_DATA_DIR="$DATA_DIR" \
SAM3_PATH="$SAM3_DIR" \
"${CMD[@]}" \
  > "$LOG_DIR/train.log" 2>&1

echo "[done] training finished: $RUN_DIR"
echo "[done] best model: $OUT_DIR/best_model.pt"
echo "[done] final model: $OUT_DIR/final_model.pt"
