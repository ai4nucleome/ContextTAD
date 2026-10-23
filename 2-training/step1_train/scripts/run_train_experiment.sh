#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  bash run_train_experiment.sh --experiment NAME --gpu-ids IDS [options]

Required:
  --experiment NAME    base | obs_input | no_pairloss | no_count | boundary_count
  --gpu-ids IDS        Comma-separated GPU IDs exposed to Accelerate, for example 0 or 0,1

Options:
  --run-id ID          Output run ID (default: train_<experiment>_<timestamp>)
  --epochs N           Number of training epochs (default: 50)
  --batch-size N       Batch size per GPU process (default: 16)
  --count-lambda FLOAT Count-loss coefficient after normalization by window bins (default: 32)
  --max-grad-norm FLOAT Gradient clipping threshold (default: 0.4)
  --master-port PORT   Accelerate main-process port for multi-GPU training
  --data-dir PATH      Processed training-data directory
  --output-root PATH   Parent directory for training runs
  -h, --help           Show this help message
EOF
}

EXP_NAME=""
GPU_SPEC_RAW=""
RUN_ID=""
EPOCHS=50
BATCH_SIZE=16
COUNT_LAMBDA=32
MAX_GRAD_NORM=0.4
MASTER_PORT=""
DATA_DIR_OVERRIDE=""
RUNS_ROOT_OVERRIDE=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment) EXP_NAME="${2:?missing value for --experiment}"; shift 2 ;;
    --gpu-ids) GPU_SPEC_RAW="${2:?missing value for --gpu-ids}"; shift 2 ;;
    --run-id) RUN_ID="${2:?missing value for --run-id}"; shift 2 ;;
    --epochs) EPOCHS="${2:?missing value for --epochs}"; shift 2 ;;
    --batch-size) BATCH_SIZE="${2:?missing value for --batch-size}"; shift 2 ;;
    --count-lambda) COUNT_LAMBDA="${2:?missing value for --count-lambda}"; shift 2 ;;
    --max-grad-norm) MAX_GRAD_NORM="${2:?missing value for --max-grad-norm}"; shift 2 ;;
    --master-port) MASTER_PORT="${2:?missing value for --master-port}"; shift 2 ;;
    --data-dir) DATA_DIR_OVERRIDE="${2:?missing value for --data-dir}"; shift 2 ;;
    --output-root) RUNS_ROOT_OVERRIDE="${2:?missing value for --output-root}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[error] unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -n "$EXP_NAME" ]] || { echo "[error] --experiment is required" >&2; usage >&2; exit 2; }
[[ -n "$GPU_SPEC_RAW" ]] || { echo "[error] --gpu-ids is required" >&2; usage >&2; exit 2; }
RUN_ID="${RUN_ID:-train_${EXP_NAME}_$(date +%Y%m%d_%H%M%S)}"
GPU_SPEC="${GPU_SPEC_RAW// /}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
TRAIN_ROOT="$PROJECT_ROOT/2-training"
CORE_DIR="$TRAIN_ROOT/core"
DATA_DIR="${DATA_DIR_OVERRIDE:-${TAD_DATA_DIR:-$PROJECT_ROOT/0-data/1_dp_train_infer_data}}"
RUNS_ROOT="${RUNS_ROOT_OVERRIDE:-${RUNS_ROOT:-$TRAIN_ROOT/step1_train/outputs}}"
RUN_DIR="$RUNS_ROOT/$RUN_ID"
OUT_DIR="$RUN_DIR/train_outputs"
LOG_DIR="$RUN_DIR/logs"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-1}"

if [[ ! -f "$DATA_DIR/window_list.json" ]]; then
  echo "[error] missing training data: $DATA_DIR/window_list.json" >&2
  exit 2
fi

IFS=',' read -r -a GPU_ARRAY <<< "$GPU_SPEC"
NPROC="${#GPU_ARRAY[@]}"
if [[ "$NPROC" -lt 1 ]]; then
  echo "[error] invalid gpu_ids: $GPU_SPEC_RAW" >&2
  exit 2
fi

EXTRA_ARGS=()
case "$EXP_NAME" in
  base)
    ;;
  obs_input)
    EXTRA_ARGS+=(--use_obs)
    ;;
  no_pairloss)
    EXTRA_ARGS+=(--pair_weight 0.0)
    ;;
  no_count)
    EXTRA_ARGS+=(--count_weight 0.0)
    ;;
  boundary_count)
    EXTRA_ARGS+=(--pair_weight 0.0 --boundary_weight 1.0)
    ;;
  *)
    echo "[error] unsupported experiment: $EXP_NAME" >&2
    exit 2
    ;;
esac

mkdir -p "$OUT_DIR" "$LOG_DIR"

LAUNCH_ARGS=(--num_processes "$NPROC" --mixed_precision "$MIXED_PRECISION")
if [[ "$NPROC" -gt 1 ]]; then
  LAUNCH_ARGS+=(--multi_gpu)
  if [[ -n "$MASTER_PORT" ]]; then
    LAUNCH_ARGS+=(--main_process_port "$MASTER_PORT")
  fi
fi

TRAIN_ARGS=(
  --data_dir "$DATA_DIR"
  --output_dir "$OUT_DIR"
  --epochs "$EPOCHS"
  --batch_size "$BATCH_SIZE"
  --grad_accum_steps "$GRAD_ACCUM_STEPS"
  --mixed_precision "$MIXED_PRECISION"
  --lr 2e-4
  --init_lr 1e-8
  --min_lr 1e-8
  --weight_decay 0.05
  --warmup_epochs 1.0
  --hidden 128
  --diag_radius 50
  --seed 42
  --train_sampling random_single
  --val_sampling random_single
  --pair_weight 1.0
  --count_weight "$COUNT_LAMBDA"
  --boundary_weight 0.0
  --pair_hard_window 8
  --max_grad_norm "$MAX_GRAD_NORM"
  --num_workers 4
  --save_every_epoch
)
TRAIN_ARGS+=("${EXTRA_ARGS[@]}")

if [[ -n "${EXTRA_TRAIN_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  USER_ARGS=(${EXTRA_TRAIN_ARGS})
  TRAIN_ARGS+=("${USER_ARGS[@]}")
fi

CMD=(accelerate launch "${LAUNCH_ARGS[@]}" "$CORE_DIR/train.py" "${TRAIN_ARGS[@]}")
{
  echo "#!/usr/bin/env bash"
  echo "set -euo pipefail"
  printf "CUDA_VISIBLE_DEVICES=%q " "$GPU_SPEC"
  printf "%q " "${CMD[@]}"
  echo
} > "$RUN_DIR/run_command.sh"
chmod +x "$RUN_DIR/run_command.sh"

{
  echo "[info] experiment=$EXP_NAME"
  echo "[info] run_id=$RUN_ID"
  echo "[info] gpu_ids=$GPU_SPEC"
  echo "[info] epochs=$EPOCHS batch_size=$BATCH_SIZE"
  echo "[info] mixed_precision=$MIXED_PRECISION grad_accum_steps=$GRAD_ACCUM_STEPS"
  echo "[info] data_dir=$DATA_DIR"
  echo "[info] output_dir=$OUT_DIR"
  echo "[info] extra_args=${EXTRA_ARGS[*]:-(none)}"
} | tee "$LOG_DIR/meta.log"

CUDA_VISIBLE_DEVICES="$GPU_SPEC" "${CMD[@]}" 2>&1 | tee "$LOG_DIR/train.log"

echo "[done] $RUN_DIR"
echo "[done] best checkpoint: $OUT_DIR/best_model.pt"
