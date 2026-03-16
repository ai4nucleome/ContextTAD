#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STEP_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_ROOT="$(cd "${STEP_DIR}/../.." && pwd)"
EXPS_ROOT="${EXPS_ROOT:-$PROJECT_ROOT/3-evaluation/common}"
EXP2_RUNNER="$EXPS_ROOT/exp2_struct_protein/run_exp2.sh"

RUN_ID="${1:-exp2_ablation_$(date +%Y%m%d_%H%M%S)}"
EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-$PROJECT_ROOT/0-data/2_eval_tads_data}"
EVAL_TOOLS_ROOT="${EVAL_TOOLS_ROOT:-$EVAL_DATA_ROOT/tools_output}"

RUN_DIR="${STEP_DIR}/outputs/exp2_struct_protein/runs/${RUN_ID}"
RESULTS_DIR="${RUN_DIR}/results"
SUMMARY_DIR="${RUN_DIR}/summary"
LATEST_LINK="${STEP_DIR}/outputs/exp2_struct_protein/latest"

mkdir -p "${RESULTS_DIR}" "${SUMMARY_DIR}"

declare -A METHOD_BED
METHOD_BED[base]="${BASE_BED:-$STEP_DIR/outputs/eval_base_epoch005_valrand_dualratio_decodefix_20260305/beds/GM12878_250M_contexttad.bed}"
METHOD_BED[no_tofe]="${NO_TOFE_BED:-$STEP_DIR/outputs/eval_no_tofe_epoch005_valrand_dualratio_decodefix_20260305/beds/GM12878_250M_contexttad.bed}"
METHOD_BED[no_text]="${NO_TEXT_BED:-$STEP_DIR/outputs/eval_no_text_epoch005_valrand_dualratio_decodefix_20260305/beds/GM12878_250M_contexttad.bed}"
METHOD_BED[no_pairloss]="${NO_PAIRLOSS_BED:-$STEP_DIR/outputs/eval_no_pairloss_epoch005_valrand_dualratio_decodefix_20260305/beds/GM12878_250M_contexttad.bed}"
METHOD_BED[obs_input]="${OBS_INPUT_BED:-$STEP_DIR/outputs/eval_obsinput_epoch005_valrand_dualratio_decodefix_20260305/beds/GM12878_250M_contexttad.bed}"

METHOD_ORDER=(
  base
  no_tofe
  no_text
  no_pairloss
  obs_input
)

if [[ ! -x "$EXP2_RUNNER" ]]; then
  echo "[error] exp2 runner not found: $EXP2_RUNNER" >&2
  exit 2
fi

for method in "${METHOD_ORDER[@]}"; do
  bed_path="${METHOD_BED[$method]}"
  if [[ ! -f "${bed_path}" ]]; then
    echo "[error] bed not found for ${method}: ${bed_path}" >&2
    exit 2
  fi
done

echo "========================================"
echo "exp2 struct protein for model ablation"
echo "run dir: ${RUN_DIR}"
echo "========================================"

for method in "${METHOD_ORDER[@]}"; do
  bed_path="${METHOD_BED[$method]}"
  method_dir="${RESULTS_DIR}/${method}"
  method_struct_dir="${method_dir}/struct_protein"
  method_fig_dir="${method_dir}/exp2_csv"
  mkdir -p "$method_dir"

  echo "[run] ${method}"
  EVAL_DATA_ROOT="$EVAL_DATA_ROOT" \
  EVAL_TOOLS_ROOT="$EVAL_TOOLS_ROOT" \
  OUTPUT_DIR="$method_struct_dir" \
  FIG_DIR="$method_fig_dir" \
  TOOL_FILTER="ContextTAD" \
  CONTEXTTAD_BED_OVERRIDE="$bed_path" \
  CLEAN_OUTPUT=1 \
  bash "$EXP2_RUNNER" > "${method_dir}/run.log" 2>&1
done

COMBINED_TSV="${SUMMARY_DIR}/StructProteins_ablation.tsv"
{
  printf "TadsFile\tresolution_kb\tprotein\tdomains_ratio\tfc_over_bg\tpval_of_peak\n"
  for method in "${METHOD_ORDER[@]}"; do
    result_file="${RESULTS_DIR}/${method}/struct_protein/StructProteins_allTools.tsv"
    if [[ ! -f "$result_file" ]]; then
      echo "[error] missing result file for ${method}: ${result_file}" >&2
      exit 1
    fi
    tail -n +2 "$result_file" | awk -F'\t' -v OFS='\t' -v tool="$method" '{print tool, $2, $3, $4, $5, $6}'
  done
} > "$COMBINED_TSV"

python3 "${SCRIPT_DIR}/summarize_exp2_struct_protein_ablation.py" \
  --input-tsv "${COMBINED_TSV}" \
  --output-csv "${SUMMARY_DIR}/exp2_ablation_results.csv" \
  --output-md "${SUMMARY_DIR}/exp2_ablation_results.md"

ln -sfn "${RUN_DIR}" "${LATEST_LINK}"

echo
echo "done"
echo "  combined: ${COMBINED_TSV}"
echo "  summary : ${SUMMARY_DIR}/exp2_ablation_results.csv"
echo "  markdown: ${SUMMARY_DIR}/exp2_ablation_results.md"
