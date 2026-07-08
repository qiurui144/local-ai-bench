#!/usr/bin/env bash
set -euo pipefail

# Run locally. Full VLM validation for SpacemiT archive model_zoo/vlm on K3 32G.
#
# Artifacts are cached under drivers/spacemit-ai/model_zoo (ignored by git),
# copied to the target one model at a time, and evaluated on the synthetic
# VLM document extraction suite in datasets/scenarios/vlm_document_extraction.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

k3_require_target_env
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/drivers/spacemit-ai/model_zoo}"
REMOTE_OUT_ROOT="${REMOTE_OUT_ROOT:-/root/k3_32g_model_zoo_vlm_full/$(date +%Y%m%d_%H%M%S)}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/local-ai-bench}"
PORT_BASE_START="${PORT_BASE_START:-18700}"
DOWNLOAD="${DOWNLOAD:-1}"

# Set VLM_MAX_CASES=0 for all cases. Use a small positive number for smoke.
VLM_MAX_CASES="${VLM_MAX_CASES:-0}"
VLM_DOC_MAX_TOKENS="${VLM_DOC_MAX_TOKENS:-192}"
CONTEXT_LADDER="${CONTEXT_LADDER:-1024,3072}"
RUN_BENCH="${RUN_BENCH:-1}"
RUN_SMOKE="${RUN_SMOKE:-0}"
RUN_UPSTREAM="${RUN_UPSTREAM:-0}"
FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-1}"
CLEAN_REMOTE_AFTER="${CLEAN_REMOTE_AFTER:-1}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
CTX_SIZE="${CTX_SIZE:-4096}"
CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"

VLM_PROMPT="${VLM_PROMPT:-请识别图片中的主要文字和关键信息，只输出简短 JSON。}"

RUN_TARS="${RUN_TARS:-1}"
RUN_PAIRS="${RUN_PAIRS:-1}"
RUN_HEAVY="${RUN_HEAVY:-1}"

VLM_TAR_COMMON=(
  Qwen3.5-4B.tar.gz
  Qwen3.5-2B.tar.gz
  Qwen3.5-0.8B.tar.gz
  fastvlm-mm-0.5b-q4_1.tar.gz
)
VLM_TAR_HEAVY=(
  qwen30ba3b-mm-q4_1.tar.gz
)
VLM_PAIR_MODELS=(
  "Qwen3VL|Qwen3VL-4B-Instruct-Q4_K_M.gguf|mmproj-Qwen3VL-4B-Instruct-F16.gguf"
  "SmolVLM|SmolVLM-256M-Instruct-f16.gguf|mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

run_cached() {
  local mode="$1" model_file="$2" mmproj_file="${3:-}" alias="$4" port="$5" out_dir="$6"
  env \
    K3_HOST="${K3_HOST}" \
    K3_USER="${K3_USER}" \
    CACHE_ROOT="${CACHE_ROOT}" \
    REMOTE_WORKDIR="${REMOTE_WORKDIR}" \
    MODE="${mode}" \
    MODEL_FILE="${model_file}" \
    MMPROJ_FILE="${mmproj_file}" \
    ALIAS="${alias}" \
    OUT_DIR="${out_dir}" \
    PORT_BASE="${port}" \
    FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE}" \
    RUN_BENCH="${RUN_BENCH}" \
    RUN_SMOKE="${RUN_SMOKE}" \
    RUN_UPSTREAM="${RUN_UPSTREAM}" \
    CTX_SIZE="${CTX_SIZE}" \
    CONTEXT_LADDER="${CONTEXT_LADDER}" \
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
    CLEAN_REMOTE_AFTER="${CLEAN_REMOTE_AFTER}" \
    CACHE_TYPE_K="${CACHE_TYPE_K}" \
    CACHE_TYPE_V="${CACHE_TYPE_V}" \
    VLM_MAX_CASES="${VLM_MAX_CASES}" \
    VLM_DOC_MAX_TOKENS="${VLM_DOC_MAX_TOKENS}" \
    VLM_PROMPT="${VLM_PROMPT}" \
    bash "${REPO_ROOT}/scripts/run_k3_32g_model_zoo_cached.sh"
}

mkdir -p "${REPO_ROOT}/reports/runs/k3-riscv-32g"

if [[ "${DOWNLOAD}" == "1" ]]; then
  log "cache SpacemiT archive model_zoo/vlm under ${CACHE_ROOT}"
  (cd "${REPO_ROOT}" && CACHE_ROOT="${CACHE_ROOT}" SCOPE=vlm bash scripts/cache_spacemit_model_zoo.sh)
fi

log "remote output root: ${REMOTE_OUT_ROOT}"
port="${PORT_BASE_START}"

if [[ "${RUN_TARS}" == "1" ]]; then
  tar_models=("${VLM_TAR_COMMON[@]}")
  if [[ "${RUN_HEAVY}" == "1" ]]; then
    tar_models=("${VLM_TAR_HEAVY[@]}" "${tar_models[@]}")
  fi
  for file in "${tar_models[@]}"; do
    alias="${file%.tar.gz}"
    out="${REMOTE_OUT_ROOT}/vlm-tar/${alias}"
    log "run VLM tar ${file} port=${port}"
    run_cached "vlm-tar" "${file}" "" "${alias}" "${port}" "${out}"
    port=$((port + 2))
  done
fi

if [[ "${RUN_PAIRS}" == "1" ]]; then
  for spec in "${VLM_PAIR_MODELS[@]}"; do
    IFS='|' read -r name model_file mmproj_file <<<"${spec}"
    out="${REMOTE_OUT_ROOT}/vlm-pair/${name}"
    log "run VLM pair ${name} port=${port}"
    run_cached "vlm-pair" "${model_file}" "${mmproj_file}" "${model_file}" "${port}" "${out}"
    port=$((port + 2))
  done
fi

cat <<EOF
Full VLM run submitted/completed sequentially.
Remote output root: ${REMOTE_OUT_ROOT}
Dataset: ${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction/cases.jsonl
Case limit: ${VLM_MAX_CASES} (0 means all)
EOF
