#!/usr/bin/env bash
set -euo pipefail

# Run locally. Push one cached SpacemiT model_zoo artifact to the K3 target and
# invoke the remote high-spec runner. Set SSHPASS in the environment when using
# password auth with sshpass.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

if [[ "${1:-}" == "--describe" ]]; then
  k3_print_target_contract
  cat <<'EOF'

Script: run_k3_32g_model_zoo_cached.sh
Purpose: copy one cached LLM/VLM artifact from drivers/ to K3, run the remote high-spec harness, and optionally clean the K3 copy.

Official references:
  Invocation/application: https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/application_tools/ai-sdk.md
  Performance baseline:   https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md

Data sources:
  llm      drivers/spacemit-ai/model_zoo/llm/<MODEL_FILE>
  vlm-tar  drivers/spacemit-ai/model_zoo/vlm/<MODEL_FILE>
  vlm-pair drivers/spacemit-ai/model_zoo/vlm[/Qwen3VL]/<MODEL_FILE> + <MMPROJ_FILE>

Call contract:
  AI SDK application layer:
    LLM -> llama-server plus llm_chat, or gateway POST /v1/chat/completions
    VLM -> gateway POST /v1/vlm/models/load then POST /v1/vlm/chat/completions with image_url
  ModelZoo performance layer:
    LLM -> llama-bench -p 128 -n 128 -mmp 0 -fa 1 -ub 128
    VLM -> SMT llama-server path, with --media-backend/--vision-backend smt and --smt-config-dir
  Exact LLM baseline retest:
    RUN_OFFICIAL_MODELZOO_BENCH=1 RUN_BENCH=0 RUN_SERVER=0 RUN_SMOKE=0

Invocation modes:
  MODE=llm      -> llama-server -m <MODEL_PATH> /v1/chat/completions
  MODE=vlm-tar  -> extract tar, llama-server SMT backend /v1/chat/completions with image_url
  MODE=vlm-pair -> llama-server -m <MODEL_PATH> --mmproj <MMPROJ_PATH> /v1/chat/completions with image_url

Required knobs:
  MODEL_FILE=<artifact filename>
  MMPROJ_FILE=<mmproj filename> for MODE=vlm-pair
  K3_HOST/K3_USER connection settings
EOF
  exit 0
fi

k3_require_target_env
CACHE_ROOT="${CACHE_ROOT:-drivers/spacemit-ai/model_zoo}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/k3_scripts}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-/root/models/spacemit-ai}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-/root/local-ai-bench}"
SYNC_VLM_DATASET="${SYNC_VLM_DATASET:-1}"
REUSE_REMOTE_EXTRACT="${REUSE_REMOTE_EXTRACT:-1}"
MODE="${MODE:-llm}" # llm | vlm-tar | vlm-pair
MODEL_FILE="${MODEL_FILE:-}"
MMPROJ_FILE="${MMPROJ_FILE:-}"
ALIAS="${ALIAS:-${MODEL_FILE}}"
OUT_DIR="${OUT_DIR:-/root/k3_32g_cached/${ALIAS%.*}-$(date +%Y%m%d_%H%M%S)}"
PORT_BASE="${PORT_BASE:-18100}"
CLEAN_REMOTE_AFTER="${CLEAN_REMOTE_AFTER:-0}"
RSYNC_PROGRESS="${RSYNC_PROGRESS:-0}"

LLM_BASE="${LLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/llm}"
VLM_BASE="${VLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/vlm}"
QWEN3VL_BASE="${QWEN3VL_BASE:-${VLM_BASE}/Qwen3VL}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
SCP_CMD=(scp "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
fi

remote() {
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "$@"
}

copy_file() {
  local src="$1" dst_dir="$2"
  [[ -s "${src}" ]] || { echo "missing local cache file: ${src}" >&2; exit 3; }
  remote "mkdir -p '${dst_dir}' '${REMOTE_SCRIPT_DIR}'"
  if command -v rsync >/dev/null 2>&1; then
    local rsync_opts=(-ah --partial)
    if [[ "${RSYNC_PROGRESS}" == "1" ]]; then
      rsync_opts+=(--info=progress2)
    fi
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e rsync "${rsync_opts[@]}" \
        -e "ssh ${SSH_OPTS[*]}" "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
    else
      rsync "${rsync_opts[@]}" \
        -e "ssh ${SSH_OPTS[*]}" "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
    fi
  else
    "${SCP_CMD[@]}" -p "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
  fi
}

copy_runner() {
  remote "mkdir -p '${REMOTE_SCRIPT_DIR}'"
  "${SCP_CMD[@]}" -p \
    scripts/run_k3_32g_model_zoo_highspec.sh \
    scripts/run_k3_32g_model_zoo_full_matrix.sh \
    "${K3_USER}@${K3_HOST}:${REMOTE_SCRIPT_DIR}/"
}

copy_vlm_dataset() {
  [[ "${SYNC_VLM_DATASET}" == "1" ]] || return 0
  remote "mkdir -p '${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction' '${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction'"
  if command -v rsync >/dev/null 2>&1; then
    local ssh_cmd="ssh ${SSH_OPTS[*]}"
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e rsync -ah --delete -e "${ssh_cmd}" \
        datasets/scenarios/vlm_document_extraction/ \
        "${K3_USER}@${K3_HOST}:${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction/"
      sshpass -e rsync -ah --delete -e "${ssh_cmd}" \
        fixtures/scenarios/vlm_document_extraction/ \
        "${K3_USER}@${K3_HOST}:${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction/"
    else
      rsync -ah --delete -e "${ssh_cmd}" \
        datasets/scenarios/vlm_document_extraction/ \
        "${K3_USER}@${K3_HOST}:${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction/"
      rsync -ah --delete -e "${ssh_cmd}" \
        fixtures/scenarios/vlm_document_extraction/ \
        "${K3_USER}@${K3_HOST}:${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction/"
    fi
  else
    tar -C . -cf - datasets/scenarios/vlm_document_extraction fixtures/scenarios/vlm_document_extraction \
      | "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "cd '${REMOTE_WORKDIR}' && tar -xf -"
  fi
}

needs_no_think() {
  case "$1" in
    Qwen3*|qwen3*|qwen30b*|deepseek-r1*|DeepSeek-R1*) return 0 ;;
    *) return 1 ;;
  esac
}

common_env() {
  printf '%q ' \
    RUN_PRIVATE="${RUN_PRIVATE:-1}" \
    RUN_UPSTREAM="${RUN_UPSTREAM:-0}" \
    RUN_SERVER="${RUN_SERVER:-1}" \
    RUN_SMOKE="${RUN_SMOKE:-1}" \
    RUN_BENCH="${RUN_BENCH:-1}" \
    RUN_OFFICIAL_MODELZOO_BENCH="${RUN_OFFICIAL_MODELZOO_BENCH:-0}" \
    PRIVATE_ENV="${PRIVATE_ENV-SPACEMIT_DISABLE_TCM=1}" \
    CACHE_TYPE_K="${CACHE_TYPE_K:-f16}" \
    CACHE_TYPE_V="${CACHE_TYPE_V:-f16}" \
    CTX_SIZE="${CTX_SIZE:-4096}" \
    CONTEXT_LADDER="${CONTEXT_LADDER-1024,3072}" \
    BENCH_PROMPT="${BENCH_PROMPT:-512}" \
    BENCH_GEN="${BENCH_GEN:-128}" \
    BENCH_REPS="${BENCH_REPS:-1}" \
    BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}" \
    SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}" \
    SMOKE_LOG_LIMIT_BYTES="${SMOKE_LOG_LIMIT_BYTES:-1048576}" \
    FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-0}" \
    THREADS="${THREADS:-8}" \
    THREADS_BATCH="${THREADS_BATCH:-8}" \
    BATCH_SIZE="${BATCH_SIZE:-1024}" \
    UBATCH_SIZE="${UBATCH_SIZE:-512}" \
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}" \
    VLM_CASES_JSONL="${VLM_CASES_JSONL:-}" \
    VLM_MAX_CASES="${VLM_MAX_CASES:-0}" \
    VLM_CASE_IDS="${VLM_CASE_IDS:-}" \
    VLM_DOC_MAX_TOKENS="${VLM_DOC_MAX_TOKENS:-192}" \
    RUN_VISUAL_PROBE="${RUN_VISUAL_PROBE:-0}" \
    RUN_VLM_DOC_PROBE="${RUN_VLM_DOC_PROBE:-1}" \
    PORT_BASE="${PORT_BASE}" \
    OUT_DIR="${OUT_DIR}"
}

vlm_env() {
  printf '%q ' \
    VLM_IMAGE_PATH="${VLM_IMAGE_PATH:-}" \
    VLM_PROMPT="${VLM_PROMPT:-请识别图片中的主要文字和关键信息，只输出简短 JSON。}"
}

copy_runner
case "${MODE}" in
  vlm-tar|vlm-pair) copy_vlm_dataset ;;
  llm)
    if [[ "${RUN_VISUAL_PROBE:-0}" == "1" ]]; then
      copy_vlm_dataset
    fi
    ;;
esac

prompt_prefix=""
chat_template_kwargs=""
if needs_no_think "${MODEL_FILE}"; then
  prompt_prefix=$'/no_think\n'
  chat_template_kwargs='{"enable_thinking":false}'
fi

case "${MODE}" in
  llm)
    [[ -n "${MODEL_FILE}" ]] || { echo "MODEL_FILE is required" >&2; exit 2; }
    local_path="${CACHE_ROOT}/llm/${MODEL_FILE}"
    remote_dir="${REMOTE_MODEL_ROOT}/llm"
    copy_file "${local_path}" "${remote_dir}"
    remote_path="${remote_dir}/${MODEL_FILE}"
    remote_image="${VLM_IMAGE_PATH:-${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction/receipt/c17.png}"
    remote_env="$(common_env) $(vlm_env) MODE=llm MODEL_URL=$(printf '%q' "${LLM_BASE}/${MODEL_FILE}") MODEL_DIR=$(printf '%q' "${remote_dir}") MODEL_FILE=$(printf '%q' "${MODEL_FILE}") MODEL_PATH=$(printf '%q' "${remote_path}") VLM_IMAGE_PATH=$(printf '%q' "${remote_image}") ALIAS=$(printf '%q' "${ALIAS}") PROMPT_PREFIX=$(printf '%q' "${prompt_prefix}") CHAT_TEMPLATE_KWARGS_JSON=$(printf '%q' "${chat_template_kwargs}")"
    remote "cd /root && env ${remote_env} bash '${REMOTE_SCRIPT_DIR}/run_k3_32g_model_zoo_highspec.sh'"
    if [[ "${CLEAN_REMOTE_AFTER}" == "1" ]]; then
      remote "rm -f '${remote_path}'"
    fi
    ;;
  vlm-tar)
    [[ -n "${MODEL_FILE}" ]] || { echo "MODEL_FILE is required" >&2; exit 2; }
    local_path="${CACHE_ROOT}/vlm/${MODEL_FILE}"
    remote_dir="${REMOTE_MODEL_ROOT}/vlm"
    remote_tar="${remote_dir}/${MODEL_FILE}"
    extract_dir="${remote_dir}/${MODEL_FILE%.tar.gz}"
    copied_remote_tar=0
    extract_vlm="${EXTRACT_VLM:-1}"
    if [[ "${REUSE_REMOTE_EXTRACT}" == "1" ]] \
      && remote "test -d '${extract_dir}' && find '${extract_dir}' -type f -iname '*.gguf' | grep -q ."; then
      echo "reuse remote VLM extract dir: ${extract_dir}"
      extract_vlm=0
    else
      copy_file "${local_path}" "${remote_dir}"
      copied_remote_tar=1
    fi
    remote_cases="${VLM_CASES_JSONL:-${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction/cases.jsonl}"
    remote_image="${VLM_IMAGE_PATH:-${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction/receipt/c17.png}"
    remote_env="$(common_env) $(vlm_env) MODE=vlm-tar EXTRACT_VLM=$(printf '%q' "${extract_vlm}") VLM_TAR_URL=$(printf '%q' "${VLM_BASE}/${MODEL_FILE}") VLM_TAR_DIR=$(printf '%q' "${remote_dir}") VLM_TAR_FILE=$(printf '%q' "${MODEL_FILE}") VLM_EXTRACT_DIR=$(printf '%q' "${extract_dir}") VLM_CASES_JSONL=$(printf '%q' "${remote_cases}") VLM_IMAGE_PATH=$(printf '%q' "${remote_image}") ALIAS=$(printf '%q' "${ALIAS:-${MODEL_FILE%.tar.gz}}") PROMPT_PREFIX=$(printf '%q' "${prompt_prefix}") CHAT_TEMPLATE_KWARGS_JSON=$(printf '%q' "${chat_template_kwargs}")"
    remote "cd /root && env ${remote_env} bash '${REMOTE_SCRIPT_DIR}/run_k3_32g_model_zoo_highspec.sh'"
    if [[ "${CLEAN_REMOTE_AFTER}" == "1" && "${copied_remote_tar}" == "1" ]]; then
      remote "rm -f '${remote_tar}'; rm -rf '${extract_dir}'"
    fi
    ;;
  vlm-pair)
    [[ -n "${MODEL_FILE}" && -n "${MMPROJ_FILE}" ]] || { echo "MODEL_FILE and MMPROJ_FILE are required" >&2; exit 2; }
    if [[ -s "${CACHE_ROOT}/vlm/Qwen3VL/${MODEL_FILE}" ]]; then
      local_model="${CACHE_ROOT}/vlm/Qwen3VL/${MODEL_FILE}"
      local_mmproj="${CACHE_ROOT}/vlm/Qwen3VL/${MMPROJ_FILE}"
      remote_dir="${REMOTE_MODEL_ROOT}/vlm/Qwen3VL"
      url_base="${QWEN3VL_BASE}"
    else
      local_model="${CACHE_ROOT}/vlm/${MODEL_FILE}"
      local_mmproj="${CACHE_ROOT}/vlm/${MMPROJ_FILE}"
      remote_dir="${REMOTE_MODEL_ROOT}/vlm"
      url_base="${VLM_BASE}"
    fi
    copied_remote_pair=0
    if [[ "${REUSE_REMOTE_EXTRACT}" == "1" ]] \
      && remote "test -s '${remote_dir}/${MODEL_FILE}' && test -s '${remote_dir}/${MMPROJ_FILE}'"; then
      echo "reuse remote VLM pair: ${remote_dir}/${MODEL_FILE} + ${remote_dir}/${MMPROJ_FILE}"
    else
      copy_file "${local_model}" "${remote_dir}"
      copy_file "${local_mmproj}" "${remote_dir}"
      copied_remote_pair=1
    fi
    remote_cases="${VLM_CASES_JSONL:-${REMOTE_WORKDIR}/datasets/scenarios/vlm_document_extraction/cases.jsonl}"
    remote_image="${VLM_IMAGE_PATH:-${REMOTE_WORKDIR}/fixtures/scenarios/vlm_document_extraction/receipt/c17.png}"
    remote_env="$(common_env) $(vlm_env) MODE=vlm-pair MODEL_URL=$(printf '%q' "${url_base}/${MODEL_FILE}") MODEL_DIR=$(printf '%q' "${remote_dir}") MODEL_FILE=$(printf '%q' "${MODEL_FILE}") MODEL_PATH=$(printf '%q' "${remote_dir}/${MODEL_FILE}") MMPROJ_URL=$(printf '%q' "${url_base}/${MMPROJ_FILE}") MMPROJ_DIR=$(printf '%q' "${remote_dir}") MMPROJ_FILE=$(printf '%q' "${MMPROJ_FILE}") VLM_CASES_JSONL=$(printf '%q' "${remote_cases}") VLM_IMAGE_PATH=$(printf '%q' "${remote_image}") ALIAS=$(printf '%q' "${ALIAS}") PROMPT_PREFIX=$(printf '%q' "${prompt_prefix}") CHAT_TEMPLATE_KWARGS_JSON=$(printf '%q' "${chat_template_kwargs}")"
    remote "cd /root && env ${remote_env} bash '${REMOTE_SCRIPT_DIR}/run_k3_32g_model_zoo_highspec.sh'"
    if [[ "${CLEAN_REMOTE_AFTER}" == "1" && "${copied_remote_pair}" == "1" ]]; then
      remote "rm -f '${remote_dir}/${MODEL_FILE}' '${remote_dir}/${MMPROJ_FILE}'"
    fi
    ;;
  *)
    echo "unknown MODE=${MODE}; use llm|vlm-tar|vlm-pair" >&2
    exit 2
    ;;
esac
