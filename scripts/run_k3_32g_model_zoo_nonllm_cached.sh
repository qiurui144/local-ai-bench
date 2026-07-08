#!/usr/bin/env bash
set -euo pipefail

# Run locally. Push one cached non-LLM SpacemiT model_zoo artifact to the K3
# target and invoke the remote non-LLM runner.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

if [[ "${1:-}" == "--describe" ]]; then
  k3_print_target_contract
  cat <<'EOF'

Script: run_k3_32g_model_zoo_nonllm_cached.sh
Purpose: copy one cached embedding/reranker/package artifact from drivers/ to K3 and run the remote non-LLM harness.

Official references:
  Invocation/application: https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/application_tools/ai-sdk.md
  Performance baseline:   https://www.spacemit.com/community/document/info?lang=zh&nodepath=ai/compute_stack/ai_compute_stack/modelzoo.md

Data sources:
  embed   drivers/spacemit-ai/model_zoo/embed/<MODEL_FILE>
  rerank  drivers/spacemit-ai/model_zoo/rerank/<MODEL_FILE>
  asr     drivers/spacemit-ai/model_zoo/asr/<MODEL_FILE>
  vlm     drivers/spacemit-ai/model_zoo/vlm/<MODEL_FILE>
  vision  drivers/spacemit-ai/model_zoo/vision/<MODEL_FILE>

Call contract:
  AI SDK application layer:
    ASR -> asr_file_demo or gateway POST /v1/asr/recognize
    Vision -> component demos or gateway POST /v1/vision/inference
    Embed/Rerank -> gateway-backed capability family; the cited quick verification section does not publish concrete curl routes
  ModelZoo performance layer:
    Vision -> onnxruntime_perf_test with SpaceMITExecutionProvider
    ASR -> official qwen3-ASR/sensevoice RTF rows where published
    Embedding/Reranker -> no official rows in the cited ModelZoo page; local measured only

Invocation modes:
  MODE=embedding   -> llama-server --embedding --pooling mean, POST /v1/embeddings
  MODE=rerank      -> llama-server --reranking --pooling rank, POST /v1/rerank
  MODE=tar-inspect -> list/extract tar package and record model contents
  MODE=file-inspect -> record file metadata only

Required knobs:
  CATEGORY=embed|rerank|asr|vlm|vision
  MODEL_FILE=<artifact filename>
  K3_HOST/K3_USER connection settings
EOF
  exit 0
fi

k3_require_target_env
CACHE_ROOT="${CACHE_ROOT:-drivers/spacemit-ai/model_zoo}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/k3_scripts}"
REMOTE_MODEL_ROOT="${REMOTE_MODEL_ROOT:-/root/models/spacemit-ai}"

MODE="${MODE:-embedding}" # embedding | rerank | tar-inspect
CATEGORY="${CATEGORY:-embed}"
MODEL_FILE="${MODEL_FILE:-}"
ALIAS="${ALIAS:-${MODEL_FILE}}"
OUT_DIR="${OUT_DIR:-/root/k3_32g_nonllm/${ALIAS%.*}-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18220}"
CLEAN_REMOTE_AFTER="${CLEAN_REMOTE_AFTER:-1}"

BASE_URL="${BASE_URL:-}"
case "${CATEGORY}" in
  embed) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/embed" ;;
  rerank) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/rerank" ;;
  asr) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/asr" ;;
  vlm) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vlm" ;;
  vision) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/vision" ;;
  *) DEFAULT_BASE_URL="https://archive.spacemit.com/spacemit-ai/model_zoo/${CATEGORY}" ;;
esac
BASE_URL="${BASE_URL:-${DEFAULT_BASE_URL}}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
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
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e rsync -ah --partial --info=progress2 \
        -e "ssh ${SSH_OPTS[*]}" "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
    else
      rsync -ah --partial --info=progress2 \
        -e "ssh ${SSH_OPTS[*]}" "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
    fi
  else
    "${SCP_CMD[@]}" -p "${src}" "${K3_USER}@${K3_HOST}:${dst_dir}/"
  fi
}

copy_runner() {
  remote "mkdir -p '${REMOTE_SCRIPT_DIR}'"
  "${SCP_CMD[@]}" -p scripts/run_k3_32g_model_zoo_nonllm_highspec.sh \
    "${K3_USER}@${K3_HOST}:${REMOTE_SCRIPT_DIR}/"
}

common_env() {
  printf '%q ' \
    MODE="${MODE}" \
    MODEL_URL="${BASE_URL}/${MODEL_FILE}" \
    MODEL_DIR="${REMOTE_MODEL_ROOT}/${CATEGORY}" \
    MODEL_FILE="${MODEL_FILE}" \
    MODEL_PATH="${REMOTE_MODEL_ROOT}/${CATEGORY}/${MODEL_FILE}" \
    ALIAS="${ALIAS}" \
    OUT_DIR="${OUT_DIR}" \
    PORT="${PORT}" \
    THREADS="${THREADS:-8}" \
    THREADS_BATCH="${THREADS_BATCH:-8}" \
    BATCH_SIZE="${BATCH_SIZE:-512}" \
    UBATCH_SIZE="${UBATCH_SIZE:-256}" \
    CTX_SIZE="${CTX_SIZE:-512}" \
    REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}" \
    EXTRACT_TAR="${EXTRACT_TAR:-1}" \
    TAR_EXTRACT_DIR="${TAR_EXTRACT_DIR:-${REMOTE_MODEL_ROOT}/${CATEGORY}/${MODEL_FILE%.tar.gz}}"
}

[[ -n "${MODEL_FILE}" ]] || { echo "MODEL_FILE is required" >&2; exit 2; }

copy_runner
local_path="${CACHE_ROOT}/${CATEGORY}/${MODEL_FILE}"
remote_dir="${REMOTE_MODEL_ROOT}/${CATEGORY}"
copy_file "${local_path}" "${remote_dir}"

remote_env="$(common_env)"
remote "cd /root && env ${remote_env} bash '${REMOTE_SCRIPT_DIR}/run_k3_32g_model_zoo_nonllm_highspec.sh'"

if [[ "${CLEAN_REMOTE_AFTER}" == "1" ]]; then
  remote "rm -f '${remote_dir}/${MODEL_FILE}'; if [[ '${MODEL_FILE}' == *.tar.gz ]]; then rm -rf '${remote_dir}/${MODEL_FILE%.tar.gz}'; fi"
fi
