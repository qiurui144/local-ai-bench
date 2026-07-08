#!/usr/bin/env bash
set -uo pipefail

# Run locally. Continue cached full-matrix validation for remaining model_zoo
# entries. A single model failure is recorded and does not stop the batch.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

k3_require_target_env
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-output/reports/k3-riscv-32g}"
REMOTE_OUT_ROOT="${REMOTE_OUT_ROOT:-/root/k3_32g_full}"
PORT_BASE_START="${PORT_BASE_START:-18160}"
VLM_IMAGE_PATH="${VLM_IMAGE_PATH:-/root/k3_images/receipt_c17.png}"
VLM_PROMPT="${VLM_PROMPT:-请识别图片中的票据类型、日期、金额和商户信息，只输出简短 JSON。}"

mkdir -p "${LOCAL_OUT_ROOT}"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

pull_dir() {
  local remote_dir="$1"
  local name
  name="$(basename "${remote_dir}")"
  if command -v rsync >/dev/null 2>&1; then
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e rsync -ah --delete \
        -e "ssh ${SSH_OPTS[*]}" \
        "${K3_USER}@${K3_HOST}:${remote_dir}/" "${LOCAL_OUT_ROOT}/${name}/" || true
    else
      rsync -ah --delete \
        -e "ssh ${SSH_OPTS[*]}" \
        "${K3_USER}@${K3_HOST}:${remote_dir}/" "${LOCAL_OUT_ROOT}/${name}/" || true
    fi
  fi
}

run_llm() {
  local file="$1" port="$2" stamp out rc
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="${REMOTE_OUT_ROOT}/${file%.gguf}-cached-${stamp}"
  log "LLM start ${file} port=${port} out=${out}"
  K3_HOST="${K3_HOST}" K3_USER="${K3_USER}" \
  MODE=llm MODEL_FILE="${file}" ALIAS="${file%.gguf}" OUT_DIR="${out}" \
  RUN_PRIVATE=1 RUN_UPSTREAM=0 RUN_SERVER=1 CLEAN_REMOTE_AFTER=1 \
  CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 CTX_SIZE=4096 CONTEXT_LADDER=1024,3072 \
  BENCH_PROMPT=512 BENCH_GEN=128 BENCH_REPS=1 BENCH_TIMEOUT=1200 \
  SMOKE_TIMEOUT=60 SMOKE_LOG_LIMIT_BYTES=1048576 \
  PORT_BASE="${port}" THREADS=8 THREADS_BATCH=8 BATCH_SIZE=1024 UBATCH_SIZE=512 REQUEST_TIMEOUT=900 \
    bash "${REPO_ROOT}/scripts/run_k3_32g_model_zoo_cached.sh"
  rc=$?
  log "LLM done ${file} rc=${rc}"
  pull_dir "${out}"
  return 0
}

run_vlm_tar() {
  local file="$1" port="$2" stamp out rc
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="${REMOTE_OUT_ROOT}/${file%.tar.gz}-cached-${stamp}"
  log "VLM tar start ${file} port=${port} out=${out}"
  K3_HOST="${K3_HOST}" K3_USER="${K3_USER}" \
  MODE=vlm-tar MODEL_FILE="${file}" ALIAS="${file%.tar.gz}" OUT_DIR="${out}" \
  RUN_PRIVATE=1 RUN_UPSTREAM=0 RUN_SERVER=1 CLEAN_REMOTE_AFTER=1 \
  CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 CTX_SIZE=4096 CONTEXT_LADDER=1024,3072 \
  BENCH_PROMPT=512 BENCH_GEN=128 BENCH_REPS=1 BENCH_TIMEOUT=1200 \
  SMOKE_TIMEOUT=60 SMOKE_LOG_LIMIT_BYTES=1048576 \
  PORT_BASE="${port}" THREADS=8 THREADS_BATCH=8 BATCH_SIZE=1024 UBATCH_SIZE=512 REQUEST_TIMEOUT=900 \
  VLM_IMAGE_PATH="${VLM_IMAGE_PATH}" VLM_PROMPT="${VLM_PROMPT}" \
    bash "${REPO_ROOT}/scripts/run_k3_32g_model_zoo_cached.sh"
  rc=$?
  log "VLM tar done ${file} rc=${rc}"
  pull_dir "${out}"
  return 0
}

run_vlm_pair() {
  local name="$1" model_file="$2" mmproj_file="$3" port="$4" stamp out rc
  stamp="$(date +%Y%m%d_%H%M%S)"
  out="${REMOTE_OUT_ROOT}/${name}-cached-${stamp}"
  log "VLM pair start ${name} port=${port} out=${out}"
  K3_HOST="${K3_HOST}" K3_USER="${K3_USER}" \
  MODE=vlm-pair MODEL_FILE="${model_file}" MMPROJ_FILE="${mmproj_file}" ALIAS="${name}" OUT_DIR="${out}" \
  RUN_PRIVATE=1 RUN_UPSTREAM=0 RUN_SERVER=1 CLEAN_REMOTE_AFTER=1 \
  CACHE_TYPE_K=f16 CACHE_TYPE_V=f16 CTX_SIZE=4096 CONTEXT_LADDER=1024,3072 \
  BENCH_PROMPT=512 BENCH_GEN=128 BENCH_REPS=1 BENCH_TIMEOUT=1200 \
  SMOKE_TIMEOUT=60 SMOKE_LOG_LIMIT_BYTES=1048576 \
  PORT_BASE="${port}" THREADS=8 THREADS_BATCH=8 BATCH_SIZE=1024 UBATCH_SIZE=512 REQUEST_TIMEOUT=900 \
  VLM_IMAGE_PATH="${VLM_IMAGE_PATH}" VLM_PROMPT="${VLM_PROMPT}" \
    bash "${REPO_ROOT}/scripts/run_k3_32g_model_zoo_cached.sh"
  rc=$?
  log "VLM pair done ${name} rc=${rc}"
  pull_dir "${out}"
  return 0
}

main() {
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "mkdir -p /root/k3_images" || true
  if [[ -f fixtures/scenarios/vlm_document_extraction/receipt/c17.png ]]; then
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e scp "${SSH_OPTS[@]}" fixtures/scenarios/vlm_document_extraction/receipt/c17.png \
        "${K3_USER}@${K3_HOST}:/root/k3_images/receipt_c17.png" || true
    else
      scp "${SSH_OPTS[@]}" fixtures/scenarios/vlm_document_extraction/receipt/c17.png \
        "${K3_USER}@${K3_HOST}:/root/k3_images/receipt_c17.png" || true
    fi
  fi

  local port="${PORT_BASE_START}"
  local llm_models=(
    SmallThinker-4B-A0.6B-Instruct.Q4_0.gguf
    qwen2.5-3b-instruct-q4_0.gguf
    Qwen3.5-2B-Q4_0.gguf
    HY-MT1.5-1.8B-Q4_K_M.gguf
    Qwen3-1.7B-Q8_0.gguf
    deepseek-r1-distill-qwen-1.5b-q4_0.gguf
    glm-edge-1.5b-chat-q4_0.gguf
    qwen2.5-1.5b-instruct-q4_0.gguf
    LFM2.5-1.2B-Instruct-Q4_0.gguf
    Qwen3.5-0.8B-Q4_0.gguf
    Qwen3-0.6B-Q4_K_M.gguf
    Qwen3-0.6B-Q4_1.gguf
    Qwen3-0.6B-Q4_0.gguf
    qwen2.5-0.5b-instruct-q4_0.gguf
  )
  for model in "${llm_models[@]}"; do
    run_llm "${model}" "${port}"
    port=$((port + 2))
  done

  local vlm_tars=(
    Qwen3.5-2B.tar.gz
    Qwen3.5-0.8B.tar.gz
    fastvlm-mm-0.5b-q4_1.tar.gz
    qwen3-asr-0.6B.tar.gz
  )
  for model in "${vlm_tars[@]}"; do
    run_vlm_tar "${model}" "${port}"
    port=$((port + 2))
  done

  run_vlm_pair SmolVLM-256M-Instruct \
    SmolVLM-256M-Instruct-f16.gguf \
    mmproj-SmolVLM-256M-Instruct-Q8_0.gguf "${port}"

  log "remaining cached matrix done"
}

main "$@"
