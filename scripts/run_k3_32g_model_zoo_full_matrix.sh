#!/usr/bin/env bash
set -euo pipefail

# Run on the SpacemiT K3 32GB target.
# Sequentially validates public SpacemiT model_zoo entries using the
# high-spec runner. Default scope is high-spec first; set SCOPE=all for the
# full public LLM/VLM matrix.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${RUNNER:-${SCRIPT_DIR}/run_k3_32g_model_zoo_highspec.sh}"
OUT_ROOT="${OUT_ROOT:-/root/k3_32g_model_zoo_full/$(date +%Y%m%d_%H%M%S)}"
SCOPE="${SCOPE:-high}" # high | llm | vlm | all

LLM_BASE="${LLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/llm}"
VLM_BASE="${VLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/vlm}"
QWEN3VL_BASE="${QWEN3VL_BASE:-${VLM_BASE}/Qwen3VL}"

mkdir -p "${OUT_ROOT}"

COMMON_ENV=(
  RUN_PRIVATE="${RUN_PRIVATE:-1}"
  RUN_UPSTREAM="${RUN_UPSTREAM:-0}"
  RUN_SERVER="${RUN_SERVER:-1}"
  CACHE_TYPE_K="${CACHE_TYPE_K:-f16}"
  CACHE_TYPE_V="${CACHE_TYPE_V:-f16}"
  CTX_SIZE="${CTX_SIZE:-4096}"
  CONTEXT_LADDER="${CONTEXT_LADDER:-1024,3072}"
  BENCH_PROMPT="${BENCH_PROMPT:-512}"
  BENCH_GEN="${BENCH_GEN:-128}"
  BENCH_REPS="${BENCH_REPS:-1}"
  THREADS="${THREADS:-8}"
  THREADS_BATCH="${THREADS_BATCH:-8}"
  BATCH_SIZE="${BATCH_SIZE:-1024}"
  UBATCH_SIZE="${UBATCH_SIZE:-512}"
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
)

LLM_HIGH=(
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
  Qwen3.5-35B-A3B-Q4_0.gguf
  Qwen3-30B-A3B-Q4_0.gguf
  LFM2-24B-A2B-Q4_0.gguf
  Qwen3-8B-Q4_K_M.gguf
)

LLM_ALL=(
  Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf
  Qwen3.5-35B-A3B-Q4_0.gguf
  Qwen3-30B-A3B-Q4_0.gguf
  LFM2-24B-A2B-Q4_0.gguf
  Qwen3-8B-Q4_K_M.gguf
  Qwen3-4B-Q4_K_M.gguf
  Qwen3.5-4B-Q4_0.gguf
  SmallThinker-4B-A0.6B-Instruct.Q4_0.gguf
  qwen2.5-7b-instruct-q4_0.gguf
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

VLM_TAR_HIGH=(
  qwen30ba3b-mm-q4_1.tar.gz
  Qwen3.5-4B.tar.gz
)

VLM_TAR_ALL=(
  qwen30ba3b-mm-q4_1.tar.gz
  Qwen3.5-4B.tar.gz
  Qwen3.5-2B.tar.gz
  Qwen3.5-0.8B.tar.gz
  fastvlm-mm-0.5b-q4_1.tar.gz
  mineru2.5-pro-2605-1.2B-original.tar.gz
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_ROOT}/matrix.log"
}

needs_no_think() {
  case "$1" in
    Qwen3*|qwen30b*|qwen3*) return 0 ;;
    *) return 1 ;;
  esac
}

run_llm() {
  local file="$1" port="$2" prefix="" chat_kwargs=""
  if needs_no_think "${file}"; then
    prefix=$'/no_think\n'
    chat_kwargs='{"enable_thinking":false}'
  fi
  log "LLM ${file} port=${port}"
  env "${COMMON_ENV[@]}" \
    MODE=llm \
    MODEL_URL="${LLM_BASE}/${file}" \
    MODEL_DIR=/root/models/spacemit-ai/llm \
    MODEL_FILE="${file}" \
    MODEL_PATH="/root/models/spacemit-ai/llm/${file}" \
    ALIAS="${file}" \
    PORT_BASE="${port}" \
    PROMPT_PREFIX="${prefix}" \
    CHAT_TEMPLATE_KWARGS_JSON="${chat_kwargs}" \
    OUT_DIR="${OUT_ROOT}/llm/${file%.gguf}" \
    bash "${RUNNER}"
}

run_vlm_tar() {
  local file="$1" port="$2" prefix="" chat_kwargs=""
  if needs_no_think "${file}"; then
    prefix=$'/no_think\n'
    chat_kwargs='{"enable_thinking":false}'
  fi
  log "VLM tar ${file} port=${port}"
  env "${COMMON_ENV[@]}" \
    MODE=vlm-tar \
    VLM_TAR_URL="${VLM_BASE}/${file}" \
    VLM_TAR_DIR=/root/models/spacemit-ai/vlm \
    VLM_TAR_FILE="${file}" \
    VLM_EXTRACT_DIR="/root/models/spacemit-ai/vlm/${file%.tar.gz}" \
    ALIAS="${file%.tar.gz}" \
    PORT_BASE="${port}" \
    PROMPT_PREFIX="${prefix}" \
    CHAT_TEMPLATE_KWARGS_JSON="${chat_kwargs}" \
    OUT_DIR="${OUT_ROOT}/vlm-tar/${file%.tar.gz}" \
    bash "${RUNNER}"
}

run_vlm_pair() {
  local name="$1" base="$2" model_file="$3" mmproj_file="$4" port="$5" prefix="" chat_kwargs=""
  if needs_no_think "${model_file}"; then
    prefix=$'/no_think\n'
    chat_kwargs='{"enable_thinking":false}'
  fi
  log "VLM pair ${name} port=${port}"
  env "${COMMON_ENV[@]}" \
    MODE=vlm-pair \
    MODEL_URL="${base}/${model_file}" \
    MODEL_DIR="/root/models/spacemit-ai/vlm/${name}" \
    MODEL_FILE="${model_file}" \
    MODEL_PATH="/root/models/spacemit-ai/vlm/${name}/${model_file}" \
    MMPROJ_URL="${base}/${mmproj_file}" \
    MMPROJ_DIR="/root/models/spacemit-ai/vlm/${name}" \
    MMPROJ_FILE="${mmproj_file}" \
    ALIAS="${model_file}" \
    PORT_BASE="${port}" \
    PROMPT_PREFIX="${prefix}" \
    CHAT_TEMPLATE_KWARGS_JSON="${chat_kwargs}" \
    OUT_DIR="${OUT_ROOT}/vlm-pair/${name}" \
    bash "${RUNNER}"
}

port="${PORT_BASE_START:-18100}"

case "${SCOPE}" in
  high)
    for f in "${LLM_HIGH[@]}"; do run_llm "$f" "$port"; port=$((port + 2)); done
    for f in "${VLM_TAR_HIGH[@]}"; do run_vlm_tar "$f" "$port"; port=$((port + 2)); done
    run_vlm_pair Qwen3VL "${QWEN3VL_BASE}" \
      Qwen3VL-4B-Instruct-Q4_K_M.gguf \
      mmproj-Qwen3VL-4B-Instruct-F16.gguf "$port"; port=$((port + 2))
    ;;
  llm)
    for f in "${LLM_ALL[@]}"; do run_llm "$f" "$port"; port=$((port + 2)); done
    ;;
  vlm)
    for f in "${VLM_TAR_ALL[@]}"; do run_vlm_tar "$f" "$port"; port=$((port + 2)); done
    run_vlm_pair Qwen3VL "${QWEN3VL_BASE}" \
      Qwen3VL-4B-Instruct-Q4_K_M.gguf \
      mmproj-Qwen3VL-4B-Instruct-F16.gguf "$port"; port=$((port + 2))
    run_vlm_pair SmolVLM "${VLM_BASE}" \
      SmolVLM-256M-Instruct-f16.gguf \
      mmproj-SmolVLM-256M-Instruct-Q8_0.gguf "$port"; port=$((port + 2))
    ;;
  all)
    OUT_ROOT="${OUT_ROOT}" SCOPE=llm PORT_BASE_START="${PORT_BASE_START:-18100}" "$0"
    OUT_ROOT="${OUT_ROOT}" SCOPE=vlm PORT_BASE_START="${VLM_PORT_BASE_START:-18300}" "$0"
    ;;
  *)
    echo "unknown SCOPE=${SCOPE}; use high|llm|vlm|all" >&2
    exit 2
    ;;
esac

log "done out=${OUT_ROOT}"
