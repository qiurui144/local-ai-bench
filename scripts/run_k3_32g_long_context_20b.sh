#!/usr/bin/env bash
set -euo pipefail

# Run locally. Starts one 20B+ K3 model at a time on the 32GB board and runs
# the long-context edge subset from this repository against its OpenAI API.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

k3_require_target_env
REMOTE_OUT_ROOT="${REMOTE_OUT_ROOT:-/root/k3_32g_long_context}"
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-output/reports/k3-riscv-32g/long-context-20b-$(date +%Y%m%d_%H%M%S)}"
REMOTE_PORT_BASE="${REMOTE_PORT_BASE:-18500}"
CTX_SIZE="${CTX_SIZE:-4096}"
MAX_INPUT_TOKENS="${MAX_INPUT_TOKENS:-3072}"
THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-8}"
SERVER_PARALLEL="${SERVER_PARALLEL:-1}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
PROMPT_CACHE_RAM="${PROMPT_CACHE_RAM:-8192}"
TIMEOUT_S="${TIMEOUT_S:-2400}"
LLAMA_SERVER_TIMEOUT_S="${LLAMA_SERVER_TIMEOUT_S:-2400}"
CONTEXT_LENGTHS="${CONTEXT_LENGTHS:-1024,3072}"
DEPTH_PERCENTS="${DEPTH_PERCENTS:-10,50,90}"
LONGBENCH_DATASETS="${LONGBENCH_DATASETS:-passage_retrieval_en,passage_count}"
LEVAL_TASKS="${LEVAL_TASKS:-quality,coursera}"
SAMPLES_PER_DATASET="${SAMPLES_PER_DATASET:-1}"
SAMPLES_PER_TASK="${SAMPLES_PER_TASK:-1}"
QUESTIONS_PER_DOCUMENT="${QUESTIONS_PER_DOCUMENT:-1}"
AIRPLANE_MANUALS="${AIRPLANE_MANUALS:-1}"
AIRPLANE_MANUAL_SCOPE="${AIRPLANE_MANUAL_SCOPE:-core}" # core | broad | all
AIRPLANE_MANUAL_PATHS="${AIRPLANE_MANUAL_PATHS:-}"
AIRPLANE_MANUAL_MAX_FILES="${AIRPLANE_MANUAL_MAX_FILES:-0}"
AIRPLANE_MANUAL_CASE_LIMIT="${AIRPLANE_MANUAL_CASE_LIMIT:-12}"
AIRPLANE_MANUAL_CONTEXT_TOKENS="${AIRPLANE_MANUAL_CONTEXT_TOKENS:-${MAX_INPUT_TOKENS}}"
AIRPLANE_MANUAL_PROMPT_BUDGET_SAFETY="${AIRPLANE_MANUAL_PROMPT_BUDGET_SAFETY:-0.70}"
SKIP_SUITES="${SKIP_SUITES:-}"
ONLY_MODELS="${ONLY_MODELS:-}" # comma-separated aliases; empty = all known 32G 20B+ models

SSH_OPTS=(
  -o StrictHostKeyChecking=no
  -o UserKnownHostsFile=/dev/null
  -o ConnectTimeout=10
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=2
)
if [[ -n "${SSHPASS:-}" ]]; then
  SSH_OPTS+=(
    -o PubkeyAuthentication=no
    -o PreferredAuthentications=password,keyboard-interactive
  )
else
  SSH_OPTS+=(
    -o BatchMode=yes
    -o IdentitiesOnly=yes
  )
fi
SSH_CMD=(ssh "${SSH_OPTS[@]}")
SCP_CMD=(scp "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
fi

remote() {
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "$@" < /dev/null
}

contains_model() {
  local alias="$1"
  [[ -z "${ONLY_MODELS}" ]] && return 0
  IFS=',' read -ra parts <<< "${ONLY_MODELS}"
  for p in "${parts[@]}"; do
    [[ "${p}" == "${alias}" ]] && return 0
  done
  return 1
}

wait_ready() {
  local port="$1"
  for _ in $(seq 1 240); do
    if curl -fsS "http://${K3_HOST}:${port}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_one() {
  local alias="$1" bin_dir="$2" model_path="$3" port="$4" extra_prefix="$5" chat_kwargs="$6" smt_config_dir="${7:-}"
  contains_model "${alias}" || return 0
  local local_out="${LOCAL_OUT_ROOT}/${alias}"
  local remote_out="${REMOTE_OUT_ROOT}/${alias}-$(date +%Y%m%d_%H%M%S)"
  mkdir -p "${local_out}"
  if ! remote "mkdir -p '${remote_out}'"; then
    echo "ssh/setup failed for ${alias}" | tee "${local_out}/FAILED"
    return 0
  fi
  if ! remote "test -s '${model_path}'"; then
    echo "missing remote model: ${model_path}" | tee "${local_out}/SKIPPED"
    return 0
  fi
  remote "pkill -f '[l]lama-server.*--port ${port}' >/dev/null 2>&1 || true" || true
  remote "spacemit-tcm-smi -c > '${remote_out}/tcm-release.txt' 2>&1 || true" || true
  local server_extra=""
  if [[ -n "${smt_config_dir}" ]]; then
    server_extra="--media-backend smt --smt-config-dir '${smt_config_dir}'"
  fi
  if ! remote "cd /root && setsid -f env SPACEMIT_DISABLE_TCM=1 '${bin_dir}/llama-server' \
    -m '${model_path}' ${server_extra} --alias '${alias}' --host 0.0.0.0 --port '${port}' \
    -c '${CTX_SIZE}' -np '${SERVER_PARALLEL}' -t '${THREADS}' -tb '${THREADS_BATCH}' -b '${BATCH_SIZE}' -ub '${UBATCH_SIZE}' \
    -ctk '${CACHE_TYPE_K}' -ctv '${CACHE_TYPE_V}' --cache-ram '${PROMPT_CACHE_RAM}' --timeout '${LLAMA_SERVER_TIMEOUT_S}' --no-webui \
    --log-file '${remote_out}/llama-server.log' \
    > '${remote_out}/llama-server.stdout.log' 2> '${remote_out}/llama-server.stderr.log' < /dev/null; \
    sleep 1; pgrep -f '[l]lama-server.*--port ${port}' | tail -1 > '${remote_out}/llama-server.pid'"; then
    echo "server launch failed for ${alias}" | tee "${local_out}/FAILED"
    return 0
  fi
  if ! wait_ready "${port}"; then
    remote "cat '${remote_out}/llama-server.pid' 2>/dev/null | xargs -r kill >/dev/null 2>&1 || true" || true
    "${SCP_CMD[@]}" -r "${K3_USER}@${K3_HOST}:${remote_out}" "${local_out}/remote" || true
    echo "server not ready for ${alias}" >&2
    echo "server not ready for ${alias}" > "${local_out}/FAILED"
    return 0
  fi
  set +e
  PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 "${REPO_ROOT}/scripts/run_long_context_suite.py" \
    --base-url "http://${K3_HOST}:${port}/v1" \
    --model "${alias}" \
    --model-name "${alias}" \
    --out-dir "${local_out}" \
    --prompt-prefix "${extra_prefix}" \
    --chat-template-kwargs "${chat_kwargs}" \
    --max-input-tokens "${MAX_INPUT_TOKENS}" \
    --timeout-s "${TIMEOUT_S}" \
    --context-lengths "${CONTEXT_LENGTHS}" \
    --depth-percents "${DEPTH_PERCENTS}" \
    --longbench-datasets "${LONGBENCH_DATASETS}" \
    --leval-tasks "${LEVAL_TASKS}" \
    --samples-per-dataset "${SAMPLES_PER_DATASET}" \
    --samples-per-task "${SAMPLES_PER_TASK}" \
    --questions-per-document "${QUESTIONS_PER_DOCUMENT}" \
    --airplane-manual-case-limit "${AIRPLANE_MANUAL_CASE_LIMIT}" \
    --airplane-manual-context-tokens "${AIRPLANE_MANUAL_CONTEXT_TOKENS}" \
    --airplane-manual-prompt-budget-safety "${AIRPLANE_MANUAL_PROMPT_BUDGET_SAFETY}" \
    --skip-suites "${SKIP_SUITES}" \
    > "${local_out}/runner.stdout.log" 2> "${local_out}/runner.stderr.log"
  local rc=$?
  set -e
  echo "${rc}" > "${local_out}/runner.rc"
  remote "cat '${remote_out}/llama-server.pid' 2>/dev/null | xargs -r kill >/dev/null 2>&1 || true" || true
  remote "spacemit-tcm-smi > '${remote_out}/tcm-after.txt' 2>&1 || true" || true
  "${SCP_CMD[@]}" -r "${K3_USER}@${K3_HOST}:${remote_out}" "${local_out}/remote" || true
  return 0
}

mkdir -p "${LOCAL_OUT_ROOT}"
cache_args=()
if [[ "${AIRPLANE_MANUALS}" == "1" ]]; then
  cache_args+=(--airplane-manuals --airplane-scope "${AIRPLANE_MANUAL_SCOPE}" --airplane-max-files "${AIRPLANE_MANUAL_MAX_FILES}")
  if [[ -n "${AIRPLANE_MANUAL_PATHS}" ]]; then
    cache_args+=(--airplane-paths "${AIRPLANE_MANUAL_PATHS}")
  fi
fi
PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}" python3 "${REPO_ROOT}/scripts/cache_long_context_suites.py" "${cache_args[@]}" \
  | tee "${LOCAL_OUT_ROOT}/cache-long-context.log"

run_one "Qwen3-30B-A3B-Q4_0" "/usr/bin" "/root/models/spacemit-ai/llm/Qwen3-30B-A3B-Q4_0.gguf" "$((REMOTE_PORT_BASE + 0))" $'/no_think\n' '{"enable_thinking":false}'
run_one "Qwen3.5-35B-A3B-Q4_0" "/usr/bin" "/root/models/spacemit-ai/llm/Qwen3.5-35B-A3B-Q4_0.gguf" "$((REMOTE_PORT_BASE + 1))" $'/no_think\n' '{"enable_thinking":false}'
run_one "Qwen3.6-35B-A3B-UD-Q4_K_XL-upstream" "/root/src/llama.cpp/build-k3/bin" "/root/models/spacemit-ai/llm/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf" "$((REMOTE_PORT_BASE + 2))" $'/no_think\n' '{"enable_thinking":false}'
run_one "qwen30ba3b-mm-q4_1" "/usr/bin" "/root/models/spacemit-ai/vlm/qwen30ba3b-mm-q4_1/qwen3vl-30b-text-q4_1.gguf" "$((REMOTE_PORT_BASE + 3))" $'/no_think\n' '{"enable_thinking":false}' "/root/models/spacemit-ai/vlm/qwen30ba3b-mm-q4_1"
run_one "LFM2-24B-A2B-Q4_0" "/usr/bin" "/root/models/spacemit-ai/llm/LFM2-24B-A2B-Q4_0.gguf" "$((REMOTE_PORT_BASE + 4))" "" '{}'

echo "long-context output: ${LOCAL_OUT_ROOT}"
