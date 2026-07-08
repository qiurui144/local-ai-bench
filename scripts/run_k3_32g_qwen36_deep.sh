#!/usr/bin/env bash
set -euo pipefail

# Run on the K3 32GB target from the repo root.
# Focus: SpacemiT model_zoo Qwen3.6-35B-A3B-UD-Q4_K_XL prefill/decode/context.

MODEL_URL="${MODEL_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/llm/Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
MODEL_DIR="${MODEL_DIR:-/root/models/spacemit-ai/llm}"
MODEL_FILE="${MODEL_FILE:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
MODEL_PATH="${MODEL_DIR}/${MODEL_FILE}"
EXPECTED_BYTES="${EXPECTED_BYTES:-22853663008}"
ALIAS="${ALIAS:-Qwen3.6-35B-A3B-UD-Q4_K_XL.gguf}"
PORT="${PORT:-18080}"
CTX_SIZE="${CTX_SIZE:-32768}"
THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
BENCH_REPS="${BENCH_REPS:-3}"
BENCH_PROMPTS="${BENCH_PROMPTS:-512,2048,4096,8192,16384,32768}"
BENCH_GEN="${BENCH_GEN:-128}"
RUN_QUALITY="${RUN_QUALITY:-0}"
RUN_CONDITIONED="${RUN_CONDITIONED:-1}"

DEFAULT_LLAMA_BIN_DIR="/root/src/llama.cpp/build-k3/bin"
if [[ -x "${DEFAULT_LLAMA_BIN_DIR}/llama-server" ]]; then
  LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-${DEFAULT_LLAMA_BIN_DIR}}"
else
  LLAMA_BIN_DIR="${LLAMA_BIN_DIR:-}"
fi
LLAMA_BENCH="${LLAMA_BENCH:-${LLAMA_BIN_DIR:+${LLAMA_BIN_DIR}/}llama-bench}"
LLAMA_SERVER="${LLAMA_SERVER:-${LLAMA_BIN_DIR:+${LLAMA_BIN_DIR}/}llama-server}"

OUT_DIR="${OUT_DIR:-output/reports/k3-riscv-32g/qwen36-$(date +%Y%m%d_%H%M%S)}"
mkdir -p "${MODEL_DIR}" "${OUT_DIR}"

echo "[1/5] ensuring model exists: ${MODEL_PATH}"
if [[ ! -s "${MODEL_PATH}" ]] || [[ "$(stat -c '%s' "${MODEL_PATH}" 2>/dev/null || echo 0)" -lt "${EXPECTED_BYTES}" ]]; then
  if command -v aria2c >/dev/null 2>&1; then
    aria2c -c -x 4 -s 4 -k 4M --file-allocation=none -d "${MODEL_DIR}" -o "${MODEL_FILE}" "${MODEL_URL}"
  else
    wget -c --progress=dot:giga -O "${MODEL_PATH}" "${MODEL_URL}"
  fi
fi
actual_bytes="$(stat -c '%s' "${MODEL_PATH}")"
if [[ "${actual_bytes}" -ne "${EXPECTED_BYTES}" ]]; then
  echo "model size mismatch: got ${actual_bytes}, expected ${EXPECTED_BYTES}" >&2
  exit 1
fi
ls -lh "${MODEL_PATH}" | tee "${OUT_DIR}/model-file.txt"

echo "[2/5] native llama-bench PP and TG sweeps"
{
  echo "LLAMA_BENCH=${LLAMA_BENCH}"
  echo "LLAMA_SERVER=${LLAMA_SERVER}"
  "${LLAMA_BENCH}" --help | head -40 || true
  "${LLAMA_SERVER}" --version || true
} > "${OUT_DIR}/runtime-info.txt" 2>&1

set +e
"${LLAMA_BENCH}" \
  -m "${MODEL_PATH}" \
  -t "${THREADS}" \
  -p "${BENCH_PROMPTS}" \
  -n 0 \
  -ctk "${CACHE_TYPE_K}" \
  -ctv "${CACHE_TYPE_V}" \
  -r "${BENCH_REPS}" \
  -o jsonl 2>&1 | tee "${OUT_DIR}/llama-bench-prefill.jsonl"
prefill_rc="${PIPESTATUS[0]}"
echo "${prefill_rc}" > "${OUT_DIR}/llama-bench-prefill.rc"

"${LLAMA_BENCH}" \
  -m "${MODEL_PATH}" \
  -t "${THREADS}" \
  -p 0 \
  -n "${BENCH_GEN}" \
  -ctk "${CACHE_TYPE_K}" \
  -ctv "${CACHE_TYPE_V}" \
  -r "${BENCH_REPS}" \
  -o jsonl 2>&1 | tee "${OUT_DIR}/llama-bench-decode.jsonl"
decode_rc="${PIPESTATUS[0]}"
echo "${decode_rc}" > "${OUT_DIR}/llama-bench-decode.rc"
set -e

if [[ "${prefill_rc}" -ne 0 || "${decode_rc}" -ne 0 ]]; then
  echo "llama-bench returned non-zero rc: prefill=${prefill_rc}, decode=${decode_rc}; continuing with server/API tests" \
    | tee "${OUT_DIR}/llama-bench-warning.txt"
fi

echo "[3/5] starting llama-server on port ${PORT}, ctx=${CTX_SIZE}, KV=${CACHE_TYPE_K}/${CACHE_TYPE_V}"
pkill -f "llama-server.*--port ${PORT}" >/dev/null 2>&1 || true
pkill -f "llama-server.*:${PORT}" >/dev/null 2>&1 || true

"${LLAMA_SERVER}" \
  -m "${MODEL_PATH}" \
  --alias "${ALIAS}" \
  --host 0.0.0.0 \
  --port "${PORT}" \
  -c "${CTX_SIZE}" \
  -t "${THREADS}" \
  -tb "${THREADS_BATCH}" \
  -b "${BATCH_SIZE}" \
  -ub "${UBATCH_SIZE}" \
  -ctk "${CACHE_TYPE_K}" \
  -ctv "${CACHE_TYPE_V}" \
  --no-webui \
  --log-file "${OUT_DIR}/llama-server.log" \
  >"${OUT_DIR}/llama-server.stdout.log" 2>"${OUT_DIR}/llama-server.stderr.log" &
SERVER_PID=$!
echo "${SERVER_PID}" > "${OUT_DIR}/llama-server.pid"

echo "[4/5] waiting for OpenAI-compatible endpoint"
for _ in $(seq 1 180); do
  if curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "${OUT_DIR}/models.json"; then
    break
  fi
  if ! kill -0 "${SERVER_PID}" >/dev/null 2>&1; then
    echo "llama-server exited during startup" >&2
    tail -120 "${OUT_DIR}/llama-server.stderr.log" >&2 || true
    exit 1
  fi
  sleep 5
done
curl -fsS "http://127.0.0.1:${PORT}/v1/models" > "${OUT_DIR}/models.json"

echo "[5/5] running harness focus dimensions"
export K3_32G_QWEN36_BASE_URL="http://127.0.0.1:${PORT}/v1"

SKIP_DIMS="accuracy,stability,concurrency,scenarios,embedding,rerank,asr,ocr,conversation_drift"
if [[ "${RUN_QUALITY}" != "1" ]]; then
  SKIP_DIMS="${SKIP_DIMS},translation,general_ability"
fi
if [[ "${RUN_CONDITIONED}" != "1" ]]; then
  SKIP_DIMS="${SKIP_DIMS},conditioned"
fi

python3 run_benchmark.py \
  --model qwen3.6-35b-a3b-k3-32g-riscv \
  --target k3-riscv-32g \
  --local-only \
  --skip "${SKIP_DIMS}" | tee "${OUT_DIR}/run_benchmark.log"

echo "reports: ${OUT_DIR}"
