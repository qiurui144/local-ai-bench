#!/usr/bin/env bash
set -euo pipefail

# Run on the SpacemiT K3 32GB target.
# Focus: highest-spec non-LLM model_zoo entries (embedding/rerank plus package
# inspection for ASR/OCR archives).

MODE="${MODE:-embedding}" # embedding | rerank | tar-inspect | file-inspect
MODEL_URL="${MODEL_URL:-}"
MODEL_DIR="${MODEL_DIR:-/root/models/spacemit-ai/nonllm}"
MODEL_FILE="${MODEL_FILE:-}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
ALIAS="${ALIAS:-${MODEL_FILE}}"
OUT_DIR="${OUT_DIR:-/root/k3_32g_nonllm/${ALIAS%.*}-$(date +%Y%m%d_%H%M%S)}"
PORT="${PORT:-18220}"

THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-8}"
BATCH_SIZE="${BATCH_SIZE:-512}"
UBATCH_SIZE="${UBATCH_SIZE:-256}"
CTX_SIZE="${CTX_SIZE:-512}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-300}"
PRIVATE_BIN_DIR="${PRIVATE_BIN_DIR:-/usr/bin}"
PRIVATE_ENV="${PRIVATE_ENV:-SPACEMIT_DISABLE_TCM=1}"
SERVER_TIMEOUT="${SERVER_TIMEOUT:-300}"
TAR_EXTRACT_DIR="${TAR_EXTRACT_DIR:-${MODEL_DIR}/${MODEL_FILE%.tar.gz}}"
EXTRACT_TAR="${EXTRACT_TAR:-1}"

mkdir -p "${OUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_DIR}/run.log"
}

download_file() {
  local url="$1" dir="$2" file="$3"
  mkdir -p "${dir}"
  local path="${dir}/${file}"
  if [[ -n "${url}" && ! -s "${path}" ]]; then
    log "downloading ${url}"
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -c -x 4 -s 4 -k 4M --file-allocation=none -d "${dir}" -o "${file}" "${url}" \
        2>&1 | tee -a "${OUT_DIR}/download.log"
    else
      wget -c --progress=dot:giga -O "${path}" "${url}" 2>&1 | tee -a "${OUT_DIR}/download.log"
    fi
  fi
  stat -c '{"path":"%n","bytes":%s}' "${path}" | tee "${OUT_DIR}/model-file.json"
  ls -lh "${path}" | tee "${OUT_DIR}/model-file.txt"
}

wait_for_server() {
  local pid="$1" port="$2"
  for _ in $(seq 1 "${SERVER_TIMEOUT}"); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${OUT_DIR}/server.models.json"; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 1
  done
  return 1
}

start_server() {
  local mode_flag="$1"
  local pooling="${2:-}"
  local out="${OUT_DIR}/server"
  mkdir -p "${out}"
  pkill -f "llama-server.*--port ${PORT}" >/dev/null 2>&1 || true
  local extra=("${mode_flag}")
  if [[ -n "${pooling}" ]]; then
    extra+=(--pooling "${pooling}")
  fi
  env ${PRIVATE_ENV} "${PRIVATE_BIN_DIR}/llama-server" \
    -m "${MODEL_PATH}" "${extra[@]}" \
    --alias "${ALIAS}" --host 127.0.0.1 --port "${PORT}" \
    -c "${CTX_SIZE}" -t "${THREADS}" -tb "${THREADS_BATCH}" -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}" \
    --no-webui --no-warmup --cache-ram 0 \
    --log-file "${out}/llama-server.log" \
    > "${out}/llama-server.stdout.log" 2> "${out}/llama-server.stderr.log" &
  local pid=$!
  echo "${pid}" > "${out}/llama-server.pid"
  if ! wait_for_server "${pid}" "${PORT}"; then
    log "server did not become ready"
    echo 1 > "${out}/server-ready.rc"
    return 1
  fi
  echo 0 > "${out}/server-ready.rc"
  return 0
}

stop_server() {
  local pid_file="${OUT_DIR}/server/llama-server.pid"
  if [[ -f "${pid_file}" ]]; then
    kill "$(cat "${pid_file}")" >/dev/null 2>&1 || true
    wait "$(cat "${pid_file}")" >/dev/null 2>&1 || true
  fi
}

run_embedding_probe() {
  BASE_URL="http://127.0.0.1:${PORT}/v1" MODEL_ALIAS="${ALIAS}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
  python3 - <<'PY' > "${OUT_DIR}/embedding-probe.jsonl" 2> "${OUT_DIR}/embedding-probe.stderr.log"
import json, math, os, statistics, time
import httpx

base = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL_ALIAS"]
timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))
client = httpx.Client(timeout=timeout)

corpus = [
    ("d0", "北京是中国的首都，也是政治和文化中心。"),
    ("d1", "上海是重要的金融中心，拥有繁忙的港口。"),
    ("d2", "Python 常用于数据分析、自动化和机器学习。"),
    ("d3", "端侧 AI 推理关注吞吐、首 token 延迟、内存和稳定性。"),
    ("d4", "蜂窝网络和 Wi-Fi 都可以提供无线连接。"),
]
queries = [
    ("中国首都是哪里", "d0"),
    ("哪座城市是金融中心", "d1"),
    ("Python 可以做什么", "d2"),
    ("边缘推理要看哪些性能指标", "d3"),
    ("无线连接有哪些方式", "d4"),
]

def embed(texts):
    t0 = time.perf_counter()
    r = client.post(f"{base}/embeddings", json={"model": model, "input": texts})
    elapsed = time.perf_counter() - t0
    item = {"status_code": r.status_code, "elapsed_s": round(elapsed, 4), "n_inputs": len(texts)}
    try:
        body = r.json()
    except Exception as exc:
        item["error"] = f"json: {exc}"
        item["body_prefix"] = r.text[:1000]
        print(json.dumps({"case": "embed", **item}, ensure_ascii=False), flush=True)
        return None, item
    vecs = [x.get("embedding", []) for x in body.get("data", [])]
    item["dim"] = len(vecs[0]) if vecs else 0
    item["usage"] = body.get("usage")
    print(json.dumps({"case": "embed", **item}, ensure_ascii=False), flush=True)
    return vecs, item

doc_vecs, doc_item = embed([d[1] for d in corpus])
query_lat = []
hits = 0
rrs = []
ndcgs = []

def cosine(a, b):
    dot = sum(x*y for x, y in zip(a, b))
    na = math.sqrt(sum(x*x for x in a))
    nb = math.sqrt(sum(y*y for y in b))
    return dot / (na * nb) if na and nb else -1.0

if doc_vecs:
    for query, expected in queries:
        vecs, item = embed([query])
        query_lat.append(item["elapsed_s"] * 1000)
        if not vecs:
            continue
        scores = [(corpus[i][0], cosine(vecs[0], doc_vecs[i])) for i in range(len(corpus))]
        scores.sort(key=lambda x: x[1], reverse=True)
        rank = [doc_id for doc_id, _ in scores].index(expected) + 1
        hits += int(rank == 1)
        rrs.append(1.0 / rank)
        ndcgs.append(1.0 / math.log2(rank + 1))
        print(json.dumps({"case": "retrieval", "query": query, "expected": expected, "rank": rank, "top": scores[:3]}, ensure_ascii=False), flush=True)

summary = {
    "case": "summary",
    "hit_at_1": hits / len(queries),
    "mrr": sum(rrs) / len(rrs) if rrs else 0,
    "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
    "latency_p50_ms": statistics.median(query_lat) if query_lat else None,
    "latency_p95_ms": sorted(query_lat)[int(len(query_lat)*0.95)-1] if query_lat else None,
}
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY
  echo "$?" > "${OUT_DIR}/embedding-probe.rc"
}

run_rerank_probe() {
  BASE_URL="http://127.0.0.1:${PORT}/v1" MODEL_ALIAS="${ALIAS}" REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
  python3 - <<'PY' > "${OUT_DIR}/rerank-probe.jsonl" 2> "${OUT_DIR}/rerank-probe.stderr.log"
import json, math, os, statistics, time
import httpx

base = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL_ALIAS"]
timeout = float(os.environ.get("REQUEST_TIMEOUT", "300"))
client = httpx.Client(timeout=timeout)

cases = [
    ("中国首都是哪里", ["上海是金融中心。", "北京是中国的首都。", "Python 是编程语言。"], 1),
    ("边缘推理关注哪些指标", ["吞吐和首 token 延迟是关键指标。", "今天适合散步。", "咖啡需要研磨。"], 0),
    ("Python 用于什么", ["北京有很多博物馆。", "Python 可用于自动化和机器学习。", "无线网络很常见。"], 1),
]
lat = []
hits = 0
rrs = []
ndcgs = []
for query, docs, expected in cases:
    payload = {"model": model, "query": query, "documents": docs, "return_documents": False}
    t0 = time.perf_counter()
    r = client.post(f"{base}/rerank", json=payload)
    elapsed = time.perf_counter() - t0
    item = {"case": "rerank", "query": query, "status_code": r.status_code, "elapsed_s": round(elapsed, 4)}
    try:
        body = r.json()
        item["body"] = body
        results = body.get("results", [])
        ranked = [x.get("index") for x in results]
        rank = ranked.index(expected) + 1 if expected in ranked else 999
        item["rank"] = rank
        hits += int(rank == 1)
        rrs.append(1.0 / rank)
        ndcgs.append(1.0 / math.log2(rank + 1))
    except Exception as exc:
        item["error"] = str(exc)
        item["body_prefix"] = r.text[:1000]
    lat.append(elapsed * 1000)
    print(json.dumps(item, ensure_ascii=False), flush=True)
summary = {
    "case": "summary",
    "hit_at_1": hits / len(cases),
    "mrr": sum(rrs) / len(rrs) if rrs else 0,
    "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
    "latency_p50_ms": statistics.median(lat),
    "latency_p95_ms": sorted(lat)[int(len(lat)*0.95)-1],
}
print(json.dumps(summary, ensure_ascii=False), flush=True)
PY
  echo "$?" > "${OUT_DIR}/rerank-probe.rc"
}

main_embedding() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}"
  if start_server "--embedding" "mean"; then
    run_embedding_probe
  fi
  stop_server
}

main_rerank() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}"
  if start_server "--reranking" "rank"; then
    run_rerank_probe
  fi
  stop_server
}

main_tar_inspect() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}"
  mkdir -p "${TAR_EXTRACT_DIR}"
  tar -tzf "${MODEL_PATH}" > "${OUT_DIR}/tar-list.txt"
  if [[ "${EXTRACT_TAR}" == "1" ]]; then
    tar -xzf "${MODEL_PATH}" -C "${TAR_EXTRACT_DIR}" --skip-old-files
  fi
  find "${TAR_EXTRACT_DIR}" -maxdepth 5 -type f -printf '%s\t%p\n' | sort -nr > "${OUT_DIR}/extracted-files.tsv" || true
  {
    echo "onnx:"
    find "${TAR_EXTRACT_DIR}" -maxdepth 5 -type f -iname '*.onnx' -print | sort || true
    echo "gguf:"
    find "${TAR_EXTRACT_DIR}" -maxdepth 5 -type f -iname '*.gguf' -print | sort || true
    echo "json:"
    find "${TAR_EXTRACT_DIR}" -maxdepth 5 -type f -iname '*.json' -print | sort || true
  } > "${OUT_DIR}/package-summary.txt"
}

main_file_inspect() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}"
  {
    echo "stat:"
    stat "${MODEL_PATH}" || true
    echo
    echo "sha256:"
    sha256sum "${MODEL_PATH}" || true
    echo
    echo "file:"
    file "${MODEL_PATH}" || true
    echo
    echo "strings-head:"
    strings -a "${MODEL_PATH}" | head -80 || true
  } > "${OUT_DIR}/file-inspect.txt" 2>&1
}

log "mode=${MODE} out=${OUT_DIR}"
case "${MODE}" in
  embedding) main_embedding ;;
  rerank) main_rerank ;;
  tar-inspect) main_tar_inspect ;;
  file-inspect) main_file_inspect ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac
log "done"
