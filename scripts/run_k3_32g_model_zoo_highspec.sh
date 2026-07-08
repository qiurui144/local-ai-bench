#!/usr/bin/env bash
set -euo pipefail

# Run on the SpacemiT K3 32GB target.
# Focus: high-spec SpacemiT model_zoo LLM/VLM GGUF packages.

MODE="${MODE:-llm}" # llm | vlm-tar | vlm-pair
MODEL_URL="${MODEL_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/llm/Qwen3-30B-A3B-Q4_0.gguf}"
MODEL_DIR="${MODEL_DIR:-/root/models/spacemit-ai/llm}"
MODEL_FILE="${MODEL_FILE:-Qwen3-30B-A3B-Q4_0.gguf}"
MODEL_PATH="${MODEL_PATH:-${MODEL_DIR}/${MODEL_FILE}}"
EXPECTED_MIN_BYTES="${EXPECTED_MIN_BYTES:-1}"
ALIAS="${ALIAS:-${MODEL_FILE}}"
OUT_DIR="${OUT_DIR:-/root/k3_32g_highspec/${ALIAS%.*}-$(date +%Y%m%d_%H%M%S)}"

THREADS="${THREADS:-8}"
THREADS_BATCH="${THREADS_BATCH:-8}"
BATCH_SIZE="${BATCH_SIZE:-1024}"
UBATCH_SIZE="${UBATCH_SIZE:-512}"
CACHE_TYPE_K="${CACHE_TYPE_K:-q8_0}"
CACHE_TYPE_V="${CACHE_TYPE_V:-q8_0}"
BENCH_PROMPT="${BENCH_PROMPT:-512}"
BENCH_GEN="${BENCH_GEN:-128}"
BENCH_REPS="${BENCH_REPS:-1}"
BENCH_TIMEOUT="${BENCH_TIMEOUT:-1800}"
SMOKE_TIMEOUT="${SMOKE_TIMEOUT:-300}"
SMOKE_LOG_LIMIT_BYTES="${SMOKE_LOG_LIMIT_BYTES:-1048576}"
RUN_PRIVATE="${RUN_PRIVATE:-1}"
RUN_UPSTREAM="${RUN_UPSTREAM:-1}"
RUN_SERVER="${RUN_SERVER:-1}"
RUN_SMOKE="${RUN_SMOKE:-1}"
RUN_BENCH="${RUN_BENCH:-1}"
RUN_OFFICIAL_MODELZOO_BENCH="${RUN_OFFICIAL_MODELZOO_BENCH:-0}"
PORT_BASE="${PORT_BASE:-18080}"
CTX_SIZE="${CTX_SIZE:-32768}"
CONTEXT_LADDER="${CONTEXT_LADDER-1024,4096,8192}"
REQUEST_TIMEOUT="${REQUEST_TIMEOUT:-900}"
PROMPT_PREFIX="${PROMPT_PREFIX:-}"
CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON:-}"
SERVER_EXTRA_ARGS="${SERVER_EXTRA_ARGS:-}"

DEFAULT_PRIVATE_BIN_DIR="/opt/spacemit-runtime/llama.cpp/bin"
if [[ -x "${DEFAULT_PRIVATE_BIN_DIR}/llama-server" ]]; then
  PRIVATE_BIN_DIR="${PRIVATE_BIN_DIR:-${DEFAULT_PRIVATE_BIN_DIR}}"
else
  PRIVATE_BIN_DIR="${PRIVATE_BIN_DIR:-/usr/bin}"
fi
UPSTREAM_BIN_DIR="${UPSTREAM_BIN_DIR:-/root/src/llama.cpp/build-k3/bin}"
PRIVATE_ENV="${PRIVATE_ENV-SPACEMIT_DISABLE_TCM=1}"
FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-0}"

VLM_TAR_URL="${VLM_TAR_URL:-https://archive.spacemit.com/spacemit-ai/model_zoo/vlm/qwen30ba3b-mm-q4_1.tar.gz}"
VLM_TAR_DIR="${VLM_TAR_DIR:-/root/models/spacemit-ai/vlm}"
VLM_TAR_FILE="${VLM_TAR_FILE:-qwen30ba3b-mm-q4_1.tar.gz}"
VLM_EXTRACT_DIR="${VLM_EXTRACT_DIR:-${VLM_TAR_DIR}/qwen30ba3b-mm-q4_1}"
EXTRACT_VLM="${EXTRACT_VLM:-1}"
VLM_IMAGE_PATH="${VLM_IMAGE_PATH:-}"
VLM_PROMPT="${VLM_PROMPT:-请识别图片中的主要文字和关键信息，只输出简短 JSON。}"
VLM_CASES_JSONL="${VLM_CASES_JSONL:-}"
VLM_MAX_CASES="${VLM_MAX_CASES:-0}"
VLM_CASE_IDS="${VLM_CASE_IDS:-}"
VLM_DOC_MAX_TOKENS="${VLM_DOC_MAX_TOKENS:-192}"
MMPROJ_URL="${MMPROJ_URL:-}"
MMPROJ_DIR="${MMPROJ_DIR:-${MODEL_DIR}}"
MMPROJ_FILE="${MMPROJ_FILE:-}"
MMPROJ_PATH="${MMPROJ_PATH:-${MMPROJ_DIR}/${MMPROJ_FILE}}"

mkdir -p "${OUT_DIR}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_DIR}/run.log"
}

download_file() {
  local url="$1" dir="$2" file="$3" min_bytes="$4"
  mkdir -p "${dir}"
  local path="${dir}/${file}"
  if [[ ! -s "${path}" ]] || [[ "$(stat -c '%s' "${path}" 2>/dev/null || echo 0)" -lt "${min_bytes}" ]]; then
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

run_cmd_capture() {
  local name="$1"
  shift
  set +e
  "$@" >"${name}.stdout.log" 2>"${name}.stderr.log"
  local rc=$?
  set -e
  echo "${rc}" > "${name}.rc"
  return 0
}

compact_large_log() {
  local path="$1" limit="${2:-${SMOKE_LOG_LIMIT_BYTES}}"
  if [[ -s "${path}" ]] && [[ "$(stat -c '%s' "${path}" 2>/dev/null || echo 0)" -gt "${limit}" ]]; then
    stat -c '%s' "${path}" > "${path%.log}.size.txt"
    tail -200 "${path}" > "${path%.log}.tail.log"
    rm -f "${path}"
  fi
}

capture_tcm_state() {
  local label="$1"
  if command -v spacemit-tcm-smi >/dev/null 2>&1; then
    spacemit-tcm-smi > "${label}" 2>&1 || true
  else
    echo "spacemit-tcm-smi not found" > "${label}"
  fi
}

release_tcm_if_requested() {
  local out="$1"
  capture_tcm_state "${out}/tcm-before.txt"
  if [[ "${FORCE_TCM_RELEASE}" == "1" ]] && command -v spacemit-tcm-smi >/dev/null 2>&1; then
    spacemit-tcm-smi -c > "${out}/tcm-release.txt" 2>&1 || true
    capture_tcm_state "${out}/tcm-after-release.txt"
  fi
}

run_api_probe() {
  local out_prefix="$1" port="$2" model_alias="$3" is_vlm="$4"
  BASE_URL="http://127.0.0.1:${port}/v1" \
  MODEL_ALIAS="${model_alias}" \
  CONTEXT_LADDER="${CONTEXT_LADDER}" \
  REQUEST_TIMEOUT="${REQUEST_TIMEOUT}" \
  VLM_IMAGE_PATH="${VLM_IMAGE_PATH}" \
  VLM_PROMPT="${VLM_PROMPT}" \
  VLM_CASES_JSONL="${VLM_CASES_JSONL}" \
  VLM_MAX_CASES="${VLM_MAX_CASES}" \
  VLM_CASE_IDS="${VLM_CASE_IDS}" \
  VLM_DOC_MAX_TOKENS="${VLM_DOC_MAX_TOKENS}" \
  PROMPT_PREFIX="${PROMPT_PREFIX}" \
  CHAT_TEMPLATE_KWARGS_JSON="${CHAT_TEMPLATE_KWARGS_JSON}" \
  IS_VLM="${is_vlm}" \
  python3 - <<'PY' > "${out_prefix}.stdout.log" 2> "${out_prefix}.stderr.log"
import base64
import json
import os
import re
import sys
import time
from decimal import Decimal, InvalidOperation
from pathlib import Path

import httpx

base_url = os.environ["BASE_URL"].rstrip("/")
model = os.environ["MODEL_ALIAS"]
timeout = float(os.environ.get("REQUEST_TIMEOUT", "900"))
ladder = [int(x) for x in os.environ.get("CONTEXT_LADDER", "").split(",") if x.strip()]
is_vlm = os.environ.get("IS_VLM") == "1"
run_visual_probe = os.environ.get("RUN_VISUAL_PROBE") == "1"
run_vlm_doc_probe = os.environ.get("RUN_VLM_DOC_PROBE", "1") == "1"
vlm_cases_jsonl = os.environ.get("VLM_CASES_JSONL", "")
vlm_max_cases = int(os.environ.get("VLM_MAX_CASES", "0") or 0)
vlm_case_ids = {x.strip() for x in os.environ.get("VLM_CASE_IDS", "").split(",") if x.strip()}
vlm_doc_max_tokens = int(os.environ.get("VLM_DOC_MAX_TOKENS", "192") or 192)
prompt_prefix = os.environ.get("PROMPT_PREFIX", "")
chat_template_kwargs = {}
raw_chat_template_kwargs = os.environ.get("CHAT_TEMPLATE_KWARGS_JSON", "").strip()
if raw_chat_template_kwargs:
    chat_template_kwargs = json.loads(raw_chat_template_kwargs)

client = httpx.Client(timeout=timeout)

def with_prefix(text):
    if prompt_prefix and not text.startswith(prompt_prefix):
        return prompt_prefix + text
    return text

def chat(messages, max_tokens=32, stream=False):
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": stream,
    }
    if chat_template_kwargs:
        payload["chat_template_kwargs"] = chat_template_kwargs
    t0 = time.perf_counter()
    if not stream:
        r = client.post(f"{base_url}/chat/completions", json=payload)
        elapsed = time.perf_counter() - t0
        item = {"ok": r.status_code == 200, "status_code": r.status_code, "elapsed_s": round(elapsed, 3)}
        try:
            body = r.json()
            message = body.get("choices", [{}])[0].get("message", {})
            content = message.get("content") or ""
            reasoning_content = message.get("reasoning_content") or message.get("reasoning") or ""
            item["usage"] = body.get("usage")
            item["content"] = content
            if reasoning_content:
                item["reasoning_content"] = reasoning_content
            item["content_empty"] = not bool(content)
            item["reasoning_chars"] = len(reasoning_content)
            item["finish_reason"] = body.get("choices", [{}])[0].get("finish_reason")
        except Exception as exc:
            item["error"] = f"decode-json: {exc}"
            item["body_prefix"] = r.text[:1000]
        return item

    first = None
    chunks = 0
    content_parts = []
    reasoning_parts = []
    with client.stream("POST", f"{base_url}/chat/completions", json=payload) as r:
        status = r.status_code
        for line in r.iter_lines():
            if not line:
                continue
            if isinstance(line, bytes):
                line = line.decode("utf-8", errors="replace")
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            if first is None:
                first = time.perf_counter()
            chunks += 1
            try:
                delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
                content_parts.append(delta.get("content") or "")
                reasoning_parts.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
            except Exception:
                pass
    elapsed = time.perf_counter() - t0
    content = "".join(content_parts)
    reasoning_content = "".join(reasoning_parts)
    item = {
        "ok": status == 200,
        "status_code": status,
        "ttft_s": None if first is None else round(first - t0, 3),
        "elapsed_s": round(elapsed, 3),
        "chunks": chunks,
        "content": content,
        "content_empty": not bool(content),
        "reasoning_chars": len(reasoning_content),
    }
    if reasoning_content:
        item["reasoning_content"] = reasoning_content
    return {
        **item,
    }

def emit(name, item):
    item["case"] = name
    print(json.dumps(item, ensure_ascii=False), flush=True)

def normalize_value(value):
    text = str(value or "").lower()
    text = text.replace("人民币", "").replace("元", "")
    return re.sub(r"[\s\t\r\n:：,，.。;；/\\|_\\-—*]+", "", text)

def expected_variants(value):
    text = str(value or "").strip()
    variants = {normalize_value(text)}
    numeric = text.replace("人民币", "").replace("元", "").replace(",", "").replace("，", "")
    numeric = numeric.strip()
    suffix = "%" if numeric.endswith("%") else ""
    if suffix:
        numeric = numeric[:-1]
    try:
        dec = Decimal(numeric)
    except (InvalidOperation, ValueError):
        return {v for v in variants if v}
    fixed = format(dec, "f")
    trimmed = fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    for candidate in {fixed, trimmed, f"{trimmed}{suffix}", f"{fixed}{suffix}"}:
        variants.add(normalize_value(candidate))
    return {v for v in variants if v}

def image_content(image_path, prompt):
    ext = Path(image_path).suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    data = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": with_prefix(prompt)},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}},
    ]

def extract_jsonish(text):
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            continue
    return None

def iter_vlm_cases(path):
    selected = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        if vlm_case_ids and case.get("id") not in vlm_case_ids:
            continue
        selected.append(case)
        if vlm_max_cases > 0 and len(selected) >= vlm_max_cases:
            break
    return selected

emit("models", {"ok": True, "models": client.get(f"{base_url}/models").json()})
emit("stream-ttft", chat([{"role": "user", "content": with_prefix("用一句中文说明你已就绪。")}], max_tokens=32, stream=True))
emit("decode-128", chat([{"role": "user", "content": with_prefix("请用中文列出 K3 32G 边缘推理测试需要关注的五个指标。")}], max_tokens=128))

for target in ladder:
    marker = f"K3_NEEDLE_{target}_竹影星轨"
    filler = (
        "以下为边缘 AI 平台验收材料，包含硬件、运行时、模型加载、吞吐、上下文和稳定性说明。"
        "请严格依据材料回答问题，不要引入材料外信息。"
    )
    chars = max(512, int(target * 1.65))
    body = (filler * ((chars // len(filler)) + 1))[:chars]
    prompt = (
        f"{body}\n\n关键暗号: {marker}\n\n"
        "问题: 上文给出的关键暗号是什么？只回答关键暗号。"
    )
    t0 = time.perf_counter()
    item = chat([{"role": "user", "content": with_prefix(prompt)}], max_tokens=32)
    item["target_context_tokens"] = target
    item["needle"] = marker
    item["needle_recall"] = marker in item.get("content", "")
    item["needle_recall_any"] = marker in (item.get("content", "") + item.get("reasoning_content", ""))
    item["wall_s"] = round(time.perf_counter() - t0, 3)
    emit(f"context-{target}", item)

if is_vlm or run_visual_probe:
    image_path = os.environ.get("VLM_IMAGE_PATH", "")
    if image_path and Path(image_path).exists():
        content = image_content(image_path, os.environ.get("VLM_PROMPT", "请描述图片。"))
        emit("vlm-image", chat([{"role": "user", "content": content}], max_tokens=128))
    else:
        emit("vlm-image", {"ok": False, "error": f"image not found: {image_path}"})

if is_vlm and run_vlm_doc_probe:
    if vlm_cases_jsonl:
        cases_path = Path(vlm_cases_jsonl)
        if not cases_path.exists():
            emit("vlm-doc-aggregate", {"ok": False, "error": f"cases not found: {vlm_cases_jsonl}"})
        else:
            case_results = []
            for case in iter_vlm_cases(cases_path):
                payload = case.get("payload", {})
                image = Path(payload.get("image_path", ""))
                if not image.is_absolute():
                    image = cases_path.parents[3] / image
                fields = payload.get("fields", [])
                golden = payload.get("golden", {})
                prompt = (
                    "请从图片中抽取结构化字段。只输出一个 JSON object，不要 Markdown。"
                    f"文档类型: {payload.get('document_type', '')}。"
                    f"必须包含字段: {', '.join(fields)}。"
                    "金额保留数字和小数，日期保留原格式。"
                )
                if not image.exists():
                    result = {
                        "ok": False,
                        "id": case.get("id"),
                        "document_type": payload.get("document_type"),
                        "error": f"image not found: {image}",
                        "field_count": len(fields),
                        "field_hits": 0,
                        "field_accuracy": 0.0,
                    }
                    emit("vlm-doc-case", result)
                    case_results.append(result)
                    continue
                item = chat([{"role": "user", "content": image_content(str(image), prompt)}], max_tokens=vlm_doc_max_tokens)
                content = item.get("content", "")
                parsed = extract_jsonish(content)
                haystack = normalize_value(content)
                if parsed is not None:
                    haystack += normalize_value(json.dumps(parsed, ensure_ascii=False))
                field_scores = {}
                for field in fields:
                    expected = golden.get(field, "")
                    variants = expected_variants(expected)
                    field_scores[field] = any(needle in haystack for needle in variants)
                hits = sum(1 for ok in field_scores.values() if ok)
                total = len(fields)
                result = {
                    "ok": item.get("ok") is True,
                    "id": case.get("id"),
                    "document_type": payload.get("document_type"),
                    "status_code": item.get("status_code"),
                    "elapsed_s": item.get("elapsed_s"),
                    "json_parse_ok": parsed is not None,
                    "field_count": total,
                    "field_hits": hits,
                    "field_accuracy": round(hits / total, 4) if total else 0.0,
                    "case_pass": total > 0 and hits == total,
                    "field_scores": field_scores,
                    "content": content,
                    "finish_reason": item.get("finish_reason"),
                    "usage": item.get("usage"),
                }
                emit("vlm-doc-case", result)
                case_results.append(result)
            latencies = sorted(float(r["elapsed_s"]) for r in case_results if r.get("elapsed_s") is not None)
            fields_total = sum(int(r.get("field_count", 0)) for r in case_results)
            fields_hit = sum(int(r.get("field_hits", 0)) for r in case_results)
            passed = sum(1 for r in case_results if r.get("case_pass"))
            aggregate = {
                "ok": bool(case_results),
                "cases": len(case_results),
                "case_pass": passed,
                "case_pass_rate": round(passed / len(case_results), 4) if case_results else 0.0,
                "field_count": fields_total,
                "field_hits": fields_hit,
                "field_accuracy": round(fields_hit / fields_total, 4) if fields_total else 0.0,
                "json_parse_rate": round(sum(1 for r in case_results if r.get("json_parse_ok")) / len(case_results), 4) if case_results else 0.0,
            }
            if latencies:
                aggregate.update({
                    "latency_avg_s": round(sum(latencies) / len(latencies), 3),
                    "latency_p50_s": round(latencies[len(latencies) // 2], 3),
                    "latency_p95_s": round(latencies[min(len(latencies) - 1, int(len(latencies) * 0.95))], 3),
                })
            emit("vlm-doc-aggregate", aggregate)
PY
  echo "$?" > "${out_prefix}.rc"
}

wait_for_server() {
  local pid="$1" port="$2" out_prefix="$3"
  for _ in $(seq 1 180); do
    if curl -fsS "http://127.0.0.1:${port}/v1/models" > "${out_prefix}.models.json"; then
      return 0
    fi
    if ! kill -0 "${pid}" >/dev/null 2>&1; then
      return 1
    fi
    sleep 5
  done
  return 1
}

run_runtime() {
  local label="$1" bin_dir="$2" env_words="$3" model_path="$4" model_alias="$5" port="$6" is_vlm="$7" mmproj_path="${8:-}" smt_config_dir="${9:-}"
  local out="${OUT_DIR}/${label}"
  mkdir -p "${out}"

  local cli="${bin_dir}/llama-cli"
  local bench="${bin_dir}/llama-bench"
  local server="${bin_dir}/llama-server"
  if [[ ! -x "${server}" ]]; then
    log "${label}: missing llama-server at ${server}; skipping"
    echo "missing server" > "${out}/SKIPPED"
    return 0
  fi

  release_tcm_if_requested "${out}"
  local server_help
  server_help="$("${server}" --help 2>&1 || true)"

  log "${label}: runtime info"
  {
    echo "bin_dir=${bin_dir}"
    echo "env=${env_words}"
    "${server}" --version || true
    printf '%s\n' "${server_help}" | grep -E 'mmproj|media-backend|vision-backend|smt-config|cache-type|ctx|batch|threads' | head -80 || true
  } > "${out}/runtime-info.txt" 2>&1

  if [[ "${RUN_SMOKE}" == "1" && -x "${cli}" ]]; then
    log "${label}: smoke llama-cli"
    timeout "${SMOKE_TIMEOUT}" env ${env_words} "${cli}" \
      -m "${model_path}" -p "${PROMPT_PREFIX}你好，请用一句话说明你已加载。" -n 16 -t "${THREADS}" -c 1024 --no-warmup -no-cnv \
      > "${out}/smoke-cli.stdout.log" 2> "${out}/smoke-cli.stderr.log" || echo "$?" > "${out}/smoke-cli.rc"
    [[ -f "${out}/smoke-cli.rc" ]] || echo 0 > "${out}/smoke-cli.rc"
    compact_large_log "${out}/smoke-cli.stdout.log"
  fi

  if [[ "${RUN_BENCH}" == "1" && -x "${bench}" ]]; then
    log "${label}: llama-bench PP${BENCH_PROMPT}"
    timeout "${BENCH_TIMEOUT}" env ${env_words} "${bench}" \
      -m "${model_path}" -t "${THREADS}" -p "${BENCH_PROMPT}" -n 0 \
      -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}" -r "${BENCH_REPS}" -o jsonl \
      > "${out}/llama-bench-pp.jsonl" 2> "${out}/llama-bench-pp.stderr.log" || echo "$?" > "${out}/llama-bench-pp.rc"
    [[ -f "${out}/llama-bench-pp.rc" ]] || echo 0 > "${out}/llama-bench-pp.rc"

    log "${label}: llama-bench TG${BENCH_GEN}"
    timeout "${BENCH_TIMEOUT}" env ${env_words} "${bench}" \
      -m "${model_path}" -t "${THREADS}" -p 0 -n "${BENCH_GEN}" \
      -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}" -r "${BENCH_REPS}" -o jsonl \
      > "${out}/llama-bench-tg.jsonl" 2> "${out}/llama-bench-tg.stderr.log" || echo "$?" > "${out}/llama-bench-tg.rc"
    [[ -f "${out}/llama-bench-tg.rc" ]] || echo 0 > "${out}/llama-bench-tg.rc"
  fi

  if [[ "${RUN_OFFICIAL_MODELZOO_BENCH}" == "1" && -x "${bench}" ]]; then
    log "${label}: official ModelZoo llama-bench PP128/TG128"
    timeout "${BENCH_TIMEOUT}" env ${env_words} "${bench}" \
      -m "${model_path}" -t "${THREADS}" -p 128 -n 128 -mmp 0 -fa 1 -ub 128 \
      -r "${BENCH_REPS}" -o jsonl \
      > "${out}/llama-bench-official-modelzoo.jsonl" 2> "${out}/llama-bench-official-modelzoo.stderr.log" \
      || echo "$?" > "${out}/llama-bench-official-modelzoo.rc"
    [[ -f "${out}/llama-bench-official-modelzoo.rc" ]] || echo 0 > "${out}/llama-bench-official-modelzoo.rc"
  fi

  if [[ "${RUN_SERVER}" != "1" ]]; then
    return 0
  fi

  log "${label}: starting llama-server port=${port} ctx=${CTX_SIZE}"
  pkill -f "llama-server.*--port ${port}" >/dev/null 2>&1 || true
  local extra=()
  if [[ -n "${mmproj_path}" ]]; then
    extra+=(--mmproj "${mmproj_path}")
  elif [[ -n "${smt_config_dir}" ]]; then
    if printf '%s\n' "${server_help}" | grep -q -- '--media-backend'; then
      extra+=(--media-backend smt --smt-config-dir "${smt_config_dir}")
    elif printf '%s\n' "${server_help}" | grep -q -- '--vision-backend'; then
      extra+=(--vision-backend smt --smt-config-dir "${smt_config_dir}")
    else
      extra+=(--media-backend smt --smt-config-dir "${smt_config_dir}")
    fi
  fi
  if [[ -n "${SERVER_EXTRA_ARGS}" ]]; then
    # Intended for simple flag/value pairs such as:
    # SERVER_EXTRA_ARGS="--image-min-tokens 1024 --cache-ram 1024".
    # Keep values shell-word-safe; complex quoting is intentionally unsupported.
    read -r -a user_extra <<< "${SERVER_EXTRA_ARGS}"
    extra+=("${user_extra[@]}")
  fi
  printf '%q ' "${extra[@]}" > "${out}/server-extra-args.txt"
  printf '\n' >> "${out}/server-extra-args.txt"
  env ${env_words} "${server}" \
    -m "${model_path}" "${extra[@]}" \
    --alias "${model_alias}" --host 127.0.0.1 --port "${port}" \
    -c "${CTX_SIZE}" -t "${THREADS}" -tb "${THREADS_BATCH}" -b "${BATCH_SIZE}" -ub "${UBATCH_SIZE}" \
    -ctk "${CACHE_TYPE_K}" -ctv "${CACHE_TYPE_V}" --no-webui \
    --log-file "${out}/llama-server.log" \
    > "${out}/llama-server.stdout.log" 2> "${out}/llama-server.stderr.log" &
  local pid=$!
  echo "${pid}" > "${out}/llama-server.pid"

  if wait_for_server "${pid}" "${port}" "${out}/server"; then
    run_api_probe "${out}/api-probe" "${port}" "${model_alias}" "${is_vlm}"
  else
    log "${label}: server did not become ready"
    echo 1 > "${out}/server-ready.rc"
  fi

  kill "${pid}" >/dev/null 2>&1 || true
  wait "${pid}" >/dev/null 2>&1 || true
}

find_first_gguf() {
  local dir="$1"
  find "${dir}" -type f -iname '*.gguf' | sort | grep -vi 'mmproj' | head -1
}

find_first_mmproj() {
  local dir="$1"
  find "${dir}" -type f -iname '*.gguf' | sort | grep -i 'mmproj' | head -1
}

find_smt_config_dir() {
  local dir="$1"
  local config
  while IFS= read -r config; do
    local candidate_dir
    candidate_dir="$(dirname "${config}")"
    if find "${candidate_dir}" -maxdepth 1 -type f -iname '*.onnx' | grep -q .; then
      echo "${candidate_dir}"
      return 0
    fi
  done < <(find "${dir}" -type f -name 'config.json' | sort)
  return 1
}

main_llm() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}" "${EXPECTED_MIN_BYTES}"
  if [[ "${RUN_PRIVATE}" == "1" ]]; then
    run_runtime "private-spacemit" "${PRIVATE_BIN_DIR}" "${PRIVATE_ENV}" "${MODEL_PATH}" "${ALIAS}" "${PORT_BASE}" "0"
  fi
  if [[ "${RUN_UPSTREAM}" == "1" && -x "${UPSTREAM_BIN_DIR}/llama-server" ]]; then
    run_runtime "upstream-k3" "${UPSTREAM_BIN_DIR}" "" "${MODEL_PATH}" "${ALIAS}" "$((PORT_BASE + 1))" "0"
  fi
}

main_vlm_tar() {
  local tar_path="${VLM_TAR_DIR}/${VLM_TAR_FILE}"
  if [[ "${EXTRACT_VLM}" == "0" && -d "${VLM_EXTRACT_DIR}" && ! -s "${tar_path}" ]]; then
    log "using existing VLM extract dir ${VLM_EXTRACT_DIR}"
  else
    download_file "${VLM_TAR_URL}" "${VLM_TAR_DIR}" "${VLM_TAR_FILE}" "${EXPECTED_MIN_BYTES}"
    tar -tzf "${tar_path}" > "${OUT_DIR}/vlm-tar-list.txt"
    if [[ "${EXTRACT_VLM}" == "1" ]]; then
      mkdir -p "${VLM_EXTRACT_DIR}"
      tar -xzf "${tar_path}" -C "${VLM_EXTRACT_DIR}" --skip-old-files
    fi
  fi
  find "${VLM_EXTRACT_DIR}" -maxdepth 4 -type f | sort > "${OUT_DIR}/vlm-extracted-files.txt" || true
  local model_path
  local mmproj_path
  local smt_config_dir=""
  model_path="$(find_first_gguf "${VLM_EXTRACT_DIR}" || true)"
  mmproj_path="$(find_first_mmproj "${VLM_EXTRACT_DIR}" || true)"
  if [[ -z "${mmproj_path}" ]]; then
    smt_config_dir="$(find_smt_config_dir "${VLM_EXTRACT_DIR}" || true)"
  fi
  {
    echo "model_path=${model_path}"
    echo "mmproj_path=${mmproj_path}"
    echo "smt_config_dir=${smt_config_dir}"
    echo "image_path=${VLM_IMAGE_PATH}"
  } | tee "${OUT_DIR}/vlm-selection.txt"
  if [[ -z "${model_path}" ]]; then
    log "no VLM main GGUF found after extraction"
    exit 2
  fi
  local vlm_alias="${ALIAS}"
  if [[ "${vlm_alias}" == "${MODEL_FILE}" ]]; then
    vlm_alias="$(basename "${model_path}")"
  fi
  if [[ "${RUN_PRIVATE}" == "1" ]]; then
    run_runtime "private-spacemit" "${PRIVATE_BIN_DIR}" "${PRIVATE_ENV}" "${model_path}" "${vlm_alias}" "${PORT_BASE}" "1" "${mmproj_path}" "${smt_config_dir}"
  fi
  if [[ "${RUN_UPSTREAM}" == "1" && -x "${UPSTREAM_BIN_DIR}/llama-server" ]]; then
    run_runtime "upstream-k3" "${UPSTREAM_BIN_DIR}" "" "${model_path}" "${vlm_alias}" "$((PORT_BASE + 1))" "1" "${mmproj_path}" "${smt_config_dir}"
  fi
}

main_vlm_pair() {
  download_file "${MODEL_URL}" "${MODEL_DIR}" "${MODEL_FILE}" "${EXPECTED_MIN_BYTES}"
  local mmproj_path=""
  if [[ -n "${MMPROJ_URL}" && -n "${MMPROJ_FILE}" ]]; then
    download_file "${MMPROJ_URL}" "${MMPROJ_DIR}" "${MMPROJ_FILE}" "1"
    mmproj_path="${MMPROJ_DIR}/${MMPROJ_FILE}"
  elif [[ -n "${MMPROJ_FILE}" && -s "${MMPROJ_PATH}" ]]; then
    mmproj_path="${MMPROJ_PATH}"
  fi
  {
    echo "model_path=${MODEL_PATH}"
    echo "mmproj_path=${mmproj_path}"
    echo "image_path=${VLM_IMAGE_PATH}"
  } | tee "${OUT_DIR}/vlm-selection.txt"
  if [[ "${RUN_PRIVATE}" == "1" ]]; then
    run_runtime "private-spacemit" "${PRIVATE_BIN_DIR}" "${PRIVATE_ENV}" "${MODEL_PATH}" "${ALIAS}" "${PORT_BASE}" "1" "${mmproj_path}"
  fi
  if [[ "${RUN_UPSTREAM}" == "1" && -x "${UPSTREAM_BIN_DIR}/llama-server" ]]; then
    run_runtime "upstream-k3" "${UPSTREAM_BIN_DIR}" "" "${MODEL_PATH}" "${ALIAS}" "$((PORT_BASE + 1))" "1" "${mmproj_path}"
  fi
}

log "mode=${MODE} out=${OUT_DIR}"
case "${MODE}" in
  llm) main_llm ;;
  vlm-tar) main_vlm_tar ;;
  vlm-pair) main_vlm_pair ;;
  *) echo "unknown MODE=${MODE}" >&2; exit 2 ;;
esac
log "done"
