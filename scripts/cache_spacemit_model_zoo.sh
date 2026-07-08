#!/usr/bin/env bash
set -euo pipefail

# Run locally. Cache public SpacemiT model_zoo artifacts under drivers/.

CACHE_ROOT="${CACHE_ROOT:-drivers/spacemit-ai/model_zoo}"
SCOPE="${SCOPE:-high}" # high | llm | vlm | vision | nonllm | official | all
ARIA2_SPLIT="${ARIA2_SPLIT:-8}"
PARALLEL_DOWNLOADS="${PARALLEL_DOWNLOADS:-3}"
STRICT_MD5="${STRICT_MD5:-0}"
LLM_BASE="${LLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/llm}"
VLM_BASE="${VLM_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/vlm}"
VISION_BASE="${VISION_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/vision}"
ASR_BASE="${ASR_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/asr}"
EMBED_BASE="${EMBED_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/embed}"
RERANK_BASE="${RERANK_BASE:-https://archive.spacemit.com/spacemit-ai/model_zoo/rerank}"
QWEN3VL_BASE="${QWEN3VL_BASE:-${VLM_BASE}/Qwen3VL}"
PPOCR_DICT_URL="${PPOCR_DICT_URL:-https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/ppocr/utils/dict/ppocrv5_dict.txt}"
MS_QWEN3_17B_Q4_0_URL="${MS_QWEN3_17B_Q4_0_URL:-https://www.modelscope.cn/models/unsloth/Qwen3-1.7B-GGUF/resolve/master/Qwen3-1.7B-Q4_0.gguf}"
MS_QWEN3_4B_Q4_0_URL="${MS_QWEN3_4B_Q4_0_URL:-https://www.modelscope.cn/models/unsloth/Qwen3-4B-GGUF/resolve/master/Qwen3-4B-Q4_0.gguf}"
MS_QWEN3_30B_A3B_Q4_0_URL="${MS_QWEN3_30B_A3B_Q4_0_URL:-https://www.modelscope.cn/models/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF/resolve/master/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf}"
MS_LLAMA2_7B_Q4_0_URL="${MS_LLAMA2_7B_Q4_0_URL:-https://www.modelscope.cn/models/TheBloke/Llama-2-7B-GGUF/resolve/master/llama-2-7b.Q4_0.gguf}"

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
  qwen3-asr-0.6B.tar.gz
  smollm2-135m-q40-agv-fc.gguf
)

VISION_ONNX_ALL=(
  resnet/resnet18.q.onnx
  resnet/resnet50.q.onnx
  resnet/resnet50.b4.q.onnx
  resnet/resnet50.fp16.onnx
  mobilenet/mobilenet_v1.q.onnx
  mobilenet/mobilenet_v2.q.onnx
  mobilenet/mobilenet_v3_small.fp16.onnx
  mobilenet/mobilenet_v3_large.fp16.onnx
  efficientnet/efficientnet_v1_b0.q.onnx
  efficientnet/efficientnet_v1_b1.q.onnx
  efficientnet/efficientnet_v2_s.q.onnx
  efficientnet/efficientnet_v1_b0.fp16.onnx
  efficientnet/efficientnet_v1_b1.fp16.onnx
  efficientnet/efficientnet_v2_s.fp16.onnx
  vit/vit_b_16.q.onnx
  vit/vit_b_16.fp16.onnx
  yolov5/yolov5n.q.onnx
  yolov5/yolov5s.q.onnx
  yolov5/yolov5m.q.onnx
  yolov6/yolov6n.q.onnx
  yolov6/yolov6s.q.onnx
  yolov8/yolov8n.q.onnx
  yolov8/yolov8s.q.onnx
  yolov8/yolov8m.q.onnx
  yolov8_seg/yolov8n-seg.q.onnx
  yolov8_seg/yolov8s-seg.q.onnx
  yolov8_seg/yolov8m-seg.q.onnx
  yolov8_pose/yolov8n-pose.q.onnx
  yolov8_pose/yolov8s-pose.q.onnx
  yolov8_pose/yolov8m-pose.q.onnx
  yolo12/yolo12n.q.onnx
  yolo12/yolo12s.q.onnx
  yolo12/yolo12m.q.onnx
  ppocr/PP-OCRv5_mobile_det.onnx
  ppocr/PP-OCRv5_mobile_rec.onnx
)

ASR_TAR_ALL=(
  sensevoice.tar.gz
  qwen3-asr-1.7B-dynq-q4km.tar.gz
)

EMBED_GGUF_ALL=(
  Bge-Small-Zh-V1.5-Q4_K_M.gguf
  Bge-Small-En-V1.5-Q4_K_M.gguf
  Jina-Embeddings-V5-Text-Small-Retrieval-Q4_K_M.gguf
  Nomic-Embed-Text-V2-Moe-Q4_0.gguf
  Qwen3-Embedding-0.6B-Q4_0.gguf
)

RERANK_GGUF_ALL=(
  Bge-Reranker-V2-M3-Q4_0.gguf
  Qwen3-Reranker-0.6B-Q4_0.gguf
)

OFFICIAL_LLM_EXTRA_URLS=(
  "Qwen3-1.7B-Q4_0.gguf|${MS_QWEN3_17B_Q4_0_URL}"
  "Qwen3-4B-Q4_0.gguf|${MS_QWEN3_4B_Q4_0_URL}"
  "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf|${MS_QWEN3_30B_A3B_Q4_0_URL}"
  "llama-2-7b.Q4_0.gguf|${MS_LLAMA2_7B_Q4_0_URL}"
)

PPOCR_ONNX=(
  ppocr/PP-OCRv5_mobile_det.onnx
  ppocr/PP-OCRv5_mobile_rec.onnx
)

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

record_cache_index() {
  local path="$1" size
  size="$(stat -c '%s' "${path}")"
  printf '%s\t%s\n' "${size}" "${path}" >> "${CACHE_ROOT}/cache-index.tsv"
}

download_url() {
  local url="$1" rel="$2"
  local dir="${CACHE_ROOT}/$(dirname "${rel}")"
  local file
  file="$(basename "${rel}")"
  local path="${dir}/${file}"
  mkdir -p "${dir}"

  log "cache ${rel}"
  fetch_file() {
    if command -v aria2c >/dev/null 2>&1; then
      aria2c -c -x "${ARIA2_SPLIT}" -s "${ARIA2_SPLIT}" -k 4M \
        --console-log-level=warn --summary-interval=0 --show-console-readout=false \
        --file-allocation=none --auto-file-renaming=false --allow-overwrite=true \
        -d "${dir}" -o "${file}" "${url}"
    else
      curl -fL --retry 5 -C - -o "${path}" "${url}"
    fi
  }

  fetch_file

  curl -fsSL --retry 3 -o "${path}.md5" "${url}.md5" || true
  if [[ -s "${path}.md5" ]]; then
    local expected actual
    expected="$(awk '{print $1; exit}' "${path}.md5")"
    actual="$(md5sum "${path}" | awk '{print $1}')"
    printf '%s  %s\n' "${actual}" "${file}" > "${path}.md5.actual"
    if [[ "${expected}" =~ ^[0-9a-fA-F]{32}$ && "${expected,,}" != "${actual}" ]]; then
      log "md5 mismatch rel=${rel} expected=${expected} actual=${actual}; retrying clean download"
      rm -f "${path}" "${path}.aria2"
      fetch_file
      actual="$(md5sum "${path}" | awk '{print $1}')"
      printf '%s  %s\n' "${actual}" "${file}" > "${path}.md5.actual"
      if [[ "${expected,,}" != "${actual}" ]]; then
        log "md5 mismatch after retry rel=${rel} expected=${expected} actual=${actual}"
        if [[ "${STRICT_MD5}" == "1" ]]; then
          return 3
        fi
      fi
    fi
    log "md5 ok/recorded ${rel} ${actual}"
  fi

  record_cache_index "${path}"
}

download_llm_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${LLM_BASE}/${file}" "llm/${file}"
  done
}

download_official_llm_extra_set() {
  local spec file url
  for spec in "${OFFICIAL_LLM_EXTRA_URLS[@]}"; do
    IFS='|' read -r file url <<< "${spec}"
    download_url "${url}" "llm/${file}"
  done
}

download_vlm_tar_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${VLM_BASE}/${file}" "vlm/${file}"
  done
}

download_vlm_pairs() {
  download_url "${QWEN3VL_BASE}/Qwen3VL-4B-Instruct-Q4_K_M.gguf" \
    "vlm/Qwen3VL/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  download_url "${QWEN3VL_BASE}/mmproj-Qwen3VL-4B-Instruct-F16.gguf" \
    "vlm/Qwen3VL/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
  download_url "${VLM_BASE}/SmolVLM-256M-Instruct-f16.gguf" \
    "vlm/SmolVLM-256M-Instruct-f16.gguf"
  download_url "${VLM_BASE}/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" \
    "vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
}

download_vision_onnx_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${VISION_BASE}/${file}" "vision/${file}"
  done
}

download_asr_tar_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${ASR_BASE}/${file}" "asr/${file}"
  done
}

download_embed_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${EMBED_BASE}/${file}" "embed/${file}"
  done
}

download_rerank_set() {
  local files=("$@")
  local file
  for file in "${files[@]}"; do
    download_url "${RERANK_BASE}/${file}" "rerank/${file}"
  done
}

download_ppocr_dict() {
  download_url "${PPOCR_DICT_URL}" "vision/ppocr/ppocrv5_dict.txt"
}

download_ppocr_assets() {
  download_vision_onnx_set "${PPOCR_ONNX[@]}"
  download_ppocr_dict
}

rebuild_cache_index_from_manifest() {
  local rel url
  : > "${CACHE_ROOT}/cache-index.tsv"
  while IFS=$'\t' read -r rel url; do
    if [[ -s "${CACHE_ROOT}/${rel}" ]]; then
      record_cache_index "${CACHE_ROOT}/${rel}"
    fi
  done < <(emit_manifest | awk -F '\t' 'NR > 1 {print $3 "\t" $4}' | sort -u)
}

download_missing_parallel() {
  local manifest missing rel url running rc
  manifest="$(mktemp)"
  missing="$(mktemp)"
  trap 'rm -f "${manifest}" "${missing}"' RETURN

  mkdir -p "${CACHE_ROOT}"
  emit_manifest | awk -F '\t' 'NR > 1 {print $3 "\t" $4}' | sort -u > "${manifest}"
  while IFS=$'\t' read -r rel url; do
    if [[ ! -s "${CACHE_ROOT}/${rel}" || -e "${CACHE_ROOT}/${rel}.aria2" ]]; then
      printf '%s\t%s\n' "${rel}" "${url}" >> "${missing}"
    fi
  done < "${manifest}"

  if [[ ! -s "${missing}" ]]; then
    log "cache already complete root=${CACHE_ROOT}"
    rebuild_cache_index_from_manifest
    return 0
  fi

  log "parallel missing downloads jobs=${PARALLEL_DOWNLOADS} root=${CACHE_ROOT}"
  cat "${missing}" > "${CACHE_ROOT}/cache-missing.tsv"
  running=0
  rc=0
  while IFS=$'\t' read -r rel url; do
    (
      download_url "${url}" "${rel}"
    ) &
    running=$((running + 1))
    if [[ "${running}" -ge "${PARALLEL_DOWNLOADS}" ]]; then
      if ! wait -n; then
        rc=1
      fi
      running=$((running - 1))
    fi
  done < "${missing}"

  while [[ "${running}" -gt 0 ]]; do
    if ! wait -n; then
      rc=1
    fi
    running=$((running - 1))
  done

  rebuild_cache_index_from_manifest
  return "${rc}"
}

emit_manifest_row() {
  local scope="$1" category="$2" rel="$3" url="$4"
  printf '%s\t%s\t%s\t%s\n' "${scope}" "${category}" "${rel}" "${url}"
}

emit_manifest() {
  local file
  printf 'scope\tcategory\trelative_path\turl\n'
  for file in "${LLM_HIGH[@]}"; do
    emit_manifest_row high llm "llm/${file}" "${LLM_BASE}/${file}"
  done
  for file in "${LLM_ALL[@]}"; do
    emit_manifest_row llm llm "llm/${file}" "${LLM_BASE}/${file}"
  done
  for file in "${OFFICIAL_LLM_EXTRA_URLS[@]}"; do
    IFS='|' read -r rel url <<< "${file}"
    emit_manifest_row llm llm "llm/${rel}" "${url}"
  done
  for file in "${VLM_TAR_ALL[@]}"; do
    emit_manifest_row vlm vlm "vlm/${file}" "${VLM_BASE}/${file}"
  done
  emit_manifest_row vlm vlm "vlm/Qwen3VL/Qwen3VL-4B-Instruct-Q4_K_M.gguf" "${QWEN3VL_BASE}/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
  emit_manifest_row vlm vlm "vlm/Qwen3VL/mmproj-Qwen3VL-4B-Instruct-F16.gguf" "${QWEN3VL_BASE}/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
  emit_manifest_row vlm vlm "vlm/SmolVLM-256M-Instruct-f16.gguf" "${VLM_BASE}/SmolVLM-256M-Instruct-f16.gguf"
  emit_manifest_row vlm vlm "vlm/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf" "${VLM_BASE}/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf"
  for file in "${VISION_ONNX_ALL[@]}"; do
    emit_manifest_row vision vision "vision/${file}" "${VISION_BASE}/${file}"
  done
  emit_manifest_row vision vision "vision/ppocr/ppocrv5_dict.txt" "${PPOCR_DICT_URL}"
  for file in "${ASR_TAR_ALL[@]}"; do
    emit_manifest_row nonllm asr "asr/${file}" "${ASR_BASE}/${file}"
  done
  emit_manifest_row nonllm vlm "vlm/qwen3-asr-0.6B.tar.gz" "${VLM_BASE}/qwen3-asr-0.6B.tar.gz"
  for file in "${EMBED_GGUF_ALL[@]}"; do
    emit_manifest_row nonllm embed "embed/${file}" "${EMBED_BASE}/${file}"
  done
  for file in "${RERANK_GGUF_ALL[@]}"; do
    emit_manifest_row nonllm rerank "rerank/${file}" "${RERANK_BASE}/${file}"
  done
}

if [[ "${1:-}" == "--manifest" ]]; then
  emit_manifest
  exit 0
fi

if [[ "${1:-}" == "--download-missing" ]]; then
  download_missing_parallel
  log "cache complete root=${CACHE_ROOT}"
  exit 0
fi

mkdir -p "${CACHE_ROOT}"
: > "${CACHE_ROOT}/cache-index.tsv"

case "${SCOPE}" in
  high)
    download_llm_set "${LLM_HIGH[@]}"
    download_vlm_tar_set "${VLM_TAR_HIGH[@]}"
    download_vlm_pairs
    ;;
  llm)
    download_llm_set "${LLM_ALL[@]}"
    download_official_llm_extra_set
    ;;
  vlm)
    download_vlm_tar_set "${VLM_TAR_ALL[@]}"
    download_vlm_pairs
    ;;
  vision)
    download_vision_onnx_set "${VISION_ONNX_ALL[@]}"
    download_ppocr_dict
    ;;
  nonllm)
    download_asr_tar_set "${ASR_TAR_ALL[@]}"
    download_vlm_tar_set qwen3-asr-0.6B.tar.gz
    download_embed_set "${EMBED_GGUF_ALL[@]}"
    download_rerank_set "${RERANK_GGUF_ALL[@]}"
    download_ppocr_assets
    ;;
  official)
    download_vision_onnx_set "${VISION_ONNX_ALL[@]}"
    download_ppocr_dict
    download_asr_tar_set "${ASR_TAR_ALL[@]}"
    download_embed_set "${EMBED_GGUF_ALL[@]}"
    download_rerank_set "${RERANK_GGUF_ALL[@]}"
    download_llm_set \
      Qwen3-0.6B-Q4_0.gguf \
      Qwen3.5-0.8B-Q4_0.gguf \
      Qwen3.5-2B-Q4_0.gguf \
      HY-MT1.5-1.8B-Q4_K_M.gguf
    download_official_llm_extra_set
    download_vlm_tar_set \
      fastvlm-mm-0.5b-q4_1.tar.gz \
      qwen30ba3b-mm-q4_1.tar.gz \
      Qwen3.5-0.8B.tar.gz \
      Qwen3.5-2B.tar.gz \
      Qwen3.5-4B.tar.gz \
      qwen3-asr-0.6B.tar.gz
    ;;
  all)
    download_llm_set "${LLM_ALL[@]}"
    download_official_llm_extra_set
    download_vlm_tar_set "${VLM_TAR_ALL[@]}"
    download_vlm_pairs
    download_vision_onnx_set "${VISION_ONNX_ALL[@]}"
    download_ppocr_dict
    download_asr_tar_set "${ASR_TAR_ALL[@]}"
    download_embed_set "${EMBED_GGUF_ALL[@]}"
    download_rerank_set "${RERANK_GGUF_ALL[@]}"
    ;;
  *)
    echo "unknown SCOPE=${SCOPE}; use high|llm|vlm|vision|nonllm|official|all" >&2
    exit 2
    ;;
esac

log "cache complete root=${CACHE_ROOT}"
