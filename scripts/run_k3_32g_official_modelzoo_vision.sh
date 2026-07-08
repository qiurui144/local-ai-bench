#!/usr/bin/env bash
set -euo pipefail

# Run locally. It caches official SpacemiT ModelZoo vision ONNX models under
# drivers/, transfers one model at a time to the K3 target, and runs the
# documented spacemit-ort benchmark across 1/2/4/8 core settings.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

k3_require_target_env
CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/drivers/spacemit-ai/model_zoo}"
REMOTE_ROOT="${REMOTE_ROOT:-/tmp/k3-model-zoo-cache/vision-official}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/reports/k3-riscv-32g/vision-official-$(date +%Y%m%d_%H%M%S)}"
DOWNLOAD="${DOWNLOAD:-1}"
CORES="${CORES:-1 2 4 8}"
RUN_TIMEOUT="${RUN_TIMEOUT:-240}"
FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-1}"

VISION_MODELS=(
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
)

SSH_BASE=(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)
SCP_BASE=(scp -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null)

if [[ -n "${SSHPASS:-}" ]]; then
  SSH_BASE=(sshpass -e "${SSH_BASE[@]}")
  SCP_BASE=(sshpass -e "${SCP_BASE[@]}")
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_ROOT}/run.log"
}

model_label() {
  local rel="$1"
  printf '%s' "${rel}" | tr '/' '_'
}

extract_avg_ms() {
  awk '/Average inference time cost total:/ {print $(NF-1)}' "$1" | tail -1
}

mkdir -p "${OUT_ROOT}/raw"
: > "${OUT_ROOT}/results.tsv"
printf 'model\tcore\tstatus\tavg_ms\traw_log\n' >> "${OUT_ROOT}/results.tsv"

if [[ "${DOWNLOAD}" == "1" ]]; then
  log "caching official vision models under ${CACHE_ROOT}"
  (cd "${REPO_ROOT}" && CACHE_ROOT="${CACHE_ROOT}" SCOPE=vision bash scripts/cache_spacemit_model_zoo.sh)
fi

log "remote=${K3_USER}@${K3_HOST} remote_root=${REMOTE_ROOT}"
"${SSH_BASE[@]}" "${K3_USER}@${K3_HOST}" "mkdir -p '${REMOTE_ROOT}'"

if [[ "${FORCE_TCM_RELEASE}" == "1" ]]; then
  log "checking and force-releasing stale TCM blocks before official ORT run"
  "${SSH_BASE[@]}" "${K3_USER}@${K3_HOST}" "spacemit-tcm-smi || true; spacemit-tcm-smi -c || true; spacemit-tcm-smi || true" \
    | tee -a "${OUT_ROOT}/tcm.log"
fi

for rel in "${VISION_MODELS[@]}"; do
  local_path="${CACHE_ROOT}/vision/${rel}"
  if [[ ! -s "${local_path}" ]]; then
    log "missing ${local_path}"
    printf '%s\t%s\t%s\t%s\t%s\n' "${rel}" "-" "MISSING_LOCAL" "-" "-" >> "${OUT_ROOT}/results.tsv"
    continue
  fi

  label="$(model_label "${rel}")"
  remote_path="${REMOTE_ROOT}/${label}"
  log "transfer ${rel}"
  "${SCP_BASE[@]}" "${local_path}" "${K3_USER}@${K3_HOST}:${remote_path}"

  for core in ${CORES}; do
    raw="${OUT_ROOT}/raw/${label}.core${core}.log"
    log "run ${rel} core=${core}"
    if [[ "${FORCE_TCM_RELEASE}" == "1" ]]; then
      "${SSH_BASE[@]}" "${K3_USER}@${K3_HOST}" "spacemit-tcm-smi -c >/dev/null 2>&1 || true"
    fi
    set +e
    "${SSH_BASE[@]}" "${K3_USER}@${K3_HOST}" \
      "set -e; export LD_LIBRARY_PATH=/usr/lib:\${LD_LIBRARY_PATH:-}; timeout '${RUN_TIMEOUT}' onnxruntime_perf_test '${remote_path}' -e spacemit -r 10 -x 1 -S 1 -s -c 1 -i 'SPACEMIT_EP_INTRA_THREAD_NUM|${core}' -I" \
      > "${raw}" 2>&1
    rc=$?
    set -e

    avg="$(extract_avg_ms "${raw}" || true)"
    if [[ "${rc}" -eq 0 && -n "${avg}" ]]; then
      status="PASS"
    elif [[ "${rc}" -eq 124 ]]; then
      status="TIMEOUT"
    else
      status="FAIL"
    fi
    printf '%s\t%s\t%s\t%s\t%s\n' "${rel}" "${core}" "${status}" "${avg:--}" "${raw}" >> "${OUT_ROOT}/results.tsv"
  done

  "${SSH_BASE[@]}" "${K3_USER}@${K3_HOST}" "rm -f '${remote_path}'" || true
done

log "done: ${OUT_ROOT}/results.tsv"
