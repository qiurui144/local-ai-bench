#!/usr/bin/env bash
set -euo pipefail

# Run locally. Probe VLM VisionEncoder ONNX latency for the K3 ModelZoo VLM
# table using the same SpacemiT ORT perf-test style as the official vision run.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

CACHE_ROOT="${CACHE_ROOT:-${REPO_ROOT}/drivers/spacemit-ai/model_zoo}"
REMOTE_VLM_ROOT="${REMOTE_VLM_ROOT:-/root/models/spacemit-ai/vlm}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/reports/k3-riscv-32g/official-modelzoo-vlm-encoder-${STAMP}}"
RUN_TIMEOUT="${RUN_TIMEOUT:-420}"
REPEATS="${REPEATS:-10}"
FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-1}"

VLM_ENCODER_ROWS=(
  "fastvlm-0.5B|fastvlm-mm-0.5b-q4_1.tar.gz|fastvlm-mm-0.5b-q4_1/fastvlm_vision.f16.onnx|4|256.47"
  "fastvlm-0.5B|fastvlm-mm-0.5b-q4_1.tar.gz|fastvlm-mm-0.5b-q4_1/fastvlm_vision.f16.onnx|8|164.50"
  "Qwen3-VL-30B-A3B|qwen30ba3b-mm-q4_1.tar.gz|qwen30ba3b-mm-q4_1/qwen3_vl_vision.q_replaceneg.onnx|4|7928.13"
  "Qwen3-VL-30B-A3B|qwen30ba3b-mm-q4_1.tar.gz|qwen30ba3b-mm-q4_1/qwen3_vl_vision.q_replaceneg.onnx|8|4753.55"
  "Qwen3.5-0.8B|Qwen3.5-0.8B.tar.gz|Qwen3.5-0.8B/qwen3_5vl_0.8b-vision-384-op23.f16.onnx|4|340.42"
  "Qwen3.5-0.8B|Qwen3.5-0.8B.tar.gz|Qwen3.5-0.8B/qwen3_5vl_0.8b-vision-384-op23.f16.onnx|8|245.61"
  "Qwen3.5-2B|Qwen3.5-2B.tar.gz|Qwen3.5-2B/qwen3_5_2b-vision-384-op23.f16.onnx|4|901.56"
  "Qwen3.5-2B|Qwen3.5-2B.tar.gz|Qwen3.5-2B/qwen3_5_2b-vision-384-op23.f16.onnx|8|794.03"
  "Qwen3.5-4B|Qwen3.5-4B.tar.gz|Qwen3.5-4B/qwen3_5_4b-vision-384-op23.f16.onnx|4|904.73"
  "Qwen3.5-4B|Qwen3.5-4B.tar.gz|Qwen3.5-4B/qwen3_5_4b-vision-384-op23.f16.onnx|8|798.71"
)

if [[ "${1:-}" == "--describe" ]]; then
  k3_print_target_contract
  cat <<'EOF'

Script: run_k3_32g_official_modelzoo_vlm_encoder_probe.sh
Purpose: probe K3 VLM VisionEncoder ONNX latency for the official ModelZoo VLM
         table rows.

Probe command shape:
  onnxruntime_perf_test <vision-encoder.onnx> -e spacemit -r 10 -x 1 -S 1 -s -c 1 \
    -i SPACEMIT_EP_INTRA_THREAD_NUM|<4-or-8> -I

Important:
  The ModelZoo page documents the VLM llama-server startup command but does not
  publish a standalone encoder benchmark command. This script uses the same ORT
  perf-test method as the official vision benchmark and labels the result as a
  VisionEncoder probe.
EOF
  exit 0
fi

k3_require_target_env
mkdir -p "${OUT_ROOT}/raw"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
fi

remote() {
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "$@"
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_ROOT}/run.log"
}

extract_avg_ms() {
  awk '/Average inference time cost total:/ {print $(NF-1)}' "$1" | tail -1
}

stream_extract_if_needed() {
  local tar_file="$1"
  local expected_rel="$2"
  local local_tar="${CACHE_ROOT}/vlm/${tar_file}"
  local remote_expected="${REMOTE_VLM_ROOT}/${expected_rel}"
  [[ -s "${local_tar}" ]] || { log "missing local tar ${local_tar}"; return 3; }
  if remote "test -s '${remote_expected}'"; then
    return 0
  fi
  log "extract ${tar_file} on target"
  remote "mkdir -p '${REMOTE_VLM_ROOT}'"
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "tar -xzf - -C '${REMOTE_VLM_ROOT}' --skip-old-files" < "${local_tar}"
}

printf 'model\tartifact\tencoder\tcore\tofficial_ms\tlocal_ms\tratio\tstatus\traw_log\n' > "${OUT_ROOT}/results.tsv"

declare -A extracted=()
for row in "${VLM_ENCODER_ROWS[@]}"; do
  IFS='|' read -r model tar_file encoder_rel core official_ms <<< "${row}"
  key="${tar_file}:${encoder_rel}"
  if [[ -z "${extracted[${key}]:-}" ]]; then
    stream_extract_if_needed "${tar_file}" "${encoder_rel}" || true
    extracted["${key}"]=1
  fi

  label="$(printf '%s_%score%s' "${model}" "${core}" | tr '/ ' '__')"
  raw="${OUT_ROOT}/raw/${label}.log"
  remote_encoder="${REMOTE_VLM_ROOT}/${encoder_rel}"
  log "run ${model} core=${core}"
  if [[ "${FORCE_TCM_RELEASE}" == "1" ]]; then
    remote "spacemit-tcm-smi -c >/dev/null 2>&1 || true" || true
  fi
  set +e
  remote "set -e; test -s '${remote_encoder}'; export LD_LIBRARY_PATH=/usr/lib:\${LD_LIBRARY_PATH:-}; export SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1; timeout '${RUN_TIMEOUT}' onnxruntime_perf_test '${remote_encoder}' -e spacemit -r '${REPEATS}' -x 1 -S 1 -s -c 1 -i 'SPACEMIT_EP_INTRA_THREAD_NUM|${core}' -I" \
    > "${raw}" 2>&1
  rc=$?
  set -e
  avg="$(extract_avg_ms "${raw}" || true)"
  status="FAIL"
  ratio="-"
  if [[ "${rc}" -eq 0 && -n "${avg}" ]]; then
    ratio="$(awk -v local="${avg}" -v official="${official_ms}" 'BEGIN { printf "%.3f", local / official }')"
    status="$(awk -v local="${avg}" -v official="${official_ms}" 'BEGIN { print (local <= official * 1.05) ? "PASS" : "RETEST_REQUIRED" }')"
  elif [[ "${rc}" -eq 124 ]]; then
    status="TIMEOUT"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "${model}" "${tar_file}" "${encoder_rel}" "${core}" "${official_ms}" "${avg:--}" "${ratio}" "${status}" "${raw}" \
    | tee -a "${OUT_ROOT}/results.tsv"
done

log "done: ${OUT_ROOT}/results.tsv"
