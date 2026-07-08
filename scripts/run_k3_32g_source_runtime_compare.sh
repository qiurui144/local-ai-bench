#!/usr/bin/env bash
set -euo pipefail

# Run locally. Compare K3 system SpacemiT runtimes with locally source-built
# SpacemiT A100 artifacts. Connection values must be provided via environment.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-${REPO_ROOT}/output/reports/k3-riscv-32g/source-runtime-compare-${STAMP}}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/k3_source_compare}"
SOURCE_LLAMA_TAR="${SOURCE_LLAMA_TAR:-${REPO_ROOT}/builds/spacemit-a100/llama-install.tar.gz}"
SOURCE_ORT_TAR="${SOURCE_ORT_TAR:-${REPO_ROOT}/builds/spacemit-a100/onnxruntime-install.tar.gz}"
LLAMA_REPEATS="${LLAMA_REPEATS:-3}"
LLAMA_THREADS="${LLAMA_THREADS:-8}"
LLAMA_TIMEOUT="${LLAMA_TIMEOUT:-1800}"
RUN_LLAMA_COMPARE="${RUN_LLAMA_COMPARE:-1}"
RUN_ORT_COMPARE="${RUN_ORT_COMPARE:-0}"
ORT_MODEL_MANIFEST="${ORT_MODEL_MANIFEST:-}"
ORT_REPEATS="${ORT_REPEATS:-10}"
ORT_TIMEOUT="${ORT_TIMEOUT:-420}"

LLAMA_MODELS=(
  "qwen3-0.6B|/root/models/spacemit-ai/llm/Qwen3-0.6B-Q4_0.gguf"
  "qwen3-4B|/root/models/spacemit-ai/llm/Qwen3-4B-Q4_0.gguf"
  "qwen3-30B-A3B|/root/models/spacemit-ai/llm/Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf"
)

if [[ "${1:-}" == "--describe" ]]; then
  k3_print_target_contract
  cat <<'EOF'

Script: run_k3_32g_source_runtime_compare.sh

llama.cpp command shape:
  llama-bench -m <gguf> -t 8 -p 128 -n 128 -mmp 0 -fa 1 -ub 128 -r 3 -o jsonl

ORT command shape, enabled with RUN_ORT_COMPARE=1 and ORT_MODEL_MANIFEST=<tsv>:
  onnxruntime_perf_test <onnx> -e spacemit -r 10 -x 1 -S 1 -s -c 1 \
    -i SPACEMIT_EP_INTRA_THREAD_NUM|<core> -I

ORT manifest TSV columns:
  label<TAB>remote_onnx_path<TAB>core_count

Pass gate:
  llama.cpp source/system tokens-per-second >= 0.95
  ORT source/system average latency <= 1.05
EOF
  exit 0
fi

k3_require_target_env
mkdir -p "${OUT_ROOT}/raw"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SSH_CMD=(ssh "${SSH_OPTS[@]}")
SCP_CMD=(scp "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SSH_CMD=(sshpass -e ssh "${SSH_OPTS[@]}")
  SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
fi

remote() {
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "$@" < /dev/null
}

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*" | tee -a "${OUT_ROOT}/run.log"
}

extract_ort_avg_ms() {
  awk '/Average inference time cost total:/ {print $(NF-1)}' "$1" | tail -1
}

install_source_tar() {
  local tar_path="$1" remote_subdir="$2"
  test -s "${tar_path}" || {
    echo "missing source artifact tar: ${tar_path}" >&2
    return 2
  }
  remote "rm -rf '${REMOTE_ROOT}/${remote_subdir}' && mkdir -p '${REMOTE_ROOT}/${remote_subdir}'"
  "${SSH_CMD[@]}" "${K3_USER}@${K3_HOST}" "tar -xzf - -C '${REMOTE_ROOT}/${remote_subdir}'" < "${tar_path}"
}

run_llama_compare() {
  install_source_tar "${SOURCE_LLAMA_TAR}" "llama-install"
  printf 'runtime\tmodel\tartifact\tstatus\tstdout\tstderr\n' > "${OUT_ROOT}/run-manifest.tsv"
  local row label model_path runtime bin lib_prefix stdout stderr remote_raw rc
  for row in "${LLAMA_MODELS[@]}"; do
    IFS='|' read -r label model_path <<< "${row}"
    if ! remote "test -s '${model_path}'"; then
      log "skip missing model ${label}: ${model_path}"
      continue
    fi
    for runtime in system source; do
      if [[ "${runtime}" == "system" ]]; then
        bin="/usr/bin/llama-bench"
        lib_prefix=""
      else
        bin="${REMOTE_ROOT}/llama-install/bin/llama-bench"
        lib_prefix="LD_LIBRARY_PATH='${REMOTE_ROOT}/llama-install/lib':\${LD_LIBRARY_PATH:-}"
      fi
      remote_raw="${REMOTE_ROOT}/raw/${runtime}_${label}.jsonl"
      stdout="${OUT_ROOT}/raw/${runtime}_${label}.jsonl"
      stderr="${OUT_ROOT}/raw/${runtime}_${label}.stderr"
      remote "mkdir -p '${REMOTE_ROOT}/raw'; spacemit-tcm-smi -c >/dev/null 2>&1 || true" || true
      log "llama ${runtime} ${label}"
      set +e
      remote "set -e; ${lib_prefix} timeout '${LLAMA_TIMEOUT}' '${bin}' -m '${model_path}' -t '${LLAMA_THREADS}' -p 128 -n 128 -mmp 0 -fa 1 -ub 128 -r '${LLAMA_REPEATS}' -o jsonl > '${remote_raw}'" \
        > "${stderr}" 2>&1
      rc=$?
      set -e
      "${SCP_CMD[@]}" "${K3_USER}@${K3_HOST}:${remote_raw}" "${stdout}" >/dev/null 2>&1 || true
      printf '%s\t%s\t%s\trc=%s\t%s\t%s\n' "${runtime}" "${label}" "$(basename "${model_path}")" "${rc}" "${stdout}" "${stderr}" \
        >> "${OUT_ROOT}/run-manifest.tsv"
    done
  done
  python3 - "$OUT_ROOT" <<'PY'
import json
import statistics
import sys
from pathlib import Path

out = Path(sys.argv[1])
rows = {}
for path in sorted((out / "raw").glob("*.jsonl")):
    parts = path.stem.split("_", 1)
    if len(parts) != 2:
        continue
    runtime, model = parts
    vals = {}
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        key = obj.get("test")
        if key and obj.get("avg_ts") is not None:
            vals.setdefault(key, []).append(float(obj["avg_ts"]))
    for key, samples in vals.items():
        rows.setdefault((model, key), {})[runtime] = samples

with (out / "summary.tsv").open("w", encoding="utf-8") as f:
    f.write("model\ttest\tsystem_tps\tsource_tps\tsource_vs_system\tstatus\tsystem_samples\tsource_samples\n")
    for (model, test), by_rt in sorted(rows.items()):
        system = by_rt.get("system", [])
        source = by_rt.get("source", [])
        if not system or not source:
            continue
        sys_mean = statistics.mean(system)
        src_mean = statistics.mean(source)
        ratio = src_mean / sys_mean if sys_mean else 0.0
        status = "PASS" if ratio >= 0.95 else "RETEST_REQUIRED"
        f.write(
            f"{model}\t{test}\t{sys_mean:.2f}\t{src_mean:.2f}\t{ratio:.3f}\t{status}\t"
            f"{','.join(f'{v:.3f}' for v in system)}\t{','.join(f'{v:.3f}' for v in source)}\n"
        )
PY
}

run_ort_compare() {
  [[ -n "${ORT_MODEL_MANIFEST}" && -s "${ORT_MODEL_MANIFEST}" ]] || {
    echo "ORT_MODEL_MANIFEST is required when RUN_ORT_COMPARE=1" >&2
    return 2
  }
  install_source_tar "${SOURCE_ORT_TAR}" "onnxruntime-install"
  printf 'model\tcore\tsystem_ms\tsource_ms\tsource_vs_system\tstatus\tsystem_log\tsource_log\n' > "${OUT_ROOT}/ort-summary.tsv"
  local label model_path core runtime bin lib_prefix raw avg system_ms source_ms
  while IFS=$'\t' read -r label model_path core; do
    [[ -z "${label}" || "${label}" == "label" ]] && continue
    remote "test -s '${model_path}'" || { log "skip missing ONNX ${label}: ${model_path}"; continue; }
    system_ms=""
    source_ms=""
    for runtime in system source; do
      if [[ "${runtime}" == "system" ]]; then
        bin="/usr/bin/onnxruntime_perf_test"
        lib_prefix="LD_LIBRARY_PATH=/usr/lib:\${LD_LIBRARY_PATH:-}"
      else
        bin="${REMOTE_ROOT}/onnxruntime-install/bin/onnxruntime_perf_test"
        lib_prefix="LD_LIBRARY_PATH='${REMOTE_ROOT}/onnxruntime-install/lib':/usr/lib:\${LD_LIBRARY_PATH:-}"
      fi
      raw="${OUT_ROOT}/raw/ort_${runtime}_${label}_core${core}.log"
      log "ort ${runtime} ${label} core=${core}"
      set +e
      remote "set -e; export SPACEMIT_EP_DENSE_ACCURACY_LEVEL=1; ${lib_prefix} timeout '${ORT_TIMEOUT}' '${bin}' '${model_path}' -e spacemit -r '${ORT_REPEATS}' -x 1 -S 1 -s -c 1 -i 'SPACEMIT_EP_INTRA_THREAD_NUM|${core}' -I" \
        > "${raw}" 2>&1
      set -e
      avg="$(extract_ort_avg_ms "${raw}" || true)"
      if [[ "${runtime}" == "system" ]]; then
        system_ms="${avg}"
      else
        source_ms="${avg}"
      fi
    done
    local ratio status
    ratio="-"
    status="FAIL"
    if [[ -n "${system_ms}" && -n "${source_ms}" ]]; then
      ratio="$(awk -v src="${source_ms}" -v sys="${system_ms}" 'BEGIN { printf "%.3f", src / sys }')"
      status="$(awk -v src="${source_ms}" -v sys="${system_ms}" 'BEGIN { print (src <= sys * 1.05) ? "PASS" : "RETEST_REQUIRED" }')"
    fi
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "${label}" "${core}" "${system_ms:--}" "${source_ms:--}" "${ratio}" "${status}" \
      "${OUT_ROOT}/raw/ort_system_${label}_core${core}.log" "${OUT_ROOT}/raw/ort_source_${label}_core${core}.log" \
      >> "${OUT_ROOT}/ort-summary.tsv"
  done < "${ORT_MODEL_MANIFEST}"
}

if [[ "${RUN_LLAMA_COMPARE}" == "1" ]]; then
  run_llama_compare
fi
if [[ "${RUN_ORT_COMPARE}" == "1" ]]; then
  run_ort_compare
fi

log "done: ${OUT_ROOT}"
