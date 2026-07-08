#!/usr/bin/env bash
set -euo pipefail

# Run locally. Retest K3 LLM artifacts with the exact SpacemiT ModelZoo
# llama-bench command. This is intentionally narrower than the full model_zoo
# quality harness: it only verifies official PP128/TG128 throughput baselines.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
source "${SCRIPT_DIR}/k3_32g_common.sh"

CACHE_ROOT="${CACHE_ROOT:-drivers/spacemit-ai/model_zoo}"
REMOTE_BASE="${REMOTE_BASE:-/root/k3_32g_official_modelzoo_llm}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
BENCH_REPS="${BENCH_REPS:-3}"
LOCAL_OUT_ROOT="${LOCAL_OUT_ROOT:-${REPO_ROOT}/output/reports/k3-riscv-32g/official-modelzoo-llm-${STAMP}}"
PRIVATE_ENV="${PRIVATE_ENV-}"
FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE:-1}"

OFFICIAL_LLM_CACHE_MODELS=(
  "Qwen3-0.6B-Q4_0.gguf:qwen3-0.6B:Q4_0"
  "Qwen3-1.7B-Q4_0.gguf:qwen3-1.7B:Q4_0"
  "Qwen3-4B-Q4_0.gguf:qwen3-4B:Q4_0"
  "Qwen3-30B-A3B-Instruct-2507-Q4_0.gguf:qwen3-moe-30B-A3B:Q4_0"
  "Qwen3.5-0.8B-Q4_0.gguf:qwen3.5-0.8B:Q4_0"
  "Qwen3.5-2B-Q4_0.gguf:qwen3.5-2B:Q4_0-official-link"
  "HY-MT1.5-1.8B-Q4_K_M.gguf:HY-MT1.5-1.8B:Q4_K_M"
  "llama-2-7b.Q4_0.gguf:llama2-7B:Q4_0"
)

if [[ "${1:-}" == "--describe" ]]; then
  k3_print_target_contract
  cat <<'EOF'

Script: run_k3_32g_official_modelzoo_llm_retest.sh
Purpose: retest cached LLM artifacts against the exact SpacemiT ModelZoo LLM
         benchmark command.

Official command:
  llama-bench -m <model> -t <cores> -p 128 -n 128 -mmp 0 -fa 1 -ub 128

Runtime behavior:
  - uses scripts/run_k3_32g_model_zoo_cached.sh
  - sets RUN_OFFICIAL_MODELZOO_BENCH=1
  - disables smoke, server, and non-official PP512/TG128 bench paths
  - leaves PRIVATE_ENV empty by default so the SpacemiT TCM runtime is enabled
  - runs spacemit-tcm-smi -c before each model by default
  - writes remote output under /root/k3_32g_official_modelzoo_llm/<timestamp>/*
  - pulls remote output into output/reports/k3-riscv-32g/official-modelzoo-llm-<timestamp>/

Important:
  Do not compare neighboring quantizations. If the exact official artifact is
  not in drivers/spacemit-ai/model_zoo/llm, this script records it as missing
  instead of substituting Q4_K_M/Q8_0/etc. Run
  scripts/cache_spacemit_model_zoo.sh --download-missing first to fetch
  archive and ModelScope official-cache entries.
EOF
  exit 0
fi

k3_require_target_env

missing=0
ran=0
failed=0

mkdir -p "${LOCAL_OUT_ROOT}"
manifest="${LOCAL_OUT_ROOT}/run-manifest.tsv"

SSH_OPTS=(-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR)
SCP_CMD=(scp "${SSH_OPTS[@]}")
if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
  SCP_CMD=(sshpass -e scp "${SSH_OPTS[@]}")
fi

pull_dir() {
  local remote_dir="$1"
  local local_dir="$2"
  mkdir -p "${local_dir}"
  if command -v rsync >/dev/null 2>&1; then
    if [[ -n "${SSHPASS:-}" ]] && command -v sshpass >/dev/null 2>&1; then
      sshpass -e rsync -ah --delete \
        -e "ssh ${SSH_OPTS[*]}" \
        "${K3_USER}@${K3_HOST}:${remote_dir}/" "${local_dir}/" || true
    else
      rsync -ah --delete \
        -e "ssh ${SSH_OPTS[*]}" \
        "${K3_USER}@${K3_HOST}:${remote_dir}/" "${local_dir}/" || true
    fi
  else
    "${SCP_CMD[@]}" -rp "${K3_USER}@${K3_HOST}:${remote_dir}/." "${local_dir}/" || true
  fi
}

printf 'official_model\tquant\tartifact\tstatus\tremote_out\tlocal_out\n' | tee "${manifest}"
for spec in "${OFFICIAL_LLM_CACHE_MODELS[@]}"; do
  IFS=: read -r file name quant <<<"${spec}"
  if [[ ! -s "${CACHE_ROOT}/llm/${file}" ]]; then
    printf '%s\t%s\t%s\tmissing-local-cache\t-\t-\n' "${name}" "${quant}" "${file}" | tee -a "${manifest}"
    missing=$((missing + 1))
    continue
  fi

  alias="official-${name}-${quant}"
  out_dir="${REMOTE_BASE}/${STAMP}/${alias}"
  local_dir="${LOCAL_OUT_ROOT}/${alias}"
  printf '%s\t%s\t%s\trun-started\t%s\t%s\n' "${name}" "${quant}" "${file}" "${out_dir}" "${local_dir}" | tee -a "${manifest}"
  set +e
  RUN_OFFICIAL_MODELZOO_BENCH=1 \
  RUN_BENCH=0 \
  RUN_SERVER=0 \
  RUN_SMOKE=0 \
  BENCH_REPS="${BENCH_REPS}" \
  PRIVATE_ENV="${PRIVATE_ENV}" \
  FORCE_TCM_RELEASE="${FORCE_TCM_RELEASE}" \
  MODE=llm \
  MODEL_FILE="${file}" \
  ALIAS="${alias}" \
  OUT_DIR="${out_dir}" \
  CACHE_ROOT="${CACHE_ROOT}" \
  bash "${SCRIPT_DIR}/run_k3_32g_model_zoo_cached.sh"
  rc=$?
  set -e
  pull_dir "${out_dir}" "${local_dir}"
  printf '%s\t%s\t%s\trun-rc=%s\t%s\t%s\n' "${name}" "${quant}" "${file}" "${rc}" "${out_dir}" "${local_dir}" | tee -a "${manifest}"
  if [[ "${rc}" -ne 0 ]]; then
    failed=$((failed + 1))
  fi
  ran=$((ran + 1))
done

printf 'summary\t-\t-\tran=%s missing=%s failed=%s\t-\t%s\n' "${ran}" "${missing}" "${failed}" "${LOCAL_OUT_ROOT}" | tee -a "${manifest}"
printf 'local_output_root=%s\n' "${LOCAL_OUT_ROOT}"
