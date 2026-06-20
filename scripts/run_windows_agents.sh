#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGETS="amd,intel"
MODELS_AMD="all"
MODELS_INTEL="all"
SKIP_COMMON="stability,translation,general_ability,conversation_drift"
SEEDS=1
INSTALL_FIRST=0
PROBE_ONLY=0
RUN_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/run_windows_agents.sh [options]

Deploy and run the Windows laptop benchmark agents from the controller.
Each enabled target runs in its own background process with separate logs.

Options:
  --targets amd|intel|amd,intel   Targets to run. Default: amd,intel
  --amd-models "all|m1 m2"        AMD model list override. Default: all
  --intel-models "all|m1 m2"      Intel model list override. Default: all
  --skip "dim1,dim2"              Extra skip list passed to run_benchmark.py
  --seeds N                       Seeds per model. Default: 1
  --install-first                 Run remote pip install before benchmark
  --probe-only                    Only run provider probes, no benchmarks
  --run-dir DIR                   Log directory. Default: reports/runs/windows-agents-<ts>

Required env:
  AMD_HOST AMD_SSH_USER AMD_SSH_PASS
  INTEL_WIN_HOST INTEL_WIN_SSH_USER INTEL_WIN_SSH_PASS

Recommended env:
  OLLAMA_AMD_BASE_URL=http://$AMD_HOST:11434/v1
  OLLAMA_INTEL_WIN_BASE_URL=http://$INTEL_WIN_HOST:11434/v1
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --targets) TARGETS="$2"; shift 2 ;;
    --amd-models) MODELS_AMD="$2"; shift 2 ;;
    --intel-models) MODELS_INTEL="$2"; shift 2 ;;
    --skip) SKIP_COMMON="$2"; shift 2 ;;
    --seeds) SEEDS="$2"; shift 2 ;;
    --install-first) INSTALL_FIRST=1; shift ;;
    --probe-only) PROBE_ONLY=1; shift ;;
    --run-dir) RUN_DIR="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if ! [[ "$SEEDS" =~ ^[0-9]+$ ]] || [[ "$SEEDS" -lt 1 ]]; then
  echo "--seeds must be a positive integer" >&2
  exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$ROOT/reports/runs/windows-agents-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"

require_env() {
  local missing=0
  for name in "$@"; do
    if [[ -z "${!name:-}" ]]; then
      echo "missing env: $name" >&2
      missing=1
    fi
  done
  return "$missing"
}

run_target() {
  local label="$1"
  local target="$2"
  local models="$3"
  local log="$RUN_DIR/${label}.log"
  local summary="$RUN_DIR/${label}.summary.tsv"
  printf "model\tprobe\tbenchmark\n" > "$summary"

  {
    echo "[$(date --iso-8601=seconds)] START $label target=$target"
    echo "models: $models"
    for model in $models; do
      echo "[$(date --iso-8601=seconds)] PROBE $model"
      if [[ "$model" == "all" ]]; then
        probe_status="skipped-all"
      else
        python scripts/probe_provider.py --model "$model"
        probe_status=$?
      fi
      bench_status="skipped"
      if [[ "$PROBE_ONLY" -eq 0 ]]; then
        args=(run_benchmark.py --target "$target" --model "$model" --skip "$SKIP_COMMON" --seeds "$SEEDS")
        if [[ "$INSTALL_FIRST" -eq 1 ]]; then
          args+=(--install-first)
        fi
        echo "[$(date --iso-8601=seconds)] BENCH $model"
        python "${args[@]}"
        bench_status=$?
      fi
      printf "%s\t%s\t%s\n" "$model" "$probe_status" "$bench_status" >> "$summary"
    done
    echo "[$(date --iso-8601=seconds)] END $label"
  } > "$log" 2>&1
}

pids=()

if [[ ",$TARGETS," == *",amd,"* ]]; then
  require_env AMD_HOST AMD_SSH_USER AMD_SSH_PASS || exit 2
  (cd "$ROOT" && run_target "amd-win-x86" "amd-win-x86" "$MODELS_AMD") &
  pids+=("$!")
fi

if [[ ",$TARGETS," == *",intel,"* ]]; then
  require_env INTEL_WIN_HOST INTEL_WIN_SSH_USER INTEL_WIN_SSH_PASS || exit 2
  (cd "$ROOT" && run_target "intel-win-x86" "intel-win-x86" "$MODELS_INTEL") &
  pids+=("$!")
fi

if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "No targets selected: $TARGETS" >&2
  exit 2
fi

printf "%s\n" "${pids[@]}" > "$RUN_DIR/pids"
status=0
for pid in "${pids[@]}"; do
  if ! wait "$pid"; then
    status=1
  fi
done

cat "$RUN_DIR"/*.summary.tsv > "$RUN_DIR/summary.tsv"
exit "$status"
