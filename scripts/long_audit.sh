#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROUNDS=20
INTERVAL_SECONDS=1900
RUN_DIR=""

usage() {
  cat <<'EOF'
Usage: scripts/long_audit.sh [--rounds N] [--interval SECONDS] [--run-dir DIR]

Runs repeated offline checks and writes per-round logs. The default 20 rounds
with a 1900 second interval takes more than 10 hours end to end.

Checks per round:
  - python compileall for core source/test trees
  - full pytest suite
  - scenario dataset verifier
  - synthetic/fixture image guard
  - static risk scan for common dangerous patterns

The script does not modify source files.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rounds)
      ROUNDS="$2"
      shift 2
      ;;
    --interval)
      INTERVAL_SECONDS="$2"
      shift 2
      ;;
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if ! [[ "$ROUNDS" =~ ^[0-9]+$ ]] || [[ "$ROUNDS" -lt 1 ]]; then
  echo "--rounds must be a positive integer" >&2
  exit 2
fi

if ! [[ "$INTERVAL_SECONDS" =~ ^[0-9]+$ ]]; then
  echo "--interval must be a non-negative integer" >&2
  exit 2
fi

if [[ -z "$RUN_DIR" ]]; then
  RUN_DIR="$ROOT/reports/runs/long-audit-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p "$RUN_DIR"

SUMMARY="$RUN_DIR/summary.tsv"
STATUS_JSONL="$RUN_DIR/status.jsonl"
PID_FILE="$RUN_DIR/pid"

echo "$$" > "$PID_FILE"
printf "round\tstarted_at\tended_at\tcompileall\tpytest\tverify_benchmark\timage_guard\tstatic_scan\n" > "$SUMMARY"

run_step() {
  local name="$1"
  local logfile="$2"
  shift 2
  {
    echo "[$(date --iso-8601=seconds)] START $name"
    echo "+ $*"
    "$@"
    local code=$?
    echo "[$(date --iso-8601=seconds)] END $name status=$code"
    return "$code"
  } >"$logfile" 2>&1
}

for ((round = 1; round <= ROUNDS; round++)); do
  round_dir="$RUN_DIR/round-$(printf '%02d' "$round")"
  mkdir -p "$round_dir"
  started_at="$(date --iso-8601=seconds)"

  git -C "$ROOT" status --short > "$round_dir/git-status.txt" 2>&1
  python -VV > "$round_dir/python-version.txt" 2>&1

  run_step "compileall" "$round_dir/compileall.log" \
    python -m compileall -q common.py run_benchmark.py benchmark tests scripts
  compileall_status=$?

  run_step "pytest" "$round_dir/pytest.log" \
    python -m pytest -q
  pytest_status=$?

  run_step "verify_benchmark" "$round_dir/verify_benchmark.log" \
    python scripts/verify_benchmark.py
  verify_status=$?

  if [[ -x "$ROOT/scripts/check_no_real_images.sh" ]]; then
    run_step "image_guard" "$round_dir/image_guard.log" \
      bash scripts/check_no_real_images.sh
    image_status=$?
  else
    echo "scripts/check_no_real_images.sh is not executable" > "$round_dir/image_guard.log"
    image_status=127
  fi

  run_step "static_scan" "$round_dir/static_scan.log" \
    rg -n "TODO|FIXME|XXX|except Exception|eval\\(|exec\\(|shell=True|pickle|yaml\\.load|verify=False|timeout=None" \
      benchmark common.py run_benchmark.py tests scripts
  scan_status=$?
  if [[ "$scan_status" -eq 1 ]]; then
    scan_status=0
  fi

  ended_at="$(date --iso-8601=seconds)"
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" \
    "$round" "$started_at" "$ended_at" "$compileall_status" "$pytest_status" \
    "$verify_status" "$image_status" "$scan_status" >> "$SUMMARY"
  printf '{"round":%s,"started_at":"%s","ended_at":"%s","compileall":%s,"pytest":%s,"verify_benchmark":%s,"image_guard":%s,"static_scan":%s}\n' \
    "$round" "$started_at" "$ended_at" "$compileall_status" "$pytest_status" \
    "$verify_status" "$image_status" "$scan_status" >> "$STATUS_JSONL"

  if [[ "$round" -lt "$ROUNDS" ]]; then
    sleep "$INTERVAL_SECONDS"
  fi
done

rm -f "$PID_FILE"
echo "completed_at=$(date --iso-8601=seconds)" > "$RUN_DIR/completed"
