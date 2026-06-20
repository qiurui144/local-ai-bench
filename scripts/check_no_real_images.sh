#!/usr/bin/env bash
# check_no_real_images.sh — PII control for scenario fixture images.
#
# Deterministic, dependency-free (bash + python3 stdlib only). Two checks:
#   1. Provenance whitelist: every .png under fixtures/scenarios/wechat_intent/
#      must have a basename (minus .png) that is an "id" present in
#      datasets/scenarios/wechat_intent/dialogs.json — i.e. it can only be an
#      output of the synthetic renderer (scripts/render_wechat_case.py).
#   2. Reference integrity: every payload.image path referenced in
#      datasets/scenarios/wechat_intent/cases.jsonl must exist on disk.
#
# Honest limitation: this is a filename-whitelist + provenance check. It
# CANNOT detect a real screenshot that has been renamed to a whitelisted id
# (e.g. someone saving a genuine WeChat capture as c1.png). The stronger
# control for that is manual: regenerate the fixtures with the renderer and
# diff the images (renderer output is deterministic for a given dialogs.json).
#
# Exit codes: 0 = clean (PASS), 1 = violation or missing input file.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURES_DIR="${ROOT}/fixtures/scenarios/wechat_intent"
DIALOGS_JSON="${ROOT}/datasets/scenarios/wechat_intent/dialogs.json"
CASES_JSONL="${ROOT}/datasets/scenarios/wechat_intent/cases.jsonl"

fail=0

for required in "${DIALOGS_JSON}" "${CASES_JSONL}"; do
    if [[ ! -f "${required}" ]]; then
        echo "FAIL: required input missing: ${required}" >&2
        exit 1
    fi
done

# --- Check 1: fixture png basenames must be ids in dialogs.json -------------
mapfile -t whitelist < <(python3 -c '
import json, sys
for d in json.load(open(sys.argv[1], encoding="utf-8")):
    print(d["id"])
' "${DIALOGS_JSON}")

checked=0
if [[ -d "${FIXTURES_DIR}" ]]; then
    while IFS= read -r -d '' png; do
        base="$(basename "${png}" .png)"
        ok=0
        for wid in "${whitelist[@]}"; do
            if [[ "${wid}" == "${base}" ]]; then
                ok=1
                break
            fi
        done
        if [[ "${ok}" -eq 0 ]]; then
            echo "FAIL: ${png} — basename '${base}' is not an id in dialogs.json (not renderer-provenanced)" >&2
            fail=1
        fi
        checked=$((checked + 1))
    done < <(find "${FIXTURES_DIR}" -type f -name '*.png' -print0 | sort -z)
fi

# --- Check 2: every payload.image in cases.jsonl must exist -----------------
refs=0
while IFS= read -r img; do
    refs=$((refs + 1))
    if [[ ! -f "${ROOT}/${img}" ]]; then
        echo "FAIL: cases.jsonl references missing image: ${img}" >&2
        fail=1
    fi
done < <(python3 -c '
import json, sys
with open(sys.argv[1], encoding="utf-8") as fh:
    for line in fh:
        line = line.strip()
        if not line:
            continue
        img = json.loads(line).get("payload", {}).get("image")
        if img:
            print(img)
' "${CASES_JSONL}")

if [[ "${fail}" -ne 0 ]]; then
    echo "FAIL: PII fixture check found violations (see above)" >&2
    exit 1
fi

echo "PASS: ${checked} fixture png(s) all renderer-whitelisted; ${refs} cases.jsonl image reference(s) all exist"
exit 0
