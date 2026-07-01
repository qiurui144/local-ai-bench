#!/usr/bin/env python3
"""Export attune model-selection catalogs from the bench SSOT.

This bench repo (models.yaml + reports/*.en.md + drivers/) is the upstream SSOT for
attune's local-base model selection (embedding / rerank / OCR / ASR / local-LLM) and
vendor driver artifacts. attune consumes the produced manifests via company-mirror
and its S8 download_with_failover flow.

Outputs two manifests under --out:
  model-catalog.yaml   — tier x role -> repo/file/engine/ep/verdict + source line refs
  driver-catalog.yaml  — tier -> vendor driver pkg name/version/sha256/url (from drivers/)

Every model-catalog entry carries `verdict` + `source` (reports file:line) per
CLAUDE.md §6.3 (数据有源). PENDING-VERIFY entries are explicitly marked.

This script ONLY reads the bench repo and writes YAML manifests — it never runs a
model or a benchmark (§1.6). Signing is out of scope here (done in CI / cloud catalog
signing endpoint with the catalog trust anchor — see spec §11 R4).

Usage:
    python scripts/export_attune_catalog.py --out dist/attune-catalog/
    python scripts/export_attune_catalog.py --matrix reports/2026-06-19-all-model-matrix-results.en.md
"""

from __future__ import annotations

import argparse
import datetime
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MATRIX = "reports/2026-06-19-all-model-matrix-results.en.md"
SCHEMA_VERSION = 1

# ── target (bench) -> attune catalog tier key ───────────────────────────────
TARGET_TO_TIER = {
    "amd-win-x86": "amd-win",
    "intel-win-x86": "intel-win",
    "intel-linux": "intel-linux",
    "jetson-agx": "nvidia-cuda",
    "k3-riscv": "riscv-k3-16g",
    "k3-riscv-16g": "riscv-k3-16g",
    "k3-riscv-8g": "riscv-k3-8g",
    "rk1820": "rk1820-npu",
    "rk3588": "rk3588-rknpu",
    # local/reference rows have no hardware target -> contribute to cpu-fallback only
    # when no measured per-target row exists for that role.
    "local/reference": "cpu-fallback",
}

# bench `Role` column prefix -> attune catalog role key.
ROLE_PREFIX_TO_ROLE = {
    "embedding": "embedding",
    "reranker": "rerank",
    "ocr": "ocr",
    "asr": "asr",
    "llm": "llm",
    "vlm": "llm",  # local VLM treated as llm-class local model in attune catalog
}

# Known per-tier OCR EP rules (bench-measured; encoded so the manifest declares intent).
# Intel DirectML OCR is unusable (CER 202%); AMD DirectML OCR is 3.4x faster than CPU.
OCR_EP_BY_TIER = {
    "amd-win": "directml",
    "intel-win": "openvino",
    "cpu-fallback": "cpu",
}


@dataclass
class Row:
    model: str
    target: str
    provider: str
    role_caps: str
    caps: str
    status: str
    verdicts: str
    metrics: str
    report: str
    line: int  # 1-based line number in the matrix report (for `source`)


def parse_matrix(matrix_path: Path) -> list[Row]:
    """Parse the markdown matrix table; capture 1-based line numbers for source refs."""
    rows: list[Row] = []
    in_table = False
    for i, raw in enumerate(matrix_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw.strip()
        if line.startswith("| Model |"):
            in_table = True
            continue
        if in_table and line.startswith("|---"):
            continue
        if in_table:
            if not line.startswith("|"):
                in_table = False
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 9:
                continue
            rows.append(
                Row(
                    model=cells[0],
                    target=cells[1],
                    provider=cells[2],
                    role_caps=cells[3],
                    caps=cells[4],
                    status=cells[5],
                    verdicts=cells[6],
                    metrics=cells[7],
                    report=cells[8],
                    line=i,
                )
            )
    return rows


def verdict_of(row: Row) -> str:
    """Map bench Status/Verdicts to attune verdict (pass/measured/pending-verify)."""
    status = row.status.upper()
    verdicts = row.verdicts.lower()
    if "fail" in verdicts:
        # A FAIL row is recorded but NOT selected; caller drops/avoids it.
        return "fail"
    if status == "PASS" or ":pass" in verdicts or "embedding:pass" in verdicts:
        return "pass"
    if status == "MEASURED" or (row.metrics not in ("", "-") and "pending" in row.metrics.lower()):
        return "measured"
    if status == "REGISTERED":
        return "pending-verify"
    if row.metrics not in ("", "-"):
        return "measured"
    return "pending-verify"


def role_of(row: Row) -> str | None:
    """Derive attune catalog role from bench Role column prefix."""
    rc = row.role_caps.lower()
    for prefix, role in ROLE_PREFIX_TO_ROLE.items():
        if rc.startswith(prefix):
            return role
    # Caps column fallback (e.g. ocr / asr / embedding listed there).
    caps = row.caps.lower()
    for prefix, role in ROLE_PREFIX_TO_ROLE.items():
        if prefix in caps:
            return role
    return None


def infer_engine_repo(row: Row, role: str) -> dict:
    """Best-effort engine/repo/model fields from the model name (declared, not guessed
    silently — name conventions in models.yaml are stable)."""
    name = row.model.lower()
    out: dict = {}
    if role == "ocr":
        out["engine"] = "ppocr" if "paddle" in name else "rapidocr"
    elif role == "asr":
        out["engine"] = "whisper" if "whisper" in name else "sensevoice"
        out["model"] = re.sub(r"-(amd|intel)-(win|linux).*$", "", row.model)
    elif role == "llm":
        out["model"] = re.sub(r"-(amd|intel|rk\d+).*$", "", row.model)
        out["engine"] = "rknn" if "rk" in name else "llama.cpp"
    elif role in ("embedding", "rerank"):
        # embedding/rerank ship as HF ONNX repos; map family -> Xenova ONNX mirror.
        if "qwen3-embedding-0.6b" in name:
            out["repo"] = "Xenova/qwen3-embedding-0.6b"
        elif "bge-m3" in name:
            out["repo"] = "Xenova/bge-m3"
        elif "bge-reranker-base" in name:
            out["repo"] = "Xenova/bge-reranker-base"
        elif "bge-reranker-v2-m3" in name:
            out["repo"] = "Xenova/bge-reranker-v2-m3"
        elif "minicpm-embed" in name:
            out["repo"] = row.model  # NPU-native, no HF ONNX mirror
        if role == "embedding":
            out["dims"] = 1024
        out.setdefault("file", "onnx/model_quantized.onnx")
    return out


def ep_for(tier: str, role: str, row: Row) -> str:
    if role == "ocr":
        return OCR_EP_BY_TIER.get(tier, "cpu")
    if tier == "amd-win":
        return "directml"
    if tier == "intel-win":
        return "openvino"
    if tier == "nvidia-cuda":
        return "cuda"
    if tier in ("rk1820-npu", "rk3588-rknpu"):
        return "rknn"
    return "cpu"


def build_model_catalog(rows: list[Row], matrix_rel: str) -> dict:
    """Pick, per (tier, role), the best PASS row (prefer pass > measured); record source."""
    # tier -> role -> (priority, entry)
    selected: dict[str, dict[str, tuple[int, dict]]] = {}
    pri = {"pass": 3, "measured": 2, "pending-verify": 1, "fail": -1}

    for row in rows:
        tier = TARGET_TO_TIER.get(row.target)
        role = role_of(row)
        if tier is None or role is None:
            continue
        v = verdict_of(row)
        if v == "fail":
            continue  # never select a FAIL row (e.g. intel directml OCR CER 202%)
        entry = {
            "verdict": v.upper().replace("_", "-"),
            "metric": row.metrics if row.metrics not in ("", "-") else "",
            "source": f"{matrix_rel}:{row.line}",
            "ep": ep_for(tier, role, row),
        }
        entry.update(infer_engine_repo(row, role))
        cur = selected.setdefault(tier, {}).get(role)
        if cur is None or pri[v] > cur[0]:
            selected[tier][role] = (pri[v], entry)

    tiers = {
        tier: {role: e for role, (_, e) in roles.items()}
        for tier, roles in sorted(selected.items())
    }
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_at": datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "harness_version": _harness_version(),
        "source_repo": "local-ai-bench",
        "tiers": tiers,
    }


def _harness_version() -> str:
    rel = REPO_ROOT / "RELEASE.md"
    if rel.exists():
        m = re.search(r"v?(\d+\.\d+\.\d+)", rel.read_text(encoding="utf-8")[:2000])
        if m:
            return m.group(1)
    return "unknown"


def build_driver_catalog() -> dict:
    """Scan drivers/<tier>/ for vendor packages -> driver-catalog (url + sha256 PENDING).

    sha256 left PENDING (we do not hash large vendor blobs in this script; CI fills them
    at publish time). url points at company-mirror attune-drivers/ path (R5: we publish
    only the official download URL + sha256, not re-host vendor binaries unless licensed)."""
    drivers_root = REPO_ROOT / "drivers"
    tier_by_dir = {
        "amd-win": "amd-npu-win",
        "intel-win": "intel-npu-win",
        "rk182x-linux": "rk1820-npu",
        "rk3588-linux": "rk3588-rknpu",
    }
    skip_ext = {".pdf"}
    tiers: dict[str, list] = {}
    if drivers_root.exists():
        for sub in sorted(drivers_root.iterdir()):
            if not sub.is_dir():
                continue
            tier = tier_by_dir.get(sub.name, sub.name)
            pkgs = []
            for f in sorted(sub.iterdir()):
                if f.is_dir() or f.suffix.lower() in skip_ext:
                    continue
                pkgs.append(
                    {
                        "name": f.name,
                        "file": f.name,
                        "sha256": "PENDING",  # CI fills at publish
                        "url": f"{{mirror}}/attune-drivers/{sub.name}/{f.name}",
                        "note": "vendor NPU driver/SDK; official source — verify license before re-host (spec §11 R5)",
                    }
                )
            if pkgs:
                tiers[tier] = pkgs
    return {"schema_version": SCHEMA_VERSION, "tiers": tiers}


def main() -> int:
    ap = argparse.ArgumentParser(description="Export attune model + driver catalogs from bench SSOT.")
    ap.add_argument("--matrix", default=DEFAULT_MATRIX, help="path to the matrix report markdown")
    ap.add_argument("--out", default="dist/attune-catalog", help="output dir for the two manifests")
    args = ap.parse_args()

    matrix_path = (REPO_ROOT / args.matrix).resolve()
    if not matrix_path.exists():
        print(f"ERROR: matrix report not found: {matrix_path}")
        return 2
    matrix_rel = str(matrix_path.relative_to(REPO_ROOT))

    rows = parse_matrix(matrix_path)
    model_catalog = build_model_catalog(rows, matrix_rel)
    driver_catalog = build_driver_catalog()

    out_dir = (REPO_ROOT / args.out).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "model-catalog.yaml").write_text(
        yaml.safe_dump(model_catalog, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (out_dir / "driver-catalog.yaml").write_text(
        yaml.safe_dump(driver_catalog, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )

    n_tiers = len(model_catalog["tiers"])
    n_entries = sum(len(v) for v in model_catalog["tiers"].values())
    print(f"wrote {out_dir}/model-catalog.yaml ({n_tiers} tiers, {n_entries} role entries)")
    print(f"wrote {out_dir}/driver-catalog.yaml ({len(driver_catalog['tiers'])} driver tiers)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
