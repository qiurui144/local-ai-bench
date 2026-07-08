#!/usr/bin/env python3
"""Cache public long-context suite data used by the edge harness."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import urllib.request
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LONG_CONTEXT_ROOT = ROOT / "drivers/long-context-suites"
LONGBENCH_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
AIRPLANE_MANUAL_REPO = "https://github.com/shiroinekotfs/airplane-manual-collection.git"
AIRPLANE_MANUAL_HEAD = "afe8288495338880e165f77bb9afe9946f366a52"

AIRPLANE_MANUAL_CORE = [
    "Airbus/A350/FDS Briefing/a350-900-flight-deck-and-systems-briefing-for-pilots.pdf",
    "Airbus/A220/QRH/a220-300-cs300-bd500-1a11-quick-reference-handbook.pdf",
    "Boeing/B737/FCOM/737MAX FCOM.pdf",
    "Boeing/B737/QRH/B737-700 Quick Reference Handbook (QRH).pdf",
    "Boeing/B737/FCTM/B737 Flight Crew Training Manual - All.pdf",
]

AIRPLANE_MANUAL_BROAD = [
    *AIRPLANE_MANUAL_CORE,
    "Boeing/B747/FCOM/747-400_FCOM.pdf",
    "Boeing/B747/QRH/747-400_QRH.pdf",
    "Boeing/B787/FCOM/787-tbc_om_tbc_c_100215_v1v2_b2p-c.pdf",
    "Boeing/B787/QRH/787-TBC_OM_TBC_C_100215_QRH_B2P-C.pdf",
    "Airbus/General/Others/airbus_abbreviations.pdf",
]


def cache_longbench(force: bool = False) -> Path:
    repo = LONG_CONTEXT_ROOT / "LongBench"
    data_dir = repo / "data"
    if data_dir.exists() and any(data_dir.glob("*.jsonl")) and not force:
        return data_dir
    repo.mkdir(parents=True, exist_ok=True)
    zip_path = repo / "data.zip"
    if force or not zip_path.exists():
        urllib.request.urlretrieve(LONGBENCH_URL, zip_path)
    tmp = repo / "_data_extract"
    if tmp.exists():
        shutil.rmtree(tmp)
    tmp.mkdir(parents=True)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(tmp)
    extracted = tmp / "data"
    if not extracted.exists():
        candidates = [p for p in tmp.rglob("*.jsonl")]
        if not candidates:
            raise RuntimeError(f"no LongBench jsonl files found in {zip_path}")
        extracted = candidates[0].parent
    if data_dir.exists():
        shutil.rmtree(data_dir)
    shutil.move(str(extracted), str(data_dir))
    shutil.rmtree(tmp)
    return data_dir


def _run(cmd: list[str], cwd: Path | None = None) -> str:
    proc = subprocess.run(
        cmd,
        cwd=cwd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )
    return proc.stdout


def _safe_name(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", path).strip("_")


def _normalize_text(raw: str) -> str:
    raw = raw.replace("\x00", " ").replace("\r\n", "\n").replace("\r", "\n")
    raw = re.sub(r"[ \t]+", " ", raw)
    raw = re.sub(r"\n{4,}", "\n\n\n", raw)
    return raw.strip()


def _text_lines(text: str) -> list[str]:
    lines = []
    for line in text.splitlines():
        line = re.sub(r"\s+", " ", line).strip()
        if 40 <= len(line) <= 220 and re.search(r"[A-Za-z]", line):
            lines.append(line)
    return lines


def _pick_span(lines: list[str], seed: int) -> tuple[str, str]:
    if len(lines) < 2:
        raise RuntimeError("not enough extracted text lines for span case")
    idx = min(len(lines) - 2, max(0, int(len(lines) * seed / 10)))
    return lines[idx], lines[idx + 1]


def _pick_keywords(text: str, seed: int) -> tuple[str, list[str], int]:
    paragraphs: list[tuple[str, str, int]] = []
    for match in re.finditer(r"(?ms)(?:^|\n\s*\n)(.*?)(?=\n\s*\n|$)", text):
        raw = match.group(1)
        stripped = raw.strip()
        normalized = re.sub(r"\s+", " ", stripped)
        if 180 <= len(normalized) <= 900:
            offset = match.start(1) + len(raw) - len(raw.lstrip())
            paragraphs.append((stripped, normalized, offset))
    if not paragraphs:
        lines = _text_lines(text)
        for i in range(0, max(0, len(lines) - 5), 5):
            paragraph = " ".join(lines[i:i + 5])
            paragraphs.append((paragraph, paragraph, max(0, text.find(lines[i]))))
    if not paragraphs:
        raise RuntimeError("not enough extracted text for keyword case")
    paragraph, normalized, offset = paragraphs[min(len(paragraphs) - 1, max(0, int(len(paragraphs) * seed / 10)))]
    candidates = re.findall(r"\b[A-Z][A-Z0-9/-]{2,}\b", normalized)
    seen = set()
    keywords = []
    for item in candidates:
        if item not in seen:
            seen.add(item)
            keywords.append(item)
        if len(keywords) >= 8:
            break
    if len(keywords) < 3:
        tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9/-]{4,}\b", normalized)
        for item in tokens:
            key = item.upper()
            if key not in seen:
                seen.add(key)
                keywords.append(item)
            if len(keywords) >= 8:
                break
    if len(keywords) < 3:
        raise RuntimeError("not enough stable keywords for keyword case")
    return paragraph, keywords, offset


def _pdf_pages(path: Path) -> int | None:
    try:
        out = _run(["pdfinfo", str(path)])
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    for line in out.splitlines():
        if line.startswith("Pages:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def _airplane_repo(force: bool = False) -> Path:
    repo = LONG_CONTEXT_ROOT / "airplane-manual-collection"
    if force and repo.exists():
        shutil.rmtree(repo)
    if not (repo / ".git").exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        _run([
            "git", "clone", "--filter=blob:none", "--no-checkout", "--depth", "1",
            AIRPLANE_MANUAL_REPO, str(repo),
        ])
    else:
        _run(["git", "fetch", "--depth", "1", "origin", "master"], cwd=repo)
    return repo


def _airplane_manual_paths(scope: str, explicit: str = "", max_files: int = 0) -> list[str]:
    if explicit:
        paths = [item.strip() for item in explicit.split(",") if item.strip()]
    elif scope == "core":
        paths = list(AIRPLANE_MANUAL_CORE)
    elif scope == "broad":
        paths = list(AIRPLANE_MANUAL_BROAD)
    elif scope == "all":
        repo = _airplane_repo()
        paths = [
            p for p in _run(["git", "ls-tree", "-r", "--name-only", "HEAD"], cwd=repo).splitlines()
            if p.lower().endswith(".pdf")
        ]
    else:
        raise ValueError(f"unknown airplane manual scope: {scope}")
    if max_files > 0:
        return paths[:max_files]
    return paths


def cache_airplane_manuals(
    *,
    scope: str = "core",
    paths_csv: str = "",
    max_files: int = 0,
    force: bool = False,
) -> Path:
    repo = _airplane_repo(force=force)
    selected = _airplane_manual_paths(scope, paths_csv, max_files)
    if not selected:
        raise RuntimeError("no airplane manual PDF paths selected")

    _run(["git", "checkout", "HEAD", "--", *selected], cwd=repo)

    extracted_dir = repo / "extracted_text"
    cases_dir = repo / "cases"
    extracted_dir.mkdir(parents=True, exist_ok=True)
    cases_dir.mkdir(parents=True, exist_ok=True)
    cases_path = cases_dir / "aviation_manual_cases.jsonl"
    if cases_path.exists():
        cases_path.unlink()

    skipped: list[str] = []
    written = 0
    with cases_path.open("w", encoding="utf-8") as out:
        for index, rel in enumerate(selected):
            pdf_path = repo / rel
            text_path = extracted_dir / f"{_safe_name(rel)}.txt"
            try:
                if force or not text_path.exists():
                    _run(["pdftotext", "-layout", "-enc", "UTF-8", str(pdf_path), str(text_path)])
                    text_path.write_text(_normalize_text(text_path.read_text(encoding="utf-8", errors="replace")),
                                         encoding="utf-8")
            except (FileNotFoundError, subprocess.CalledProcessError) as exc:
                skipped.append(f"{rel}: pdftotext failed: {exc}")
                continue
            text = text_path.read_text(encoding="utf-8", errors="replace")
            lines = _text_lines(text)
            try:
                anchor, answer = _pick_span(lines, 3 + (index % 4))
                paragraph, keywords, keyword_offset = _pick_keywords(text, 5 + (index % 3))
            except RuntimeError as exc:
                skipped.append(f"{rel}: {exc}")
                continue
            manual_id = _safe_name(rel).removesuffix(".pdf")
            common = {
                "manual_id": manual_id,
                "source_path": rel,
                "source_url": f"https://github.com/shiroinekotfs/airplane-manual-collection/blob/master/{rel.replace(' ', '%20')}",
                "repo_commit": AIRPLANE_MANUAL_HEAD,
                "text_path": str(text_path.relative_to(repo)),
                "pdf_pages": _pdf_pages(pdf_path),
                "text_chars": len(text),
            }
            out.write(json.dumps({
                **common,
                "case_id": f"{manual_id}:span",
                "case_type": "span_recall",
                "anchor": anchor,
                "answer": answer,
                "offset": max(0, text.find(anchor)),
            }, ensure_ascii=False) + "\n")
            written += 1
            out.write(json.dumps({
                **common,
                "case_id": f"{manual_id}:keywords",
                "case_type": "keyword_recall",
                "paragraph": paragraph,
                "keywords": keywords,
                "offset": keyword_offset,
            }, ensure_ascii=False) + "\n")
            written += 1
            out.write(json.dumps({
                **common,
                "case_id": f"{manual_id}:needle",
                "case_type": "manual_needle",
            }, ensure_ascii=False) + "\n")
            written += 1

    if written == 0:
        raise RuntimeError("no usable aviation manual cases generated; skipped: " + "; ".join(skipped))

    skipped_path = cases_dir / "skipped.txt"
    skipped_path.write_text("\n".join(skipped) + ("\n" if skipped else ""), encoding="utf-8")

    readme = cases_dir / "README.md"
    readme.write_text(
        "# Aviation Manual Long-Context Cases\n\n"
        "Generated from shiroinekotfs/airplane-manual-collection for local benchmark use. "
        "The upstream repository states that materials may be unverified for real flight use; "
        "these cases are for benchmark quality/risk analysis only.\n",
        encoding="utf-8",
    )
    return cases_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-longbench", action="store_true")
    parser.add_argument("--airplane-manuals", action="store_true",
                        help="Cache selected aviation manuals and generate long-context cases.")
    parser.add_argument("--airplane-scope", choices=("core", "broad", "all"), default="core")
    parser.add_argument("--airplane-paths", default="",
                        help="Comma-separated PDF paths inside airplane-manual-collection.")
    parser.add_argument("--airplane-max-files", type=int, default=0)
    args = parser.parse_args()
    if not args.skip_longbench:
        data_dir = cache_longbench(force=args.force)
        files = sorted(p.name for p in data_dir.glob("*.jsonl"))
        print(f"LongBench data: {data_dir}")
        print(f"jsonl files: {len(files)}")
        for name in files[:20]:
            print(f"- {name}")
    if args.airplane_manuals:
        cases_path = cache_airplane_manuals(
            scope=args.airplane_scope,
            paths_csv=args.airplane_paths,
            max_files=args.airplane_max_files,
            force=args.force,
        )
        rows = [line for line in cases_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        print(f"Airplane manual cases: {cases_path}")
        print(f"cases: {len(rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
