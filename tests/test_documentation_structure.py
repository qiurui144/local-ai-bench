import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWER_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
PUBLIC_REPORT_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*(?:\.(?:en|zh))?\.md$")


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return output.splitlines()


def test_docs_use_lowercase_kebab_case_names():
    bad_paths = []
    for path in _tracked_files():
        if not path.startswith("docs/"):
            continue
        parts = Path(path).parts[1:]
        for part in parts:
            stem = part[:-3] if part.endswith(".md") else part
            if not LOWER_KEBAB_RE.fullmatch(stem):
                bad_paths.append(path)
                break

    assert bad_paths == []


def test_public_reports_use_fixed_lowercase_names():
    bad_paths = []
    for path in _tracked_files():
        if not path.startswith("reports/"):
            continue
        rel = Path(path)
        if len(rel.parts) != 2 or not PUBLIC_REPORT_RE.fullmatch(rel.name):
            bad_paths.append(path)

    assert bad_paths == []


def test_no_tracked_local_report_records():
    bad_prefixes = ("reports/runs/",)
    bad_paths = []
    for path in _tracked_files():
        if path.startswith(bad_prefixes) or re.match(r"^reports/20[0-9]{2}[-/]", path):
            bad_paths.append(path)

    assert bad_paths == []


def test_tracked_markdown_is_utf8_compatible():
    bad_paths = []
    for path in _tracked_files():
        if not path.endswith(".md"):
            continue
        try:
            (REPO_ROOT / path).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            bad_paths.append(path)

    assert bad_paths == []
