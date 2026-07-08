import re
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWER_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REPORT_PATH_RE = re.compile(
    r"^reports/(?:"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.(?:en|zh))?\.md|"
    r"(?:selection|archive)/[a-z0-9]+(?:-[a-z0-9]+)*\.(?:en|zh)\.md|"
    r"platforms/[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*\.(?:en|zh)\.md|"
    r"evidence/[a-z0-9]+(?:-[a-z0-9]+)*\.evidence\.(?:en|zh)\.md"
    r")$"
)


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return output.splitlines()


def _markdown_stem(part: str) -> str:
    if not part.endswith(".md"):
        return part
    stem = part[:-3]
    for suffix in (".en", ".zh"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def test_docs_use_lowercase_kebab_case_names():
    bad_paths = []
    for path in _tracked_files():
        if not path.startswith("docs/"):
            continue
        parts = Path(path).parts[1:]
        for part in parts:
            stem = _markdown_stem(part)
            if not LOWER_KEBAB_RE.fullmatch(stem):
                bad_paths.append(path)
                break

    assert bad_paths == []


def test_public_reports_use_fixed_lowercase_names():
    bad_paths = []
    for path in _tracked_files():
        if not path.startswith("reports/"):
            continue
        if not REPORT_PATH_RE.fullmatch(path):
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
