import re
import subprocess
from pathlib import Path
from urllib.parse import unquote


REPO_ROOT = Path(__file__).resolve().parents[1]
LOWER_KEBAB_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
REPORT_PATH_RE = re.compile(
    r"^reports/(?:"
    r"[a-z0-9]+(?:-[a-z0-9]+)*(?:\.(?:en|zh))?\.md|"
    r"(?:selection|archive)/[a-z0-9]+(?:-[a-z0-9]+)*\.(?:en|zh)\.md|"
    r"platforms/[a-z0-9]+(?:-[a-z0-9]+)*/[a-z0-9]+(?:-[a-z0-9]+)*\.(?:en|zh)\.md|"
    r"quality-dimensions/[a-z0-9]+(?:-[a-z0-9]+)*(?:\.(?:en|zh)\.md|\.json|\.tsv)|"
    r"evidence/[a-z0-9]+(?:-[a-z0-9]+)*\.evidence\.(?:en|zh)\.md"
    r")$"
)
MARKDOWN_LINK_RE = re.compile(r"(?<!!)\[[^\]]+\]\(([^)]+)\)")
EXTERNAL_LINK_PREFIXES = ("http://", "https://", "mailto:", "app://")


def _tracked_files() -> list[str]:
    output = subprocess.check_output(["git", "ls-files"], cwd=REPO_ROOT, text=True)
    return output.splitlines()


def _portable_stem(part: str) -> str:
    stem = Path(part).stem
    for suffix in (".en", ".zh", ".schema"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def test_docs_use_lowercase_kebab_case_names():
    bad_paths = []
    for path in _tracked_files():
        if not path.startswith("docs/"):
            continue
        parts = Path(path).parts[1:]
        for part in parts:
            stem = _portable_stem(part)
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


def test_index_links_resolve_to_tracked_files():
    tracked = set(_tracked_files())
    bad_links = []
    for path in tracked:
        if not path.endswith(".md"):
            continue
        if Path(path).name not in {"index.md", "index.en.md", "index.zh.md"}:
            continue
        text = (REPO_ROOT / path).read_text(encoding="utf-8")
        for line_no, line in enumerate(text.splitlines(), 1):
            for match in MARKDOWN_LINK_RE.finditer(line):
                raw = match.group(1).strip()
                if not raw or raw.startswith(("#", *EXTERNAL_LINK_PREFIXES)):
                    continue
                target = raw.split("#", 1)[0].strip("<>")
                if not target:
                    continue
                resolved = (REPO_ROOT / path).parent.joinpath(unquote(target)).resolve()
                try:
                    rel = str(resolved.relative_to(REPO_ROOT))
                except ValueError:
                    bad_links.append((path, line_no, raw))
                    continue
                if rel not in tracked:
                    bad_links.append((path, line_no, raw))

    assert bad_links == []


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
