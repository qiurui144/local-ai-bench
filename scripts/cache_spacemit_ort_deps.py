#!/usr/bin/env python3
"""Cache SpacemiT ONNX Runtime CMake dependencies for offline builds."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import sys
import tempfile
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DEPS = ROOT / "drivers/spacemit-source/onnxruntime/cmake/deps.txt"
DEFAULT_MIRROR = ROOT / "drivers/ort-cmake-deps-mirror"


@dataclass(frozen=True)
class Dependency:
    name: str
    url: str
    sha1: str

    @property
    def mirror_relative_path(self) -> Path:
        if not self.url.startswith("https://"):
            return Path(self.url)
        return Path(self.url.removeprefix("https://"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download SpacemiT ONNX Runtime CMake dependencies into a local CMake mirror."
    )
    parser.add_argument("--deps", type=Path, default=DEFAULT_DEPS, help="Path to onnxruntime/cmake/deps.txt.")
    parser.add_argument("--mirror", type=Path, default=DEFAULT_MIRROR, help="CMake dependency mirror root.")
    parser.add_argument("--jobs", type=int, default=min(8, (os.cpu_count() or 4)), help="Parallel downloads.")
    parser.add_argument("--timeout", type=int, default=120, help="Per-request timeout in seconds.")
    parser.add_argument("--retries", type=int, default=3, help="Retry count per URL candidate.")
    parser.add_argument("--force", action="store_true", help="Redownload files even when the SHA1 already matches.")
    parser.add_argument("--manifest", action="store_true", help="Print the resolved dependency manifest and exit.")
    parser.add_argument("--download-missing", action="store_true",
                        help="Download dependencies that are absent or fail SHA1 validation.")
    parser.add_argument("--limit", help="Comma-separated dependency names to download.")
    return parser.parse_args()


def read_deps(path: Path) -> list[Dependency]:
    deps: list[Dependency] = []
    with path.open("r", encoding="utf-8", newline="") as file:
        for row in csv.reader(file, delimiter=";"):
            if len(row) != 3 or row[0].startswith("#"):
                continue
            deps.append(Dependency(name=row[0], url=row[1], sha1=row[2]))
    return deps


def github_url_from_nexus_url(url: str) -> str | None:
    marker = "/onnxruntime_build_deps/"
    if marker not in url:
        return None
    encoded = url.split(marker, 1)[1]
    prefix = "github.com_"
    if not encoded.startswith(prefix):
        return None
    payload = encoded.removeprefix(prefix)

    if "_archive_refs_tags_" in payload:
        repo_part, tag_file = payload.split("_archive_refs_tags_", 1)
        owner, repo = repo_part.split("_", 1)
        return f"https://github.com/{owner}/{repo}/archive/refs/tags/{tag_file}"

    if "_archive_" in payload:
        repo_part, ref_file = payload.split("_archive_", 1)
        owner, repo = repo_part.split("_", 1)
        commit_named = re_match_commit_named(ref_file)
        if commit_named:
            ref, filename = commit_named
            return f"https://github.com/{owner}/{repo}/archive/{ref}/{filename}"
        return f"https://github.com/{owner}/{repo}/archive/{ref_file}"

    if "_releases_download_" in payload:
        repo_part, release_part = payload.split("_releases_download_", 1)
        owner, repo = repo_part.split("_", 1)
        tag, filename = release_part.split("_", 1)
        return f"https://github.com/{owner}/{repo}/releases/download/{tag}/{filename}"

    return None


def re_match_commit_named(ref_file: str) -> tuple[str, str] | None:
    if len(ref_file) <= 41 or ref_file[40] != "_":
        return None
    ref = ref_file[:40]
    if all(ch in "0123456789abcdefABCDEF" for ch in ref):
        return ref, ref_file[41:]
    return None


def sha1sum(path: Path) -> str:
    digest = hashlib.sha1()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_url(url: str, dest: Path, timeout: int) -> None:
    request = urllib.request.Request(url, headers={"User-Agent": "vlm-llm-benchmark/ort-cache"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        dest.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(delete=False, dir=dest.parent) as tmp:
            tmp_path = Path(tmp.name)
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                tmp.write(chunk)
    tmp_path.replace(dest)


def cache_one(dep: Dependency, mirror: Path, timeout: int, retries: int, force: bool) -> tuple[str, str]:
    dest = mirror / dep.mirror_relative_path
    if dest.exists() and not force:
        actual = sha1sum(dest)
        if actual == dep.sha1:
            return dep.name, "cached"

    upstream = github_url_from_nexus_url(dep.url)
    candidates = [candidate for candidate in (upstream, dep.url) if candidate]
    errors: list[str] = []
    for candidate in candidates:
        for attempt in range(1, retries + 1):
            try:
                download_url(candidate, dest, timeout)
                actual = sha1sum(dest)
                if actual != dep.sha1:
                    raise ValueError(f"sha1 mismatch: expected {dep.sha1}, got {actual}")
                return dep.name, "downloaded"
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{candidate} attempt {attempt}: {exc}")
                if dest.exists():
                    dest.unlink()

    return dep.name, "failed: " + " | ".join(errors[-3:])


def print_manifest(deps: list[Dependency], mirror: Path) -> None:
    print("name\tmirror_path\tsha1\tupstream_url\toriginal_url")
    for dep in deps:
        print(
            f"{dep.name}\t{mirror / dep.mirror_relative_path}\t{dep.sha1}\t"
            f"{github_url_from_nexus_url(dep.url) or ''}\t{dep.url}"
        )


def main() -> int:
    args = parse_args()
    deps = read_deps(args.deps)
    if args.limit:
        wanted = {item.strip() for item in args.limit.split(",") if item.strip()}
        deps = [dep for dep in deps if dep.name in wanted]

    if args.manifest:
        print_manifest(deps, args.mirror)
        return 0

    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {
            pool.submit(cache_one, dep, args.mirror, args.timeout, args.retries, args.force): dep.name
            for dep in deps
        }
        for future in as_completed(futures):
            name, status = future.result()
            print(f"{name}\t{status}", flush=True)
            if status.startswith("failed:"):
                failures.append(name)

    if failures:
        print("failed dependencies: " + ", ".join(sorted(failures)), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
