#!/usr/bin/env python3
"""Cache Rockchip RKNN3 model-zoo artifacts listed in models.yaml."""

from __future__ import annotations

import argparse
import http.cookiejar
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CACHE_ROOT = ROOT / "drivers/rockchip-rknn3-models"
MODEL_ZOO_PREFIX = "/RKNN3_SDK/rknn3_models/"
ARTIFACT_FIELDS = ("rknn_model_path", "rknn_vision_model_path")
LENOVO_SHARE_PREFIX = "rknn3_models"


@dataclass(frozen=True)
class Artifact:
    model_name: str
    field: str
    category: str
    device_path: str
    expected_size: int = 0
    source_url: str = ""

    @property
    def relative_path(self) -> Path:
        path = self.device_path
        if path.startswith(MODEL_ZOO_PREFIX):
            path = path.removeprefix(MODEL_ZOO_PREFIX)
        return Path(path.lstrip("/"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build or populate a local cache for RKNN3 model-zoo .rknn files."
    )
    parser.add_argument("--models-yaml", type=Path, default=ROOT / "models.yaml")
    parser.add_argument("--cache-root", type=Path, default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--scope", choices=("all", "llm", "vlm", "others"), default="all")
    parser.add_argument("--models", default="", help="Comma-separated model names to include.")
    parser.add_argument("--manifest", action="store_true", help="Print artifact manifest and exit.")
    parser.add_argument("--check-cache", action="store_true", help="Report cached/missing artifacts and exit.")
    parser.add_argument("--download-missing", action="store_true", help="Populate missing artifacts.")
    parser.add_argument("--force", action="store_true", help="Replace existing cached files.")
    parser.add_argument("--jobs", type=int, default=int(os.environ.get("ROCKCHIP_RKNN3_DOWNLOAD_JOBS", "1")),
                        help="Parallel download jobs for Lenovo Filez tree downloads.")
    parser.add_argument("--lenovo-tree-root", default=os.environ.get("ROCKCHIP_RKNN3_LENOVO_TREE_ROOT", ""),
                        help="Enumerate this Filez tree relative to rknn3_models, e.g. v1.0.4.")
    parser.add_argument("--from-url-base", default=os.environ.get("ROCKCHIP_RKNN3_BASE_URL", ""),
                        help="HTTP mirror root for rknn3_models, e.g. https://mirror/rknn3_models.")
    parser.add_argument("--from-lenovo-share-url", default=os.environ.get("ROCKCHIP_RKNN3_LENOVO_SHARE_URL", ""),
                        help="Lenovo Filez share URL that contains RKNN3_SDK.")
    parser.add_argument("--from-lenovo-password", default=os.environ.get("ROCKCHIP_RKNN3_LENOVO_PASSWORD", ""),
                        help="Extraction code for --from-lenovo-share-url.")
    parser.add_argument("--from-local-root", type=Path,
                        default=Path(os.environ["ROCKCHIP_RKNN3_LOCAL_ROOT"])
                        if os.environ.get("ROCKCHIP_RKNN3_LOCAL_ROOT") else None,
                        help="Local root that contains v1.0.4/... .rknn files.")
    parser.add_argument("--from-ssh-root", default=os.environ.get("ROCKCHIP_RKNN3_SSH_ROOT", MODEL_ZOO_PREFIX.rstrip("/")),
                        help="Remote root that contains v1.0.4/... .rknn files.")
    parser.add_argument("--ssh-host", default=os.environ.get("RK3588_HOST", ""))
    parser.add_argument("--ssh-user", default=os.environ.get("RK3588_USER", ""))
    parser.add_argument("--sync-to-device", action="store_true",
                        help="Copy cached artifacts to --device-root on the RK target.")
    parser.add_argument("--device-root", default=MODEL_ZOO_PREFIX.rstrip("/"))
    return parser.parse_args()


def load_artifacts(models_yaml: Path, *, scope: str, models_csv: str) -> list[Artifact]:
    data = yaml.safe_load(models_yaml.read_text(encoding="utf-8"))
    wanted = {item.strip() for item in models_csv.split(",") if item.strip()}
    artifacts: list[Artifact] = []
    for model in data.get("models", []):
        name = str(model.get("name", ""))
        if wanted and name not in wanted:
            continue
        if model.get("target") not in {"rk182x-linux", "rk3588-linux"}:
            continue
        for field in ARTIFACT_FIELDS:
            path = model.get(field)
            if not path:
                continue
            rel_parts = Artifact(name, field, "", str(path)).relative_path.parts
            category = rel_parts[1] if len(rel_parts) > 1 and rel_parts[0].startswith("v") else rel_parts[0]
            if scope != "all" and category != scope:
                continue
            artifacts.append(Artifact(name, field, category, str(path)))
    return sorted(set(artifacts), key=lambda a: (str(a.relative_path), a.model_name, a.field))


def _json_from_response(response) -> dict:
    raw = response.read().decode("utf-8", errors="replace")
    return json.loads(raw)


class LenovoFilezClient:
    def __init__(self, share_url: str, password: str) -> None:
        if not share_url or not password:
            raise RuntimeError("set ROCKCHIP_RKNN3_LENOVO_SHARE_URL and ROCKCHIP_RKNN3_LENOVO_PASSWORD")
        self.share_url = share_url
        self.password = password
        self.cookie_jar = http.cookiejar.CookieJar()
        self.opener = urllib.request.build_opener(
            urllib.request.HTTPCookieProcessor(self.cookie_jar),
            urllib.request.HTTPRedirectHandler(),
        )
        self.page_url = ""
        self.base_url = ""
        self.share_id = ""
        self.token = ""

    def open(self) -> None:
        with self.opener.open(self.share_url, timeout=60) as response:
            self.page_url = response.geturl()
            html = response.read().decode("utf-8", errors="replace")
        parsed = urllib.parse.urlparse(self.page_url)
        self.base_url = f"{parsed.scheme}://{parsed.netloc}"
        match = re.search(r"var\s+linkInfo\s*=\s*(\{.*?\});", html, flags=re.S)
        if match:
            info = json.loads(match.group(1))
            self.share_id = str(info.get("_id") or "")
        if not self.share_id:
            parts = [part for part in parsed.path.split("/") if part]
            if len(parts) >= 3 and parts[-2] == "view":
                self.share_id = parts[-1]
        if not self.share_id:
            raise RuntimeError("could not discover Lenovo Filez share id")
        data = urllib.parse.urlencode({"password": self.password}).encode()
        req = urllib.request.Request(
            f"{self.base_url}/v2/delivery/auth/{self.share_id}",
            data=data,
            headers={
                "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
                "X-Requested-With": "XMLHttpRequest",
                "Referer": self.page_url,
            },
            method="POST",
        )
        with self.opener.open(req, timeout=60) as response:
            auth = _json_from_response(response)
        self.token = str(auth.get("token") or (auth.get("data") or {}).get("token") or "")
        if not self.token:
            raise RuntimeError(f"Lenovo Filez auth failed: {auth.get('message') or auth}")

    def metadata(self, artifact: Artifact) -> dict:
        rel = Path(LENOVO_SHARE_PREFIX) / artifact.relative_path
        return self.metadata_for_share_rel(rel)

    def metadata_for_share_rel(self, rel: Path) -> dict:
        quoted = "/".join(urllib.parse.quote(part) for part in rel.parts)
        query = urllib.parse.urlencode({
            "token": self.token,
            "orderby": "name",
            "sort": "asc",
            "offset": 0,
            "limit": 1000,
        })
        req = urllib.request.Request(
            f"{self.base_url}/v2/delivery/metadata/{self.share_id}/{quoted}?{query}",
            headers={"X-Requested-With": "XMLHttpRequest", "Referer": self.page_url},
        )
        with self.opener.open(req, timeout=60) as response:
            meta = _json_from_response(response)
        if meta.get("message"):
            raise FileNotFoundError(f"{rel}: {meta.get('message')}")
        return meta

    def list_tree(self, tree_root: str) -> list[Artifact]:
        rel_root = Path(LENOVO_SHARE_PREFIX) / tree_root.strip("/")
        artifacts: list[Artifact] = []

        def walk(share_rel: Path) -> None:
            meta = self.metadata_for_share_rel(share_rel)
            if not meta.get("is_dir"):
                artifacts.append(self._artifact_from_meta(meta))
                return
            for item in meta.get("content") or []:
                path = str(item.get("path") or "")
                if not path:
                    continue
                item_rel = Path(LENOVO_SHARE_PREFIX) / path.removeprefix(MODEL_ZOO_PREFIX).lstrip("/")
                if item.get("is_dir"):
                    walk(item_rel)
                else:
                    artifacts.append(self._artifact_from_meta(item))

        walk(rel_root)
        return sorted(artifacts, key=lambda a: str(a.relative_path))

    def _artifact_from_meta(self, meta: dict) -> Artifact:
        path = str(meta.get("path") or "")
        if not path.startswith(MODEL_ZOO_PREFIX):
            raise RuntimeError(f"unexpected Filez path: {path}")
        rel = Path(path.removeprefix(MODEL_ZOO_PREFIX).lstrip("/"))
        parts = rel.parts
        category = parts[1] if len(parts) > 1 and parts[0].startswith("v") else (parts[0] if parts else "")
        model_name = f"filez:{parts[-2] if len(parts) >= 2 else rel.stem}"
        return Artifact(
            model_name=model_name,
            field="filez_path",
            category=category,
            device_path=path,
            expected_size=int(meta.get("bytes") or 0),
            source_url=str(meta.get("download_url") or ""),
        )

    def download(self, artifact: Artifact, cache_root: Path, force: bool) -> bool:
        dst = cache_root / artifact.relative_path
        if _is_cached(artifact, cache_root) and not force:
            return True
        if artifact.source_url:
            url = artifact.source_url
            expected_size = artifact.expected_size
        else:
            meta = self.metadata(artifact)
            if meta.get("is_dir"):
                raise IsADirectoryError(str(artifact.relative_path))
            url = meta.get("download_url")
            expected_size = int(meta.get("bytes") or 0)
        if not url:
            raise RuntimeError(f"missing download_url for {artifact.relative_path}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".part")
        req = urllib.request.Request(str(url), headers={"Referer": self.page_url})
        opener = urllib.request.urlopen if artifact.source_url else self.opener.open
        with opener(req, timeout=900) as response, tmp.open("wb") as file:
            shutil.copyfileobj(response, file)
        if expected_size and tmp.stat().st_size != expected_size:
            raise RuntimeError(
                f"size mismatch for {artifact.relative_path}: got {tmp.stat().st_size}, expected {expected_size}"
            )
        tmp.replace(dst)
        return True


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def print_manifest(artifacts: list[Artifact], cache_root: Path) -> None:
    print("model\tartifact\tcategory\texpected_size\trelative_path\tcache_path\tdevice_path")
    for artifact in artifacts:
        rel = artifact.relative_path
        print(
            f"{artifact.model_name}\t{artifact.field}\t{artifact.category}\t{artifact.expected_size}\t{rel}\t"
            f"{cache_root / rel}\t{artifact.device_path}"
        )


def _is_cached(artifact: Artifact, cache_root: Path) -> bool:
    path = cache_root / artifact.relative_path
    if not path.exists() or path.stat().st_size <= 0:
        return False
    return artifact.expected_size <= 0 or path.stat().st_size == artifact.expected_size


def cache_status(artifacts: list[Artifact], cache_root: Path) -> tuple[list[Artifact], list[Artifact]]:
    cached = []
    missing = []
    for artifact in artifacts:
        if _is_cached(artifact, cache_root):
            cached.append(artifact)
        else:
            missing.append(artifact)
    return cached, missing


def write_index(artifacts: list[Artifact], cache_root: Path) -> None:
    cache_root.mkdir(parents=True, exist_ok=True)
    index = cache_root / "cache-index.tsv"
    with index.open("w", encoding="utf-8") as file:
        file.write("size\tsha256\trelative_path\n")
        for artifact in artifacts:
            path = cache_root / artifact.relative_path
            if path.exists() and path.stat().st_size > 0:
                file.write(f"{path.stat().st_size}\t{sha256sum(path)}\t{artifact.relative_path}\n")


def copy_from_local(artifact: Artifact, source_root: Path, cache_root: Path, force: bool) -> bool:
    src = source_root / artifact.relative_path
    dst = cache_root / artifact.relative_path
    if _is_cached(artifact, cache_root) and not force:
        return True
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def download_from_url(artifact: Artifact, base_url: str, cache_root: Path, force: bool) -> bool:
    dst = cache_root / artifact.relative_path
    if _is_cached(artifact, cache_root) and not force:
        return True
    quoted = "/".join(urllib.parse.quote(part) for part in artifact.relative_path.parts)
    url = base_url.rstrip("/") + "/" + quoted
    dst.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=600) as response:
        tmp = dst.with_suffix(dst.suffix + ".part")
        with tmp.open("wb") as file:
            shutil.copyfileobj(response, file)
        tmp.replace(dst)
    return True


def _ssh_base(args: argparse.Namespace) -> list[str]:
    if not args.ssh_host or not args.ssh_user:
        raise RuntimeError("set RK3588_HOST and RK3588_USER, or pass --ssh-host/--ssh-user")
    opts = [
        "-o", "StrictHostKeyChecking=no",
        "-o", "UserKnownHostsFile=/dev/null",
        "-o", "ConnectTimeout=10",
    ]
    password = os.environ.get("SSHPASS") or os.environ.get("RK3588_PASS")
    if password:
        os.environ["SSHPASS"] = password
        return ["sshpass", "-e", "ssh", *opts]
    return ["ssh", "-o", "BatchMode=yes", *opts]


def _scp_base(args: argparse.Namespace) -> list[str]:
    ssh = _ssh_base(args)
    if "ssh" in ssh:
        idx = ssh.index("ssh")
        return [*ssh[:idx], "scp", *ssh[idx + 1:]]
    return ["scp"]


def copy_from_ssh(artifact: Artifact, args: argparse.Namespace, cache_root: Path, force: bool) -> bool:
    dst = cache_root / artifact.relative_path
    if _is_cached(artifact, cache_root) and not force:
        return True
    dst.parent.mkdir(parents=True, exist_ok=True)
    remote_path = f"{args.from_ssh_root.rstrip('/')}/{artifact.relative_path}"
    remote = f"{args.ssh_user}@{args.ssh_host}:{remote_path}"
    proc = subprocess.run([*_scp_base(args), remote, str(dst)], text=True,
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return proc.returncode == 0


def sync_to_device(artifacts: list[Artifact], args: argparse.Namespace, cache_root: Path) -> None:
    ssh = _ssh_base(args)
    scp = _scp_base(args)
    remote_base = f"{args.ssh_user}@{args.ssh_host}"
    for artifact in artifacts:
        src = cache_root / artifact.relative_path
        if not src.exists():
            print(f"missing local artifact for sync: {artifact.relative_path}", file=sys.stderr)
            continue
        remote_dir = f"{args.device_root.rstrip('/')}/{artifact.relative_path.parent}"
        subprocess.run([*ssh, remote_base, f"mkdir -p {shell_quote(remote_dir)}"], check=True)
        subprocess.run([*scp, str(src), f"{remote_base}:{remote_dir}/"], check=True)


def shell_quote(value: str) -> str:
    return "'" + value.replace("'", "'\"'\"'") + "'"


def main() -> int:
    args = parse_args()
    artifacts = load_artifacts(args.models_yaml, scope=args.scope, models_csv=args.models)
    lenovo_client = None
    if args.from_lenovo_share_url:
        lenovo_client = LenovoFilezClient(args.from_lenovo_share_url, args.from_lenovo_password)
        lenovo_client.open()
    if args.lenovo_tree_root:
        if not lenovo_client:
            raise RuntimeError("--lenovo-tree-root requires --from-lenovo-share-url and --from-lenovo-password")
        artifacts = lenovo_client.list_tree(args.lenovo_tree_root)
        if args.scope != "all":
            artifacts = [artifact for artifact in artifacts if artifact.category == args.scope]
        wanted = {item.strip() for item in args.models.split(",") if item.strip()}
        if wanted:
            artifacts = [artifact for artifact in artifacts if artifact.model_name in wanted]
    if args.manifest:
        print_manifest(artifacts, args.cache_root)
        return 0

    cached, missing = cache_status(artifacts, args.cache_root)
    if args.check_cache:
        print(f"cache_root: {args.cache_root}")
        print(f"artifacts: {len(artifacts)} cached: {len(cached)} missing: {len(missing)}")
        for artifact in missing[:100]:
            print(f"MISSING\t{artifact.relative_path}")
        return 0 if not missing else 1

    if args.download_missing:
        failures: list[Artifact] = []
        def download_one(artifact: Artifact) -> tuple[Artifact, bool, str]:
            ok = False
            if args.from_local_root:
                ok = copy_from_local(artifact, args.from_local_root, args.cache_root, args.force)
            if not ok and args.from_url_base:
                try:
                    ok = download_from_url(artifact, args.from_url_base, args.cache_root, args.force)
                except Exception as exc:  # noqa: BLE001
                    return artifact, False, f"URL failed {artifact.relative_path}: {exc}"
            if not ok and lenovo_client:
                try:
                    ok = lenovo_client.download(artifact, args.cache_root, args.force)
                except Exception as exc:  # noqa: BLE001
                    return artifact, False, f"Lenovo Filez failed {artifact.relative_path}: {exc}"
            if not ok and args.ssh_host and args.ssh_user:
                ok = copy_from_ssh(artifact, args, args.cache_root, args.force)
            return artifact, ok, ""

        if args.jobs > 1 and lenovo_client and args.lenovo_tree_root:
            with ThreadPoolExecutor(max_workers=args.jobs) as pool:
                futures = [pool.submit(download_one, artifact) for artifact in artifacts]
                for future in as_completed(futures):
                    artifact, ok, error = future.result()
                    if error:
                        print(error, file=sys.stderr)
                    print(f"{'OK' if ok else 'MISSING'}\t{artifact.relative_path}", flush=True)
                    if not ok:
                        failures.append(artifact)
        else:
            for artifact in artifacts:
                artifact, ok, error = download_one(artifact)
                if error:
                    print(error, file=sys.stderr)
                print(f"{'OK' if ok else 'MISSING'}\t{artifact.relative_path}", flush=True)
                if not ok:
                    failures.append(artifact)
        write_index(artifacts, args.cache_root)
        if failures:
            print(f"missing artifacts: {len(failures)}", file=sys.stderr)
            return 1

    if args.sync_to_device:
        sync_to_device(artifacts, args, args.cache_root)

    if not args.download_missing and not args.sync_to_device:
        print_manifest(artifacts, args.cache_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
