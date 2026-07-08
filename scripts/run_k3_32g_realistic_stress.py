#!/usr/bin/env python3
"""Run a focused K3 32GB realistic workflow control test.

This is intentionally not a full ModelZoo matrix.  It retests the selected
Qwen3 LLM/VLM controls, then exercises OCR, ASR, embedding, and reranker
building blocks with resource snapshots.  If a scheduler/gateway is absent,
the script records that limitation and treats the run as a raw model-server
control rather than proof of queueing/admission behavior.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import shlex
import statistics
import subprocess
import sys
import threading
import time
import urllib.request
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Any, Iterable

import httpx


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = ROOT / "output" / "reports" / "k3-riscv-32g"
MODEL_CACHE = ROOT / "drivers" / "spacemit-ai" / "model_zoo"
REMOTE_MODEL_ROOT = "/root/models/spacemit-ai"
REMOTE_WORK_ROOT = "/root/local-ai-bench"
PPOCR_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/"
    "ppocr/utils/dict/ppocrv5_dict.txt"
)


def now_stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log(message: str) -> None:
    print(f"[{time.strftime('%F %T')}] {message}", flush=True)


def write_jsonl(path: Path, item: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(len(ordered) * q) - 1))
    return ordered[idx]


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", "", str(text or ""))


def edit_distance(a: str, b: str) -> int:
    if a == b:
        return 0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]


def cer(ref: str, hyp: str) -> float:
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)
    return edit_distance(ref_n, hyp_n) / max(1, len(ref_n))


def normalize_zh_asr(text: str) -> str:
    table = str.maketrans(
        {
            "開": "开",
            "時": "时",
            "間": "间",
            "點": "点",
            "臺": "台",
            "九": "9",
            "五": "5",
            "零": "0",
            "〇": "0",
            "一": "1",
            "二": "2",
            "三": "3",
            "四": "4",
            "六": "6",
            "七": "7",
            "八": "8",
            "：": "",
            ":": "",
            "。": "",
            "，": "",
            ",": "",
            ".": "",
        }
    )
    return normalize_text(text.translate(table))


def zh_asr_cer(ref: str, hyp: str) -> float:
    ref_n = normalize_zh_asr(ref)
    hyp_n = normalize_zh_asr(hyp)
    return edit_distance(ref_n, hyp_n) / max(1, len(ref_n))


def ned(ref: str, hyp: str) -> float:
    ref_n = normalize_text(ref)
    hyp_n = normalize_text(hyp)
    return edit_distance(ref_n, hyp_n) / max(1, max(len(ref_n), len(hyp_n)))


def expected_variants(value: Any) -> set[str]:
    text = str(value or "").strip()
    variants = {normalize_value(text)}
    numeric = text.replace("人民币", "").replace("元", "").replace(",", "").replace("，", "").strip()
    suffix = "%" if numeric.endswith("%") else ""
    if suffix:
        numeric = numeric[:-1]
    try:
        dec = Decimal(numeric)
    except (InvalidOperation, ValueError):
        return {v for v in variants if v}
    fixed = format(dec, "f")
    trimmed = fixed.rstrip("0").rstrip(".") if "." in fixed else fixed
    for candidate in {fixed, trimmed, f"{fixed}{suffix}", f"{trimmed}{suffix}"}:
        variants.add(normalize_value(candidate))
    return {v for v in variants if v}


def normalize_value(value: Any) -> str:
    text = str(value or "").lower()
    text = text.replace("人民币", "").replace("元", "")
    return re.sub(r"[\s\t\r\n:：,，.。;；/\\|_\-—*]+", "", text)


def extract_jsonish(text: str) -> Any:
    raw = (text or "").strip()
    raw = re.sub(r"^```(?:json)?", "", raw, flags=re.I).strip()
    raw = re.sub(r"```$", "", raw).strip()
    candidates = [raw]
    for opener, closer in [("{", "}"), ("[", "]")]:
        start = raw.find(opener)
        end = raw.rfind(closer)
        if start != -1 and end > start:
            candidates.append(raw[start : end + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except Exception:
            pass
    return None


@dataclass
class Remote:
    host: str
    user: str
    password: str

    def _env(self) -> dict[str, str]:
        env = os.environ.copy()
        env["SSHPASS"] = self.password
        return env

    def ssh_base(self) -> list[str]:
        return [
            "sshpass",
            "-e",
            "ssh",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ServerAliveInterval=30",
            "-o",
            "ConnectTimeout=10",
            f"{self.user}@{self.host}",
        ]

    def scp_base(self) -> list[str]:
        return [
            "sshpass",
            "-e",
            "scp",
            "-o",
            "StrictHostKeyChecking=no",
            "-o",
            "ConnectTimeout=10",
        ]

    def run(self, cmd: str, *, timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
        remote_cmd = "bash -lc " + shlex.quote(cmd)
        proc = subprocess.run(
            self.ssh_base() + [remote_cmd],
            text=True,
            capture_output=True,
            timeout=timeout,
            env=self._env(),
        )
        if check and proc.returncode != 0:
            raise RuntimeError(f"remote command failed rc={proc.returncode}\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
        return proc

    def put(self, local: Path, remote_dir: str) -> None:
        self.run(f"mkdir -p {shlex.quote(remote_dir)}", check=True)
        subprocess.run(
            self.scp_base() + ["-p", str(local), f"{self.user}@{self.host}:{remote_dir}/"],
            check=True,
            env=self._env(),
        )

    def put_dir(self, local_dir: Path, remote_parent: str) -> None:
        self.run(f"mkdir -p {shlex.quote(remote_parent)}", check=True)
        subprocess.run(
            self.scp_base() + ["-rp", str(local_dir), f"{self.user}@{self.host}:{remote_parent}/"],
            check=True,
            env=self._env(),
        )

    def stream_extract_tar(self, local_tar: Path, remote_dir: str) -> None:
        self.run(f"mkdir -p {shlex.quote(remote_dir)}", check=True)
        with local_tar.open("rb") as f:
            subprocess.run(
                self.ssh_base()
                + [
                    "bash -lc "
                    + shlex.quote(f"tar -xzf - -C {shlex.quote(remote_dir)} --skip-old-files"),
                ],
                stdin=f,
                check=True,
                env=self._env(),
            )


class ResourceSampler:
    def __init__(self, remote: Remote, pid: str, label: str, out: Path, interval_s: float) -> None:
        self.remote = remote
        self.pid = pid
        self.label = label
        self.out = out
        self.interval_s = interval_s
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        self.thread.join(timeout=max(2.0, self.interval_s + 2.0))

    def _loop(self) -> None:
        while not self.stop_event.is_set():
            cmd = f"""
set +e
echo TS=$(date +%s.%N)
echo LABEL={shlex.quote(self.label)}
echo PID={shlex.quote(self.pid)}
ps -p {shlex.quote(self.pid)} -o pid=,ppid=,stat=,rss=,vsz=,pcpu=,pmem=,etime=,comm= 2>/dev/null
cat /proc/loadavg 2>/dev/null
awk '/MemTotal|MemAvailable|MemFree|SwapTotal|SwapFree/ {{print}}' /proc/meminfo 2>/dev/null
spacemit-tcm-smi 2>/dev/null | head -20
"""
            proc = self.remote.run(cmd, timeout=15)
            write_jsonl(
                self.out,
                {
                    "ts": time.time(),
                    "label": self.label,
                    "pid": self.pid,
                    "returncode": proc.returncode,
                    "stdout": proc.stdout,
                    "stderr": proc.stderr,
                },
            )
            self.stop_event.wait(self.interval_s)


class OpenAIClient:
    def __init__(self, base_url: str, timeout_s: float = 900.0) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def models(self) -> dict[str, Any]:
        return self.client.get(f"{self.base_url}/models").json()

    def chat(
        self,
        model: str,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        stream: bool = False,
        extra: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": stream,
        }
        if extra:
            payload.update(extra)
        t0 = time.perf_counter()
        if not stream:
            r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
            elapsed = time.perf_counter() - t0
            item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_s": round(elapsed, 3)}
            try:
                body = r.json()
                choice = body.get("choices", [{}])[0]
                message = choice.get("message", {})
                item.update(
                    {
                        "content": message.get("content") or "",
                        "reasoning_content": message.get("reasoning_content") or message.get("reasoning") or "",
                        "finish_reason": choice.get("finish_reason"),
                        "usage": body.get("usage"),
                    }
                )
            except Exception as exc:
                item.update({"error": f"json decode failed: {exc}", "body_prefix": r.text[:1000]})
            return item

        first = None
        chunks = 0
        content: list[str] = []
        reasoning: list[str] = []
        with self.client.stream("POST", f"{self.base_url}/chat/completions", json=payload) as r:
            status = r.status_code
            for line in r.iter_lines():
                if not line:
                    continue
                if isinstance(line, bytes):
                    line = line.decode("utf-8", errors="replace")
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                if first is None:
                    first = time.perf_counter()
                chunks += 1
                try:
                    delta = json.loads(data).get("choices", [{}])[0].get("delta", {})
                    content.append(delta.get("content") or "")
                    reasoning.append(delta.get("reasoning_content") or delta.get("reasoning") or "")
                except Exception:
                    pass
        elapsed = time.perf_counter() - t0
        return {
            "status_code": status,
            "ok": status == 200,
            "ttft_s": None if first is None else round(first - t0, 3),
            "elapsed_s": round(elapsed, 3),
            "chunks": chunks,
            "content": "".join(content),
            "reasoning_content": "".join(reasoning),
        }

    def embeddings(self, model: str, texts: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = self.client.post(f"{self.base_url}/embeddings", json={"model": model, "input": texts})
        elapsed = time.perf_counter() - t0
        item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_s": round(elapsed, 4)}
        try:
            body = r.json()
            item["body"] = body
        except Exception as exc:
            item.update({"error": str(exc), "body_prefix": r.text[:1000]})
        return item

    def rerank(self, model: str, query: str, documents: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = self.client.post(
            f"{self.base_url}/rerank",
            json={"model": model, "query": query, "documents": documents, "return_documents": False},
        )
        elapsed = time.perf_counter() - t0
        item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_s": round(elapsed, 4)}
        try:
            item["body"] = r.json()
        except Exception as exc:
            item.update({"error": str(exc), "body_prefix": r.text[:1000]})
        return item


def ensure_local_file(path: Path, url: str | None = None) -> None:
    if path.exists() and path.stat().st_size > 0:
        return
    if not url:
        raise FileNotFoundError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    log(f"download {url}")
    urllib.request.urlretrieve(url, path)


def ensure_remote_file(remote: Remote, local: Path, remote_dir: str) -> None:
    remote_path = f"{remote_dir.rstrip('/')}/{local.name}"
    proc = remote.run(f"test -s {shlex.quote(remote_path)}")
    if proc.returncode != 0:
        log(f"copy {local.name} -> {remote_dir}")
        remote.put(local, remote_dir)


def wait_for_server(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    url = f"http://{host}:{port}/v1/models"
    while time.monotonic() < deadline:
        try:
            r = httpx.get(url, timeout=2.0)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(2)
    return False


def start_server(
    remote: Remote,
    *,
    run_dir: str,
    label: str,
    port: int,
    model_path: str,
    alias: str,
    extra_args: list[str] | None = None,
    ctx: int = 4096,
    batch: int = 1024,
    ubatch: int = 512,
    cache_k: str = "q8_0",
    cache_v: str = "q8_0",
    startup_timeout_s: int = 240,
) -> tuple[str, str]:
    out = f"{run_dir.rstrip('/')}/{label}"
    args = [
        "/usr/bin/llama-server",
        "-m",
        model_path,
        "--alias",
        alias,
        "--host",
        "0.0.0.0",
        "--port",
        str(port),
        "-c",
        str(ctx),
        "-t",
        "8",
        "-tb",
        "8",
        "-b",
        str(batch),
        "-ub",
        str(ubatch),
        "-ctk",
        cache_k,
        "-ctv",
        cache_v,
        "--no-webui",
        "--no-warmup",
        "--cache-ram",
        "0",
        "--log-file",
        f"{out}/llama-server.log",
    ]
    if extra_args:
        args[1:1] = extra_args
    quoted_args = " ".join(shlex.quote(a) for a in args)
    cmd = f"""
set -e
mkdir -p {shlex.quote(out)}
for old_pid in $(pgrep -x llama-server 2>/dev/null || true); do
  old_cmd=$(tr '\\000' ' ' < /proc/$old_pid/cmdline 2>/dev/null || true)
  case "$old_cmd" in
    *"--port {port}"*) kill "$old_pid" >/dev/null 2>&1 || true ;;
  esac
done
sleep 1
spacemit-tcm-smi > {shlex.quote(out)}/tcm-before.txt 2>&1 || true
setsid env SPACEMIT_DISABLE_TCM=1 {quoted_args} > {shlex.quote(out)}/llama-server.stdout.log 2> {shlex.quote(out)}/llama-server.stderr.log < /dev/null &
pid=$!
echo "$pid" > {shlex.quote(out)}/llama-server.pid
echo "$pid"
"""
    proc = remote.run(cmd, timeout=30, check=True)
    pid = proc.stdout.strip().splitlines()[-1]
    if not wait_for_server(remote.host, port, startup_timeout_s):
        logs = remote.run(f"tail -160 {shlex.quote(out)}/llama-server.log {shlex.quote(out)}/llama-server.stderr.log 2>/dev/null || true")
        raise RuntimeError(f"{label} server did not become ready on port {port}; pid={pid}\n{logs.stdout}")
    return pid, out


def stop_server(remote: Remote, out: str) -> None:
    cmd = f"""
set +e
if [ -f {shlex.quote(out)}/llama-server.pid ]; then
  pid=$(cat {shlex.quote(out)}/llama-server.pid)
  kill "$pid" >/dev/null 2>&1 || true
  sleep 2
  kill -9 "$pid" >/dev/null 2>&1 || true
fi
spacemit-tcm-smi > {shlex.quote(out)}/tcm-after.txt 2>&1 || true
"""
    remote.run(cmd, timeout=30)


def image_content(path: Path, prompt: str) -> list[dict[str, Any]]:
    ext = path.suffix.lower().lstrip(".") or "png"
    mime = "jpeg" if ext in {"jpg", "jpeg"} else ext
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "image_url", "image_url": {"url": f"data:image/{mime};base64,{data}"}},
    ]


def audio_content(path: Path, prompt: str) -> list[dict[str, Any]]:
    data = base64.b64encode(path.read_bytes()).decode("ascii")
    return [
        {"type": "text", "text": prompt},
        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
    ]


def run_preflight(remote: Remote, out_dir: Path, remote_run_dir: str) -> dict[str, Any]:
    log("preflight")
    checks = {
        "date": remote.run("date -Is || true").stdout.strip(),
        "uname": remote.run("uname -a || true").stdout.strip(),
        "df": remote.run("df -h /root /root/models 2>/dev/null || df -h /root || true").stdout,
        "meminfo": remote.run("awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo || true").stdout,
        "tcm": remote.run("spacemit-tcm-smi 2>&1 || true").stdout,
        "versions": remote.run("dpkg -l | grep -E 'llama.cpp-tools-spacemit|spacemit-onnxruntime|spacemit-tcm' || true").stdout,
        "scheduler_ready": None,
    }
    scheduler = remote.run("curl -fsS --max-time 2 http://127.0.0.1:8080/ready 2>&1 || true").stdout.strip()
    checks["scheduler_ready"] = scheduler
    (out_dir / "preflight.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
    remote.run(f"mkdir -p {shlex.quote(remote_run_dir)}", check=True)
    return checks


def run_llm(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("LLM retest: Qwen3-30B-A3B-Q4_0")
    label = "llm-qwen3-30b"
    trace = out_dir / "trace.jsonl"
    model_path = f"{REMOTE_MODEL_ROOT}/llm/Qwen3-30B-A3B-Q4_0.gguf"
    pid = ""
    server_out = ""
    sampler: ResourceSampler | None = None
    results: list[dict[str, Any]] = []
    try:
        pid, server_out = start_server(
            remote,
            run_dir=remote_run_dir,
            label=label,
            port=args.llm_port,
            model_path=model_path,
            alias="Qwen3-30B-A3B-Q4_0",
            ctx=args.llm_ctx,
            startup_timeout_s=args.startup_timeout,
        )
        sampler = ResourceSampler(remote, pid, label, out_dir / "resource" / f"{label}.jsonl", args.resource_interval)
        sampler.start()
        client = OpenAIClient(f"http://{remote.host}:{args.llm_port}/v1", timeout_s=args.request_timeout)
        models = {"phase": "llm", "case": "models", "models": client.models()}
        write_jsonl(trace, models)
        no_think = {"chat_template_kwargs": {"enable_thinking": False}}
        cases = [
            ("stream_ttft", "/no_think\n用一句中文说明你已就绪。", 32, True),
            ("decode_128", "/no_think\n请用中文列出 K3 32G 边缘推理测试需要关注的五个指标。", 128, False),
        ]
        for case_id, prompt, max_tokens, stream in cases:
            item = client.chat(
                "Qwen3-30B-A3B-Q4_0",
                [{"role": "user", "content": prompt}],
                max_tokens=max_tokens,
                stream=stream,
                extra=no_think,
            )
            item.update({"phase": "llm", "case": case_id})
            write_jsonl(trace, item)
            results.append(item)
        for target in args.llm_contexts:
            marker = f"K3_NEEDLE_{target}_竹影星轨"
            filler = "以下为边缘 AI 平台验收材料，包含硬件、运行时、模型加载、吞吐、上下文和稳定性说明。"
            chars = max(512, int(target * 1.65))
            body = (filler * ((chars // len(filler)) + 1))[:chars]
            prompt = f"/no_think\n{body}\n\n关键暗号: {marker}\n\n问题: 上文给出的关键暗号是什么？只回答关键暗号。"
            item = client.chat(
                "Qwen3-30B-A3B-Q4_0",
                [{"role": "user", "content": prompt}],
                max_tokens=32,
                extra=no_think,
            )
            item.update(
                {
                    "phase": "llm",
                    "case": f"context_{target}",
                    "target_context_tokens": target,
                    "needle": marker,
                    "needle_recall": marker in item.get("content", ""),
                    "needle_recall_any": marker in (item.get("content", "") + item.get("reasoning_content", "")),
                }
            )
            write_jsonl(trace, item)
            results.append(item)
    finally:
        if sampler:
            sampler.stop()
        if server_out:
            stop_server(remote, server_out)
    summary = {
        "phase": "llm",
        "model": "Qwen3-30B-A3B-Q4_0",
        "cases": len(results),
        "ok_cases": sum(1 for r in results if r.get("ok")),
        "latency_s": {r.get("case", "?"): r.get("elapsed_s") for r in results},
        "ttft_s": next((r.get("ttft_s") for r in results if r.get("case") == "stream_ttft"), None),
        "context_recall": {str(r.get("target_context_tokens")): r.get("needle_recall_any") for r in results if "target_context_tokens" in r},
    }
    (out_dir / "llm-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def iter_vlm_cases(limit: int) -> Iterable[dict[str, Any]]:
    cases_path = ROOT / "datasets" / "scenarios" / "vlm_document_extraction" / "cases.jsonl"
    count = 0
    with cases_path.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            yield json.loads(line)
            count += 1
            if limit and count >= limit:
                break


def score_vlm_case(case: dict[str, Any], content: str) -> dict[str, Any]:
    payload = case.get("payload", {})
    fields = payload.get("fields", [])
    golden = payload.get("golden", {})
    parsed = extract_jsonish(content)
    haystack = normalize_value(content)
    if parsed is not None:
        haystack += normalize_value(json.dumps(parsed, ensure_ascii=False))
    field_scores = {}
    for field in fields:
        variants = expected_variants(golden.get(field, ""))
        field_scores[field] = any(v in haystack for v in variants)
    hits = sum(1 for ok in field_scores.values() if ok)
    total = len(fields)
    return {
        "json_parse_ok": parsed is not None,
        "field_count": total,
        "field_hits": hits,
        "field_accuracy": round(hits / total, 4) if total else 0.0,
        "case_pass": total > 0 and hits == total,
        "field_scores": field_scores,
    }


def run_vlm(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("VLM retest: Qwen3VL-4B + mmproj")
    label = "vlm-qwen3vl-4b"
    trace = out_dir / "trace.jsonl"
    model_path = f"{REMOTE_MODEL_ROOT}/vlm/Qwen3VL/Qwen3VL-4B-Instruct-Q4_K_M.gguf"
    mmproj_path = f"{REMOTE_MODEL_ROOT}/vlm/Qwen3VL/mmproj-Qwen3VL-4B-Instruct-F16.gguf"
    pid = ""
    server_out = ""
    sampler: ResourceSampler | None = None
    case_results: list[dict[str, Any]] = []
    try:
        pid, server_out = start_server(
            remote,
            run_dir=remote_run_dir,
            label=label,
            port=args.vlm_port,
            model_path=model_path,
            alias="Qwen3VL-4B-Instruct-Q4_K_M",
            extra_args=["--mmproj", mmproj_path],
            ctx=4096,
            startup_timeout_s=args.startup_timeout,
        )
        sampler = ResourceSampler(remote, pid, label, out_dir / "resource" / f"{label}.jsonl", args.resource_interval)
        sampler.start()
        client = OpenAIClient(f"http://{remote.host}:{args.vlm_port}/v1", timeout_s=args.request_timeout)
        for case in iter_vlm_cases(args.vlm_cases):
            payload = case["payload"]
            image = ROOT / payload["image_path"]
            prompt = (
                "请从图片中抽取结构化字段。只输出一个 JSON object，不要 Markdown。"
                f"文档类型: {payload.get('document_type', '')}。"
                f"必须包含字段: {', '.join(payload.get('fields', []))}。"
                "金额保留数字和小数，日期保留原格式。"
            )
            item = client.chat(
                "Qwen3VL-4B-Instruct-Q4_K_M",
                [{"role": "user", "content": image_content(image, prompt)}],
                max_tokens=192,
            )
            scored = score_vlm_case(case, item.get("content", ""))
            result = {
                **item,
                **scored,
                "phase": "vlm",
                "case": "doc_extract",
                "id": case.get("id"),
                "document_type": payload.get("document_type"),
            }
            write_jsonl(trace, result)
            case_results.append(result)
    finally:
        if sampler:
            sampler.stop()
        if server_out:
            stop_server(remote, server_out)
    lat = [float(r["elapsed_s"]) for r in case_results if r.get("elapsed_s") is not None]
    fields_total = sum(int(r.get("field_count", 0)) for r in case_results)
    fields_hit = sum(int(r.get("field_hits", 0)) for r in case_results)
    summary = {
        "phase": "vlm",
        "model": "Qwen3VL-4B-Instruct-Q4_K_M + mmproj",
        "cases": len(case_results),
        "case_pass": sum(1 for r in case_results if r.get("case_pass")),
        "case_pass_rate": round(sum(1 for r in case_results if r.get("case_pass")) / len(case_results), 4) if case_results else 0,
        "field_accuracy": round(fields_hit / fields_total, 4) if fields_total else 0,
        "json_parse_rate": round(sum(1 for r in case_results if r.get("json_parse_ok")) / len(case_results), 4) if case_results else 0,
        "latency_avg_s": round(sum(lat) / len(lat), 3) if lat else None,
        "latency_p50_s": percentile(lat, 0.5),
        "latency_p95_s": percentile(lat, 0.95),
    }
    (out_dir / "vlm-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_embedding(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("Embedding test: BGE small zh")
    label = "embed-bge-zh"
    local_model = MODEL_CACHE / "embed" / "Bge-Small-Zh-V1.5-Q4_K_M.gguf"
    ensure_remote_file(remote, local_model, f"{REMOTE_MODEL_ROOT}/embed")
    model_path = f"{REMOTE_MODEL_ROOT}/embed/{local_model.name}"
    pid = ""
    server_out = ""
    sampler: ResourceSampler | None = None
    trace = out_dir / "trace.jsonl"
    try:
        pid, server_out = start_server(
            remote,
            run_dir=remote_run_dir,
            label=label,
            port=args.embed_port,
            model_path=model_path,
            alias=local_model.name,
            extra_args=["--embedding", "--pooling", "mean"],
            ctx=512,
            batch=512,
            ubatch=256,
            startup_timeout_s=args.startup_timeout,
        )
        sampler = ResourceSampler(remote, pid, label, out_dir / "resource" / f"{label}.jsonl", args.resource_interval)
        sampler.start()
        client = OpenAIClient(f"http://{remote.host}:{args.embed_port}/v1", timeout_s=args.request_timeout)
        corpus = [
            ("d0", "北京是中国的首都，也是政治和文化中心。"),
            ("d1", "上海是重要的金融中心，拥有繁忙的港口。"),
            ("d2", "Python 常用于数据分析、自动化和机器学习。"),
            ("d3", "端侧 AI 推理关注吞吐、首 token 延迟、内存和稳定性。"),
            ("d4", "OCR 会把图片中的文字识别为可检索文本。"),
        ]
        queries = [
            ("中国首都是哪里", "d0"),
            ("哪座城市是金融中心", "d1"),
            ("Python 可以做什么", "d2"),
            ("边缘推理要看哪些性能指标", "d3"),
            ("图片文字识别是什么任务", "d4"),
        ]
        doc_item = client.embeddings(local_model.name, [x[1] for x in corpus])
        write_jsonl(trace, {"phase": "embedding", "case": "corpus", **doc_item})
        doc_vecs = [x.get("embedding", []) for x in doc_item.get("body", {}).get("data", [])]

        def cosine(a: list[float], b: list[float]) -> float:
            dot = sum(x * y for x, y in zip(a, b))
            na = math.sqrt(sum(x * x for x in a))
            nb = math.sqrt(sum(y * y for y in b))
            return dot / (na * nb) if na and nb else -1.0

        hits = 0
        rrs: list[float] = []
        ndcgs: list[float] = []
        lat_ms: list[float] = []
        for query, expected in queries:
            item = client.embeddings(local_model.name, [query])
            lat_ms.append(float(item["elapsed_s"]) * 1000)
            vecs = [x.get("embedding", []) for x in item.get("body", {}).get("data", [])]
            if vecs and doc_vecs:
                scores = [(corpus[i][0], cosine(vecs[0], doc_vecs[i])) for i in range(len(corpus))]
                scores.sort(key=lambda x: x[1], reverse=True)
                rank = [doc_id for doc_id, _ in scores].index(expected) + 1
                hits += int(rank == 1)
                rrs.append(1.0 / rank)
                ndcgs.append(1.0 / math.log2(rank + 1))
                item.update({"rank": rank, "expected": expected, "top": scores[:3]})
            item.update({"phase": "embedding", "case": "query", "query": query})
            write_jsonl(trace, item)
    finally:
        if sampler:
            sampler.stop()
        if server_out:
            stop_server(remote, server_out)
    summary = {
        "phase": "embedding",
        "model": local_model.name,
        "hit_at_1": hits / 5,
        "mrr": sum(rrs) / len(rrs) if rrs else 0,
        "ndcg_at_5": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
        "latency_p50_ms": percentile(lat_ms, 0.5),
        "latency_p95_ms": percentile(lat_ms, 0.95),
    }
    (out_dir / "embedding-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def run_reranker(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("Reranker test: BGE reranker v2 m3")
    label = "rerank-bge-v2-m3"
    local_model = MODEL_CACHE / "rerank" / "Bge-Reranker-V2-M3-Q4_0.gguf"
    ensure_remote_file(remote, local_model, f"{REMOTE_MODEL_ROOT}/rerank")
    model_path = f"{REMOTE_MODEL_ROOT}/rerank/{local_model.name}"
    trace = out_dir / "trace.jsonl"
    pid = ""
    server_out = ""
    sampler: ResourceSampler | None = None
    cases = [
        ("中国首都是哪里", ["上海是金融中心。", "北京是中国的首都。", "Python 是编程语言。"], 1),
        ("边缘推理关注哪些指标", ["吞吐和首 token 延迟是关键指标。", "今天适合散步。", "咖啡需要研磨。"], 0),
        ("OCR 识别什么内容", ["图片中的文字会被 OCR 提取。", "电池容量单位是 mAh。", "向量数据库用于检索。"], 0),
    ]
    ranks: list[int] = []
    lat_ms: list[float] = []
    try:
        pid, server_out = start_server(
            remote,
            run_dir=remote_run_dir,
            label=label,
            port=args.rerank_port,
            model_path=model_path,
            alias=local_model.name,
            extra_args=["--reranking", "--pooling", "rank"],
            ctx=512,
            batch=512,
            ubatch=256,
            startup_timeout_s=args.startup_timeout,
        )
        sampler = ResourceSampler(remote, pid, label, out_dir / "resource" / f"{label}.jsonl", args.resource_interval)
        sampler.start()
        client = OpenAIClient(f"http://{remote.host}:{args.rerank_port}/v1", timeout_s=args.request_timeout)
        for query, docs, expected in cases:
            item = client.rerank(local_model.name, query, docs)
            results = item.get("body", {}).get("results", [])
            ranked = [x.get("index") for x in results]
            rank = ranked.index(expected) + 1 if expected in ranked else 999
            ranks.append(rank)
            lat_ms.append(float(item["elapsed_s"]) * 1000)
            item.update({"phase": "reranker", "case": "query", "query": query, "rank": rank, "expected": expected})
            write_jsonl(trace, item)
    finally:
        if sampler:
            sampler.stop()
        if server_out:
            stop_server(remote, server_out)
    summary = {
        "phase": "reranker",
        "model": local_model.name,
        "hit_at_1": sum(1 for r in ranks if r == 1) / len(cases),
        "mrr": sum(1.0 / r for r in ranks) / len(ranks) if ranks else 0,
        "ndcg": sum(1.0 / math.log2(r + 1) for r in ranks) / len(ranks) if ranks else 0,
        "latency_p50_ms": percentile(lat_ms, 0.5),
        "latency_p95_ms": percentile(lat_ms, 0.95),
    }
    (out_dir / "reranker-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def ensure_ppocr(remote: Remote) -> None:
    ppocr_dir = MODEL_CACHE / "vision" / "ppocr"
    ensure_local_file(ppocr_dir / "PP-OCRv5_mobile_det.onnx", "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/ppocr/PP-OCRv5_mobile_det.onnx")
    ensure_local_file(ppocr_dir / "PP-OCRv5_mobile_rec.onnx", "https://archive.spacemit.com/spacemit-ai/model_zoo/vision/ppocr/PP-OCRv5_mobile_rec.onnx")
    ensure_local_file(ppocr_dir / "ppocrv5_dict.txt", PPOCR_DICT_URL)
    for file in ["PP-OCRv5_mobile_det.onnx", "PP-OCRv5_mobile_rec.onnx", "ppocrv5_dict.txt"]:
        ensure_remote_file(remote, ppocr_dir / file, f"{REMOTE_MODEL_ROOT}/vision/ppocr")
    remote.put_dir(ROOT / "datasets" / "ocr", f"{REMOTE_WORK_ROOT}/datasets")


def run_ocr(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("OCR test: SpacemiT ModelZoo PP-OCRv5 mobile")
    ensure_ppocr(remote)
    remote_out = f"{remote_run_dir}/ocr-ppocrv5"
    remote.run(f"mkdir -p {shlex.quote(remote_out)}", check=True)
    code = r'''
import json, math, re, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

model_dir = Path("/root/models/spacemit-ai/vision/ppocr")
data_root = Path("/root/local-ai-bench/datasets/ocr")
manifest = data_root / "manifest.jsonl"
providers = ort.get_available_providers()
det_sess = ort.InferenceSession(str(model_dir / "PP-OCRv5_mobile_det.onnx"), providers=providers)
rec_sess = ort.InferenceSession(str(model_dir / "PP-OCRv5_mobile_rec.onnx"), providers=providers)
chars = ["blank"] + (model_dir / "ppocrv5_dict.txt").read_text(encoding="utf-8").splitlines() + [" "]

def norm(text):
    return re.sub(r"\s+", "", str(text or ""))

def edit(a, b):
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb))
        prev = cur
    return prev[-1]

def rec_pre(img):
    h, w = img.shape[:2]
    img_h = 48
    new_w = max(1, int(math.ceil(img_h * w / h)))
    im = cv2.resize(img, (new_w, img_h))
    x = ((im.astype("float32").transpose(2, 0, 1) / 255.0) - 0.5) / 0.5
    return x[None].astype("float32")

def crop_whitespace(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray < 245
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return img
    pad = 8
    x0 = max(0, int(xs.min()) - pad)
    x1 = min(img.shape[1], int(xs.max()) + pad + 1)
    y0 = max(0, int(ys.min()) - pad)
    y1 = min(img.shape[0], int(ys.max()) + pad + 1)
    return img[y0:y1, x0:x1]

def decode(pred):
    idx = pred.argmax(axis=2)[0]
    prob = pred.max(axis=2)[0]
    out, ps, prev = [], [], None
    for raw_i, p in zip(idx, prob):
        i = int(raw_i)
        if i != 0 and i != prev and i < len(chars):
            out.append(chars[i])
            ps.append(float(p))
        prev = i
    return "".join(out), (sum(ps) / len(ps) if ps else 0.0)

def det_smoke(img):
    h, w = img.shape[:2]
    ratio = 736.0 / min(h, w) if min(h, w) < 736 else 1.0
    rh = max(32, int(round(h * ratio / 32) * 32))
    rw = max(32, int(round(w * ratio / 32) * 32))
    im = cv2.resize(img, (rw, rh))
    x = ((im.astype("float32") / 255.0) - 0.5) / 0.5
    x = x.transpose(2, 0, 1)[None].astype("float32")
    pred = det_sess.run(None, {det_sess.get_inputs()[0].name: x})[0][0, 0]
    return int((pred > 0.3).sum())

refs, hyps, latencies = [], [], []
print(json.dumps({"phase": "ocr", "case": "providers", "providers": providers, "det_providers": det_sess.get_providers(), "rec_providers": rec_sess.get_providers()}, ensure_ascii=False), flush=True)
for line in manifest.read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    img = cv2.imread(str(data_root / item["image"]))
    t0 = time.perf_counter()
    det_pixels = det_smoke(img)
    rec_img = crop_whitespace(img)
    pred = rec_sess.run(None, {rec_sess.get_inputs()[0].name: rec_pre(rec_img)})[0]
    hyp, conf = decode(pred)
    elapsed = time.perf_counter() - t0
    ref = item["text"]
    refs.append(ref)
    hyps.append(hyp)
    latencies.append(elapsed * 1000)
    ref_n, hyp_n = norm(ref), norm(hyp)
    row = {
        "phase": "ocr",
        "case": "sample",
        "uid": item.get("uid"),
        "text": ref,
        "prediction": hyp,
        "confidence": round(conf, 4),
        "latency_ms": round(elapsed * 1000, 2),
        "cer": edit(ref_n, hyp_n) / max(1, len(ref_n)),
        "ned": edit(ref_n, hyp_n) / max(1, max(len(ref_n), len(hyp_n))),
        "det_positive_pixels": det_pixels,
        "mode": "ppocrv5_det_smoke_whitespace_crop_rec_single_line",
        "rec_crop_shape": list(rec_img.shape[:2]),
    }
    print(json.dumps(row, ensure_ascii=False), flush=True)

total_chars = sum(len(norm(x)) for x in refs)
total_edits = sum(edit(norm(r), norm(h)) for r, h in zip(refs, hyps))
ned_scores = [edit(norm(r), norm(h)) / max(1, max(len(norm(r)), len(norm(h)))) for r, h in zip(refs, hyps)]
lat_sorted = sorted(latencies)
summary = {
    "phase": "ocr",
    "case": "summary",
    "model": "PP-OCRv5_mobile_det+rec.onnx",
    "samples": len(refs),
    "cer": total_edits / max(1, total_chars),
    "ned": sum(ned_scores) / len(ned_scores) if ned_scores else None,
    "latency_p50_ms": lat_sorted[len(lat_sorted)//2] if lat_sorted else None,
    "latency_p95_ms": lat_sorted[min(len(lat_sorted)-1, math.ceil(len(lat_sorted)*0.95)-1)] if lat_sorted else None,
    "mode": "det model smoke + whitespace-crop rec for single-line manifest",
}
print(json.dumps(summary, ensure_ascii=False), flush=True)
'''
    proc = remote.run(f"python3 - <<'PY'\n{code}\nPY", timeout=args.request_timeout, check=True)
    trace = out_dir / "trace.jsonl"
    summary: dict[str, Any] = {"phase": "ocr", "error": "summary missing"}
    for line in proc.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        item = json.loads(line)
        write_jsonl(trace, item)
        if item.get("case") == "summary":
            summary = item
    (out_dir / "ocr-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ocr-remote.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return summary


def ensure_asr_package(remote: Remote) -> tuple[str, str]:
    base = f"{REMOTE_MODEL_ROOT}/vlm/qwen3-asr-0.6B"
    probe = remote.run(f"find {shlex.quote(base)} -type f -name 'config.json' -printf '%h\\n' 2>/dev/null | head -1")
    config_dir = probe.stdout.strip()
    if not config_dir:
        tar_path = MODEL_CACHE / "vlm" / "qwen3-asr-0.6B.tar.gz"
        if not tar_path.exists():
            raise FileNotFoundError(tar_path)
        log("extract qwen3-asr-0.6B on K3")
        remote.stream_extract_tar(tar_path, f"{REMOTE_MODEL_ROOT}/vlm")
        probe = remote.run(f"find {shlex.quote(base)} -type f -name 'config.json' -printf '%h\\n' 2>/dev/null | head -1")
        config_dir = probe.stdout.strip()
    model_probe = remote.run(f"find {shlex.quote(base)} -type f -iname '*text*.gguf' | head -1")
    model_path = model_probe.stdout.strip()
    if not config_dir or not model_path:
        raise RuntimeError("qwen3-asr package missing config or text gguf")
    return config_dir, model_path


def run_asr(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> dict[str, Any]:
    log("ASR test: qwen3-asr-0.6B")
    config_dir, model_path = ensure_asr_package(remote)
    label = "asr-qwen3-0.6b"
    trace = out_dir / "trace.jsonl"
    pid = ""
    server_out = ""
    sampler: ResourceSampler | None = None
    result: dict[str, Any] = {}
    try:
        pid, server_out = start_server(
            remote,
            run_dir=remote_run_dir,
            label=label,
            port=args.asr_port,
            model_path=model_path,
            alias="qwen3-asr-0.6B",
            extra_args=["--media-backend", "smt", "--smt-config-dir", config_dir],
            ctx=2048,
            batch=512,
            ubatch=256,
            cache_k="q8_0",
            cache_v="q8_0",
            startup_timeout_s=args.startup_timeout,
        )
        sampler = ResourceSampler(remote, pid, label, out_dir / "resource" / f"{label}.jsonl", args.resource_interval)
        sampler.start()
        client = OpenAIClient(f"http://{remote.host}:{args.asr_port}/v1", timeout_s=args.request_timeout)
        manifest = json.loads((ROOT / "datasets" / "asr" / "manifest.jsonl").read_text(encoding="utf-8").splitlines()[0])
        wav_path = ROOT / "datasets" / "asr" / manifest["audio"]
        prompt_variants = ["<asr_text>", "language Chinese<asr_text>", "请转写音频。<asr_text>"]
        for prompt in prompt_variants:
            item = client.chat(
                "qwen3-asr-0.6B",
                [{"role": "user", "content": audio_content(wav_path, prompt)}],
                max_tokens=128,
            )
            prediction = item.get("content", "")
            item.update(
                {
                    "phase": "asr",
                    "case": "sample",
                    "prompt": prompt,
                    "uid": manifest.get("uid"),
                    "text": manifest.get("text"),
                    "prediction": prediction,
                    "duration_s": manifest.get("duration"),
                    "cer": cer(manifest.get("text", ""), prediction),
                    "normalized_cer": zh_asr_cer(manifest.get("text", ""), prediction),
                    "rtf": (float(item.get("elapsed_s") or 0) / float(manifest.get("duration") or 1)),
                }
            )
            write_jsonl(trace, item)
            if item.get("ok") and prediction:
                result = item
                break
        if not result:
            result = item
    finally:
        if sampler:
            sampler.stop()
        if server_out:
            stop_server(remote, server_out)
    summary = {
        "phase": "asr",
        "model": "qwen3-asr-0.6B",
        "ok": bool(result.get("ok") and result.get("prediction")),
        "cer": result.get("cer"),
        "normalized_cer": result.get("normalized_cer"),
        "rtf": result.get("rtf"),
        "elapsed_s": result.get("elapsed_s"),
        "prediction": result.get("prediction"),
    }
    (out_dir / "asr-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def render_report(out_dir: Path, summaries: dict[str, dict[str, Any]], preflight: dict[str, Any]) -> None:
    lines = [
        "# K3 32G Realistic Workflow Control Run",
        "",
        f"Run directory: `{out_dir}`",
        "",
        "## Scope",
        "",
        "- Raw model servers were used as controls. No scheduler/gateway `/ready`, `/capacity`, `/events`, or async job API was proven in this run.",
        "- Workflows covered: bounded Qwen3 LLM, Qwen3VL document extraction, PP-OCRv5 OCR, qwen3-ASR, BGE embedding, and BGE reranker.",
        "- Resource samples are in `resource/*.jsonl`; per-request evidence is in `trace.jsonl`.",
        "",
        "## Results",
        "",
        "| Phase | Model | Key result | Latency | Verdict |",
        "| --- | --- | --- | --- | --- |",
    ]
    llm = summaries.get("llm")
    if llm:
        lines.append(
            f"| LLM | `{llm.get('model')}` | {llm.get('ok_cases')}/{llm.get('cases')} cases; context recall {llm.get('context_recall')} | TTFT {llm.get('ttft_s')}s | {'PASS' if llm.get('ok_cases') == llm.get('cases') else 'CHECK'} |"
        )
    else:
        lines.append("| LLM | - | skipped in this run | - | SKIP |")
    vlm = summaries.get("vlm", {})
    lines.append(
        f"| VLM | `{vlm.get('model')}` | pass {vlm.get('case_pass')}/{vlm.get('cases')}; field acc {vlm.get('field_accuracy')} | p95 {vlm.get('latency_p95_s')}s | {'PASS' if vlm.get('field_accuracy') == 1 else 'CHECK'} |"
    )
    ocr = summaries.get("ocr", {})
    lines.append(
        f"| OCR | `{ocr.get('model')}` | CER {ocr.get('cer'):.4f}, NED {ocr.get('ned'):.4f}, samples {ocr.get('samples')} | p95 {ocr.get('latency_p95_ms'):.1f}ms | {'PASS' if (ocr.get('cer') or 1) <= 0.10 else 'CHECK'} |"
        if "cer" in ocr
        else f"| OCR | `{ocr.get('model')}` | {ocr.get('error')} | - | CHECK |"
    )
    asr = summaries.get("asr", {})
    lines.append(
        f"| ASR | `{asr.get('model')}` | raw CER {asr.get('cer')}, normalized CER {asr.get('normalized_cer')}, RTF {asr.get('rtf')} | {asr.get('elapsed_s')}s | {'PASS' if asr.get('ok') else 'CHECK'} |"
    )
    emb = summaries.get("embedding", {})
    lines.append(
        f"| Embedding | `{emb.get('model')}` | Hit@1 {emb.get('hit_at_1')}, MRR {emb.get('mrr')} | p95 {emb.get('latency_p95_ms')}ms | {'PASS' if emb.get('hit_at_1') == 1.0 else 'CHECK'} |"
    )
    rer = summaries.get("reranker", {})
    lines.append(
        f"| Reranker | `{rer.get('model')}` | Hit@1 {rer.get('hit_at_1')}, MRR {rer.get('mrr')} | p95 {rer.get('latency_p95_ms')}ms | {'PASS' if rer.get('hit_at_1') == 1.0 else 'CHECK'} |"
    )
    lines += [
        "",
        "## Risk Evaluation Against `docs/k3-realistic-stress-plan.md`",
        "",
        "- **Scheduler risk remains open:** this run cannot prove queue wait, async admission, holder fairness, `/capacity`, `/events`, or backpressure because it uses raw model servers.",
        "- **Long request isolation risk:** Qwen3-30B long-context requests must be async or strictly deadline/token bounded. A raw sync server has no admission control and can block realtime retrieval.",
        "- **Memory/disk risk:** K3 had limited free root space after large model extraction; keep only the hot set on disk and clean stale packages before long soaks.",
        "- **VLM risk:** Qwen3VL quality is a useful control but latency is high; production document upload should prefer compact VLM for sync paths and reserve Qwen3VL/high-spec VLM for async.",
        "- **OCR risk:** PP-OCRv5 is the correct OCR path from `vision/ppocr`, not a VLM substitute. This script validates det ONNX smoke and rec on the single-line OCR manifest; full multi-line document OCR still needs detector postprocess hardening.",
        "- **ASR risk:** qwen3-ASR requires the base64 `input_audio` path with `<asr_text>`; file URL input remains unsafe unless separately fixed.",
        "- **Realtime retrieval:** embedding and reranker fit the realtime class, but they still need scheduler priority protection from LLM/VLM queues.",
        "",
        "## Preflight Notes",
        "",
        "```text",
        (preflight.get("tcm") or "").strip(),
        "```",
    ]
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def redacted_run_config(args: argparse.Namespace, remote_run_dir: str) -> dict[str, Any]:
    data = {**vars(args), "remote_run_dir": remote_run_dir}
    for key in ("k3_host", "k3_user", "k3_pass"):
        if key in data:
            data[key] = "<redacted>"
    return data


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--k3-host", default=os.environ.get("K3_HOST") or os.environ.get("K3_32G_HOST") or "")
    parser.add_argument("--k3-user", default=os.environ.get("K3_USER") or os.environ.get("K3_32G_USER") or "")
    parser.add_argument("--k3-pass", default=os.environ.get("K3_PASS") or os.environ.get("K3_32G_PASS") or os.environ.get("SSHPASS") or "")
    parser.add_argument("--out-dir", default="")
    parser.add_argument("--remote-run-dir", default="")
    parser.add_argument("--request-timeout", type=int, default=1200)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--resource-interval", type=float, default=5.0)
    parser.add_argument("--llm-contexts", default="1024,3072")
    parser.add_argument("--llm-ctx", type=int, default=4096)
    parser.add_argument("--vlm-cases", type=int, default=10)
    parser.add_argument("--skip-llm", action="store_true")
    parser.add_argument("--skip-vlm", action="store_true")
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--llm-port", type=int, default=18900)
    parser.add_argument("--vlm-port", type=int, default=18910)
    parser.add_argument("--embed-port", type=int, default=18920)
    parser.add_argument("--rerank-port", type=int, default=18921)
    parser.add_argument("--asr-port", type=int, default=18922)
    args = parser.parse_args()
    missing = [name for name, value in (("K3 host", args.k3_host), ("K3 user", args.k3_user), ("K3 password", args.k3_pass)) if not value]
    if missing:
        raise SystemExit(
            "Missing K3 connection setting(s): "
            + ", ".join(missing)
            + ". Provide --k3-host/--k3-user/--k3-pass or K3_HOST/K3_USER/K3_PASS."
        )
    args.llm_contexts = [int(x) for x in str(args.llm_contexts).split(",") if x.strip()]
    return args


def main() -> int:
    args = parse_args()
    stamp = now_stamp()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / f"realistic-stress-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_run_dir = args.remote_run_dir or f"/root/k3_32g_realistic_stress/realistic-stress-{stamp}"
    (out_dir / "run-config.json").write_text(
        json.dumps(redacted_run_config(args, remote_run_dir), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    remote = Remote(args.k3_host, args.k3_user, args.k3_pass)
    preflight = run_preflight(remote, out_dir, remote_run_dir)
    summaries: dict[str, dict[str, Any]] = {}
    try:
        if not args.skip_llm:
            summaries["llm"] = run_llm(remote, out_dir, remote_run_dir, args)
        if not args.skip_vlm:
            summaries["vlm"] = run_vlm(remote, out_dir, remote_run_dir, args)
        if not args.skip_ocr:
            summaries["ocr"] = run_ocr(remote, out_dir, remote_run_dir, args)
        if not args.skip_asr:
            summaries["asr"] = run_asr(remote, out_dir, remote_run_dir, args)
        if not args.skip_embedding:
            summaries["embedding"] = run_embedding(remote, out_dir, remote_run_dir, args)
        if not args.skip_reranker:
            summaries["reranker"] = run_reranker(remote, out_dir, remote_run_dir, args)
    finally:
        remote.run("pkill -f '[l]lama-server.*--port 189' >/dev/null 2>&1 || true; spacemit-tcm-smi -c >/dev/null 2>&1 || true")
        final_state = remote.run("spacemit-tcm-smi 2>&1 || true").stdout
        (out_dir / "tcm-final.txt").write_text(final_state, encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(out_dir, summaries, preflight)
    log(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
