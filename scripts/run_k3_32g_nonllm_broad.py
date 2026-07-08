#!/usr/bin/env python3
"""Broader non-LLM coverage run for K3 32GB.

This runner expands the OCR/ASR/embedding/reranker checks beyond the small
realistic-control sample.  It intentionally runs components sequentially on raw
model servers, so it measures model/runtime behavior but not scheduler fairness.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import random
import re
import shlex
import shutil
import statistics
import subprocess
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import numpy as np
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
MODEL_CACHE = ROOT / "drivers" / "spacemit-ai" / "model_zoo"
DEFAULT_OUT_ROOT = ROOT / "output" / "reports" / "k3-riscv-32g"
REMOTE_MODEL_ROOT = "/root/models/spacemit-ai"
REMOTE_WORK_ROOT = "/root/local-ai-bench/nonllm-broad"
PPOCR_DICT_URL = (
    "https://raw.githubusercontent.com/PaddlePaddle/PaddleOCR/main/"
    "ppocr/utils/dict/ppocrv5_dict.txt"
)


def stamp() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


def log(msg: str) -> None:
    print(f"[{time.strftime('%F %T')}] {msg}", flush=True)


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
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
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
            "UserKnownHostsFile=/dev/null",
            "-o",
            "LogLevel=ERROR",
            "-o",
            "ConnectTimeout=10",
        ]

    def run(self, cmd: str, *, timeout: int | None = None, check: bool = False) -> subprocess.CompletedProcess[str]:
        proc = subprocess.run(
            self.ssh_base() + ["bash -lc " + shlex.quote(cmd)],
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
                self.ssh_base() + ["bash -lc " + shlex.quote(f"tar -xzf - -C {shlex.quote(remote_dir)} --skip-old-files")],
                stdin=f,
                check=True,
                env=self._env(),
            )


class ResourceSampler:
    def __init__(self, remote: Remote, pid: str, label: str, path: Path, interval_s: float) -> None:
        self.remote = remote
        self.pid = pid
        self.label = label
        self.path = path
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
awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {{print}}' /proc/meminfo 2>/dev/null
spacemit-tcm-smi 2>/dev/null | head -20
"""
            proc = self.remote.run(cmd, timeout=15)
            write_jsonl(
                self.path,
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


def parse_max_rss(path: Path) -> dict[str, Any]:
    max_rss = 0
    samples = 0
    if not path.exists():
        return {"max_rss_kb": None, "max_rss_gib": None, "samples": 0}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except Exception:
            continue
        stdout = item.get("stdout") or ""
        for raw in stdout.splitlines():
            parts = raw.split()
            if len(parts) >= 4 and parts[0].isdigit() and parts[3].isdigit():
                max_rss = max(max_rss, int(parts[3]))
                samples += 1
    return {"max_rss_kb": max_rss or None, "max_rss_gib": round(max_rss / 1024 / 1024, 3) if max_rss else None, "samples": samples}


def wait_for_server(host: str, port: int, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"http://{host}:{port}/v1/models", timeout=2.0)
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
    extra_args: list[str],
    ctx: int,
    startup_timeout_s: int,
) -> tuple[str, str]:
    out = f"{run_dir.rstrip('/')}/{label}"
    args = [
        "/usr/bin/llama-server",
        "-m",
        model_path,
        *extra_args,
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
        "512",
        "-ub",
        "256",
        "--no-webui",
        "--no-warmup",
        "--cache-ram",
        "0",
        "--log-file",
        f"{out}/llama-server.log",
    ]
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
setsid env SPACEMIT_DISABLE_TCM=1 {quoted_args} > {shlex.quote(out)}/stdout.log 2> {shlex.quote(out)}/stderr.log < /dev/null &
pid=$!
echo "$pid" > {shlex.quote(out)}/pid
echo "$pid"
"""
    proc = remote.run(cmd, timeout=30, check=True)
    pid = proc.stdout.strip().splitlines()[-1]
    if not wait_for_server(remote.host, port, startup_timeout_s):
        logs = remote.run(f"tail -160 {shlex.quote(out)}/llama-server.log {shlex.quote(out)}/stderr.log 2>/dev/null || true")
        raise RuntimeError(f"{label} server did not become ready on port {port}; pid={pid}\n{logs.stdout}")
    return pid, out


def stop_server(remote: Remote, out: str) -> None:
    cmd = f"""
set +e
if [ -f {shlex.quote(out)}/pid ]; then
  pid=$(cat {shlex.quote(out)}/pid)
  kill "$pid" >/dev/null 2>&1 || true
  sleep 2
  kill -9 "$pid" >/dev/null 2>&1 || true
fi
spacemit-tcm-smi > {shlex.quote(out)}/tcm-after.txt 2>&1 || true
"""
    remote.run(cmd, timeout=30)


def ensure_remote_file(remote: Remote, local: Path, remote_dir: str) -> str:
    if not local.exists():
        raise FileNotFoundError(local)
    remote_path = f"{remote_dir.rstrip('/')}/{local.name}"
    proc = remote.run(f"test -s {shlex.quote(remote_path)}")
    if proc.returncode != 0:
        log(f"copy {local.name} -> {remote_dir}")
        remote.put(local, remote_dir)
    return remote_path


def pick_font(preferred: list[str], size: int) -> ImageFont.FreeTypeFont:
    for p in preferred:
        path = Path(p)
        if path.exists():
            return ImageFont.truetype(str(path), size=size)
    return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=size)


def draw_text_image(text: str, image_path: Path, *, style: str, font_path: str, font_size: int) -> None:
    font = ImageFont.truetype(font_path, size=font_size)
    tmp = Image.new("RGB", (32, 32), "white")
    draw = ImageDraw.Draw(tmp)
    bbox = draw.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    margin_x = 24
    margin_y = 16
    bg = (255, 255, 255)
    fg = (0, 0, 0)
    if "low_contrast" in style:
        bg = (236, 236, 230)
        fg = (94, 94, 88)
    elif "gray" in style:
        bg = (232, 235, 238)
        fg = (24, 27, 31)
    image = Image.new("RGB", (w + margin_x * 2, h + margin_y * 2), bg)
    draw = ImageDraw.Draw(image)
    draw.text((margin_x, margin_y - bbox[1]), text, font=font, fill=fg)
    if "noise" in style:
        arr = np.asarray(image).astype(np.int16)
        rng = np.random.default_rng(20260706 + len(text) + font_size)
        noise = rng.normal(0, 9, arr.shape)
        arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
        image = Image.fromarray(arr, "RGB")
    if "rotate" in style:
        image = image.rotate(2.0, expand=True, fillcolor=bg)
    image.save(image_path)


def generate_ocr_assets(out_dir: Path) -> Path:
    root = out_dir / "generated" / "ocr"
    images = root / "images"
    if (root / "manifest.jsonl").exists():
        return root
    images.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.jsonl"
    font_regular = "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"
    font_bold = "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc"
    font_serif = "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc"
    font_mono = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    styles = [
        ("clean_regular_32", font_regular, 32),
        ("small_regular_22", font_regular, 22),
        ("low_contrast_serif_28", font_serif, 28),
        ("gray_noise_bold_28", font_bold, 28),
        ("rotate2_regular_28", font_regular, 28),
    ]
    texts = [
        ("zh_time_notice", "开放时间早上9点至下午5点"),
        ("zh_contract", "合同编号：2026-BJ-0042"),
        ("zh_tax_invoice", "增值税专用发票  税率13%"),
        ("zh_amount", "总金额（大写）：壹万元整"),
        ("zh_phone", "手机号码：138-0013-8000"),
        ("zh_address", "地址：上海市浦东新区张江高科技园区"),
        ("zh_clause", "备注：请于三个工作日内完成付款"),
        ("en_invoice", "Invoice No. INV-2026-00123"),
        ("en_amount", "Date: 2026-06-17 Amount: $520.00"),
        ("bilingual_name", "客户姓名 / Customer Name: 李明"),
        ("mixed_sku", "SKU: K3-32G-RISCV 批次 A07"),
        ("digits_symbols", "SN: A9Z7-0042  ¥1,280.00"),
    ]
    with manifest.open("w", encoding="utf-8") as f:
        for src in (ROOT / "datasets" / "ocr" / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
            item = json.loads(src)
            uid = f"orig_{item['uid']}"
            dst = images / f"{uid}.png"
            shutil.copy2(ROOT / "datasets" / "ocr" / item["image"], dst)
            f.write(json.dumps({**item, "uid": uid, "image": f"images/{dst.name}", "style": "original_fixture"}, ensure_ascii=False) + "\n")
        for text_id, text in texts:
            for style, font_path, font_size in styles:
                uid = f"{text_id}_{style}"
                dst = images / f"{uid}.png"
                draw_text_image(text, dst, style=style, font_path=font_path if Path(font_path).exists() else font_mono, font_size=font_size)
                f.write(
                    json.dumps(
                        {
                            "uid": uid,
                            "image": f"images/{dst.name}",
                            "text": text,
                            "source": "synthetic-broad",
                            "style": style,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
    return root


def read_wav(path: Path) -> tuple[int, np.ndarray]:
    with wave.open(str(path), "rb") as w:
        if w.getnchannels() != 1 or w.getsampwidth() != 2:
            raise ValueError(f"unsupported wav: {path}")
        sr = w.getframerate()
        data = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2").astype(np.float32) / 32768.0
    return sr, data


def write_wav(path: Path, sr: int, data: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    clipped = np.clip(data, -0.999, 0.999)
    pcm = (clipped * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def speed_resample(data: np.ndarray, factor: float) -> np.ndarray:
    # factor > 1 shortens audio; factor < 1 lengthens audio.
    old = np.arange(len(data))
    new_len = max(1, int(len(data) / factor))
    new_pos = np.linspace(0, len(data) - 1, new_len)
    return np.interp(new_pos, old, data).astype(np.float32)


def add_noise(data: np.ndarray, snr_db: float) -> np.ndarray:
    rng = np.random.default_rng(20260706 + int(snr_db * 10))
    signal_power = float(np.mean(data * data)) or 1e-8
    noise_power = signal_power / (10 ** (snr_db / 10))
    noise = rng.normal(0, math.sqrt(noise_power), data.shape).astype(np.float32)
    return data + noise


def generate_asr_assets(out_dir: Path) -> Path:
    root = out_dir / "generated" / "asr"
    wavs = root / "wavs"
    if (root / "manifest.jsonl").exists():
        return root
    wavs.mkdir(parents=True, exist_ok=True)
    base_dir = ROOT / "datasets" / "asr" / "models" / "sherpa-onnx-sense-voice-zh-en-ja-ko-yue-2024-07-17" / "test_wavs"
    zh_src = base_dir / "zh.wav"
    sr, zh = read_wav(zh_src)
    ref = "开放时间早上9点至下午5点"
    variants: list[tuple[str, np.ndarray, str | None, bool]] = [
        ("zh_original", zh, ref, True),
        ("zh_pad_1s", np.concatenate([np.zeros(sr, dtype=np.float32), zh, np.zeros(sr, dtype=np.float32)]), ref, True),
        ("zh_low_volume_045", zh * 0.45, ref, True),
        ("zh_high_volume_140", zh * 1.40, ref, True),
        ("zh_noise_20db", add_noise(zh, 20.0), ref, True),
        ("zh_noise_10db", add_noise(zh, 10.0), ref, True),
        ("zh_speed_090", speed_resample(zh, 0.90), ref, True),
        ("zh_speed_110", speed_resample(zh, 1.10), ref, True),
    ]
    for lang in ["en", "ja", "yue", "ko"]:
        src = base_dir / f"{lang}.wav"
        if src.exists():
            sr_lang, data = read_wav(src)
            if sr_lang != sr:
                continue
            variants.append((f"{lang}_smoke", data, None, False))
    with (root / "manifest.jsonl").open("w", encoding="utf-8") as f:
        for uid, data, text, score in variants:
            path = wavs / f"{uid}.wav"
            write_wav(path, sr, data)
            f.write(
                json.dumps(
                    {
                        "uid": uid,
                        "audio": f"wavs/{path.name}",
                        "text": text,
                        "score": score,
                        "duration": round(len(data) / sr, 3),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    return root


def ensure_ppocr(remote: Remote) -> None:
    ppocr_dir = MODEL_CACHE / "vision" / "ppocr"
    for filename in ["PP-OCRv5_mobile_det.onnx", "PP-OCRv5_mobile_rec.onnx", "ppocrv5_dict.txt"]:
        ensure_remote_file(remote, ppocr_dir / filename, f"{REMOTE_MODEL_ROOT}/vision/ppocr")


def run_ocr(remote: Remote, out_dir: Path, remote_run_dir: str, local_ocr_root: Path, args: argparse.Namespace) -> dict[str, Any]:
    log("OCR broad test: PP-OCRv5")
    ensure_ppocr(remote)
    remote_parent = f"{REMOTE_WORK_ROOT}/{out_dir.name}"
    remote.put_dir(local_ocr_root, remote_parent)
    remote_ocr_root = f"{remote_parent}/{local_ocr_root.name}"
    code = r'''
import json, math, re, time
from pathlib import Path
import cv2
import numpy as np
import onnxruntime as ort

model_dir = Path("/root/models/spacemit-ai/vision/ppocr")
data_root = Path("__DATA_ROOT__")
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

def rec_pre(img):
    h, w = img.shape[:2]
    img_h = 48
    new_w = max(1, int(math.ceil(img_h * w / h)))
    im = cv2.resize(img, (new_w, img_h))
    x = ((im.astype("float32").transpose(2, 0, 1) / 255.0) - 0.5) / 0.5
    return x[None].astype("float32")

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

refs, hyps, lats, rows = [], [], [], []
print(json.dumps({"phase": "ocr", "case": "providers", "providers": providers, "det_providers": det_sess.get_providers(), "rec_providers": rec_sess.get_providers()}, ensure_ascii=False), flush=True)
for line in (data_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines():
    if not line.strip():
        continue
    item = json.loads(line)
    img = cv2.imread(str(data_root / item["image"]))
    t0 = time.perf_counter()
    det_pixels = det_smoke(img)
    crop = crop_whitespace(img)
    pred = rec_sess.run(None, {rec_sess.get_inputs()[0].name: rec_pre(crop)})[0]
    hyp, conf = decode(pred)
    elapsed_ms = (time.perf_counter() - t0) * 1000
    ref = item["text"]
    ref_n, hyp_n = norm(ref), norm(hyp)
    row = {
        "phase": "ocr",
        "case": "sample",
        "uid": item.get("uid"),
        "style": item.get("style"),
        "text": ref,
        "prediction": hyp,
        "confidence": round(conf, 4),
        "latency_ms": round(elapsed_ms, 2),
        "cer": edit(ref_n, hyp_n) / max(1, len(ref_n)),
        "ned": edit(ref_n, hyp_n) / max(1, max(len(ref_n), len(hyp_n))),
        "det_positive_pixels": det_pixels,
        "crop_shape": list(crop.shape[:2]),
    }
    refs.append(ref)
    hyps.append(hyp)
    lats.append(elapsed_ms)
    rows.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)

total_chars = sum(len(norm(x)) for x in refs)
total_edits = sum(edit(norm(r), norm(h)) for r, h in zip(refs, hyps))
by_style = {}
for row in rows:
    s = row.get("style") or "unknown"
    by_style.setdefault(s, {"n": 0, "cer_sum": 0.0, "lat": []})
    by_style[s]["n"] += 1
    by_style[s]["cer_sum"] += row["cer"]
    by_style[s]["lat"].append(row["latency_ms"])
for s, v in by_style.items():
    v["cer_avg"] = v["cer_sum"] / v["n"]
    v["latency_p95_ms"] = sorted(v["lat"])[min(len(v["lat"]) - 1, math.ceil(len(v["lat"]) * 0.95) - 1)]
    del v["cer_sum"]
    del v["lat"]
summary = {
    "phase": "ocr",
    "case": "summary",
    "model": "PP-OCRv5_mobile_det+rec.onnx",
    "samples": len(refs),
    "cer": total_edits / max(1, total_chars),
    "sample_cer_avg": sum(r["cer"] for r in rows) / len(rows) if rows else None,
    "ned_avg": sum(r["ned"] for r in rows) / len(rows) if rows else None,
    "latency_p50_ms": sorted(lats)[len(lats)//2] if lats else None,
    "latency_p95_ms": sorted(lats)[min(len(lats)-1, math.ceil(len(lats)*0.95)-1)] if lats else None,
    "by_style": by_style,
    "mode": "det smoke + whitespace-crop recognition on expanded single-line OCR set",
}
print(json.dumps(summary, ensure_ascii=False), flush=True)
'''
    code = code.replace("__DATA_ROOT__", remote_ocr_root)
    proc = remote.run(f"python3 - <<'PY'\n{code}\nPY", timeout=args.ocr_timeout, check=True)
    summary: dict[str, Any] = {"phase": "ocr", "error": "summary missing"}
    for line in proc.stdout.splitlines():
        if not line.strip().startswith("{"):
            continue
        item = json.loads(line)
        write_jsonl(out_dir / "trace.jsonl", item)
        if item.get("case") == "summary":
            summary = item
    (out_dir / "ocr-broad-summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "ocr-broad.stderr.log").write_text(proc.stderr, encoding="utf-8")
    return summary


class OpenAIClient:
    def __init__(self, base_url: str, timeout_s: float) -> None:
        self.base_url = base_url.rstrip("/")
        self.client = httpx.Client(timeout=timeout_s)

    def embeddings(self, model: str, texts: list[str]) -> tuple[list[list[float]], dict[str, Any]]:
        t0 = time.perf_counter()
        r = self.client.post(f"{self.base_url}/embeddings", json={"model": model, "input": texts})
        elapsed = time.perf_counter() - t0
        item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_ms": round(elapsed * 1000, 2), "n_inputs": len(texts)}
        try:
            body = r.json()
            vecs = [x.get("embedding", []) for x in body.get("data", [])]
            item["dim"] = len(vecs[0]) if vecs else 0
            item["usage"] = body.get("usage")
            return vecs, item
        except Exception as exc:
            item["error"] = str(exc)
            item["body_prefix"] = r.text[:1000]
            return [], item

    def rerank(self, model: str, query: str, documents: list[str]) -> dict[str, Any]:
        t0 = time.perf_counter()
        r = self.client.post(
            f"{self.base_url}/rerank",
            json={"model": model, "query": query, "documents": documents, "return_documents": False},
        )
        elapsed = time.perf_counter() - t0
        item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_ms": round(elapsed * 1000, 2)}
        try:
            item["body"] = r.json()
        except Exception as exc:
            item["error"] = str(exc)
            item["body_prefix"] = r.text[:1000]
        return item

    def chat_audio(self, model: str, wav_path: Path, prompt: str) -> dict[str, Any]:
        data = base64.b64encode(wav_path.read_bytes()).decode("ascii")
        payload = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "input_audio", "input_audio": {"data": data, "format": "wav"}},
                    ],
                }
            ],
            "temperature": 0,
            "max_tokens": 128,
        }
        t0 = time.perf_counter()
        r = self.client.post(f"{self.base_url}/chat/completions", json=payload)
        elapsed = time.perf_counter() - t0
        item: dict[str, Any] = {"status_code": r.status_code, "ok": r.status_code == 200, "elapsed_s": round(elapsed, 3)}
        try:
            body = r.json()
            item["content"] = body.get("choices", [{}])[0].get("message", {}).get("content") or ""
            item["usage"] = body.get("usage")
        except Exception as exc:
            item["error"] = str(exc)
            item["body_prefix"] = r.text[:1000]
        return item


def finite_vector_stats(vecs: list[list[float]]) -> dict[str, Any]:
    total = 0
    finite = 0
    nan = 0
    inf = 0
    for vec in vecs:
        for x in vec:
            total += 1
            if isinstance(x, (int, float)) and math.isfinite(float(x)):
                finite += 1
            elif isinstance(x, float) and math.isnan(x):
                nan += 1
            else:
                inf += 1
    return {"values": total, "finite": finite, "nan": nan, "inf_or_invalid": inf, "finite_ratio": finite / total if total else 0.0}


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return -1.0
    if any(not math.isfinite(float(x)) for x in a + b):
        return -1.0
    dot = sum(float(x) * float(y) for x, y in zip(a, b))
    na = math.sqrt(sum(float(x) * float(x) for x in a))
    nb = math.sqrt(sum(float(y) * float(y) for y in b))
    return dot / (na * nb) if na and nb else -1.0


def retrieval_sets() -> dict[str, dict[str, Any]]:
    zh_docs = [
        ("zh_d0", "北京是中国的首都，也是政治和文化中心。"),
        ("zh_d1", "上海是重要的金融中心，拥有繁忙的港口。"),
        ("zh_d2", "Python 常用于数据分析、自动化和机器学习。"),
        ("zh_d3", "端侧 AI 推理关注吞吐、首 token 延迟、内存和稳定性。"),
        ("zh_d4", "OCR 会把图片中的文字识别为可检索文本。"),
        ("zh_d5", "语音识别系统会把音频转换成文字。"),
        ("zh_d6", "向量数据库通过相似度检索相关文档。"),
        ("zh_d7", "重排序模型会对候选文档进行精排。"),
        ("zh_d8", "K3 32G 平台需要关注内存余量和磁盘空间。"),
        ("zh_d9", "异步队列可以隔离长文本和多模态任务。"),
        ("zh_d10", "发票号码和金额是票据 OCR 的关键字段。"),
        ("zh_d11", "ASR 对噪声、音量和语速变化比较敏感。"),
    ]
    zh_queries = [
        ("中国首都是哪里", "zh_d0"),
        ("哪座城市是金融中心", "zh_d1"),
        ("Python 可以做什么", "zh_d2"),
        ("边缘推理要看哪些性能指标", "zh_d3"),
        ("图片文字识别是什么任务", "zh_d4"),
        ("音频转文字是什么", "zh_d5"),
        ("向量数据库如何找文档", "zh_d6"),
        ("候选文档精排用什么模型", "zh_d7"),
        ("K3 32G 需要关注哪些资源", "zh_d8"),
        ("长文本任务应该如何隔离", "zh_d9"),
        ("票据 OCR 要抽取什么", "zh_d10"),
        ("语音识别受什么影响", "zh_d11"),
    ]
    en_docs = [
        ("en_d0", "Paris is the capital of France and a major cultural center."),
        ("en_d1", "Python is widely used for automation, data science, and machine learning."),
        ("en_d2", "Optical character recognition extracts text from images."),
        ("en_d3", "Speech recognition converts audio into text."),
        ("en_d4", "Embedding models map text into vectors for retrieval."),
        ("en_d5", "Rerankers sort candidate passages by relevance."),
        ("en_d6", "Long context language model requests can consume significant memory."),
        ("en_d7", "Asynchronous queues protect realtime requests from slow jobs."),
    ]
    en_queries = [
        ("What city is the capital of France?", "en_d0"),
        ("What is Python used for?", "en_d1"),
        ("What does OCR extract?", "en_d2"),
        ("What converts audio to text?", "en_d3"),
        ("What maps text to vectors?", "en_d4"),
        ("What sorts candidate passages?", "en_d5"),
        ("Why are long context requests risky?", "en_d6"),
        ("How do queues protect realtime requests?", "en_d7"),
    ]
    mixed_docs = [
        ("mx_d0", "客户姓名 / Customer Name: 李明"),
        ("mx_d1", "Invoice No. INV-2026-00123 contains the invoice identifier."),
        ("mx_d2", "开放时间 Open hours are 9:00 to 17:00."),
        ("mx_d3", "K3 32G edge inference requires memory and queue monitoring."),
        ("mx_d4", "ASR 音频识别 should normalize Traditional and Simplified Chinese."),
        ("mx_d5", "OCR 文档识别 should use PP-OCR instead of a VLM substitute."),
    ]
    mixed_queries = [
        ("Customer Name 是谁", "mx_d0"),
        ("Which document has invoice identifier?", "mx_d1"),
        ("开放时间是什么", "mx_d2"),
        ("K3 edge inference 要监控什么", "mx_d3"),
        ("ASR 中文评分要做什么", "mx_d4"),
        ("OCR 应该用什么模型", "mx_d5"),
    ]
    return {
        "zh": {"docs": zh_docs, "queries": zh_queries},
        "en": {"docs": en_docs, "queries": en_queries},
        "mixed": {"docs": mixed_docs, "queries": mixed_queries},
    }


def run_embedding_models(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    models = [
        ("Bge-Small-Zh-V1.5-Q4_K_M.gguf", "embed", "BGE-Zh"),
        ("Bge-Small-En-V1.5-Q4_K_M.gguf", "embed", "BGE-En"),
        ("Jina-Embeddings-V5-Text-Small-Retrieval-Q4_K_M.gguf", "embed", "Jina"),
        ("Nomic-Embed-Text-V2-Moe-Q4_0.gguf", "embed", "Nomic"),
        ("Qwen3-Embedding-0.6B-Q4_0.gguf", "embed", "Qwen3-Embedding"),
    ]
    summaries: list[dict[str, Any]] = []
    sets = retrieval_sets()
    batch_texts = [f"批量 embedding 压测文本 {i}: K3 32G retrieval workload sample." for i in range(64)]
    for filename, category, label in models:
        local = MODEL_CACHE / category / filename
        if not local.exists():
            summaries.append({"phase": "embedding", "model": filename, "status": "missing-local-cache"})
            continue
        log(f"Embedding broad test: {filename}")
        remote_path = ensure_remote_file(remote, local, f"{REMOTE_MODEL_ROOT}/{category}")
        server_out = ""
        sampler: ResourceSampler | None = None
        summary: dict[str, Any] = {"phase": "embedding", "model": filename, "label": label}
        try:
            pid, server_out = start_server(
                remote,
                run_dir=remote_run_dir,
                label=f"embed-{label.lower().replace('_','-')}",
                port=args.embed_port,
                model_path=remote_path,
                alias=filename,
                extra_args=["--embedding", "--pooling", "mean"],
                ctx=512,
                startup_timeout_s=args.startup_timeout,
            )
            sampler_path = out_dir / "resource" / f"embed-{label}.jsonl"
            sampler = ResourceSampler(remote, pid, f"embed-{label}", sampler_path, args.resource_interval)
            sampler.start()
            client = OpenAIClient(f"http://{remote.host}:{args.embed_port}/v1", args.request_timeout)
            set_results: dict[str, Any] = {}
            all_lat: list[float] = []
            bad_vectors = 0
            for set_name, data in sets.items():
                docs = data["docs"]
                queries = data["queries"]
                doc_vecs, doc_item = client.embeddings(filename, [x[1] for x in docs])
                stats = finite_vector_stats(doc_vecs)
                if stats["finite_ratio"] < 1.0:
                    bad_vectors += 1
                write_jsonl(out_dir / "trace.jsonl", {"phase": "embedding", "model": filename, "case": "docs", "set": set_name, **doc_item, "vector_stats": stats})
                hits = 0
                rrs: list[float] = []
                ndcgs: list[float] = []
                lat: list[float] = []
                for query, expected in queries:
                    vecs, item = client.embeddings(filename, [query])
                    q_stats = finite_vector_stats(vecs)
                    if q_stats["finite_ratio"] < 1.0:
                        bad_vectors += 1
                    lat.append(item["elapsed_ms"])
                    all_lat.append(item["elapsed_ms"])
                    rank = 999
                    top: list[tuple[str, float]] = []
                    if doc_vecs and vecs and stats["finite_ratio"] == 1.0 and q_stats["finite_ratio"] == 1.0:
                        scores = [(docs[i][0], cosine(vecs[0], doc_vecs[i])) for i in range(len(docs))]
                        scores.sort(key=lambda x: x[1], reverse=True)
                        ids = [doc_id for doc_id, _ in scores]
                        if expected in ids:
                            rank = ids.index(expected) + 1
                        top = scores[:3]
                    hits += int(rank == 1)
                    if rank != 999:
                        rrs.append(1.0 / rank)
                        ndcgs.append(1.0 / math.log2(rank + 1))
                    write_jsonl(
                        out_dir / "trace.jsonl",
                        {
                            "phase": "embedding",
                            "model": filename,
                            "case": "query",
                            "set": set_name,
                            "query": query,
                            "expected": expected,
                            "rank": rank,
                            "top": top,
                            **item,
                            "vector_stats": q_stats,
                        },
                    )
                set_results[set_name] = {
                    "queries": len(queries),
                    "hit_at_1": hits / len(queries),
                    "mrr": sum(rrs) / len(rrs) if rrs else 0,
                    "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
                    "latency_p50_ms": percentile(lat, 0.50),
                    "latency_p95_ms": percentile(lat, 0.95),
                }
            batch_results = []
            for n in [1, 8, 32, 64]:
                vecs, item = client.embeddings(filename, batch_texts[:n])
                stats = finite_vector_stats(vecs)
                batch = {"n": n, "elapsed_ms": item["elapsed_ms"], "per_text_ms": item["elapsed_ms"] / n, "vector_stats": stats}
                batch_results.append(batch)
                write_jsonl(out_dir / "trace.jsonl", {"phase": "embedding", "model": filename, "case": "batch", **batch, "status_code": item.get("status_code")})
            if sampler:
                sampler.stop()
                sampler = None
            summary.update(
                {
                    "status": "ok",
                    "sets": set_results,
                    "overall_hit_at_1": statistics.mean([v["hit_at_1"] for v in set_results.values()]),
                    "latency_p50_ms": percentile(all_lat, 0.50),
                    "latency_p95_ms": percentile(all_lat, 0.95),
                    "batch": batch_results,
                    "bad_vector_batches": bad_vectors,
                    "resource": parse_max_rss(sampler_path),
                }
            )
        except Exception as exc:
            summary.update({"status": "error", "error": str(exc)})
        finally:
            if sampler:
                sampler.stop()
            if server_out:
                stop_server(remote, server_out)
        write_jsonl(out_dir / "embedding-broad-summary.jsonl", summary)
        summaries.append(summary)
    return summaries


def rerank_cases() -> list[dict[str, Any]]:
    distractors = [
        "上海拥有繁忙的港口和金融机构。",
        "Python 是一种常用编程语言。",
        "咖啡豆需要研磨后冲泡。",
        "电池容量通常用 mAh 表示。",
        "图像分类模型会输出类别标签。",
        "天气预报包含温度和降水概率。",
        "Wi-Fi 和蜂窝网络都能提供无线连接。",
        "数据库索引用于加速查询。",
        "合同通常包含甲方乙方和金额。",
        "发票号码可以用于财务核验。",
        "Paris is a major cultural center in France.",
        "Python is used for automation and machine learning.",
        "Long context requests can consume substantial memory.",
        "Asynchronous jobs can isolate slow multimodal tasks.",
        "OCR extracts text from images.",
        "Speech recognition converts audio into text.",
        "Vector search retrieves similar passages.",
        "A reranker sorts candidate documents by relevance.",
        "Invoices contain numbers, dates, and payment amounts.",
        "Queue backpressure protects realtime services.",
        "K3 32G needs memory and disk headroom monitoring.",
        "Traditional and Simplified Chinese normalization affects ASR scoring.",
        "Document upload should not block realtime search.",
        "Batch size affects embedding throughput.",
        "Candidate count affects reranker latency.",
        "Model warmup can change first request latency.",
        "A gateway can expose capacity and job status.",
        "Cancellation and TTL prevent stale long-running jobs.",
        "Noise can reduce ASR recognition quality.",
        "Low contrast text can reduce OCR accuracy.",
        "The report records p50 and p95 latency.",
        "TCM state must be logged before and after runs.",
        "The root filesystem can fill up with extracted models.",
        "Small VLMs may pass runtime but fail extraction quality.",
        "Embedding dimensions should not contain NaN values.",
        "Reranking top 50 is more expensive than top 10.",
        "Chinese OCR includes amounts and invoice fields.",
        "English OCR includes invoice IDs and dates.",
        "Bilingual documents mix Chinese and English fields.",
        "Resource sampling tracks RSS and CPU use.",
        "OpenAI-compatible APIs simplify probes.",
        "Raw model servers cannot prove scheduler fairness.",
        "Long prompt prefill can dominate latency.",
        "Decoder speed affects completion latency.",
        "ASR RTF compares elapsed time with audio duration.",
        "The selected OCR path is PP-OCRv5.",
        "The selected default reranker is BGE v2 m3.",
        "The selected default Chinese embedding is BGE small zh.",
        "Qwen reranker is higher latency on K3.",
        "Jina and Nomic embeddings are alternative specs.",
    ]
    return [
        {
            "set": "zh",
            "query": "OCR 应该识别图片中的什么内容",
            "relevant": "OCR 会把图片中的文字识别为可检索文本。",
            "distractors": distractors,
        },
        {
            "set": "zh",
            "query": "长文本请求为什么需要异步队列",
            "relevant": "长文本 LLM 请求会占用大量内存和时间，需要异步队列隔离。",
            "distractors": distractors,
        },
        {
            "set": "zh",
            "query": "K3 32G 风险需要关注哪些资源",
            "relevant": "K3 32G 风险评估需要关注内存余量、磁盘空间、队列等待和取消机制。",
            "distractors": distractors,
        },
        {
            "set": "en",
            "query": "What does a reranker do?",
            "relevant": "A reranker sorts candidate documents by relevance after retrieval.",
            "distractors": distractors,
        },
        {
            "set": "en",
            "query": "Why are long context LLM requests risky?",
            "relevant": "Long context language model requests can consume significant memory and block realtime work.",
            "distractors": distractors,
        },
        {
            "set": "mixed",
            "query": "OCR 文档识别应该用什么模型",
            "relevant": "OCR 文档识别 should use PP-OCRv5 instead of a VLM substitute.",
            "distractors": distractors,
        },
    ]


def run_reranker_models(remote: Remote, out_dir: Path, remote_run_dir: str, args: argparse.Namespace) -> list[dict[str, Any]]:
    models = [
        ("Bge-Reranker-V2-M3-Q4_0.gguf", "rerank", "BGE-Reranker"),
        ("Qwen3-Reranker-0.6B-Q4_0.gguf", "rerank", "Qwen3-Reranker"),
    ]
    cases = rerank_cases()
    candidate_sizes = [3, 10, 20, 50]
    summaries: list[dict[str, Any]] = []
    for filename, category, label in models:
        local = MODEL_CACHE / category / filename
        if not local.exists():
            summaries.append({"phase": "reranker", "model": filename, "status": "missing-local-cache"})
            continue
        log(f"Reranker broad test: {filename}")
        remote_path = ensure_remote_file(remote, local, f"{REMOTE_MODEL_ROOT}/{category}")
        server_out = ""
        sampler: ResourceSampler | None = None
        summary: dict[str, Any] = {"phase": "reranker", "model": filename, "label": label}
        try:
            pid, server_out = start_server(
                remote,
                run_dir=remote_run_dir,
                label=f"rerank-{label.lower().replace('_','-')}",
                port=args.rerank_port,
                model_path=remote_path,
                alias=filename,
                extra_args=["--reranking", "--pooling", "rank"],
                ctx=1024,
                startup_timeout_s=args.startup_timeout,
            )
            sampler_path = out_dir / "resource" / f"rerank-{label}.jsonl"
            sampler = ResourceSampler(remote, pid, f"rerank-{label}", sampler_path, args.resource_interval)
            sampler.start()
            client = OpenAIClient(f"http://{remote.host}:{args.rerank_port}/v1", args.request_timeout)
            by_size: dict[str, Any] = {}
            all_lat: list[float] = []
            for size in candidate_sizes:
                hits = 0
                rrs: list[float] = []
                ndcgs: list[float] = []
                lat: list[float] = []
                for case in cases:
                    docs = [case["relevant"]] + case["distractors"][: size - 1]
                    # Deterministic shuffle so relevant is not always index 0.
                    rng = random.Random(f"{filename}-{size}-{case['query']}")
                    rng.shuffle(docs)
                    expected = docs.index(case["relevant"])
                    item = client.rerank(filename, case["query"], docs)
                    ranked = [x.get("index") for x in item.get("body", {}).get("results", [])]
                    rank = ranked.index(expected) + 1 if expected in ranked else 999
                    hits += int(rank == 1)
                    if rank != 999:
                        rrs.append(1.0 / rank)
                        ndcgs.append(1.0 / math.log2(rank + 1))
                    lat.append(item["elapsed_ms"])
                    all_lat.append(item["elapsed_ms"])
                    write_jsonl(
                        out_dir / "trace.jsonl",
                        {
                            "phase": "reranker",
                            "model": filename,
                            "case": "query",
                            "set": case["set"],
                            "candidate_size": size,
                            "query": case["query"],
                            "expected_index": expected,
                            "rank": rank,
                            **item,
                        },
                    )
                by_size[str(size)] = {
                    "queries": len(cases),
                    "hit_at_1": hits / len(cases),
                    "mrr": sum(rrs) / len(rrs) if rrs else 0,
                    "ndcg": sum(ndcgs) / len(ndcgs) if ndcgs else 0,
                    "latency_p50_ms": percentile(lat, 0.50),
                    "latency_p95_ms": percentile(lat, 0.95),
                }
            if sampler:
                sampler.stop()
                sampler = None
            summary.update(
                {
                    "status": "ok",
                    "by_candidate_size": by_size,
                    "overall_hit_at_1": statistics.mean([v["hit_at_1"] for v in by_size.values()]),
                    "latency_p50_ms": percentile(all_lat, 0.50),
                    "latency_p95_ms": percentile(all_lat, 0.95),
                    "resource": parse_max_rss(sampler_path),
                }
            )
        except Exception as exc:
            summary.update({"status": "error", "error": str(exc)})
        finally:
            if sampler:
                sampler.stop()
            if server_out:
                stop_server(remote, server_out)
        write_jsonl(out_dir / "reranker-broad-summary.jsonl", summary)
        summaries.append(summary)
    return summaries


def ensure_asr_package(remote: Remote, name: str, local_tar: Path, remote_category: str) -> tuple[str, str, str]:
    base = f"{REMOTE_MODEL_ROOT}/{remote_category}"
    name_l = name.lower()
    existing_dirs = remote.run(f"find {shlex.quote(base)} -maxdepth 5 -type d 2>/dev/null || true").stdout.lower()
    if local_tar.exists() and name_l not in existing_dirs:
        # The package may already be extracted under a vendor-specific name, so
        # the find below remains authoritative after this best-effort extract.
        log(f"stream extract ASR package {local_tar.name}")
        remote.stream_extract_tar(local_tar, base)
    probe = remote.run(f"find {shlex.quote(base)} -type f -name 'config.json' -printf '%h\\n' 2>/dev/null || true")
    config_dirs = [x.strip() for x in probe.stdout.splitlines() if x.strip()]
    tokens = [x for x in re.split(r"[^0-9a-zA-Z.]+", name_l) if x]
    config_dir = ""
    for candidate in config_dirs:
        c = candidate.lower()
        if all(token in c for token in tokens if token not in {"qwen3", "asr"}):
            config_dir = candidate
            break
    if not config_dir:
        for candidate in config_dirs:
            if "asr" in candidate.lower():
                config_dir = candidate
                break
    if not config_dir and config_dirs:
        config_dir = config_dirs[0]
    model_probe = remote.run(f"find {shlex.quote(config_dir or base)} -type f -iname '*text*.gguf' 2>/dev/null | head -1")
    model_path = model_probe.stdout.strip()
    if not config_dir or not model_path:
        raise RuntimeError(f"ASR package missing config/text gguf for {name}")
    return config_dir, model_path, base


def copy_asr_assets(remote: Remote, local_asr_root: Path, out_dir: Path) -> str:
    remote_parent = f"{REMOTE_WORK_ROOT}/{out_dir.name}"
    remote.put_dir(local_asr_root, remote_parent)
    return f"{remote_parent}/{local_asr_root.name}"


def run_asr_models(remote: Remote, out_dir: Path, remote_run_dir: str, local_asr_root: Path, args: argparse.Namespace) -> list[dict[str, Any]]:
    remote_asr_root = copy_asr_assets(remote, local_asr_root, out_dir)
    local_manifest = [json.loads(x) for x in (local_asr_root / "manifest.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]
    models = [
        {
            "label": "qwen3-asr-0.6B",
            "alias": "qwen3-asr-0.6B",
            "local_tar": MODEL_CACHE / "vlm" / "qwen3-asr-0.6B.tar.gz",
            "remote_category": "vlm",
            "match": "qwen3-asr-0.6B",
        }
    ]
    if args.include_asr_17b:
        models.append(
            {
                "label": "qwen3-asr-1.7B-q4km",
                "alias": "qwen3-asr-1.7B-q4km",
                "local_tar": MODEL_CACHE / "asr" / "qwen3-asr-1.7B-dynq-q4km.tar.gz",
                "remote_category": "asr",
                "match": "qwen3-asr-1.7B",
            }
        )
    summaries: list[dict[str, Any]] = []
    for model in models:
        log(f"ASR broad test: {model['label']}")
        summary: dict[str, Any] = {"phase": "asr", "model": model["label"]}
        server_out = ""
        sampler: ResourceSampler | None = None
        try:
            config_dir, model_path, _ = ensure_asr_package(remote, model["match"], model["local_tar"], model["remote_category"])
            pid, server_out = start_server(
                remote,
                run_dir=remote_run_dir,
                label=f"asr-{model['label']}",
                port=args.asr_port,
                model_path=model_path,
                alias=model["alias"],
                extra_args=["--media-backend", "smt", "--smt-config-dir", config_dir],
                ctx=2048,
                startup_timeout_s=args.startup_timeout,
            )
            sampler_path = out_dir / "resource" / f"asr-{model['label']}.jsonl"
            sampler = ResourceSampler(remote, pid, f"asr-{model['label']}", sampler_path, args.resource_interval)
            sampler.start()
            client = OpenAIClient(f"http://{remote.host}:{args.asr_port}/v1", args.request_timeout)
            rows = []
            for sample in local_manifest:
                wav_path = local_asr_root / sample["audio"]
                item = client.chat_audio(model["alias"], wav_path, "<asr_text>")
                pred = item.get("content", "")
                row = {
                    "phase": "asr",
                    "model": model["label"],
                    "case": "sample",
                    "uid": sample["uid"],
                    "duration_s": sample["duration"],
                    "score": sample["score"],
                    "text": sample.get("text"),
                    "prediction": pred,
                    "rtf": (float(item.get("elapsed_s") or 0) / float(sample["duration"] or 1)),
                    **item,
                }
                if sample.get("score") and sample.get("text"):
                    row["cer"] = cer(sample["text"], pred)
                    row["normalized_cer"] = zh_asr_cer(sample["text"], pred)
                rows.append(row)
                write_jsonl(out_dir / "trace.jsonl", row)
            if sampler:
                sampler.stop()
                sampler = None
            scored = [r for r in rows if r.get("score")]
            rtf = [float(r["rtf"]) for r in rows if r.get("rtf") is not None]
            summary.update(
                {
                    "status": "ok",
                    "samples": len(rows),
                    "scored_samples": len(scored),
                    "raw_cer_avg": statistics.mean([r["cer"] for r in scored]) if scored else None,
                    "normalized_cer_avg": statistics.mean([r["normalized_cer"] for r in scored]) if scored else None,
                    "rtf_p50": percentile(rtf, 0.50),
                    "rtf_p95": percentile(rtf, 0.95),
                    "resource": parse_max_rss(sampler_path),
                }
            )
        except Exception as exc:
            summary.update({"status": "error", "error": str(exc)})
        finally:
            if sampler:
                sampler.stop()
            if server_out:
                stop_server(remote, server_out)
        write_jsonl(out_dir / "asr-broad-summary.jsonl", summary)
        summaries.append(summary)
    if args.cleanup_asr_17b:
        remote.run(f"rm -rf {shlex.quote(REMOTE_MODEL_ROOT + '/asr/qwen3-asr-1.7B-dynq-q4km')} {shlex.quote(REMOTE_MODEL_ROOT + '/asr/Qwen3-ASR-1.7B')} >/dev/null 2>&1 || true")
    return summaries


def render_report(out_dir: Path, summaries: dict[str, Any], preflight: dict[str, Any]) -> None:
    lines = [
        "# K3 32G Non-LLM Broad Coverage Run",
        "",
        f"Run directory: `{out_dir}`",
        "",
        "## Scope",
        "",
        "- OCR: expanded synthetic line OCR set with original fixtures, multiple fonts, small text, low contrast, noise, and slight rotation.",
        "- ASR: qwen3-ASR on Chinese perturbations plus multilingual smoke wavs.",
        "- Embedding: BGE-Zh, BGE-En, Jina, Nomic, and Qwen3 embedding across zh/en/mixed retrieval and batch sizes.",
        "- Reranker: BGE and Qwen3 rerankers across zh/en/mixed queries and candidate sizes 3/10/20/50.",
        "- Raw model servers only; this does not prove scheduler admission or mixed-load priority.",
        "",
        "## Summary",
        "",
        "### OCR",
        "",
    ]
    ocr = summaries.get("ocr", {})
    if ocr:
        lines += [
            f"- Model: `{ocr.get('model')}`",
            f"- Samples: {ocr.get('samples')}, CER {ocr.get('cer'):.4f}, sample-CER avg {ocr.get('sample_cer_avg'):.4f}, NED avg {ocr.get('ned_avg'):.4f}",
            f"- Latency p50/p95: {ocr.get('latency_p50_ms'):.1f}ms / {ocr.get('latency_p95_ms'):.1f}ms",
            "",
        ]
    lines += ["### ASR", "", "| Model | Samples | Scored | Normalized CER avg | RTF p50 / p95 | RSS GiB | Status |", "|---|---:|---:|---:|---:|---:|---|"]
    for item in summaries.get("asr", []):
        res = item.get("resource") or {}
        lines.append(
            f"| `{item.get('model')}` | {item.get('samples')} | {item.get('scored_samples')} | {item.get('normalized_cer_avg')} | {item.get('rtf_p50')} / {item.get('rtf_p95')} | {res.get('max_rss_gib')} | {item.get('status')} |"
        )
    lines += ["", "### Embedding", "", "| Model | Overall Hit@1 | p50 / p95 ms | Batch64 per text ms | Bad vector batches | RSS GiB | Status |", "|---|---:|---:|---:|---:|---:|---|"]
    for item in summaries.get("embedding", []):
        batch64 = next((b for b in item.get("batch", []) if b.get("n") == 64), {})
        res = item.get("resource") or {}
        lines.append(
            f"| `{item.get('model')}` | {item.get('overall_hit_at_1')} | {item.get('latency_p50_ms')} / {item.get('latency_p95_ms')} | {batch64.get('per_text_ms')} | {item.get('bad_vector_batches')} | {res.get('max_rss_gib')} | {item.get('status')} |"
        )
    lines += ["", "### Reranker", "", "| Model | Overall Hit@1 | p50 / p95 ms | top50 p95 ms | RSS GiB | Status |", "|---|---:|---:|---:|---:|---|"]
    for item in summaries.get("reranker", []):
        top50 = (item.get("by_candidate_size") or {}).get("50", {})
        res = item.get("resource") or {}
        lines.append(
            f"| `{item.get('model')}` | {item.get('overall_hit_at_1')} | {item.get('latency_p50_ms')} / {item.get('latency_p95_ms')} | {top50.get('latency_p95_ms')} | {res.get('max_rss_gib')} | {item.get('status')} |"
        )
    lines += [
        "",
        "## Risk Notes",
        "",
        "- Embedding and reranker latency remains small compared with VLM/long-LLM, but scheduler priority is still required under mixed load.",
        "- Reranker latency scales with candidate count; use top20 as a safer default unless top50 latency is acceptable for the product path.",
        "- PP-OCRv5 line OCR is usable, but rotated/low-contrast samples expose the margin where detector postprocess and image normalization matter.",
        "- ASR scoring must normalize Traditional/Simplified Chinese and Chinese numerals; noisy or speed-shifted audio should be evaluated by normalized CER and RTF together.",
        "- Current run is sequential. It cannot replace the mixed workflow stress plan with `/capacity`, `/jobs`, cancellation, TTL, and backpressure.",
        "",
        "## Preflight",
        "",
        "```text",
        (preflight.get("df") or "").strip(),
        "",
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
    parser.add_argument("--request-timeout", type=int, default=600)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--ocr-timeout", type=int, default=900)
    parser.add_argument("--resource-interval", type=float, default=5.0)
    parser.add_argument("--embed-port", type=int, default=19020)
    parser.add_argument("--rerank-port", type=int, default=19021)
    parser.add_argument("--asr-port", type=int, default=19022)
    parser.add_argument("--skip-ocr", action="store_true")
    parser.add_argument("--skip-asr", action="store_true")
    parser.add_argument("--skip-embedding", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--include-asr-17b", action="store_true")
    parser.add_argument("--cleanup-asr-17b", action="store_true", default=True)
    args = parser.parse_args()
    missing = [name for name, value in (("K3 host", args.k3_host), ("K3 user", args.k3_user), ("K3 password", args.k3_pass)) if not value]
    if missing:
        raise SystemExit(
            "Missing K3 connection setting(s): "
            + ", ".join(missing)
            + ". Provide --k3-host/--k3-user/--k3-pass or K3_HOST/K3_USER/K3_PASS."
        )
    return args


def main() -> int:
    args = parse_args()
    run_stamp = stamp()
    out_dir = Path(args.out_dir) if args.out_dir else DEFAULT_OUT_ROOT / f"nonllm-broad-{run_stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    remote_run_dir = args.remote_run_dir or f"/root/k3_32g_nonllm_broad/nonllm-broad-{run_stamp}"
    remote = Remote(args.k3_host, args.k3_user, args.k3_pass)
    (out_dir / "run-config.json").write_text(json.dumps(redacted_run_config(args, remote_run_dir), ensure_ascii=False, indent=2), encoding="utf-8")
    preflight = {
        "df": remote.run("df -h /root /root/models 2>/dev/null || df -h /root || true").stdout,
        "meminfo": remote.run("awk '/MemTotal|MemAvailable|SwapTotal|SwapFree/ {print}' /proc/meminfo || true").stdout,
        "tcm": remote.run("spacemit-tcm-smi 2>&1 || true").stdout,
        "versions": remote.run("dpkg -l | grep -E 'llama.cpp-tools-spacemit|spacemit-onnxruntime|spacemit-tcm' || true").stdout,
    }
    (out_dir / "preflight.json").write_text(json.dumps(preflight, ensure_ascii=False, indent=2), encoding="utf-8")
    ocr_root = generate_ocr_assets(out_dir)
    asr_root = generate_asr_assets(out_dir)
    summaries: dict[str, Any] = {}
    try:
        if not args.skip_ocr:
            summaries["ocr"] = run_ocr(remote, out_dir, remote_run_dir, ocr_root, args)
        if not args.skip_asr:
            summaries["asr"] = run_asr_models(remote, out_dir, remote_run_dir, asr_root, args)
        if not args.skip_embedding:
            summaries["embedding"] = run_embedding_models(remote, out_dir, remote_run_dir, args)
        if not args.skip_reranker:
            summaries["reranker"] = run_reranker_models(remote, out_dir, remote_run_dir, args)
    finally:
        remote.run("pkill -f '[l]lama-server.*--port 190' >/dev/null 2>&1 || true; spacemit-tcm-smi -c >/dev/null 2>&1 || true")
        final = {
            "df": remote.run("df -h /root /root/models 2>/dev/null || df -h /root || true").stdout,
            "tcm": remote.run("spacemit-tcm-smi 2>&1 || true").stdout,
            "llama": remote.run("pgrep -a llama-server || true").stdout,
        }
        (out_dir / "final-state.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(summaries, ensure_ascii=False, indent=2), encoding="utf-8")
    render_report(out_dir, summaries, preflight)
    log(f"done: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
