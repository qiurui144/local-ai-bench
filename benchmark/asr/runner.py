"""ASR benchmark runner — CER (Chinese) / WER / RTF over an audio manifest.

The transcription backend is **ONNX-based** (e.g. sherpa-onnx SenseVoice, the
model the K23 eval used to hit CER 1.17 % / RTF 0.086). Because ONNX runtimes
and the model checkpoint are large optional dependencies, the backend is
loaded lazily and the dimension **degrades gracefully**:

- no manifest (no dataset)            → ``status: "blocked", reason: "no dataset"``
- sherpa-onnx / model missing         → ``status: "blocked", reason: "no asr backend"``
- both present                        → real transcription + CER/WER/RTF verdict

A custom transcriber can be injected (``transcribe_fn``) for testing or for a
different ONNX pipeline, so the scoring/aggregation logic is exercised on CPU
without shipping a model.
"""

from __future__ import annotations

import logging
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Optional

from common import ModelConfig, summarize_latencies

from .datasets import AudioSample, load_asr_manifest, wav_duration_s
from .metrics import corpus_cer, corpus_wer, rtf, validate_transcript

logger = logging.getLogger(__name__)

# A transcriber takes an audio Path and returns the decoded text. Subprocess
# helpers may return their JSON body so runtime/provider metadata can be kept.
Transcriber = Callable[[Path], str | dict[str, Any]]

_DEFAULT_THRESHOLDS = {
    "cer_max": 0.15,     # 15 % CER ceiling for clean Chinese speech
    "rtf_max": 1.0,      # must be real-time capable
}


def _http_transcriber(base_url: str) -> Transcriber:
    """HTTP transcriber: POST raw WAV to <base_url>/asr/transcribe."""
    import httpx

    url = base_url.rstrip("/")
    # Strip /v1 suffix — rk-asr endpoint is at /asr/transcribe, not /v1/asr/...
    if url.endswith("/v1"):
        url = url[:-3]
    endpoint = f"{url}/asr/transcribe"

    def _transcribe(path: Path) -> str:
        with open(path, "rb") as f:
            data = f.read()
        resp = httpx.post(
            endpoint,
            content=data,
            headers={"Content-Type": "audio/wav"},
            timeout=180.0,
        )
        resp.raise_for_status()
        body = resp.json()
        return body.get("result", "")

    return _transcribe


def _try_whisper_ov_transcriber(
    model_dir: Path | str | None,
    *,
    prefer_npu: bool = True,
) -> Optional[Transcriber]:
    """Whisper OV: encoder on NPU (or GPU fallback), decoder on CPU.

    Confirmed architecture (Intel AI Boost, 2026-06-22):
      encoder device=NPU  → 115 ms static [1,80,3000]
      decoder device=CPU  → autoregressive, dynamic shapes

    optimum-intel ≥ 1.18 supports per-sub-model device dict; encoder compiles
    with static shape on NPU while decoder runs CPU.  Falls back gracefully to
    GPU encoder if NPU is unavailable.
    """
    if not model_dir:
        return None
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        return None
    try:
        import openvino as ov
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
        from transformers import AutoProcessor

        core = ov.Core()
        available = core.available_devices

        if prefer_npu and "NPU" in available:
            enc_device = "NPU"
        elif "GPU" in available:
            enc_device = "GPU"
        else:
            enc_device = "CPU"

        device_map = {
            "encoder_model": enc_device,
            "decoder_model": "CPU",
            "decoder_with_past_model": "CPU",
        }

        processor = AutoProcessor.from_pretrained(str(model_dir))
        model = OVModelForSpeechSeq2Seq.from_pretrained(str(model_dir), device=device_map)
        logger.info("Whisper OV transcriber loaded: encoder=%s decoder=CPU", enc_device)
    except Exception as e:
        logger.info("Whisper OV transcriber init failed: %s", e)
        return None

    def _transcribe(path: Path) -> str:  # pragma: no cover - needs model + OV
        import soundfile as sf  # type: ignore
        audio, sr = sf.read(str(path), dtype="float32")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        ids = model.generate(inputs.input_features)
        texts = processor.batch_decode(ids, skip_special_tokens=True)
        return texts[0] if texts else ""

    return _transcribe


def _try_whisper_ov_subprocess_transcriber(
    model_dir: Path | str | None,
    *,
    device: str = "CPU",
) -> Optional[Transcriber]:
    """Whisper OV in a child process.

    This is slower than a warm in-process model, but robust on Windows targets
    where OpenVINO/optimum speech generation can terminate the interpreter in
    native code. A child-process crash becomes a per-sample error; the benchmark
    process and report generation continue.
    """
    if not model_dir:
        return None
    model_dir = Path(model_dir)
    if not (model_dir / "config.json").exists():
        return None
    helper = Path(__file__).resolve().parents[2] / "scripts" / "whisper_ov_transcribe.py"
    if not helper.exists():
        logger.info("Whisper OV subprocess helper missing: %s", helper)
        return None

    def _transcribe(path: Path) -> str:  # pragma: no cover - needs model + OV
        proc = subprocess.run(
            [
                sys.executable,
                str(helper),
                "--model-dir", str(model_dir),
                "--wav", str(path),
                "--device", device,
                "--language", "zh",
                "--task", "transcribe",
            ],
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=300,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper_ov_subprocess failed rc={proc.returncode}: {proc.stderr[-1000:]}"
            )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"whisper_ov_subprocess produced no output: {proc.stderr[-1000:]}")
        body = json.loads(lines[-1])
        if proc.stderr.strip():
            body["_stderr_tail"] = proc.stderr[-2000:]
        return body

    return _transcribe


def _infer_whisper_model_type(model_cfg: ModelConfig) -> str:
    explicit = getattr(model_cfg, "asr_model_type", None)
    if explicit:
        return explicit
    haystack = " ".join(
        str(v or "")
        for v in (
            getattr(model_cfg, "name", ""),
            getattr(model_cfg, "model_id", ""),
            getattr(model_cfg, "hf_repo", ""),
        )
    ).lower()
    if "large" in haystack and "turbo" in haystack:
        return "whisper-large-v3-turbo"
    for size in ("tiny", "base", "small", "medium"):
        if size in haystack:
            return f"whisper-{size}"
    return "whisper-base"


def _try_whisper_amd_npu_subprocess_transcriber(model_cfg: ModelConfig) -> Optional[Transcriber]:
    """AMD RyzenAI Whisper ONNX NPU helper in a child process.

    This backend is intentionally strict. If a model declares
    ``asr_backend: whisper_amd_npu_subprocess`` it must not silently fall back
    to sherpa/SenseVoice CPU, because that would produce a false NPU result.
    """
    helper = Path(__file__).resolve().parents[2] / "scripts" / "whisper_amd_npu_transcribe.py"
    if not helper.exists():
        logger.info("AMD Whisper NPU subprocess helper missing: %s", helper)
        return None
    device = str(getattr(model_cfg, "asr_device", "") or "npu").lower()
    if device == "auto":
        device = "npu"
    config_file = getattr(model_cfg, "asr_config_file", None)
    if device == "npu" and not (config_file and Path(config_file).exists()):
        logger.info("AMD Whisper NPU config missing: %s", config_file)
        return None

    py = getattr(model_cfg, "asr_python", None) or sys.executable
    model_dir = getattr(model_cfg, "asr_model_dir", None)
    model_type = _infer_whisper_model_type(model_cfg)

    def _transcribe(path: Path) -> str:  # pragma: no cover - needs RyzenAI runtime
        cmd = [
            py,
            str(helper),
            "--wav", str(path),
            "--model-type", model_type,
            "--device", device,
            "--language", "zh",
        ]
        if model_dir:
            cmd.extend(["--model-dir", str(model_dir)])
        if config_file:
            cmd.extend(["--config-file", str(config_file)])
        proc = subprocess.run(
            cmd,
            text=True,
            encoding="utf-8",
            errors="replace",
            capture_output=True,
            timeout=600,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"whisper_amd_npu_subprocess failed rc={proc.returncode}: {proc.stderr[-1000:]}"
            )
        lines = [line for line in proc.stdout.splitlines() if line.strip()]
        if not lines:
            raise RuntimeError(f"whisper_amd_npu_subprocess produced no output: {proc.stderr[-1000:]}")
        body = json.loads(lines[-1])
        if proc.stderr.strip():
            body["_stderr_tail"] = proc.stderr[-2000:]
        return body

    return _transcribe


def _try_sherpa_transcriber(model_dir: Path | str | None) -> Optional[Transcriber]:
    """Build a sherpa-onnx SenseVoice transcriber if the dep + model exist."""
    if not model_dir:
        return None
    model_dir = Path(model_dir)
    try:
        import sherpa_onnx  # type: ignore
    except Exception:
        return None
    model = model_dir / "model.onnx"
    tokens = model_dir / "tokens.txt"
    if not (model.exists() and tokens.exists()):
        return None
    try:
        recognizer = sherpa_onnx.OfflineRecognizer.from_sense_voice(  # type: ignore
            model=str(model), tokens=str(tokens),
        )
    except Exception as e:  # pragma: no cover - backend-specific
        logger.info("sherpa-onnx init failed: %s", e)
        return None

    def _transcribe(path: Path) -> str:  # pragma: no cover - needs model
        import soundfile as sf  # type: ignore

        audio, sr = sf.read(str(path), dtype="float32")
        stream = recognizer.create_stream()
        stream.accept_waveform(sr, audio)
        recognizer.decode_stream(stream)
        return stream.result.text

    return _transcribe


def run_asr(
    model_cfg: ModelConfig,
    *,
    manifest_path: Path | str | None = None,
    audio_root: Path | str | None = None,
    asr_model_dir: Path | str | None = None,
    num_samples: Optional[int] = None,
    thresholds: Optional[dict] = None,
    transcribe_fn: Optional[Transcriber] = None,
) -> dict:
    """Transcribe a manifest + score CER/WER/RTF. Degrades gracefully (BLOCKED)."""
    thresholds = thresholds or _DEFAULT_THRESHOLDS
    samples: list[AudioSample] = (
        load_asr_manifest(manifest_path, audio_root=audio_root, num_samples=num_samples)
        if manifest_path else []
    )
    if not samples:
        return {"benchmark": "asr", "model": model_cfg.name, "status": "blocked",
                "reason": "no dataset (manifest missing or empty)",
                "verdict": "SKIP"}

    backend = getattr(model_cfg, "asr_backend", "") or "auto"
    # HTTP backend: if base_url is configured, use remote HTTP transcription
    http_base = getattr(model_cfg, "base_url", None)
    # Whisper OV backend: asr_backend=whisper_ov/subprocess + ov_model_dir in model config
    whisper_ov_dir = (
        getattr(model_cfg, "ov_model_dir", None)
        if backend in {"whisper_ov", "whisper_ov_subprocess"}
        else None
    )
    if transcribe_fn is not None:
        transcriber = transcribe_fn
    elif backend in {"whisper_amd_npu", "whisper_amd_npu_subprocess"}:
        transcriber = _try_whisper_amd_npu_subprocess_transcriber(model_cfg)
        if transcriber is None:
            return {
                "benchmark": "asr",
                "model": model_cfg.name,
                "status": "blocked",
                "reason": "AMD Whisper NPU backend unavailable (helper/config/model files or RyzenAI VitisAIExecutionProvider not ready)",
                "num_samples": len(samples),
                "verdict": "SKIP",
            }
    elif backend in {"whisper_ov", "whisper_ov_subprocess"}:
        device = str(getattr(model_cfg, "asr_device", "") or "CPU")
        transcriber = (
            _try_whisper_ov_subprocess_transcriber(whisper_ov_dir, device=device)
            if backend == "whisper_ov_subprocess"
            else _try_whisper_ov_transcriber(whisper_ov_dir)
        )
        if transcriber is None:
            return {
                "benchmark": "asr",
                "model": model_cfg.name,
                "status": "blocked",
                "reason": "Whisper OpenVINO backend unavailable (ov_model_dir/helper/runtime not ready)",
                "num_samples": len(samples),
                "verdict": "SKIP",
            }
    elif backend == "sherpa":
        transcriber = _try_sherpa_transcriber(asr_model_dir)
        if transcriber is None:
            return {
                "benchmark": "asr",
                "model": model_cfg.name,
                "status": "blocked",
                "reason": "sherpa-onnx ASR backend unavailable",
                "num_samples": len(samples),
                "verdict": "SKIP",
            }
    else:
        transcriber = (
            (
                _http_transcriber(http_base)
                if (http_base and (
                    getattr(model_cfg, "task_type", None) == "asr"
                    or "asr" in (getattr(model_cfg, "capabilities", ()) or ())
                ))
                else None
            )
            or _try_whisper_ov_transcriber(whisper_ov_dir)
            or _try_sherpa_transcriber(asr_model_dir)
        )
    if transcriber is None:
        return {"benchmark": "asr", "model": model_cfg.name, "status": "blocked",
                "reason": "no asr backend (set base_url_env for HTTP, asr_backend: whisper_ov + ov_model_dir, asr_backend: whisper_amd_npu_subprocess + asr_config_file, or provide sherpa-onnx model)",
                "num_samples": len(samples), "verdict": "SKIP"}

    refs: list[str] = []
    hyps: list[str] = []
    latencies: list[float] = []
    rtfs: list[float] = []
    total_audio_s = 0.0
    empty_outputs = errors = 0
    per_sample: list[dict] = []

    for s in samples:
        dur = s.duration_s or wav_duration_s(s.audio)
        total_audio_s += dur
        t0 = time.monotonic()
        try:
            raw = transcriber(s.audio)
            sample_extra: dict[str, Any] = {}
            if isinstance(raw, dict):
                hyp = str(raw.get("result", "") or "")
                if isinstance(raw.get("_runtime"), dict):
                    sample_extra["runtime"] = raw["_runtime"]
                if isinstance(raw.get("_perf"), dict):
                    sample_extra["backend_perf"] = raw["_perf"]
                if raw.get("_stderr_tail"):
                    sample_extra["backend_stderr_tail"] = str(raw["_stderr_tail"])
            else:
                hyp = raw
        except Exception as e:
            errors += 1
            logger.info("  [asr] %s transcribe failed: %s", s.uid, e)
            hyp = ""
            sample_extra = {}
        proc_s = time.monotonic() - t0
        latencies.append(proc_s * 1000)
        if dur > 0:
            rtfs.append(rtf(proc_s, dur))
        if not validate_transcript(hyp)["ok"]:
            empty_outputs += 1
        refs.append(s.text)
        hyps.append(hyp)
        per_sample.append({
            "uid": s.uid,
            "audio": str(s.audio),
            "reference": s.text,
            "hypothesis": hyp,
            "duration_s": dur,
            "latency_ms": proc_s * 1000,
            "rtf": rtf(proc_s, dur) if dur > 0 else 0.0,
        } | sample_extra)

    overall_cer = corpus_cer(refs, hyps)
    overall_wer = corpus_wer(refs, hyps)
    mean_rtf = sum(rtfs) / len(rtfs) if rtfs else 0.0

    aggregate = {
        "num_samples": len(samples),
        "total_audio_s": total_audio_s,
        "cer": overall_cer,
        "wer": overall_wer,
        "rtf_mean": mean_rtf,
        "latency_ms_stats": summarize_latencies(latencies),
        "empty_output_count": empty_outputs,
        "error_count": errors,
        "data_source": samples[0].source,
    }

    reasons: list[str] = []
    if empty_outputs == len(samples):
        reasons.append("FAIL: all transcripts empty (broken backend)")
    if overall_cer > thresholds.get("cer_max", 1.0):
        reasons.append(f"FAIL: CER {overall_cer*100:.2f}% > {thresholds['cer_max']*100:.0f}%")
    if mean_rtf > thresholds.get("rtf_max", 1.0):
        reasons.append(f"FAIL: RTF {mean_rtf:.3f} > {thresholds['rtf_max']} (not real-time)")
    if empty_outputs and empty_outputs < len(samples):
        reasons.append(f"WARN: {empty_outputs} empty transcript(s)")

    verdict = "FAIL" if any(r.startswith("FAIL") for r in reasons) else (
        "WARN" if reasons else "PASS"
    )

    return {
        "benchmark": "asr",
        "model": model_cfg.name,
        "status": "ok",
        "verdict": verdict,
        "verdict_reasons": reasons,
        "aggregate": aggregate,
        "per_sample": per_sample,
    }
