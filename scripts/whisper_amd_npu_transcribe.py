"""One-shot AMD RyzenAI Whisper ONNX transcription helper.

This mirrors the AMD RyzenAI-SW Whisper demo execution model but returns a
single JSON line for the benchmark runner. It uses VitisAIExecutionProvider
when --device npu is selected and keeps the main benchmark process isolated
from native ONNX Runtime/RyzenAI failures.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000

MODEL_REPOS = {
    "whisper-tiny": ("amd/whisper-tiny-onnx-npu", ("tiny_encoder.onnx", "encoder_model.onnx"), ("tiny_decoder.onnx", "decoder_model.onnx")),
    "whisper-base": ("amd/whisper-base-onnx-npu", ("base_encoder.onnx", "encoder_model.onnx"), ("base_decoder.onnx", "decoder_model.onnx")),
    "whisper-small": ("amd/whisper-small-onnx-npu", ("encoder_model.onnx", "small_encoder.onnx"), ("decoder_model.onnx", "small_decoder.onnx")),
    "whisper-medium": ("amd/whisper-medium-onnx-npu", ("encoder_model.onnx", "medium_encoder.onnx"), ("decoder_model.onnx", "medium_decoder.onnx")),
    "whisper-large-v3-turbo": ("amd/whisper-large-turbo-onnx-npu", ("encoder_model.onnx", "large_encoder.onnx"), ("decoder_model.onnx", "large_decoder.onnx")),
}


def _load_wav(path: Path) -> np.ndarray:
    import soundfile as sf  # type: ignore

    audio, sr = sf.read(str(path), dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sr == SAMPLE_RATE:
        return np.asarray(audio, dtype=np.float32)
    try:
        import librosa  # type: ignore

        return np.asarray(librosa.resample(audio, orig_sr=sr, target_sr=SAMPLE_RATE), dtype=np.float32)
    except Exception:
        duration = len(audio) / float(sr)
        src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
        dst_len = max(1, int(duration * SAMPLE_RATE))
        dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype("float32")


def _find_model_files(model_type: str, model_dir: Path | None) -> tuple[Path, Path]:
    if model_type not in MODEL_REPOS:
        raise ValueError(f"unsupported model_type: {model_type}")
    repo_id, encoder_names, decoder_names = MODEL_REPOS[model_type]
    local_dir = model_dir
    if local_dir is None:
        from huggingface_hub import snapshot_download  # type: ignore

        local_dir = Path(snapshot_download(repo_id=repo_id))
    if not local_dir.exists():
        raise FileNotFoundError(f"model_dir not found: {local_dir}")

    def pick(candidates: tuple[str, ...], pattern: str) -> Path:
        for name in candidates:
            path = local_dir / name
            if path.exists():
                return path
        matches = sorted(local_dir.glob(pattern))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"no {pattern} model under {local_dir}")

    return pick(encoder_names, "*encoder*.onnx"), pick(decoder_names, "*decoder*.onnx")


def _resolve_demo_path(value: str, demo_root: Path) -> str:
    path = Path(value)
    if path.is_absolute():
        return str(path)
    text = value.replace("\\", "/")
    if text.startswith("./"):
        text = text[2:]
    return str(demo_root / text)


def _synthesized_npu_config(model_key: str) -> dict:
    return {
        "encoder": {
            "cache_key": f"whisper_{model_key}_encoder",
            "cache_dir": "./cache/",
            "config_file": "./config/vitisai_config_whisper_encoder.json",
        },
        "decoder": {
            "cache_key": f"whisper_{model_key}_decoder",
            "cache_dir": "./cache/",
            "config_file": "./config/vitisai_config_whisper_decoder.json",
        },
    }


def _provider_options(config_file: Path, model_type: str, device: str) -> tuple[list, list]:
    if device == "cpu":
        return ["CPUExecutionProvider"], ["CPUExecutionProvider"]
    if device != "npu":
        raise ValueError(f"unsupported device: {device}")
    with config_file.open(encoding="utf-8") as f:
        config = json.load(f)
    model_key = model_type.replace("whisper-", "")
    demo_root = config_file.parent.parent
    model_node = ((config.get("whisper") or {}).get(model_key) or {})
    npu_node = model_node.get("npu") or _synthesized_npu_config(model_key)

    def build(opts: dict) -> list:
        provider_config = opts.get("config_file")
        if not provider_config:
            return ["CPUExecutionProvider"]
        cfg_path = _resolve_demo_path(str(provider_config), demo_root)
        if not Path(cfg_path).exists():
            raise FileNotFoundError(f"VitisAI provider config missing: {cfg_path}")
        cache_dir = _resolve_demo_path(str(opts.get("cache_dir", "./cache/")), demo_root)
        os.makedirs(cache_dir, exist_ok=True)
        return [
            (
                "VitisAIExecutionProvider",
                {
                    "config_file": cfg_path,
                    "cache_dir": cache_dir,
                    "cache_key": str(opts.get("cache_key", "")),
                },
            ),
            "CPUExecutionProvider",
        ]

    return build(npu_node["encoder"]), build(npu_node["decoder"])


class WhisperONNX:
    def __init__(
        self,
        encoder_path: Path,
        decoder_path: Path,
        model_type: str,
        encoder_providers: list,
        decoder_providers: list,
        language: str | None,
    ) -> None:
        import onnxruntime as ort  # type: ignore
        from transformers import WhisperFeatureExtractor, WhisperTokenizer  # type: ignore

        self.encoder = ort.InferenceSession(str(encoder_path), providers=encoder_providers)
        self.decoder = ort.InferenceSession(str(decoder_path), providers=decoder_providers)
        self.feature_extractor = WhisperFeatureExtractor.from_pretrained(f"openai/{model_type}")
        self.tokenizer = WhisperTokenizer.from_pretrained(f"openai/{model_type}")
        self.eos_token = self.tokenizer.eos_token_id
        self.max_length = min(448, self.decoder.get_inputs()[0].shape[1])
        if not isinstance(self.max_length, int):
            raise ValueError("decoder input shape must be static")
        if language:
            self.tokenizer.set_prefix_tokens(language=language, task="transcribe")
            self.initial_tokens = list(self.tokenizer.prefix_tokens)
        else:
            self.initial_tokens = [self.tokenizer.convert_tokens_to_ids("<|startoftranscript|>")]

    def transcribe(self, audio: np.ndarray, chunk_length_s: int = 30) -> str:
        chunks: list[str] = []
        stride = SAMPLE_RATE * chunk_length_s
        overlap = SAMPLE_RATE
        for start in range(0, len(audio), max(1, stride - overlap)):
            chunk = audio[start:min(start + stride, len(audio))]
            inputs = self.feature_extractor(chunk, sampling_rate=SAMPLE_RATE, return_tensors="np")
            input_name = self.encoder.get_inputs()[0].name
            encoder_out = self.encoder.run(None, {input_name: inputs["input_features"]})[0]
            chunks.append(self._decode_chunk(encoder_out))
            if start + stride >= len(audio):
                break
        return " ".join(p for p in chunks if p).strip()

    def _decode_chunk(self, encoder_out: np.ndarray) -> str:
        tokens = list(self.initial_tokens)
        decoder_inputs = self.decoder.get_inputs()
        input_ids_name = decoder_inputs[0].name
        encoder_out_name = decoder_inputs[1].name
        if decoder_inputs[0].type != "tensor(int64)":
            input_ids_name, encoder_out_name = encoder_out_name, input_ids_name

        for _ in range(len(tokens), self.max_length):
            decoder_input = np.full((1, self.max_length), self.eos_token, dtype=np.int64)
            decoder_input[0, :len(tokens)] = tokens
            outputs = self.decoder.run(None, {
                input_ids_name: decoder_input,
                encoder_out_name: encoder_out,
            })
            next_token = int(np.argmax(outputs[0][0, len(tokens) - 1]))
            if next_token == self.eos_token:
                break
            tokens.append(next_token)
        return self.tokenizer.decode(
            tokens[len(self.initial_tokens):],
            skip_special_tokens=True,
        ).strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model-dir")
    parser.add_argument("--model-type", required=True, choices=sorted(MODEL_REPOS))
    parser.add_argument("--config-file")
    parser.add_argument("--device", choices=["cpu", "npu"], default="npu")
    parser.add_argument("--language", default="zh")
    args = parser.parse_args()

    t0 = time.monotonic()
    model_dir = Path(args.model_dir) if args.model_dir else None
    encoder_path, decoder_path = _find_model_files(args.model_type, model_dir)
    config_file = Path(args.config_file) if args.config_file else None
    if args.device == "npu":
        if not config_file:
            raise FileNotFoundError("--config-file is required for --device npu")
        encoder_providers, decoder_providers = _provider_options(config_file, args.model_type, args.device)
    else:
        encoder_providers, decoder_providers = ["CPUExecutionProvider"], ["CPUExecutionProvider"]

    model = WhisperONNX(
        encoder_path=encoder_path,
        decoder_path=decoder_path,
        model_type=args.model_type,
        encoder_providers=encoder_providers,
        decoder_providers=decoder_providers,
        language=args.language,
    )
    text = model.transcribe(_load_wav(Path(args.wav)))
    print(json.dumps({
        "result": text,
        "_perf": {"latency_ms": round((time.monotonic() - t0) * 1000, 1)},
        "_runtime": {
            "device": args.device,
            "encoder_providers": model.encoder.get_providers(),
            "decoder_providers": model.decoder.get_providers(),
        },
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
