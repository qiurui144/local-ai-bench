"""One-shot Whisper OpenVINO transcription helper.

This script is intentionally process-isolated from ``serve_ov_extras.py``.
Some OpenVINO/optimum Whisper combinations can terminate the interpreter in
native code during generation. Running the ASR request in a child process keeps
the HTTP service alive and turns native failures into ordinary request errors.
"""

from __future__ import annotations

import argparse
import json
import time


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--wav", required=True)
    parser.add_argument("--device", default="CPU")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--task", default="transcribe")
    args = parser.parse_args()

    import numpy as np  # type: ignore
    import soundfile as sf  # type: ignore
    from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
    from transformers import AutoProcessor  # type: ignore

    t0 = time.monotonic()
    processor = AutoProcessor.from_pretrained(args.model_dir)
    model = OVModelForSpeechSeq2Seq.from_pretrained(args.model_dir, device=args.device)

    audio, sr = sf.read(args.wav, dtype="float32")
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        try:
            import librosa  # type: ignore
            audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
        except Exception:
            duration = len(audio) / float(sr)
            src_x = np.linspace(0.0, duration, num=len(audio), endpoint=False)
            dst_len = max(1, int(duration * 16000))
            dst_x = np.linspace(0.0, duration, num=dst_len, endpoint=False)
            audio = np.interp(dst_x, src_x, audio).astype("float32")

    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    generate_kwargs = {}
    if hasattr(processor, "get_decoder_prompt_ids"):
        try:
            generate_kwargs["forced_decoder_ids"] = processor.get_decoder_prompt_ids(
                language=args.language,
                task=args.task,
            )
        except Exception:
            pass
    ids = model.generate(inputs.input_features, **generate_kwargs)
    texts = processor.batch_decode(ids, skip_special_tokens=True)
    text = texts[0].strip() if texts else ""
    print(json.dumps({
        "result": text,
        "_perf": {"latency_ms": round((time.monotonic() - t0) * 1000, 1)},
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
