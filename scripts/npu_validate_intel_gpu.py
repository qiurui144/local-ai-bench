"""
Intel iGPU (Arc) OpenVINO validation — embedding / reranker / ASR on GPU device.

Context: NPU requires static-shape models; OV Hub INT8 models use dynamic shapes.
This script tests the same models on GPU (iGPU Arc) which supports dynamic shapes.
Run: python npu_validate_intel_gpu.py
"""

import json
import time
import sys
import io
from pathlib import Path

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_PATH = Path(r"C:\npu_gpu_validation_results.json")
OV_MODELS = Path(r"C:\ov_models")
DEVICE = "GPU"


def timer(fn):
    t0 = time.perf_counter()
    result = fn()
    return result, (time.perf_counter() - t0) * 1000


def validate_embedding_gpu():
    print("\n" + "="*60, flush=True)
    print(f"TASK: Embedding on {DEVICE} (BGE-base-en-v1.5 INT8)", flush=True)
    MODEL_DIR = str(OV_MODELS / "embedding" / "bge-base-en-v1.5-int8-ov")
    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}"}
    try:
        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForFeatureExtraction.from_pretrained(
            MODEL_DIR, device=DEVICE))
        print(f"  Loaded in {load_ms:.0f}ms", flush=True)
        sentences = [
            "What is the capital of France?",
            "OpenVINO accelerates deep learning inference on Intel hardware.",
            "Embedding models convert text to dense vectors.",
        ]
        latencies = []
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True, max_length=128)
            _, lat = timer(lambda: model(**inputs))
            latencies.append(lat)
            print(f"  '{sent[:50]}' -> {lat:.1f}ms", flush=True)
        avg = sum(latencies) / len(latencies)
        print(f"  [OK] avg={avg:.1f}ms", flush=True)
        return {"status": "PASS", "device": DEVICE, "model": "bge-base-en-v1.5-int8-ov",
                "load_ms": round(load_ms), "avg_latency_ms": round(avg, 1), "samples": len(sentences)}
    except Exception as e:
        print(f"  [FAIL] {e}", flush=True)
        return {"status": "FAIL", "error": str(e)}


def validate_reranker_gpu():
    print("\n" + "="*60, flush=True)
    print(f"TASK: Reranker on {DEVICE} (BGE-reranker-base INT8)", flush=True)
    MODEL_DIR = str(OV_MODELS / "reranker" / "bge-reranker-base-int8-ov")
    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}"}
    try:
        from optimum.intel import OVModelForSequenceClassification
        from transformers import AutoTokenizer
        import torch
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForSequenceClassification.from_pretrained(
            MODEL_DIR, device=DEVICE))
        print(f"  Loaded in {load_ms:.0f}ms", flush=True)
        pairs = [
            ("What is AI?", "Artificial intelligence simulates human intelligence."),
            ("Capital of France", "Paris is the capital of France."),
            ("Capital of France", "Tokyo is the capital of Japan."),
        ]
        latencies, scores = [], []
        for q, d in pairs:
            inputs = tokenizer(q, d, return_tensors="pt", padding=True, truncation=True, max_length=512)
            out, lat = timer(lambda: model(**inputs))
            score = torch.sigmoid(out.logits[0][0]).item()
            latencies.append(lat)
            scores.append(score)
            print(f"  '{q[:30]}' | '{d[:40]}' -> score={score:.3f} {lat:.1f}ms", flush=True)
        avg = sum(latencies) / len(latencies)
        print(f"  [OK] avg={avg:.1f}ms, scores={[round(s,3) for s in scores]}", flush=True)
        return {"status": "PASS", "device": DEVICE, "model": "bge-reranker-base-int8-ov",
                "load_ms": round(load_ms), "avg_latency_ms": round(avg, 1), "scores": scores}
    except Exception as e:
        print(f"  [FAIL] {e}", flush=True)
        return {"status": "FAIL", "error": str(e)}


def validate_asr_gpu():
    print("\n" + "="*60, flush=True)
    print(f"TASK: ASR on {DEVICE} (Whisper-base INT8)", flush=True)
    MODEL_DIR = str(OV_MODELS / "asr" / "whisper-base-int8-ov")
    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}"}
    # Create test WAV
    TEST_AUDIO = Path(r"C:\npu_asr_test.wav")
    if not TEST_AUDIO.exists():
        import wave
        import struct
        with wave.open(str(TEST_AUDIO), 'w') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(16000)
            wf.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
    try:
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_DIR, device=DEVICE))
        print(f"  Loaded in {load_ms:.0f}ms", flush=True)
        import soundfile as sf
        audio, sr = sf.read(str(TEST_AUDIO), dtype='float32')
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
        _, lat = timer(lambda: model.generate(inputs.input_features))
        print(f"  Inference: {lat:.1f}ms  [OK]", flush=True)
        return {"status": "PASS", "device": DEVICE, "model": "whisper-base-int8-ov",
                "load_ms": round(load_ms), "inference_ms": round(lat, 1)}
    except ImportError:
        # Try with librosa
        try:
            from optimum.intel.openvino import OVModelForSpeechSeq2Seq
            from transformers import AutoProcessor
            import librosa
            processor = AutoProcessor.from_pretrained(MODEL_DIR)
            model, load_ms = timer(lambda: OVModelForSpeechSeq2Seq.from_pretrained(
                MODEL_DIR, device=DEVICE))
            audio, sr = librosa.load(str(TEST_AUDIO), sr=16000)
            inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
            _, lat = timer(lambda: model.generate(inputs.input_features))
            return {"status": "PASS", "device": DEVICE, "model": "whisper-base-int8-ov",
                    "load_ms": round(load_ms), "inference_ms": round(lat, 1)}
        except Exception as e2:
            print(f"  [FAIL] {e2}", flush=True)
            return {"status": "FAIL", "error": str(e2)}
    except Exception as e:
        print(f"  [FAIL] {e}", flush=True)
        return {"status": "FAIL", "error": str(e)}


def probe():
    print("\n" + "="*60, flush=True)
    print("PROBE: OpenVINO Devices", flush=True)
    import openvino as ov
    core = ov.Core()
    devices = core.available_devices
    info = {}
    for d in devices:
        try:
            name = core.get_property(d, "FULL_DEVICE_NAME")
            info[d] = name
            print(f"  {d}: {name}", flush=True)
        except Exception:
            info[d] = "unknown"
    return {"devices": devices, "device_names": info}


def main():
    results = {"platform": "Intel Windows", "test_device": DEVICE,
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
               "note": "NPU requires static shapes; GPU tested as production alternative"}
    results["probe"] = probe()
    results["embedding"] = validate_embedding_gpu()
    results["reranker"] = validate_reranker_gpu()
    results["asr"] = validate_asr_gpu()

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved -> {RESULTS_PATH}", flush=True)

    print("\n" + "="*60, flush=True)
    print("SUMMARY:", flush=True)
    for k, v in results.items():
        if isinstance(v, dict) and "status" in v:
            icon = "[OK]" if "PASS" in v["status"] else "[FAIL]" if "FAIL" in v["status"] else "[SKIP]"
            print(f"  {icon} {k}: {v['status']} (device={v.get('device','?')})", flush=True)


if __name__ == "__main__":
    main()
