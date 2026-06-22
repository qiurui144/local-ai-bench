"""
Intel Core Ultra NPU 全面验证脚本 (OpenVINO device='NPU')

测试任务: Embedding / Reranker / OCR / ASR / LLM
目标机: Intel Windows 192.168.100.116
运行方式:
  python npu_validate_intel.py --task all
  python npu_validate_intel.py --task embedding,reranker
  python npu_validate_intel.py --task asr

依赖 (Intel 机器需提前安装):
  pip install openvino openvino-genai optimum[openvino] transformers
  pip install paddleocr paddlepaddle  (OCR 任务)
  pip install openai-whisper           (ASR 转换用)

模型存放: C:\ov_models\embedding\  C:\ov_models\reranker\  C:\ov_models\asr\  C:\ov_models\ocr\
"""

import argparse
import json
import time
import sys
import io
from pathlib import Path

# Fix GBK console encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_PATH = Path(r"C:\npu_validation_results.json")

# ─── 工具 ────────────────────────────────────────────────────────────────────

def timer(fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def save_results(results: dict):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_PATH}", flush=True)


# ─── Embedding ───────────────────────────────────────────────────────────────

def validate_embedding():
    """BGE-base-en-v1.5 INT8 on NPU via OpenVINO."""
    print("\n" + "="*60, flush=True)
    print("TASK: Embedding (BGE-base-en-v1.5 INT8 OpenVINO NPU)", flush=True)

    MODEL_DIR = r"C:\ov_models\embedding\bge-base-en-v1.5-int8-ov"
    DEVICE = "NPU"

    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}",
                "download_cmd": "python scripts/npu_download_intel.py --task embedding"}

    try:
        from optimum.intel import OVModelForFeatureExtraction
        from transformers import AutoTokenizer

        print(f"Loading from {MODEL_DIR} on {DEVICE}...", flush=True)
        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForFeatureExtraction.from_pretrained(
            MODEL_DIR, device=DEVICE))
        print(f"Loaded in {load_ms:.0f}ms", flush=True)

        sentences = [
            "What is the capital of France?",
            "The quick brown fox jumps over the lazy dog.",
            "OpenVINO accelerates deep learning inference.",
            "人工智能正在改变世界。",
            "Embedding models convert text to dense vectors.",
        ]

        latencies = []
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="pt", padding=True, truncation=True)
            _, lat = timer(lambda: model(**inputs))
            latencies.append(lat)
            print(f"  '{sent[:40]}' → {lat:.1f}ms", flush=True)

        avg_lat = sum(latencies) / len(latencies)
        result = {
            "status": "PASS",
            "device": DEVICE,
            "model": "bge-base-en-v1.5-int8-ov",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
            "samples": len(sentences),
        }
        print(f"\n[OK] Embedding NPU: avg={avg_lat:.1f}ms", flush=True)
        return result

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


# ─── Reranker ────────────────────────────────────────────────────────────────

def validate_reranker():
    """BGE-reranker-base INT8 on NPU via OpenVINO."""
    print("\n" + "="*60, flush=True)
    print("TASK: Reranker (bge-reranker-base INT8 OpenVINO NPU)", flush=True)

    MODEL_DIR = r"C:\ov_models\reranker\bge-reranker-base-int8-ov"
    DEVICE = "NPU"

    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}",
                "download_cmd": "python scripts/npu_download_intel.py --task reranker"}

    try:
        from optimum.intel import OVModelForSequenceClassification
        from transformers import AutoTokenizer
        import torch

        tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForSequenceClassification.from_pretrained(
            MODEL_DIR, device=DEVICE))
        print(f"Loaded in {load_ms:.0f}ms", flush=True)

        pairs = [
            ("What is AI?", "Artificial intelligence is the simulation of human intelligence."),
            ("Capital of France", "Paris is the capital and most populous city of France."),
            ("Capital of France", "The Eiffel Tower is a famous landmark in Paris."),
        ]

        latencies = []
        for query, doc in pairs:
            inputs = tokenizer(query, doc, return_tensors="pt",
                               padding=True, truncation=True, max_length=512)
            out, lat = timer(lambda: model(**inputs))
            score = torch.sigmoid(out.logits[0][0]).item()
            latencies.append(lat)
            print(f"  '{query[:30]}' vs '{doc[:40]}' → score={score:.3f} {lat:.1f}ms",
                  flush=True)

        avg_lat = sum(latencies) / len(latencies)
        result = {
            "status": "PASS",
            "device": DEVICE,
            "model": "bge-reranker-base-int8-ov",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
            "samples": len(pairs),
        }
        print(f"\n[OK] Reranker NPU: avg={avg_lat:.1f}ms", flush=True)
        return result

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


# ─── OCR ─────────────────────────────────────────────────────────────────────

def validate_ocr():
    """PP-OCR v4 on Intel NPU via OpenVINO with static input shapes.

    Confirmed (2026-06-22): det 33ms, rec 11ms (H=48 required), cls 3ms.
    Model dir: C:\\ov_models\\ocr\\ppocr-v4\\  with det.xml/bin + rec.xml/bin + cls.xml/bin.
    Uses OV Core directly (not rapidocr-openvino) to force device=NPU.
    """
    print("\n" + "="*60, flush=True)
    print("TASK: OCR (PP-OCR v4 OpenVINO NPU — static shapes)", flush=True)

    OCR_DIR = Path(r"C:\ov_models\ocr\ppocr-v4")
    DET_XML = OCR_DIR / "det.xml"
    REC_XML = OCR_DIR / "rec.xml"
    CLS_XML = OCR_DIR / "cls.xml"
    DEVICE = "NPU"

    if not DET_XML.exists():
        # Try rapidocr-openvino fallback (iGPU auto-device; not NPU)
        try:
            import openvino as ov
            core = ov.Core()
            devices = core.available_devices
            npu_ok = "NPU" in devices
            from rapidocr_openvino import RapidOCR  # type: ignore
            engine = RapidOCR()
            import numpy as np
            dummy = np.zeros((64, 200, 3), dtype=np.uint8)
            _, lat = timer(lambda: engine(dummy))
            return {
                "status": "PASS_IGPU",
                "device": "iGPU (rapidocr-openvino auto-device, not NPU)",
                "latency_ms": round(lat, 1),
                "available_devices": list(devices),
                "npu_available": npu_ok,
                "npu_note": f"NPU static-shape path needs OV IR in {OCR_DIR}. "
                            "Convert: mo --input_model ppocr_v4_det.onnx --input [1,3,640,640]",
            }
        except Exception as e:
            return {
                "status": "SKIP",
                "reason": f"OV IR not found at {OCR_DIR}; rapidocr fallback also failed: {e}",
                "prep_cmd": "python scripts/npu_download_intel.py --task ocr",
            }

    try:
        import openvino as ov
        import numpy as np

        core = ov.Core()
        print(f"  Available devices: {core.available_devices}", flush=True)

        results = {}

        # Detection: static [1,3,640,640]
        print("  Loading det model (static [1,3,640,640])…", flush=True)
        det_model, det_load_ms = timer(lambda: core.compile_model(str(DET_XML), DEVICE))
        dummy_det = np.zeros((1, 3, 640, 640), dtype=np.float32)
        _, det_lat = timer(lambda: det_model([dummy_det]))
        print(f"    det: load {det_load_ms:.0f}ms  infer {det_lat:.1f}ms", flush=True)
        results["det"] = {"load_ms": round(det_load_ms, 0), "latency_ms": round(det_lat, 1)}

        # Recognition: static [1,3,48,320]  (H=48 required for AvgPool NPU constraint)
        print("  Loading rec model (static [1,3,48,320])…", flush=True)
        rec_model, rec_load_ms = timer(lambda: core.compile_model(str(REC_XML), DEVICE))
        dummy_rec = np.zeros((1, 3, 48, 320), dtype=np.float32)
        _, rec_lat = timer(lambda: rec_model([dummy_rec]))
        print(f"    rec: load {rec_load_ms:.0f}ms  infer {rec_lat:.1f}ms", flush=True)
        results["rec"] = {"load_ms": round(rec_load_ms, 0), "latency_ms": round(rec_lat, 1)}

        # Classifier: static [1,3,48,192]
        if CLS_XML.exists():
            print("  Loading cls model (static [1,3,48,192])…", flush=True)
            cls_model, cls_load_ms = timer(lambda: core.compile_model(str(CLS_XML), DEVICE))
            dummy_cls = np.zeros((1, 3, 48, 192), dtype=np.float32)
            _, cls_lat = timer(lambda: cls_model([dummy_cls]))
            print(f"    cls: load {cls_load_ms:.0f}ms  infer {cls_lat:.1f}ms", flush=True)
            results["cls"] = {"load_ms": round(cls_load_ms, 0), "latency_ms": round(cls_lat, 1)}

        print(f"\n[OK] OCR NPU: det {results['det']['latency_ms']}ms  "
              f"rec {results['rec']['latency_ms']}ms  "
              f"pipeline ~{sum(v['latency_ms'] for v in results.values()):.0f}ms", flush=True)
        return {"status": "PASS", "device": DEVICE, **results}

    except Exception as e:
        return {"status": "FAIL", "error": str(e),
                "note": "Check VPUX compiler; ensure model converted with correct static shapes"}


# ─── ASR ─────────────────────────────────────────────────────────────────────

def validate_asr():
    """Whisper-base INT8 on Intel NPU/GPU: encoder=NPU, decoder=CPU (split device).

    Confirmed (2026-06-22): encoder 115ms on Intel AI Boost NPU.
    Uses optimum-intel device dict (encoder_model=NPU, decoder*=CPU).
    Falls back to GPU encoder if NPU unavailable.
    """
    print("\n" + "="*60, flush=True)
    print("TASK: ASR (Whisper-base INT8 — encoder NPU, decoder CPU)", flush=True)

    MODEL_DIR = r"C:\ov_models\asr\whisper-base-int8-ov"
    TEST_AUDIO = r"C:\npu_asr_test.wav"

    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}",
                "download_cmd": "python scripts/npu_download_intel.py --task asr"}

    if not Path(TEST_AUDIO).exists():
        try:
            import wave
            import struct
            with wave.open(TEST_AUDIO, 'w') as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(16000)
                wf.writeframes(struct.pack('<' + 'h' * 16000, *([0] * 16000)))
        except Exception as e:
            return {"status": "SKIP", "reason": f"Cannot create test audio: {e}"}

    try:
        import openvino as ov
        from optimum.intel.openvino import OVModelForSpeechSeq2Seq  # type: ignore
        from transformers import AutoProcessor  # type: ignore
        import soundfile as sf  # type: ignore

        core = ov.Core()
        available = list(core.available_devices)
        print(f"  Available devices: {available}", flush=True)

        enc_device = "NPU" if "NPU" in available else ("GPU" if "GPU" in available else "CPU")
        device_map = {
            "encoder_model": enc_device,
            "decoder_model": "CPU",
            "decoder_with_past_model": "CPU",
        }
        print(f"  Loading: encoder={enc_device}, decoder=CPU", flush=True)

        processor = AutoProcessor.from_pretrained(MODEL_DIR)
        model, load_ms = timer(lambda: OVModelForSpeechSeq2Seq.from_pretrained(
            MODEL_DIR, device=device_map))
        print(f"  Loaded in {load_ms:.0f}ms", flush=True)

        audio, sr = sf.read(TEST_AUDIO, dtype="float32")
        inputs = processor(audio, sampling_rate=16000, return_tensors="pt")

        # Warm-up (NPU first-run compiles)
        try:
            model.generate(inputs.input_features)
        except Exception:
            pass

        latencies = []
        for i in range(3):
            _, lat = timer(lambda: model.generate(inputs.input_features))
            latencies.append(lat)
            print(f"  Run {i+1}: {lat:.1f}ms", flush=True)

        avg_lat = sum(latencies) / len(latencies)
        print(f"\n[OK] ASR: encoder={enc_device} avg {avg_lat:.1f}ms (decoder=CPU)", flush=True)
        return {
            "status": "PASS",
            "encoder_device": enc_device,
            "decoder_device": "CPU",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
            "model": "whisper-base-int8-ov",
        }

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


def validate_asr_diarization():
    """Speaker diarization probe: VAD (CPU) + Whisper NPU + speaker embed NPU.

    Pipeline architecture (Intel):
      1. VAD: Silero VAD ONNX on CPU → detect speech segments
      2. ASR: Whisper-base encoder on NPU → transcribe each segment
      3. Speaker embed: ERes2Net / CAM++ ONNX on NPU (static shape) → d-vectors
      4. Clustering: agglomerative clustering on CPU → assign speaker IDs

    This function probes each stage independently and reports PASS/FAIL per stage.
    Full diarization pipeline requires: silero_vad.onnx + whisper-base-int8-ov +
    speaker_embed.onnx (static, e.g. [1, 16000] fixed window).
    """
    print("\n" + "="*60, flush=True)
    print("TASK: ASR Diarization (VAD → Whisper NPU → speaker embed)", flush=True)

    SILERO_VAD = Path(r"C:\ov_models\asr\silero_vad.onnx")
    WHISPER_DIR = Path(r"C:\ov_models\asr\whisper-base-int8-ov")
    SPEAKER_EMBED = Path(r"C:\ov_models\asr\speaker_embed.onnx")

    results: dict[str, dict] = {}

    # Stage 1: VAD probe
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np

        if not SILERO_VAD.exists():
            results["vad"] = {"status": "SKIP", "reason": f"Not found: {SILERO_VAD}",
                              "install": "pip install silero-vad; copy onnx to above path"}
        else:
            vad_sess, load_ms = timer(lambda: ort.InferenceSession(
                str(SILERO_VAD), providers=["CPUExecutionProvider"]))
            dummy = np.zeros((1, 512), dtype=np.float32)
            # Silero VAD expects (batch, samples) at 16kHz, 512 sample chunks
            h = np.zeros((2, 1, 64), dtype=np.float32)
            c = np.zeros((2, 1, 64), dtype=np.float32)
            _, lat = timer(lambda: vad_sess.run(None, {"input": dummy, "h": h, "c": c,
                                                        "sr": np.array(16000)}))
            results["vad"] = {"status": "PASS", "device": "CPU",
                              "load_ms": round(load_ms, 0), "latency_ms": round(lat, 1)}
            print(f"  VAD: CPU {lat:.1f}ms [OK]", flush=True)
    except Exception as e:
        results["vad"] = {"status": "FAIL", "error": str(e)}

    # Stage 2: Whisper NPU (reuse validate_asr result)
    if WHISPER_DIR.exists():
        print("  Whisper NPU: calling validate_asr()…", flush=True)
        results["whisper_npu"] = validate_asr()
    else:
        results["whisper_npu"] = {"status": "SKIP",
                                   "reason": f"Not found: {WHISPER_DIR}"}

    # Stage 3: Speaker embedding on NPU
    try:
        import onnxruntime as ort  # type: ignore
        import numpy as np
        import openvino as ov  # type: ignore

        if not SPEAKER_EMBED.exists():
            results["speaker_embed"] = {
                "status": "SKIP",
                "reason": f"Not found: {SPEAKER_EMBED}",
                "note": "Download ERes2Net or CAM++ speaker model, export to ONNX with "
                        "static shape [1, 16000], then run mo to convert to OV IR for NPU",
                "example": "pip install wespeaker; wespeaker --export onnx --static-shape"
            }
        else:
            # Try NPU via OV Core (static shape speaker model)
            core = ov.Core()
            if "NPU" in core.available_devices:
                spk_model, load_ms = timer(
                    lambda: core.compile_model(str(SPEAKER_EMBED), "NPU"))
                dummy_audio = np.zeros((1, 16000), dtype=np.float32)
                _, lat = timer(lambda: spk_model([dummy_audio]))
                results["speaker_embed"] = {
                    "status": "PASS", "device": "NPU",
                    "load_ms": round(load_ms, 0), "latency_ms": round(lat, 1)}
                print(f"  Speaker embed: NPU {lat:.1f}ms [OK]", flush=True)
            else:
                # Fallback: CPU via onnxruntime
                sess, load_ms = timer(lambda: ort.InferenceSession(
                    str(SPEAKER_EMBED), providers=["CPUExecutionProvider"]))
                dummy_audio = np.zeros((1, 16000), dtype=np.float32)
                inp_name = sess.get_inputs()[0].name
                _, lat = timer(lambda: sess.run(None, {inp_name: dummy_audio}))
                results["speaker_embed"] = {
                    "status": "PASS_CPU", "device": "CPU (NPU not available)",
                    "load_ms": round(load_ms, 0), "latency_ms": round(lat, 1)}
    except Exception as e:
        results["speaker_embed"] = {"status": "FAIL", "error": str(e)}

    # Stage 4: Clustering (always CPU, numpy-only)
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        dummy_embeds = np.random.randn(10, 192)  # 10 segments, 192-dim d-vectors
        _, lat = timer(lambda: AgglomerativeClustering(n_clusters=2).fit(dummy_embeds))
        results["clustering"] = {"status": "PASS", "device": "CPU", "latency_ms": round(lat, 1)}
        print(f"  Clustering: CPU {lat:.1f}ms [OK]", flush=True)
    except ImportError:
        results["clustering"] = {"status": "SKIP",
                                  "reason": "pip install scikit-learn",
                                  "note": "Clustering runs on CPU; no GPU needed"}
    except Exception as e:
        results["clustering"] = {"status": "FAIL", "error": str(e)}

    overall = ("PASS" if all(v.get("status", "").startswith("PASS")
                              for v in results.values()) else "PARTIAL")
    print(f"\n[Diarization] Overall: {overall}", flush=True)
    return {"status": overall, "stages": results}


# ─── LLM NPU ─────────────────────────────────────────────────────────────────

def validate_llm_npu():
    """Qwen2.5-1.5B-INT4 on Intel NPU via openvino_genai."""
    print("\n" + "="*60, flush=True)
    print("TASK: LLM NPU (Qwen2.5-1.5B-INT4 openvino_genai device=NPU)", flush=True)

    MODEL_DIR = r"C:\ov_models\qwen2.5-1.5b-int4-ov"
    DEVICE = "NPU"

    if not Path(MODEL_DIR).exists():
        return {"status": "SKIP", "reason": f"Model not found: {MODEL_DIR}"}

    try:
        import openvino_genai as ov_genai

        print(f"Loading on {DEVICE}...", flush=True)
        pipe, load_ms = timer(lambda: ov_genai.LLMPipeline(MODEL_DIR, DEVICE))
        print(f"Loaded in {load_ms:.0f}ms", flush=True)

        prompt = "What is the capital of France? Answer in one word."
        token_count = 0
        first_token_time = None
        gen_start = time.perf_counter()

        def streamer(word):
            nonlocal token_count, first_token_time
            if first_token_time is None:
                first_token_time = (time.perf_counter() - gen_start) * 1000
            token_count += 1
            return False

        config = ov_genai.GenerationConfig()
        config.max_new_tokens = 50
        config.do_sample = False

        pipe.generate(prompt, config, streamer)
        total_ms = (time.perf_counter() - gen_start) * 1000
        tps = token_count / (total_ms / 1000)

        result = {
            "status": "PASS",
            "device": DEVICE,
            "model": "qwen2.5-1.5b-int4-ov",
            "load_ms": round(load_ms, 0),
            "ttft_ms": round(first_token_time or 0, 1),
            "tps": round(tps, 1),
            "gpu_baseline_tps": 34.0,
            "gpu_baseline_ttft_ms": 192,
        }
        print(f"\n[OK] LLM NPU: ttft={result['ttft_ms']}ms tps={result['tps']}", flush=True)
        return result

    except Exception as e:
        print(f"  FAIL: {e}", flush=True)
        # NPU might not support this model yet, try to get device list
        try:
            import openvino as ov
            core = ov.Core()
            return {
                "status": "FAIL",
                "device": DEVICE,
                "error": str(e),
                "available_devices": core.available_devices,
                "note": "Qwen2.5-1.5B may require OV>=2026.0 for NPU; try GPU or CPU",
            }
        except Exception:
            return {"status": "FAIL", "error": str(e)}


# ─── Device Probe ─────────────────────────────────────────────────────────────

def probe_devices():
    """List all OpenVINO available devices and properties."""
    print("\n" + "="*60, flush=True)
    print("PROBE: OpenVINO Available Devices", flush=True)
    try:
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
    except Exception as e:
        return {"error": str(e)}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        help="all | embedding | reranker | ocr | asr | asr_diarization | llm_npu | probe")
    args = parser.parse_args()

    tasks = args.task.split(",") if args.task != "all" else \
        ["probe", "embedding", "reranker", "ocr", "asr", "asr_diarization", "llm_npu"]

    results = {"platform": "Intel Windows (Core Ultra 7 155H / Arc iGPU / AI Boost NPU)",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    task_map = {
        "probe": probe_devices,
        "embedding": validate_embedding,
        "reranker": validate_reranker,
        "ocr": validate_ocr,
        "asr": validate_asr,
        "asr_diarization": validate_asr_diarization,
        "llm_npu": validate_llm_npu,
    }

    for task in tasks:
        if task in task_map:
            print(f"\n{'#'*60}", flush=True)
            print(f"# Running: {task}", flush=True)
            results[task] = task_map[task]()
        else:
            print(f"Unknown task: {task}", flush=True)

    save_results(results)

    # Summary
    print("\n" + "="*60, flush=True)
    print("VALIDATION SUMMARY:", flush=True)
    for task, r in results.items():
        if isinstance(r, dict) and "status" in r:
            status = r["status"]
            icon = "[OK]" if "PASS" in status else ("[SKIP]" if "SKIP" in status else "[FAIL]")
            print(f"  {icon} {task}: {status}", flush=True)


if __name__ == "__main__":
    main()
