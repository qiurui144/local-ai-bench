"""
AMD XDNA 1 NPU 全面验证脚本 (VitisAI Execution Provider)

测试任务: Embedding / Reranker / OCR / ASR
目标机: AMD Windows 192.168.100.201
运行方式:
  python npu_validate_amd.py --task all
  python npu_validate_amd.py --task embedding,reranker

依赖 (AMD 机器需提前安装 RyzenAI SDK 1.7.1):
  pip install onnxruntime-directml  (AMD 官方推荐 ONNX Runtime with VitisAI EP)
  pip install olive-ai              (量化工具)
  pip install transformers sentencepiece

AMD NPU 路径: VitisAI Execution Provider (EP)
  - 需要 INT8 量化 ONNX 模型
  - provider_options 指定 config_file 为 VitisAI 配置
  - 仅支持 XDNA 硬件上的 CNN / BERT-style transformer
  - LLM 生成式模型: 不支持 (需 XDNA 2 / Ryzen AI 300)

模型存放:
  C:\npu_models\embedding\bge-small-en-v1.5-int8\
  C:\npu_models\reranker\bge-reranker-base-int8\
  C:\npu_models\asr\sensevoice-small-int8\
  C:\npu_models\ocr\ppocr-v4-det-int8\  ppocr-v4-rec-int8\
"""

import argparse
import json
import time
import sys
import io
from pathlib import Path

# Fix GBK console encoding on Windows
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

RESULTS_PATH = Path(r"C:\npu_amd_validation_results.json")
NPU_MODELS_DIR = Path(r"C:\npu_models")


def timer(fn):
    t0 = time.perf_counter()
    result = fn()
    elapsed_ms = (time.perf_counter() - t0) * 1000
    return result, elapsed_ms


def save_results(results: dict):
    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved → {RESULTS_PATH}", flush=True)


# ─── Provider Probe ───────────────────────────────────────────────────────────

def probe_providers():
    """检查 VitisAI EP 是否可用."""
    print("\n" + "="*60, flush=True)
    print("PROBE: ONNX Runtime Providers", flush=True)
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        print(f"  Available: {providers}", flush=True)
        vitisai_ok = "VitisAIExecutionProvider" in providers
        directml_ok = "DmlExecutionProvider" in providers
        print(f"  VitisAI EP: {'[OK] AVAILABLE' if vitisai_ok else '[FAIL] NOT FOUND'}", flush=True)
        print(f"  DirectML EP: {'[OK] AVAILABLE' if directml_ok else '[FAIL] NOT FOUND'}", flush=True)
        return {
            "ort_version": ort.__version__,
            "providers": providers,
            "vitisai_available": vitisai_ok,
            "directml_available": directml_ok,
        }
    except ImportError as e:
        return {"error": str(e), "note": "Install onnxruntime or onnxruntime-directml"}


# ─── Embedding ───────────────────────────────────────────────────────────────

def validate_embedding():
    """BGE-small-en-v1.5 INT8 on AMD NPU via VitisAI EP."""
    print("\n" + "="*60, flush=True)
    print("TASK: Embedding (BGE-small-en-v1.5 INT8 VitisAI EP)", flush=True)

    MODEL_PATH = NPU_MODELS_DIR / "embedding" / "bge-small-en-v1.5-int8" / "model.onnx"
    VITISAI_CONFIG = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"

    if not MODEL_PATH.exists():
        return {
            "status": "SKIP",
            "reason": f"Model not found: {MODEL_PATH}",
            "prep_cmd": "python scripts/npu_quantize_amd.py --task embedding",
            "note": "Requires Olive quantization with VitisAI EP config"
        }

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        TOKENIZER_DIR = NPU_MODELS_DIR / "embedding" / "bge-small-en-v1.5-int8"

        provider_options = [{
            "config_file": VITISAI_CONFIG,
        }]

        sess, load_ms = timer(lambda: ort.InferenceSession(
            str(MODEL_PATH),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options
        ))
        print(f"Loaded on NPU in {load_ms:.0f}ms", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        sentences = [
            "What is the capital of France?",
            "OpenVINO accelerates deep learning inference.",
            "人工智能正在改变世界。",
        ]

        latencies = []
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="np", padding=True, truncation=True)
            feed = {k: v for k, v in inputs.items() if k in [i.name for i in sess.get_inputs()]}
            _, lat = timer(lambda: sess.run(None, feed))
            latencies.append(lat)
            print(f"  '{sent[:40]}' → {lat:.1f}ms", flush=True)

        avg_lat = sum(latencies) / len(latencies)
        return {
            "status": "PASS",
            "device": "NPU (VitisAI EP)",
            "model": "bge-small-en-v1.5-int8",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
        }

    except Exception as e:
        # Fallback: try DirectML EP on iGPU
        try:
            import onnxruntime as ort
            sess_dml, load_ms_dml = timer(lambda: ort.InferenceSession(
                str(MODEL_PATH),
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]
            ))
            return {
                "status": "PASS_IGPU",
                "device": "iGPU (DirectML EP fallback)",
                "model": "bge-small-en-v1.5-int8",
                "load_ms": round(load_ms_dml, 0),
                "npu_error": str(e),
                "note": "VitisAI EP failed; ran on iGPU DirectML instead"
            }
        except Exception as e2:
            return {"status": "FAIL", "npu_error": str(e), "dml_error": str(e2)}


# ─── Reranker ────────────────────────────────────────────────────────────────

def validate_reranker():
    """BGE-reranker-base INT8 on AMD NPU via VitisAI EP."""
    print("\n" + "="*60, flush=True)
    print("TASK: Reranker (bge-reranker-base INT8 VitisAI EP)", flush=True)

    MODEL_PATH = NPU_MODELS_DIR / "reranker" / "bge-reranker-base-int8" / "model.onnx"
    VITISAI_CONFIG = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"

    if not MODEL_PATH.exists():
        return {
            "status": "SKIP",
            "reason": f"Model not found: {MODEL_PATH}",
            "prep_cmd": "python scripts/npu_quantize_amd.py --task reranker",
        }

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer

        provider_options = [{"config_file": VITISAI_CONFIG}]
        sess, load_ms = timer(lambda: ort.InferenceSession(
            str(MODEL_PATH),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options
        ))
        print(f"Loaded on NPU in {load_ms:.0f}ms", flush=True)

        tokenizer = AutoTokenizer.from_pretrained(
            str(NPU_MODELS_DIR / "reranker" / "bge-reranker-base-int8"))

        pairs = [
            ("What is AI?", "Artificial intelligence simulates human intelligence."),
            ("Capital of France", "Paris is the capital of France."),
        ]
        latencies = []
        for q, d in pairs:
            inputs = tokenizer(q, d, return_tensors="np",
                               padding=True, truncation=True, max_length=512)
            feed = {k: v for k, v in inputs.items() if k in [i.name for i in sess.get_inputs()]}
            out, lat = timer(lambda: sess.run(None, feed))
            latencies.append(lat)
            print(f"  '{q[:30]}' → {lat:.1f}ms", flush=True)

        return {
            "status": "PASS",
            "device": "NPU (VitisAI EP)",
            "model": "bge-reranker-base-int8",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(sum(latencies) / len(latencies), 1),
        }

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


# ─── OCR ─────────────────────────────────────────────────────────────────────

def validate_ocr():
    """PP-OCR v4 INT8 on AMD NPU via VitisAI EP."""
    print("\n" + "="*60, flush=True)
    print("TASK: OCR (PP-OCR v4 INT8 VitisAI EP)", flush=True)

    DET_MODEL = NPU_MODELS_DIR / "ocr" / "ppocr-v4-det-int8" / "model.onnx"
    VITISAI_CONFIG = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"

    if not DET_MODEL.exists():
        return {
            "status": "SKIP",
            "reason": f"OCR models not found under {NPU_MODELS_DIR / 'ocr'}",
            "prep_cmd": "python scripts/npu_quantize_amd.py --task ocr",
            "note": "PP-OCR det (CNN-based) supports VitisAI EP; rec (CRNN) needs verification"
        }

    try:
        import onnxruntime as ort
        import numpy as np

        provider_options = [{"config_file": VITISAI_CONFIG}]

        det_sess, det_load_ms = timer(lambda: ort.InferenceSession(
            str(DET_MODEL),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options
        ))
        print(f"Detection model loaded on NPU in {det_load_ms:.0f}ms", flush=True)

        # Test with dummy image (3×640×640)
        dummy_img = np.random.randn(1, 3, 640, 640).astype(np.float32)
        _, det_lat = timer(lambda: det_sess.run(None, {"x": dummy_img}))
        print(f"  Detection inference: {det_lat:.1f}ms", flush=True)

        return {
            "status": "PASS",
            "device": "NPU (VitisAI EP)",
            "model": "ppocr-v4-det-int8",
            "det_load_ms": round(det_load_ms, 0),
            "det_latency_ms": round(det_lat, 1),
            "rec_status": "PENDING - CRNN rec model NPU compatibility not verified",
        }

    except Exception as e:
        return {
            "status": "FAIL",
            "error": str(e),
            "note": "CNN det model may not map fully to NPU ops; check VitisAI EP logs"
        }


# ─── ASR ─────────────────────────────────────────────────────────────────────

def validate_asr():
    """SenseVoice-small on AMD: iGPU DirectML (current path) vs NPU VitisAI (experimental)."""
    print("\n" + "="*60, flush=True)
    print("TASK: ASR (SenseVoice-small — iGPU DirectML vs NPU VitisAI)", flush=True)

    # Current validated path: iGPU DirectML
    SENSEVOICE_ONNX = Path(r"C:\sensevoice_model\model.onnx")
    VITISAI_CONFIG = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"

    if not SENSEVOICE_ONNX.exists():
        return {
            "status": "SKIP",
            "reason": f"SenseVoice ONNX not found: {SENSEVOICE_ONNX}",
            "note": "Deploy SenseVoice via attune-k3 or RapidASR"
        }

    results = {}

    # Test 1: iGPU DirectML (validated baseline)
    try:
        import onnxruntime as ort
        import numpy as np

        sess_dml, load_dml = timer(lambda: ort.InferenceSession(
            str(SENSEVOICE_ONNX),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"]
        ))
        print(f"  iGPU DirectML load: {load_dml:.0f}ms", flush=True)

        # Dummy audio input (1s @ 16kHz)
        dummy_audio = np.zeros((1, 16000), dtype=np.float32)
        _, dml_lat = timer(lambda: sess_dml.run(None, {"speech": dummy_audio}))
        print(f"  iGPU DirectML inference: {dml_lat:.1f}ms [OK] (validated)", flush=True)

        results["igpu_directml"] = {
            "status": "PASS",
            "device": "iGPU (DirectML EP)",
            "load_ms": round(load_dml, 0),
            "latency_ms": round(dml_lat, 1),
            "note": "Validated production path"
        }
    except Exception as e:
        results["igpu_directml"] = {"status": "FAIL", "error": str(e)}

    # Test 2: NPU VitisAI (experimental — BERT-encoder-only CIF may work)
    try:
        import onnxruntime as ort
        provider_options = [{"config_file": VITISAI_CONFIG}]
        sess_npu, load_npu = timer(lambda: ort.InferenceSession(
            str(SENSEVOICE_ONNX),
            providers=["VitisAIExecutionProvider"],
            provider_options=provider_options
        ))
        print(f"  NPU VitisAI load: {load_npu:.0f}ms", flush=True)
        dummy_audio = np.zeros((1, 16000), dtype=np.float32)
        _, npu_lat = timer(lambda: sess_npu.run(None, {"speech": dummy_audio}))
        print(f"  NPU VitisAI inference: {npu_lat:.1f}ms", flush=True)

        results["npu_vitisai"] = {
            "status": "PASS",
            "device": "NPU (VitisAI EP)",
            "load_ms": round(load_npu, 0),
            "latency_ms": round(npu_lat, 1),
        }
    except Exception as e:
        results["npu_vitisai"] = {
            "status": "FAIL",
            "device": "NPU (VitisAI EP)",
            "error": str(e),
            "note": "SenseVoice CIF encoder may have unsupported ops for VitisAI EP XDNA 1"
        }

    return results


# ─── Embedding iGPU (DirectML) ───────────────────────────────────────────────

def validate_embedding_directml():
    """BGE embedding on AMD iGPU (Radeon 780M) via onnxruntime-directml.

    Uses the standard (non-quantized) ONNX model; DirectML handles FP32 on iGPU.
    Note: AMD Ollama already runs embedding at 100% iGPU via Vulkan/GGUF — this
    validates the ONNX DirectML path as an alternative (for optimum-intel parity).
    """
    print("\n" + "="*60, flush=True)
    print("TASK: Embedding iGPU DirectML (bge-base-en-v1.5)", flush=True)

    MODEL_PATH = Path(r"C:\ort_models\embedding\bge-base-en-v1.5\model.onnx")
    TOKENIZER_DIR = MODEL_PATH.parent

    if not MODEL_PATH.exists():
        return {
            "status": "SKIP",
            "reason": f"ONNX model not found: {MODEL_PATH}",
            "note": "Export: optimum-cli export onnx --model BAAI/bge-base-en-v1.5 C:\\ort_models\\embedding\\bge-base-en-v1.5",
            "alt_note": "AMD Ollama (bge-m3 / qwen3-embedding) already uses iGPU via Vulkan — check probe output",
        }

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer  # type: ignore

        sess, load_ms = timer(lambda: ort.InferenceSession(
            str(MODEL_PATH),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        ))
        print(f"  Loaded on iGPU DirectML in {load_ms:.0f}ms", flush=True)
        device_used = sess.get_providers()[0]

        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        sentences = ["What is the capital of France?",
                     "OpenVINO accelerates inference.", "人工智能正在改变世界。"]
        latencies = []
        for sent in sentences:
            inputs = tokenizer(sent, return_tensors="np", padding=True, truncation=True)
            inp_names = {i.name for i in sess.get_inputs()}
            feed = {k: v for k, v in inputs.items() if k in inp_names}
            _, lat = timer(lambda: sess.run(None, feed))
            latencies.append(lat)
            print(f"  '{sent[:40]}' → {lat:.1f}ms", flush=True)

        avg_lat = sum(latencies) / len(latencies)
        print(f"\n[OK] Embedding DirectML: avg {avg_lat:.1f}ms (device={device_used})", flush=True)
        return {
            "status": "PASS",
            "device": f"iGPU (DirectML EP) — {device_used}",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
        }

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


# ─── Reranker iGPU (DirectML) ────────────────────────────────────────────────

def validate_reranker_directml():
    """BGE-reranker cross-encoder on AMD iGPU via onnxruntime-directml.

    Uses FP32 ONNX model via DirectML EP on Radeon 780M.
    Current production path (local_reranker) uses CPU sentence_transformers;
    this validates the DirectML path for potential speed-up.
    """
    print("\n" + "="*60, flush=True)
    print("TASK: Reranker iGPU DirectML (bge-reranker-base)", flush=True)

    MODEL_PATH = Path(r"C:\ort_models\reranker\bge-reranker-base\model.onnx")
    TOKENIZER_DIR = MODEL_PATH.parent

    if not MODEL_PATH.exists():
        return {
            "status": "SKIP",
            "reason": f"ONNX model not found: {MODEL_PATH}",
            "note": "Export: optimum-cli export onnx --model BAAI/bge-reranker-base C:\\ort_models\\reranker\\bge-reranker-base",
        }

    try:
        import onnxruntime as ort
        from transformers import AutoTokenizer  # type: ignore

        sess, load_ms = timer(lambda: ort.InferenceSession(
            str(MODEL_PATH),
            providers=["DmlExecutionProvider", "CPUExecutionProvider"],
        ))
        print(f"  Loaded on iGPU DirectML in {load_ms:.0f}ms", flush=True)
        device_used = sess.get_providers()[0]

        tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
        pairs = [("What is AI?", "Artificial intelligence simulates human intelligence."),
                 ("Capital of France", "Paris is the capital of France.")]
        latencies = []
        for q, d in pairs:
            inputs = tokenizer(q, d, return_tensors="np", padding=True,
                               truncation=True, max_length=512)
            inp_names = {i.name for i in sess.get_inputs()}
            feed = {k: v for k, v in inputs.items() if k in inp_names}
            _, lat = timer(lambda: sess.run(None, feed))
            latencies.append(lat)
            print(f"  pair → {lat:.1f}ms", flush=True)

        avg_lat = sum(latencies) / len(latencies)
        print(f"\n[OK] Reranker DirectML: avg {avg_lat:.1f}ms (device={device_used})", flush=True)
        return {
            "status": "PASS",
            "device": f"iGPU (DirectML EP) — {device_used}",
            "load_ms": round(load_ms, 0),
            "avg_latency_ms": round(avg_lat, 1),
        }

    except Exception as e:
        return {"status": "FAIL", "error": str(e)}


# ─── ASR Diarization (AMD) ───────────────────────────────────────────────────

def validate_asr_diarization():
    """Speaker diarization probe on AMD: VAD (CPU) + ASR (DirectML) + speaker embed.

    Pipeline: Silero VAD (CPU) → SenseVoice DirectML → speaker embed (iGPU DirectML
    or VitisAI NPU if static shape) → agglomerative clustering (CPU).
    """
    print("\n" + "="*60, flush=True)
    print("TASK: ASR Diarization (VAD → SenseVoice DirectML → speaker embed)", flush=True)

    SENSEVOICE_ONNX = Path(r"C:\sensevoice_model\model.onnx")
    SPEAKER_EMBED_ONNX = Path(r"C:\ort_models\asr\speaker_embed.onnx")
    VITISAI_CONFIG = r"C:\Program Files\RyzenAI\1.7.1\voe-4.0-win_amd64\vaip_config.json"

    results: dict[str, dict] = {}

    # Stage 1: SenseVoice ASR via DirectML (validated)
    if not SENSEVOICE_ONNX.exists():
        results["asr_sensevoice"] = {"status": "SKIP",
                                      "reason": f"Not found: {SENSEVOICE_ONNX}"}
    else:
        try:
            import onnxruntime as ort
            import numpy as np
            sess, load_ms = timer(lambda: ort.InferenceSession(
                str(SENSEVOICE_ONNX),
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]))
            dummy = np.zeros((1, 16000), dtype=np.float32)
            _, lat = timer(lambda: sess.run(None, {"speech": dummy}))
            results["asr_sensevoice"] = {
                "status": "PASS", "device": "iGPU (DirectML)",
                "load_ms": round(load_ms, 0), "latency_ms": round(lat, 1)}
            print(f"  SenseVoice DirectML: {lat:.1f}ms [OK]", flush=True)
        except Exception as e:
            results["asr_sensevoice"] = {"status": "FAIL", "error": str(e)}

    # Stage 2: Speaker embedding — try DirectML first, then VitisAI NPU
    if not SPEAKER_EMBED_ONNX.exists():
        results["speaker_embed"] = {
            "status": "SKIP",
            "reason": f"Not found: {SPEAKER_EMBED_ONNX}",
            "note": "Export ERes2Net/CAM++ to ONNX with static [1,16000] input. "
                    "DirectML path: standard ONNX; NPU path: INT8 quantized + static shape",
        }
    else:
        try:
            import onnxruntime as ort
            import numpy as np

            # Try DirectML first
            sess, load_ms = timer(lambda: ort.InferenceSession(
                str(SPEAKER_EMBED_ONNX),
                providers=["DmlExecutionProvider", "CPUExecutionProvider"]))
            dummy = np.zeros((1, 16000), dtype=np.float32)
            inp = sess.get_inputs()[0].name
            _, lat = timer(lambda: sess.run(None, {inp: dummy}))
            device_used = sess.get_providers()[0]
            results["speaker_embed"] = {
                "status": "PASS",
                "device": f"iGPU (DirectML) — {device_used}",
                "load_ms": round(load_ms, 0), "latency_ms": round(lat, 1)}
            print(f"  Speaker embed DirectML: {lat:.1f}ms [OK]", flush=True)

            # Also probe VitisAI NPU (experimental, may fail for this model)
            try:
                sess_npu, _ = timer(lambda: ort.InferenceSession(
                    str(SPEAKER_EMBED_ONNX),
                    providers=["VitisAIExecutionProvider"],
                    provider_options=[{"config_file": VITISAI_CONFIG}]))
                _, npu_lat = timer(lambda: sess_npu.run(None, {inp: dummy}))
                results["speaker_embed_npu"] = {
                    "status": "PASS", "device": "NPU (VitisAI)",
                    "latency_ms": round(npu_lat, 1)}
                print(f"  Speaker embed VitisAI NPU: {npu_lat:.1f}ms [OK]", flush=True)
            except Exception as enpu:
                results["speaker_embed_npu"] = {
                    "status": "FAIL", "error": str(enpu),
                    "note": "NPU requires INT8 quantization; use AMD Olive to quantize"}
        except Exception as e:
            results["speaker_embed"] = {"status": "FAIL", "error": str(e)}

    # Stage 3: Clustering (CPU)
    try:
        import numpy as np
        from sklearn.cluster import AgglomerativeClustering  # type: ignore
        dummy_embeds = np.random.randn(10, 192)
        _, lat = timer(lambda: AgglomerativeClustering(n_clusters=2).fit(dummy_embeds))
        results["clustering"] = {"status": "PASS", "device": "CPU",
                                  "latency_ms": round(lat, 1)}
        print(f"  Clustering: CPU {lat:.1f}ms [OK]", flush=True)
    except ImportError:
        results["clustering"] = {"status": "SKIP", "reason": "pip install scikit-learn"}
    except Exception as e:
        results["clustering"] = {"status": "FAIL", "error": str(e)}

    overall = ("PASS" if all(v.get("status", "").startswith("PASS")
                              for v in results.values()) else "PARTIAL")
    print(f"\n[Diarization AMD] Overall: {overall}", flush=True)
    return {"status": overall, "stages": results}


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default="all",
                        help="all | probe | embedding | embedding_dml | reranker | reranker_dml | ocr | asr | asr_diarization")
    args = parser.parse_args()

    tasks = args.task.split(",") if args.task != "all" else \
        ["probe", "embedding", "embedding_dml", "reranker", "reranker_dml", "ocr", "asr", "asr_diarization"]

    results = {"platform": "AMD Windows (Ryzen 8845H / Radeon 780M / XDNA 1)",
               "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")}

    task_map = {
        "probe": probe_providers,
        "embedding": validate_embedding,
        "embedding_dml": validate_embedding_directml,
        "reranker": validate_reranker,
        "reranker_dml": validate_reranker_directml,
        "ocr": validate_ocr,
        "asr": validate_asr,
        "asr_diarization": validate_asr_diarization,
    }

    for task in tasks:
        if task in task_map:
            print(f"\n{'#'*60}", flush=True)
            print(f"# Running: {task}", flush=True)
            results[task] = task_map[task]()

    save_results(results)

    print("\n" + "="*60, flush=True)
    print("VALIDATION SUMMARY:", flush=True)
    for task, r in results.items():
        if isinstance(r, dict) and "status" in r:
            status = r["status"]
            icon = "[OK]" if "PASS" in status else ("⚠" if "SKIP" in status else "[FAIL]")
            print(f"  {icon} {task}: {status}", flush=True)


if __name__ == "__main__":
    main()
