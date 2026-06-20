"""
RK3588 RKNN HTTP Adapter — 在 RK3588 本机运行（不在 controller）。
提供 OpenAI-compat /v1/completions 和 /v1/chat/completions。
从 models.yaml extra.rknn_model_path 加载 .rknn 文件。

部署方式（RK3588 机器上）：
  python -m benchmark.backends.rknn_adapter --model /path/to/model.rknn --port 8080
"""
from __future__ import annotations
import argparse
import time
from typing import Optional

try:
    from rkllm.api import RKLLM  # rkllm 官方 Python binding
    _RKLLM_AVAILABLE = True
except ImportError:
    _RKLLM_AVAILABLE = False


def _build_app(model_path: str, verbose: bool = False):
    """构建 Flask app，返回 app 对象（支持 pytest 测试注入）。"""
    try:
        from flask import Flask, request, jsonify
    except ImportError:
        raise RuntimeError("pip install flask on RK3588 first")

    app = Flask(__name__)
    _model: Optional[object] = None

    def _get_model():
        nonlocal _model
        if _model is None:
            if not _RKLLM_AVAILABLE:
                raise RuntimeError("rkllm not installed; run on RK3588 with rkllm-toolkit")
            _model = RKLLM(model_path)
        return _model

    @app.route("/v1/models")
    def list_models():
        return jsonify({"object": "list", "data": [{"id": "rknn-model", "object": "model"}]})

    @app.route("/v1/chat/completions", methods=["POST"])
    def chat_completions():
        body = request.get_json()
        messages = body.get("messages", [])
        prompt = messages[-1].get("content", "") if messages else ""
        model = _get_model()
        t0 = time.monotonic()
        result = model.run(prompt)
        elapsed = time.monotonic() - t0
        return jsonify({
            "id": "rknn-0",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": "rknn-model",
            "choices": [{"message": {"role": "assistant", "content": result}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": len(prompt.split()),
                      "completion_tokens": len(result.split()),
                      "total_tokens": len(prompt.split()) + len(result.split()),
                      "_latency_s": elapsed},
        })

    return app


def main():
    parser = argparse.ArgumentParser(description="RKNN OpenAI-compat HTTP adapter")
    parser.add_argument("--model", required=True, help="Path to .rknn model file")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    app = _build_app(args.model, args.verbose)
    app.run(host="0.0.0.0", port=args.port)


if __name__ == "__main__":
    main()
