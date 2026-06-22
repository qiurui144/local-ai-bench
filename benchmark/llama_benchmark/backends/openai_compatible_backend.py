"""OpenAI-compatible 通用后端。

覆盖所有实现 OpenAI REST API 协议的推理服务：
  - LLM：vLLM、LMDeploy、SGLang、LocalAI、LiteLLM
  - Embedding：Infinity、HuggingFace TEI（Text Embeddings Inference）
  - Rerank：Jina Reranker API、TEI reranker 端点
  - ASR：OpenAI Whisper API（/v1/audio/transcriptions）

配置示例（models.yaml）::

    - name: "qwen2.5-7b-vllm"
      type: llm
      backend: openai_compatible
      openai_base_url: "http://localhost:8000/v1"
      openai_model: "Qwen/Qwen2.5-7B-Instruct"
      openai_api_key: "EMPTY"

    - name: "bge-m3-tei"
      type: embedding
      backend: openai_compatible
      openai_base_url: "http://localhost:8080"   # TEI 不需要 /v1 前缀
      openai_model: "BAAI/bge-m3"
"""

from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_backend(BackendType.OPENAI_COMPATIBLE.value)
class OpenAICompatibleBackend(AbstractModelBackend):
    """OpenAI REST API 兼容后端。

    使用 httpx 直接调用，不依赖 openai 包，无需额外安装。
    自动识别 TEI（无 /v1 前缀）和标准 OpenAI 格式（含 /v1 前缀）。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._base_url: str = config.openai_base_url or "http://localhost:8000/v1"
        # 统一去除尾部斜杠
        self._base_url = self._base_url.rstrip("/")
        self._api_key: str = config.openai_api_key or "EMPTY"
        self._model_name: str = config.openai_model or config.name
        self._headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    def load(self) -> None:
        """检查服务可达性。"""
        import httpx

        health_urls = [
            f"{self._base_url}/health",
            f"{self._base_url}/models",
            # TEI / Infinity
            f"{self._base_url.rstrip('/v1')}/health",
        ]
        for url in health_urls:
            try:
                resp = httpx.get(url, headers=self._headers, timeout=5)
                if resp.status_code < 500:
                    logger.info(
                        f"OpenAI-compatible 服务已连接: {self._base_url} "
                        f"(model={self._model_name})"
                    )
                    return
            except Exception:
                continue

        logger.warning(
            f"OpenAI-compatible 服务健康检查失败: {self._base_url}，"
            "继续尝试（服务可能不提供 /health 端点）"
        )

    def unload(self) -> None:
        pass  # 远程服务，无需释放本地资源

    # ── LLM 接口 ──────────────────────────────────────────────────────────────

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        import httpx

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        if not self.config.extra.get("ollama_think", True):
            payload["options"] = {"think": False}
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    def generate_with_stats(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """非流式生成，同时返回 usage token 统计。

        返回 (response_text, stats_dict)，stats_dict 包含：
          - prompt_tokens:      输入 token 数
          - completion_tokens:  输出 token 数
          - tokens_per_second:  估算值（需要 total_ms 外部计时）
        """
        import httpx
        import time as _time

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": False,
        }
        start_ns = _time.perf_counter_ns()
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=300,
        )
        elapsed_ms = (_time.perf_counter_ns() - start_ns) / 1_000_000
        resp.raise_for_status()
        data = resp.json()

        usage = data.get("usage", {})
        completion_tokens = usage.get("completion_tokens", 0)
        tps = (completion_tokens / elapsed_ms * 1000) if elapsed_ms > 0 else 0.0

        stats = {
            "prompt_tokens": usage.get("prompt_tokens", 0),
            "completion_tokens": completion_tokens,
            "total_tokens": usage.get("total_tokens", 0),
            "tokens_per_second": round(tps, 2),
            "total_latency_ms": round(elapsed_ms, 2),
        }
        text = data["choices"][0]["message"]["content"]
        return text, stats

    def generate_with_logprobs(
        self,
        prompt: str,
        candidates: List[str],
        max_tokens: int = 1,
    ) -> Dict[str, float]:
        """从 /v1/chat/completions logprobs 中提取候选 token 的概率。

        兼容 vLLM / LMDeploy / SGLang（均支持 top_logprobs 参数）。
        """
        import httpx

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": 20,  # 拿足够多以覆盖 A/B/C/D
            "stream": False,
        }
        resp = httpx.post(
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # 解析 top_logprobs
        result: Dict[str, float] = {c: float("-inf") for c in candidates}
        try:
            logprobs_content = (
                data["choices"][0]
                .get("logprobs", {})
                .get("content", [])
            )
            if logprobs_content:
                top = {
                    item["token"]: item["logprob"]
                    for item in logprobs_content[0].get("top_logprobs", [])
                }
                for candidate in candidates:
                    token = candidate.strip()
                    if token in top:
                        result[candidate] = top[token]
                    elif token[0] in top:
                        result[candidate] = top[token[0]]
        except (KeyError, IndexError):
            pass

        return result

    def measure_ttft(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Tuple[float, float]:
        """通过 SSE 流式响应测量 TTFT（Time To First Token）。"""
        import httpx

        messages = [{"role": "user", "content": prompt}]
        payload = {
            "model": self._model_name,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": 0.0,
            "stream": True,
        }

        start = time.perf_counter_ns()
        ttft_ms = 0.0
        first_token_seen = False

        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers=self._headers,
            json=payload,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                data_str = line[5:].strip()
                if data_str == "[DONE]":
                    break
                try:
                    chunk = json.loads(data_str)
                    delta = chunk["choices"][0].get("delta", {})
                    if not first_token_seen and delta.get("content"):
                        ttft_ms = (time.perf_counter_ns() - start) / 1_000_000
                        first_token_seen = True
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue

        total_ms = (time.perf_counter_ns() - start) / 1_000_000
        return ttft_ms, total_ms

    # ── Embedding 接口 ────────────────────────────────────────────────────────

    def embed(self, texts: List[str]) -> np.ndarray:
        """调用 /v1/embeddings（OpenAI 标准格式，兼容 Infinity / TEI）。"""
        import httpx

        payload = {"model": self._model_name, "input": texts}
        resp = httpx.post(
            f"{self._base_url}/embeddings",
            headers=self._headers,
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = [item["embedding"] for item in data["data"]]
        return np.array(embeddings, dtype=np.float32)

    # ── Rerank 接口 ───────────────────────────────────────────────────────────

    def rerank_score(self, query: str, documents: List[str]) -> List[float]:
        """调用 /v1/rerank（Jina / TEI reranker 格式）。

        若服务端不支持 rerank 端点，自动降级到 embedding cosine 相似度。
        """
        import httpx

        # 尝试原生 rerank 端点（TEI / Jina 格式）
        payload = {
            "model": self._model_name,
            "query": query,
            "documents": documents,
            "return_documents": False,
        }
        try:
            resp = httpx.post(
                f"{self._base_url}/rerank",
                headers=self._headers,
                json=payload,
                timeout=120,
            )
            resp.raise_for_status()
            results = resp.json().get("results", [])
            # 按原始顺序排列分数
            scores = [0.0] * len(documents)
            for item in results:
                scores[item["index"]] = item["relevance_score"]
            return scores
        except Exception:
            logger.debug("rerank 端点不可用，降级到 embedding cosine 相似度")

        # 降级：embedding cosine 相似度（同 OllamaBackend）
        pairs = [f"Query: {query}\nDocument: {doc}" for doc in documents]
        query_emb = self.embed([query])[0]
        doc_embs = self.embed(pairs)
        q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        d_norm = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        return (d_norm @ q_norm).tolist()

    # ── ASR 接口 ──────────────────────────────────────────────────────────────

    def transcribe(
        self,
        audio_path: str,
        language: Optional[str] = "en",
        beam_size: int = 5,
    ) -> Tuple[str, float]:
        """调用 /v1/audio/transcriptions（OpenAI Whisper API 格式）。"""
        import httpx

        start = time.perf_counter_ns()
        with open(audio_path, "rb") as f:
            files = {"file": (audio_path, f, "audio/wav")}
            data = {"model": self._model_name}
            if language:
                data["language"] = language

            headers = {"Authorization": f"Bearer {self._api_key}"}
            resp = httpx.post(
                f"{self._base_url}/audio/transcriptions",
                headers=headers,
                data=data,
                files=files,
                timeout=300,
            )
        resp.raise_for_status()
        latency_ms = (time.perf_counter_ns() - start) / 1_000_000
        return resp.json().get("text", ""), latency_ms

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        base.update({
            "openai_base_url": self._base_url,
            "openai_model": self._model_name,
        })
        return base

    def _require_loaded(self) -> None:
        pass  # 无状态，不需要显式初始化检查
