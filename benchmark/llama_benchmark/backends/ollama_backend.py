"""OllamaBackend：通过 Ollama REST API 调用 LLM / Embedding / Rerank / Whisper。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_backend(BackendType.OLLAMA.value)
class OllamaBackend(AbstractModelBackend):
    """通过 Ollama HTTP API 进行推理的后端。

    Ollama 作为独立服务运行，本类仅封装 HTTP 调用。
    load() 检查服务健康状态和模型是否已拉取。
    unload() 为空操作（Ollama 自行管理模型内存）。
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        self._client = None
        self._base_url: str = "http://localhost:11434"

    def load(self) -> None:
        try:
            import ollama
        except ImportError:
            raise ImportError("请安装 ollama 包: pip install ollama")

        self._client = ollama.Client(host=self._base_url)
        self._check_service_health()
        logger.info(f"OllamaBackend 已连接: {self._base_url}, 模型: {self.config.ollama_model}")

    def configure(self, base_url: str) -> None:
        """在 load() 前设置 Ollama 服务地址。"""
        self._base_url = base_url

    def unload(self) -> None:
        """Ollama 自行管理模型内存，此处无需操作。"""
        self._client = None

    def _check_service_health(self) -> None:
        """检查 Ollama 服务是否可达，以及目标模型是否已加载。"""
        import httpx
        try:
            resp = httpx.get(f"{self._base_url}/api/tags", timeout=5)
            resp.raise_for_status()
        except Exception as e:
            raise ConnectionError(
                f"无法连接 Ollama 服务 {self._base_url}: {e}\n"
                "请确认 Ollama 已启动: ollama serve"
            )

        models_data = resp.json()
        available = [m["name"] for m in models_data.get("models", [])]
        model_name = self.config.ollama_model
        # 检查精确匹配或前缀匹配（ollama 名称可能带/不带 tag）
        if not any(m.startswith(model_name.split(":")[0]) for m in available):
            logger.warning(
                f"模型 '{model_name}' 未在 Ollama 中找到。"
                f"可用模型: {available}\n"
                f"请先执行: ollama pull {model_name}"
            )

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        system: Optional[str] = None,
        stream: bool = False,
    ) -> str:
        """文本生成（非流式）。"""
        self._require_loaded()

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})

        response = self._client.chat(
            model=self.config.ollama_model,
            messages=messages,
            options={
                "temperature": temperature,
                "num_predict": max_tokens,
                "seed": 42,
            },
            stream=False,
        )
        return response.message.content

    def generate_with_logprobs(
        self,
        prompt: str,
        candidates: List[str],
        max_tokens: int = 1,
    ) -> Dict[str, float]:
        """获取候选 token 的 logprob，用于多选题评分（MMLU / HellaSwag）。

        Ollama v0.12.11+ 支持 logprobs 参数。
        返回 {candidate: log_probability} 映射。
        """
        self._require_loaded()
        import httpx

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "options": {"temperature": 0.0, "num_predict": max_tokens, "seed": 42},
            "stream": False,
            "logprobs": True,
        }

        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=60,
        )
        resp.raise_for_status()
        data = resp.json()

        # 提取第一个 token 的 logprobs
        logprobs_data = data.get("logprobs", {})
        top_logprobs = logprobs_data.get("top_logprobs", [{}])

        # 在 top_logprobs 中查找每个候选 token
        result: Dict[str, float] = {}
        first_top = top_logprobs[0] if top_logprobs else {}
        for candidate in candidates:
            # 尝试精确匹配，再尝试首字母（"A"/"B"/"C"/"D"）
            token = candidate.strip()
            if token in first_top:
                result[candidate] = first_top[token]
            elif token[0] in first_top:
                result[candidate] = first_top[token[0]]
            else:
                result[candidate] = float("-inf")

        return result

    def generate_with_stats(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> Tuple[str, Dict[str, Any]]:
        """非流式生成，同时返回 Ollama 服务端性能元数据。

        返回 (response_text, stats_dict)，stats_dict 包含：
          - eval_count:           实际生成的 token 数
          - eval_duration_ns:     生成阶段耗时（纳秒）
          - prompt_eval_count:    输入 token 数
          - prompt_eval_duration_ns: prompt 处理耗时（纳秒）
          - load_duration_ns:     模型加载耗时（纳秒，已缓存时接近 0）
          - tokens_per_second:    生成速率（eval_count / eval_duration）
        """
        self._require_loaded()
        import httpx

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "options": {"temperature": temperature, "num_predict": max_tokens, "seed": 42},
            "stream": False,
        }
        resp = httpx.post(
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=300,
        )
        resp.raise_for_status()
        data = resp.json()

        eval_count = data.get("eval_count", 0)
        eval_duration_ns = data.get("eval_duration", 0)
        prompt_eval_duration_ns = data.get("prompt_eval_duration", 0)
        load_duration_ns = data.get("load_duration", 0)

        tps = (eval_count / eval_duration_ns * 1e9) if eval_duration_ns > 0 else 0.0

        stats = {
            "eval_count": eval_count,
            "eval_duration_ns": eval_duration_ns,
            "prompt_eval_count": data.get("prompt_eval_count", 0),
            "prompt_eval_duration_ns": prompt_eval_duration_ns,
            "load_duration_ns": load_duration_ns,
            "tokens_per_second": round(tps, 2),
        }
        return data.get("response", ""), stats

    def measure_ttft(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Tuple[float, float]:
        """测量 TTFT（Time To First Token）和总延迟（ms）。

        通过流式响应计时：记录收到第一个 token 的时间。
        返回 (ttft_ms, total_latency_ms)。
        """
        self._require_loaded()
        import httpx
        import json

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
            "stream": True,
        }

        start = time.perf_counter_ns()
        ttft_ms = 0.0
        first_token = True
        last_chunk: Dict[str, Any] = {}

        with httpx.stream(
            "POST",
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if first_token and chunk.get("response"):
                    ttft_ms = (time.perf_counter_ns() - start) / 1_000_000
                    first_token = False
                if chunk.get("done"):
                    last_chunk = chunk
                    break

        total_ms = (time.perf_counter_ns() - start) / 1_000_000

        # 保存最后一帧包含的服务端元数据供调用方使用
        self._last_ttft_stats = {
            "eval_count": last_chunk.get("eval_count", 0),
            "eval_duration_ns": last_chunk.get("eval_duration", 0),
            "prompt_eval_count": last_chunk.get("prompt_eval_count", 0),
            "prompt_eval_duration_ns": last_chunk.get("prompt_eval_duration", 0),
            "load_duration_ns": last_chunk.get("load_duration", 0),
        }
        return ttft_ms, total_ms

    def stream_with_token_timing(
        self,
        prompt: str,
        max_tokens: int = 128,
    ) -> Tuple[List[str], List[float], float]:
        """流式生成并记录每个 token 的到达时间戳。

        返回 (token_texts, inter_token_ms, total_ms)：
          - token_texts:    各 token 文本片段列表
          - inter_token_ms: 相邻 token 间隔毫秒列表（[0] = TTFT）
          - total_ms:       完整生成总耗时（毫秒）
        """
        self._require_loaded()
        import httpx
        import json

        payload = {
            "model": self.config.ollama_model,
            "prompt": prompt,
            "options": {"temperature": 0.0, "num_predict": max_tokens},
            "stream": True,
        }

        token_texts: List[str] = []
        timestamps_ns: List[int] = []

        start_ns = time.perf_counter_ns()

        with httpx.stream(
            "POST",
            f"{self._base_url}/api/generate",
            json=payload,
            timeout=120,
        ) as response:
            for line in response.iter_lines():
                if not line:
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                token = chunk.get("response", "")
                if token:
                    timestamps_ns.append(time.perf_counter_ns())
                    token_texts.append(token)
                if chunk.get("done"):
                    break

        total_ms = (time.perf_counter_ns() - start_ns) / 1_000_000

        # 计算相邻 token 间隔（inter_token_ms[0] = TTFT）
        inter_token_ms: List[float] = []
        if timestamps_ns:
            inter_token_ms.append((timestamps_ns[0] - start_ns) / 1_000_000)
            for i in range(1, len(timestamps_ns)):
                inter_token_ms.append((timestamps_ns[i] - timestamps_ns[i - 1]) / 1_000_000)

        return token_texts, inter_token_ms, total_ms

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量 Embedding，返回 shape=(N, dim) 的 numpy 数组。"""
        self._require_loaded()
        import httpx

        payload = {
            "model": self.config.ollama_model,
            "input": texts,
        }
        resp = httpx.post(
            f"{self._base_url}/api/embed",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        embeddings = data.get("embeddings", [])
        return np.array(embeddings, dtype=np.float32)

    def rerank_score(self, query: str, documents: List[str]) -> List[float]:
        """Rerank 评分：通过 cross-encoder 模式计算 query-doc 相似度。

        将 query 和每个 document 拼接后做 embedding，
        用 cosine 相似度作为相关性分数。
        """
        self._require_loaded()

        # 构造 cross-encoder 格式的 query-doc 对
        pairs = [f"Query: {query}\nDocument: {doc}" for doc in documents]
        query_emb = self.embed([query])[0]
        doc_embs = self.embed(pairs)

        # 归一化后计算 cosine 相似度
        query_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-8)
        scores = (doc_norms @ query_norm).tolist()
        return scores

    def get_model_info(self) -> Dict[str, Any]:
        """获取 Ollama 中的模型信息。"""
        import httpx
        try:
            resp = httpx.post(
                f"{self._base_url}/api/show",
                json={"model": self.config.ollama_model},
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            return {
                "name": self.config.name,
                "type": self.config.type.value,
                "backend": self.config.backend.value,
                "ollama_model": self.config.ollama_model,
                "modelfile": data.get("modelfile", ""),
                "parameters": data.get("parameters", ""),
            }
        except Exception:
            return super().get_model_info()

    def _require_loaded(self) -> None:
        if self._client is None:
            raise RuntimeError("OllamaBackend 未初始化，请先调用 load()")
