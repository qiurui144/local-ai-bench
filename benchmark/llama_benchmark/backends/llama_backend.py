"""LlamaCppBackend：直接调用 llama-cpp-python，用于 RISC-V/RVV 或批量推理场景。"""

from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_backend(BackendType.LLAMA_CPP.value)
class LlamaCppBackend(AbstractModelBackend):
    """llama-cpp-python 进程内推理后端（Fallback）。

    适用场景：
    - RISC-V/RVV 平台（Ollama 无官方 riscv64 发行版）
    - 需要精细控制 n_gpu_layers / n_batch 等参数

    安装（RISC-V RVV）::

        CMAKE_ARGS="-DGGML_RVV=on" pip install llama-cpp-python --no-cache-dir

    安装（CUDA）::

        CMAKE_ARGS="-DGGML_CUDA=on" pip install llama-cpp-python --no-cache-dir
    """

    def load(self) -> None:
        try:
            from llama_cpp import Llama
        except ImportError:
            raise ImportError(
                "请安装 llama-cpp-python: pip install llama-cpp-python\n"
                "RISC-V RVV: CMAKE_ARGS='-DGGML_RVV=on' pip install llama-cpp-python\n"
                "CUDA: CMAKE_ARGS='-DGGML_CUDA=on' pip install llama-cpp-python"
            )

        model_path = str(self.config.path)
        logger.info(f"加载 llama.cpp 模型: {model_path}")

        self._model = Llama(
            model_path=model_path,
            n_ctx=self.config.context_length,
            n_gpu_layers=self.config.n_gpu_layers,
            n_threads=self.config.n_threads,
            n_batch=self.config.batch_size,
            verbose=False,
            logits_all=True,
            embedding=(self.config.type.value == "embedding"),
        )
        logger.info(f"llama.cpp 模型加载完成: {self.config.name}")

    def unload(self) -> None:
        if self._model is not None:
            del self._model
            self._model = None
            logger.info(f"llama.cpp 模型已释放: {self.config.name}")

    def generate(
        self,
        prompt: str,
        max_tokens: int = 512,
        temperature: float = 0.0,
        stop: Optional[List[str]] = None,
    ) -> str:
        self._require_loaded()
        output = self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stop=stop or [],
            echo=False,
        )
        return output["choices"][0]["text"]

    def get_logprobs(self, prompt: str, candidates: List[str]) -> Dict[str, float]:
        """获取候选 token 的 logprob（用于多选题评分）。

        通过 create_completion 接口获取 logprobs，避免直接调用内部 forward 方法。
        """
        self._require_loaded()

        # 使用 logprobs 参数获取 top-k token 概率
        output = self._model(
            prompt,
            max_tokens=1,
            temperature=0.0,
            logprobs=len(candidates) + 10,  # 请求足够多的 top token
            echo=False,
        )

        top_logprobs: Dict[str, float] = {}
        choices = output.get("choices", [])
        if choices and choices[0].get("logprobs"):
            top_logprobs = choices[0]["logprobs"].get("top_logprobs", [{}])[0]

        result: Dict[str, float] = {}
        for candidate in candidates:
            token = candidate.strip()
            # 尝试直接匹配（含空格前缀，llama.cpp tokenizer 行为）
            for key in (token, f" {token}", f"▁{token}"):
                if key in top_logprobs:
                    result[candidate] = top_logprobs[key]
                    break
            else:
                result[candidate] = float("-inf")
        return result

    def embed(self, texts: List[str]) -> np.ndarray:
        """批量 Embedding。"""
        self._require_loaded()
        embeddings = []
        for text in texts:
            emb = self._model.create_embedding(text)
            embeddings.append(emb["data"][0]["embedding"])
        return np.array(embeddings, dtype=np.float32)

    def measure_ttft(self, prompt: str, max_tokens: int = 128) -> Tuple[float, float]:
        """测量 TTFT 和总延迟（ms），通过流式生成计时。"""
        self._require_loaded()
        ttft_ms = 0.0
        first_token = True
        start = time.perf_counter_ns()

        for _ in self._model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            stream=True,
        ):
            if first_token:
                ttft_ms = (time.perf_counter_ns() - start) / 1_000_000
                first_token = False

        total_ms = (time.perf_counter_ns() - start) / 1_000_000
        return ttft_ms, total_ms

    def get_model_info(self) -> Dict[str, Any]:
        base = super().get_model_info()
        if self._model is not None:
            base.update({
                "n_ctx": self._model.n_ctx(),
                "n_gpu_layers": self.config.n_gpu_layers,
                "hardware": self.config.hardware.value,
            })
        return base

    def _require_loaded(self) -> None:
        if self._model is None:
            raise RuntimeError("LlamaCppBackend 未初始化，请先调用 load()")
