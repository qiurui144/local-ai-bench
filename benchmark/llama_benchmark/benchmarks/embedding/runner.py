"""EmbeddingBenchmarkRunner。"""

from __future__ import annotations

from typing import List

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import TaskResult
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.EMBEDDING.value)
class EmbeddingBenchmarkRunner(AbstractBenchmarkRunner):
    """Embedding 模型 benchmark runner。"""

    supported_model_types = [ModelType.EMBEDDING.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(self.app_config.ollama.base_url)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        from benchmark.llama_benchmark.benchmarks.embedding.retrieval import run_retrieval
        from benchmark.llama_benchmark.benchmarks.embedding.similarity import run_similarity

        emb_cfg = self.app_config.benchmarks.embedding
        task_results: List[TaskResult] = []

        if emb_cfg.retrieval.enabled:
            logger.info(f"[{self.model_config.name}] 开始 Embedding 检索...")
            dataset = emb_cfg.mteb_tasks[0] if emb_cfg.mteb_tasks else "NQ"
            task_results.append(
                run_retrieval(
                    self._backend,
                    emb_cfg.retrieval,
                    self.model_config.name,
                    dataset,
                )
            )

        if emb_cfg.similarity.enabled:
            logger.info(f"[{self.model_config.name}] 开始语义相似度...")
            task_results.append(
                run_similarity(self._backend, emb_cfg.similarity, self.model_config.name)
            )

        return task_results

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
