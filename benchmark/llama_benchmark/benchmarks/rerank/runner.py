"""RerankBenchmarkRunner：BEIR 数据集 NDCG/MRR/MAP 评测。"""

from __future__ import annotations

import time
from typing import List

import numpy as np
from tqdm import tqdm

from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
from benchmark.llama_benchmark.core.config import ModelType
from benchmark.llama_benchmark.core.registry import create_backend, register_runner
from benchmark.llama_benchmark.core.result import (
    BenchmarkStatus,
    MetricResult,
    TaskResult,
)
from benchmark.llama_benchmark.metrics.ranking import MAPMetric, MRRMetric, NDCGMetric
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_runner(ModelType.RERANK.value)
class RerankBenchmarkRunner(AbstractBenchmarkRunner):
    """Rerank 模型 benchmark runner。

    流程：BM25 初检（top-100）→ Rerank 重排 → NDCG/MRR/MAP 评估。
    """

    supported_model_types = [ModelType.RERANK.value]

    def setup(self) -> None:
        self._backend = create_backend(self.model_config)
        if hasattr(self._backend, "configure"):
            self._backend.configure(self.app_config.ollama.base_url)
        self._backend.load()

    def run(self) -> List[TaskResult]:
        rerank_cfg = self.app_config.benchmarks.rerank
        if not rerank_cfg.tasks.enabled:
            return []

        task_results: List[TaskResult] = []
        for dataset_name in rerank_cfg.beir_datasets:
            logger.info(f"[{self.model_config.name}] Rerank BEIR: {dataset_name}")
            result = self._evaluate_beir(
                dataset_name,
                rerank_cfg.k_values,
                rerank_cfg.tasks,
            )
            task_results.append(result)
        return task_results

    def _evaluate_beir(self, dataset_name: str, k_values: List[int], config) -> TaskResult:
        start_time = time.time()
        task_name = f"rerank_{dataset_name}"

        try:
            corpus, queries, qrels = self._load_beir_dataset(dataset_name, config.dataset_path)
        except Exception as e:
            return TaskResult(
                task_name=task_name,
                model_name=self.model_config.name,
                metrics=[],
                num_samples=0,
                duration_seconds=0.0,
                status=BenchmarkStatus.ERROR,
                error_message=f"BEIR 数据集加载失败: {e}",
            )

        if config.num_samples:
            query_ids = list(queries.keys())[: config.num_samples]
        else:
            query_ids = list(queries.keys())

        # BM25 初检
        from rank_bm25 import BM25Okapi
        corpus_ids = list(corpus.keys())
        corpus_texts = [corpus[cid]["text"] for cid in corpus_ids]
        tokenized_corpus = [t.lower().split() for t in corpus_texts]
        bm25 = BM25Okapi(tokenized_corpus)

        all_relevance: List[List[float]] = []
        ndcg_10 = NDCGMetric(k=10)
        mrr_10 = MRRMetric(k=10)
        map_metric = MAPMetric()

        for qid in tqdm(query_ids, desc=f"Rerank {dataset_name}"):
            query_text = queries[qid]
            tokenized_q = query_text.lower().split()
            bm25_scores = bm25.get_scores(tokenized_q)

            # 取 top-100
            top_indices = np.argsort(-bm25_scores)[:100]
            candidate_ids = [corpus_ids[i] for i in top_indices]
            candidate_texts = [corpus[cid]["text"] for cid in candidate_ids]

            # Rerank
            try:
                rerank_scores = self._backend.rerank_score(query_text, candidate_texts)
            except Exception as e:
                logger.warning(f"Rerank 失败 query={qid}: {e}")
                rerank_scores = list(bm25_scores[top_indices])

            # 按 rerank 分数重排
            reranked = sorted(
                zip(candidate_ids, rerank_scores),
                key=lambda x: x[1],
                reverse=True,
            )

            # 构建相关性标签
            query_qrels = qrels.get(qid, {})
            relevance = [float(query_qrels.get(cid, 0)) for cid, _ in reranked]
            all_relevance.append(relevance)

        ndcg_val = ndcg_10.compute(all_relevance, [])
        mrr_val = mrr_10.compute(all_relevance, [])
        map_val = map_metric.compute(all_relevance, [])

        ndcg_threshold = config.thresholds.get("ndcg_at_10", None)
        ndcg_status = BenchmarkStatus.PASS
        if ndcg_threshold and not ndcg_threshold.check(ndcg_val):
            ndcg_status = BenchmarkStatus.FAIL

        overall = (
            BenchmarkStatus.FAIL
            if ndcg_status == BenchmarkStatus.FAIL
            else BenchmarkStatus.PASS
        )

        return TaskResult(
            task_name=task_name,
            model_name=self.model_config.name,
            metrics=[
                MetricResult(
                    name="ndcg_at_10",
                    value=round(ndcg_val, 4),
                    higher_is_better=True,
                    threshold=ndcg_threshold.min_value if ndcg_threshold else None,
                    status=ndcg_status,
                ),
                MetricResult(name="mrr_at_10", value=round(mrr_val, 4), higher_is_better=True),
                MetricResult(name="map", value=round(map_val, 4), higher_is_better=True),
            ],
            num_samples=len(query_ids),
            duration_seconds=time.time() - start_time,
            status=overall,
            metadata={"dataset": dataset_name},
        )

    def _load_beir_dataset(
        self, dataset_name: str, dataset_path=None
    ) -> tuple:
        """加载 BEIR 数据集，优先本地路径，否则从 HuggingFace 下载。"""
        try:
            from beir import util
            from beir.datasets.data_loader import GenericDataLoader

            if dataset_path:
                data_path = str(dataset_path)
            else:
                url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset_name}.zip"
                data_path = util.download_and_unzip(url, "data/beir")

            corpus, queries, qrels = GenericDataLoader(data_path).load(split="test")
            return corpus, queries, qrels
        except ImportError:
            raise ImportError("请安装 beir: pip install beir")

    def teardown(self) -> None:
        if self._backend is not None:
            self._backend.unload()
            self._backend = None
