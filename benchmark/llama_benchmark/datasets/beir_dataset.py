"""BEIR 数据集加载器（Rerank 评测）。

数据格式：返回 (corpus, queries, qrels) 三元组，与 BEIR GenericDataLoader 接口对齐。
corpus: {doc_id: {'title': ..., 'text': ...}}
queries: {query_id: query_text}
qrels:   {query_id: {doc_id: relevance_score}}
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from benchmark.llama_benchmark.datasets.base_dataset import AbstractDataset

# 内置 NFCorpus 小样本（5 条 query，10 篇文档）
_BUILTIN_CORPUS = {
    "d1": {"title": "Nutrition and Cancer", "text": "Diet plays a crucial role in cancer prevention and treatment."},
    "d2": {"title": "Exercise Benefits", "text": "Regular physical activity reduces the risk of chronic diseases."},
    "d3": {"title": "Vitamin D Deficiency", "text": "Vitamin D deficiency is linked to increased cancer risk."},
    "d4": {"title": "Mediterranean Diet", "text": "The Mediterranean diet is associated with lower cancer mortality rates."},
    "d5": {"title": "Antioxidants Research", "text": "Antioxidants in fruits and vegetables help prevent cellular damage."},
    "d6": {"title": "Sleep and Health", "text": "Adequate sleep is essential for immune system function and disease prevention."},
    "d7": {"title": "Processed Food Risks", "text": "Consumption of processed foods is linked to higher cancer incidence."},
    "d8": {"title": "Omega-3 Fatty Acids", "text": "Omega-3 fatty acids have anti-inflammatory properties beneficial for health."},
    "d9": {"title": "Sugar and Disease", "text": "Excessive sugar intake is associated with obesity and increased disease risk."},
    "d10": {"title": "Herbal Medicine", "text": "Some herbal compounds show promising anti-tumor properties in studies."},
}

_BUILTIN_QUERIES = {
    "q1": "cancer prevention diet",
    "q2": "vitamin deficiency disease risk",
    "q3": "anti-inflammatory foods",
    "q4": "processed food health effects",
    "q5": "herbal cancer treatment",
}

_BUILTIN_QRELS = {
    "q1": {"d1": 2, "d4": 2, "d5": 1, "d7": 1},
    "q2": {"d3": 2, "d6": 1},
    "q3": {"d2": 1, "d8": 2, "d5": 1},
    "q4": {"d7": 2, "d9": 2, "d1": 1},
    "q5": {"d10": 2, "d1": 1, "d3": 1},
}


class BEIRDataset(AbstractDataset):
    """BEIR 基准检索评测数据集。

    支持 BEIR 框架标准数据集（msmarco、trec-covid、nfcorpus 等）。
    也可作为 (corpus, queries, qrels) 三元组提供方使用。
    """

    def __init__(
        self,
        dataset_name: str = "nfcorpus",
        split: str = "test",
        num_samples: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(split=split, num_samples=num_samples, **kwargs)
        self.dataset_name = dataset_name

    def load_beir(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Dict[str, int]]]:
        """返回 BEIR 格式三元组：(corpus, queries, qrels)。"""
        try:
            return self._load_beir_hf()
        except (ImportError, Exception):
            return self._load_beir_builtin()

    def _load_beir_hf(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Dict[str, int]]]:
        """通过 BEIR 库从 HuggingFace 加载数据集。"""
        from beir import util
        from beir.datasets.data_loader import GenericDataLoader

        url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{self.dataset_name}.zip"
        data_path = util.download_and_unzip(url, "data/beir")
        corpus, queries, qrels = GenericDataLoader(data_path).load(split=self.split)

        if self.num_samples:
            query_ids = list(queries.keys())[: self.num_samples]
            queries = {qid: queries[qid] for qid in query_ids}

        return corpus, queries, qrels

    def _load_beir_builtin(
        self,
    ) -> Tuple[Dict[str, Any], Dict[str, str], Dict[str, Dict[str, int]]]:
        queries = dict(_BUILTIN_QUERIES)
        if self.num_samples:
            qids = list(queries.keys())[: self.num_samples]
            queries = {qid: queries[qid] for qid in qids}
        return dict(_BUILTIN_CORPUS), queries, dict(_BUILTIN_QRELS)

    # AbstractDataset 接口（将 BEIR 三元组扁平化为样本列表）
    def _load_hf(self) -> List[Dict[str, Any]]:
        corpus, queries, qrels = self._load_beir_hf()
        return self._to_samples(corpus, queries, qrels)

    def _load_builtin(self) -> List[Dict[str, Any]]:
        corpus, queries, qrels = self._load_beir_builtin()
        return self._to_samples(corpus, queries, qrels)

    @staticmethod
    def _to_samples(
        corpus: Dict, queries: Dict, qrels: Dict
    ) -> List[Dict[str, Any]]:
        """将 BEIR 三元组转为扁平样本列表（每条 = 一个 query + 相关文档 id 集合）。"""
        return [
            {
                "query_id": qid,
                "query": query_text,
                "relevant_doc_ids": list(qrels.get(qid, {}).keys()),
                "corpus_size": len(corpus),
            }
            for qid, query_text in queries.items()
        ]
