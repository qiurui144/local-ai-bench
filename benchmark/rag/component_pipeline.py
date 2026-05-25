"""RAG component-wise pipeline stages and failure taxonomy.

Implements PDF Chapter 1 — Component-wise RAG evaluation.

The RAG pipeline is decomposed into 8 stages, each independently observable
and assessable:

    Ingest -> Chunk -> Index -> Retrieve -> Rerank -> PromptPack -> Generate -> Cite

Each stage emits a structured trace record (see schemas/rag_trace.schema.json)
that downstream evaluation modules consume. Failures are categorized via
`FailureKind` for failure-mode triage.

Why per-stage rather than end-to-end only:
- A bad answer can fail at any layer; attributing without staged traces
  forces guesswork.
- Latency budgets are owned per-stage; rolling up to end-to-end hides where
  the SLO blew up.
- Quality regressions (e.g. embedding model swap) localize to one stage.

Reference: production RAG evaluation playbook, Ch 1.
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional


class FailureKind(str, Enum):
    """Failure-mode taxonomy across the 8 RAG stages.

    Categories are designed so each failed answer can be tagged with exactly
    one primary kind plus optional secondaries; this keeps post-mortems
    actionable rather than narrative.
    """

    # Retrieval-side
    RETRIEVAL_MISS = "retrieval_miss"  # gold doc not in top-k
    RETRIEVAL_NOISE = "retrieval_noise"  # top-k full of off-topic chunks
    CHUNK_TOO_SMALL = "chunk_too_small"  # answer split across chunks
    CHUNK_TOO_LARGE = "chunk_too_large"  # signal diluted by noise

    # Generation-side
    HALLUCINATION = "hallucination"  # claim with no source backing
    PARTIAL_ANSWER = "partial_answer"  # missing some required facts
    OVER_REFUSAL = "over_refusal"  # refused when evidence sufficient
    UNDER_REFUSAL = "under_refusal"  # answered when evidence absent

    # Citation-side
    CITATION_MISSING = "citation_missing"  # claim has no cite at all
    CITATION_WRONG = "citation_wrong"  # cite points to wrong source
    CITATION_FABRICATED = "citation_fabricated"  # cite is invented

    # Cross-cutting
    PROMPT_INJECTION = "prompt_injection"  # adversarial input leaked through
    LATENCY_BLOWUP = "latency_blowup"  # stage exceeded budget
    SCHEMA_VIOLATION = "schema_violation"  # output failed format contract


@dataclass
class StageTrace:
    """Trace record for a single pipeline stage.

    Persisted for both offline and online runs; the offline/online alignment
    module diffs trace populations to spot drift.
    """

    stage: str
    started_at: float
    duration_ms: float
    input_summary: str
    output_summary: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    failure: Optional[FailureKind] = None
    error: Optional[str] = None


@dataclass
class PipelineTrace:
    """End-to-end trace for one query."""

    query_id: str
    query_text: str
    stages: List[StageTrace] = field(default_factory=list)
    final_answer: str = ""
    final_citations: List[str] = field(default_factory=list)

    def total_latency_ms(self) -> float:
        return sum(s.duration_ms for s in self.stages)

    def stage_named(self, name: str) -> Optional[StageTrace]:
        for s in self.stages:
            if s.stage == name:
                return s
        return None


class _Stage:
    """Base class for the 8 stages.

    Subclasses override `run`. The base wraps timing/error capture so every
    stage emits a uniform StageTrace.
    """

    name: str = "stage"

    def run(self, payload: Any) -> Any:  # pragma: no cover - abstract
        raise NotImplementedError

    def execute(self, payload: Any) -> tuple[Any, StageTrace]:
        t0 = time.perf_counter()
        in_sum = self._summarize(payload)
        try:
            out = self.run(payload)
            err: Optional[str] = None
        except Exception as exc:  # noqa: BLE001 — caller controls retry
            out = None
            err = f"{type(exc).__name__}: {exc}"
        dt_ms = (time.perf_counter() - t0) * 1000.0
        trace = StageTrace(
            stage=self.name,
            started_at=t0,
            duration_ms=dt_ms,
            input_summary=in_sum,
            output_summary=self._summarize(out),
            error=err,
        )
        return out, trace

    @staticmethod
    def _summarize(obj: Any) -> str:
        if obj is None:
            return "<none>"
        text = repr(obj)
        return text[:200]


class IngestStage(_Stage):
    """Ingest: raw doc -> normalized text + structural metadata."""

    name = "ingest"

    def __init__(self, normalizer: Optional[Callable[[str], str]] = None) -> None:
        self.normalizer = normalizer or (lambda s: s.strip())

    def run(self, raw_text: str) -> Dict[str, Any]:
        text = self.normalizer(raw_text)
        return {"text": text, "char_count": len(text)}


class ChunkStage(_Stage):
    """Chunk: text -> list of (idx, chunk_text, span)."""

    name = "chunk"

    def __init__(self, target_size: int = 400, overlap: int = 50) -> None:
        if target_size <= 0:
            raise ValueError("target_size must be > 0")
        if overlap < 0 or overlap >= target_size:
            raise ValueError("overlap must be in [0, target_size)")
        self.target_size = target_size
        self.overlap = overlap

    def run(self, ingested: Dict[str, Any]) -> List[Dict[str, Any]]:
        text = ingested["text"]
        chunks: List[Dict[str, Any]] = []
        step = self.target_size - self.overlap
        for i, start in enumerate(range(0, max(len(text), 1), step)):
            end = min(start + self.target_size, len(text))
            chunks.append(
                {
                    "idx": i,
                    "text": text[start:end],
                    "span": (start, end),
                }
            )
            if end >= len(text):
                break
        return chunks


class IndexStage(_Stage):
    """Index: chunks -> embedded + persisted records.

    Embedding function is injected to keep this module framework-agnostic.
    """

    name = "index"

    def __init__(self, embed_fn: Callable[[str], List[float]]) -> None:
        self.embed_fn = embed_fn

    def run(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out = []
        for c in chunks:
            out.append({**c, "embedding": self.embed_fn(c["text"])})
        return out


class RetrieveStage(_Stage):
    """Retrieve: query -> top-k chunk candidates by cosine similarity."""

    name = "retrieve"

    def __init__(
        self,
        embed_fn: Callable[[str], List[float]],
        corpus: List[Dict[str, Any]],
        top_k: int = 20,
    ) -> None:
        self.embed_fn = embed_fn
        self.corpus = corpus
        self.top_k = top_k

    @staticmethod
    def _cosine(a: List[float], b: List[float]) -> float:
        num = sum(x * y for x, y in zip(a, b))
        da = sum(x * x for x in a) ** 0.5
        db = sum(y * y for y in b) ** 0.5
        if da == 0 or db == 0:
            return 0.0
        return num / (da * db)

    def run(self, query: str) -> List[Dict[str, Any]]:
        q = self.embed_fn(query)
        scored = [
            {**c, "score": self._cosine(q, c["embedding"])} for c in self.corpus
        ]
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[: self.top_k]


class RerankStage(_Stage):
    """Rerank: top-k candidates -> reordered top-n via reranker model."""

    name = "rerank"

    def __init__(
        self,
        rerank_fn: Callable[[str, List[str]], List[float]],
        top_n: int = 5,
    ) -> None:
        self.rerank_fn = rerank_fn
        self.top_n = top_n

    def run(
        self, payload: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        query: str = payload["query"]
        candidates: List[Dict[str, Any]] = payload["candidates"]
        texts = [c["text"] for c in candidates]
        scores = self.rerank_fn(query, texts)
        for c, s in zip(candidates, scores):
            c["rerank_score"] = s
        candidates.sort(key=lambda x: x["rerank_score"], reverse=True)
        return candidates[: self.top_n]


class PromptPackStage(_Stage):
    """PromptPack: query + reranked chunks -> final prompt + cite-index map."""

    name = "prompt_pack"

    def __init__(self, system_preamble: str = "Answer using only the sources.") -> None:
        self.system_preamble = system_preamble

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        query: str = payload["query"]
        chunks: List[Dict[str, Any]] = payload["chunks"]
        cite_map = {f"[S{i+1}]": c for i, c in enumerate(chunks)}
        sources_block = "\n\n".join(
            f"[S{i+1}] {c['text']}" for i, c in enumerate(chunks)
        )
        prompt = (
            f"{self.system_preamble}\n\n"
            f"Sources:\n{sources_block}\n\n"
            f"Question: {query}\n"
            f"Answer (cite as [S#]):"
        )
        return {"prompt": prompt, "cite_map": cite_map}


class GenerateStage(_Stage):
    """Generate: prompt -> answer string via LLM."""

    name = "generate"

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        self.generate_fn = generate_fn

    def run(self, packed: Dict[str, Any]) -> str:
        return self.generate_fn(packed["prompt"])


class CiteStage(_Stage):
    """Cite: parse answer -> extract `[S#]` citations and verify against cite_map.

    Returns the parsed citations and flags `FailureKind.CITATION_FABRICATED`
    if any cite tag isn't in the map.
    """

    name = "cite"

    def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        import re

        answer: str = payload["answer"]
        cite_map: Dict[str, Dict[str, Any]] = payload["cite_map"]
        tags = re.findall(r"\[S\d+\]", answer)
        fabricated = [t for t in tags if t not in cite_map]
        return {
            "citations": tags,
            "fabricated": fabricated,
            "ok": len(fabricated) == 0,
        }


@dataclass
class ComponentPipeline:
    """Composes 8 stages, runs a query, returns a PipelineTrace.

    Wiring is explicit so each project can swap stages (e.g. use a hybrid
    retriever or skip the reranker for cost).
    """

    ingest: IngestStage
    chunk: ChunkStage
    index: IndexStage
    retrieve: RetrieveStage
    rerank: Optional[RerankStage]
    prompt_pack: PromptPackStage
    generate: GenerateStage
    cite: CiteStage

    def offline_index(self, raw_docs: List[str]) -> List[Dict[str, Any]]:
        all_chunks: List[Dict[str, Any]] = []
        for doc in raw_docs:
            ingested, _ = self.ingest.execute(doc)
            chunks, _ = self.chunk.execute(ingested)
            indexed, _ = self.index.execute(chunks)
            all_chunks.extend(indexed)
        # bind into the retrieve stage
        self.retrieve.corpus = all_chunks
        return all_chunks

    def query(self, query_text: str) -> PipelineTrace:
        trace = PipelineTrace(query_id=str(uuid.uuid4()), query_text=query_text)
        retrieved, t = self.retrieve.execute(query_text)
        trace.stages.append(t)
        candidates = retrieved or []
        if self.rerank is not None:
            reranked, t = self.rerank.execute(
                {"query": query_text, "candidates": candidates}
            )
            trace.stages.append(t)
            top = reranked or []
        else:
            top = candidates[: self.prompt_pack.__dict__.get("top_n", 5)]
        packed, t = self.prompt_pack.execute({"query": query_text, "chunks": top})
        trace.stages.append(t)
        answer, t = self.generate.execute(packed or {"prompt": ""})
        trace.stages.append(t)
        trace.final_answer = answer or ""
        cite_res, t = self.cite.execute(
            {"answer": trace.final_answer, "cite_map": (packed or {}).get("cite_map", {})}
        )
        trace.stages.append(t)
        if cite_res and not cite_res.get("ok", True):
            t.failure = FailureKind.CITATION_FABRICATED
        trace.final_citations = (cite_res or {}).get("citations", [])
        return trace
