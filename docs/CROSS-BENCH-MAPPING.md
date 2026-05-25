# Cross-bench mapping: vlm-llm-benchmark <-> attune-bench

This repository (`vlm-llm-benchmark`) is the **validation framework**:
academic-grade methodology, metrics, judge calibration, regression CI,
and the methodology for evaluating model and pipeline quality.

`attune-bench` (Rust, separate repository) is the **product
performance harness**: criterion-driven micro-benchmarks of attune's
hot paths (vault, retrieval, agent dispatch, plugin install). It
generates HTML reports for tracking attune's own performance budgets.

They are complementary. This document records how to combine them.

## Mapping by dimension

| Dimension | `vlm-llm-benchmark` module | `attune-bench` bench |
|---|---|---|
| Retrieval accuracy | `benchmark/rag/retrieval_metrics.py` (NDCG / MRR / bpref / ERR / RBP) | `benches/retrieval_accuracy.rs` (precision/recall on attune's local corpus) |
| Generation quality | `benchmark/rag/answer_relevance.py` + `groundedness.py` | n/a (attune-bench does not score generations) |
| Reranker | `benchmark/rag/reranker.py` | `benches/retrieval_accuracy.rs` (reranker on/off) |
| Vault unlock latency | `benchmark/rag/canary.py::latency_budget` style | `benches/vault_unlock.rs` (criterion) |
| Plugin install | n/a | `benches/plugin_install.rs` |
| Chat E2E | `benchmark/rag/component_pipeline.py` traces | `benches/chat_e2e.rs` (wall-clock E2E) |
| Encryption overhead | n/a (out of scope) | `benches/encrypt_overhead.rs` |
| HDBSCAN clustering | n/a (out of scope) | `benches/hdbscan_accuracy.rs` |
| Token savings | `benchmark/rag/canary.py` shadow comparison | `benches/token_savings.rs` |
| Agent dispatch | `benchmark/rag/regression_ci.py` for quality | `benches/agent_dispatch.rs` for latency |

## Unified threshold convention

When both repos measure the same thing (e.g. retrieval), use these
shared thresholds:

| Metric | Pass | Warn | Fail | Source |
|---|---|---|---|---|
| NDCG@10 | >=0.65 | 0.50-0.65 | <0.50 | `quality_gate_matrix.yaml` |
| Vault unlock P50 | <100 ms | 100-200 | >200 ms | attune-bench |
| Plugin install P95 | <2 s | 2-5 s | >5 s | attune-bench |
| Groundedness | >=0.90 | 0.80-0.90 | <0.80 | `quality_gate_matrix.yaml` |

When a metric is owned by one repo, the other defers (no duplication).

## Embedding attune-bench HTML reports

attune-bench produces a criterion HTML report under
`target/criterion/<bench>/report/index.html`. To embed those alongside
this framework's reports:

```bash
# 1. In attune-bench, run criterion.
cd /path/to/attune-bench
cargo bench --bench retrieval_accuracy

# 2. Copy the HTML output into vlm-llm-benchmark's reports directory.
mkdir -p /path/to/vlm-llm-benchmark/reports/attune-bench
cp -r target/criterion/* /path/to/vlm-llm-benchmark/reports/attune-bench/

# 3. Link from vlm-llm-benchmark's release report.
echo "[attune-bench criterion]" \
     "(reports/attune-bench/retrieval_accuracy/report/index.html)" \
     >> /path/to/vlm-llm-benchmark/reports/runs/<ts>/README.md
```

## Boundary: what stays in attune-bench

attune-bench is **kept** in its own repository. It tests attune's
Rust internals; pulling those crates into the Python framework is
out of scope. The contract:

- attune-bench: production performance budgets, criterion harness.
- vlm-llm-benchmark: validation methodology + RAG quality.

If you want a single dashboard combining both, write a thin
aggregator that reads each repo's `reports/` directory.

## Why two repos

- attune-bench is a hot path tied to attune's Cargo workspace.
  Cross-compiling, criterion lifecycle, and Rust-only deps belong
  with the project that uses them.
- vlm-llm-benchmark is the methodology library. Many projects can
  depend on it (attune, attune-pro, cloud, RV) without inheriting
  attune's Rust toolchain.

The cost of separation is the small effort to sync threshold
conventions; that is the purpose of this document.
