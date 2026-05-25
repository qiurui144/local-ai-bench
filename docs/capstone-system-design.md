# Capstone — Reference RAG Validation System Design

This appendix sketches an end-to-end system that uses every component
in `benchmark/rag/` and `benchmark/rigor/`. It is meant to be a
reference architecture you can crib for your own deployment.

## 1. Objective

Operate a production RAG service with continuous quality and safety
guardrails. Required properties:

- Every release passes a documented gate matrix before promotion.
- Every production request emits a trace amenable to per-stage
  failure attribution.
- A canary process can roll back the candidate within minutes
  of a degradation.
- Drift between offline benchmark and online behavior is bounded
  and monitored.

## 2. Components

```
                +---------------------+
   user request | front-end gateway   |
                +----------+----------+
                           |
                           v
                +---------------------+
                | traffic splitter    | <-- TrafficSplitter (canary.py)
                +----------+----------+
                  control | candidate
                          |        |
                          v        v
                +---------------+ +---------------+
                | control RAG   | | candidate RAG |
                +---------------+ +---------------+
                          |        |
                          +----+---+
                               v
                +--------------------------+
                | trace logger             | <-- rag_trace.schema.json
                +-----------+--------------+
                            |
                            v
   +------------------+   +-------+   +----------------------+
   | online monitor   | <-+ logs  +-> | offline runner       |
   +--------+---------+   +-------+   +----------+-----------+
            |                                    |
            v                                    v
   +--------------------+              +--------------------+
   | drift detector +   |              | golden set + judge |
   | canary gate        |              | regression CI      |
   +--------+-----------+              +----------+---------+
            |                                     |
            v                                     v
        +-------+                          +--------------+
        | rollback policy                  | gate matrix  |
        +-------+                          +--------------+
```

## 3. Key decisions

- **Trace schema is the source of truth.** Every other tool reads
  these JSON traces. If a stage cannot emit a trace, it does not
  ship.
- **Judge calibration runs daily**, not per-release. A judge that
  silently drifts is a worse failure mode than a model that
  silently drifts.
- **Bucketed gate matrix.** A "global PASS" is never accepted; the
  per-bucket PASS must hold.
- **Canary gates require >=50 samples per window.** Faster
  promotion is not worth false positives.
- **Reproducibility snapshot per release.** When a future bug
  surfaces, we can recreate the run on the same versions and
  hardware class.

## 4. Reuse across this repository

| Subsystem | Module(s) |
|---|---|
| Failure attribution | `benchmark/rag/component_pipeline.py` |
| Offline runner | `benchmark/rag/offline_online_alignment.py::OfflineRunner` |
| Retrieval metrics | `benchmark/rag/retrieval_metrics.py` |
| Reranker eval | `benchmark/rag/reranker.py` |
| Relevance scoring | `benchmark/rag/answer_relevance.py` |
| Groundedness | `benchmark/rag/groundedness.py` |
| Judge prompts | `benchmark/rag/judge_prompts.py` |
| Judge calibration | `benchmark/rag/judge_calibration.py` |
| Judge attacks | `benchmark/rag/judge_attacks.py` |
| Regression CI | `benchmark/rag/regression_ci.py` |
| Canary | `benchmark/rag/canary.py` |
| Drift | `benchmark/rag/drift_detection.py` |
| Statistical tests | `benchmark/rigor/statistical_tests.py` |
| Effect sizes | `benchmark/rigor/effect_size.py` |
| Multi-seed | `benchmark/rigor/multi_seed_runner.py` |
| Reproducibility | `benchmark/rigor/reproducibility.py` |
| Calibration probs | `benchmark/rigor/calibration.py` |
| Inter-rater | `benchmark/rigor/inter_rater.py` |
| Ablation | `benchmark/rigor/ablation.py` |
| CV | `benchmark/rigor/cross_validation.py` |
| Power | `benchmark/rigor/power_analysis.py` |
| OOD | `benchmark/rigor/ood_assessment.py` |

## 5. What this system explicitly does not do

- It does not train models. The bench is read-only on the model
  side; training pipelines are someone else's concern.
- It does not store user data beyond the trace logs, and trace
  logs hash any user identifier.
- It does not promise full-coverage testing. The gates catch
  known classes of failure; unknown unknowns require manual
  audit and case-study postmortems.
