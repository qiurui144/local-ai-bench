# Translation datasets

Parallel corpora for the translation benchmark dimension
(`benchmark/translation/`).

## Files

| File | Source | Pairs | Notes |
|---|---|---|---|
| `custom_zh_en.jsonl` | **synthetic / hand-authored** | ~60 | Product / engineering domain (AI infra, RAG, support). Each line carries an optional `glossary` for L3 terminology scoring. No real PII. |

Flores-200 (zh↔en) is **not vendored** — it is pulled at runtime from the
HuggingFace `facebook/flores` dataset (`devtest` split) by
`benchmark.translation.datasets.load_flores`. When the dataset or network is
unavailable, that loader falls back to a tiny built-in synthetic set so
offline / CI runs still work (provenance is flagged `source="builtin"`).

## JSONL schema

One JSON object per line:

```json
{"src": "向量化是 RAG 流程的第一步。", "tgt": "Vectorization is the first step of the RAG pipeline.", "domain": "ai_infra", "glossary": {"向量化": "vectorization", "RAG": "RAG"}}
```

- `src` / `tgt` — required parallel sentences.
- `src_lang` / `tgt_lang` — optional (default `zh` / `en`).
- `domain` — optional free-form tag for per-domain breakouts.
- `glossary` — optional `{src_term: tgt_term}`; required target strings are
  scored by exact-match rate at task level **L3**.

## Provenance honesty

`custom_zh_en.jsonl` is **synthetic** (authored for this benchmark, not
sampled from a production log) to avoid any customer PII. Replace it with your
own reviewed parallel corpus to evaluate against your real domain — keep the
same JSONL schema and the loader works unchanged.
