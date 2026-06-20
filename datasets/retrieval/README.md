# Retrieval datasets (embedding + rerank)

The embedding and rerank dimensions share a single retrieval-set format so both
are scored against the same gold relevance.

## Files

- `cmteb_zh_subset.jsonl` — **you provide this**. A Chinese retrieval set
  exported to JSONL (e.g. a [C-MTEB](https://github.com/FlagOpen/FlagEmbedding)
  CMedQAv2 / T2Retrieval subset). It is **not shipped** — retrieval corpora can
  carry licensing constraints. When this file is absent the harness falls back
  to the built-in synthetic Chinese set in
  `benchmark/embedding/datasets.py::load_builtin_retrieval` (flagged
  `source="builtin"`, ~6 hand-authored queries — for offline smoke / unit tests
  only, **not** a real benchmark score).

## JSONL schema

One object per line:

```json
{"query": "如何重置密码？", "candidates": ["在登录页点击忘记密码…", "本店周末营业…", "在设置页修改密码…"], "relevant": [0, 2], "qid": "q1", "domain": "support"}
```

| field | type | required | meaning |
|---|---|---|---|
| `query` | string | yes | the search query |
| `candidates` | string[] | yes | candidate documents to rank |
| `relevant` | int[] | yes | indices into `candidates` that are relevant (gold) |
| `qid` | string | no | query id (defaults to line number) |
| `domain` | string | no | `support` / `tech` / `legal` / … |
| `meta` | object | no | free-form provenance |

Rows with empty `candidates` or empty `relevant` are skipped (they have no
scorable gold and would silently distort recall/MRR).

## Provenance honesty

- A shipped JSONL is tagged `source="custom"`.
- The built-in offline fallback is tagged `source="builtin"` and is **synthetic
  / hand-authored** — it exists so unit tests and air-gapped smoke runs work, and
  never masquerades as a real benchmark corpus. Every report prints
  `data_source` so a builtin-fallback number is never mistaken for a real one.
