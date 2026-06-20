import json
from pathlib import Path

from benchmark.conditioned.context_corpus import (
    CHARS_PER_TOKEN, BuiltContext, build_context, load_needles,
)

PARAS = [f"第{i}号案情段落," + "某甲与某乙因合同纠纷诉至法院。" * 6 for i in range(40)]
FACTS = [("n25", 0.25, "虚构卷宗Z-77存放在第41柜。"),
         ("n50", 0.50, "虚构账户尾号9923的余额为615元。"),
         ("n75", 0.75, "虚构站点K-12的巡检周期为7天。")]


def test_build_is_deterministic():
    a = build_context(4096, FACTS, PARAS)
    b = build_context(4096, FACTS, PARAS)
    assert isinstance(a, BuiltContext) and a.text == b.text


def test_token_budget_within_5pct():
    ctx = build_context(8192, [], PARAS)
    assert abs(ctx.est_tokens - 8192) / 8192 <= 0.05
    assert abs(len(ctx.text) / CHARS_PER_TOKEN - 8192) / 8192 <= 0.05


def test_needles_present_at_requested_depths_in_order():
    ctx = build_context(4096, FACTS, PARAS)
    pos = [ctx.text.find(f[2]) for f in FACTS]
    assert all(p >= 0 for p in pos) and pos == sorted(pos)
    for (fid, want, _), realized in zip(FACTS, (ctx.insertions[f[0]] for f in FACTS)):
        assert abs(realized - want) <= 0.15, (fid, want, realized)


def test_load_needles_missing_returns_none(tmp_path):
    assert load_needles(tmp_path / "nope.jsonl") is None


def test_shipped_needles_file_is_valid_and_synthetic():
    path = Path(__file__).resolve().parents[2] / "datasets/conditioned/needles.jsonl"
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(rows) == 8
    assert sum(1 for r in rows if r["role"] == "task") == 5
    assert sum(1 for r in rows if r["role"] == "needle") == 3
    for r in rows:
        assert {"id", "role", "depth", "fact", "question", "answer"} <= set(r)
        assert "虚构" in r["fact"]          # needle 全合成,防训练数据污染(spec §11)
        assert r["answer"] in r["fact"]
