"""general_ability runner:全 stub 离线测试(数据集/后端均不联网)。

per-dataset stub 的 item 形状对齐各真实 Dataset.load() 输出;BLOCKED 语义
(load 失败 / synthetic_fallback)绝不假 PASS(spec §7)。
"""
import pytest

import benchmark.general_ability.runner as ga
from benchmark.general_ability.runner import parse_choice_letter, run_general_ability


class _FakeBackend:
    def __init__(self, reply="#### 18", lp=None, fail=False):
        self.reply, self.lp, self.fail = reply, lp, fail

    def generate(self, prompt, max_tokens=512, temperature=0.0, **kw):
        if self.fail:
            raise RuntimeError("HTTP 500")
        return self.reply

    def generate_with_logprobs(self, prompt, candidates, max_tokens=1):
        if self.fail:
            raise RuntimeError("HTTP 500")
        return self.lp or {c: float("-inf") for c in candidates}


class _Cfg:
    name = "m1"
    hf_repo = "org/m1"
    port = 8123
    base_url = "http://localhost:8123/v1"


def _patch_datasets(monkeypatch, synthetic=False, raise_load=False):
    # per-dataset stub:item 形状对齐各真实 Dataset.load() 输出
    # (gsm8k: question/answer "#### N";mmlu: question/choices/answer 字母/subject;
    #  hellaswag: activity_label/ctx/endings/label 0-3 整数)
    def _make(item):
        class _DS:
            synthetic_fallback = synthetic

            def __init__(self, **kw):
                if raise_load:
                    raise RuntimeError("hub unreachable")

            def load(self):
                return [dict(item)] * 4

        return _DS

    monkeypatch.setattr(ga, "GSM8KDataset", _make(
        {"question": "1+1?", "answer": "x\n#### 18"}))
    monkeypatch.setattr(ga, "MMLUDataset", _make(
        {"question": "1+1?", "choices": ["a", "b", "c", "d"],
         "answer": "B", "subject": "s"}))
    monkeypatch.setattr(ga, "HellaSwagDataset", _make(
        {"activity_label": "A", "ctx": "c",
         "endings": ["e1", "e2", "e3", "e4"], "label": 1}))


def test_happy_path_all_tasks_pass(monkeypatch):
    _patch_datasets(monkeypatch)
    monkeypatch.setattr(ga, "make_backend", lambda m: _FakeBackend(
        reply="#### 18", lp={"A": -3.0, "B": -0.1, "C": -5.0, "D": -6.0}))
    out = run_general_ability(_Cfg(), {"thresholds": {
        "gsm8k_min": 0.5, "mmlu_min": 0.5, "hellaswag_min": 0.5}})
    assert out["verdict"] == "PASS"
    assert out["tasks"]["gsm8k"]["accuracy"] == 1.0      # 手算:4/4 全对
    assert out["tasks"]["mmlu"]["verdict"] == "PASS"     # 全选 B == answer "B"
    assert out["tasks"]["hellaswag"]["verdict"] == "PASS"  # B == label 1


@pytest.mark.parametrize("kw", [{"raise_load": True}, {"synthetic": True}])
def test_unusable_dataset_blocks_never_passes(monkeypatch, kw):
    """数据加载失败 / synthetic_fallback → BLOCKED,绝不假 PASS(spec §7)。"""
    _patch_datasets(monkeypatch, **kw)
    monkeypatch.setattr(ga, "make_backend", lambda m: _FakeBackend())
    out = run_general_ability(_Cfg(), {})
    assert out["verdict"] == "BLOCKED"
    assert all(t["verdict"] == "BLOCKED" for t in out["tasks"].values())


def test_endpoint_all_errors_blocked(monkeypatch):
    _patch_datasets(monkeypatch)
    monkeypatch.setattr(ga, "make_backend", lambda m: _FakeBackend(fail=True))
    out = run_general_ability(_Cfg(), {})
    assert out["tasks"]["gsm8k"]["verdict"] == "BLOCKED"   # error_rate 100%


def test_below_threshold_fails(monkeypatch):
    _patch_datasets(monkeypatch)
    monkeypatch.setattr(ga, "make_backend", lambda m: _FakeBackend(
        reply="#### 999", lp={"A": -0.1, "B": -3.0, "C": -5.0, "D": -6.0}))
    out = run_general_ability(_Cfg(), {})
    assert out["verdict"] == "FAIL"


def test_parse_choice_letter_adversarial():
    assert parse_choice_letter("The answer is (A). just kidding it is (B)") == "A"
    assert parse_choice_letter("Answer: C") == "C"
    assert parse_choice_letter("(D) because ...") == "D"
    assert parse_choice_letter("no letters here 42") is None
    assert parse_choice_letter("") is None


def test_backend_init_failure_returns_blocked(monkeypatch):
    """If make_backend raises, run_general_ability must return BLOCKED (not crash)."""
    _patch_datasets(monkeypatch)
    monkeypatch.setattr(ga, "make_backend", lambda m: (_ for _ in ()).throw(
        RuntimeError("backend unavailable: model not found")))
    out = run_general_ability(_Cfg(), {})
    assert out["verdict"] == "BLOCKED"
    assert "backend unavailable" in out.get("reason", "")
