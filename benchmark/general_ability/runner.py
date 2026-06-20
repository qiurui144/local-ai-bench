"""general_ability 维度:gsm8k / mmlu / hellaswag,与 accuracy/scenarios 同级判定。

复用 llama_benchmark 的数据集(pin SHA)与评分原语(extract_answer /
build_prompt / predict_choice);runner 本体按主 harness 语义:数据缺失或
synthetic_fallback → BLOCKED(空跑不得 PASS),error_rate>20% → FAIL。
HellaSwag 以 A-D 选择题正确率评分(chat API 上的确定性近似,非
length-normalized logprob — 方法记入 block["method"],per §6.3 claim 纪律)。
"""
from __future__ import annotations

import logging
import re

from benchmark.general_ability.backend_adapter import make_backend
from benchmark.llama_benchmark.benchmarks.llm.gsm8k import GSM8K_FEW_SHOT, extract_answer
from benchmark.llama_benchmark.benchmarks.llm.mmlu import build_prompt, predict_choice
from benchmark.llama_benchmark.datasets.gsm8k_dataset import GSM8KDataset
from benchmark.llama_benchmark.datasets.hellaswag_dataset import HellaSwagDataset
from benchmark.llama_benchmark.datasets.mmlu_dataset import MMLUDataset
from benchmark.registry import worst_verdict

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLDS = {"gsm8k_min": 0.55, "mmlu_min": 0.55, "hellaswag_min": 0.60}
DEFAULT_MMLU_SUBJECTS = ["professional_law", "logical_fallacies",
                         "computer_security", "elementary_mathematics"]

_LETTER_RE = re.compile(
    r"answer\s*(?:is)?\s*[:：]?\s*\(?([ABCD])\)?|^\s*\(?([ABCD])\)?(?:[).\s]|$)",
    re.IGNORECASE | re.MULTILINE,
)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def parse_choice_letter(text: str):
    """取输出中第一个权威 A-D(adversarial 尾随 'just kidding (B)' 不生效)。
    Qwen3/Qwen3-VL thinking-mode 先剥离 <think>...</think>;再剥离 markdown bold
    (**A.**) — rkllm3 server 上的 Qwen3-VL-2B 以加粗格式返回选项。"""
    stripped = _THINK_RE.sub("", text or "").strip()
    stripped = stripped.replace("**", "")   # **A.** → A.
    m = _LETTER_RE.search(stripped)
    return (m.group(1) or m.group(2)).upper() if m else None


def _blocked(reason: str) -> dict:
    return {"verdict": "BLOCKED", "reason": reason, "accuracy": 0.0, "n": 0, "errors": 0}


def _choose(backend, prompt: str):
    """优先 logprob 选择;后端不回 logprobs(全 -inf)时降级 generate+解析。"""
    try:
        predicted, lp = predict_choice(backend, prompt)
        if any(v != float("-inf") for v in lp.values()):
            return predicted
    except Exception:
        pass
    return parse_choice_letter(backend.generate(prompt, max_tokens=16, temperature=0.0))


def _eval(samples, answer_one, floor: float, extra: dict | None = None) -> dict:
    """逐题评测:answer_one(item)->bool;单题异常计 error 不崩跑。"""
    correct = errors = 0
    for item in samples:
        try:
            ok = answer_one(item)
        except Exception:
            errors += 1
            continue
        if ok:
            correct += 1
    n = len(samples)
    if n and errors == n:
        return _blocked("all requests errored — endpoint unusable for chat")
    block = {"accuracy": round(correct / n, 4) if n else 0.0, "n": n, "errors": errors,
             "error_rate": round(errors / n, 4) if n else 0.0,
             "verdict": "PASS", "verdict_reasons": [], **(extra or {})}
    if block["error_rate"] > 0.2:
        block["verdict"] = "FAIL"
        block["verdict_reasons"].append(f"error_rate {block['error_rate']:.0%} > 20%")
    if block["accuracy"] < floor:
        block["verdict"] = "FAIL"
        block["verdict_reasons"].append(f"accuracy {block['accuracy']:.3f} < {floor}")
    return block


def _load(ds_cls, **kw):
    """加载数据集;失败或 synthetic_fallback → (None, blocked_block)。"""
    try:
        ds = ds_cls(**kw)
        samples = ds.load()
    except Exception as e:
        return None, _blocked(f"dataset load failed: {e}")
    if ds.synthetic_fallback:
        return None, _blocked("synthetic fallback rejected — not a general-ability score")
    return samples, None


def _run_gsm8k(backend, task_cfg: dict, floor: float) -> dict:
    samples, blocked = _load(GSM8KDataset, split=task_cfg.get("split", "test"),
                             num_samples=task_cfg.get("num_samples", 100))
    if blocked:
        return blocked

    def answer_one(item) -> bool:
        ref = extract_answer(item["answer"])
        text = backend.generate(GSM8K_FEW_SHOT + f"Q: {item['question']}\nA:",
                                max_tokens=512, temperature=0.0)
        pred = extract_answer(text)
        return pred is not None and ref is not None and pred == ref

    return _eval(samples, answer_one, floor)


def _run_mmlu(backend, task_cfg: dict, floor: float) -> dict:
    subjects = task_cfg.get("subjects", DEFAULT_MMLU_SUBJECTS)
    samples: list = []
    for s in subjects:
        part, blocked = _load(MMLUDataset, subject=s,
                              num_samples=task_cfg.get("per_subject", 25))
        if blocked:
            return blocked
        samples += part
    return _eval(samples,
                 lambda it: _choose(backend, build_prompt(it["question"], it["choices"])) == it["answer"],
                 floor, {"subjects": len(subjects)})


def _run_hellaswag(backend, task_cfg: dict, floor: float) -> dict:
    samples, blocked = _load(HellaSwagDataset, split=task_cfg.get("split", "validation"),
                             num_samples=task_cfg.get("num_samples", 100))
    if blocked:
        return blocked

    def answer_one(item) -> bool:
        q = f"Choose the most plausible ending.\n{item['activity_label']}: {item['ctx']}"
        pred = _choose(backend, build_prompt(q, item["endings"]))
        return pred is not None and "ABCD".index(pred) == int(item["label"])

    return _eval(samples, answer_one, floor, {"method": "choice_letter"})


GENERAL_TASKS = {"gsm8k": _run_gsm8k, "mmlu": _run_mmlu, "hellaswag": _run_hellaswag}


def run_general_ability(model_cfg, cfg: dict) -> dict:
    thresholds = {**DEFAULT_THRESHOLDS, **(cfg.get("thresholds") or {})}
    tasks_cfg = cfg.get("tasks") or {}
    try:
        backend = make_backend(model_cfg)
    except Exception as exc:
        return {"verdict": "BLOCKED", "reason": f"backend unavailable: {exc}", "metrics": {}}
    out = {"benchmark": "general_ability", "model": model_cfg.name,
           "tasks": {}, "verdict": "PASS", "verdict_reasons": []}
    for name, fn in GENERAL_TASKS.items():
        block = fn(backend, tasks_cfg.get(name) or {}, thresholds.get(f"{name}_min", 0.0))
        out["tasks"][name] = block
        if block["verdict"] == "BLOCKED":
            out["verdict_reasons"].append(f"[{name}] BLOCKED: {block.get('reason')}")
        for r in block.get("verdict_reasons", []):
            out["verdict_reasons"].append(f"[{name}] {r}")
    verdicts = [b["verdict"] for b in out["tasks"].values()]
    out["verdict"] = ("BLOCKED" if verdicts and all(v == "BLOCKED" for v in verdicts)
                      else worst_verdict(verdicts))
    return out
