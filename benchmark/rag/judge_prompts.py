"""LLM-as-judge prompt templates and scoring (PDF Chapter 7).

Three principles drive this module:
1. *Evidence-only* scoring: the judge must only consider the supplied
   evidence chunks. Anything the model "knows" from pretraining should
   not influence the verdict.
2. *Variance control*: at least N judge runs per item; aggregate via
   median (default) or majority vote.
3. *Structured output*: the judge must return JSON conforming to a
   strict schema so downstream code does not parse free text.

Beyond the PDF we add:
- A G-Eval-style chain-of-thought prompting template (Liu et al. 2023).
- Few-shot calibration examples plumbed via the JudgeConfig.

This module is *prompt assembly only*; the actual LLM call is left to the
backends module so callers can plug in Ollama / OpenAI / Anthropic / Gemini.

References
----------
- Liu, Y. et al. (2023). G-Eval: NLG Evaluation Using GPT-4 with Better
  Human Alignment. EMNLP.
- Zheng, L. et al. (2023). Judging LLM-as-a-Judge with MT-Bench and
  Chatbot Arena. NeurIPS.
- Es, S. et al. (2023). RAGAs.
"""
from __future__ import annotations

import json
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Prompt assembly
# ---------------------------------------------------------------------------


GROUNDEDNESS_SYSTEM_PROMPT = """You are a strict evidence-only evaluator.

You will be given (1) a user question, (2) a candidate answer with possibly inline
citations, and (3) the retrieved evidence passages.

Your job is to mark each atomic claim in the candidate answer as:
- supported: the claim is directly entailed by the evidence
- unsupported: the claim is not entailed by the evidence (regardless of whether
  it might be true in the world)

Rules:
1. Do NOT use any outside knowledge. If you don't see it in the evidence,
   it's unsupported.
2. Citations attached to a claim must point to evidence that actually supports
   the claim. If a cite is wrong, mark the cite as wrong even if some other
   piece of evidence supports the claim.
3. Return STRICT JSON exactly as the requested schema. No prose, no markdown.

Output schema:
{
  "claims": [
    {
      "claim_id": "c0",
      "text": "...",
      "supported": true,
      "supporting_evidence_ids": ["e_3"],
      "incorrect_citations": []
    },
    ...
  ]
}"""


ANSWER_RELEVANCE_SYSTEM_PROMPT = """You are a strict relevance evaluator.

You will be given a user question and a candidate answer. Score the answer
on three axes (each in [0, 1]):

- intent_satisfied: does the answer address what was asked?
- claim_coverage: what fraction of the question's sub-asks are addressed?
- on_topic: does the answer stay focused without irrelevant tangents?

Return STRICT JSON:
{
  "intent_satisfied": 0.0,
  "claim_coverage": 0.0,
  "on_topic": 0.0,
  "rationale": "one sentence"
}"""


CHOICE_PROMPT_TEMPLATE = """You will be shown a question and two candidate answers (A and B).
Pick the better answer.

Question: {question}

Answer A: {answer_a}
Answer B: {answer_b}

Evidence:
{evidence}

Return STRICT JSON:
{{
  "winner": "A" | "B" | "tie",
  "rationale": "one sentence",
  "confidence": 0.0
}}"""


GEVAL_COT_PREFIX = """Before producing the JSON output, internally reason step-by-step:
1. Identify the atomic claims in the answer.
2. For each claim, search the evidence for entailment.
3. Note which citations are correctly mapped.
Do not output the reasoning; only emit the final JSON."""


# ---------------------------------------------------------------------------
# Config + prompt builders
# ---------------------------------------------------------------------------


@dataclass
class JudgeConfig:
    """How the judge should be invoked.

    `n_runs` enforces variance control: each item is judged this many times
    and the dispatch helper aggregates.
    `temperature` should be 0.0 for deterministic-leaning judging; raise only
    if you intentionally want a distribution over judgments.
    `few_shot_examples` are appended to the system prompt to anchor the judge
    in known-good labels (see judge_calibration.py).
    """

    n_runs: int = 3
    temperature: float = 0.0
    use_cot: bool = True
    few_shot_examples: List[Dict[str, Any]] = field(default_factory=list)
    aggregator: str = "median"  # "median" | "majority" | "mean"


def build_groundedness_prompt(
    question: str,
    answer: str,
    evidence: Sequence[Dict[str, str]],
    config: JudgeConfig,
) -> List[Dict[str, str]]:
    """Build chat messages for the groundedness judge.

    `evidence` is a list of {id, text} chunks.
    Returns OpenAI-style messages: [{role, content}, ...].
    """
    system_parts = [GROUNDEDNESS_SYSTEM_PROMPT]
    if config.use_cot:
        system_parts.append(GEVAL_COT_PREFIX)
    for ex in config.few_shot_examples:
        system_parts.append("Example:\n" + json.dumps(ex, ensure_ascii=False, indent=2))

    user_block = [
        f"Question: {question}",
        f"Candidate answer: {answer}",
        "Evidence:",
    ]
    for ev in evidence:
        user_block.append(f"[{ev['id']}] {ev['text']}")
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": "\n\n".join(user_block)},
    ]


def build_relevance_prompt(
    question: str,
    answer: str,
    config: JudgeConfig,
) -> List[Dict[str, str]]:
    system_parts = [ANSWER_RELEVANCE_SYSTEM_PROMPT]
    if config.use_cot:
        system_parts.append(GEVAL_COT_PREFIX)
    return [
        {"role": "system", "content": "\n\n".join(system_parts)},
        {"role": "user", "content": f"Question: {question}\n\nAnswer: {answer}"},
    ]


def build_pairwise_prompt(
    question: str,
    answer_a: str,
    answer_b: str,
    evidence: Sequence[Dict[str, str]],
) -> List[Dict[str, str]]:
    evidence_str = "\n".join(f"[{e['id']}] {e['text']}" for e in evidence)
    return [
        {
            "role": "user",
            "content": CHOICE_PROMPT_TEMPLATE.format(
                question=question,
                answer_a=answer_a,
                answer_b=answer_b,
                evidence=evidence_str,
            ),
        },
    ]


# ---------------------------------------------------------------------------
# Aggregation across N runs (variance control)
# ---------------------------------------------------------------------------


def aggregate_runs(
    run_outputs: Sequence[Dict[str, Any]],
    field_name: str,
    method: str = "median",
) -> float:
    """Aggregate `field_name` across N judge runs.

    `method`:
      - "median": robust to one outlier judge
      - "mean": linear; appropriate when judges produce continuous scores
      - "majority": for binary fields (returns 1.0 if majority True else 0.0)
    """
    values = [r.get(field_name) for r in run_outputs if field_name in r]
    if not values:
        return float("nan")
    if method == "median":
        return float(statistics.median(values))
    if method == "mean":
        return float(statistics.fmean(v for v in values if v is not None))
    if method == "majority":
        truthy = sum(1 for v in values if v)
        return 1.0 if truthy > len(values) / 2 else 0.0
    raise ValueError(f"unknown aggregator: {method}")


def run_judge(
    invoke_fn: Callable[[List[Dict[str, str]], float], str],
    messages: List[Dict[str, str]],
    config: JudgeConfig,
) -> List[Dict[str, Any]]:
    """Invoke a chat completion `n_runs` times and parse JSON.

    `invoke_fn(messages, temperature) -> str` is whatever the caller's
    backend exposes. We re-run on parse failures with the same prompt;
    after 3 failures we surface an error record so calibration can see it.
    """
    outputs: List[Dict[str, Any]] = []
    for _ in range(config.n_runs):
        attempt = 0
        parsed: Optional[Dict[str, Any]] = None
        last_err = ""
        while attempt < 3 and parsed is None:
            try:
                raw = invoke_fn(messages, config.temperature)
                parsed = _strip_to_json(raw)
            except Exception as e:  # noqa: BLE001
                last_err = repr(e)
                parsed = None
            attempt += 1
        if parsed is None:
            outputs.append({"_parse_error": last_err})
        else:
            outputs.append(parsed)
    return outputs


def _strip_to_json(raw: str) -> Dict[str, Any]:
    """Best-effort JSON extraction.

    Handles judges that wrap output in ```json ... ``` despite the prompt.
    We do NOT use Python's `eval` here. We locate the first '{' and last
    '}' and json.loads the substring.
    """
    s = raw.strip()
    if not s:
        raise ValueError("empty judge output")
    # Strip code fences.
    if s.startswith("```"):
        first_nl = s.find("\n")
        if first_nl > 0:
            s = s[first_nl + 1 :]
        if s.endswith("```"):
            s = s[:-3]
        s = s.strip()
    start = s.find("{")
    end = s.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object found in judge output")
    return json.loads(s[start : end + 1])
