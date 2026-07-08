"""Public-suite long-context adapters.

This module intentionally implements an edge subset, not the official
leaderboard pipelines. It preserves source identity and scoring rules while
keeping dependencies small enough for RISC-V targets and OpenAI-compatible
llama.cpp servers.
"""
from __future__ import annotations

import json
import math
import re
import statistics
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from common import ModelConfig, infer_sync


CHARS_PER_TOKEN_EN = 4.0
CHARS_PER_TOKEN_ZH = 1.6
DEFAULT_MAX_INPUT_TOKENS = 3072


@dataclass(frozen=True)
class TextFit:
    text: str
    original_est_tokens: int
    final_est_tokens: int
    truncated: bool


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    zh_chars = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    ratio = CHARS_PER_TOKEN_ZH if zh_chars > len(text) * 0.15 else CHARS_PER_TOKEN_EN
    return max(1, math.ceil(len(text) / ratio))


def _fit_text_middle(text: str, max_tokens: int) -> TextFit:
    original = _estimate_tokens(text)
    if original <= max_tokens:
        return TextFit(text=text, original_est_tokens=original, final_est_tokens=original, truncated=False)
    chars_per_token = len(text) / max(original, 1)
    budget_chars = max(256, int(max_tokens * chars_per_token))
    head = max(64, budget_chars // 2)
    tail = max(64, budget_chars - head)
    fitted = text[:head] + "\n\n[... middle truncated for K3 edge context budget ...]\n\n" + text[-tail:]
    return TextFit(
        text=fitted,
        original_est_tokens=original,
        final_est_tokens=_estimate_tokens(fitted),
        truncated=True,
    )


def _fit_text_window(text: str, center_offset: int, max_tokens: int) -> tuple[TextFit, dict]:
    original = _estimate_tokens(text)
    if original <= max_tokens:
        return (
            TextFit(text=text, original_est_tokens=original, final_est_tokens=original, truncated=False),
            {"window_start": 0, "window_end": len(text)},
        )
    chars_per_token = len(text) / max(original, 1)
    budget_chars = max(512, int(max_tokens * chars_per_token))
    center = min(max(0, int(center_offset)), len(text))
    start = max(0, min(center - budget_chars // 2, max(0, len(text) - budget_chars)))
    end = min(len(text), start + budget_chars)
    fitted = text[start:end]
    if start > 0:
        fitted = "[... preceding manual text omitted for K3 context budget ...]\n\n" + fitted
    if end < len(text):
        fitted = fitted + "\n\n[... following manual text omitted for K3 context budget ...]"
    return (
        TextFit(
            text=fitted,
            original_est_tokens=original,
            final_est_tokens=_estimate_tokens(fitted),
            truncated=True,
        ),
        {"window_start": start, "window_end": end},
    )


def _fit_context_prompt(prefix: str, context: str, suffix: str, max_input_tokens: int) -> tuple[str, dict]:
    fixed_tokens = _estimate_tokens(prefix + suffix)
    budget = max(256, max_input_tokens - fixed_tokens)
    fit = _fit_text_middle(context, budget)
    prompt = prefix + fit.text + suffix
    return prompt, {
        "original_context_est_tokens": fit.original_est_tokens,
        "final_context_est_tokens": fit.final_est_tokens,
        "truncated": fit.truncated,
        "max_input_tokens": max_input_tokens,
    }


def _safe_prompt_budget(max_input_tokens: int, safety: float) -> int:
    return max(256, int(max_input_tokens * max(0.25, min(1.0, safety))))


def _fit_window_prompt(
    prefix: str,
    text: str,
    center_offset: int,
    suffix: str,
    *,
    max_input_tokens: int,
    target_context_tokens: int,
    safety: float,
) -> tuple[str, TextFit, dict]:
    fixed_tokens = _estimate_tokens(prefix + suffix)
    budget = min(target_context_tokens, max(256, _safe_prompt_budget(max_input_tokens, safety) - fixed_tokens))
    fit, window = _fit_text_window(text, center_offset, budget)
    return prefix + fit.text + suffix, fit, window


def _read_jsonl(path: Path, limit: int | None = None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip()).lower()


def _norm_compact(text: str) -> str:
    text = text or ""
    text = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    text = re.sub(r"<think>[\s\S]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r"[\s`'\".,;:!?()\[\]{}<>，。！？；：（）【】]+", "", text)
    return text.lower()


def _extract_option(text: str) -> str:
    raw = (text or "").strip()
    m = re.search(r"(?i)(?:^|[^A-Z])\(?([ABCD])\)?(?:[^A-Z]|$)", raw)
    return m.group(1).upper() if m else ""


def _extract_expected_option(answer: str) -> str:
    return _extract_option(answer) or (answer or "").strip()[:1].upper()


def _qa_f1(prediction: str, ground_truth: str) -> float:
    pred_tokens = _norm_compact(prediction).split()
    gt_tokens = _norm_compact(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = 0
    gt_counts: dict[str, int] = {}
    for token in gt_tokens:
        gt_counts[token] = gt_counts.get(token, 0) + 1
    for token in pred_tokens:
        if gt_counts.get(token, 0) > 0:
            common += 1
            gt_counts[token] -= 1
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens)
    recall = common / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def _score_longbench(dataset: str, prediction: str, answers: list[str], all_classes=None) -> float:
    pred = prediction or ""
    if dataset == "passage_count":
        numbers = re.findall(r"\d+", pred)
        if not numbers:
            return 0.0
        return max(1.0 if str(ans) in numbers else 0.0 for ans in answers)
    if dataset.startswith("passage_retrieval"):
        numbers = re.findall(r"\d+", pred)
        best = 0.0
        for ans in answers:
            m = re.search(r"(?:Paragraph|段落)\s*(\d+)", str(ans))
            if m and m.group(1) in numbers:
                best = 1.0
        return best
    if all_classes:
        compact = _norm_compact(pred)
        matches = [cls for cls in all_classes if _norm_compact(str(cls)) in compact]
        return max((1.0 / len(matches)) if ans in matches else 0.0 for ans in answers) if matches else 0.0
    return max(_qa_f1(pred, str(ans)) for ans in answers)


def _technical_terms(text: str) -> list[str]:
    seen = set()
    terms = []
    for item in re.findall(r"\b[A-Za-z][A-Za-z0-9/-]{2,}\b", text or ""):
        key = item.upper()
        if key not in seen:
            seen.add(key)
            terms.append(item)
    return terms


def _span_recall_score(prediction: str, expected: str) -> float:
    pred = _norm_compact(prediction)
    exp = _norm_compact(expected)
    if not pred or not exp:
        return 0.0
    if exp in pred:
        return 1.0
    terms = _technical_terms(expected)
    if not terms:
        return 0.0
    hits = sum(1 for term in terms if _norm_compact(term) in pred)
    return min(1.0, hits / min(6, len(terms)))


def _keyword_recall_score(prediction: str, keywords: list[str]) -> float:
    if not keywords:
        return 0.0
    compact = _norm_compact(prediction)
    hits = sum(1 for term in keywords if _norm_compact(str(term)) in compact)
    return hits / len(keywords)


def _emit_progress(cfg: dict, event: dict) -> None:
    if not cfg.get("progress_log"):
        return
    print(json.dumps({"event": "long_context_case", **event}, ensure_ascii=False), file=sys.stderr, flush=True)


def _append_case_result(cfg: dict, suite: str, case: dict) -> None:
    path_raw = cfg.get("case_result_log")
    if not path_raw:
        return
    path = Path(path_raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps({"suite": suite, **case}, ensure_ascii=False) + "\n")


def _chat(
    model_cfg: ModelConfig,
    prompt: str,
    *,
    max_tokens: int,
    timeout_s: float,
) -> tuple[dict, str]:
    t0 = time.perf_counter()
    try:
        result = infer_sync(
            model_cfg,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=0.0,
            timeout_s=timeout_s,
        )
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        return {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(elapsed, 3),
            "latency_ms": round(elapsed * 1000, 1),
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "finish_reason": "",
        }, ""
    elapsed = time.perf_counter() - t0
    return {
        "ok": result.ok,
        "error": result.error,
        "elapsed_s": round(elapsed, 3),
        "latency_ms": round(result.latency_ms, 1),
        "prompt_tokens": result.input_tokens,
        "completion_tokens": result.output_tokens,
        "finish_reason": result.finish_reason,
    }, result.content


def _load_nihs_haystack(root: Path) -> tuple[str, str]:
    suite_dir = root / "drivers/long-context-suites/needle-in-a-haystack/needlehaystack/PaulGrahamEssays"
    if not suite_dir.exists():
        return "", "missing needle-in-a-haystack PaulGrahamEssays"
    chunks = []
    for path in sorted(suite_dir.glob("*.txt")):
        chunks.append(path.read_text(encoding="utf-8", errors="replace"))
    return "\n\n".join(chunks), ""


def _build_nihs_context(haystack: str, context_tokens: int, depth_percent: int, code: str) -> tuple[str, dict]:
    fit = _fit_text_middle(haystack, context_tokens)
    context = fit.text
    idx = int(len(context) * (depth_percent / 100.0))
    needle = f"\n\nThe secret K3 validation code is {code}.\n\n"
    rendered = context[:idx] + needle + context[idx:]
    return rendered, {
        "context_tokens": context_tokens,
        "depth_percent": depth_percent,
        "code": code,
        "haystack_original_est_tokens": fit.original_est_tokens,
        "haystack_final_est_tokens": fit.final_est_tokens,
        "haystack_truncated": fit.truncated,
    }


def _run_nihs(model_cfg: ModelConfig, cfg: dict, root: Path) -> dict:
    haystack, err = _load_nihs_haystack(root)
    if err:
        return {"suite": "needle-in-a-haystack", "verdict": "BLOCKED", "reason": err, "cases": []}
    lengths = [int(v) for v in cfg.get("context_lengths", [1024, 3072])]
    depths = [int(v) for v in cfg.get("depth_percents", [10, 50, 90])]
    timeout_s = float(cfg.get("timeout_s", 900))
    max_tokens = int(cfg.get("max_tokens", 48))
    cases = []
    for context_tokens in lengths:
        for depth in depths:
            code = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"k3-nihs-{context_tokens}-{depth}"))
            context, meta = _build_nihs_context(haystack, context_tokens, depth, code)
            prompt = (
                "You are given a long context. Find the secret K3 validation code in the context.\n\n"
                f"{context}\n\nQuestion: What is the secret K3 validation code? "
                "Reply with only the exact code."
            )
            call, response = _chat(model_cfg, prompt, max_tokens=max_tokens, timeout_s=timeout_s)
            score = 1.0 if code.lower() in response.lower() else 0.0
            case = {**meta, **call, "response": response, "score": score}
            cases.append(case)
            _append_case_result(cfg, "needle_in_a_haystack", case)
            _emit_progress(cfg, {
                "suite": "needle_in_a_haystack",
                "context_tokens": context_tokens,
                "depth_percent": depth,
                "ok": call["ok"],
                "score": score,
                "elapsed_s": call["elapsed_s"],
            })
    measured = [c for c in cases if c.get("ok")]
    recall = statistics.mean(c["score"] for c in measured) if measured else 0.0
    return {
        "suite": "needle-in-a-haystack",
        "source": "https://github.com/gkamradt/needle-in-a-haystack",
        "mode": "edge_subset_uuid",
        "verdict": "PASS" if measured else "BLOCKED",
        "recall": round(recall, 3),
        "measured": len(measured),
        "errors": len(cases) - len(measured),
        "cases": cases,
    }


def _longbench_data_dir(root: Path) -> Path:
    return root / "drivers/long-context-suites/LongBench/data"


def _run_longbench(model_cfg: ModelConfig, cfg: dict, root: Path) -> dict:
    data_dir = _longbench_data_dir(root)
    prompt_file = root / "drivers/long-context-suites/LongBench/LongBench/config/dataset2prompt.json"
    if not data_dir.exists():
        return {
            "suite": "LongBench",
            "verdict": "BLOCKED",
            "reason": "LongBench data dir missing; run scripts/cache_long_context_suites.py",
            "cases": [],
        }
    dataset2prompt = json.loads(prompt_file.read_text(encoding="utf-8"))
    datasets = cfg.get("datasets", ["passage_retrieval_en", "passage_count"])
    samples_per_dataset = int(cfg.get("samples_per_dataset", 1))
    max_input_tokens = int(cfg.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS))
    timeout_s = float(cfg.get("timeout_s", 900))
    max_tokens = int(cfg.get("max_tokens", 64))
    cases = []
    for dataset in datasets:
        path = data_dir / f"{dataset}.jsonl"
        if not path.exists():
            cases.append({"dataset": dataset, "ok": False, "error": f"missing {path}"})
            continue
        prompt_format = dataset2prompt.get(dataset, "{context}\n\n{input}")
        for row in _read_jsonl(path, samples_per_dataset):
            context = str(row.get("context") or "")
            fit_context = _fit_text_middle(context, max_input_tokens)
            prompt_row = dict(row)
            prompt_row["context"] = fit_context.text
            prompt = prompt_format.format(**prompt_row)
            prompt_fit = _fit_text_middle(prompt, max_input_tokens)
            call, response = _chat(model_cfg, prompt_fit.text, max_tokens=max_tokens, timeout_s=timeout_s)
            answers = [str(a) for a in (row.get("answers") or [])]
            score = _score_longbench(dataset, response, answers, row.get("all_classes")) if call["ok"] else 0.0
            case = {
                "dataset": dataset,
                "id": row.get("_id"),
                "ok": call["ok"],
                "error": call["error"],
                "score": round(score, 4),
                "response": response,
                "answers": answers,
                "length": row.get("length"),
                "context_original_est_tokens": fit_context.original_est_tokens,
                "context_final_est_tokens": fit_context.final_est_tokens,
                "context_truncated": fit_context.truncated or prompt_fit.truncated,
                **{k: v for k, v in call.items() if k not in {"ok", "error"}},
            }
            cases.append(case)
            _append_case_result(cfg, "longbench", case)
            _emit_progress(cfg, {
                "suite": "longbench",
                "dataset": dataset,
                "id": row.get("_id"),
                "ok": call["ok"],
                "score": round(score, 4),
                "elapsed_s": call["elapsed_s"],
            })
    measured = [c for c in cases if c.get("ok")]
    score = statistics.mean(c["score"] for c in measured) if measured else 0.0
    return {
        "suite": "LongBench",
        "source": "https://github.com/THUDM/LongBench",
        "mode": "edge_subset_v1_auto_metric",
        "verdict": "PASS" if measured else "BLOCKED",
        "score": round(score, 4),
        "measured": len(measured),
        "errors": len(cases) - len(measured),
        "cases": cases,
    }


def _run_leval(model_cfg: ModelConfig, cfg: dict, root: Path) -> dict:
    data_dir = root / "drivers/long-context-suites/leval/LEval-data/Closed-ended-tasks"
    if not data_dir.exists():
        return {
            "suite": "L-Eval",
            "verdict": "BLOCKED",
            "reason": "L-Eval closed-ended data missing",
            "cases": [],
        }
    tasks = cfg.get("tasks", ["quality", "coursera"])
    samples_per_task = int(cfg.get("samples_per_task", 1))
    questions_per_document = int(cfg.get("questions_per_document", 1))
    max_input_tokens = int(cfg.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS))
    timeout_s = float(cfg.get("timeout_s", 900))
    max_tokens = int(cfg.get("max_tokens", 48))
    cases = []
    for task in tasks:
        path = data_dir / f"{task}.jsonl"
        if not path.exists():
            cases.append({"task": task, "ok": False, "error": f"missing {path}"})
            continue
        for doc_index, row in enumerate(_read_jsonl(path, samples_per_task)):
            context = str(row.get("input") or "")
            instructions = list(row.get("instructions") or [])[:questions_per_document]
            outputs = list(row.get("outputs") or [])[:questions_per_document]
            for question_index, (question, answer) in enumerate(zip(instructions, outputs)):
                prefix = "Read the long context and answer the multiple-choice question.\n\nContext:\n"
                suffix = (
                    "\n\nQuestion:\n"
                    f"{question}\n\nReply with only the option letter, such as A, B, C, or D."
                )
                prompt, fit = _fit_context_prompt(prefix, context, suffix, max_input_tokens)
                call, response = _chat(model_cfg, prompt, max_tokens=max_tokens, timeout_s=timeout_s)
                expected = _extract_expected_option(str(answer))
                predicted = _extract_option(response)
                score = 1.0 if predicted and predicted == expected else 0.0
                case = {
                    "task": task,
                    "doc_index": doc_index,
                    "question_index": question_index,
                    "ok": call["ok"],
                    "error": call["error"],
                    "score": score if call["ok"] else 0.0,
                    "predicted": predicted,
                    "expected": expected,
                    "response": response,
                    "context_original_est_tokens": fit["original_context_est_tokens"],
                    "context_final_est_tokens": fit["final_context_est_tokens"],
                    "context_truncated": fit["truncated"],
                    "max_input_tokens": fit["max_input_tokens"],
                    **{k: v for k, v in call.items() if k not in {"ok", "error"}},
                }
                cases.append(case)
                _append_case_result(cfg, "leval", case)
                _emit_progress(cfg, {
                    "suite": "leval",
                    "task": task,
                    "doc_index": doc_index,
                    "question_index": question_index,
                    "ok": call["ok"],
                    "score": score if call["ok"] else 0.0,
                    "elapsed_s": call["elapsed_s"],
                })
    measured = [c for c in cases if c.get("ok")]
    accuracy = statistics.mean(c["score"] for c in measured) if measured else 0.0
    return {
        "suite": "L-Eval",
        "source": "https://github.com/openlmlab/LEval",
        "mode": "edge_subset_closed_ended_exact_match",
        "verdict": "PASS" if measured else "BLOCKED",
        "accuracy": round(accuracy, 4),
        "measured": len(measured),
        "errors": len(cases) - len(measured),
        "cases": cases,
    }


def _aviation_manual_cases_path(root: Path) -> Path:
    return root / "drivers/long-context-suites/airplane-manual-collection/cases/aviation_manual_cases.jsonl"


def _load_manual_text(root: Path, case: dict) -> str:
    suite_root = root / "drivers/long-context-suites/airplane-manual-collection"
    path = suite_root / str(case.get("text_path") or "")
    return path.read_text(encoding="utf-8", errors="replace")


def _run_aviation_manuals(model_cfg: ModelConfig, cfg: dict, root: Path) -> dict:
    cases_path = _aviation_manual_cases_path(root)
    if not cases_path.exists():
        return {
            "suite": "Aviation Manual Long Context",
            "verdict": "BLOCKED",
            "reason": "aviation manual cases missing; run scripts/cache_long_context_suites.py --airplane-manuals",
            "cases": [],
        }
    rows = _read_jsonl(cases_path)
    case_limit = int(cfg.get("case_limit", 12))
    if case_limit > 0:
        rows = rows[:case_limit]
    max_input_tokens = int(cfg.get("max_input_tokens", DEFAULT_MAX_INPUT_TOKENS))
    context_tokens = int(cfg.get("target_context_tokens", max_input_tokens))
    prompt_budget_safety = float(cfg.get("prompt_budget_safety", 0.70))
    timeout_s = float(cfg.get("timeout_s", 900))
    max_tokens = int(cfg.get("max_tokens", 96))
    depth_percents = [int(v) for v in cfg.get("depth_percents", [10, 50, 90])]
    cases = []
    for idx, row in enumerate(rows):
        try:
            text = _load_manual_text(root, row)
        except OSError as exc:
            cases.append({
                "case_id": row.get("case_id"),
                "case_type": row.get("case_type"),
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            continue

        case_type = row.get("case_type")
        source = {
            "case_id": row.get("case_id"),
            "case_type": case_type,
            "manual_id": row.get("manual_id"),
            "source_path": row.get("source_path"),
            "pdf_pages": row.get("pdf_pages"),
            "text_chars": row.get("text_chars"),
        }
        if case_type == "span_recall":
            prefix = (
                "You are checking an aircraft manual excerpt. Use only the context.\n\n"
                "Context:\n"
            )
            suffix = (
                "Question: In the context, what line immediately follows this anchor line?\n"
                f"ANCHOR: {row.get('anchor')}\n\n"
                "Reply with only the following line."
            )
            prompt, fit, window = _fit_window_prompt(
                prefix,
                text,
                int(row.get("offset", 0)),
                "\n\n" + suffix,
                max_input_tokens=max_input_tokens,
                target_context_tokens=context_tokens,
                safety=prompt_budget_safety,
            )
            call, response = _chat(model_cfg, prompt, max_tokens=max_tokens, timeout_s=timeout_s)
            score = _span_recall_score(response, str(row.get("answer") or "")) if call["ok"] else 0.0
            case = {
                **source,
                **call,
                "score": round(score, 4),
                "response": response,
                "answer": row.get("answer"),
                "context_original_est_tokens": fit.original_est_tokens,
                "context_final_est_tokens": fit.final_est_tokens,
                "context_truncated": fit.truncated,
                **window,
            }
        elif case_type == "keyword_recall":
            offset = int(row.get("offset", 0))
            paragraph = str(row.get("paragraph") or "")
            marked = text
            if paragraph:
                pos = text.find(paragraph[:80], max(0, offset - 2000))
                if pos < 0:
                    pos = offset
                end = min(len(text), pos + len(paragraph))
                marked = text[:pos] + "\n<<<TARGET PARAGRAPH>>>\n" + text[pos:end] + "\n<<<END TARGET>>>\n" + text[end:]
                offset = pos
            prefix = (
                "You are checking an aircraft manual excerpt. Use only the marked target paragraph.\n\n"
                "Context:\n"
            )
            suffix = (
                "Question: List exact technical terms or acronyms from the target paragraph. "
                "Return a comma-separated list only."
            )
            prompt, fit, window = _fit_window_prompt(
                prefix,
                marked,
                offset,
                "\n\n" + suffix,
                max_input_tokens=max_input_tokens,
                target_context_tokens=context_tokens,
                safety=prompt_budget_safety,
            )
            call, response = _chat(model_cfg, prompt, max_tokens=max_tokens, timeout_s=timeout_s)
            keywords = [str(v) for v in (row.get("keywords") or [])]
            score = _keyword_recall_score(response, keywords) if call["ok"] else 0.0
            case = {
                **source,
                **call,
                "score": round(score, 4),
                "response": response,
                "keywords": keywords,
                "context_original_est_tokens": fit.original_est_tokens,
                "context_final_est_tokens": fit.final_est_tokens,
                "context_truncated": fit.truncated,
                **window,
            }
        elif case_type == "manual_needle":
            depth = depth_percents[idx % len(depth_percents)]
            code = str(uuid.uuid5(uuid.NAMESPACE_DNS, f"k3-aviation-{row.get('case_id')}-{depth}"))
            fixed = (
                "\n\nK3 aviation validation code: "
                + code
                + "\n\nQuestion: What is the K3 aviation validation code? Reply with only the exact code."
            )
            fit = _fit_text_middle(
                text,
                min(context_tokens, max(256, _safe_prompt_budget(max_input_tokens, prompt_budget_safety) - _estimate_tokens(fixed))),
            )
            insert_at = int(len(fit.text) * (depth / 100.0))
            context = fit.text[:insert_at] + f"\n\nK3 aviation validation code: {code}\n\n" + fit.text[insert_at:]
            prompt = (
                "You are given a long aircraft manual excerpt. Find the K3 aviation validation code.\n\n"
                f"{context}\n\nQuestion: What is the K3 aviation validation code? Reply with only the exact code."
            )
            call, response = _chat(model_cfg, prompt, max_tokens=48, timeout_s=timeout_s)
            score = 1.0 if code.lower() in response.lower() else 0.0
            case = {
                **source,
                **call,
                "score": score,
                "response": response,
                "depth_percent": depth,
                "code": code,
                "context_original_est_tokens": fit.original_est_tokens,
                "context_final_est_tokens": fit.final_est_tokens,
                "context_truncated": fit.truncated,
            }
        else:
            case = {**source, "ok": False, "error": f"unknown aviation manual case_type: {case_type}"}
        cases.append(case)
        _append_case_result(cfg, "aviation_manuals", case)
        _emit_progress(cfg, {
            "suite": "aviation_manuals",
            "case_id": row.get("case_id"),
            "case_type": case_type,
            "ok": case.get("ok", False),
            "score": case.get("score", 0.0),
            "elapsed_s": case.get("elapsed_s"),
        })

    measured = [c for c in cases if c.get("ok")]
    by_type = {}
    for case in measured:
        key = str(case.get("case_type"))
        by_type.setdefault(key, []).append(float(case.get("score", 0.0)))
    type_scores = {key: round(statistics.mean(vals), 4) for key, vals in by_type.items() if vals}
    score = statistics.mean(c["score"] for c in measured) if measured else 0.0
    errors = len(cases) - len(measured)
    verdict = "PASS"
    if not measured:
        verdict = "BLOCKED"
    elif errors:
        verdict = "WARN"
    return {
        "suite": "Aviation Manual Long Context",
        "source": "https://github.com/shiroinekotfs/airplane-manual-collection",
        "mode": "edge_subset_pdf_text_window_and_manual_needle",
        "verdict": verdict,
        "score": round(score, 4),
        "type_scores": type_scores,
        "measured": len(measured),
        "errors": errors,
        "cases": cases,
    }


def _summary_latency(suites: Iterable[dict]) -> dict:
    latencies = []
    for suite in suites:
        for case in suite.get("cases", []):
            if case.get("ok") and case.get("elapsed_s") is not None:
                latencies.append(float(case["elapsed_s"]))
    if not latencies:
        return {}
    latencies_sorted = sorted(latencies)
    p95_index = min(len(latencies_sorted) - 1, math.ceil(len(latencies_sorted) * 0.95) - 1)
    return {
        "mean_s": round(statistics.mean(latencies), 3),
        "p95_s": round(latencies_sorted[p95_index], 3),
        "max_s": round(max(latencies), 3),
        "n": len(latencies),
    }


def run_long_context(model_cfg: ModelConfig, cfg: dict, root: Path) -> dict:
    cfg = dict(cfg or {})
    suites_cfg = cfg.get("suites") or {}
    block = {
        "benchmark": "long_context",
        "model": model_cfg.name,
        "mode": "edge_subset_not_leaderboard_full",
        "required_for": "20B+ chat models",
        "sources": {
            "needle_in_a_haystack": "https://github.com/gkamradt/needle-in-a-haystack",
            "longbench": "https://github.com/THUDM/LongBench",
            "leval": "https://github.com/openlmlab/LEval",
            "aviation_manuals": "https://github.com/shiroinekotfs/airplane-manual-collection",
        },
        "suites": {},
        "verdict": "PASS",
        "verdict_reasons": [],
    }
    if "needle_in_a_haystack" not in cfg.get("skip_suites", []):
        block["suites"]["needle_in_a_haystack"] = _run_nihs(
            model_cfg, {**cfg, **(suites_cfg.get("needle_in_a_haystack") or {})}, root
        )
    if "longbench" not in cfg.get("skip_suites", []):
        block["suites"]["longbench"] = _run_longbench(
            model_cfg, {**cfg, **(suites_cfg.get("longbench") or {})}, root
        )
    if "leval" not in cfg.get("skip_suites", []):
        block["suites"]["leval"] = _run_leval(
            model_cfg, {**cfg, **(suites_cfg.get("leval") or {})}, root
        )
    if "aviation_manuals" not in cfg.get("skip_suites", []):
        block["suites"]["aviation_manuals"] = _run_aviation_manuals(
            model_cfg, {**cfg, **(suites_cfg.get("aviation_manuals") or {})}, root
        )

    suites = list(block["suites"].values())
    measured = sum(int(s.get("measured", 0)) for s in suites)
    blocked = [name for name, suite in block["suites"].items() if suite.get("verdict") == "BLOCKED"]
    warned = [name for name, suite in block["suites"].items() if suite.get("verdict") == "WARN"]
    block["summary"] = {
        "measured_cases": measured,
        "blocked_suites": blocked,
        "latency": _summary_latency(suites),
    }
    thresholds = cfg.get("thresholds") or {}
    niah_min = float(thresholds.get("nihs_recall_min", 0.80))
    longbench_min = float(thresholds.get("longbench_score_min", 0.20))
    leval_min = float(thresholds.get("leval_accuracy_min", 0.20))
    aviation_min = float(thresholds.get("aviation_manual_score_min", 0.60))

    if measured == 0:
        block["verdict"] = "BLOCKED"
        block["verdict_reasons"].append("no long-context suite produced measured cases")
    if blocked and block["verdict"] == "PASS":
        block["verdict"] = "WARN"
        block["verdict_reasons"].append(f"blocked suites: {', '.join(blocked)}")
    if warned and block["verdict"] == "PASS":
        block["verdict"] = "WARN"
        block["verdict_reasons"].append(f"warned suites: {', '.join(warned)}")

    niah = block["suites"].get("needle_in_a_haystack") or {}
    if niah.get("measured") and float(niah.get("recall", 0.0)) < niah_min:
        block["verdict"] = "FAIL"
        block["verdict_reasons"].append(f"NIHS recall {niah.get('recall')} < {niah_min}")
    lb = block["suites"].get("longbench") or {}
    if lb.get("measured") and float(lb.get("score", 0.0)) < longbench_min and block["verdict"] == "PASS":
        block["verdict"] = "WARN"
        block["verdict_reasons"].append(f"LongBench subset score {lb.get('score')} < {longbench_min}")
    leval = block["suites"].get("leval") or {}
    if leval.get("measured") and float(leval.get("accuracy", 0.0)) < leval_min and block["verdict"] == "PASS":
        block["verdict"] = "WARN"
        block["verdict_reasons"].append(f"L-Eval subset accuracy {leval.get('accuracy')} < {leval_min}")
    aviation = block["suites"].get("aviation_manuals") or {}
    if aviation.get("measured") and float(aviation.get("score", 0.0)) < aviation_min:
        block["verdict"] = "FAIL"
        block["verdict_reasons"].append(
            f"Aviation manual score {aviation.get('score')} < {aviation_min}"
        )
    return block
