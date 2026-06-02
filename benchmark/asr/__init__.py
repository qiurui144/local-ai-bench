"""ASR benchmark dimension — Chinese CER / WER / RTF over an audio manifest.

Synced from the K23 edge eval SenseVoice methodology
(``2026-06-02_yolo_asr_ui_eval.md`` §C: real AISHELL utterances, custom numpy
pipeline → ONNX CTC → greedy decode, CER 1.17 % / RTF 0.086). Adapted here to a
pluggable ONNX backend (sherpa-onnx SenseVoice by default) that degrades
gracefully to BLOCKED when the runtime / model / dataset is absent — so the
scoring logic ships and is unit-tested on CPU without shipping a model.

- ``metrics`` : pure-Python WER / CER (edit distance) + RTF + transcript
  validation (empty output = FAIL). CPU-only, fully unit-testable.
- ``datasets``: ASR manifest loader (audio path + reference transcript).
- ``runner``  : ``run_asr`` — transcribe + score CER/WER/RTF with PASS/WARN/FAIL
  (or SKIP/blocked) verdict.
"""

from __future__ import annotations

__all__ = [
    "datasets",
    "metrics",
    "runner",
]
