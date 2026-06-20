# ASR datasets

The ASR dimension scores Chinese **CER / WER / RTF** over an audio manifest.

## Files

- `manifest.template.jsonl` — a 2-line example manifest (shipped). Copy it to
  `manifest.jsonl` and point each row at a real audio file (e.g. an
  [AISHELL](https://www.openslr.org/33/) test subset) on your machine.
- `manifest.jsonl` — **you provide this** (gitignored alongside other binary
  fixtures). Audio files themselves are **not shipped** (large + may carry PII /
  licensing constraints, same policy as `fixtures/`).

When `manifest.jsonl` is absent the ASR dimension reports
`status: "blocked", reason: "no dataset"` and the verdict is `SKIP` — it never
crashes the run.

## Manifest JSONL schema

One object per line:

```json
{"audio": "wav/BAC009S0764W0121.wav", "text": "甚至出现交易几乎停滞的情况", "duration": 4.2, "uid": "u1"}
```

| field | type | required | meaning |
|---|---|---|---|
| `audio` | string | yes | path to a 16 kHz PCM WAV (absolute, or relative to `audio_root`) |
| `text` | string | yes | reference transcript (ground truth) |
| `duration` | number | no | audio seconds (read from the WAV header if omitted; needed for RTF) |
| `uid` | string | no | utterance id |

## ONNX backend

Transcription uses a pluggable ONNX backend, default
[sherpa-onnx](https://github.com/k2-fsa/sherpa-onnx) SenseVoice (the model the
K23 edge eval used to hit CER 1.17 % / RTF 0.086). Point
`models.yaml::benchmarks.asr.model_dir` at a directory containing `model.onnx`
+ `tokens.txt`, and `pip install sherpa-onnx soundfile`. Without the backend the
dimension reports `blocked / no asr backend` (the CER/WER/RTF scoring logic is
still fully unit-tested on CPU via an injected transcriber).
