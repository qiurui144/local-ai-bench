# K3 RISC-V 16G CPU ORT / sherpa

**Last updated:** 2026-07-14
**Chinese version:** [cpu-ort.zh.md](cpu-ort.zh.md)
**Legacy source:** [../../k3-riscv.en.md](../../k3-riscv.en.md)

## Scope

This page covers legacy K3 16G CPU-side ONNX Runtime and sherpa-onnx bottom-model evidence. It is not a current v1 NAS contract artifact.

## Workload Results

| Workload | Model/path | Key metric | Status | Decision |
|---|---|---:|---|---|
| Embedding | `bge-m3` ONNX | 77ms cold search | PASS | Legacy local retrieval default |
| Reranker | `bge-reranker-base` ORT | ranking correct | PASS | Legacy local reranker default |
| OCR | PP-OCRv4 + layout | 315ms | PASS | Legacy local OCR default |
| ASR | sherpa SenseVoice + diarization | RTF 0.17 | PASS | Legacy local ASR default |

## Decision

These rows support the K3 16G local bottom-model architecture, but they remain legacy evidence until the P3 contract retest emits required v1 artifacts.

## Evidence

| Detail | Report |
|---|---|
| Legacy full report | [../../k3-riscv.en.md](../../k3-riscv.en.md) |
