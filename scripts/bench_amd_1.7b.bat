@echo off
set OLLAMA_AMD_BASE_URL=http://localhost:11434/v1
set PYTHONUNBUFFERED=1
cd /d C:\Users\happy\vlm-llm-benchmark
python run_benchmark.py --model qwen3-1.7b-amd --seeds 3 --skip stability,concurrency,conditioned,scenarios,conversation_drift,translation,asr,ocr,embedding,rerank,prefill_decode,ttft,throughput >> C:\Users\happy\bench_1.7b_ga_fixed.log 2>&1
