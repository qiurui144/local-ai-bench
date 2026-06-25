@echo off
set OV_INTEL_QWEN3_4B_BASE_URL=http://localhost:8084/v1
set PYTHONUNBUFFERED=1
cd /d C:\Users\happy\vlm-llm-benchmark
python run_benchmark.py --model qwen3-4b-igpu-intel-win --seeds 3 --skip stability,concurrency,conditioned,scenarios,conversation_drift,translation,asr,ocr,embedding,rerank,prefill_decode,ttft,throughput >> C:\Users\happy\bench_4b_igpu_ga.log 2>&1
