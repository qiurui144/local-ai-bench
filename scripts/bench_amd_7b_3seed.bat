@echo off
set OLLAMA_AMD_BASE_URL=http://localhost:11434/v1
set PYTHONUNBUFFERED=1
cd /d C:\Users\happy\vlm-llm-benchmark
python run_benchmark.py --model qwen2.5-7b-amd-win --seeds 3 >> C:\Users\happy\amd_7b_3seed.log 2>&1
