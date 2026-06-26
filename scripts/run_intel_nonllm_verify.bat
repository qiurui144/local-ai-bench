@echo off
REM Intel non-LLM 3-seed verification runner
REM Launch: wmic process call create "cmd /c C:\Users\happy\run_intel_nonllm_verify.bat"
REM Output log: C:\Users\happy\intel_nonllm_verify.log

set OLLAMA_INTEL_WIN_BASE_URL=http://localhost:11434/v1
set PYTHONUNBUFFERED=1

cd /d C:\Users\happy\vlm-llm-benchmark

"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe" scripts\launch_intel_nonllm_verify.py > C:\Users\happy\intel_nonllm_verify.log 2>&1
