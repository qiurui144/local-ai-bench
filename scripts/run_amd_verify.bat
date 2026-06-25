@echo off
REM AMD full 3-seed verification runner
REM Launched via: wmic process call create "cmd /c C:\Users\happy\run_amd_verify.bat"
REM Output log: C:\Users\happy\amd_full_verify.log

set OLLAMA_AMD_BASE_URL=http://localhost:11434/v1
set ORT_AMD_EXTRAS_BASE_URL=http://localhost:8091/v1
set PYTHONUNBUFFERED=1

cd /d C:\Users\happy\vlm-llm-benchmark

"C:\Users\happy\AppData\Local\Programs\Python\Python311\python.exe" scripts\launch_amd_full_verify.py > C:\Users\happy\amd_full_verify.log 2>&1
