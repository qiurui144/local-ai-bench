@echo off
REM Intel sequential 3-seed benchmark runner
REM Launched via: wmic process call create "cmd /c C:\Users\happy\run_intel_verify.bat"
REM Output log: C:\Users\happy\intel_sequential.log

set OV_INTEL_QWEN25_7B_BASE_URL=http://localhost:8085/v1
set OV_INTEL_QWEN3_4B_BASE_URL=http://localhost:8084/v1
set OV_INTEL_EXTRAS_BASE_URL=http://localhost:8081/v1
set PYTHONUNBUFFERED=1

cd /d C:\Users\happy\vlm-llm-benchmark

"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe" scripts\launch_intel_sequential.py > C:\Users\happy\intel_sequential.log 2>&1
