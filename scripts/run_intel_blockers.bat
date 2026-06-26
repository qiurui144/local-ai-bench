@echo off
REM Intel blocker resolution runner
REM Launch: wmic process call create "cmd /c C:\Users\happy\run_intel_blockers.bat"
REM Output log: C:\Users\happy\intel_blockers.log

set PYTHONUNBUFFERED=1
cd /d C:\Users\happy\vlm-llm-benchmark
"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe" scripts\launch_intel_blockers.py > C:\Users\happy\intel_blockers.log 2>&1
