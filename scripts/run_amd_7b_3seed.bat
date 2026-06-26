@echo off
set PYTHONUNBUFFERED=1
cd /d C:\Users\happy\vlm-llm-benchmark
"C:\Users\happy\AppData\Local\Programs\Python\Python312\python.exe" scripts\launch_amd_7b_3seed.py > C:\Users\happy\amd_7b_3seed.log 2>&1
