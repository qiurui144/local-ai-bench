"""Fire AMD full verification as a detached background process (run this from SSH)."""
import subprocess, sys, os, time

DETACHED_PROCESS = 0x00000008
CREATE_NEW_PROCESS_GROUP = 0x00000200

cwd = r"C:\Users\happy\vlm-llm-benchmark"
logfile = r"C:\Users\happy\amd_full_verify_driver.log"

env = dict(os.environ)
env["OLLAMA_AMD_BASE_URL"] = "http://localhost:11434/v1"
env["ORT_AMD_EXTRAS_BASE_URL"] = "http://localhost:8091/v1"
env["PYTHONUNBUFFERED"] = "1"

cmd = [sys.executable, r"scripts\launch_amd_full_verify.py"]

with open(logfile, "w") as f:
    f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Firing AMD full 3-seed verification\n")
    proc = subprocess.Popen(
        cmd, cwd=cwd, env=env,
        stdout=f, stderr=f,
        creationflags=DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP,
    )

print(f"PID: {proc.pid}  driver_log: {logfile}")
print(f"Detail log: C:\\Users\\happy\\amd_full_verify.log")
print(f"Reports: {cwd}\\output\\reports\\")
