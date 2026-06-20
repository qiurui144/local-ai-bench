# Run once in an elevated PowerShell on the AMD Windows laptop.
#
# Purpose:
# - Install Python, Ollama, VC++ runtime, and benchmark Python dependencies.
# - Enable OpenSSH Server for controller-side deployment.
# - Configure Ollama for LAN access and iGPU acceleration.
# - Pull the default AMD benchmark models.
#
# Ryzen AI / VitisAI NPU packages are not installed by this script because
# they require the AMD Ryzen AI Software package and platform-specific driver
# validation. See docs/WINDOWS-RESOURCE-MATERIALS.md.

$ErrorActionPreference = "Stop"

winget install -e --id Python.Python.3.11 --silent --accept-source-agreements --accept-package-agreements
winget install -e --id Ollama.Ollama --silent --accept-source-agreements --accept-package-agreements
winget install -e --id Microsoft.VCRedist.2015+.x64 --silent --accept-source-agreements --accept-package-agreements

Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd
Set-Service -Name sshd -StartupType Automatic

New-NetFirewallRule `
  -Name "OpenSSH-Server-In-TCP" `
  -DisplayName "OpenSSH Server" `
  -Enabled True `
  -Direction Inbound `
  -Protocol TCP `
  -Action Allow `
  -LocalPort 22 `
  -ErrorAction SilentlyContinue

New-NetFirewallRule `
  -Name "Ollama-11434" `
  -DisplayName "Ollama 11434" `
  -Enabled True `
  -Direction Inbound `
  -Protocol TCP `
  -Action Allow `
  -LocalPort 11434 `
  -ErrorAction SilentlyContinue

setx OLLAMA_HOST 0.0.0.0 | Out-Null
setx OLLAMA_IGPU_ENABLE 1 | Out-Null

$ollama = Join-Path $env:LOCALAPPDATA "Programs\Ollama\ollama.exe"
if (Test-Path $ollama) {
  Start-Process -FilePath $ollama -ArgumentList "serve" -WindowStyle Hidden
} else {
  Start-Process -FilePath "ollama.exe" -ArgumentList "serve" -WindowStyle Hidden
}

ollama pull llama3.2:3b
ollama pull qwen2.5:7b
ollama pull qwen3:0.6b
ollama pull qwen3-embedding:0.6b
ollama pull bge-m3:latest
ollama pull llava:7b

python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python -m pip uninstall -y onnxruntime onnxruntime-directml
python -m pip install onnxruntime-directml rapidocr-onnxruntime --no-deps

Write-Host "AMD bootstrap complete. Verify from the controller:"
Write-Host "  ssh <user>@<amd-ip> hostname"
Write-Host "  curl http://<amd-ip>:11434/api/version"
Write-Host "  python run_benchmark.py --target amd-win-x86 --model rapidocr-amd-directml --skip stability"
