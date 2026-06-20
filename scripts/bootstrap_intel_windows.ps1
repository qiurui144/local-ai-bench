# Run once in an elevated PowerShell on the Intel Windows laptop.
#
# Purpose:
# - Install Python and Ollama on a clean Windows system.
# - Enable OpenSSH Server for controller-side deployment.
# - Open firewall ports for SSH and Ollama.
# - Pull the default Intel CPU benchmark models.

$ErrorActionPreference = "Stop"

winget install -e --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
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

ollama pull llama3.2:1b
ollama pull qwen2.5:3b
ollama pull qwen3-embedding:0.6b

python -m pip install --upgrade pip
python -m pip install -r requirements-windows.txt
python -m pip uninstall -y onnxruntime onnxruntime-directml
python -m pip install openvino==2025.4.1 openvino-telemetry==2025.2.0 onnxruntime-directml rapidocr-onnxruntime rapidocr-openvino --no-deps

Write-Host "Bootstrap complete. Verify from the controller:"
Write-Host "  ssh <user>@<intel-ip> hostname"
Write-Host "  curl http://<intel-ip>:11434/api/version"
Write-Host "  python scripts\validate_windows_accel.py"
