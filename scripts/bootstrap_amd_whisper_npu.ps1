param(
  [string]$Root = "$env:USERPROFILE\models\amd-whisper-npu",
  [string]$Python = "$env:USERPROFILE\py312-ryzenai\python.exe",
  [string[]]$Models = @("whisper-base"),
  [switch]$InstallDeps
)

$ErrorActionPreference = "Stop"

if (!(Test-Path $Python)) {
  throw "RyzenAI Python not found: $Python"
}

$Models = @($Models | ForEach-Object { $_ -split "," } | ForEach-Object { $_.Trim() } | Where-Object { $_ })

$configDir = Join-Path $Root "config"
$modelRoot = Join-Path $Root "models"
New-Item -ItemType Directory -Force -Path $configDir, $modelRoot | Out-Null

$baseUrl = "https://raw.githubusercontent.com/amd/RyzenAI-SW/main/Demos/ASR/Whisper/config"
$configFiles = @(
  "model_config.json",
  "vitisai_config_whisper_encoder.json",
  "vitisai_config_whisper_decoder.json"
)
foreach ($file in $configFiles) {
  $dest = Join-Path $configDir $file
  Invoke-WebRequest -Uri "$baseUrl/$file" -OutFile $dest
}

if ($InstallDeps) {
  & $Python -m pip install --upgrade "numpy<2" huggingface_hub transformers soundfile librosa
}

$downloadCode = @'
import sys
from pathlib import Path
from huggingface_hub import snapshot_download

repo_map = {
    "whisper-tiny": "amd/whisper-tiny-onnx-npu",
    "whisper-base": "amd/whisper-base-onnx-npu",
    "whisper-small": "amd/whisper-small-onnx-npu",
    "whisper-medium": "amd/whisper-medium-onnx-npu",
    "whisper-large-v3-turbo": "amd/whisper-large-turbo-onnx-npu",
}

model = sys.argv[1]
dest = Path(sys.argv[2]) / model
if model not in repo_map:
    raise SystemExit(f"Unsupported model: {model}")
dest.mkdir(parents=True, exist_ok=True)
snapshot_download(repo_id=repo_map[model], local_dir=str(dest), local_dir_use_symlinks=False)
print(dest)
'@

foreach ($model in $Models) {
  $downloadScript = Join-Path $Root "download_whisper_npu.py"
  Set-Content -Path $downloadScript -Value $downloadCode -Encoding UTF8
  & $Python $downloadScript $model $modelRoot
}

Write-Host "AMD Whisper NPU assets prepared under $Root"
Write-Host "Config: $(Join-Path $configDir 'model_config.json')"
