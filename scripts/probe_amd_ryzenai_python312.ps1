# Build a standalone Python 3.12 probe environment for AMD RyzenAI.
#
# This avoids the Windows Python installer path. RyzenAI 1.7.1 ships CPython
# 3.12 wheels, while the AMD benchmark laptop may still use Python 3.11 for the
# main DirectML/Ollama benchmark stack.

$ErrorActionPreference = "Stop"

$RyzenAiRoot = "C:\Program Files\RyzenAI\1.7.1"
$ProbeRoot = Join-Path $env:USERPROFILE "py312-ryzenai"
$EmbedZip = Join-Path $env:TEMP "python-3.12.10-embed-amd64.zip"
$PythonExe = Join-Path $ProbeRoot "python.exe"

if (!(Test-Path $RyzenAiRoot)) {
  throw "RyzenAI root not found: $RyzenAiRoot"
}

if (!(Test-Path $PythonExe)) {
  New-Item -ItemType Directory -Force -Path $ProbeRoot | Out-Null
  Invoke-WebRequest `
    -Uri "https://www.python.org/ftp/python/3.12.10/python-3.12.10-embed-amd64.zip" `
    -OutFile $EmbedZip
  Expand-Archive -Path $EmbedZip -DestinationPath $ProbeRoot -Force

  $Pth = Join-Path $ProbeRoot "python312._pth"
  (Get-Content $Pth) |
    ForEach-Object { if ($_ -eq "#import site") { "import site" } else { $_ } } |
    Set-Content $Pth -Encoding ASCII
}

$GetPip = Join-Path $env:TEMP "get-pip.py"
Invoke-WebRequest -Uri "https://bootstrap.pypa.io/get-pip.py" -OutFile $GetPip
& $PythonExe $GetPip
& $PythonExe -m pip install --upgrade pip

Push-Location $RyzenAiRoot
& $PythonExe -m pip install setuptools wheel protobuf numpy==1.26.4
& $PythonExe -m pip install --no-deps `
  .\onnxruntime_vitisai-1.23.3-cp312-cp312-win_amd64.whl `
  .\onnxruntime_providers_ryzenai-0.11.1-py3-none-win_amd64.whl `
  .\ryzenai_onnx_utils-1.7.1-py3-none-any.whl `
  .\voe-1.7.1-py3-none-win_amd64.whl
Pop-Location

# RapidOCR PP-OCR pipeline dependencies for the benchmark VitisAI helper.
# Keep NumPy < 2 because onnxruntime_vitisai 1.23.3 is built against NumPy 1.x.
& $PythonExe -m pip install --no-deps rapidocr-onnxruntime==1.4.4
& $PythonExe -m pip install --force-reinstall `
  numpy==1.26.4 `
  opencv-python-headless==4.10.0.84 `
  pillow `
  pyclipper `
  shapely `
  pyyaml `
  six `
  tqdm `
  coloredlogs `
  flatbuffers `
  sympy `
  colorlog `
  ml-dtypes `
  rich

$env:PATH = "$RyzenAiRoot\deployment;$RyzenAiRoot\onnxruntime\bin;$RyzenAiRoot\xrt;$env:PATH"
& $PythonExe -c "import json, onnxruntime as ort, sys; print(json.dumps({'exe': sys.executable, 'providers': ort.get_available_providers()}))"

$RepoRoot = (Get-Location).Path
$OcrHelper = Join-Path $RepoRoot "scripts\ocr_vitisai_rapidocr.py"
if (Test-Path $OcrHelper) {
  $env:PYTHONIOENCODING = "utf-8"
  & $PythonExe $OcrHelper --probe
}
