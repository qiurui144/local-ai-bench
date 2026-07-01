param(
    [string]$VenvPath = "$env:USERPROFILE\ov-llm-venv",
    [string]$PythonExe = "python",
    [string]$OpenVinoVersion = "2026.2.1",
    [string]$OpenVinoTokenizersVersion = "2026.2.1.0",
    [string]$OptimumVersion = "2.2.0",
    [string]$OptimumIntelVersion = "2.0.0"
)

$ErrorActionPreference = "Stop"

if (!(Test-Path (Join-Path $VenvPath "Scripts\python.exe"))) {
    Write-Host "Creating Intel OpenVINO LLM venv at $VenvPath"
    & $PythonExe -m venv --system-site-packages $VenvPath
}

$VenvPython = Join-Path $VenvPath "Scripts\python.exe"
Write-Host "Using $VenvPython"

& $VenvPython -m pip install --upgrade --no-deps `
    "openvino==$OpenVinoVersion" `
    "openvino-tokenizers==$OpenVinoTokenizersVersion" `
    "optimum==$OptimumVersion" `
    "optimum-intel==$OptimumIntelVersion"

& $VenvPython -c "import sys; print(sys.executable); import openvino; print('openvino', openvino.__version__, openvino.__file__); from optimum.intel import OVModelForCausalLM; print('OVModelForCausalLM import ok')"

Write-Host "Set OV_INTEL_LLM_PYTHON=$VenvPython before launching Intel OpenVINO LLM services."
