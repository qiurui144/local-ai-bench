# Multi-Platform Deployment SOP

Step-by-step runbook for deploying and running the local-ai-bench harness on all supported
remote target platforms. Covers SSH prerequisites, runtime startup, model pull, benchmark
invocation, and per-platform skip-dimension recommendations.

**Priority order** (reflects typical hardware availability):

1. [AMD Linux](../reports/amd-linux.en.md) — current Ryzen 8845H target state (`amd-linux-x86`)
2. [AMD Windows](#1-amd-windows--ryzen-8845h--radeon-780m-vulkan) — historical dual-boot path, use only when the machine is explicitly booted into Windows
3. [Intel Windows](#2-intel-windows--openvinoigpu-primary-cpu-baseline-explicit-only)
4. [Rockchip RK3588 Linux](#3-rockchip-rk3588-linux--rknn-npu)
5. [SpacemiT K3 Linux / RISC-V](#4-spacemit-k3-linux--risc-v-riscv64)
6. [Intel Linux](#5-intel-linux--openvinovllm-accelerated-runtime-cpu-baseline-explicit-only)
7. [Other ARM / macOS / Jetson](#6-other-arm--macos--jetson-brief)

---

## 0. Common Setup (Controller Side — Read Before Any Platform)

All platforms require the same controller-side prerequisites. Complete this section once.

### 0.1 Required controller tools

```bash
# Linux (Debian/Ubuntu)
sudo apt-get install -y sshpass rsync openssh-client curl

# macOS
brew install hudochenkov/sshpass/sshpass rsync curl
```

Verify:

```bash
sshpass -V
rsync --version
ssh -V
```

### 0.2 Environment variables

Each target platform reads its connection credentials from environment variables — **no IPs or
passwords are ever committed to `targets.yaml` or any source file** (global rule §1.4).

Create or extend your `.env` file (copy from `.env.example`):

```bash
cp .env.example .env
# Then edit .env — it is gitignored
```

Full variable table:

| Platform | Variable | Description |
|---|---|---|
| AMD Windows | `AMD_HOST` | IP/hostname of the AMD Windows machine |
| AMD Windows | `AMD_SSH_USER` | SSH username (Microsoft account: `email@domain`) |
| AMD Windows | `AMD_SSH_PASS` | SSH password |
| AMD Windows | `OLLAMA_AMD_BASE_URL` | Ollama endpoint seen from controller, e.g. `http://$AMD_HOST:11434/v1` |
| AMD Linux | `AMD_LINUX_HOST` | IP/hostname of the AMD Linux machine |
| AMD Linux | `AMD_LINUX_SSH_USER` | SSH username |
| AMD Linux | `AMD_LINUX_SSH_PASS` | SSH password |
| AMD Linux | `OLLAMA_AMD_LINUX_BASE_URL` | Target-local Ollama endpoint, normally `http://localhost:11434/v1` |
| Intel Windows | `INTEL_WIN_HOST` | IP/hostname of the Intel Windows machine |
| Intel Windows | `INTEL_WIN_SSH_USER` | SSH username |
| Intel Windows | `INTEL_WIN_SSH_PASS` | SSH password |
| Intel Windows | `OLLAMA_INTEL_WIN_BASE_URL` | Ollama endpoint, e.g. `http://$INTEL_WIN_HOST:11434/v1` |
| RK3588 | `RK3588_HOST` | IP/hostname of the RK3588 board |
| RK3588 | `RK3588_USER` | SSH username (typically `pi` or `rock`) |
| RK3588 | `RK3588_PASS` | SSH password |
| RK3588 | `RK3588_LLM_BASE_URL` | RKNN adapter endpoint, e.g. `http://$RK3588_HOST:8080/v1` |
| SpacemiT K3 | `K3_HOST` | IP/hostname of the K3 board |
| SpacemiT K3 | `K3_USER` | SSH username |
| SpacemiT K3 | `K3_PASS` | SSH password |
| SpacemiT K3 | `K3_LLM_BASE_URL` | llama-server endpoint, e.g. `http://$K3_HOST:8080/v1` |
| Intel Linux | `INTEL_LINUX_HOST` | IP/hostname of the Intel Linux machine |
| Intel Linux | `INTEL_LINUX_SSH_USER` | SSH username |
| Intel Linux | `INTEL_LINUX_SSH_PASS` | SSH password |
| Intel Linux | `OV_INTEL_LINUX_BASE_URL` | Target-local OpenVINO/OpenAI-compatible endpoint, normally `http://localhost:8080/v1` |
| Intel Linux | `OLLAMA_INTEL_LINUX_BASE_URL` | Target-local Ollama endpoint for explicit CPU baselines, normally `http://localhost:11434/v1` |
| Jetson | `JETSON_HOST` | IP/hostname of the Jetson board |
| Jetson | `JETSON_USER` | SSH username |
| Jetson | `JETSON_PASS` | SSH password |

Source the file before running benchmarks:

```bash
source .env
# or: set -a && source .env && set +a   # export all to subprocesses
```

### 0.3 targets.yaml — registered platform pool

`targets.yaml` (repo root) maps a short target identifier to its platform metadata. The harness
uses this when you pass `--target <id>`. Connection details come from the env vars above — only
the variable *names* are stored in `targets.yaml`.

To verify the file parses correctly:

```bash
python3 -c "import yaml; t = yaml.safe_load(open('targets.yaml')); print(list(t['targets'].keys()))"
# Expected: ['local', 'amd-win-x86', 'rk3588-linux', 'k3-riscv', 'jetson-agx', 'intel-win-x86', 'intel-linux']
```

### 0.4 SSH keypair (recommended over password auth)

Password auth via `sshpass` works but is less robust for long sessions. For stable CI/automation
use key-based auth:

```bash
ssh-keygen -t ed25519 -C "vlm-bench-controller" -f ~/.ssh/vlm_bench_ed25519
ssh-copy-id -i ~/.ssh/vlm_bench_ed25519.pub $AMD_SSH_USER@$AMD_HOST
```

Once keys are installed, remove `ssh_pass_env` references from your env or leave them empty —
the SSH client will fall through to key auth automatically.

---

## 1. AMD Windows — Ryzen 8845H + Radeon 780M (Vulkan)

> **Current 192.168.100.201 state:** the Ryzen 8845H machine is currently booted into AMD Linux and should be targeted as `amd-linux-x86` (`AMD_LINUX_HOST`, `AMD_LINUX_SSH_USER`, `AMD_LINUX_SSH_PASS`). This Windows section is retained for historical dual-boot reproduction only; do not use it unless the machine is explicitly booted into Windows.

**Target ID**: `amd-win-x86`  
**Runtime**: Ollama with Vulkan backend (RDNA3 iGPU, 17.9 GiB shared memory pool)  
**NPU**: AMD XDNA 16-TOPS — **not yet accessible via Ollama** (future: VitisAI EP path)

### 1.1 Prerequisites (Windows side — one-time)

1. Install [Ollama for Windows](https://ollama.com/download/windows).
2. Install [OpenSSH Server](https://learn.microsoft.com/en-us/windows-server/administration/openssh/openssh_install_firstuse):
   ```powershell
   Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
   Start-Service sshd
   Set-Service -Name sshd -StartupType 'Automatic'
   ```
3. Open Windows Firewall for inbound TCP 22 (SSH) and TCP 11434 (Ollama):
   ```powershell
   New-NetFirewallRule -Name "Ollama" -DisplayName "Ollama LLM" -Protocol TCP -LocalPort 11434 -Action Allow -Direction Inbound
   ```
4. Note your SSH username. For Microsoft accounts this is the email-style format
   (`happy@example.com`), not the short display name.

### 1.2 Start Ollama with Vulkan GPU + LAN access

Ollama must be started with `OLLAMA_HOST=0.0.0.0` so the controller can reach it.
The `HSA_OVERRIDE_GFX_VERSION=gfx1102` workaround forces the Radeon 780M to be recognized by
the ROCm/HIP stack embedded in Ollama's Vulkan path.

```bash
# From the controller (Linux/macOS), start Ollama on the AMD machine via SSH
sshpass -p "$AMD_SSH_PASS" ssh -o ServerAliveInterval=30 "$AMD_SSH_USER@$AMD_HOST" \
  'wmic process call create "cmd /c setx /M OLLAMA_HOST 0.0.0.0 && setx /M HSA_OVERRIDE_GFX_VERSION gfx1102 && ollama.exe serve"'
```

`wmic process call create` launches the command in Session 0 (system session), allowing Ollama
to stay alive even after the SSH session closes.

Alternatively, on the Windows machine directly:

```cmd
setx /M OLLAMA_HOST 0.0.0.0
setx /M HSA_OVERRIDE_GFX_VERSION gfx1102
ollama.exe serve
```

Wait for Ollama to be ready (controller side):

```bash
until curl -sf "http://$AMD_HOST:11434/api/version" > /dev/null; do
  echo "Waiting for Ollama..."; sleep 5
done
echo "Ollama ready"
```

### 1.3 Pull a model

```bash
sshpass -p "$AMD_SSH_PASS" ssh "$AMD_SSH_USER@$AMD_HOST" \
  'wmic process call create "cmd /c ollama.exe pull llama3.2:3b"'
# Monitor progress
sshpass -p "$AMD_SSH_PASS" ssh "$AMD_SSH_USER@$AMD_HOST" 'ollama list'
```

**VRAM guidance** (17.9 GiB shared — Windows reserves ~2 GiB for display):

| Model size | Recommended quant | Ollama layers on GPU |
|---|---|---|
| ≤ 3B | Q4_K_M | 29/29 (full) |
| 7B | Q4_K_M | 29/29 (full) |
| 14B | Q4_K_M | ~14/32 (partial) |
| 32B+ | Not recommended | OOM |

### 1.4 Run the benchmark

```bash
export OLLAMA_AMD_BASE_URL="http://$AMD_HOST:11434/v1"

python run_benchmark.py \
  --target amd-win-x86 \
  --model llama3.2-3b-amd-win \
  --seeds 3 \
  --skip stability,embedding,rerank,asr
```

### 1.5 GPU recognition troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `HSA_STATUS_ERROR_INVALID_AGENT` | 780M gfx version not detected | Set `HSA_OVERRIDE_GFX_VERSION=gfx1102` in environment |
| Ollama says `0 GPU layers` | OLLAMA_HOST not set or Vulkan init failed | Restart with `OLLAMA_HOST=0.0.0.0` and confirm `ollama ps` shows `gpu` |
| Very low throughput (~2 tok/s) | Falling back to CPU | Stop the run and fix acceleration; continue only for an explicit CPU baseline |
| `Connection refused` on port 11434 | Firewall blocking | Check `netstat -ano | findstr 11434` on Windows |

---

## 2. Intel Windows — OpenVINO/iGPU Primary, CPU Baseline Explicit Only

**Target ID**: `intel-win-x86`  
**Runtime**: OpenVINO/iGPU services for normal LLM/embedding/rerank runs; Ollama CPU only for
explicit CPU baseline or diagnostics
**Use case**: Core Ultra / Arc iGPU Windows machines

### 2.1 Prerequisites (Windows side — one-time)

Same OpenSSH Server setup as AMD Windows (see §1.1 steps 1–3). Skip the GPU-specific
`HSA_OVERRIDE_GFX_VERSION` variable.

### 2.2 CPU baseline path (explicit only)

Normal Intel Windows LLM/VLM calibration should use the `*-igpu-intel-win` OpenVINO models.
Do not run Ollama CPU LLM/VLM as part of the normal model matrix. If a CPU baseline is
requested, label the report as CPU baseline and exclude it from accelerator performance
conclusions.

Start Ollama only for that explicit CPU baseline:

```bash
# Start Ollama on the Intel Windows machine from the controller
sshpass -p "$INTEL_WIN_SSH_PASS" ssh -o ServerAliveInterval=30 \
  "$INTEL_WIN_SSH_USER@$INTEL_WIN_HOST" \
  'wmic process call create "cmd /c setx /M OLLAMA_HOST 0.0.0.0 && ollama.exe serve"'
```

Wait for readiness:

```bash
until curl -sf "http://$INTEL_WIN_HOST:11434/api/version" > /dev/null; do
  echo "Waiting..."; sleep 5
done
```

### 2.3 Pull a CPU-baseline model

For CPU baseline only, prefer small quantized models:

```bash
sshpass -p "$INTEL_WIN_SSH_PASS" ssh "$INTEL_WIN_SSH_USER@$INTEL_WIN_HOST" \
  'wmic process call create "cmd /c ollama.exe pull qwen3:0.6b"'
```

| RAM available | Recommended model | Expected throughput |
|---|---|---|
| 8 GB | llama3.2:1b (Q4) | ~5–12 tok/s |
| 16 GB | llama3.2:3b (Q4) | ~3–8 tok/s |
| 32 GB | qwen2.5:7b (Q4) | ~1–4 tok/s |

### 2.4 Run a CPU-baseline benchmark

```bash
export OLLAMA_INTEL_WIN_BASE_URL="http://$INTEL_WIN_HOST:11434/v1"

python run_benchmark.py \
  --target intel-win-x86 \
  --model llama3.2-1b-intel-win \
  --seeds 3 \
  --skip stability,embedding,rerank,asr,conditioned
```

For the stronger Intel CPU baseline:

```bash
python run_benchmark.py \
  --target intel-win-x86 \
  --model qwen2.5-3b-intel-win \
  --seeds 3 \
  --skip stability,concurrency,conditioned,scenarios
```

CPU-mode LLM/VLM throughput is not a normal production signal for this target. Skip long
dimensions such as `stability`, `concurrency`, `conditioned`, and `scenarios` unless the
baseline request explicitly requires them.

For target-local single-model runs, `scenarios` must not auto-load another same-machine model
as the L2 judge. Use L1-only scenarios by default; enable L2 only when the judge is served by
separate hardware or an external endpoint.

---

## 3. Rockchip RK3588 Linux — RKNN NPU

**Target ID**: `rk3588-linux`  
**Runtime**: Custom `rknn_adapter.py` Flask server (OpenAI-compat) + RKNN Toolkit Lite 2  
**Accelerator**: Rockchip NPU (6 TOPS INT8), Mali-G610 GPU (optional, via Ollama fallback)

### 3.1 Prerequisites (RK3588 board — one-time)

Ensure the board runs an RKNN-capable kernel (OrangePi/Rock 5B vendor kernel or Armbian RKNN):

```bash
# On RK3588 — check NPU device node
ls /dev/rknpu0           # should exist
cat /sys/class/devfreq/fdab0000.npu/cur_freq   # current NPU frequency in Hz
# Expected: 700000000 or 1000000000 (700 MHz / 1 GHz)
```

Install Python dependencies:

```bash
# On RK3588
pip3 install rknn-toolkit-lite2 flask requests numpy
```

Verify RKNN Toolkit Lite 2 version (must match your kernel RKNN driver version):

```bash
python3 -c "from rknnlite.api import RKNNLite; print('OK')"
```

### 3.2 Convert or obtain a .rknn model

RKNN models must be pre-converted on a host with full rknn-toolkit2 (not the lite variant).
Conversion is a separate process not covered here; see the RKNN Toolkit 2 documentation.

Place the converted model at a known path on the RK3588 board:

```bash
# Example: /home/pi/models/qwen2.5-0.5b.rknn
scp qwen2.5-0.5b.rknn $RK3588_USER@$RK3588_HOST:/home/pi/models/
```

### 3.3 Sync harness code to the RK3588

```bash
rsync -az --exclude='output/' --exclude='__pycache__/' \
  ./ $RK3588_USER@$RK3588_HOST:/home/pi/local-ai-bench/
```

Or use the harness sync helper if `--target` + `--sync-only` is wired:

```bash
python run_benchmark.py --target rk3588-linux --model qwen2.5-0.5b-rk3588 --sync-only
```

### 3.4 Start the RKNN adapter server

The adapter (`benchmark/backends/rknn_adapter.py`) exposes a minimal OpenAI-compatible
`/v1/chat/completions` endpoint that loads the RKNN model and runs inference on the NPU.

```bash
sshpass -p "$RK3588_PASS" ssh -o ServerAliveInterval=30 $RK3588_USER@$RK3588_HOST \
  "nohup python3 /home/pi/local-ai-bench/benchmark/backends/rknn_adapter.py \
   --model /home/pi/models/qwen2.5-0.5b.rknn \
   --port 8080 \
   > /tmp/rknn_adapter.log 2>&1 &"
```

Wait for the adapter to be ready:

```bash
until curl -sf "http://$RK3588_HOST:8080/v1/models" > /dev/null; do
  echo "Waiting for RKNN adapter..."; sleep 3
done
echo "RKNN adapter ready"
```

Monitor logs:

```bash
sshpass -p "$RK3588_PASS" ssh $RK3588_USER@$RK3588_HOST "tail -f /tmp/rknn_adapter.log"
```

### 3.5 Run the benchmark

```bash
export RK3588_LLM_BASE_URL="http://$RK3588_HOST:8080/v1"

python run_benchmark.py \
  --target rk3588-linux \
  --model qwen2.5-0.5b-rk3588 \
  --seeds 3 \
  --skip stability,conditioned,general_ability,embedding,rerank,asr
```

**Why skip those dims?** The RKNN adapter only implements text-generation endpoints.
`general_ability` requires dataset downloads (air-gapped boards); `embedding`/`rerank` require
dedicated models; `asr` requires sherpa-onnx which is ARM64-compatible but memory-constrained.

### 3.6 RK3588 troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| `/dev/rknpu0` missing | Wrong kernel | Use vendor kernel or Armbian RKNN build |
| `ImportError: No module named rknnlite` | Missing toolkit | `pip3 install rknn-toolkit-lite2` |
| `RKNN version mismatch` | Toolkit vs driver version conflict | Match rknn-toolkit-lite2 version to kernel RKNN driver version |
| NPU stuck at 0 Hz | NPU power-gated | Trigger inference once to wake it; check devfreq governor |
| Adapter port 8080 blocked | Firewall | `sudo ufw allow 8080/tcp` (Ubuntu) |

---

## 4. SpacemiT K3 Linux — RISC-V (riscv64)

**Target ID**: `k3-riscv`  
**Runtime**: llama.cpp `llama-server` (built with `GGML_RVV=ON` for SpacemiT X60 + RVV 1.0)  
**ISA**: RISC-V RVA22 + RVV 1.0, SpacemiT X60 8-core, ~4–16 tok/s for small models

### 4.1 Build llama.cpp with RVV support (on the K3 board)

```bash
# On K3 board (or via SSH)
sshpass -p "$K3_PASS" ssh $K3_USER@$K3_HOST << 'EOF'
  git clone https://github.com/ggerganov/llama.cpp /home/user/llama.cpp
  cd /home/user/llama.cpp
  cmake -B build \
    -DGGML_RVV=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -DLLAMA_BUILD_SERVER=ON
  cmake --build build --target llama-server -j$(nproc)
EOF
```

Verify RVV detection:

```bash
sshpass -p "$K3_PASS" ssh $K3_USER@$K3_HOST \
  "/home/user/llama.cpp/build/bin/llama-server --version"
# Should mention RVV in build flags
```

### 4.2 Transfer a GGUF model to the K3

Small quantized GGUF models work best within the K3's ~4 GB LPDDR5:

```bash
# Transfer from controller
scp qwen2.5-0.5b-q4_k_m.gguf $K3_USER@$K3_HOST:/home/user/models/
```

Recommended models for K3:
- `qwen2.5-0.5b` Q4_K_M — fastest, good quality-per-parameter
- `llama3.2-1b` Q4_K_M — good general-purpose baseline
- `qwen3-0.6b` Q4_K_M — latest Qwen generation

### 4.3 Start llama-server

```bash
sshpass -p "$K3_PASS" ssh -o ServerAliveInterval=30 $K3_USER@$K3_HOST \
  "nohup /home/user/llama.cpp/build/bin/llama-server \
   --model /home/user/models/qwen2.5-0.5b-q4_k_m.gguf \
   --host 0.0.0.0 \
   --port 8080 \
   --threads $(nproc) \
   --alias qwen2.5-0.5b \
   > /tmp/llama_server.log 2>&1 &"
```

Wait for readiness:

```bash
until curl -sf "http://$K3_HOST:8080/v1/models" > /dev/null; do
  echo "Waiting for llama-server..."; sleep 5
done
echo "llama-server ready"
```

### 4.4 Run the benchmark

```bash
export K3_LLM_BASE_URL="http://$K3_HOST:8080/v1"

python run_benchmark.py \
  --target k3-riscv \
  --model qwen2.5-0.5b-k3 \
  --seeds 3 \
  --skip stability,embedding,rerank,asr,conditioned,general_ability
```

**Performance expectations**:
- TTFT: ~0.5–2 s (prefill phase, RVV-accelerated)
- Throughput: ~4–16 tok/s (decode phase, single-thread bottleneck)
- Larger models (7B+): not recommended, insufficient RAM

### 4.5 K3 / RISC-V troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Very slow build | K3 has limited CPU resources | Cross-compile on x86 host; or use `--jobs 2` to avoid OOM during compile |
| `Illegal instruction` crash | RVV not detected or kernel lacks RVV support | Check `cat /proc/cpuinfo | grep isa`; ensure kernel has `rvv` in isa string |
| OOM during model load | Model too large for 4 GB RAM | Use Q4_K_M quantization; 0.5B or 1B models only |
| Port 8080 unreachable | Firewall or binding issue | Check `ss -tlnp | grep 8080` on K3; confirm `--host 0.0.0.0` |

---

## 5. Intel Linux — OpenVINO/vLLM Accelerated Runtime, CPU Baseline Explicit Only

**Target ID**: `intel-linux`  
**Runtime**: OpenVINO, vLLM, or another accelerated OpenAI-compatible runtime by default;
Ollama CPU only for explicit CPU baseline or diagnostics
**Use case**: x86 Linux servers with accelerator-backed inference, or labeled CPU baselines

AMD Linux and Intel Linux post-Windows sequencing is maintained in
[`docs/amd-intel-linux-test-plan.md`](amd-intel-linux-test-plan.md).

### 5.1 Runtime choice

| Scenario | Recommended runtime | Notes |
|---|---|---|
| Intel Arc/Xe GPU | OpenVINO OpenAI-compatible service | Preferred Intel Linux LLM/VLM path |
| NVIDIA GPU available | vLLM | Full benchmark support, best performance |
| No GPU (CPU-only) | Ollama | CPU baseline only; not a normal LLM/VLM matrix path |

### 5.2 OpenVINO setup (Intel GPU path)

```bash
# On the Intel Linux machine
export OV_INTEL_LINUX_BASE_URL=http://localhost:8080/v1
export OV_INTEL_LINUX_MODEL_ROOT=/path/to/openvino-models
export OV_INTEL_LINUX_LLM_DEVICE=GPU

python scripts/serve_ov_intel.py \
  --llm "$OV_INTEL_LINUX_MODEL_ROOT/qwen3-0.6b-int4-ov" \
  --llm-device GPU \
  --host 0.0.0.0 \
  --port 8080
```

The Linux runner can also restart this service per model when the matching
model directory exists under `OV_INTEL_LINUX_MODEL_ROOT`.

### 5.3 vLLM setup (NVIDIA GPU path)

```bash
# On the Intel Linux machine
pip install vllm

# Start vLLM server
vllm serve Qwen/Qwen2.5-7B-Instruct \
  --host 0.0.0.0 \
  --port 8000 \
  --max-model-len 4096 \
  &
```

Wait for startup (vLLM can take 60–120 s to load):

```bash
until curl -sf "http://$INTEL_LINUX_HOST:8000/v1/models" > /dev/null; do
  echo "Waiting for vLLM..."; sleep 10
done
```

### 5.4 Ollama setup (explicit CPU baseline only)

```bash
# On the Intel Linux machine
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_HOST=0.0.0.0 ollama serve &
ollama pull llama3.2:3b
```

### 5.5 Run the benchmark

```bash
python scripts/run_linux_full_matrix.py \
  --target intel-linux \
  --models qwen3-0.6b-openvino-intel-linux \
  --seeds 3 \
  --tag intel-linux-YYYYMMDD-qwen3-06b-ov \
  --detach
```

CPU-only LLM/VLM baselines must be explicit:

```bash
python scripts/run_linux_full_matrix.py \
  --target intel-linux \
  --models qwen2.5-7b-intel-linux \
  --seeds 3 \
  --tag intel-linux-YYYYMMDD-q25-7b-cpu-baseline \
  --allow-cpu-llm-vlm \
  --detach
```

Keep CPU-baseline reports out of accelerator performance conclusions.

---

## 6. Other ARM / macOS / Jetson (Brief)

### 6.1 NVIDIA Jetson (jetson-agx)

**Target ID**: `jetson-agx` | **Runtime**: Ollama (CUDA) | **Accelerator**: CUDA (Ampere)

```bash
# On Jetson — Ollama supports Jetson with CUDA backend
curl -fsSL https://ollama.com/install.sh | sh
OLLAMA_HOST=0.0.0.0 ollama serve &
ollama pull llama3.2:3b

# Benchmark from controller
export OLLAMA_JETSON_BASE_URL="http://$JETSON_HOST:11434/v1"
python run_benchmark.py --target jetson-agx --model llama3.2-3b-jetson --seeds 3 \
  --skip stability,asr,embedding,rerank
```

GPU memory on Jetson AGX Orin is 64 GB unified — 7B models run comfortably.

### 6.2 macOS Apple Silicon

No `targets.yaml` entry needed for local macOS. Run directly:

```bash
brew install ollama
ollama serve &    # Metal backend auto-enabled
ollama pull qwen3:0.6b
python run_benchmark.py --model qwen3-0.6b-local --seeds 3 --skip stability,asr
```

### 6.3 Raspberry Pi 5

Raspberry Pi 5 (BCM2712, 4–8 GB): llama.cpp CPU-only path. Performance similar to K3 RISC-V.

```bash
# On RPi 5 — build llama.cpp (no special CPU flags needed, NEON is auto-detected)
cmake -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_BUILD_SERVER=ON
cmake --build build --target llama-server -j4

# Recommended skip set for RPi 5
python run_benchmark.py --target rpi5-linux --model qwen2.5-0.5b-rpi5 \
  --skip stability,conditioned,general_ability,embedding,rerank,asr
```

---

## Appendix A — targets.yaml Full Example

Complete `targets.yaml` with all supported platform entries. All IP addresses and credentials
come from environment variables only.

```yaml
# Target Pool — platform/arch/connection/runtime registry
# IPs/passwords injected via env vars only — never hardcoded (§1.4)
targets:
  local:
    platform: linux
    arch: x86_64
    connection: local
    runtime: vllm

  amd-win-x86:
    platform: windows
    arch: x86_64
    connection: ssh
    ip_env: AMD_HOST
    ssh_user_env: AMD_SSH_USER
    ssh_pass_env: AMD_SSH_PASS
    runtime: ollama
    runtime_port: 11434
    accelerator: vulkan
    npu: amd-xdna
    remote_workdir: "C:\\Users\\happy\\local-ai-bench"
    python_cmd: "C:\\Users\\happy\\py311\\python.exe"

  rk3588-linux:
    platform: linux
    arch: aarch64
    connection: ssh
    ip_env: RK3588_HOST
    ssh_user_env: RK3588_USER
    ssh_pass_env: RK3588_PASS
    runtime: rknn
    accelerator: rknn-npu
    mali_gpu: mali-g610
    remote_workdir: "/home/pi/local-ai-bench"
    python_cmd: "python3"

  k3-riscv:
    platform: linux
    arch: riscv64
    connection: ssh
    ip_env: K3_HOST
    ssh_user_env: K3_USER
    ssh_pass_env: K3_PASS
    runtime: llama_cpp
    runtime_port: 8080
    remote_workdir: "/home/user/local-ai-bench"
    python_cmd: "python3"

  jetson-agx:
    platform: linux
    arch: aarch64
    connection: ssh
    ip_env: JETSON_HOST
    ssh_user_env: JETSON_USER
    ssh_pass_env: JETSON_PASS
    runtime: ollama
    runtime_port: 11434
    accelerator: cuda
    remote_workdir: "/home/user/local-ai-bench"
    python_cmd: "python3"

  intel-win-x86:
    platform: windows
    arch: x86_64
    connection: ssh
    ip_env: INTEL_WIN_HOST
    ssh_user_env: INTEL_WIN_SSH_USER
    ssh_pass_env: INTEL_WIN_SSH_PASS
    runtime: ollama
    runtime_port: 11434
    accelerator: cpu
    remote_workdir: "C:\\Users\\user\\local-ai-bench"
    python_cmd: "python.exe"

  intel-linux:
    platform: linux
    arch: x86_64
    connection: ssh
    ip_env: INTEL_LINUX_HOST
    ssh_user_env: INTEL_LINUX_SSH_USER
    ssh_pass_env: INTEL_LINUX_SSH_PASS
    runtime: generic
    runtime_port: 8080
    accelerator: openvino-gpu
    accelerator_profiles: [openvino-gpu, openvino-cpu, cpu]
    env_overrides:
      OLLAMA_INTEL_LINUX_BASE_URL: http://localhost:11434/v1
      OV_INTEL_LINUX_BASE_URL: http://localhost:8080/v1
      INTEL_LINUX_BASE_URL: http://localhost:8080/v1
    remote_workdir: "/home/user/local-ai-bench"
    python_cmd: "python3"
```

---

## Appendix B — `--target` Command Reference

| Flag | Description |
|---|---|
| `--target <id>` | Override default target; `<id>` must match a key in `targets.yaml` |
| `--model <name>` | Model name as defined in `models.yaml` |
| `--seeds N` | Run N independent seeds; verdict = worst across seeds |
| `--skip <dims>` | Comma-separated list of dimension keys to skip |
| `--compare A B` | Offline replaceability comparison of two saved reports |
| `--sync-only` | Rsync harness code to remote target without running benchmarks |

Example — benchmark on AMD Windows with 3 seeds, skipping slow dimensions:

```bash
python run_benchmark.py \
  --target amd-win-x86 \
  --model llama3.2-3b-amd-win \
  --seeds 3 \
  --skip stability,embedding,rerank,asr
```

---

## Appendix C — Skip-Dimension Matrix by Platform

Recommended `--skip` combinations. "●" = run, "–" = skip (not supported or impractical).

| Dimension | local (x86 GPU) | amd-win-x86 | intel-win-x86 | rk3588-linux | k3-riscv | intel-linux (GPU) | intel-linux (CPU) |
|---|---|---|---|---|---|---|---|
| `accuracy` | ● | ● | ● | ● | ● | ● | ● |
| `ttft` | ● | ● | ● | ● | ● | ● | ● |
| `throughput` | ● | ● | ● | ● | ● | ● | ● |
| `prefill_decode` | ● | ● | ● | ● | ● | ● | ● |
| `concurrency` | ● | ● | – | – | – | ● | – |
| `stability` | ● | – | – | – | – | ● | – |
| `translation` | ● | ● | ● | ● | ● | ● | ● |
| `embedding` | ● | – | – | – | – | ● | – |
| `rerank` | ● | – | – | – | – | ● | – |
| `asr` | ● | – | – | – | – | ● | – |
| `general_ability` | ● | ● | – | – | – | ● | – |
| `conditioned` | ● | ● | – | – | – | ● | – |
| `scenarios` | ● | ● | ● | ● | – | ● | ● |

Notes:
- `concurrency` / `stability`: require sustained throughput; skip on CPU-only baselines and
  low-RAM targets.
- `scenarios`: L2 judge must not run on the same target machine as the model under test during
  target-local single-model runs. Use L1-only or an external/independent judge.
- `embedding` / `rerank` / `asr`: require dedicated model capabilities; only run when the model
  is `embedding_capable` / `rerank_capable` / `asr_capable` in `models.yaml`.
- `general_ability`: requires HuggingFace dataset download; skip on air-gapped boards.

---

## Appendix D — Troubleshooting Matrix

### D.1 SSH connectivity

| Symptom | Diagnosis | Fix |
|---|---|---|
| `ssh: connect to host ... port 22: Connection refused` | SSH server not running | Start OpenSSH Server on the target; check firewall |
| `Permission denied (publickey,password)` | Wrong credentials or user format | For Windows Microsoft accounts use `email@domain` format |
| `sshpass: invalid option` | sshpass version too old | Upgrade via package manager |
| SSH hangs after ~30 s | Network idle timeout | Add `-o ServerAliveInterval=30 -o ServerAliveCountMax=3` |
| `Host key verification failed` | Known-hosts mismatch | `ssh-keyscan $HOST >> ~/.ssh/known_hosts` or add `-o StrictHostKeyChecking=no` for trusted LAN |

### D.2 GPU / accelerator not recognized

| Platform | Symptom | Fix |
|---|---|---|
| AMD Windows | `0 GPU layers` in Ollama | Set `HSA_OVERRIDE_GFX_VERSION=gfx1102` before `ollama serve` |
| AMD Windows | Low throughput despite GPU | Run `ollama ps`; stop and fix acceleration unless this is an explicit CPU baseline |
| RK3588 | NPU `cur_freq` = 0 | Send one inference request to wake the NPU; check devfreq governor |
| Jetson | CUDA not detected | Verify CUDA toolkit installed; `nvidia-smi` should show the Jetson GPU |
| Intel Arc | No acceleration | Arc support via OpenVINO is experimental; fallback to CPU |

### D.3 Out-of-memory errors

| Platform | RAM | Symptom | Fix |
|---|---|---|---|
| AMD Windows | 17.9 GB shared | `Error: model too large` | Use Q4_K_M; max 7B models |
| RK3588 | 8 GB | Adapter OOM | Use 0.5B or 1B models; Q4 quantization only |
| K3 RISC-V | 4 GB | llama-server killed | Use 0.5B Q4_K_M; avoid 3B+ |
| Intel (CPU) | Variable | Swap thrashing, very slow | Reduce `max_model_len`; use smaller model |

### D.4 Model load timeout / hang

| Symptom | Likely cause | Fix |
|---|---|---|
| `Timeout (600s)` during `accuracy` | Model still loading | Increase `wait_for_server` timeout in `models.yaml`; check server logs |
| Benchmark exits immediately with exit 2 | Endpoint unreachable | Run `python3 scripts/probe_provider.py --model <name>` first |
| RKNN adapter `Port already in use` | Previous instance still running | `sshpass ... ssh $RK3588_USER@$RK3588_HOST "pkill -f rknn_adapter.py"` |
| vLLM `CUDA error: no kernel image` | CUDA / driver mismatch | Rebuild vLLM against installed CUDA version; check `nvcc --version` |

### D.5 Benchmark result anomalies

| Symptom | Cause | Fix |
|---|---|---|
| All quality dims show `WARN` | Synthetic provenance data | Add curated cases via `scripts/curate_scenario_case.py` |
| `INCONCLUSIVE` from `--compare` | Single-seed runs | Re-run both models with `--seeds 3` |
| Reports from different targets not comparable | `hardware_profile` mismatch | Performance side forced to `INCONCLUSIVE`; this is correct behavior |
| `BLOCKED` on `general_ability` | HF dataset unreachable | Set `TRANSLATION_OFFLINE=1` or `--skip general_ability` on air-gapped targets |
