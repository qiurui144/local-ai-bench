# vlm-llm-benchmark — AI 工作指令

> 全局规范见 `/home/qiurui/.claude/CLAUDE.md`(§3.1 spec-first / §6.3 baseline 诚信 / §7.2 RC 四节门)。本文件只放项目特定约束。

## 北极星

**「模型 X 放进 AI-box 产品,能不能替换 Y?性能够不够、效果掉不掉?」**

定位:性能 × 模型效果双轴综合测试平台,一条命令出可替换性结论(`--compare` → REPLACEABLE / NOT_REPLACEABLE / INCONCLUSIVE)。任何新维度 / 重构先问是否服务这句话。

## 大改三问(任一不确定 → 停,先写 spec)

1. 服务北极星吗?
2. 在追学术指标牺牲产品吗?
3. 偏离单机 vLLM / OpenAI-compatible 端点 / 边缘硬件约束吗?

## 13 维度地图(`run_benchmark.py::DIMENSIONS` 注册表 = 唯一接线点)

| 维度 | quality(进 exit code) | gate(能力门) | --skip key |
|---|---|---|---|
| accuracy | ✅ | 总是 | `accuracy` |
| ttft | — | 总是 | `ttft` |
| throughput | — | 总是 | `throughput` |
| prefill_decode | — | 总是 | `prefill_decode` |
| concurrency | — | 总是 | `concurrency` |
| stability | — | 总是 | `stability` |
| translation | ✅ | `translation_capable` | `translation` |
| embedding | ✅ | `embedding_capable` | `embedding` |
| rerank | ✅ | `rerank_capable` | `rerank` |
| asr | ✅ | `asr_capable` | `asr` |
| general_ability | ✅ | chat-capable | `general_ability` |
| conditioned | ✅ | chat-capable | `conditioned` |
| scenarios | ✅ | chat-capable | `scenarios` |

新维度 = `benchmark/<dim>/` 包 + `DIMENSIONS` 里一个 `DimensionSpec`(run/gate/render)+ `models.yaml::benchmarks.<dim>` 块 + `benchmark/report/sections.py` render hook + 测试。`QUALITY_DIMS` 由表派生,不手写。

## 关键纪律

- **空跑不得 PASS**:零实测的整轮 exit 2;点名 `--model X` 出 error exit 2。
- **合成数据封顶 WARN**:`synthetic` provenance / 离线 Flores fallback / `synthetic_fallback` 数据集永不产出 PASS;general_ability 数据缺失或合成 → BLOCKED。
- **BLOCKED 语义**:前置缺失 ≠ 通过。BLOCKED 计为 WARN(exit 1),绝不静默 0。
- **provenance 强制**:scenarios 每 case 记 provenance(synthetic 封顶 WARN,curated/dataset 才解锁 PASS);数据源记入报告 `dataset_sources`。
- **单 seed 排名是噪声**:质量结论必须 `--seeds 3` 报 mean±std;`--compare` 对单 seed 数据硬封顶 INCONCLUSIVE,不可配置绕过。
- **报告 schema v1**:每份报告带 `schema_version` / `harness_version` / `hardware_profile` / `condition`;`--compare` 拒比 legacy(无 schema_version)、不同 harness_version / condition 的报告;hardware_profile 不一致 → 性能侧强制 INCONCLUSIVE。
- **`benchmark/llama_benchmark/` 是 library 不是 harness**:只许适配层(`benchmark/general_ability/backend_adapter.py`)单向消费,不复制实现、不双向 import。
- **`benchmark/rag/` 是教学/方法论轨**,不进 verdict 链。
- **阈值校准**:general_ability / conditioned 初始阈值未经实测校准,首跑后变更记 RELEASE.md。

## 测试原则（铁律，违反即停）

### 串行原则 — 禁止并发 benchmark

**触发**：任何"同时跑两个模型 / 两个维度 / 两个平台"的想法。

**硬约束**：
- **性能测试（ttft / throughput / prefill_decode / concurrency / stability）** 必须 GPU 独占，任何并发都使数据失真，结论作废
- **质量测试（general_ability / translation / embedding / rerank / asr）** 并发虽不影响 accuracy，但会大幅拉长耗时、提高超时风险、增大 seed 间方差
- **LLM/VLM 默认禁止 CPU-only 测试**：除非用户明确要求 CPU baseline / 特殊诊断，否则不启动纯 CPU LLM/VLM；CPU-only 结果必须标注为 baseline，不进入常规模型选型/性能结论
- **scenarios 禁止同机自动加载第二模型做 L2 judge**：target-local 单模型测试默认 L1-only；L2 judge 必须来自独立硬件/外部服务，否则违反单机单模型原则
- **唯一例外**：不同目标机可并行（AMD 机和 Intel 机同时跑，互不影响）

**正确顺序**：同一台测试机上，一个模型跑完 → 进程退出 → 再启动下一个。

**不同测试机之间可以并行**：AMD / Intel / RK3588 / K3 各自独立硬件，互不影响，可同时跑。

### 目标机交互模式 — "投递后脱离"

**触发**：任何在目标机上启动 benchmark / 验证脚本的操作。

**硬约束**：
1. **传脚本和物料** → `scp` 到目标机，确认文件到位
2. **后台启动** → Windows 用 `Start-Process ... -WindowStyle Hidden`（脱离 SSH 会话），Linux 用 `nohup ... &`；**禁止前台 SSH 阻塞执行**
3. **断开连接** → SSH 可断，进程独立运行，不依赖 SSH 保持
4. **判断依据只有日志和报告文件**：
   - 目标机自身输出的 `.md` / `.json` 报告 = 唯一真相
   - 报告未生成 = **我方脚本有问题**（参数错误 / 路径错误 / 依赖缺失），不是目标机问题
   - 禁止凭 SSH 返回值 / exit code 判断 benchmark 结果
5. **轮询检查**：`dir output/reports | findstr <model>` 检查报告文件是否生成；`tasklist | findstr python` 确认进程是否存活

**反模式（违反即停）**：
- SSH 前台阻塞运行 benchmark（断网 = 进程死）
- 报告未生成就说"可能是网络问题"（先查脚本）
- 并发启动两个模型的 benchmark 节省时间
- 凭 SSH 连接成功判断进程存活（要用 `tasklist` 实查）

### Thinking 模型在边缘设备上的使用原则

**触发**：为 AMD / Intel / RK3588 等边缘设备选模型时。

**结论（已实测确认，2026-06-23）**：
- **禁止在边缘 iGPU 上使用 thinking 模型**（Qwen3 系列默认 thinking ON）
- AMD 780M / Intel Arc：10–15 TPS，每道 MCQ 题生成 300–800 think tokens → 每题 30–60s → **交互场景完全不可用**
- `think=false` 不是解决方案：AMD Qwen3-1.7B think=false 实测 GA 仍 FAIL（MCQ 格式遵从问题）；Intel Qwen3-4B think=false 未经测试但性能改善有限
- **边缘设备 LLM 首选**：非 thinking 模型（Qwen2.5 系列 / LLaMA 系列），实测 TTFT < 2s

## 常用命令

```bash
python3 -m pytest tests/ -q        # 579+ 全离线,无 GPU / 无端点
ruff check .                       # 仓库 ruff-green,保持
python run_benchmark.py --model qwen3-vl-8b-instruct --seeds 3 --skip stability
python run_benchmark.py --compare qwen2.5-vl-7b-fp16 qwen3-vl-8b-instruct
python run_benchmark.py --model all --skip general_ability,conditioned,scenarios
```

## 指针

- 开发者上手 / 架构:[DEVELOP.md](DEVELOP.md);版本史 SSOT:[RELEASE.md](RELEASE.md)
- 设计 spec:`docs/superpowers/specs/`(本定位:`2026-06-11-platform-positioning.md`)
- 架构评审:`reports/2026-06-11-architecture-review.md`
- 贡献指南:[docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)

---

## drivers/ — 软件栈 & 模型文件清单（本地 HDD，不入 GitHub）

> `drivers/` 是 `/mnt/hdd/vlm-llm-benchmark/drivers/` 的软链接（约 37 GB，2026-06-23）。
> 内容不进 git（已 `.gitignore`）。第三方仓库（如 attune-k3、attune-pro）通过此路径调用软件栈和模型文件。
> **原则**：`drivers/` 仅存放驱动、SDK、模型文件、验证脚本；**benchmark 报告不放此处**（→ `benchmark-runs/`）。

### 目录结构

```
/mnt/hdd/vlm-llm-benchmark/
├── drivers/           # 驱动 / SDK / 模型文件（仅物料，不放报告）
│   ├── amd-win/      # AMD Windows 平台驱动 & SDK（14 GB）
│   │   ├── NPU_RAI1.5_280_WHQL.zip       # RyzenAI NPU 驱动 v1.5 (WHQL 认证)
│   │   ├── NPU_RAI1.6.1_314_WHQL.zip     # RyzenAI NPU 驱动 v1.6.1 (WHQL 认证，推荐)
│   │   ├── ryzen-ai-lt-1.7.1.exe         # RyzenAI SDK v1.7.1 安装包
│   │   ├── RAI_1.7.1_Linux_NPU_XRT.zip   # RyzenAI Linux NPU XRT 包
│   │   ├── ryzen_ai-1.7.1.tgz            # RyzenAI SDK tarball
│   │   ├── ort_models/                   # ORT+DirectML 推理模型（389 MB）
│   │   │   ├── embedding/                # BGE-base-en-v1.5 INT8 ONNX（3-seed PASS 2026-06-23）
│   │   │   └── reranker/                 # BGE-reranker-base INT8 ONNX（3-seed PASS 2026-06-23）
│   │   ├── ov_models/                    # AMD VitisAI NPU 专属模型（空，VitisAI EP 不可用）
│   │   └── scripts/                      # AMD 验证脚本（npu_validate_amd.py / serve_ort_extras_amd.py）
│   │
│   ├── intel-win/    # Intel Windows 平台 SDK & 模型（15 GB）
│   │   ├── ov_models/
│   │   │   ├── llm/                      # OpenVINO INT4 量化 LLM（14 GB）
│   │   │   │   ├── qwen2.5-1.5b-int4-ov/ # Qwen2.5-1.5B INT4（GA PASS: 34TPS/192ms P50 TTFT）
│   │   │   │   ├── qwen2.5-7b-int4-ov/   # Qwen2.5-7B INT4（GA PENDING，port 8085 运行中）
│   │   │   │   ├── qwen3-0.6b-int4-ov/   # Qwen3-0.6B INT4（小参考模型）
│   │   │   │   ├── qwen3-1.7b-int4-ov/   # Qwen3-1.7B INT4
│   │   │   │   └── qwen3-4b-int4-ov/     # Qwen3-4B INT4（GA FAIL 3-seed 2026-06-24）
│   │   │   ├── embedding/                # BGE-base-en-v1.5-int8-ov（3-seed PASS 2026-06-23）
│   │   │   ├── reranker/                 # BGE-reranker-base-int8-ov（3-seed PASS 2026-06-23）
│   │   │   ├── asr/                      # whisper-base-int8-ov（FAIL CER=54%, RTF=3.64）
│   │   │   └── ocr/                      # PP-OCR（空）
│   │   ├── scripts/                      # Intel 验证脚本（serve_ov_intel.py / serve_ov_extras.py / npu_validate_intel.py）
│   │   └── sdk/                          # Intel SDK（按需补充）
│   │
├── rk182x-linux/     # RK1820/1828 Linux 平台 SDK & 模型（8 GB）
│   ├── RK1820_RK1828_AI_SDK_V1.0.0.tgz  # SDK v1.0.0 tarball（存档）
│   ├── RK1820_RK1828_AI_SDK_V1.0.4.tgz  # SDK v1.0.4 tarball（当前稳定版）
│   ├── RKNN3_SDK/
│   │   ├── datasets/                 # 评测数据集（gsm8k / cmmlu / mmbench）
│   │   │   └── v1.0.5b2/            # runtime beta tarballs（rknn3-runtime.tgz / toolkit.tgz）
│   │   ├── rknn3_models/v1.0.4/     # RKNN3 Model Zoo (v1.0.4)
│   │   │   ├── llm/CoPaw-flash-4B/  # CoPaw-flash-4B (8k/32k ctx, .rknn + .embed.bin + .gguf)
│   │   │   ├── llm/Qwen3-1.7B/      # Qwen3-1.7B（.ldtmp 格式）
│   │   │   ├── vlm/                 # VLM 模型
│   │   │   ├── cnn/                 # CNN 模型
│   │   │   ├── others/              # 其他模型
│   │   │   └── yolo/                # YOLO 系列
│   │   └── v1.0.4/rknn3-toolkit-1.0.4/  # rknn3-toolkit Python 包
│   └── docs/                        # 官方 PDF 文档（Quick Start / Release Notes）
│
└── rk3588-linux/     # RK3588+RK182X 平台（192.168.100.206，2026-06-24 上线）
    ├── rknn3_models/   # RKNN3 模型（从 rk182x-linux/RKNN3_SDK/rknn3_models/ 复用）
    └── scripts/        # RK3588 专属验证/benchmark 脚本
```

### 平台 → SDK 版本对应

| 平台 | 当前 SDK 版本 | 关键文件 |
|---|---|---|
| AMD Windows (Ryzen 8845H) | Ollama 0.30.8（100% iGPU Radeon 780M Vulkan）+ ORT 1.x + DirectML；RyzenAI SDK v1.7.1 + NPU 驱动 v1.6.1（NPU仅CNN，LLM不支持） | `amd-win/ryzen-ai-lt-1.7.1.exe` |
| Intel Windows (Core Ultra 7 155H) | OpenVINO 2026.2.1 + optimum-intel 2.0.0（OVModelForCausalLM device=GPU Arc ✓）；openvino-genai DLL broken；serve_ov_extras.py port 8081（embedding+reranker+asr） | `intel-win/ov_models/llm/` |
| RK1820/1828 Linux | RKNN3 SDK v1.0.4（stable），v1.0.5b2（beta runtime）| `rk182x-linux/RK1820_RK1828_AI_SDK_V1.0.4.tgz` |
| **RK3588+RK182X Linux** (192.168.100.206) | RKNN3 runtime 已装（librknnrt.so / librknn3_api.so）；rknn-smi v? 已装；rknn_toolkit_lite2==2.3.2（在 /userdata/model_hub/embedding/.venv）；RK1828 via PCIe（Product: RM1828SA0-F, Serial: R1BCA260200353）；Python 3.11.2；**rknn Python toolkit 未全局安装** | 需 sudo 运行 rknn-smi；/ 分区 93% 满（剩 468M）→ 数据放 /userdata（1.2G 可用）|

> **AMD 模型选型**（2026-06-22）：Ollama Vulkan iGPU 路径使用标准 GGUF Q4 模型。DirectML 路径（ORT）用于 embedding/reranker/OCR（serve_ort_extras_amd.py port 8091）。AMD VitisAI EP 未安装（vitisai_available=false）。
>
> **Intel 版本锁定说明**（2026-06-22）：openvino-genai 存在 DLL export 冲突（非版本问题）；上游 rapidocr PR 计划中（OV 2026 兼容，待安排）。

### NPU 验证脚本（`scripts/`）

| 脚本 | 用途 |
|---|---|
| `scripts/npu_validate_intel.py` | Intel NPU 全面验证：probe / embedding / reranker / ocr / asr / llm_npu |
| `scripts/npu_download_intel.py` | Intel NPU 验证模型下载（hf-mirror）：bge / reranker / whisper |
| `scripts/npu_validate_amd.py` | AMD NPU/iGPU 全面验证：probe / embedding / reranker / ocr / asr |

**运行前提（Intel）**：optimum[openvino] 已安装，`npu_validate_intel.py --task probe` 看到 `NPU: Intel(R) AI Boost`。
**运行前提（AMD NPU）**：RyzenAI SDK 1.7.1 从 `drivers/amd-win/ryzen-ai-lt-1.7.1.exe` 安装，VitisAI EP 出现在 `ort.get_available_providers()`。

### benchmark-runs/ — 原始评测报告（与 drivers/ 同级）

```
benchmark-runs/
├── amd-win/           # 从 AMD 机 output/reports/ 同步的原始 .json/.md/.html
└── intel-win/         # 从 Intel 机 output/reports/ 同步的原始 .json/.md/.html
```

不进 git（HDD only）。平台报告汇总见 `reports/<platform>.en.md`（git 追踪）。

### 新增软件栈 / 模型文件的规范

1. **按平台分目录**：`amd-win/` / `rk182x-linux/` / `rk3588-linux/` / `intel-win/` / `k3-riscv/`（如需新建）
2. **按类别分子目录**：驱动放根，SDK 放 `<SDK名>/`，模型放 `models/` 或 `rknn3_models/`
3. **更新此清单**：每次新增文件同步更新本节，标明版本和用途
4. **大文件不进 git**：`drivers/` 整个目录已 `.gitignore`；本 `CLAUDE.md` 也不进 git

### 目标机访问路径

| 机器 | 访问方式 | 备注 |
|---|---|---|
| AMD Windows `192.168.100.201` | SSH 文件传输 | `happyqiurui@163.com` / `qr@1205RI` |
| Intel Windows `192.168.100.116` | SSH 文件传输 | `happyqiurui@163.com` / `qr@1205RI` (Microsoft 账户格式) |
| RK3588+RK182X Linux `192.168.100.206` | SSH rsync | `linaro` / `linaro` |
| K3 RISC-V `192.168.100.215` | SSH rsync | `root` / `bianbu` (旧 IP 140 已废弃，2026-06-21 更新) |
