> [English](./README.md)

# local-ai-bench

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![CI](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml/badge.svg)](https://github.com/qiurui144/local-ai-bench/actions/workflows/ci.yml)

一个可复现的 **性能 × 模型质量综合评测平台**，用于 AI-box 模型选型。通过 **vLLM**（或任意 OpenAI 兼容端点）对 **VLM**（视觉语言）和 **LLM**（纯文本）模型进行 **13 个注册维度** 的评测——性能轴涵盖延迟、吞吐量、并发与稳定性；模型质量轴涵盖准确率、翻译、embedding/rerank 检索、ASR、**通用能力**（gsm8k / mmlu / hellaswag）和真实场景质量；另有 **conditioned** 能力曲线（上下文长度梯度 + prefix-cache 冷/热状态 A/B）——并附带面向生产部署的完整 **RAG / LLM 验证框架**。

平台围绕以下核心问题构建：*"模型 X 能否替换生产中的模型 Y——性能够不够、质量掉不掉？"* `--compare baseline candidate` 将已保存的报告转化为自动化的 **REPLACEABLE / NOT_REPLACEABLE / INCONCLUSIVE** 判定，附带逐指标 2σ 证据；`conditioned` 维度将能力表达为**条件曲线**（输入长度、缓存状态、硬件归因报告），而非单一数据点。

📋 版本历史：[RELEASE.md](RELEASE.md) · 🛠 开发指南与架构：[DEVELOP.md](DEVELOP.md)

---

## 验证框架（v0.2 新增）

除 vLLM harness 外，本仓库现在在 `benchmark/rigor/` 和 `benchmark/rag/` 下提供完整的学术级验证框架，实现了：

- **严谨性基础**（`benchmark/rigor/`）：统计检验、效应量、多 seed 运行器、可复现性快照、概率校准（ECE/Brier/Platt/Isotonic）、评分者一致性（Cohen/Fleiss/Krippendorff）、消融实验编排器、交叉验证、功效分析，以及 OOD/子组评估。
- **RAG 方法论**（`benchmark/rag/`）：RAG 评测 playbook 全部 12 章——组件流水线追踪、离线/在线对齐、检索指标（NDCG/MRR/MAP/bpref/ERR/RBP）、重排序器评估 + 排名融合（RRF/Borda/CombSUM/CombMNZ）、答案相关性、基于 RAGAS strict faithfulness 的声明级接地性、G-Eval CoT LLM 裁判提示、裁判校准含位置/冗长/自我偏好偏差检测、裁判攻击防护（注入、泄露、扰动）、含 flake 控制器的回归 CI、含回滚策略的金丝雀发布，以及含 PSI / JS 散度 / 时间队列 / 自动策展的漂移检测。
- **附录**：`benchmark/rag/{schemas,rubrics,labs}/` 中的 3 个 JSON schema、5 个 YAML rubric 和 8 个可运行实验室；`docs/case-studies/` 中的 6 个生产案例研究；另有 120 题面试题库和一份 capstone 系统设计文档（位于 `docs/`）。

配套文档：

- `docs/ACADEMIC-RIGOR.md` — 框架执行的 12 项原则。
- `docs/BASELINES.md` — 参考基线与阈值默认值。
- `docs/REPRODUCIBILITY.md` — 版本锁定策略与快照格式。
- `docs/CITATION.md` — 引用本工作及底层方法的 bibtex。
- `docs/CONTRIBUTING.md` — 贡献指南（harness + 方法论）。
- `docs/capstone-system-design.md` — 端到端参考架构。
- `reports/2026-06-19-all-model-matrix-results.en.md` 和 `reports/2026-06-19-all-model-matrix-results.json` — 最新完整 Windows 模型矩阵、评测证据摘要与模型选型指南。
- `reports/2026-06-19-git-readiness.en.md` — 推送前质量、安全与开源准备度检查清单。

运行验证测试：

```bash
python -m pytest tests/rigor tests/rag -q
```

以实验室为例运行：

```bash
python -m benchmark.rag.labs.lab2_retrieval_metrics
python -m benchmark.rag.labs.lab4_groundedness_audit
python -m benchmark.rag.labs.lab8_drift_detection
```

原有的 `algo-base/llama-benchmark` 数据集基础设施已并入 `benchmark/llama_benchmark/`。

---

## 评测维度

| 维度 | 内容 | 重要性 |
|---|---|---|
| **Accuracy** | 分类精度、实体召回、事实召回，以及针对 golden set 的 **must-not-say 违规**检测 | 能捕捉纯困惑度无法发现的数字偏移错误（如 ¥120 vs ¥1200） |
| **TTFT** | 首 token 延迟 P50 / P95（流式输出） | UX 基线——超过 2s 体验明显变差 |
| **Throughput** | 持续负载下的聚合 tokens-per-second | 容量规划 |
| **Concurrency** | 在 1 / 5 / 10 / 30 / 50 并发请求下的成功率 + P50/P95 | 生产负载形态 |
| **Stability** | 30 分钟持续运行；前 5 分钟与后 5 分钟的延迟漂移 | 内存泄漏、KV-cache 抖动 |
| **Token budget** | 输入/输出 token 分布 + 截断率（在 accuracy 维度内测量） | 成本监控 + 静默截断检测 |
| **PP / TG split** | 分别测量 Prefill (PP) vs Decode (TG) tokens-per-second（llama-bench 风格） | 聚合吞吐量掩盖了两种不同的硬件瓶颈——prefill 受计算限制，decode 受带宽限制 |
| **Translation** | zh↔en 机器翻译质量（SacreBLEU / chrF / COMET）+ 逐语言对延迟，覆盖 3 个任务层级 | 验证模型的双语部署就绪度 |
| **Embedding** | 检索 recall@k / MRR / nDCG@10 + 单次查询延迟 P50（常驻态）+ RSS 双区分 + 数值验证 | AI-box RAG 检索器的核心——质量、真实 chat 查询延迟/内存，以及零向量/NaN 向量检测门 |
| **Rerank** | 独立重排序器 nDCG@10 / MRR + 逐对延迟（区别于 RAG 内部重排序器） | 二阶段重排质量与延迟成本的权衡（实时 vs 离线） |
| **ASR** | 基于音频 manifest 的中文 CER / WER / RTF（ONNX 后端） | 语音转录准确率 + 实时能力 |
| **General ability** | 通过版本锁定的 HF 数据集测试 gsm8k（数学推理）/ mmlu（知识，4 科目）/ hellaswag（常识）准确率，复用通过进程内适配器接入的 `llama_benchmark` 数据集。HellaSwag 以选项字母准确率（A–D）通过 chat API 评分（确定性近似，**非**长度归一化 logprob——方法已记入报告）。无法访问或合成回退数据集 → `BLOCKED`，不产出虚假分数 | 在 golden set 上表现优秀的模型，可能在通用推理能力上相比被替换模型有所退步 |
| **Conditioned** | 能力表达为**曲线**，而非单点：任务质量 + needle 召回 + TTFT/TPS 跨上下文长度梯度（1k → 32k，受模型最大长度截止）+ prefix-cache 冷/热 A/B（TTFT 加速 + 输出一致性检查——缓存绝不允许改变答案） | "在 1k token 下表现良好"无法说明 16k 的情况；缓存热状态的演示掩盖了冷启动延迟 |
| **Scenarios** | 8 个真实场景任务（S1-S8）：微信截图意图识别（VLM）、案例逻辑矛盾检测、文章知识评级、**指令跟随**、**结构化提取**、**函数调用**、**VLM 文档提取**、**对抗稳定性**——L1 确定性 + L2 多 seed LLM 裁判含锚点校准。265 个精选用例。 | 测量标准化评测套件无法覆盖的内容：模型在产品真实输入分布上的行为 |
| **Conversation drift** | 在 0 / 5 / 10 / 20 轮历史对话（填充语料）下的质量曲线。最大下降 > 15% → DRIFT/FAIL。长会话部署的前提条件。 | "在 1k token 下表现良好"无法说明第 50 轮的退化情况 |

通过/警告/失败由 `golden/expectations.json::*_acceptance_criteria` 和 `models.yaml::benchmarks.*.thresholds` 中的阈值决定——退出码 `0` PASS / `1` WARN / `2` FAIL，可直接用于 CI 消费。指定模型（`--model <name>`）出错（如端点未就绪）时退出码为 `2`；在 `--model all` 模式下，下线模型按"运行已启动的模型"的约定被跳过——但如果**零个**模型产生任何测量结果，则运行退出码为 `2`：空运行不能报告成功。质量维度报告 `BLOCKED`（前置条件缺失——如 `general_ability` / `conditioned` 数据集在离线主机上不可访问）计为 WARN（退出码 `1`），绝不静默通过；`--skip general_ability,conditioned` 可恢复 v0.3 之前的默认运行行为。

**可替换性判定（`--compare BASELINE CANDIDATE`）：** 北极星问题可从已保存报告中获得自动化答案（离线——无需重新运行）：

```bash
python run_benchmark.py --model qwen2.5-vl-7b-fp16   --seeds 3 --skip stability
python run_benchmark.py --model qwen3-vl-8b-instruct --seeds 3 --skip stability
python run_benchmark.py --compare qwen2.5-vl-7b-fp16 qwen3-vl-8b-instruct
```

判定与退出码：`REPLACEABLE`（0）/ `INCONCLUSIVE`（1）/ `NOT_REPLACEABLE`（2），逐指标 Δ / σ / 显著性证据写入 `output/reports/compare_*.{json,md}`。判定纪律硬编码，不可配置：**REPLACEABLE** 要求所有共享质量指标均在 2σ 内**且**候选模型通过自身性能阈值；任何显著质量退步 → **NOT_REPLACEABLE**；**单 seed 数据上限为 INCONCLUSIVE**（单次运行的排名是噪声——请先生成 `--seeds 3` 数据）。来自不同 `harness_version` 或 `condition` 的报告将被拒绝，不含 `schema_version` 的遗留报告将被拒绝，`hardware_profile` 不一致时性能侧强制为 INCONCLUSIVE（质量侧仍会比较）——每份报告都携带 schema-v1 信封（`schema_version` / `harness_version` / `hardware_profile` / `condition`）以支持这些检查。

**HTML 可视化报告：** 每次评测运行都会在 JSON/Markdown 旁自动生成独立的 `output/reports/<model>_<ts>.html`，可在任意浏览器中查看。包含质量雷达图（9 轴）、性能表格、逐场景柱状图和对话漂移折线图。`--compare` 模式生成 `compare_*.html`，含并排雷达图叠加和 REPLACEABLE/INCONCLUSIVE/NOT_REPLACEABLE 判定徽章。

**多 seed 运行（`--seeds N`，默认 1）：** 采样型 LLM 的单次运行质量数字存在统计噪声——排名可能在两次运行间翻转，因此单个数字不构成断言（CLAUDE.md §2.3：报告 mean ± std，不能只报单个分数）。使用 `--seeds N` 时，完整评测套件对每个模型重复运行 N 次，报告中增加顶层 `multi_seed` 块（`n_seeds` 以及通过 `benchmark/rigor/multi_seed_runner.aggregate` 对所有 N 次运行中存在的数值质量指标计算的 `mean`/`std`/`ci95_lower`/`ci95_upper`），在 Markdown 报告中渲染为"Multi-seed"章节。诚实说明：v1 **不**锁定每次调用的采样 seed——方差来源是模型在相同重复运行中的温度噪声；退出码判定取 N 次运行中的**最差**判定（任一次运行 FAIL 即为 FAIL——判定不做平均）。

---

## 翻译场景（`benchmark/translation/`）

评测任意 LLM（服务 OpenAI 兼容端点）在 zh↔en 语言对上的**机器翻译质量与延迟**。通过 `models.yaml` 中的 `translation_capable: true` 标志按模型启用；是标准 `run_benchmark.py` 流程中的一个维度（通过 `--skip translation` 跳过）。

### 指标

| 指标 | 包 | 计算 | 备注 |
|---|---|---|---|
| **SacreBLEU** | `sacrebleu` | CPU | 语料 BLEU，可复现分词（中文目标语使用 `zh` 分词器）。包缺失时回退纯 Python 实现。 |
| **chrF** | `sacrebleu` | CPU | 字符 n-gram F 分——无需分词，适合中文的稳健指标。 |
| **COMET** | `unbabel-comet` | **推荐 GPU** | 神经质量估计（`Unbabel/wmt22-comet-da`）。无 CUDA GPU 或包缺失时自动跳过并提示 `"COMET requires GPU/DGX"`——不会导致运行崩溃。 |
| **术语匹配率** | — | CPU | 必需术语表的 L3 精确匹配率。 |

每个指标都经过数值验证（假设非空、`0 ≤ BLEU/chrF ≤ 100`、有限非 NaN/Inf），使静默失败的模型以 FAIL 而非似是而非的数字呈现。

### 任务层级

- **L1 — 单句**：直接 zh↔en 句子翻译（原始充分性 + 流畅性）。
- **L2 — 上下文一致性**：将 3–5 句段落作为整体翻译，以保证代词指代 / 时态 / 命名实体在句子边界间保持一致。
- **L3 — 术语**：领域文本，要求将必要技术术语（`prompt` / `embedding` / `向量化` 等）翻译为规范译法；以精确匹配术语率评分。

### 数据集

- **Flores-200** zh↔en（devtest 分片，默认 100 句子子集）——运行时从非封控纯 parquet HF 镜像 [`haoranxu/FLORES-200`](https://huggingface.co/datasets/haoranxu/FLORES-200) 拉取（ALMA 论文的评测镜像；1012 个 devtest 句子），锁定到 commit SHA。无 `trust_remote_code`，无 auth token——纯数据，无上游代码执行。通过 `FLORES_DATASET=<repo>` / `FLORES_REVISION=<commit-sha>` 覆盖。（`facebook/flores` 为封控且基于脚本——在 `datasets>=3` 上无授权 token 时无法加载。）离线/气隙主机回退到内置的小型合成集（设置 `TRANSLATION_OFFLINE=1` 强制使用）；回退行为会明显记录、写入报告的 `dataset_sources`，并将翻译判定上限设为 `WARN`——合成分数不会伪装成 Flores-200 结果。
- **自定义产品领域语料**——`datasets/translation/custom_zh_en.jsonl`（约 60 条手工编写的合成对，无 PII，覆盖 AI 基础设施 / 工程 / 支持领域；L3 术语表内联）。使用相同 JSONL schema（`{src, tgt, domain, glossary}`）替换为您自己的审核语料。

### 使用方法

```bash
# 仅运行翻译维度，使用 LLM 主力模型
python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,concurrency,stability

# 强制使用离线 Flores 回退（气隙主机）
TRANSLATION_OFFLINE=1 python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,concurrency,stability
```

阈值位于 `models.yaml::benchmarks.translation.thresholds`（`bleu_min`、`chrf_min`、`term_match_rate_min` 等）；golden 用例位于 `golden/expectations.json::translation_cases`。仅 CPU 测试（无需 vLLM/GPU）：

```bash
python -m pytest tests/translation -q
```

---

## 真实场景维度（`benchmark/scenarios/`）

三个来自产品真实输入分布的任务，是标准 `run_benchmark.py` 流程中的一个维度（通过 `--skip scenarios` 跳过）：

- **S1 `wechat_intent`**（VLM）——读取合成的微信风格聊天截图，提取消息内容并分类聊天意图（8 个意图标签）。
- **S2 `case_logic`**（LLM）——在案例叙述片段中发现矛盾（`time_conflict` / `causal_break` / `fact_mismatch`）以及整体一致性标签。
- **S3 `article_knowledge`**（LLM）——将自媒体文章声明判定为 `accurate` / `inaccurate` / `unverifiable`，并给出 A–D 知识等级。

**评分**采用双层结构：**L1** 确定性指标（针对用例标签的准确率 / 召回率 / F1）+ **L2** LLM 裁判，以多 seed（N=3）运行，含来自 `golden/scenarios.json` 的配对锚点校准。裁判模型在 `models.yaml::benchmarks.scenarios.judge_model` 中设置（`null` 时自动选择），**强制与被测模型不同**；`num_cases` 限制每个场景的用例数量。

**数据采用双轨制**，provenance 逐用例记录于 `datasets/scenarios/<scenario>/cases.jsonl`（每个场景 5 个 seed 用例）：

- **合成数据**——`scripts/render_wechat_case.py`（PIL 渲染器）生成截图用例。`provenance: "synthetic"` 将场景判定上限设为 `WARN`——纯合成数据不会产生 PASS。
- **精选数据**——`scripts/curate_scenario_case.py` 接收经过审核的真实世界用例（`provenance: "curated"` / `"dataset"`），解锁完整 PASS 判定。

`scripts/check_no_real_images.sh` 是图片 fixture 的 PII 控制措施：`fixtures/scenarios/` 下的每个 png 必须以 `dialogs.json` 渲染器 id（provenance 白名单）命名，`cases.jsonl` 中引用的每个 `payload.image` 必须存在——真实截图不得进入 git。

缺失 `cases.jsonl` 会使场景进入 `BLOCKED` 状态（计为 WARN，绝不虚假 PASS）。

**注意：** 开箱即用时，此维度每个场景附带 5 个合成 seed 用例，因此默认运行**设计上**报告 `WARN`（退出码 1），直到通过 `scripts/curate_scenario_case.py` 添加精选或数据集轨道用例——纯合成分数有意不产出干净的 PASS。使用 `--skip scenarios` 可恢复之前的退出行为。

---

## AI-box 核心能力（`benchmark/embedding`、`benchmark/rerank`、`benchmark/asr`）

从 K23 边缘 AI-box 评测同步的核心检索 + 语音能力，从边缘 `llama.cpp`/`ONNX` 适配到服务 **OpenAI 兼容**端点（vLLM / sglang / llama.cpp server / Ollama）。每项都是标准 `run_benchmark.py` 流程中的可选维度，通过 `models.yaml` 中的 `*_capable: true` 标志按模型启用，可跳过（`--skip embedding,rerank,asr`）。

### Embedding（`benchmark/embedding/`）

对 embedding 部署的检索质量 + 延迟/内存特征评测——RAG 流水线的检索器阶段。

| 指标 | 计算 | 备注 |
|---|---|---|
| **recall@1 / @5 / @10** | CPU（NumPy） | 逐查询：embed 查询 + 候选文档，cosine top-k，测量 gold 文档的落点 |
| **MRR** | CPU | 第一个相关文档的平均倒数排名 |
| **nDCG@10** | CPU | 归一化 DCG，二值相关性 |
| **单次查询延迟 P50** | 常驻态模型 | 针对常驻端点计时——真实 chat 查询路径（非逐进程 CLI 加载） |
| **RSS 双区分** | 本地进程 | *批次 RSS*（因逻辑批次 KV 而膨胀）vs *常驻查询 RSS*（≈ 权重 + 小 KV——真实 chat 内存）；远程端点报告 `available: false` |
| **数值验证** | CPU | NaN / Inf / **零向量** / 维度漂移检查——零向量为硬 FAIL（经典的"快但错"陷阱） |

数据集：`datasets/retrieval/cmteb_zh_subset.jsonl` 中的中文检索集（由您提供；如 C-MTEB 子集），内置中文合成回退用于离线/单元测试运行（标记为 `source="builtin"`，不会被误认为真实分数）。参见 [`datasets/retrieval/README.md`](datasets/retrieval/README.md)。

### Rerank（`benchmark/rerank/`）

**独立**重排序器评测（区别于 `benchmark/rag/reranker.py` 中的 RAG 内部重排序器）：通过服务端点对每个（查询，候选文档）对评分（是/否相关，有 logprob 时使用），重新排序，并在与 embedding 相同的检索 gold 上报告 **nDCG@10 / MRR**，以及**逐对延迟 P50** 和正例 vs 负例分数分离健全性检查。逐对延迟会报告但不设门控——高质量重排序器可以是纯离线使用。

### ASR（`benchmark/asr/`）

基于音频 manifest 的中文 **CER / WER / RTF**。CER（字符错误率）是中文的主要指标；RTF < 1.0 表示具备实时能力。转录后端为可插拔 ONNX（默认 sherpa-onnx SenseVoice）；当运行时 / 模型 / 数据集缺失时，维度报告 `blocked`，判定为 `SKIP` 而非崩溃。音频文件不随仓库附带——manifest schema 参见 [`datasets/asr/README.md`](datasets/asr/README.md)。

### 使用方法

```bash
# 仅运行 embedding 维度，使用 embedding 主力模型
python run_benchmark.py --model qwen3-embedding-0.6b \
    --skip accuracy,ttft,throughput,prefill_decode,concurrency,stability,translation,rerank,asr

# 仅 CPU 测试（无需 vLLM / GPU / 模型）——指标数学 + 数值门控：
python -m pytest tests/embedding tests/rerank tests/asr tests/performance -q
```

> **vLLM / GPU 注意事项**：指标/验证逻辑已在 CPU 上使用注入的 fake 数据完整单元测试。端到端数值（真实 embedder 上的 recall、rerank nDCG、ASR CER、PP/TG tok/s）需要服务端点 + GPU/ONNX 后端，在此**不运行**——请针对实时部署运行 `run_benchmark.py` 以生成这些数值。

---

## 参考模型矩阵（`models.yaml`）

harness 与供应商无关——任何提供 **OpenAI 兼容**端点的系统均可使用（vLLM、sglang、lmdeploy、llama.cpp server、Ollama 0.21+ 等）。开箱即用提供 **10 个参考模型矩阵**：4 个 VLM/LLM 对话模型加 6 个 embedding / rerank / ASR 模型。

| 角色 | 模型 | 量化 | 端口 | 显存 | 最低硬件 |
|---|---|---|---|---|---|
| 🌟 VLM 主力 | Qwen3-VL-8B-Instruct | BF16 | 8001 | 20 GB | A100-40G |
| 📍 VLM 基线 | Qwen2.5-VL-7B-Instruct | BF16 | 8002 | 18 GB | A100-40G |
| 🌟 LLM 主力 | Qwen3-30B-A3B-Instruct-2507-FP8（MoE） | FP8 | 9001 | 35 GB | H100-80G |
| 🌟🌟🌟 LLM 旗舰 | Qwen3-235B-A22B-Instruct-2507-FP8（MoE） | FP8 | 9002 | 240 GB | 8×H100-80G |
| 🌟 Embedding 主力 | Qwen3-Embedding-0.6B | — | 9101 | 4 GB | A100-40G |
| Embedding 高精度 | Qwen3-Embedding-4B | — | 9102 | 12 GB | A100-40G |
| Reranker（生成式） | Qwen3-Reranker-4B | — | 9201 | 12 GB | A100-40G |
| 🌟 Reranker（实时） | bge-reranker-v2-m3 | Q8_0 | 9202 | 1 GB | K3-X100（CPU） |
| Reranker（延迟下限） | bge-reranker-base | Q4_K_M | 9203 | 1 GB | K3-X100（CPU） |
| ASR 主力 | SenseVoiceSmall | INT8 | —（本地 ONNX） | 1 GB | CPU |

> 注：`scripts/prepare_offline.sh` MODEL_SET 分级（minimal/standard/full）仅覆盖 4 个对话模型；embedding / rerank / ASR 模型目前没有离线下载路径（参见 RELEASE.md 已知问题）。

**无降级设计**：如果您有 DGX 级别的硬件，请运行真实模型。旗舰 MoE 条目每次前向传播仅激活约 3B / 约 22B 参数，因此在保持质量的同时，延迟上能与小得多的密集模型媲美。

通过向 `models.yaml` 追加条目来接入您自己的模型——只需 `(name, hf_repo, port, role)` 四个必填字段；其他字段为文档提示。

---

## 快速开始

### 前置条件

- Linux（已测试 Ubuntu 22.04 或 24.04），需要 **CUDA 兼容 GPU**
- Python 3.10+
- 约 50 GB 可用磁盘空间用于默认 80 GB 模型矩阵（或 16 GB 用于最小集）

### Windows 快速开始（Ollama 或 llama.cpp）

无需 GPU 服务器——在 Windows 机器上本地运行 Ollama 或 llama.cpp，并从同一网络的任意机器指向 harness。

**步骤 1 — 在 Windows 机器上（PowerShell，一次性设置）**

```powershell
# 启用 OpenSSH 远程管理（可选但推荐）
Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0
Start-Service sshd; Set-Service -Name sshd -StartupType Automatic
New-NetFirewallRule -Name sshd -DisplayName "OpenSSH" -Enabled True -Direction Inbound -Protocol TCP -LocalPort 22 -Action Allow

# 安装 Ollama
winget install Ollama.Ollama --silent

# 拉取模型
ollama pull qwen2.5:7b

# 开放 Ollama 端口（使其他机器可访问）
New-NetFirewallRule -Name Ollama -DisplayName "Ollama" -Enabled True -Direction Inbound -Protocol TCP -LocalPort 11434 -Action Allow
```

**步骤 2 — 在开发机上，配置 models.yaml**

```yaml
models:
  - name: qwen2.5-7b-win-ollama
    provider: ollama
    port: 11434
    base_url_override: http://192.168.1.100:11434/v1   # Windows 机器 IP
    model_id: qwen2.5:7b
    task_type: text_only
    notes: "Windows Ollama — qwen2.5:7b"
```

**步骤 3 — 探测并评测**

```bash
# 验证端点可达且功能正常
python3 scripts/probe_provider.py --model qwen2.5-7b-win-ollama

# 运行评测（跳过 GPU 密集维度）
python run_benchmark.py --model qwen2.5-7b-win-ollama \
    --skip stability,embedding,rerank,asr
# → output/reports/qwen2.5-7b-win-ollama_<ts>.json + .html
```

### 供应商配置

harness 通过 HTTP 与任意 OpenAI 兼容端点通信。支持的 `provider:` 值：

| 供应商 | `provider:` 值 | 备注 |
|----------|-------------------|-------|
| 本地 vLLM | `local_vllm` | 默认。`port: 8000` |
| **llama.cpp server** | `llama_cpp` | `llama-server --port 8080 --model model.gguf` |
| Ollama | `ollama` | 支持 Linux + macOS + **Windows** |
| OpenAI 云端 | `openai` | `api_key_env: OPENAI_API_KEY`；429 自动重试 |
| DashScope | `dashscope` | `api_key_env: DASHSCOPE_API_KEY` |
| DeepSeek | `deepseek` | `api_key_env: DEEPSEEK_API_KEY`；429 自动重试 |

### 3 步部署

```bash
# 1. 在有网络的机器上——下载所有制品
git clone https://github.com/qiurui144/local-ai-bench.git
cd local-ai-bench
MODEL_SET=standard bash scripts/prepare_offline.sh
# MODEL_SET 选项：
#   minimal  (~16 GB) — 仅 VLM 主力
#   standard (~80 GB) — VLM ×2 + LLM-30B  [推荐]
#   full    (~320 GB) — 全部 4 个模型，含 235B

# 2. （可选）打包以离线传输到气隙 GPU 主机
tar czf local-ai-bench-bundle.tar.gz local-ai-bench/
scp local-ai-bench-bundle.tar.gz dgx:/data/

# 3. 在 GPU 主机上
cd /path/to/local-ai-bench
sudo bash scripts/bootstrap.sh   # 安装 vLLM，链接模型到 HF cache
bash run.sh                      # 默认：VLM 主力，跳过 30 分钟稳定性测试
```

### 针对性运行

```bash
# 用候选模型替换基线——核心"X 能替换 Y 吗？"问题
bash vllm_configs/start_all.sh   # 先在 start_all.sh 中取消注释基线
python run_benchmark.py --model qwen2.5-vl-7b-fp16  --skip stability
python run_benchmark.py --model qwen3-vl-8b-instruct --skip stability
cat output/reports/matrix_*.md

# 仅运行 LLM 并发扫描
python run_benchmark.py --model qwen3-30b-a3b-instruct-2507-fp8 \
    --skip accuracy,ttft,throughput,stability

# 旗舰 235B 冒烟测试（需要 8×H100）
python run_benchmark.py --model qwen3-235b-a22b-instruct-2507-fp8 \
    --skip concurrency,stability
```

---

## 使用您自己的数据

本仓库**设计上不附带**任何 fixture 图片——VLM 评测需要真实世界的截图 / 扫描件 / 照片，这些内容通常含有 PII。参见 [`fixtures/README.md`](fixtures/README.md) 了解：

- 图片应放置的位置（每个 `golden/expectations.json::cases[].image` 对应一张）
- 如何编写您自己的 golden-set 条目（`must_identify_entities`、`must_identify_facts`、**`must_not_say`**）
- 为什么 `.gitignore` 默认排除二进制 fixture

附带的 `golden/expectations.json` 是一个合成的 9 用例演示。请替换为您自己的 ground truth 以评测您所在领域。

---

## 仓库结构

```
local-ai-bench/
├── run.sh                    # 一行入口
├── run_benchmark.py          # 主调度器
├── models.yaml               # 模型矩阵（编辑此文件添加/移除模型）
├── common.py                 # vLLM 客户端 + 共享工具
├── requirements.txt          # httpx / pyyaml / Pillow / pynvml / pydantic / loguru / numpy / scipy / sacrebleu / pytest …
├── benchmark/
│   ├── accuracy.py           # golden-set 驱动的准确率
│   ├── performance.py        # TTFT / throughput / concurrency / stability / PP-TG split
│   ├── translation/          # zh<->en 机器翻译：SacreBLEU/chrF/COMET + 延迟（L1/L2/L3）
│   ├── embedding/            # 检索 recall@k/MRR/nDCG + 延迟/RSS + 数值验证
│   ├── rerank/               # 独立重排序器 nDCG/MRR + 逐对延迟
│   ├── asr/                  # 中文 CER/WER/RTF（ONNX 后端，优雅 BLOCKED）
│   ├── scenarios/            # 真实场景维度（wechat_intent / case_logic / article_knowledge）
│   ├── general_ability/      # 通过 llama_benchmark 适配器的 gsm8k / mmlu / hellaswag（锁定数据集）
│   ├── conditioned/          # 上下文梯度 + needle + prefix-cache 冷/热条件曲线
│   ├── registry.py           # DimensionSpec 注册表 + 共享判定语义（唯一可信源）
│   ├── report/               # 逐维度 Markdown 渲染钩子
│   │   ├── html_report.py            # 含 Chart.js 雷达/柱状/漂移图的独立 HTML 报告
│   │   └── sections.py               # 逐维度 Markdown 渲染钩子
│   ├── compare.py            # --compare 可替换性判定（2σ 纪律，退出码 0/1/2）
│   ├── rigor/                # 统计严谨性库（多 seed、效应量、校准等）
│   ├── rag/                  # RAG 验证框架（12 章 + labs/rubrics/schemas）
│   ├── llama_benchmark/      # 并入的遗留 harness（库；CLI 目前不可用——参见 RELEASE.md）
│   ├── llama_configs/        # llama_benchmark 配置（models.yaml / benchmarks.yaml / devices）
│   └── llama_baselines/      # llama_benchmark 实测基线（K1-SpacemiT 运行结果 + trend.md）
├── vllm_configs/
│   ├── launch_helpers.sh     # vllm serve 辅助函数
│   └── start_all.sh          # 批量模型启动（默认：仅 VLM 主力）
├── scripts/
│   ├── prepare_offline.sh    # 联网主机：拉取 wheels + 模型
│   ├── bootstrap.sh          # GPU 主机：安装 vLLM，链接模型
│   ├── probe_provider.py         # 端点冒烟测试（可达性 / JSON 模式 / seed / VLM）
│   ├── verify_benchmark.py       # 数据集完整性验证（通过 registry 自动检测场景）
│   ├── render_wechat_case.py # 合成微信截图渲染器（场景 S1 fixture）
│   ├── curate_scenario_case.py  # 接收经审核的真实世界场景用例（精选/数据集轨道）
│   ├── derive_cail_cases.py  # 从 CAIL2018 派生数据集轨道 S2 用例（HF 版本锁定）
│   ├── derive_cail_dialogs.py   # 从 CAIL 语料派生数据集轨道 S1 对话
│   ├── check_no_real_images.sh  # PII 控制：fixture 图片 provenance 白名单
│   └── setup_zerotier.sh     # 可选：ZeroTier VPN 用于远程部署
├── datasets/
│   ├── translation/          # zh<->en 平行语料（自定义 JSONL；Flores 运行时加载）
│   ├── retrieval/            # embedding/rerank 检索集（自定义 JSONL；内置回退）
│   ├── asr/                  # ASR manifest 模板（音频 + 参考转录文本）
│   ├── scenarios/            # 真实场景用例（wechat_intent / case_logic / article_knowledge）
│   └── conditioned/          # conditioned 上下文梯度维度的 needle 探针
├── fixtures/
│   └── README.md             # 自带数据指南
├── golden/
│   └── expectations.json     # 验收标准（逐维度）+ 演示用例
├── docs/                     # 深度内容：严谨性 / 基线 / 可复现性 / 案例研究 / spec
├── tests/                    # 离线测试套件（无需 GPU）——参见 tests/TESTING.md
└── .github/workflows/ci.yml  # lint / 语法 / shellcheck + 完整离线 pytest 套件
```

---

## 可选：通过 ZeroTier 部署到远程气隙 GPU 主机

如果您希望通过扁平 L2 VPN 将评测推送到远程 DGX，附带的 `scripts/setup_zerotier.sh` 会自动安装 ZeroTier 并加入您在 [my.zerotier.com](https://my.zerotier.com) 创建的网络：

```bash
ZEROTIER_NETWORK_ID=<your-16-hex-id> sudo -E bash scripts/setup_zerotier.sh
# 然后在 https://my.zerotier.com/network/<your-id> 批准新节点
```

这完全是可选的——直接 SSH 或 `scp` 同样有效。

---

## 常见问题

**Q：为什么把 VLM 和 LLM 放在同一个仓库？**
A：许多团队在大版本模型升级时（如 Qwen2.5 → Qwen3）会同时替换两者。放在同一个 harness 中，可以对照共享基线进行比较，并运行跨模态验收标准。

**Q：为什么特别选用 vLLM？**
A：它是事实上的 OpenAI 兼容服务栈，具有强大的持续批处理、paged-attention 以及 FP8 / AWQ 支持。harness 本身只通过 HTTP 通信，因此可以指向任何兼容端点——但启动脚本假定使用 vLLM。

**Q：能在消费级 GPU（4090 / 3090 / 7900 XT）上运行吗？**
A：235B 旗舰——不行。30B-A3B FP8——勉强（RTX 4090 为 24 GB，FP8 需要约 35 GB）。8B VLM——可以，需谨慎（bf16 → fp16）。相应调整 `models.yaml::quantization` 和 `dtype`。

**Q：我的模型 OpenAI 兼容，但不在 HuggingFace 上。**
A：在 `models.yaml` 中设置 `hf_repo: null` 并跳过 `prepare_offline.sh`——直接在 `start_all.sh` 中指向您的端点 URL。

**Q：如何在没有 GPU 服务器的 Windows 上运行？**
A：使用 Windows 版 Ollama（`winget install Ollama.Ollama`），设置 `provider: ollama` 并用 `base_url_override` 指向 Windows 机器 IP。harness 只需要 HTTP 访问端点。如需远程管理，启用 OpenSSH（`Add-WindowsCapability -Online -Name OpenSSH.Server~~~~0.0.1.0`）。参见上方"Windows 快速开始"。

**Q：如何使用 llama.cpp 代替 vLLM？**
A：启动 `llama-server --port 8080 --model model.gguf`（它提供 OpenAI 兼容端点），在 models.yaml 中设置 `provider: llama_cpp` 和 `port: 8080`。运行 `python3 scripts/probe_provider.py --model <name>` 验证。harness 的 429 重试仅针对云供应商激活；本地 llama.cpp 不需要。

**Q：HTML 报告中多个维度的质量分数显示为 0。**
A：`SKIPPED`（模型不具备该能力）或 `BLOCKED`（前置条件缺失——如数据集不可访问）的维度在雷达图中显示为 0。运行 `python3 scripts/verify_benchmark.py` 检查数据集完整性，或用 `--skip <dim>` 跳过尚未配置的维度。

**Q：`--compare` 返回 INCONCLUSIVE，即使两个模型都运行正常。**
A：以下情况会返回 INCONCLUSIVE：（a）任一报告只用了 1 个 seed——用 `--seeds 3` 重新运行；（b）报告间的硬件配置不同——性能侧强制 INCONCLUSIVE，但质量侧仍会比较；（c）报告来自不同 harness 版本——在同一版本上重新生成两份报告。

**Q：如何添加新的评测维度？**
A：在 `benchmark/<dim>/` 下新建包，在 `run_benchmark.py::DIMENSIONS` 中添加 `DimensionSpec` 条目（运行函数 + 能力门控 + `benchmark/report/sections.py` 中的渲染钩子），在 `models.yaml::benchmarks` 中添加阈值。参见 `docs/CONTRIBUTING.md` 和 `DEVELOP.md`。

---

## 贡献

欢迎 PR——参见 [docs/CONTRIBUTING.md](docs/CONTRIBUTING.md)。本项目的互动受 [Code of Conduct](docs/CODE_OF_CONDUCT.md) 约束。

维护者偏好小而专注的 PR，而非大规模重构。新的模型适配器、硬件配置和评测维度尤其受欢迎。**严禁将真实 PII 提交到 fixtures/ 目录。**

## 许可证

[Apache License 2.0](LICENSE)

## 致谢

- [vLLM](https://github.com/vllm-project/vllm) — 让一切成为可能的服务栈
- [Qwen](https://github.com/QwenLM/Qwen3) — 默认矩阵使用的参考模型系列
- [HuggingFace Hub](https://huggingface.co) — 模型分发平台
