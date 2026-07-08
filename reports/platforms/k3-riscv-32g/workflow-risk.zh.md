# K3 工作流风险

**最后更新：** 2026-07-08
**英文版本：** [workflow-risk.en.md](workflow-risk.en.md)
**来源：** [旧完整报告](../../k3-riscv-32g.en.md)、[docs/k3-realistic-stress-plan.md](../../../docs/k3-realistic-stress-plan.md)

## 范围

本报告把原始模型结果转换为 K3 32G 的产品工作流风险，覆盖常见 embedding、reranker、OCR、ASR、VLM、LLM 和长文档工作流。

## 工作流结论

| 工作流 | 推荐路径 | 风险等级 | 必要控制 |
|---|---|---|---|
| 实时 RAG | BGE-Zh embedding -> BGE reranker top-k <=20 -> 有界 Qwen3-30B 回答 | 中 | context 上限、token 上限、单请求超时 |
| 文档 OCR | 先 PP-OCRv5 OCR，只有视觉理解才用 VLM | 低/中 | 分离 OCR 和 VLM 队列；避免 VLM-as-OCR |
| 同步 VLM 上传 | Qwen3.5-2B SMT | 中 | TCM 预检、超时、可见 fallback |
| 异步高质量 VLM | Qwen3VL-4B、qwen30ba3b-mm 或 Qwen3.5-35B+mmproj | 高 | 队列、TTL、取消、单任务准入 |
| ASR | qwen3-ASR 0.6B SMT | 低/中 | 评分前归一化中文繁简和数字 |
| 飞行手册长文档 | 离线 text/OCR -> embedding -> reranker -> 引用证据窗口 -> 异步 LLM | 高 | tokenizer-aware 裁剪、队列隔离、内存保护 |
| 模型扫测 | 本地 `drivers/spacemit-ai/model_zoo` 作 canonical cache，K3 只作 working cache | 中 | 证据采集后删除 K3 非热模型副本 |

## 长上下文 / 飞行手册

| 模型/窗口 | 结果 | 结论 |
|---|---|---|
| `Qwen3-4B-Q4_0`，飞行手册 1K window | score 1.0，但 E2E 175.453s | 链路可用；同步体验失败 |
| `Qwen3-4B-Q4_0`，飞行手册 3K window | 首个请求超过 5 分钟未完成 | 不适合同步 |
| `Qwen3-0.6B-Q4_0`，naive 1K windows | 出现 context-overflow 错误 | 字符裁剪不安全 |
| `Qwen3-0.6B-Q4_0`，safety-budget windows | 无 overflow，score 0.5958 | API 稳定，质量低于门禁 |
| `Qwen3-30B` / 35B 档 | 短 probe 通过，但长文本接近内存/延迟极限 | 只作为异步 verifier，并加准入控制 |

## 资源风险

| 资源 | 观察到的问题 | 控制方式 |
|---|---|---|
| 内存 | 30B realistic 1K 达到 30.488GiB RSS | 单大模型任务，长上下文请求拒绝或排队 |
| TCM | stale blocks 导致 ORT/SMT 失败 | media run 前预检 `spacemit-tcm-smi`，释放 stale blocks |
| 存储 | 模型扫测会填满 K3 root fs | canonical cache 留本地，K3 只保留热工作副本 |
| 调度 | raw model-server 测试不能证明公平性/背压 | 实现 `/capacity`、queue wait、取消、TTL、异步状态 |
| 功耗 | 未采集板级功耗 | 不能做 K3 TPS/W 横向比较 |

## 产品门禁

| 门禁 | 要求 |
|---|---|
| 准入 | 大 LLM/VLM 任务必须检查空闲内存、TCM 状态和当前活跃任务。 |
| 超时 | 同步路径需要短硬超时；慢速质量模型必须异步。 |
| 取消 | 异步文档和长上下文任务必须支持取消和 TTL 清理。 |
| 证据窗口 | 长文档必须用检索和 tokenizer-aware 裁剪，不能全手册直灌。 |
| 指标 | 每次运行采集延迟、RSS、TCM 状态、队列等待、错误类型、模型/runtime 版本。 |
