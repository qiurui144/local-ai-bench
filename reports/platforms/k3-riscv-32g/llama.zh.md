# K3 llama.cpp / GGUF 路径

**最后更新：** 2026-07-08
**英文版本：** [llama.en.md](llama.en.md)
**来源：** [CPU/RVV/IME 运行报告](../../runs/k3-riscv-32g/20260704/k3-riscv-32g-cpu.en.md)、[旧完整报告](../../k3-riscv-32g.en.md)

## 范围

本路径覆盖 SpacemiT private `llama.cpp`、源码构建 SpacemiT `llama.cpp`、必要时的 upstream fallback、GGUF+mmproj VLM、embedding 和 reranker。Qwen 系列需要 no-think 处理，OpenAI-compatible content 才可靠。

## LLM 结果

| 模型 | 关键指标 | 状态 | 结论 |
|---|---:|---|---|
| `Qwen3-30B-A3B-Q4_0` | PP512 33.69 tok/s，TG128 9.80 tok/s；GA 通过 | 通过但有限制 | K3 有界 prompt 主 LLM |
| `Qwen3.6-35B-A3B-UD-Q4_K_XL` | PP512 30.49 tok/s，TG128 6.75 tok/s；1K/3K needle 通过 | LLM 通过，不是 VLM | 仅大 LLM 候选 |
| `Qwen3.5-35B-A3B-Q4_0` | PP512 29.31 tok/s，TG128 6.48 tok/s；1K/3K 通过 | LLM 通过，不是 VLM | 大 LLM 对照 |
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | TTFT 1.071s，decode128 29.374s；3K needle 297.122s | 通过，只适合异步 | 单服务 LLM+VLM 候选 |
| `LFM2-24B-A2B-Q4_0` | PP512 55.48 tok/s，TG128 15.36 tok/s | 性能通过，质量混合 | 仅吞吐候选 |
| `Qwen3-8B-Q4_K_M` | PP512 25.97 tok/s，TG128 4.24 tok/s | 性能通过，质量混合 | context/perf smoke |
| `Qwen3-4B-Q4_K_M` | PP512 42.14 tok/s，TG128 7.30 tok/s | smoke 通过 | 小模型对照 |
| `Qwen3-0.6B-Q4_0` | PP512 198.51 tok/s，TG128 37.53 tok/s | smoke 通过 | pipeline smoke，不是质量默认 |

## llama.cpp VLM

| 模型 | 视觉路径 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| `Qwen3.5-35B-A3B-Q4_K_M + mmproj-F16` | mtmd | 29/30 文档，平均/p95 68.822/78.774s | 通过 | 异步单服务候选 |
| `Qwen3VL-4B + mmproj` | mtmd | 30/30 文档；realistic p95 78.064s | 通过但慢 | 质量控制/异步 |
| `SmolVLM-256M + mmproj` | mtmd | 图片路径可跑但文档质量失败 | 质量失败 | 仅 runtime smoke |

## Embedding 和 Reranker

| 角色 | 模型 | 关键指标 | 状态 | 结论 |
|---|---|---:|---|---|
| Embedding | `Bge-Small-Zh-V1.5-Q4_K_M` | p95 5.85ms，overall Hit@1 0.9722 | 通过 | 默认 |
| Embedding | `Nomic-Embed-Text-V2-Moe-Q4_0` | p95 21.21ms，overall Hit@1 0.9722 | 通过 | 备选 |
| Embedding | `Qwen3-Embedding-0.6B-Q4_0` | p95 45.27ms，overall Hit@1 0.9722 | 通过 | 较慢备选 |
| Embedding | `Bge-Small-En-V1.5-Q4_K_M` | finite vector ratio 0.0 | 失败 | 不使用 |
| Reranker | `Bge-Reranker-V2-M3-Q4_0` | top50 Hit@1 1.0；top20 p95 1333ms | 通过 | 默认，限制 top-k |
| Reranker | `Qwen3-Reranker-0.6B-Q4_0` | top50 p95 18804ms；top50 Hit@1 0.8333 | 通过但慢/离线 | 不作在线默认 |

## 源码 Runtime 等价

| Runtime 行 | 结果 | 结论 |
|---|---|---|
| 源码构建 llama.cpp vs 系统版本 | 6/6 PP/TG 行通过 source/system >=0.95 | 源码 runtime 可作为优化基线 |
| Qwen3.6 private package loader 问题 | 早期 private package 缺失 `blk.40.ssm_conv1d.weight` 失败；后续当前系统 probe 可服务文本 | 每次复测都要保留 runtime 版本证据 |

## 已知限制

| 限制 | 影响 |
|---|---|
| Qwen3/Qwen3.5 thinking mode | 使用 `/no_think` 或 `enable_thinking=false`；否则 final `content` 可能为空。 |
| 长上下文 | 30B/35B 超过小窗口后延迟和内存风险高。 |
| VLM 图片支持 | 纯文本 35B GGUF 会报告 `vision_backend=none`；除非加载 mmproj/vision 路径，否则图片输入失败。 |
| 资源余量 | 30B realistic 1K 达到 30.488GiB RSS，因此必须做准入控制。 |
