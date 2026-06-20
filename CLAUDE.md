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
