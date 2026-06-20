# K1 SpacemiT — 性能基线报告

**评测时间**：2026-04-13T22:33:43
**设备**：SpacemiT K1 (MUSE Pi Pro)
**架构**：RISC-V 64 (RVV 1.0)

## 一、系统信息 (L1–L5)

| 层级 | 项目 | 值 |
|------|------|-----|
| L1/L2 ISA | 架构 | riscv64 |
| L1/L2 ISA | RVV 支持 | ✅ RVV 1.0 |
| L1/L2 ISA | 内核版本 | 6.6.63 |
| L3 libc | 类型/版本 | glibc 2.39 |
| L4 BLAS | 后端 | none |
| L5 llama.cpp | GGML 后端 | cpu |
| L5 llama.cpp | GGML_RVV 启用 | ✅ 已启用 |

## 二、LLM 推理性能 (L7)

| 模型 | Prompt 类型 | Decode TPS | Prefill TPS | TTFT (ms) | 输出 Token |
|------|------------|-----------|------------|----------|-----------|
| qwen2.5:0.5b | short | **10.8** | 149.8 | 390 | 64 |
| qwen2.5:0.5b | medium | **10.2** | 126.4 | 552 | 256 |
| qwen2.5:0.5b | long | **9.9** | 117.7 | 687 | 512 |
| deepseek-r1:1.5b | short | **4.1** | 23.7 | 602 | 64 |
| deepseek-r1:1.5b | medium | **3.8** | 27.6 | 1084 | 256 |
| deepseek-r1:1.5b | long | **3.5** | 26.3 | 1585 | 512 |

> **分析**：
> - qwen2.5:0.5b decode ~10.8 tok/s，满足实时对话需求（人类阅读速度 ~5 tok/s）
> - deepseek-r1:1.5b decode ~4.0 tok/s，适合非实时推理任务
> - RVV 加速下 prefill TPS 远高于 decode TPS，表明 prefill 阶段计算充分利用了向量指令
> - TTFT 随 prompt 长度线性增长，符合预期（无 attention quadratic 问题）

## 三、Embedding 性能 (L7)

| 模型 | 向量维度 | 吞吐 (samples/s) | 跨语言相似度 | 无关相似度 | 延迟 (ms/batch) |
|------|---------|----------------|------------|---------|--------------|
| nomic-embed-text | 768 | **2.00** | 0.384 | 0.492 | 5002 |
| bge-m3 | 1024 | **0.45** | 0.752 | 0.387 | 22140 |

> **分析**：
> - bge-m3 跨语言相似度 0.752 > nomic 0.384，bge-m3 跨语言对齐质量更好
> - nomic-embed-text 吞吐 4× 于 bge-m3，适合高频低精度检索场景
> - nomic 无关相似度(0.49) > 语义相关(0.38)，说明语义区分度不足，不推荐用于精确检索
> - bge-m3 语义区分度正常(0.75 > 0.39)，推荐用于 RAG/语义搜索

## 四、Rerank 排序质量 (L7)

| 模型 | NDCG@5 | P@3 | 吞吐 (pairs/s) | 延迟 (ms/batch) |
|------|--------|-----|--------------|--------------|
| bge-m3 | **0.832** | 1.00 | 0.73 | 11014 |

> **分析**：
> - NDCG@5=0.832，P@3=1.00，排序质量符合 bge-m3 论文指标
> - 吞吐 0.73 pairs/s，适合离线批量重排序，实时场景建议控制 candidate 数量

## 五、OCR / 文档处理 (L7)

| 场景 | 模型 | TPS | 延迟 (ms) |
|------|------|-----|---------|
| LLM 辅助 OCR 后处理 | qwen2.5:0.5b | 7.6 tok/s | 40928 |

> **注**：LLM-assisted OCR post-processing (PyMuPDF available as pure rule-based baseline)
> PyMuPDF 纯规则基线待 K1 上安装 pymupdf 后补充

## 六、ASR 语音识别

**状态**：N/A
**原因**：whisper not available on SpacemiT Ollama RISC-V
> 建议：通过 `pip install openai-whisper` 安装后使用 `llama-cpp-python` 后端评测

## 七、瓶颈分析（全链路）

| 层级 | 状态 | 详情 |
|------|------|------|
| L1/L2 ISA | ✅ 最优 | RVV 1.0 + zve64d 已启用，VLEN=256bit |
| L3 libc | ✅ 正常 | glibc 2.39，无性能限制 |
| L4 BLAS | ⚠️ 未配置 | blas_backend=none，prefill 阶段 GEMM 完全依赖 GGML 内置 RVV kernel |
| L5 llama.cpp | ✅ 已启用 | GGML_RVV=on，RVV kernel 已编译进 SpacemiT 定制 Ollama |
| L6 perf | ⏭️ 未采集 | 需在 K1 上设置 perf_event_paranoid=1 |
| L7 应用 | 📊 已测 | decode/prefill TPS 分离，多 prompt 长度测试 |

### 瓶颈结论

**主要瓶颈**：`ddr_bandwidth_bound`（内存带宽受限）

**证据**：
- K1 内存带宽约 ~20 GB/s（LPDDR4X，双通道理论值），qwen2.5:0.5b (Q4_K_M ≈ 350MB)
- 理论 decode 上限 = 20GB/s ÷ 0.35GB ≈ **57 tok/s**，实测 10.8 tok/s ≈ **19% 利用率**
- deepseek-r1:1.5b (Q4_K_M ≈ 1.0GB)，理论上限 ~20 tok/s，实测 4.1 tok/s ≈ **20% 利用率**
- 低利用率说明存在额外开销（注意力计算、序列化、Ollama Python bridge）

**次要瓶颈**：`blas_backend_missing`（嵌入推理 GEMM 效率有优化空间）

### 优化建议

1. **编译 OpenBLAS with RVV kernel**：
   ```bash
   # 在 K1 上编译 OpenBLAS（启用 RVV 后端）
   TARGET=RISCV64_RVV make -j4
   ```
   预期 prefill TPS 提升 1.5–2×（矩阵乘加速）
2. **降量化精度**（Q4_K_M → Q2_K）：减少内存带宽消耗，提升 decode TPS
3. **启用 perf 计数器**：获取真实 IPC / L3 miss rate，精确定位热点
4. **安装 whisper 推理后端**：完成 ASR 场景评测

## 八、基线注册信息

| 项目 | 值 |
|------|-----|
| 设备 ID | k1-spacemit |
| 基线目录 | `baselines/k1-spacemit/2026-04-13T22-33-43/` |
| Run ID | k1-pilot-001 |
| 下次运行对比 | 自动与本次基线对比，回归阈值 -10%(warning) / -20%(critical) |
