"""配置模型：使用 Pydantic v2 进行类型安全的配置加载与验证。"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class ModelType(str, Enum):
    LLM = "llm"
    EMBEDDING = "embedding"
    WHISPER = "whisper"
    ASR = "asr"              # 多后端 ASR 对比（ONNX Whisper / SenseVoice / FunASR）
    OCR = "ocr"              # 多分辨率 OCR 对比（RapidOCR / PP-OCR）
    RERANK = "rerank"
    DOCLING = "docling"
    SPEAKER = "speaker"      # 说话人分离 / 说话人确认


class BackendType(str, Enum):
    # ── LLM / Embedding / Rerank（通用） ──────────────────────────────────────
    OLLAMA = "ollama"                        # 主后端：Ollama REST API
    LLAMA_CPP = "llama_cpp"                  # Fallback：RISC-V/RVV、批量推理
    OPENAI_COMPATIBLE = "openai_compatible"  # vLLM / LMDeploy / SGLang / Infinity / TEI
    SENTENCE_TRANSFORMERS = "sentence_transformers"  # Embedding/Rerank 直连 CPU/GPU

    # ── ASR ───────────────────────────────────────────────────────────────────
    FASTER_WHISPER = "faster_whisper"        # Whisper 高精度 Fallback（CTranslate2）
    WHISPER_CPP = "whisper_cpp"              # whisper.cpp（CPU/CUDA/Metal，低内存）
    WHISPER_ONNX = "whisper_onnx"            # ONNX Whisper（RISC-V/无GPU，tiny/base/small）
    FUNASR = "funasr"                        # FunASR / SenseVoice（中文强，需 torch）
    SENSEVOICE_ONNX = "sensevoice_onnx"      # SenseVoice ONNX（无 torch，K1/嵌入式）

    # ── OCR ───────────────────────────────────────────────────────────────────
    RAPIDOCR = "rapidocr"                    # RapidOCR PP-OCR ONNX（中英混合，嵌入式）

    # ── 文档解析 ──────────────────────────────────────────────────────────────
    DOCLING = "docling"                      # IBM Docling（TableFormer + EasyOCR）
    MINER_U = "miner_u"                      # MinerU（PDF 学术文档，PaddleOCR）
    MARKER = "marker"                        # Marker（surya 模型，CPU/GPU）
    UNSTRUCTURED = "unstructured"            # Unstructured（混合规则+模型，企业常用）
    PYMUPDF = "pymupdf"                      # PyMuPDF（规则解析基线，无GPU需求）

    # ── 说话人分析 ────────────────────────────────────────────────────────────
    WESPEAKER = "wespeaker"                  # WeSpeaker（AISHELL，ECAPA-TDNN/ResNet/CAM++）
    PYANNOTE = "pyannote"                    # pyannote.audio（最广泛使用，支持离线）
    NEMO_SPEAKER = "nemo_speaker"            # NVIDIA NeMo MSDD（Multi-Scale Diarization）


class HardwareType(str, Enum):
    AUTO = "auto"        # Ollama 自动检测
    CPU = "cpu"
    CPU_AVX2 = "cpu_avx2"
    CPU_AVX512 = "cpu_avx512"
    CPU_RVV = "cpu_rvv"  # RISC-V Vector Extension（仅 llama_cpp 后端）
    CUDA = "cuda"
    ROCM = "rocm"
    METAL = "metal"


class DeviceConfig(BaseModel):
    """远程设备 SSH 连接配置（用于跨网络 Ollama 访问及远端 ISA 采集）。"""

    host: str
    user: str
    password: Optional[str] = None
    key_file: Optional[str] = None
    ssh_port: int = 22
    ollama_remote_port: int = 11434
    local_tunnel_port: int = 11435
    name: str = ""
    arch_hint: str = ""


class OllamaServiceConfig(BaseModel):
    base_url: str = "http://localhost:11434"
    timeout_seconds: int = 300
    # 可选远程设备配置：填写后 Runner 自动建立 SSH 隧道并采集远端 ISA
    device: Optional[DeviceConfig] = None


class FallbackConfig(BaseModel):
    backend: BackendType
    path: Optional[Path] = None
    extra: Dict[str, Any] = Field(default_factory=dict)


class ModelConfig(BaseModel):
    name: str
    type: ModelType
    backend: BackendType
    # Ollama 后端
    ollama_model: Optional[str] = None
    # OpenAI-compatible 后端（vLLM / LMDeploy / SGLang / Infinity / TEI）
    openai_base_url: Optional[str] = None   # e.g. "http://localhost:8000/v1"
    openai_api_key: str = "EMPTY"           # vLLM/LMDeploy 默认不需要真实 key
    openai_model: Optional[str] = None      # 服务端模型名称（vLLM 需要）
    # 本地模型路径（llama_cpp / faster_whisper / whisper_cpp / funasr / marker / miner_u）
    path: Optional[Path] = None
    hardware: HardwareType = HardwareType.AUTO
    context_length: int = 4096
    # llama_cpp 专用
    n_gpu_layers: int = -1
    n_threads: Optional[int] = None
    batch_size: int = 512
    # 可选 fallback 配置（当主后端精度不达标时切换）
    fallback: Optional[FallbackConfig] = None
    extra: Dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_backend_fields(self) -> "ModelConfig":
        if self.backend == BackendType.OLLAMA and not self.ollama_model:
            raise ValueError(f"模型 '{self.name}': backend=ollama 时必须指定 ollama_model")
        if self.backend == BackendType.OPENAI_COMPATIBLE and not self.openai_base_url:
            raise ValueError(
                f"模型 '{self.name}': backend=openai_compatible 时必须指定 openai_base_url"
            )
        _path_required = (
            BackendType.LLAMA_CPP,
            BackendType.FASTER_WHISPER,
            BackendType.WHISPER_CPP,
        )
        if self.backend in _path_required and self.path is None:
            raise ValueError(
                f"模型 '{self.name}': backend={self.backend.value} 时必须指定 path"
            )
        if self.hardware == HardwareType.CPU_RVV and self.backend != BackendType.LLAMA_CPP:
            raise ValueError("hardware=cpu_rvv 仅支持 backend=llama_cpp")
        return self


# ─── Benchmark 任务配置 ───────────────────────────────────────────────────────

class ThresholdConfig(BaseModel):
    min_value: Optional[float] = None
    max_value: Optional[float] = None

    @model_validator(mode="after")
    def _check_order(self) -> "ThresholdConfig":
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"ThresholdConfig: min_value ({self.min_value}) 不能大于 max_value ({self.max_value})"
            )
        return self

    def check(self, value: float) -> bool:
        if self.min_value is not None and value < self.min_value:
            return False
        if self.max_value is not None and value > self.max_value:
            return False
        return True


class BenchmarkTaskConfig(BaseModel):
    enabled: bool = True
    num_samples: Optional[int] = None  # None = 全量
    few_shot: int = 0
    dataset_path: Optional[Path] = None
    thresholds: Dict[str, ThresholdConfig] = Field(default_factory=dict)
    extra: Dict[str, Any] = Field(default_factory=dict)


class PerformanceConfig(BaseModel):
    enabled: bool = True
    num_warmup_requests: int = Field(default=3, ge=0)
    num_test_requests: int = Field(default=20, ge=1)
    prompt_lengths: List[int] = Field(default_factory=lambda: [128, 512, 1024])
    output_lengths: List[int] = Field(default_factory=lambda: [128, 256])
    thresholds: Dict[str, ThresholdConfig] = Field(default_factory=dict)
    # 并发压测（需要 profiling.enabled=True 才会触发）
    concurrency_levels: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])

    @field_validator("prompt_lengths", "output_lengths", "concurrency_levels", mode="after")
    @classmethod
    def _non_empty_positive(cls, v: List[int], info) -> List[int]:
        if not v:
            raise ValueError(f"{info.field_name} 不能为空列表")
        if any(x <= 0 for x in v):
            raise ValueError(f"{info.field_name} 中所有值必须为正整数，得到: {v}")
        return v


class ProfilingConfig(BaseModel):
    """软件栈瓶颈分析配置。enabled=True 时启用全链路瓶颈检测。"""
    enabled: bool = False
    sample_interval_ms: int = Field(default=100, ge=10)   # 最小 10ms，防止过度采样
    concurrency_levels: List[int] = Field(default_factory=lambda: [1, 2, 4, 8])
    stress_requests_per_level: int = Field(default=12, ge=1)
    # Context Length 扩展曲线
    context_lengths: List[int] = Field(
        default_factory=lambda: [128, 512, 1024, 2048, 4096, 8192]
    )
    context_scaling_output_tokens: int = Field(default=64, ge=1)
    # 持续负载测试（热降频检测）
    sustained_load_duration_s: int = Field(default=60, ge=10)  # 最短 10s 才有意义
    sustained_load_window_s: int = Field(default=10, ge=1)
    # perf 性能计数器（需要 perf_event_paranoid ≤ 1）
    perf_enabled: bool = False

    @field_validator("concurrency_levels", "context_lengths", mode="after")
    @classmethod
    def _non_empty_positive(cls, v: List[int], info) -> List[int]:
        if not v:
            raise ValueError(f"{info.field_name} 不能为空列表")
        if any(x <= 0 for x in v):
            raise ValueError(f"{info.field_name} 中所有值必须为正整数，得到: {v}")
        return v

    @model_validator(mode="after")
    def _window_le_duration(self) -> "ProfilingConfig":
        if self.sustained_load_window_s > self.sustained_load_duration_s:
            raise ValueError(
                f"sustained_load_window_s ({self.sustained_load_window_s}) "
                f"不能大于 sustained_load_duration_s ({self.sustained_load_duration_s})"
            )
        return self


class LLMBenchmarkConfig(BaseModel):
    mmlu: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    gsm8k: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    hellaswag: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    truthfulqa: BenchmarkTaskConfig = Field(
        default_factory=lambda: BenchmarkTaskConfig(enabled=False)
    )
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)


class WhisperBenchmarkConfig(BaseModel):
    wer_cer: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    dataset: str = "librispeech"
    language: str = "en"
    beam_size: int = 5


class EmbeddingBenchmarkConfig(BaseModel):
    retrieval: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    similarity: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    mteb_tasks: List[str] = Field(default_factory=list)


class RerankBenchmarkConfig(BaseModel):
    beir_datasets: List[str] = Field(
        default_factory=lambda: ["msmarco", "trec-covid", "nfcorpus"]
    )
    k_values: List[int] = Field(default_factory=lambda: [1, 3, 5, 10, 100])
    tasks: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)


class DoclingBenchmarkConfig(BaseModel):
    parse_accuracy: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    table_extraction: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    throughput: BenchmarkTaskConfig = Field(
        default_factory=lambda: BenchmarkTaskConfig(enabled=True, num_samples=10)
    )
    document_types: List[str] = Field(default_factory=lambda: ["pdf", "docx", "pptx"])


class SpeakerBenchmarkConfig(BaseModel):
    """说话人分离 / 说话人确认 benchmark 配置。"""
    diarization: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    verification: BenchmarkTaskConfig = Field(
        default_factory=lambda: BenchmarkTaskConfig(enabled=False)
    )
    # 评测数据集名称列表（本地 RTTM 目录 or HuggingFace 数据集名）
    datasets: List[str] = Field(default_factory=lambda: ["ami"])
    # DER 评测参数
    collar: float = 0.25         # NIST 标准边界忽略时间窗（秒）
    skip_overlap: bool = False   # 是否跳过重叠语音段计算 DER


class ASRBenchmarkConfig(BaseModel):
    """多后端 ASR 对比 benchmark 配置。"""
    rtf: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    # 测试音频时长列表（秒），用于 RTF 扩展曲线
    audio_durations_s: List[float] = Field(default_factory=lambda: [3.0, 5.0, 10.0, 15.0])
    # 语言列表（影响特征提取和 token 解码）
    languages: List[str] = Field(default_factory=lambda: ["en", "zh"])
    # 参考文本（用于 CER 计算），key = 语言
    reference_texts: Dict[str, str] = Field(default_factory=dict)
    num_warmup: int = 1
    num_runs: int = 3


class OCRBenchmarkConfig(BaseModel):
    """多分辨率 OCR benchmark 配置。"""
    accuracy: BenchmarkTaskConfig = Field(default_factory=BenchmarkTaskConfig)
    # 输入图像分辨率缩放系数列表（1.0=原始, 0.5=50%降采样）
    input_scales: List[float] = Field(default_factory=lambda: [1.0, 0.5])
    # 测试语言（影响模型选择：'ch' 中英混合, 'en' 仅英文）
    lang: str = "ch"
    num_warmup: int = 1
    num_runs: int = 3


class BenchmarksConfig(BaseModel):
    llm: LLMBenchmarkConfig = Field(default_factory=LLMBenchmarkConfig)
    whisper: WhisperBenchmarkConfig = Field(default_factory=WhisperBenchmarkConfig)
    asr: ASRBenchmarkConfig = Field(default_factory=ASRBenchmarkConfig)
    ocr: OCRBenchmarkConfig = Field(default_factory=OCRBenchmarkConfig)
    embedding: EmbeddingBenchmarkConfig = Field(default_factory=EmbeddingBenchmarkConfig)
    rerank: RerankBenchmarkConfig = Field(default_factory=RerankBenchmarkConfig)
    docling: DoclingBenchmarkConfig = Field(default_factory=DoclingBenchmarkConfig)
    speaker: SpeakerBenchmarkConfig = Field(default_factory=SpeakerBenchmarkConfig)
    profiling: ProfilingConfig = Field(default_factory=ProfilingConfig)


class GlobalConfig(BaseModel):
    output_dir: Path = Path("outputs")
    log_level: str = "INFO"
    seed: int = 42
    parallel_workers: int = 1
    timeout_seconds: int = 3600

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, v: str) -> str:
        valid = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if v.upper() not in valid:
            raise ValueError(f"log_level 必须是 {valid} 之一")
        return v.upper()


class AppConfig(BaseModel):
    global_config: GlobalConfig = Field(default_factory=GlobalConfig)
    ollama: OllamaServiceConfig = Field(default_factory=OllamaServiceConfig)
    models: List[ModelConfig]
    benchmarks: BenchmarksConfig = Field(default_factory=BenchmarksConfig)

    @classmethod
    def load(cls, models_path: str, benchmarks_path: str) -> "AppConfig":
        """从 YAML 文件加载完整配置。"""
        with open(models_path, "r", encoding="utf-8") as f:
            models_data = yaml.safe_load(f)
        with open(benchmarks_path, "r", encoding="utf-8") as f:
            benchmarks_data = yaml.safe_load(f)

        return cls(
            global_config=models_data.get("global", {}),
            ollama=models_data.get("ollama", {}),
            models=models_data.get("models", []),
            benchmarks=benchmarks_data,
        )

    def get_models_by_type(self, model_type: ModelType) -> List[ModelConfig]:
        return [m for m in self.models if m.type == model_type]

    def get_model(self, name: str) -> Optional[ModelConfig]:
        for m in self.models:
            if m.name == name:
                return m
        return None

    def get_benchmark_config(self, model_type: ModelType) -> Any:
        mapping = {
            ModelType.LLM: self.benchmarks.llm,
            ModelType.WHISPER: self.benchmarks.whisper,
            ModelType.EMBEDDING: self.benchmarks.embedding,
            ModelType.RERANK: self.benchmarks.rerank,
            ModelType.DOCLING: self.benchmarks.docling,
            ModelType.SPEAKER: self.benchmarks.speaker,
        }
        return mapping[model_type]
