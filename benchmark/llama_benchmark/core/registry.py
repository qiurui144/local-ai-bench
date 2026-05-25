"""Runner 与 Backend 注册表：自动发现已注册的实现。"""

from __future__ import annotations

import logging
from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from benchmark.llama_benchmark.core.base_runner import AbstractBenchmarkRunner
    from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
    from benchmark.llama_benchmark.core.config import AppConfig, BackendType, ModelConfig, ModelType

logger = logging.getLogger(__name__)

_runner_registry: Dict[str, Type["AbstractBenchmarkRunner"]] = {}
_backend_registry: Dict[str, Type["AbstractModelBackend"]] = {}


def register_runner(model_type: str):
    """装饰器：将 Runner 类注册到注册表。"""
    def decorator(cls: Type["AbstractBenchmarkRunner"]):
        _runner_registry[model_type] = cls
        return cls
    return decorator


def register_backend(backend_type: str):
    """装饰器：将 Backend 类注册到注册表。"""
    def decorator(cls: Type["AbstractModelBackend"]):
        _backend_registry[backend_type] = cls
        return cls
    return decorator


def create_runner(
    model_config: "ModelConfig",
    app_config: "AppConfig",
) -> "AbstractBenchmarkRunner":
    """根据模型类型创建对应的 Runner 实例。"""
    # 延迟导入以触发注册
    _ensure_runners_loaded()

    key = model_config.type.value
    if key not in _runner_registry:
        raise ValueError(
            f"未找到模型类型 '{key}' 对应的 Runner，"
            f"已注册: {list(_runner_registry.keys())}"
        )
    return _runner_registry[key](model_config, app_config)


def create_backend(model_config: "ModelConfig") -> "AbstractModelBackend":
    """根据后端类型创建对应的 Backend 实例。"""
    _ensure_backends_loaded()

    key = model_config.backend.value
    if key not in _backend_registry:
        raise ValueError(
            f"未找到后端类型 '{key}' 对应的 Backend，"
            f"已注册: {list(_backend_registry.keys())}"
        )
    return _backend_registry[key](model_config)


def _ensure_runners_loaded() -> None:
    """触发所有 runner 模块的导入，以完成注册。"""
    import importlib
    runner_modules = [
        "llama_benchmark.benchmarks.llm.runner",
        "llama_benchmark.benchmarks.whisper.runner",
        "llama_benchmark.benchmarks.embedding.runner",
        "llama_benchmark.benchmarks.rerank.runner",
        "llama_benchmark.benchmarks.docling.runner",
        "llama_benchmark.benchmarks.speaker.runner",
        "llama_benchmark.benchmarks.asr.runner",       # ASR 多后端对比
        "llama_benchmark.benchmarks.ocr.runner",       # OCR 多分辨率对比
    ]
    for module in runner_modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            logger.debug("Runner 模块可选依赖缺失，跳过: %s（%s）", module, exc)
        except Exception as exc:
            logger.warning("Runner 模块加载失败，该 Runner 不可用: %s — %s", module, exc)


def _ensure_backends_loaded() -> None:
    """触发所有 backend 模块的导入，以完成注册。"""
    import importlib
    backend_modules = [
        # LLM / Embedding / Rerank
        "llama_benchmark.backends.ollama_backend",
        "llama_benchmark.backends.llama_backend",
        "llama_benchmark.backends.openai_compatible_backend",
        "llama_benchmark.backends.sentence_transformers_backend",
        # ASR
        "llama_benchmark.backends.whisper_backend",
        "llama_benchmark.backends.whisper_cpp_backend",
        "llama_benchmark.backends.whisper_onnx_backend",
        "llama_benchmark.backends.funasr_backend",
        "llama_benchmark.backends.sensevoice_onnx_backend",
        # OCR
        "llama_benchmark.backends.rapidocr_backend",
        # 文档解析
        "llama_benchmark.backends.docling_backend",
        "llama_benchmark.backends.miner_u_backend",
        "llama_benchmark.backends.marker_backend",
        "llama_benchmark.backends.unstructured_backend",
        "llama_benchmark.backends.pymupdf_backend",
        # 说话人分析
        "llama_benchmark.backends.wespeaker_backend",
        "llama_benchmark.backends.pyannote_backend",
        "llama_benchmark.backends.nemo_speaker_backend",
    ]
    for module in backend_modules:
        try:
            importlib.import_module(module)
        except ImportError as exc:
            logger.debug("Backend 模块可选依赖缺失，跳过: %s（%s）", module, exc)
        except Exception as exc:
            logger.warning("Backend 模块加载失败，该 Backend 不可用: %s — %s", module, exc)


def list_runners() -> Dict[str, str]:
    _ensure_runners_loaded()
    return {k: v.__name__ for k, v in _runner_registry.items()}


def list_backends() -> Dict[str, str]:
    _ensure_backends_loaded()
    return {k: v.__name__ for k, v in _backend_registry.items()}
