"""推理后端包。

通过导入触发 @register_backend 装饰器注册，使 create_backend() 能按名称查找。
"""

from benchmark.llama_benchmark.core.registry import create_backend

# 触发所有后端注册
from benchmark.llama_benchmark.backends import (  # noqa: F401
    ollama_backend,
    llama_backend,
    whisper_backend,
    docling_backend,
    openai_compatible_backend,
    sentence_transformers_backend,
    whisper_cpp_backend,
    funasr_backend,
    miner_u_backend,
    marker_backend,
    unstructured_backend,
    pymupdf_backend,
    # 说话人分析后端
    wespeaker_backend,
    pyannote_backend,
    nemo_speaker_backend,
)

__all__ = ["create_backend"]
