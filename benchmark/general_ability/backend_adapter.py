"""harness ModelConfig → llama_benchmark OpenAICompatibleBackend(Q1-A 适配层)。

单向消费:不复制 backend 实现(arch review F6),只做配置翻译。错误语义:
backend 方法对 HTTP/网络错误抛异常,由 runner 逐题捕获计 error。
"""
from __future__ import annotations

from benchmark.llama_benchmark.backends.openai_compatible_backend import (
    OpenAICompatibleBackend,
)
from benchmark.llama_benchmark.core.config import ModelConfig as LBModelConfig


def make_backend(model_cfg) -> OpenAICompatibleBackend:
    lb_cfg = LBModelConfig(
        name=model_cfg.name,
        type="llm",
        backend="openai_compatible",
        openai_base_url=model_cfg.base_url,
        openai_model=getattr(model_cfg, "model_id", None) or model_cfg.hf_repo or model_cfg.name,
    )
    return OpenAICompatibleBackend(lb_cfg)
