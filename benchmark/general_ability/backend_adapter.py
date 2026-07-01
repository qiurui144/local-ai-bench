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
    benchmarks = getattr(model_cfg, "benchmarks", None) or {}
    ga_cfg = benchmarks.get("general_ability") or {}
    lb_cfg = LBModelConfig(
        name=model_cfg.name,
        type="llm",
        backend="openai_compatible",
        openai_base_url=model_cfg.base_url,
        openai_model=getattr(model_cfg, "model_id", None) or model_cfg.hf_repo or model_cfg.name,
        extra={
            "ollama_think": getattr(model_cfg, "ollama_think", True),
            "timeout_s": float(ga_cfg.get("timeout_s", benchmarks.get("backend_timeout_s", 600.0))),
            "retry_attempts": int(ga_cfg.get("retry_attempts", benchmarks.get("backend_retry_attempts", 8))),
            "retry_initial_s": float(ga_cfg.get("retry_initial_s", benchmarks.get("backend_retry_initial_s", 3.0))),
            "retry_max_s": float(ga_cfg.get("retry_max_s", benchmarks.get("backend_retry_max_s", 30.0))),
        },
    )
    return OpenAICompatibleBackend(lb_cfg)
