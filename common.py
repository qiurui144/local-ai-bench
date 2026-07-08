"""vLLM OpenAI 兼容客户端 + 共享工具

所有 benchmark 脚本共用的：
- 调用 vLLM / llama.cpp / Ollama / 云端 API 的同步 / 异步客户端（带 VL 图片输入）
- 获取 usage（prompt_tokens / completion_tokens）
- 流式捕获 TTFT
- 从 models.yaml 加载配置
- 云端 provider（openai / deepseek / dashscope）429 限速自动退避重试

依赖：openai>=1.40 (chat.completions.create with stream=True)
    httpx
    pyyaml
    pynvml（可选，不装就跳 VRAM 监控）
    Pillow（图片 base64）
"""

from __future__ import annotations

import base64
import ipaddress
import json
import logging
import os
import time
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import yaml

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# 项目根路径
# ────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent


# ────────────────────────────────────────────────────────────
# 目标平台配置
# ────────────────────────────────────────────────────────────


@dataclass
class TargetConfig:
    name: str
    platform: str                    # linux | windows | macos
    arch: str                        # x86_64 | aarch64 | riscv64
    connection: str = "local"        # local | ssh
    ip_env: Optional[str] = None
    ssh_user_env: Optional[str] = None
    ssh_pass_env: Optional[str] = None
    runtime: str = "vllm"            # vllm | ollama | llama_cpp | rknn | generic
    runtime_port: int = 0
    accelerator: str = "cpu"         # primary/default accelerator for legacy callers
    accelerator_profiles: tuple = () # advertised execution engines, e.g. cpu/gpu/npu
    npu: Optional[str] = None
    mali_gpu: Optional[str] = None
    env_overrides: dict = field(default_factory=dict)
    remote_workdir: Optional[str] = None
    python_cmd: str = "python3"
    notes: str = ""

    def supports_accelerator(self, accelerator: str) -> bool:
        """Return whether this target advertises an execution engine."""
        profiles = set(self.accelerator_profiles or ())
        profiles.add(self.accelerator)
        return accelerator in profiles

    @property
    def ip(self) -> str:
        if self.ip_env:
            return os.environ.get(self.ip_env, "")
        return "localhost"

    @property
    def ssh_user(self) -> str:
        return os.environ.get(self.ssh_user_env or "", "") if self.ssh_user_env else ""

    @property
    def ssh_pass(self) -> str:
        return os.environ.get(self.ssh_pass_env or "", "") if self.ssh_pass_env else ""

    def is_local(self) -> bool:
        return self.connection == "local"


_TARGET_CACHE: dict = {}


def load_targets() -> dict[str, "TargetConfig"]:
    path = (Path.cwd() / "targets.yaml").resolve()
    if not path.exists():
        return {"local": TargetConfig(name="local", platform="linux", arch="x86_64")}
    if path not in _TARGET_CACHE:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        _TARGET_CACHE[path] = {
            k: TargetConfig(name=k, **v)
            for k, v in (raw.get("targets") or {}).items()
        }
    return _TARGET_CACHE[path]


# ────────────────────────────────────────────────────────────
# 模型配置
# ────────────────────────────────────────────────────────────


@dataclass
class ModelConfig:
    name: str
    hf_repo: str = ""                    # optional: cloud models have no HF repo
    port: int = 0                        # optional: cloud models use provider URL
    vram_estimate_gb: float = 0.0       # optional: not applicable to cloud
    role: str = ""                       # documentation only
    # Cross-platform provider support (v0.4+)
    provider: str = "local_vllm"        # local_vllm | llama_cpp | ollama | deepseek | dashscope | openai | generic
    model_id: Optional[str] = None      # API model name for cloud (e.g. "deepseek-chat")
    api_key_env: Optional[str] = None   # env-var name holding the API key
    base_url_env: Optional[str] = None      # env-var name holding base URL (e.g. OLLAMA_AMD_BASE_URL)
    base_url_override: Optional[str] = None  # explicit endpoint fallback (wins over provider routing)
    # Existing optional fields
    quantization: Optional[str] = None
    hardware_min: str = "A100-40G"
    task_type: str = "vlm"             # vlm | text_only
    rerank_native: bool = False        # rerank dim: True → native /v1/rerank (BERT
                                       # cross-encoder single pass) vs generative yes/no
    ocr_backend: str = "auto"          # ocr dim: auto | rapidocr | paddleocr | vitisai | directml | openvino
    ocr_model_dir: Optional[str] = None
    asr_backend: str = "auto"          # asr dim: auto | whisper_ov | whisper_ov_subprocess | whisper_amd_npu_subprocess | sherpa
    asr_model_dir: Optional[str] = None
    asr_model_type: Optional[str] = None
    asr_config_file: Optional[str] = None
    asr_device: str = "auto"
    asr_python: Optional[str] = None
    ov_model_dir: Optional[str] = None
    max_model_len: Optional[int] = None  # conditioned dim: 阶梯超限 SKIPPED 的依据;
                                         # 未配置时运行时探 /v1/models
    parameter_size_b: Optional[float] = None
    notes: str = ""
    capabilities: tuple = ()           # typed positive capability set; load_models 由
                                       # *_capable hint 派生(hint 留一个 minor 作 alias)
    target: Optional[str] = None       # targets.yaml 中的 key（None = "local"）
    benchmarks: dict = field(default_factory=dict)  # per-model benchmark overrides
    # Optional text prepended to each user prompt. Useful for runtimes where the
    # model-specific non-thinking control is prompt-level rather than API-level.
    prompt_prefix: str = ""
    # Extra chat template controls passed through to llama.cpp / other OpenAI
    # compatible servers. Example: {"enable_thinking": false} for Qwen thinking
    # templates that otherwise return reasoning_content with empty content.
    chat_template_kwargs: dict = field(default_factory=dict)
    # Ollama qwen3 thinking mode: set False to inject options.think=false, disabling chain-of-thought
    # tokens that otherwise fill max_tokens before any content is emitted (empty hyp/answer bug).
    ollama_think: bool = True
    # Custom readiness check URL; overrides the default /v1/models poll.
    # Use for services (e.g. ASR) that expose /health instead of /v1/models.
    readiness_url: Optional[str] = None

    @property
    def base_url(self) -> str:
        # 1. env-var (设备 IP 可配置): OLLAMA_AMD_BASE_URL=http://192.168.x.x:11434/v1
        if self.base_url_env:
            url = os.environ.get(self.base_url_env, "").strip()
            if url:
                return url.rstrip("/")
        # 2. explicit override (models.yaml 文档化默认值，可被 env-var 覆盖)
        if self.base_url_override:
            return self.base_url_override.strip().rstrip("/")
        if self.provider == "deepseek":
            return "https://api.deepseek.com/v1"
        if self.provider == "dashscope":
            return "https://dashscope.aliyuncs.com/compatible-mode/v1"
        if self.provider == "openai":
            return "https://api.openai.com/v1"
        if self.provider == "ollama":
            return f"http://localhost:{self.port or 11434}/v1"
        if self.provider == "llama_cpp":
            return f"http://localhost:{self.port or 8080}/v1"
        if self.provider == "local_onnx":
            return ""   # no HTTP endpoint — callers must check port == 0
        return f"http://localhost:{self.port}/v1"  # local_vllm / generic

    @property
    def auth_header(self) -> str:
        """Bearer token for Authorization header; 'Bearer EMPTY' for local endpoints."""
        if self.api_key_env:
            key = os.environ.get(self.api_key_env, "")
            if key:
                return f"Bearer {key}"
        return "Bearer EMPTY"

    @property
    def effective_model_id(self) -> str:
        """Model identifier passed in API payload 'model' field."""
        return self.model_id or self.hf_repo or self.name

    @property
    def is_vlm(self) -> bool:
        return self.task_type != "text_only"


_HINT_TO_CAP = {
    "translation_capable": "translation", "embedding_capable": "embedding",
    "rerank_capable": "rerank", "asr_capable": "asr", "ocr_capable": "ocr",
}
_NON_CHAT_CAPS = ("embedding", "rerank", "asr", "ocr", "rerank_native")


def _is_chat_capable(cfg: "ModelConfig") -> bool:
    """只有纯 chat/VLM 模型才触发 chat 维度。
    embedding / rerank / asr / ocr 专用模型明确排除。"""
    non_chat = {"embedding", "rerank", "rerank_native", "asr", "ocr"}
    caps = getattr(cfg, "capabilities", None)
    if caps is not None:
        if non_chat.intersection(caps):
            return False
        return True
    # legacy fallback（无 capabilities 字段的旧 stub）
    if getattr(cfg, "rerank_native", False):
        return False
    return True


def _derive_capabilities(raw: dict) -> tuple:
    caps = {cap for hint, cap in _HINT_TO_CAP.items() if raw.get(hint)}
    if raw.get("rerank_native"):
        caps.add("rerank_native")
    if not caps & set(_NON_CHAT_CAPS):
        caps.add("chat")               # 纯 chat/VLM 端点(translation 模型也是 chat)
    if "translation" in caps:
        caps.add("chat")               # translation 走 chat 端点
    return tuple(sorted(caps))


def load_models(yaml_path: Path | str = "models.yaml") -> list[ModelConfig]:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    # models.yaml carries documentation-only hint fields (dtype, translation_capable, ...)
    # that are not ModelConfig fields; keep them out of the constructor instead of
    # forcing every entry to match the dataclass exactly.
    known = {f.name for f in fields(ModelConfig)}
    out = []
    for m in data.get("models", []):
        kwargs = {k: v for k, v in m.items() if k in known}
        kwargs["capabilities"] = _derive_capabilities(m)
        out.append(ModelConfig(**kwargs))
    return out


def load_benchmarks_config(yaml_path: Path | str = "models.yaml") -> dict:
    with open(yaml_path, encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("benchmarks", {})


# ────────────────────────────────────────────────────────────
# 请求 payload 构造
# ────────────────────────────────────────────────────────────


def encode_image_data_url(image_path: Path) -> str:
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    ext = image_path.suffix.lower().lstrip(".")
    mime = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "webp": "webp"}.get(ext, "jpeg")
    return f"data:image/{mime};base64,{b64}"


VLM_PROMPT = """你是法律证据分类助手。分析图片内容，严格按 JSON 格式输出（只输出 JSON）：

{
  "category": "communication|financial|contract|identity|legal_document|receipt|other",
  "subcategory": "具体子类（如 wechat_chat_with_transfer / bank_statement 等）",
  "description": "一句话描述图片内容",
  "key_entities": ["人名/金额/日期/银行卡号等关键实体"],
  "key_facts": ["2-5 条关键事实"]
}"""


TEXT_PROMPT = """你是法律文本分析助手。简要分析以下内容（200 字以内），输出 JSON：

{
  "summary": "一句话核心内容",
  "entities": ["人名/金额/日期"],
  "legal_risks": ["可能的法律风险点"]
}

内容："""


# ────────────────────────────────────────────────────────────
# 同步客户端（用 httpx 直调，避免 openai SDK 的额外开销）
# ────────────────────────────────────────────────────────────

# Providers that may return HTTP 429 rate-limit; local providers only retry
# transient service restart failures.
_CLOUD_PROVIDERS = {"openai", "deepseek", "dashscope"}
_TRANSIENT_LOCAL_STATUSES = {502, 503, 504}
_LOCAL_ENDPOINT_HOSTNAMES = {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}


def _is_local_endpoint_url(base_url: str) -> bool:
    """Return True for loopback/private endpoints, even under an OpenAI-compatible provider."""
    host = (urlparse(base_url).hostname or "").lower()
    if not host:
        return False
    if host in _LOCAL_ENDPOINT_HOSTNAMES or host.endswith(".local"):
        return True
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    return ip.is_loopback or ip.is_private or ip.is_link_local


def _is_cloud_endpoint(model_cfg: ModelConfig) -> bool:
    """Cloud provider with a public endpoint; local OpenAI-compatible servers are not cloud."""
    return model_cfg.provider in _CLOUD_PROVIDERS and not _is_local_endpoint_url(model_cfg.base_url)


def _post_with_retry(
    url: str,
    payload: dict,
    headers: dict,
    timeout_s: float,
    is_cloud: bool,
) -> "httpx.Response":
    """POST with exponential-backoff retry for expected transient failures.

    Cloud providers retry 429 rate limits. Local providers keep 429 as an
    immediate error but retry 502/503/504 and transport errors, which can happen
    during supervised local service restarts.
    """
    max_retries = 3 if is_cloud else 6
    last_exc: Exception | None = None
    for attempt in range(max_retries):
        try:
            r = httpx.post(url, json=payload, headers=headers, timeout=timeout_s)
        except (httpx.TransportError, httpx.TimeoutException) as exc:
            last_exc = exc
            if attempt == max_retries - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                "transient transport error from %s, retry %d/%d in %ds: %r",
                url, attempt + 1, max_retries, wait, exc,
            )
            time.sleep(wait)
            continue

        retry_status = (
            r.status_code == 429 if is_cloud else r.status_code in _TRANSIENT_LOCAL_STATUSES
        )
        if not retry_status or attempt == max_retries - 1:
            return r
        wait = 2 ** attempt
        logger.warning(
            "transient HTTP %s from %s, retry %d/%d in %ds",
            r.status_code, url, attempt + 1, max_retries, wait,
        )
        time.sleep(wait)
    if last_exc:
        raise last_exc
    return r  # unreachable, satisfies type checkers


@dataclass
class InferResult:
    model: str
    ok: bool = False
    error: str = ""
    # 输出
    content: str = ""
    parsed_json: Optional[dict] = None
    # Token
    input_tokens: int = 0
    output_tokens: int = 0
    finish_reason: str = ""
    # 时延（ms）
    latency_ms: float = 0.0
    ttft_ms: float = 0.0               # 流式场景
    # 吞吐
    tokens_per_sec: float = 0.0        # output_tokens / (latency_ms / 1000)

    def looks_truncated(self, max_tokens_requested: int) -> bool:
        """判断是否被截断（eval_count 接近 max_tokens）"""
        return (
            self.finish_reason == "length"
            or (max_tokens_requested > 0 and self.output_tokens >= max_tokens_requested * 0.95)
        )


def _strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from model output (vLLM/OpenVINO Qwen3 thinking mode).

    Strips complete <think>...</think> blocks and any unclosed <think> block
    (model hit max_tokens during reasoning, leaving no closing tag).
    """
    import re
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
    # Also strip unclosed <think> (no </think> because max_tokens hit during reasoning)
    stripped = re.sub(r"<think>[\s\S]*", "", stripped, flags=re.IGNORECASE)
    return stripped.strip()


def _apply_prompt_prefix(prompt: str, model_cfg: ModelConfig) -> str:
    prefix = str(getattr(model_cfg, "prompt_prefix", "") or "")
    if not prefix or prompt.startswith(prefix):
        return prompt
    return f"{prefix}{prompt}"


def _apply_ollama_think_controls(payload: dict, model_cfg: ModelConfig, max_tokens: int) -> None:
    """Request non-thinking Ollama output and leave enough budget for older Qwen3 builds.

    Some Ollama/OpenAI-compatible versions ignore the non-thinking flag for Qwen3
    but still stream hidden reasoning tokens before visible content. The token
    floor prevents short benchmark requests from ending as reasoning-only output.
    """
    if getattr(model_cfg, "ollama_think", True):
        return
    payload["think"] = False
    options = dict(payload.get("options") or {})
    options["think"] = False
    payload["options"] = options
    payload["max_tokens"] = max(max_tokens, 2048)


def _apply_chat_template_kwargs(payload: dict, model_cfg: ModelConfig) -> None:
    kwargs = getattr(model_cfg, "chat_template_kwargs", None) or {}
    if kwargs:
        payload["chat_template_kwargs"] = dict(kwargs)


def infer_sync(
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout_s: float = 600.0,
    seed: Optional[int] = None,
    prior_messages: Optional[list[dict]] = None,
) -> InferResult:
    """单次同步推理（非流式），返回 token 统计 + 延迟 + 解析内容"""
    prompt = _apply_prompt_prefix(prompt, model_cfg)
    messages: list[dict] = list(prior_messages) if prior_messages else []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.effective_model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if seed is not None:
        payload["seed"] = seed   # OpenAI 兼容采样种子(vLLM 支持);judge 多 seed 用
    _apply_ollama_think_controls(payload, model_cfg, max_tokens)
    _apply_chat_template_kwargs(payload, model_cfg)
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    try:
        r = _post_with_retry(
            url, payload,
            headers={"Authorization": model_cfg.auth_header},
            timeout_s=timeout_s,
            is_cloud=_is_cloud_endpoint(model_cfg),
        )
        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    if r.status_code != 200:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"HTTP {r.status_code}: {r.text[:200]}",
            latency_ms=elapsed,
        )

    # 200 + 非 JSON body（代理错误页 / 截断响应）必须降级为 ok=False,
    # 不能抛异常崩掉整个 benchmark 套件。下同其余 3 个客户端。
    try:
        data = r.json()
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"invalid JSON in 200 response: {type(e).__name__}: {e}",
            latency_ms=elapsed,
        )
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {}) or {}
    content = msg.get("content", "") or ""
    # Ollama 0.30+ Qwen3 thinking models put reasoning in a separate 'reasoning' field
    # and leave 'content' empty when max_tokens was exhausted during thinking.
    # Retry once with 4× max_tokens so the model has room to emit the actual answer.
    if not content and (msg.get("reasoning") or msg.get("thinking")):
        bumped = min(max(max_tokens * 4, 2048), 4096)
        if bumped > max_tokens:
            logger.debug("Ollama thinking model returned empty content (reasoning-only); "
                         "retrying with max_tokens=%d", bumped)
            retry_payload = dict(payload)
            retry_payload["max_tokens"] = bumped
            try:
                r2 = _post_with_retry(
                    url, retry_payload,
                    headers={"Authorization": model_cfg.auth_header},
                    timeout_s=timeout_s,
                    is_cloud=_is_cloud_endpoint(model_cfg),
                )
                if r2.status_code == 200:
                    d2 = r2.json()
                    c2 = (d2.get("choices", [{}])[0].get("message", {}) or {}).get("content", "") or ""
                    if c2:
                        content = c2
                        data = d2
                        choice = d2.get("choices", [{}])[0]
            except Exception:
                pass  # keep original empty content on retry failure
    # vLLM/OpenVINO Qwen3 thinking mode puts <think>...</think> blocks in content.
    # Strip them (including unclosed blocks when model hit max_tokens during reasoning).
    content = _strip_think_tags(content)
    usage = data.get("usage", {}) or {}
    input_tokens = int(usage.get("prompt_tokens", 0))
    output_tokens = int(usage.get("completion_tokens", 0))

    # 尝试解析 JSON（失败不算致命）
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0

    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=content,
        parsed_json=parsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=choice.get("finish_reason", ""),
        latency_ms=elapsed,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 流式客户端（测 TTFT）
# ────────────────────────────────────────────────────────────


def infer_stream(
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
    timeout_s: float = 600.0,
    seed: Optional[int] = None,
) -> InferResult:
    """流式推理，返回 TTFT + 总延迟 + usage"""
    prompt = _apply_prompt_prefix(prompt, model_cfg)
    messages: list[dict] = []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.effective_model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},   # vLLM 流式里带 usage
    }
    if seed is not None:
        payload["seed"] = seed   # 缓存 A/B 一致性校验需确定性采样(spec §11)
    _apply_ollama_think_controls(payload, model_cfg, max_tokens)
    _apply_chat_template_kwargs(payload, model_cfg)
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    ttft_ms = 0.0
    content_parts: list[str] = []
    input_tokens = 0
    output_tokens = 0
    finish_reason = ""
    first_token_seen = False

    try:
        with httpx.stream("POST", url, json=payload, timeout=timeout_s,
                         headers={"Authorization": model_cfg.auth_header,
                                  "Accept": "text/event-stream"}) as r:
            if r.status_code != 200:
                return InferResult(
                    model=model_cfg.name,
                    ok=False,
                    error=f"HTTP {r.status_code}",
                    latency_ms=(time.monotonic() - t0) * 1000,
                )
            for line in r.iter_lines():
                if not line or not line.startswith("data:"):
                    continue
                chunk_data = line[5:].strip()
                if chunk_data == "[DONE]":
                    break
                try:
                    chunk = json.loads(chunk_data)
                except json.JSONDecodeError:
                    continue
                # 捕获 TTFT
                choices = chunk.get("choices", [])
                if choices:
                    delta = choices[0].get("delta", {})
                    piece = delta.get("content", "")
                    if piece and not first_token_seen:
                        ttft_ms = (time.monotonic() - t0) * 1000
                        first_token_seen = True
                    if piece:
                        content_parts.append(piece)
                    if choices[0].get("finish_reason"):
                        finish_reason = choices[0]["finish_reason"]
                # usage 在最后一条 chunk（include_usage）
                usage = chunk.get("usage")
                if usage:
                    input_tokens = int(usage.get("prompt_tokens", 0))
                    output_tokens = int(usage.get("completion_tokens", 0))

        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    content = _strip_think_tags("".join(content_parts))
    # 解析 JSON（同 sync）
    parsed = None
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError:
        import re
        m = re.search(r"\{[\s\S]*\}", content)
        if m:
            try:
                parsed = json.loads(m.group(0))
            except Exception:
                pass

    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0

    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=content,
        parsed_json=parsed,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        finish_reason=finish_reason,
        latency_ms=elapsed,
        ttft_ms=ttft_ms,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 异步客户端（并发测试）
# ────────────────────────────────────────────────────────────


async def infer_async(
    client: httpx.AsyncClient,
    model_cfg: ModelConfig,
    *,
    prompt: str = VLM_PROMPT,
    image_path: Optional[Path] = None,
    max_tokens: int = 800,
    temperature: float = 0.1,
) -> InferResult:
    """异步版本，供并发测试调用"""
    prompt = _apply_prompt_prefix(prompt, model_cfg)
    messages: list[dict] = []
    if image_path and model_cfg.is_vlm:
        messages.append({
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": encode_image_data_url(image_path)}},
                {"type": "text", "text": prompt},
            ],
        })
    else:
        messages.append({"role": "user", "content": prompt})

    payload = {
        "model": model_cfg.effective_model_id,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    _apply_ollama_think_controls(payload, model_cfg, max_tokens)
    _apply_chat_template_kwargs(payload, model_cfg)
    url = f"{model_cfg.base_url}/chat/completions"

    t0 = time.monotonic()
    try:
        r = await client.post(url, json=payload,
                              headers={"Authorization": model_cfg.auth_header})
        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"{type(e).__name__}: {e}",
            latency_ms=(time.monotonic() - t0) * 1000,
        )

    if r.status_code != 200:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"HTTP {r.status_code}",
            latency_ms=elapsed,
        )
    try:
        data = r.json()
    except Exception as e:
        return InferResult(
            model=model_cfg.name,
            ok=False,
            error=f"invalid JSON in 200 response: {type(e).__name__}: {e}",
            latency_ms=elapsed,
        )
    choice = data.get("choices", [{}])[0]
    msg = choice.get("message", {}) or {}
    usage = data.get("usage", {}) or {}
    output_tokens = int(usage.get("completion_tokens", 0))
    tps = (output_tokens / (elapsed / 1000)) if elapsed > 0 and output_tokens > 0 else 0.0
    return InferResult(
        model=model_cfg.name,
        ok=True,
        content=_strip_think_tags(msg.get("content", "") or ""),
        input_tokens=int(usage.get("prompt_tokens", 0)),
        output_tokens=output_tokens,
        finish_reason=choice.get("finish_reason", ""),
        latency_ms=elapsed,
        tokens_per_sec=tps,
    )


# ────────────────────────────────────────────────────────────
# 健康检查
# ────────────────────────────────────────────────────────────


_LOCAL_PROVIDERS = {"local_vllm", "llama_cpp", "ollama", "generic"}


def wait_model_ready(model_cfg: ModelConfig, timeout_s: float = 300.0) -> bool:
    """等本地 server 就绪。云端 provider 直接视为就绪。

    本地 provider 包括：local_vllm / llama_cpp / ollama / generic。
    云端 provider（openai / deepseek / dashscope 等）直接返回 True，不轮询。
    若 model_cfg.readiness_url 设置，则轮询该 URL（用于自定义 API 如 ASR /health）。
    """
    if model_cfg.provider not in _LOCAL_PROVIDERS and _is_cloud_endpoint(model_cfg):
        return True  # cloud endpoints are always "ready"
    if model_cfg.readiness_url:
        # Custom readiness URL (e.g. ASR services that expose /health, not /v1/models)
        check_url = model_cfg.readiness_url
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                r = httpx.get(check_url, timeout=5.0)
                if r.status_code == 200:
                    return True
            except Exception:
                pass
            time.sleep(5)
        return False
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            r = httpx.get(f"{model_cfg.base_url}/models", timeout=5.0,
                          headers={"Authorization": model_cfg.auth_header})
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(5)
    return False


# Alias used by probe_provider.py and new unit tests.
wait_for_server = wait_model_ready


# ────────────────────────────────────────────────────────────
# Embedding 客户端（OpenAI 兼容 /v1/embeddings）
# ────────────────────────────────────────────────────────────


@dataclass
class EmbedResult:
    """单次 embedding 调用结果（一个或多个输入文本）。"""

    model: str
    ok: bool = False
    error: str = ""
    embeddings: Optional[list] = None  # list[list[float]]
    input_tokens: int = 0
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.embeddings is None:
            self.embeddings = []


def _embed_one_batch(
    model_cfg: ModelConfig,
    inputs: list,
    url: str,
    timeout_s: float,
) -> EmbedResult:
    """POST one /v1/embeddings request; always returns EmbedResult (ok may be False)."""
    _zero = EmbedResult(model=model_cfg.name, ok=False, error="", latency_ms=0.0)
    try:
        r = httpx.post(
            url,
            json={"model": model_cfg.effective_model_id, "input": inputs},
            timeout=timeout_s,
            headers={"Authorization": model_cfg.auth_header},
        )
    except Exception as exc:
        _zero.error = f"{type(exc).__name__}: {exc}"
        return _zero
    if r.status_code != 200:
        _zero.error = f"HTTP {r.status_code}: {getattr(r, 'text', '')[:200]}"
        return _zero
    # Some edge servers (rkllm3-server) return HTTP 200 + empty body for unsupported
    # batch sizes — getattr guard covers mock objects used in tests.
    if not getattr(r, "content", True):
        _zero.error = "empty body (server may not support batch input)"
        return _zero
    try:
        data = r.json()
    except Exception as exc:
        _zero.error = f"invalid json in 200 response: {type(exc).__name__}: {exc}"
        return _zero
    rows = sorted(data.get("data", []), key=lambda d: d.get("index", 0))
    vectors = [row.get("embedding", []) for row in rows]
    usage = data.get("usage", {}) or {}
    return EmbedResult(
        model=model_cfg.name,
        ok=bool(vectors),
        embeddings=vectors,
        error="" if vectors else "no embedding data in response",
        input_tokens=int(usage.get("prompt_tokens", 0) or usage.get("total_tokens", 0)),
        latency_ms=0.0,
    )


def infer_embedding(
    model_cfg: ModelConfig,
    inputs,
    *,
    timeout_s: float = 600.0,
) -> EmbedResult:
    """调用 OpenAI 兼容 /v1/embeddings，返回向量 + 延迟 + token 统计。

    ``inputs`` 可以是单个字符串或字符串列表。vLLM / sglang / llama.cpp server /
    Ollama 等都暴露这个端点；服务端不支持时返回 ok=False（调用方据此 graceful）。

    某些 RKNN/edge 服务端（如 rkllm3-server）只接受 batch_size=1 的请求，批量输入
    时返回 HTTP 200 空 body。检测到这种情况时自动降级为逐条请求（sequential fallback）。
    """
    if isinstance(inputs, str):
        inputs = [inputs]
    inputs = list(inputs)
    url = f"{model_cfg.base_url}/embeddings"

    t0 = time.monotonic()

    # Try full batch first (standard for vLLM / Ollama / llama.cpp)
    result = _embed_one_batch(model_cfg, inputs, url, timeout_s)
    if result.ok and len(result.embeddings) == len(inputs):
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    # Single-item input that failed: propagate error directly (no point in fallback)
    if len(inputs) == 1:
        result.latency_ms = (time.monotonic() - t0) * 1000
        return result

    # Multi-item batch failed (empty body / JSON error): fall back to sequential.
    # This handles RKNN/edge servers that only support batch_size=1.
    logger.debug("embedding batch failed for %s (%s); falling back to sequential (n=%d)",
                 model_cfg.name, result.error, len(inputs))
    all_vecs: list = []
    total_tokens = 0
    for text in inputs:
        r = _embed_one_batch(model_cfg, [text], url, timeout_s)
        if not r.ok or not r.embeddings:
            r.latency_ms = (time.monotonic() - t0) * 1000
            return r
        all_vecs.append(r.embeddings[0])
        total_tokens += r.input_tokens

    if not all_vecs:
        return EmbedResult(model=model_cfg.name, ok=False,
                           error="no embeddings produced",
                           latency_ms=(time.monotonic() - t0) * 1000)
    return EmbedResult(
        model=model_cfg.name,
        ok=True,
        embeddings=all_vecs,
        input_tokens=total_tokens,
        latency_ms=(time.monotonic() - t0) * 1000,
    )


# ────────────────────────────────────────────────────────────
# Rerank 客户端（OpenAI 兼容 /v1/rerank — BERT cross-encoder 单 pass）
# ────────────────────────────────────────────────────────────


@dataclass
class RerankResult:
    """One native ``/v1/rerank`` call: a relevance score per input document.

    Distinct from the generative yes/no reranker proxy: a BERT cross-encoder
    served by llama.cpp ``--reranking`` (``--pooling rank``) / vLLM scores the
    whole candidate list in a single pass. ``scores`` is aligned to the input
    ``documents`` order (re-sorted by the response ``index`` so backend ordering
    never silently shuffles relevance).
    """

    model: str
    ok: bool = False
    error: str = ""
    scores: Optional[list] = None  # list[float], aligned to input documents
    latency_ms: float = 0.0

    def __post_init__(self) -> None:
        if self.scores is None:
            self.scores = []


def infer_rerank(
    model_cfg: ModelConfig,
    query: str,
    documents,
    *,
    timeout_s: float = 600.0,
) -> RerankResult:
    """Call a native ``/v1/rerank`` endpoint; return per-document relevance scores.

    Single forward pass over ``[CLS] query [SEP] doc [SEP]`` for each document —
    no generation, no KV cache. The GGUF model carries its own tokenizer, so the
    serving host needs no Python ``transformers``/``tokenizers`` (the path that
    unblocks reranking on tokenizer-less edge devices). Endpoints that don't
    expose ``/v1/rerank`` return ok=False so the caller degrades gracefully.

    Response contract (cohere-style, as served by llama.cpp / vLLM / sglang):
      ``{"results": [{"index": i, "relevance_score": s}, ...]}``
    """
    documents = list(documents)
    if not documents:
        return RerankResult(model=model_cfg.name, ok=True, scores=[])
    payload = {"model": model_cfg.effective_model_id, "query": query, "documents": documents}
    url = f"{model_cfg.base_url}/rerank"

    t0 = time.monotonic()
    try:
        r = httpx.post(url, json=payload, timeout=timeout_s,
                       headers={"Authorization": model_cfg.auth_header})
        elapsed = (time.monotonic() - t0) * 1000
    except Exception as e:
        return RerankResult(model=model_cfg.name, ok=False,
                            error=f"{type(e).__name__}: {e}",
                            latency_ms=(time.monotonic() - t0) * 1000)

    if r.status_code != 200:
        return RerankResult(model=model_cfg.name, ok=False,
                            error=f"HTTP {r.status_code}: {r.text[:200]}",
                            latency_ms=elapsed)

    try:
        data = r.json()
    except Exception as e:
        return RerankResult(model=model_cfg.name, ok=False,
                            error=f"invalid JSON in 200 response: {type(e).__name__}: {e}",
                            latency_ms=elapsed)
    # Accept "results" (cohere/llama.cpp) or "data" (some OpenAI-style backends).
    rows = data.get("results", data.get("data", []))
    scores = [0.0] * len(documents)
    seen = 0
    for row in rows:
        idx = int(row.get("index", -1))
        if not (0 <= idx < len(documents)):
            continue
        # llama.cpp/cohere use "relevance_score"; some use "score".
        scores[idx] = float(row.get("relevance_score", row.get("score", 0.0)))
        seen += 1
    if seen != len(documents):
        return RerankResult(model=model_cfg.name, ok=False,
                            error=f"rerank returned {seen} scores != {len(documents)} docs",
                            scores=scores, latency_ms=elapsed)
    return RerankResult(model=model_cfg.name, ok=True, scores=scores, latency_ms=elapsed)


# ────────────────────────────────────────────────────────────
# VRAM 监控（可选）
# ────────────────────────────────────────────────────────────


def get_vram_info(device_index: int = 0) -> dict:
    """使用 pynvml 读 GPU 显存；pynvml 未装则返回空 dict"""
    try:
        import pynvml  # type: ignore
        pynvml.nvmlInit()
        h = pynvml.nvmlDeviceGetHandleByIndex(device_index)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "used_mb": mem.used // (1024**2),
            "total_mb": mem.total // (1024**2),
            "used_ratio": mem.used / mem.total,
        }
    except Exception:
        return {}


# ────────────────────────────────────────────────────────────
# 统计工具
# ────────────────────────────────────────────────────────────


def percentile(values: list[float], p: float) -> float:
    """线性插值 percentile（numpy 默认 'linear' 法）。

    截断索引法在小样本下严重失真（N=5 时 p95 直接取 max），benchmark
    尾延迟 / TPS 统计必须插值。
    """
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (len(s) - 1) * p / 100.0
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (rank - lo)


def summarize_latencies(latencies_ms: list[float]) -> dict:
    if not latencies_ms:
        return {"p50": 0, "p95": 0, "p99": 0, "min": 0, "max": 0, "count": 0}
    s = sorted(latencies_ms)
    return {
        "p50": percentile(s, 50),
        "p95": percentile(s, 95),
        "p99": percentile(s, 99),
        "min": s[0],
        "max": s[-1],
        "count": len(s),
    }


def proc_rss_mb(pid: int) -> float:
    """读 /proc/<pid>/status VmHWM（峰值 RSS，MB）。

    用于 embedding/asr 模块区分「批量 RSS」（含大 logical-batch KV）与「常驻查询
    RSS」（产品对话查询路径的真实内存）。仅在能读到目标进程的本机有效；读不到
    （远端服务 / 无权限 / 非 Linux）返回 0.0，由调用方标注 unavailable。
    """
    try:
        with open(f"/proc/{pid}/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024.0  # kB -> MB
    except Exception:
        pass
    return 0.0
