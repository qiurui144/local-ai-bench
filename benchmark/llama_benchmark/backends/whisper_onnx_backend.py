"""WhisperOnnxBackend：FP32 ONNX Whisper 推理后端。

不依赖 torch / faster-whisper / whisper.cpp，仅需 onnxruntime + numpy + scipy。
支持模型：whisper-tiny / whisper-base / whisper-small（FP32 ONNX）。
适用于 RISC-V / 嵌入式 / 无 GPU 环境（如 SpacemiT K1）。

模型来源::

    https://huggingface.co/onnx-community/whisper-tiny
    https://huggingface.co/onnx-community/whisper-base

模型文件::

    {model_dir}/encoder_fp32.onnx   — encoder
    {model_dir}/decoder_fp32.onnx   — merged decoder (含 KV-cache)
"""

from __future__ import annotations

import math
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

# HuggingFace 模型下载地址模板（FP32 ONNX）
_HF_URLS = {
    "whisper-tiny": {
        "encoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/encoder_model_fp32.onnx",
        "decoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-tiny/resolve/main/onnx/decoder_model_merged_fp32.onnx",
    },
    "whisper-base": {
        "encoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/encoder_model_fp32.onnx",
        "decoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-base/resolve/main/onnx/decoder_model_merged_fp32.onnx",
    },
    "whisper-small": {
        "encoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-small/resolve/main/onnx/encoder_model_fp32.onnx",
        "decoder_fp32.onnx": "https://huggingface.co/onnx-community/whisper-small/resolve/main/onnx/decoder_model_merged_fp32.onnx",
    },
}

# Whisper 特殊 token
_SOT     = 50258   # <|startoftranscript|>
_LANG_EN = 50259   # <|en|>
_LANG_ZH = 50260   # <|zh|>
_TASK    = 50363   # <|transcribe|>
_NO_TS   = 50362   # <|notimestamps|>
_EOT     = 50257   # <|endoftext|>

# Mel 频谱参数（固定，与 Whisper 一致）
_N_MELS      = 80
_N_FFT       = 400
_HOP_LENGTH  = 160
_CHUNK_LEN_S = 30
_N_SAMPLES   = _CHUNK_LEN_S * 16000
_N_FRAMES    = _N_SAMPLES // _HOP_LENGTH


def _build_mel_filters() -> np.ndarray:
    """构建 80 个 Mel 滤波器（Hz → Mel → Hz 转换）。"""
    def hz_to_mel(f):
        return 2595 * math.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    sr, fmin, fmax = 16000, 0.0, 8000.0
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, _N_MELS + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])
    bin_freqs = np.fft.rfftfreq(_N_FFT, 1.0 / sr)
    fbank = np.zeros((_N_MELS, _N_FFT // 2 + 1))
    for m in range(_N_MELS):
        fl, fc, fr = freqs[m], freqs[m + 1], freqs[m + 2]
        for k, f in enumerate(bin_freqs):
            if fl <= f <= fc:
                fbank[m, k] = (f - fl) / (fc - fl)
            elif fc < f <= fr:
                fbank[m, k] = (fr - f) / (fr - fc)
    return fbank


_MEL_FILTERS = _build_mel_filters()
_HANN_WINDOW = np.hanning(_N_FFT).astype(np.float32)


def log_mel_spectrogram(audio: np.ndarray) -> np.ndarray:
    """audio: float32 (N,) → mel: float32 (1, 80, N_FRAMES)。"""
    if len(audio) < _N_SAMPLES:
        audio = np.pad(audio, (0, _N_SAMPLES - len(audio)))
    else:
        audio = audio[:_N_SAMPLES]
    # 末尾补零，防止最后几帧越界（每帧需要 N_FFT=400 个样本，而末尾步长只有 HOP_LENGTH=160）
    audio = np.pad(audio, (0, _N_FFT))
    frames = np.stack([
        audio[i * _HOP_LENGTH: i * _HOP_LENGTH + _N_FFT] * _HANN_WINDOW
        for i in range(_N_FRAMES)
    ])
    stft = np.fft.rfft(frames, n=_N_FFT)
    mag = np.abs(stft) ** 2
    mel = _MEL_FILTERS @ mag.T
    mel = np.log10(np.maximum(mel, 1e-10))
    mel = np.maximum(mel, mel.max() - 8.0)
    mel = (mel + 4.0) / 4.0
    return mel.astype(np.float32)[np.newaxis, :, :]


def _download_file(url: str, dst: Path, max_retries: int = 3) -> None:
    """将 url 流式下载到 dst，用临时文件 + rename 保证原子性。

    失败后指数退避重试（最多 max_retries 次），全部失败则抛出异常。
    """
    tmp = dst.with_suffix(".tmp")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    last_exc: Optional[Exception] = None
    for attempt in range(1, max_retries + 1):
        try:
            with urllib.request.urlopen(req, timeout=300) as resp:
                content_len = resp.headers.get("Content-Length")
                total_mb = int(content_len) // (1024 * 1024) if content_len else 0
                logger.info(
                    "[download] %s (%s MB) attempt %d/%d …",
                    dst.name, total_mb or "?", attempt, max_retries,
                )
                with tmp.open("wb") as f:
                    shutil.copyfileobj(resp, f, length=1 << 20)  # 1 MB chunks
            size_mb = tmp.stat().st_size // (1024 * 1024)
            tmp.rename(dst)
            logger.info("[ok] %s (%d MB)", dst.name, size_mb)
            return
        except Exception as exc:
            last_exc = exc
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "[retry] %s 下载失败（attempt %d/%d，%ds 后重试）：%s",
                    dst.name, attempt, max_retries, wait, exc,
                )
                time.sleep(wait)
    raise RuntimeError(f"下载 {dst.name} 失败（{max_retries} 次重试后放弃）：{last_exc}") from last_exc


def _download_model(model_name: str, model_dir: str, max_retries: int = 3) -> bool:
    """下载 FP32 ONNX 模型文件，已存在则跳过。失败返回 False。"""
    urls = _HF_URLS.get(model_name)
    if not urls:
        logger.warning("未知模型名称: %s，跳过下载", model_name)
        return os.path.exists(os.path.join(model_dir, "encoder_fp32.onnx"))

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    for fname, url in urls.items():
        dst = Path(model_dir) / fname
        if dst.exists() and dst.stat().st_size > 1024:
            logger.info("[skip] %s (%d MB)", fname, dst.stat().st_size // (1024 * 1024))
            continue
        try:
            _download_file(url, dst, max_retries=max_retries)
        except RuntimeError as exc:
            logger.error("%s", exc)
            return False
    return True


@register_backend(BackendType.WHISPER_ONNX.value)
class WhisperOnnxBackend(AbstractModelBackend):
    """ONNX Whisper 推理后端（FP32，支持 tiny/base/small）。

    配置示例（models.yaml）::

        - name: whisper-base-onnx
          type: asr
          backend: whisper_onnx
          path: /home/user/models/whisper-base-onnx   # 留空则自动下载
          extra:
            model_name: whisper-base   # 用于自动下载
            language: en               # en / zh
            max_new_tokens: 100
    """

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        model_dir = str(self.config.path) if self.config.path else None
        model_name = self.config.extra.get("model_name", "whisper-tiny")

        if model_dir is None:
            # 默认存储路径
            model_dir = os.path.expanduser(f"~/llm-bench-models/{model_name}-onnx")

        if not _download_model(model_name, model_dir):
            raise RuntimeError(f"模型文件下载失败或不完整: {model_dir}")

        enc_path = os.path.join(model_dir, "encoder_fp32.onnx")
        dec_path = os.path.join(model_dir, "decoder_fp32.onnx")

        self._enc_sess = ort.InferenceSession(enc_path)
        self._dec_sess = ort.InferenceSession(dec_path)
        self._model_dir = model_dir
        self._dec_input_names = {i.name for i in self._dec_sess.get_inputs()}
        self._dec_output_names = [o.name for o in self._dec_sess.get_outputs()]

        # 从 decoder KV-cache 输出推断 n_layers 和 n_heads（精确，不依赖查表）
        # decoder 输出命名：present.{layer}.decoder.key / present.{layer}.decoder.value
        # present.{layer}.decoder.key shape: (1, n_heads, seq, head_dim)
        import re as _re
        n_layers = 0
        n_heads = 0
        head_dim = 64
        for out in self._dec_sess.get_outputs():
            m = _re.match(r"present\.(\d+)\.decoder\.key", out.name)
            if m:
                layer_idx = int(m.group(1))
                n_layers = max(n_layers, layer_idx + 1)
                if len(out.shape) >= 2:
                    n_heads = out.shape[1] if out.shape[1] and out.shape[1] > 0 else n_heads
                    if len(out.shape) >= 4 and out.shape[3]:
                        head_dim = out.shape[3]

        # fallback：从 encoder 输出的 hidden_dim 查表（shape 可能含动态维度 None）
        if n_layers == 0:
            n_state = self._enc_sess.get_outputs()[0].shape[-1]
            _dim_map = {384: (6, 4, 64), 512: (8, 6, 64), 768: (12, 12, 64)}
            n_heads, n_layers, head_dim = _dim_map.get(n_state, (8, 6, 64))
            logger.debug(
                "decoder KV 输出解析失败，使用 encoder dim=%d 查表：n_heads=%d, n_layers=%d",
                n_state, n_heads, n_layers,
            )

        self._n_heads, self._n_layers, self._head_dim = n_heads, n_layers, head_dim

        logger.info(
            "WhisperOnnxBackend 加载完成: %s (n_heads=%d, n_layers=%d, head_dim=%d)",
            model_name, self._n_heads, self._n_layers, self._head_dim,
        )

    def unload(self) -> None:
        self._enc_sess = None
        self._dec_sess = None
        logger.info(f"WhisperOnnxBackend 已释放: {self.config.name}")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        max_new_tokens: int = 100,
    ) -> Tuple[str, float, float]:
        """转录 float32 音频数组。

        Parameters
        ----------
        audio : np.ndarray
            float32 (N,)，采样率 16000 Hz。
        language : str, optional
            "en" 或 "zh"，覆盖配置中的默认语言。
        max_new_tokens : int
            最大生成 token 数。

        Returns
        -------
        text : str
            转录文本。
        enc_latency_ms : float
            encoder 推理时间（ms）。
        total_latency_ms : float
            总推理时间（encoder + decoder）（ms）。
        """
        self._require_loaded()
        lang = language or self.config.extra.get("language", "en")
        lang_token = _LANG_ZH if lang == "zh" else _LANG_EN
        max_tokens = self.config.extra.get("max_new_tokens", max_new_tokens)

        mel = log_mel_spectrogram(audio)

        # Encoder
        t0 = time.perf_counter()
        enc_out = self._enc_sess.run(None, {"input_features": mel})[0]
        enc_ms = (time.perf_counter() - t0) * 1000
        n_audio_ctx = enc_out.shape[1]

        # Decoder — greedy decode with KV-cache
        # KV-cache 命名：past_key_values.{layer}.decoder.{key/value}（self-attn）
        #              past_key_values.{layer}.encoder.{key/value}（cross-attn）
        # 输出命名：  present.{layer}.decoder.{key/value}
        #              present.{layer}.encoder.{key/value}
        t1 = time.perf_counter()
        tokens = np.array([[_SOT, lang_token, _TASK, _NO_TS]], dtype=np.int64)
        kv: dict[str, np.ndarray] = {}
        has_cache_branch = "use_cache_branch" in self._dec_input_names

        for step in range(max_tokens):
            feed: dict = {
                "input_ids": tokens[:, -1:],
                "encoder_hidden_states": enc_out,
            }
            if has_cache_branch:
                feed["use_cache_branch"] = np.array([step > 0])

            for layer in range(self._n_layers):
                for kv_type in ("key", "value"):
                    # self-attention KV
                    sa = f"past_key_values.{layer}.decoder.{kv_type}"
                    if sa in self._dec_input_names:
                        feed[sa] = kv.get(sa, np.zeros(
                            (1, self._n_heads, 0, self._head_dim), dtype=np.float32))
                    # cross-attention KV（encoder 输出，step>0 可复用）
                    ea = f"past_key_values.{layer}.encoder.{kv_type}"
                    if ea in self._dec_input_names:
                        feed[ea] = kv.get(ea, np.zeros(
                            (1, self._n_heads, n_audio_ctx, self._head_dim), dtype=np.float32))

            out = self._dec_sess.run(None, feed)
            logits = out[0]  # (1, seq, vocab)
            next_tok = int(np.argmax(logits[0, -1]))

            # 更新 KV-cache：present.X.Y → past_key_values.X.Y
            for i, name in enumerate(self._dec_output_names[1:], 1):
                inp = name.replace("present.", "past_key_values.")
                if inp in self._dec_input_names:
                    kv[inp] = out[i]

            if next_tok == _EOT:
                break
            tokens = np.concatenate([tokens, [[next_tok]]], axis=1)

        dec_ms = (time.perf_counter() - t1) * 1000
        total_ms = enc_ms + dec_ms

        # 简单 token 解码（无 tiktoken 时降级）
        gen_ids = tokens[0, 4:].tolist()
        text = self._decode_tokens(gen_ids)

        return text, enc_ms, total_ms

    def _decode_tokens(self, ids: list[int]) -> str:
        try:
            import tiktoken
            enc = tiktoken.get_encoding("gpt2")
            return enc.decode([i for i in ids if i < 50257]).strip()
        except Exception:
            return f"[{len(ids)} tokens]"
