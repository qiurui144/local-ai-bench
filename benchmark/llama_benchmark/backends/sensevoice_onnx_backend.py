"""SenseVoiceOnnxBackend：SenseVoice Small ONNX 推理后端。

不依赖 torch / funasr，仅需 onnxruntime + numpy。
支持中英文多语言，CTC 解码，无自回归循环，比 Whisper 快约 5-10×。

预装路径（SpaceMIT K1）::

    /usr/share/spacemit-asr/sensevoice/model_quant_optimized.onnx

模型规格::

    输入：speech (batch, T, 560)，speech_lengths (batch,)，language (batch,)，textnorm (batch,)
    输出：ctc_logits (batch, T', vocab)，encoder_out_lens (batch,)
    特征：80-dim fbank → LFR(m=7, n=6) → 560-dim
    词表：SentencePiece 25055 tokens（含中英）
"""

from __future__ import annotations

import math
import os
import time
from functools import lru_cache
from typing import Dict, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType, ModelConfig
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)

_DEFAULT_MODEL_PATH = "/usr/share/spacemit-asr/sensevoice/model_quant_optimized.onnx"
_DEFAULT_MVN_PATH   = "/usr/share/spacemit-asr/sensevoice/am.mvn"

# SenseVoice 语言 ID
_LANG_AUTO = 0
_LANG_ZH   = 3
_LANG_EN   = 4

# LFR（Low Frame Rate）参数，与官方预训练权重对齐
_LFR_M = 7   # 堆叠帧数
_LFR_N = 6   # 步长


@lru_cache(maxsize=4)
def _build_mel_matrix(sr: int, n_mels: int, n_fft: int) -> np.ndarray:
    """构建 Mel filterbank 矩阵 (n_mels, n_fft//2+1)，按参数缓存。

    使用 lru_cache 避免每次 transcribe() 调用重复计算（O(n_mels × n_fft) 双重循环）。
    """
    def hz_to_mel(f):
        return 2595 * math.log10(1 + f / 700)
    def mel_to_hz(m):
        return 700 * (10 ** (m / 2595) - 1)

    fmin, fmax = 0.0, sr / 2.0
    mel_min, mel_max = hz_to_mel(fmin), hz_to_mel(fmax)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    freqs = np.array([mel_to_hz(m) for m in mels])
    bin_freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    fbank_matrix = np.zeros((n_mels, n_fft // 2 + 1))
    for m in range(n_mels):
        fl, fc, fr = freqs[m], freqs[m + 1], freqs[m + 2]
        for k, f in enumerate(bin_freqs):
            if fl <= f <= fc:
                fbank_matrix[m, k] = (f - fl) / (fc - fl)
            elif fc < f <= fr:
                fbank_matrix[m, k] = (fr - f) / (fr - fc)
    return fbank_matrix


def _compute_fbank(audio: np.ndarray, sr: int = 16000,
                   n_mels: int = 80, n_fft: int = 512,
                   hop_length: int = 160) -> np.ndarray:
    """提取 fbank 特征，返回 (T, 80) float32。Mel 矩阵已缓存，不重复计算。"""
    fbank_matrix = _build_mel_matrix(sr, n_mels, n_fft)
    window = np.hanning(n_fft).astype(np.float32)

    n_frames = max(1, (len(audio) - n_fft) // hop_length + 1)
    if len(audio) < n_fft:
        audio = np.pad(audio, (0, n_fft - len(audio)))

    frames = np.stack([
        audio[i * hop_length: i * hop_length + n_fft] * window
        for i in range(n_frames)
    ])
    mag = np.abs(np.fft.rfft(frames, n=n_fft)) ** 2
    mel = mag @ fbank_matrix.T
    log_mel = np.log(np.maximum(mel, 1e-10))
    # 全局 CMVN（均值/方差归一化），与 SenseVoice 训练预处理对齐
    mean = log_mel.mean(axis=0, keepdims=True)
    std = log_mel.std(axis=0, keepdims=True) + 1e-8
    return ((log_mel - mean) / std).astype(np.float32)


def _lfr_transform(feats: np.ndarray, lfr_m: int = _LFR_M,
                   lfr_n: int = _LFR_N) -> np.ndarray:
    """Low Frame Rate 特征堆叠，返回 (T', lfr_m * D)。"""
    T, D = feats.shape
    left_pad = lfr_m // 2
    padded = np.concatenate([np.tile(feats[:1], (left_pad, 1)), feats], axis=0)
    out = []
    t = 0
    while t * lfr_n < T:
        start = t * lfr_n
        chunk = padded[start: start + lfr_m]
        if len(chunk) < lfr_m:
            chunk = np.pad(chunk, ((0, lfr_m - len(chunk)), (0, 0)), mode="edge")
        out.append(chunk.reshape(-1))
        t += 1
    return np.array(out, dtype=np.float32)


def _ctc_greedy_decode(ctc_logits: np.ndarray) -> list[int]:
    """CTC greedy decode：去掉 blank(0) 和重复。"""
    token_ids = np.argmax(ctc_logits, axis=-1).tolist()
    decoded: list[int] = []
    prev = None
    for t in token_ids:
        if t != prev and t != 0:
            decoded.append(t)
        prev = t
    return decoded


@register_backend(BackendType.SENSEVOICE_ONNX.value)
class SenseVoiceOnnxBackend(AbstractModelBackend):
    """SenseVoice Small ONNX 推理后端（无 torch 依赖）。

    配置示例（models.yaml）::

        - name: sensevoice-small
          type: asr
          backend: sensevoice_onnx
          path: /usr/share/spacemit-asr/sensevoice/model_quant_optimized.onnx
          extra:
            language: auto   # auto / zh / en
    """

    def load(self) -> None:
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("请安装 onnxruntime: pip install onnxruntime")

        model_path = str(self.config.path) if self.config.path else _DEFAULT_MODEL_PATH
        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"SenseVoice 模型不存在: {model_path}\n"
                f"在 SpaceMIT K1 上请确认: {_DEFAULT_MODEL_PATH}"
            )

        try:
            self._sess = ort.InferenceSession(model_path)
        except Exception as exc:
            raise RuntimeError(
                f"SenseVoice ONNX 模型加载失败: {model_path}\n"
                f"原因: {exc}\n"
                f"请检查文件完整性或 onnxruntime 版本（当前：{ort.__version__}）"
            ) from exc
        self._input_names = {i.name for i in self._sess.get_inputs()}
        self._model = self._sess  # 使 is_loaded 属性生效
        logger.info("SenseVoiceOnnxBackend 加载完成: %s（inputs=%s）",
                    model_path, sorted(self._input_names))

    def unload(self) -> None:
        self._sess = None
        self._model = None  # 使 is_loaded 属性失效
        logger.info(f"SenseVoiceOnnxBackend 已释放: {self.config.name}")

    def transcribe(
        self,
        audio: np.ndarray,
        language: Optional[str] = None,
        sr: int = 16000,
    ) -> Tuple[str, float]:
        """转录 float32 音频数组。

        Parameters
        ----------
        audio : np.ndarray
            float32 (N,)，采样率 16000 Hz。
        language : str, optional
            "auto" / "zh" / "en"，None 使用配置默认值。
        sr : int
            采样率，默认 16000。

        Returns
        -------
        token_repr : str
            CTC 解码结果（token 数量）或文本（如有 SentencePiece 词表）。
        latency_ms : float
            推理时间（ms），不含特征提取。
        """
        self._require_loaded()

        lang_str = language or self.config.extra.get("language", "auto")
        lang_id = {"zh": _LANG_ZH, "en": _LANG_EN}.get(lang_str, _LANG_AUTO)

        # 特征提取
        feats = _compute_fbank(audio, sr)
        feats = _lfr_transform(feats)

        speech = feats[np.newaxis, :, :]                            # (1, T', 560)
        speech_lengths = np.array([feats.shape[0]], dtype=np.int32)
        language_arr = np.array([lang_id], dtype=np.int32)
        textnorm_arr = np.array([15], dtype=np.int32)               # ITN

        t0 = time.perf_counter()
        out = self._sess.run(None, {
            "speech":         speech,
            "speech_lengths": speech_lengths,
            "language":       language_arr,
            "textnorm":       textnorm_arr,
        })
        latency_ms = (time.perf_counter() - t0) * 1000

        ctc_logits = out[0][0]  # (T', vocab)
        token_ids = _ctc_greedy_decode(ctc_logits)
        text = self._decode_ids(token_ids)

        return text, latency_ms

    def _decode_ids(self, ids: list[int]) -> str:
        """尝试 SentencePiece 解码，失败则返回 token 统计并记录 debug 日志。"""
        try:
            import sentencepiece as spm
        except ImportError:
            logger.debug("sentencepiece 未安装，CTC 输出为 token 计数（pip install sentencepiece）")
            return f"[{len(ids)} tokens]"

        model_dir = os.path.dirname(
            str(self.config.path) if self.config.path else _DEFAULT_MODEL_PATH
        )
        sp_path = os.path.join(model_dir, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        if not os.path.exists(sp_path):
            logger.debug("SentencePiece 模型不存在: %s，CTC 输出为 token 计数", sp_path)
            return f"[{len(ids)} tokens]"

        try:
            sp = spm.SentencePieceProcessor()
            sp.Load(sp_path)
            return sp.Decode(ids).strip()
        except Exception as exc:
            logger.warning("SentencePiece 解码失败（%s），回退到 token 计数: %s", sp_path, exc)
            return f"[{len(ids)} tokens]"
