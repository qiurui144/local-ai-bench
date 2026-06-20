"""RapidOCRBackend：基于 PP-OCR v3 ONNX 的 OCR 推理后端。

不依赖 PaddlePaddle，仅需 rapidocr-onnxruntime + opencv + numpy。
支持中英文混合识别，支持输入分辨率缩放（降采样加速）。

安装（RISC-V / Linux）::

    apt install python3-opencv python3-pyclipper python3-shapely
    pip install rapidocr-onnxruntime --no-deps

安装（x86 / ARM64）::

    pip install rapidocr-onnxruntime

配置示例（models.yaml）::

    - name: ppocr-v3-ch
      type: ocr
      backend: rapidocr
      extra:
        lang: ch           # ch（中英混合）或 en（仅英文）
        input_scale: 1.0   # 1.0=原始, 0.5=50%降采样
        det_box_thresh: 0.3
"""

from __future__ import annotations

import time
from typing import List, Optional, Tuple

import numpy as np

from benchmark.llama_benchmark.core.base_model import AbstractModelBackend
from benchmark.llama_benchmark.core.config import BackendType
from benchmark.llama_benchmark.core.registry import register_backend
from benchmark.llama_benchmark.utils.logging import get_logger

logger = get_logger(__name__)


@register_backend(BackendType.RAPIDOCR.value)
class RapidOCRBackend(AbstractModelBackend):
    """RapidOCR PP-OCR v3 推理后端，支持中英文混合识别和分辨率缩放。"""

    def load(self) -> None:
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            raise ImportError(
                "请安装 rapidocr-onnxruntime:\n"
                "  RISC-V: apt install python3-opencv python3-pyclipper python3-shapely && "
                "pip install rapidocr-onnxruntime --no-deps\n"
                "  x86/ARM64: pip install rapidocr-onnxruntime"
            )

        lang = self.config.extra.get("lang", "ch")
        self._input_scale = float(self.config.extra.get("input_scale", 1.0))
        self._ocr = RapidOCR(lang=lang)
        self._model = self._ocr  # 使 is_loaded 属性生效
        logger.info(
            f"RapidOCRBackend 加载完成: lang={lang}, input_scale={self._input_scale}"
        )

    def unload(self) -> None:
        self._ocr = None
        self._model = None  # 使 is_loaded 属性失效
        logger.info(f"RapidOCRBackend 已释放: {self.config.name}")

    def recognize(
        self,
        image: np.ndarray,
        input_scale: Optional[float] = None,
    ) -> Tuple[List[Tuple[list, str, float]], float]:
        """识别 BGR 图像中的文字。

        Parameters
        ----------
        image : np.ndarray
            BGR 图像，shape (H, W, 3)。
        input_scale : float, optional
            分辨率缩放系数，None 使用模型配置中的默认值。
            1.0=原始，0.5=50%降采样（面积 1/4，速度提升约 2-4×）。

        Returns
        -------
        results : list[(box, text, confidence)]
            每个文字区域的包围框、文本、置信度。
        latency_ms : float
            推理时间（ms）。
        """
        self._require_loaded()
        import cv2

        scale = input_scale if input_scale is not None else self._input_scale
        if abs(scale - 1.0) > 1e-6:
            h, w = image.shape[:2]
            new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        t0 = time.perf_counter()
        try:
            raw_result, _ = self._ocr(image)
        except Exception as exc:
            latency_ms = (time.perf_counter() - t0) * 1000
            logger.error(
                "RapidOCR 推理异常（非空图像情况）：%s，图像 shape=%s，latency=%.1fms",
                exc, image.shape, latency_ms,
            )
            raise RuntimeError(f"RapidOCR 模型执行失败: {exc}") from exc
        latency_ms = (time.perf_counter() - t0) * 1000

        results = []
        if raw_result:
            for item in raw_result:
                if item and len(item) >= 2:
                    box, text = item[0], item[1]
                    conf = float(item[2]) if len(item) > 2 else 1.0
                    results.append((box, text, conf))

        if not results:
            logger.debug(
                "RapidOCR 未检测到文字（正常情况，如空白图像）：图像 shape=%s", image.shape
            )

        return results, latency_ms

    def get_text(self, image: np.ndarray, input_scale: Optional[float] = None) -> Tuple[str, float]:
        """识别图像中全部文字并拼接为字符串。

        Returns
        -------
        text : str
            拼接后的识别文本。
        latency_ms : float
            推理时间（ms）。
        """
        results, latency_ms = self.recognize(image, input_scale)
        text = " ".join(r[1] for r in results)
        return text, latency_ms
