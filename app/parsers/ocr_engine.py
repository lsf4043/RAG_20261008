"""
OCR 引擎封装（PaddleOCR）

为什么单独抽一层：项目里原来的调用写死了 PaddleOCR 2.x 的参数
（``use_angle_cls`` / ``show_log`` / ``cls=True``）和返回结构，而实际安装的
3.x 已经改名并换了返回类型（``OCRResult`` 对象，靠 ``rec_texts`` / ``rec_scores``
取文本），于是“上传图片/扫描件”一调就抛参数错误。

这里把 2.x / 3.x 的差异收敛到一个函数里：
- 引擎只初始化一次（懒加载 + 锁），避免每次上传都重新加载模型；
- ``ocr_image()`` 统一返回按阅读顺序排好的文本行，并做置信度过滤。
"""
from __future__ import annotations

import threading
from typing import Any, List, Optional, Sequence, Tuple

from app.config import settings
from app.utils.logger import get_logger

logger = get_logger(__name__)

_engine: Any = None
_engine_lock = threading.Lock()


def get_ocr_engine() -> Any:
    """懒加载 PaddleOCR 引擎（2.x / 3.x 均可用）"""
    global _engine
    if _engine is not None:
        return _engine

    with _engine_lock:
        if _engine is not None:
            return _engine

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:  # pragma: no cover - 环境相关
            raise RuntimeError(
                "未安装 paddleocr：请执行 pip install 'paddleocr>=2.7,<3' 或安装 3.x 后重试"
            ) from exc

        # 3.x 用 use_textline_orientation，2.x 用 use_angle_cls；两者都不传 show_log
        for kwargs in (
            {"lang": "ch", "use_textline_orientation": True},
            {"lang": "ch", "use_angle_cls": True},
            {"lang": "ch"},
        ):
            try:
                _engine = PaddleOCR(**kwargs)
                logger.info("PaddleOCR 初始化成功，参数: %s", list(kwargs))
                return _engine
            except (TypeError, ValueError) as exc:
                logger.debug("PaddleOCR 参数 %s 不被支持: %s", list(kwargs), exc)
            except Exception as exc:  # noqa: BLE001 - 模型下载失败等
                logger.error("PaddleOCR 初始化失败: %s", exc)
                raise

        raise RuntimeError("PaddleOCR 初始化失败：没有可用的参数组合")


def _normalize_result(raw: Any) -> List[Tuple[str, float, Optional[Sequence[Sequence[float]]]]]:
    """
    把 2.x / 3.x 的返回结构统一成 [(text, score, box)]

    - 3.x: [OCRResult]，OCRResult.rec_texts / rec_scores / rec_polys
    - 2.x: [[ [box, (text, score)], ... ]]
    """
    items: List[Tuple[str, float, Optional[Sequence[Sequence[float]]]]] = []
    if raw is None:
        return items

    result_list = raw if isinstance(raw, (list, tuple)) else [raw]
    for page in result_list:
        if page is None:
            continue

        # ---- 3.x ----
        texts = getattr(page, "rec_texts", None)
        if texts is None and isinstance(page, dict):
            texts = page.get("rec_texts")
        if texts is not None:
            scores = getattr(page, "rec_scores", None)
            if scores is None and isinstance(page, dict):
                scores = page.get("rec_scores")
            boxes = getattr(page, "rec_polys", None)
            if boxes is None and isinstance(page, dict):
                boxes = page.get("rec_polys")
            scores = list(scores) if scores is not None else [1.0] * len(texts)
            boxes = list(boxes) if boxes is not None else [None] * len(texts)
            for text, score, box in zip(texts, scores, boxes):
                items.append((str(text), float(score), box))
            continue

        # ---- 2.x ----
        try:
            for line in page:
                box = line[0]
                text, score = line[1][0], line[1][1]
                items.append((str(text), float(score), box))
        except (TypeError, IndexError):
            logger.debug("无法识别的 OCR 返回结构: %s", type(page))
    return items


def _sort_reading_order(
    items: List[Tuple[str, float, Optional[Sequence[Sequence[float]]]]]
) -> List[str]:
    """按从上到下、从左到右还原阅读顺序"""
    located = []
    for text, score, box in items:
        try:
            ys = [float(point[1]) for point in box]  # type: ignore[index]
            xs = [float(point[0]) for point in box]  # type: ignore[index]
            y_center, x_center = sum(ys) / len(ys), sum(xs) / len(xs)
        except Exception:  # noqa: BLE001
            located.append((float("inf"), 0.0, text))
            continue
        located.append((y_center, x_center, text))

    if not located:
        return []

    located.sort(key=lambda item: (item[0], item[1]))

    lines: List[List[Tuple[float, float, str]]] = []
    for y, x, text in located:
        if lines and abs(y - lines[-1][0][0]) < 12:
            lines[-1].append((y, x, text))
        else:
            lines.append([(y, x, text)])

    result: List[str] = []
    for line in lines:
        line.sort(key=lambda item: item[1])
        joined = " ".join(part[2] for part in line if part[2])
        if joined.strip():
            result.append(joined)
    return result


def ocr_image(image_path: str, min_confidence: float | None = None) -> List[str]:
    """
    对单张图片做 OCR

    Returns:
        文本行列表（已按阅读顺序排序、已做置信度过滤）
    """
    threshold = settings.OCR_MIN_CONFIDENCE if min_confidence is None else min_confidence
    engine = get_ocr_engine()

    raw = None
    # 3.x 推荐 predict；2.x 只有 ocr
    if hasattr(engine, "predict"):
        try:
            raw = engine.predict(input=image_path)
        except TypeError:
            try:
                raw = engine.predict(image_path)
            except Exception as exc:  # noqa: BLE001
                logger.debug("predict(image_path) 失败: %s", exc)
        except Exception as exc:  # noqa: BLE001
            logger.debug("predict(input=...) 失败: %s", exc)

    if raw is None and hasattr(engine, "ocr"):
        try:
            raw = engine.ocr(image_path)
        except TypeError:
            raw = engine.ocr(image_path, cls=True)

    items = [(text, score, box) for text, score, box in _normalize_result(raw) if score >= threshold]
    if not items:
        return []
    return _sort_reading_order(items)
