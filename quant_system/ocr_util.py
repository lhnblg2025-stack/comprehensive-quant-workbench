"""
OCR 文字识别工具 — 替代 AI 识图 API，降低算力开销。

用法:
    from quant_system.ocr_util import ocr_image, ocr_image_path, is_ocr_error
    text = ocr_image_path("截图.png")
    if is_ocr_error(text):
        # 错误串以 "[OCR_ERROR:<CODE>]" 开头，不应作为正文入库
        ...
    text = ocr_image(np_array)  # numpy array / PIL Image

P2-Q28-fix(L371): 错误不再以含糊的 "[OCR…]" 字符串返回（可能被调用方当正文入库），
统一为结构化错误码格式 "[OCR_ERROR:<CODE>] message"，并提供 is_ocr_error() 判定。

依赖: tesseract (系统级), pytesseract (Python)
安装: sudo apt install tesseract-ocr tesseract-ocr-chi-sim
      pip install pytesseract Pillow numpy
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Union

logger = logging.getLogger("ocr_util")

# P2-Q28-fix(L371): 结构化错误码
OCR_ERROR_PREFIX = "[OCR_ERROR:"


def _ocr_error(code: str, message: str) -> str:
    """返回带结构化错误码的错误字符串，便于调用方通过 is_ocr_error() 识别。"""
    return f"{OCR_ERROR_PREFIX}{code}] {message}"


def is_ocr_error(text: str) -> bool:
    """判断 OCR 返回串是否为错误（而非识别正文）。"""
    return isinstance(text, str) and text.startswith(OCR_ERROR_PREFIX)

# 图片格式支持
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

try:
    import pytesseract
    HAS_TESSERACT = True
except ImportError:
    HAS_TESSERACT = False


def check_available() -> bool:
    """检查 OCR 环境是否就绪。"""
    if not HAS_TESSERACT:
        logger.warning("pytesseract 未安装")
        return False
    try:
        pytesseract.get_tesseract_version()
        return True
    except Exception:
        logger.warning("tesseract 系统命令不可用")
        return False


def ocr_image(
    image, lang: str = "chi_sim+eng",
    config: str = "--psm 6",
) -> str:
    """
    对 PIL Image / numpy ndarray 执行 OCR。

    Args:
        image: PIL.Image 或 numpy.ndarray (OpenCV/HWC)
        lang: 识别语言，默认中英文
        config: tesseract 配置，默认 psm 6（统一文本块）

    Returns:
        识别出的文本
    """
    if not HAS_TESSERACT:
        return _ocr_error("NO_TESSERACT", "pytesseract 未安装")
    if not HAS_PIL:
        return _ocr_error("NO_PIL", "Pillow 未安装")

    try:
        # 如果是 numpy 数组，转为 PIL
        if not isinstance(image, Image.Image):
            try:
                from PIL import Image as PILImage
                image = PILImage.fromarray(image)
            except Exception:
                return _ocr_error("IMG_CONVERT", "无法转换图片格式")

        text = pytesseract.image_to_string(image, lang=lang, config=config)
        return text.strip()
    except Exception as e:
        logger.error(f"OCR 识别失败: {e}")
        return _ocr_error("RECOGNITION_FAILED", str(e))


def ocr_image_path(
    path: Union[str, Path],
    lang: str = "chi_sim+eng",
    config: str = "--psm 6",
) -> str:
    """
    对图片文件路径执行 OCR。

    Args:
        path: 图片文件路径
        lang: 识别语言
        config: tesseract 配置

    Returns:
        识别出的文本
    """
    if not HAS_PIL:
        return _ocr_error("NO_PIL", "Pillow 未安装")
    try:
        image = Image.open(str(path))
        return ocr_image(image, lang=lang, config=config)
    except FileNotFoundError:
        return _ocr_error("FILE_NOT_FOUND", f"文件不存在: {path}")
    except Exception as e:
        logger.error(f"OCR 文件识别失败 {path}: {e}")
        return _ocr_error("RECOGNITION_FAILED", str(e))


def ocr_data_region(
    path: Union[str, Path],
    crop_box: tuple[int, int, int, int] | None = None,
    lang: str = "chi_sim+eng",
) -> dict:
    """
    对截图的某个区域做 OCR，返回结构化数据。
    适用于从截图提取表格/数字/列表。

    P2-Q28-fix(L371): docstring 与实际返回键对齐——实际返回 {text, words,
    word_count}（无 confidence/bboxes 键）；错误时返回 {text, error, error_code}。

    Args:
        path: 图片路径
        crop_box: (left, top, right, bottom) 裁剪区域
        lang: 识别语言

    Returns:
        {text, words, word_count} 或 {text, error, error_code}
    """
    if not HAS_PIL or not HAS_TESSERACT:
        return {"text": "", "error": "OCR 不可用", "error_code": "NO_DEPENDENCY"}

    try:
        image = Image.open(str(path))
        if crop_box:
            image = image.crop(crop_box)

        # 获取详细数据
        data = pytesseract.image_to_data(
            image, lang=lang, output_type=pytesseract.Output.DICT
        )
        words = []
        for i in range(len(data["text"])):
            if data["text"][i].strip():
                words.append({
                    "text": data["text"][i],
                    "conf": int(data["conf"][i]),
                    "x": data["left"][i],
                    "y": data["top"][i],
                    "w": data["width"][i],
                    "h": data["height"][i],
                })

        full_text = " ".join(w["text"] for w in words if w["conf"] > 0)
        return {
            "text": full_text,
            "words": words,
            "word_count": len(words),
        }
    except Exception as e:
        return {"text": "", "error": str(e), "error_code": "DATA_REGION_FAILED"}


def ocr_cli(path: str, lang: str = "chi_sim") -> str:
    """直接调用 tesseract CLI，避免 Python 库的兼容问题。"""
    try:
        result = subprocess.run(
            ["tesseract", path, "stdout", "-l", lang, "--psm", "6"],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return _ocr_error("CLI_ERROR", result.stderr[:200])
    except FileNotFoundError:
        return _ocr_error("CLI_UNAVAILABLE", "tesseract CLI 不可用")
    except subprocess.TimeoutExpired:
        return _ocr_error("CLI_TIMEOUT", "OCR 超时")
    except Exception as e:
        return _ocr_error("CLI_FAILED", str(e))


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("用法: python3 ocr_util.py <图片路径>")
        sys.exit(1)
    print(ocr_image_path(sys.argv[1]))
