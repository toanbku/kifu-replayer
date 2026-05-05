"""Read the move number printed on each detected stone.

Black stones display white digits; white stones display black digits. We crop
a square centered on the stone, normalise it (so the digit ends up dark on a
light background), then pass the result to Tesseract in single-word mode
restricted to digits.

Performance note: tesseract has ~50ms startup per call, so we only do ONE
call per stone and rely on the smart validator to clean up OCR errors using
parity / sequence constraints. We DO try a couple of preprocessings, but only
keep one (the higher-confidence run) per call.
"""
from __future__ import annotations
import cv2
import numpy as np
import pytesseract
import re
from concurrent.futures import ThreadPoolExecutor


_TESS_CONFIG = "--psm 8 -c tessedit_char_whitelist=0123456789"


def _crop_stone(board_img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    h, w = board_img.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    return board_img[y0:y1, x0:x1].copy()


def _binarize(crop_bgr: np.ndarray, color: str) -> np.ndarray:
    """Return a binary image where the digit is BLACK on a WHITE background.

    Tesseract works best with that polarity and high contrast.
    """
    if crop_bgr.size == 0:
        return crop_bgr
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    # Upscale 4x with cubic interpolation for cleaner edges.
    big = cv2.resize(
        gray, (gray.shape[1] * 4, gray.shape[0] * 4), interpolation=cv2.INTER_CUBIC
    )
    _, t = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    h, w = t.shape
    # Crop to inner 70% — kills the stone outline so only digit pixels remain.
    pad_h = int(h * 0.15)
    pad_w = int(w * 0.15)
    t = t[pad_h : h - pad_h, pad_w : w - pad_w]
    # We want digit BLACK on WHITE.
    if color == "B":
        # Black stone -> the body is mostly black, digit is white.
        # After OTSU, the larger area (background = stone body) is one class.
        # Force background to white.
        if t.mean() < 127:
            t = cv2.bitwise_not(t)
    else:
        # White stone -> body is white, digit is black.
        if t.mean() < 127:
            t = cv2.bitwise_not(t)
    # Add a small white border to help tesseract.
    t = cv2.copyMakeBorder(t, 8, 8, 8, 8, cv2.BORDER_CONSTANT, value=255)
    return t


def _ocr_one(img: np.ndarray) -> tuple[int | None, float]:
    """Return best (number, confidence). None if no digits detected."""
    if img is None or img.size == 0:
        return None, 0.0
    try:
        data = pytesseract.image_to_data(
            img, config=_TESS_CONFIG, output_type=pytesseract.Output.DICT
        )
    except Exception:
        return None, 0.0
    best: tuple[int | None, float] = (None, -1.0)
    for i, txt in enumerate(data.get("text", [])):
        if not txt:
            continue
        digits = re.sub(r"\D", "", txt)
        if not digits:
            continue
        try:
            n = int(digits)
        except ValueError:
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, KeyError):
            conf = 0.0
        if conf > best[1]:
            best = (n, max(conf, 0.0))
    return best


def _ocr_stone(args):
    board_img, stone, half = args
    crop = _crop_stone(board_img, stone["cx"], stone["cy"], half)
    bin_img = _binarize(crop, stone["color"])
    n, conf = _ocr_one(bin_img)
    candidates = [(n, conf)] if n is not None else []
    return {**stone, "candidates": candidates}


def read_numbers(
    board_img: np.ndarray,
    stones: list[dict],
    spacing: float,
    workers: int = 8,
) -> list[dict]:
    """Annotate each stone with `candidates`: list of (number, confidence)."""
    half = max(6, int(round(spacing * 0.55)))
    inputs = [(board_img, s, half) for s in stones if s["color"] != "empty"]
    if not inputs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_ocr_stone, inputs))
