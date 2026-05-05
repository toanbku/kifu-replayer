"""Read the move number printed on each detected stone.

Black stones display white digits; white stones display black digits. We crop
a square centered on the stone, normalise it (so the digit ends up dark on a
light background), then pass the result to Tesseract in single-word mode
restricted to digits.

We run a small set of complementary preprocessings (OTSU, adaptive mean,
sharpened, scale variants) and keep up to TOP_K candidates per stone with
their confidences. Multiple candidates give the validator more options to
disambiguate via parity / replay simulation.
"""
from __future__ import annotations
import cv2
import numpy as np
import pytesseract
import re
from concurrent.futures import ThreadPoolExecutor


_TESS_CONFIG_PSM8 = "--psm 8 -c tessedit_char_whitelist=0123456789"
_TESS_CONFIG_PSM7 = "--psm 7 -c tessedit_char_whitelist=0123456789"

# Each stone keeps up to this many distinct number candidates (different
# numbers, the highest confidence wins per number).
TOP_K = 4


def _crop_stone(board_img: np.ndarray, cx: int, cy: int, half: int) -> np.ndarray:
    h, w = board_img.shape[:2]
    x0, x1 = max(cx - half, 0), min(cx + half, w)
    y0, y1 = max(cy - half, 0), min(cy + half, h)
    return board_img[y0:y1, x0:x1].copy()


def _prep_variants(crop_bgr: np.ndarray, color: str) -> list[np.ndarray]:
    """Return a list of binarised crops to feed Tesseract.

    Each variant is BLACK digit on WHITE background, with whitespace padding.
    We keep a small, complementary set: OTSU (good for clean digits) and
    sharpen+OTSU (good for blurred / low-contrast digits). Adaptive
    thresholding tends to break stone outlines into the digit area; we don't
    use it.
    """
    variants: list[np.ndarray] = []
    if crop_bgr.size == 0:
        return variants
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)

    big = cv2.resize(
        gray, (gray.shape[1] * 4, gray.shape[0] * 4),
        interpolation=cv2.INTER_CUBIC,
    )

    # Path 1: OTSU global threshold.
    _, t_otsu = cv2.threshold(big, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Path 2: sharpen + OTSU. Helps thin/light digits.
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    sharp = cv2.filter2D(big, -1, kernel)
    _, t_sharp = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    for raw in (t_otsu, t_sharp):
        t = raw
        h, w = t.shape
        pad_h = int(h * 0.15)
        pad_w = int(w * 0.15)
        t = t[pad_h: h - pad_h, pad_w: w - pad_w]
        if t.size == 0:
            continue
        # Force background -> white.
        if t.mean() < 127:
            t = cv2.bitwise_not(t)
        t = cv2.copyMakeBorder(t, 12, 12, 12, 12, cv2.BORDER_CONSTANT, value=255)
        variants.append(t)
    return variants


def _ocr_one(img: np.ndarray, config: str) -> list[tuple[int, float]]:
    """Return ALL (number, confidence) pairs Tesseract emitted for this image."""
    if img is None or img.size == 0:
        return []
    try:
        data = pytesseract.image_to_data(
            img, config=config, output_type=pytesseract.Output.DICT
        )
    except Exception:
        return []
    out: list[tuple[int, float]] = []
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
        if not (1 <= n <= 999):
            continue
        try:
            conf = float(data["conf"][i])
        except (ValueError, KeyError):
            conf = 0.0
        out.append((n, max(conf, 0.0)))
    return out


def _merge_candidates(
    runs: list[list[tuple[int, float]]], top_k: int
) -> list[tuple[int, float]]:
    """Across multiple OCR runs, keep the highest-confidence value per number,
    then return the top_k by confidence (descending).

    A vote bonus: if a number shows up in multiple variants, we boost its
    confidence by sqrt(count) up to a small ceiling. Numbers that only show
    up once stay at their measured confidence.
    """
    by_num: dict[int, list[float]] = {}
    for run in runs:
        for n, c in run:
            by_num.setdefault(n, []).append(c)
    scored: list[tuple[int, float]] = []
    for n, confs in by_num.items():
        best = max(confs)
        # Mild vote bonus, capped so a single 95% reading doesn't get crushed
        # by three 30% misreads.
        bonus = min(15.0, 5.0 * (len(confs) - 1))
        scored.append((n, min(99.0, best + bonus)))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_k]


def _ocr_stone(args):
    board_img, stone, half = args
    crop = _crop_stone(board_img, stone["cx"], stone["cy"], half)
    variants = _prep_variants(crop, stone["color"])
    runs: list[list[tuple[int, float]]] = []
    # PSM 8 = single word (default); PSM 7 = single line. Try both on the
    # FIRST variant only; if that yields a high-confidence reading, skip
    # the second variant. This roughly halves the OCR time on easy boards.
    for idx, v in enumerate(variants):
        runs.append(_ocr_one(v, _TESS_CONFIG_PSM8))
        runs.append(_ocr_one(v, _TESS_CONFIG_PSM7))
        # Early exit if the merged top candidate is already very confident.
        merged = _merge_candidates(runs, 1)
        if merged and merged[0][1] >= 80.0:
            break
    candidates = _merge_candidates(runs, TOP_K)
    return {**stone, "candidates": candidates}


def read_numbers(
    board_img: np.ndarray,
    stones: list[dict],
    spacing: float,
    workers: int = 8,
) -> list[dict]:
    """Annotate each stone with `candidates`: list of (number, confidence)
    sorted by confidence descending, up to TOP_K entries.
    """
    half = max(6, int(round(spacing * 0.55)))
    inputs = [(board_img, s, half) for s in stones if s["color"] != "empty"]
    if not inputs:
        return []
    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_ocr_stone, inputs))
