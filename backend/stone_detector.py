"""Classify each grid intersection as Black / White / Empty.

Approach: sample a thin annulus (ring) centered on each intersection, with the
ring sized to fall just inside the typical stone radius. We use a ring
intentionally:

* Stones in kifu images carry a printed move number at their CENTER, so the
  centre pixels are misleading (a white stone with a black "100" averages
  to a mid-gray instead of bright white).
* The ring at ~50% of grid spacing is on the stone body (clear B/W color)
  if a stone is there, or on the board surface (wood color) if not.

After computing the ring's median intensity for every intersection we cluster
the histogram into three groups: dark (Black stones), wood (Empty), bright
(White stones). Thresholds adapt to the actual board lighting.
"""
from __future__ import annotations
import cv2
import numpy as np

EMPTY = "empty"
BLACK = "B"
WHITE = "W"


def _ring_median(gray: np.ndarray, cx: int, cy: int, r_in: int, r_out: int) -> float:
    """Median intensity inside the annulus r_in <= dist <= r_out."""
    h, w = gray.shape
    x0, x1 = max(cx - r_out, 0), min(cx + r_out + 1, w)
    y0, y1 = max(cy - r_out, 0), min(cy + r_out + 1, h)
    if x1 <= x0 or y1 <= y0:
        return 128.0
    yy, xx = np.mgrid[y0:y1, x0:x1]
    d2 = (xx - cx) ** 2 + (yy - cy) ** 2
    mask = (d2 >= r_in * r_in) & (d2 <= r_out * r_out)
    vals = gray[y0:y1, x0:x1][mask]
    if vals.size == 0:
        return 128.0
    return float(np.median(vals))


def classify_stones(
    board_img: np.ndarray,
    xs: list[int],
    ys: list[int],
    spacing: float,
) -> list[dict]:
    """For each grid intersection, classify as empty/black/white."""
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)

    # Ring radii: capture pixels on the stone body but outside the printed
    # number area. Roughly 32%..43% of the grid spacing -- this stays inside
    # the stone (~46% radius) and clear of any 3-digit number that may
    # span the centre.
    r_in = max(2, int(round(spacing * 0.32)))
    r_out = max(r_in + 2, int(round(spacing * 0.43)))

    samples: list[dict] = []
    for ri, cy in enumerate(ys):
        for ci, cx in enumerate(xs):
            v = _ring_median(gray, cx, cy, r_in, r_out)
            samples.append({"row": ri, "col": ci, "cx": cx, "cy": cy, "ring": v})

    rings = np.array([s["ring"] for s in samples], dtype=np.float32)
    # The empty-board (wood) intensity is the dominant mode of the histogram.
    # Use the median of all samples as a robust estimate of board color.
    board_v = float(np.median(rings))

    # Stones are clearly darker (Black) or brighter (White) than the board.
    # Use generous absolute offsets, with a fallback to "everything more than
    # 30 below board is dark / 30 above is bright". On well-printed kifu
    # images the gap is far larger than 30; pick a threshold midway between
    # the wood mode and each tail mode.

    dark_pop = rings[rings < board_v - 15]
    bright_pop = rings[rings > board_v + 15]
    if dark_pop.size:
        dark_thr = float((board_v + np.median(dark_pop)) / 2.0)
    else:
        dark_thr = board_v - 30.0
    if bright_pop.size:
        bright_thr = float((board_v + np.median(bright_pop)) / 2.0)
    else:
        bright_thr = board_v + 30.0

    # Safety: ensure thresholds are at least board_v ± 20.
    dark_thr = min(dark_thr, board_v - 20.0)
    bright_thr = max(bright_thr, board_v + 20.0)

    for s in samples:
        v = s["ring"]
        if v <= dark_thr:
            s["color"] = BLACK
        elif v >= bright_thr:
            s["color"] = WHITE
        else:
            s["color"] = EMPTY
        s["mean"] = v  # keep for back-compat with existing callers
        s["std"] = 0.0
    return samples
