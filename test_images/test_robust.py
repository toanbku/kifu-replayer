"""Robustness checks: re-run extraction on noisy / blurred / re-scaled
versions of the synthetic kifu image. Aim: still >= 95% accuracy."""
from __future__ import annotations
import io
import os
import sys
import json
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from kifu_extractor import extract_kifu  # noqa


def to_bytes(img) -> bytes:
    ok, b = cv2.imencode(".png", img)
    return b.tobytes()


def evaluate(name: str, img: np.ndarray, gt: dict) -> tuple[int, int]:
    LETTERS = "ABCDEFGHJKLMNOPQRST"
    expected = {}
    for m in gt["moves"]:
        co = m["coord"]
        rc = (19 - int(co[1:]), LETTERS.index(co[0]))
        expected[rc] = (m["n"], m["color"])
    res = extract_kifu(to_bytes(img))
    detected = {(m["row"], m["col"]): (m["n"], m["color"]) for m in res["moves"]}
    pos = sum(1 for rc in expected if rc in detected)
    color = sum(1 for rc, (en, ec) in expected.items() if rc in detected and detected[rc][1] == ec)
    n_match = sum(1 for rc, (en, ec) in expected.items() if rc in detected and detected[rc][0] == en)
    total = len(expected)
    print(
        f"  {name:25s}  pos {pos}/{total}  color {color}/{total}  num {n_match}/{total}  "
        f"({n_match*100/total:.1f}%)"
    )
    return n_match, total


def main():
    img = cv2.imread(os.path.join(HERE, "synthetic_kifu.png"))
    with open(os.path.join(HERE, "ground_truth.json")) as f:
        gt = json.load(f)

    print("Running robustness suite...")

    cases = []
    cases.append(("baseline", img.copy()))
    # JPEG-compressed reload
    ok, jpg = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 65])
    cases.append(("jpeg-q65", cv2.imdecode(jpg, cv2.IMREAD_COLOR)))
    # Down-scaled to 600x600 then back up
    small = cv2.resize(img, (600, 600), interpolation=cv2.INTER_AREA)
    cases.append(("downsampled-600", cv2.resize(small, (1200, 1200), interpolation=cv2.INTER_CUBIC)))
    # Slight blur
    cases.append(("blur-sigma1.5", cv2.GaussianBlur(img, (0, 0), sigmaX=1.5)))
    # Brightness shift (darken)
    dark = (img.astype(np.int16) - 25).clip(0, 255).astype(np.uint8)
    cases.append(("darkened", dark))
    # Brightness shift (brighten)
    bright = (img.astype(np.int16) + 25).clip(0, 255).astype(np.uint8)
    cases.append(("brightened", bright))
    # Small padding (center board on a larger green canvas, simulating background)
    h, w = img.shape[:2]
    pad = 100
    canvas = np.zeros((h + 2 * pad, w + 2 * pad, 3), dtype=np.uint8)
    canvas[:] = (60, 100, 60)  # tatami-ish dark green
    canvas[pad:pad + h, pad:pad + w] = img
    cases.append(("on-green-bg", canvas))

    total_correct = 0
    total = 0
    for name, c in cases:
        nm, tot = evaluate(name, c, gt)
        total_correct += nm
        total += tot
    print(f"\nOverall: {total_correct}/{total}  =  {total_correct*100/total:.2f}%")


if __name__ == "__main__":
    main()
