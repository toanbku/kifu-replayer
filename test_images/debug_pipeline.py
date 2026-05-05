"""Step-by-step debug of the pipeline."""
import os
import sys
import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from board_detector import auto_crop_board
from grid_detector import detect_grid, grid_spacing
from stone_detector import classify_stones

img = cv2.imread(os.path.join(HERE, "synthetic_kifu.png"))
print("Image shape:", img.shape)

cropped, bbox = auto_crop_board(img)
print("Bbox:", bbox)
print("Cropped shape:", cropped.shape)

xs, ys = detect_grid(cropped)
print("Grid xs:", xs)
print("Grid ys:", ys)
spacing = grid_spacing(xs, ys)
print("Spacing:", spacing)

samples = classify_stones(cropped, xs, ys, spacing)
counts = {"empty": 0, "B": 0, "W": 0}
for s in samples:
    counts[s["color"]] += 1
print("Counts:", counts)

# Show distribution of mean intensities
import collections
means = [s["mean"] for s in samples]
print("Mean stats:", min(means), np.median(means), max(means))
print("p20=", np.percentile(means, 20), "p80=", np.percentile(means, 80))

# Save overlay
out = cropped.copy()
for x in xs:
    cv2.line(out, (int(x), 0), (int(x), out.shape[0]-1), (0, 0, 255), 1)
for y in ys:
    cv2.line(out, (0, int(y)), (out.shape[1]-1, int(y)), (0, 0, 255), 1)
for s in samples:
    if s["color"] == "B":
        col = (255, 0, 0)
    elif s["color"] == "W":
        col = (0, 255, 0)
    else:
        col = (0, 0, 0)
    cv2.circle(out, (int(s["cx"]), int(s["cy"])), 4, col, -1)

cv2.imwrite(os.path.join(HERE, "debug_overlay.png"), out)
print("Wrote debug_overlay.png")
