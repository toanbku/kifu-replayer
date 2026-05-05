"""Board detection: find the playing area inside the image."""
from __future__ import annotations
import cv2
import numpy as np


def detect_board(img: np.ndarray) -> tuple[int, int, int, int]:
    """Find the bounding box of the Go board in the image.

    Returns (x0, y0, x1, y1) -- the rectangle that contains the wood-colored
    playing surface. We use HSV color filtering because Go boards are almost
    always a warm wood/orange color that contrasts strongly with stones and
    the (usually green/dark) background.
    """
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Wood/board color: hue in orange range, moderate-to-high saturation,
    # high value. We're permissive on hue to handle different board tints.
    lower = np.array([5, 40, 80])
    upper = np.array([35, 220, 255])
    mask = cv2.inRange(hsv, lower, upper)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        # Fallback: assume the whole image is the board.
        return 0, 0, w, h

    # Pick the largest contour by area.
    cnt = max(contours, key=cv2.contourArea)
    x, y, bw, bh = cv2.boundingRect(cnt)

    # Sanity check: board must occupy at least ~25% of the image area;
    # otherwise fall back to the full image.
    if bw * bh < 0.25 * w * h:
        return 0, 0, w, h
    # Pad slightly inside so we drop the wooden frame and label margin.
    return x, y, x + bw, y + bh


def auto_crop_board(img: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Crop the image down to the board bounding box. Returns (cropped, bbox)."""
    x0, y0, x1, y1 = detect_board(img)
    return img[y0:y1, x0:x1].copy(), (x0, y0, x1, y1)
