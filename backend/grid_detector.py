"""Grid detection: find the 19x19 lattice of intersection points.

Strategy:
1. Adaptive-threshold the (cropped) board so dark grid lines + stone outlines
   become foreground.
2. Morphologically open with a long horizontal kernel to keep only horizontal
   line pixels; do the same with a vertical kernel for vertical lines.
3. Project the resulting line images onto each axis (sum rows / sum cols).
4. Find peaks in the projection -- those are the line positions.
5. Cluster nearby peaks and pick exactly 19 evenly-spaced positions.

If we can't find clean lines (e.g., board mostly covered with stones), we fall
back to fitting a 19-spaced grid by detecting stones and using their inferred
spacing.
"""
from __future__ import annotations
import cv2
import numpy as np
from scipy.signal import find_peaks


BOARD_SIZE = 19


def _project_lines(line_img: np.ndarray, axis: int) -> np.ndarray:
    """Sum pixels along the axis, returning a 1-D projection.

    axis=0 sums rows -> profile across columns (use to find vertical lines).
    axis=1 sums cols -> profile across rows (use to find horizontal lines).
    """
    return line_img.sum(axis=axis).astype(np.float32)


def _extract_line_positions(profile: np.ndarray, expected: int = BOARD_SIZE) -> list[int]:
    """Find `expected` evenly-spaced peaks in a 1D projection."""
    n = len(profile)
    if n == 0:
        return []
    # Smooth a bit to coalesce close peaks.
    k = max(3, n // 200)
    if k % 2 == 0:
        k += 1
    smooth = cv2.GaussianBlur(profile.reshape(-1, 1), (1, k), 0).flatten()

    # Estimate spacing: total span / (expected-1).
    # Set min distance between peaks to ~60% of expected spacing.
    est_spacing = n / (expected + 1)
    min_dist = max(3, int(est_spacing * 0.6))
    height_thresh = max(smooth.mean() * 1.2, np.percentile(smooth, 70))
    peaks, _ = find_peaks(smooth, distance=min_dist, height=height_thresh)
    return peaks.tolist()


def _pick_19_evenly_spaced(peaks: list[int], img_size: int) -> list[int]:
    """From a noisy list of peaks, pick 19 positions that look evenly spaced.

    Strategy: estimate spacing from median diff of consecutive peaks; predict
    expected positions starting from the first/last peak; snap each predicted
    position to the nearest actual peak.
    """
    if len(peaks) < 2:
        # Fall back to evenly dividing the image.
        return [int(round(i * img_size / (BOARD_SIZE + 1))) for i in range(1, BOARD_SIZE + 1)]

    peaks = sorted(peaks)

    if len(peaks) == BOARD_SIZE:
        return peaks

    # If we have too many peaks, the right answer is a dense subset that's
    # evenly spaced. If too few, we interpolate.
    diffs = np.diff(peaks)
    spacing = float(np.median(diffs))

    # Pick start and end as the first and last peak that are reasonable.
    start = peaks[0]
    end = peaks[-1]

    span = end - start
    estimated_count = round(span / spacing) + 1
    if estimated_count == BOARD_SIZE:
        # Use the actual range; place 19 positions evenly between start/end,
        # then snap each to the nearest detected peak.
        targets = np.linspace(start, end, BOARD_SIZE)
    else:
        # Build 19 positions centered on the median.
        center = (start + end) / 2
        spacing = span / max(estimated_count - 1, 1)
        targets = np.array([center + (i - (BOARD_SIZE - 1) / 2) * spacing for i in range(BOARD_SIZE)])

    snapped: list[int] = []
    peaks_arr = np.array(peaks)
    for t in targets:
        nearest_idx = int(np.argmin(np.abs(peaks_arr - t)))
        if abs(peaks_arr[nearest_idx] - t) < spacing * 0.4:
            snapped.append(int(peaks_arr[nearest_idx]))
        else:
            snapped.append(int(round(t)))
    return snapped


def detect_grid(board_img: np.ndarray) -> tuple[list[int], list[int]]:
    """Return (xs, ys) -- 19 column x-positions and 19 row y-positions."""
    h, w = board_img.shape[:2]
    gray = cv2.cvtColor(board_img, cv2.COLOR_BGR2GRAY)
    # Bring up grid contrast: tophat highlights dark thin strokes against bright background.
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    binary = cv2.adaptiveThreshold(
        blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 25, 10
    )

    # Suppress big stone blobs so they don't bias the projection.
    # Erode everything; thin grid lines vanish under heavy erosion, but stones still huge.
    # We then subtract the heavily-eroded blob mask from binary to remove stones.
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    blobs = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
    grid_only = cv2.subtract(binary, blobs)

    # Extract horizontal and vertical lines via long structuring elements.
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(20, w // 15), 1))
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(20, h // 15)))
    h_lines = cv2.morphologyEx(grid_only, cv2.MORPH_OPEN, h_kernel)
    v_lines = cv2.morphologyEx(grid_only, cv2.MORPH_OPEN, v_kernel)

    # Project and find peaks.
    h_profile = _project_lines(h_lines, axis=1)  # one value per row
    v_profile = _project_lines(v_lines, axis=0)  # one value per column
    h_peaks = _extract_line_positions(h_profile)
    v_peaks = _extract_line_positions(v_profile)

    ys = _pick_19_evenly_spaced(h_peaks, h)
    xs = _pick_19_evenly_spaced(v_peaks, w)
    return xs, ys


def grid_spacing(xs: list[int], ys: list[int]) -> float:
    """Return the median spacing between adjacent grid lines (in pixels)."""
    dx = np.median(np.diff(xs)) if len(xs) > 1 else 0
    dy = np.median(np.diff(ys)) if len(ys) > 1 else 0
    return float((dx + dy) / 2)
