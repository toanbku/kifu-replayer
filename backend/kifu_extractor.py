"""Top-level pipeline: image bytes in, structured kifu out."""
from __future__ import annotations
import io
import base64
import numpy as np
import cv2
from PIL import Image

from board_detector import auto_crop_board
from grid_detector import detect_grid, grid_spacing, BOARD_SIZE
from stone_detector import classify_stones
from number_ocr import read_numbers
from validator import assign_moves


def _row_to_go_y(row: int) -> int:
    """Visual row 0 (top) corresponds to Go coordinate y=19; row 18 -> y=1."""
    return BOARD_SIZE - row


def _col_to_letter(col: int) -> str:
    """Go uses A..T excluding I (so columns are A,B,C,D,E,F,G,H,J,K,L,M,N,O,P,Q,R,S,T)."""
    letters = "ABCDEFGHJKLMNOPQRST"
    return letters[col]


def extract_kifu(image_bytes: bytes, return_debug: bool = False) -> dict:
    """Run the full extraction pipeline.

    Returns a dict like:
    {
        'board_size': 19,
        'moves': [
            {'n': 1, 'color': 'B', 'row': 14, 'col': 3, 'go_coord': 'D5',
             'confidence': 91.4, 'ocr_supported': True, 'conflict': False},
            ...
        ],
        'warnings': [str, ...],
        'debug': {...}            # only if return_debug
    }
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        # Try PIL as fallback for unusual formats.
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img = cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)
    if img is None:
        raise ValueError("Could not decode image")

    # 1. Crop to board area.
    board_img, bbox = auto_crop_board(img)

    # 2. Detect 19x19 grid.
    xs, ys = detect_grid(board_img)
    spacing = grid_spacing(xs, ys)

    # 3. Classify each intersection (B / W / empty).
    samples = classify_stones(board_img, xs, ys, spacing)

    stones = [s for s in samples if s["color"] != "empty"]

    warnings: list[str] = []

    # 4. OCR numbers.
    stones_with_candidates = read_numbers(board_img, stones, spacing)

    # 5. Assign numbers using parity + uniqueness solver.
    moves_raw = assign_moves(stones_with_candidates)

    # 6. Build clean moves list with Go-style coordinates.
    moves = []
    for s in moves_raw:
        moves.append(
            {
                "n": s["number"],
                "color": s["color"],
                "row": s["row"],          # 0 = top of image
                "col": s["col"],          # 0 = left of image
                "go_x": s["col"] + 1,     # 1..19 left to right
                "go_y": _row_to_go_y(s["row"]),  # 1..19 bottom to top
                "go_coord": f"{_col_to_letter(s['col'])}{_row_to_go_y(s['row'])}",
                "confidence": round(s.get("confidence", 0.0), 1),
                "ocr_supported": bool(s.get("ocr_supported", False)),
                "conflict": bool(s.get("conflict", False)),
            }
        )

    # Sanity warnings.
    if len(moves) >= 2:
        # Sequence gap warning
        nums = [m["n"] for m in moves]
        max_n = max(nums)
        missing = sorted(set(range(1, max_n + 1)) - set(nums))
        if missing:
            warnings.append(
                f"{len(missing)} move number(s) missing from board (likely captured): "
                f"{missing[:10]}{' ...' if len(missing) > 10 else ''}"
            )
        low_conf = [m for m in moves if not m["ocr_supported"]]
        if low_conf:
            warnings.append(
                f"{len(low_conf)} move(s) had no OCR reading; numbers were inferred from parity/sequence."
            )

    result = {
        "board_size": BOARD_SIZE,
        "moves": moves,
        "total_stones": len(moves),
        "warnings": warnings,
    }

    if return_debug:
        # Encode debug overlay PNG showing detected grid + classifications.
        debug_img = _make_debug_overlay(board_img.copy(), xs, ys, samples, moves)
        ok, png = cv2.imencode(".png", debug_img)
        if ok:
            result["debug_overlay_png_b64"] = base64.b64encode(png.tobytes()).decode()
        result["debug"] = {
            "bbox": bbox,
            "grid_xs": xs,
            "grid_ys": ys,
            "spacing": spacing,
        }

    return result


def _make_debug_overlay(
    board_img: np.ndarray,
    xs: list[int],
    ys: list[int],
    samples: list[dict],
    moves: list[dict],
) -> np.ndarray:
    """Draw the detected grid + classified stones + assigned numbers on the image."""
    img = board_img
    h, w = img.shape[:2]
    # Draw grid lines (thin red).
    for x in xs:
        cv2.line(img, (int(x), 0), (int(x), h - 1), (0, 0, 255), 1)
    for y in ys:
        cv2.line(img, (0, int(y)), (w - 1, int(y)), (0, 0, 255), 1)

    by_pos = {(m["row"], m["col"]): m for m in moves}
    for s in samples:
        pos = (s["row"], s["col"])
        if s["color"] == "empty":
            continue
        m = by_pos.get(pos)
        cx, cy = int(s["cx"]), int(s["cy"])
        circle_color = (0, 255, 0) if (m and m["ocr_supported"]) else (0, 165, 255)
        cv2.circle(img, (cx, cy), 4, circle_color, -1)
        if m:
            label = str(m["n"])
            cv2.putText(
                img,
                label,
                (cx - 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.putText(
                img,
                label,
                (cx - 12, cy - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 0, 0),
                1,
                cv2.LINE_AA,
            )
    return img
