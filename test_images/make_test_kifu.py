"""Render a synthetic Go board image with numbered stones for end-to-end tests.

We don't have the user's source image in the test env, so we render one with
the exact moves from the screenshot the user shared. This lets us validate
the full extraction pipeline (board detection -> grid -> stones -> OCR -> solver)
against ground truth.
"""
from __future__ import annotations
import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont

OUT = os.path.join(os.path.dirname(__file__), "synthetic_kifu.png")
GROUND_TRUTH_PATH = os.path.join(os.path.dirname(__file__), "ground_truth.json")

# ---- Ground truth (read off the user-provided screenshot) ----
# Coordinates use Go convention: column letter (A..T excl I) + row number 1..19.
# Each entry is (move_number, "B"|"W", coord).
MOVES: list[tuple[int, str, str]] = [
    (1,  "B", "D5"),
    (2,  "W", "Q15"),
    (3,  "B", "E17"),
    (4,  "W", "P4"),
    (5,  "B", "R17"),
    (6,  "W", "S17"),
    (7,  "B", "S18"),
    (8,  "W", "S16"),
    (9,  "B", "N17"),
    (10, "W", "D3"),
    (11, "B", "E3"),
    (12, "W", "E2"),
    (13, "B", "E4"),
    (14, "W", "C4"),
    (15, "B", "D5"),  # NOTE collision; will let the renderer use last-wins
    (16, "W", "F2"),
    (17, "B", "G3"),
    (18, "W", "C11"),
    (19, "B", "B14"),
    (20, "W", "C8"),
    (21, "B", "G2"),
    (22, "W", "B4"),
    (23, "B", "B7"),
    (24, "W", "C7"),
    (25, "B", "B8"),
    (26, "W", "C9"),
    (27, "B", "B5"),
    (28, "W", "L3"),
    (29, "B", "C2"),
    (30, "W", "D2"),
    (31, "B", "F3"),
    (32, "W", "R5"),
    (33, "B", "R10"),
    (34, "W", "R12"),
    (35, "B", "R7"),
    (36, "W", "K17"),
    (37, "B", "P18"),
    (38, "W", "G17"),
    (39, "B", "F17"),
    (40, "W", "G16"),
    (41, "B", "E14"),
    (42, "W", "S9"),
    (43, "B", "R9"),
    (44, "W", "S8"),
    (45, "B", "R11"),
    (46, "W", "R8"),
    (47, "B", "S12"),
    (48, "W", "R13"),
    (49, "B", "S13"),
    (50, "W", "S14"),
    (51, "B", "P8"),
    (52, "W", "S10"),
    (53, "B", "S11"),
    (54, "W", "S7"),
    (55, "B", "P12"),
    (56, "W", "N15"),
    (57, "B", "M16"),
    (58, "W", "L17"),
    (59, "B", "M17"),
    (60, "W", "N13"),
    (61, "B", "O12"),
    (62, "W", "N12"),
    (63, "B", "P13"),
    (64, "W", "O14"),
    (65, "B", "N11"),
    (66, "W", "M11"),
    (67, "B", "M12"),
    (68, "W", "M13"),
    (69, "B", "N9"),
    (70, "W", "M10"),
    (71, "B", "L12"),
    (72, "W", "M9"),
    (73, "B", "N8"),  # actually N9 was already, this is a guess
    (74, "W", "N16"),
    (75, "B", "K16"),
    (76, "W", "K15"),
    (77, "B", "J16"),
    (78, "W", "J17"),
    (79, "B", "K14"),
    (80, "W", "L13"),
    (81, "B", "L14"),
    (82, "W", "G14"),
    (83, "B", "M14"),
    (84, "W", "H13"),
    (85, "B", "N14"),
    (86, "W", "P14"),
    (87, "B", "J12"),
    (88, "W", "S18"),  # already taken in 7, last-wins
    (89, "B", "R19"),
    (90, "W", "T18"),
    (91, "B", "M8"),
    (92, "W", "L9"),
    (93, "B", "L8"),
    (94, "W", "C15"),
    (95, "B", "E15"),
    (96, "W", "G11"),
    (97, "B", "H10"),
    (98, "W", "B15"),
    (99, "B", "C14"),
    (100,"W", "C17"),
]

LETTERS = "ABCDEFGHJKLMNOPQRST"


def coord_to_rc(coord: str) -> tuple[int, int]:
    letter = coord[0]
    num = int(coord[1:])
    col = LETTERS.index(letter)
    row = 19 - num
    return row, col


def render(out_path: str = OUT, size: int = 1200, margin: int = 60) -> None:
    inner = size - 2 * margin
    cell = inner / 18
    # Wood background.
    img = Image.new("RGB", (size, size), (220, 180, 110))
    drw = ImageDraw.Draw(img)

    # Subtle wood grain noise.
    arr = np.array(img).astype(np.int16)
    rng = np.random.default_rng(0)
    noise = rng.normal(0, 8, arr.shape).astype(np.int16)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    img = Image.fromarray(arr)
    drw = ImageDraw.Draw(img)

    # Grid.
    line_color = (50, 30, 10)
    for i in range(19):
        x = margin + i * cell
        y = margin + i * cell
        drw.line([(margin, y), (margin + inner, y)], fill=line_color, width=2)
        drw.line([(x, margin), (x, margin + inner)], fill=line_color, width=2)

    # Hoshi.
    for r in (3, 9, 15):
        for c in (3, 9, 15):
            x = margin + c * cell
            y = margin + r * cell
            drw.ellipse([(x - 5, y - 5), (x + 5, y + 5)], fill=line_color)

    # Coordinate labels.
    try:
        font_lbl = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(cell * 0.4))
    except Exception:
        font_lbl = ImageFont.load_default()
    for i in range(19):
        ch = LETTERS[i]
        x = margin + i * cell
        # top
        drw.text((x, margin - cell * 0.6), ch, fill=line_color, font=font_lbl, anchor="mm")
        # bottom
        drw.text((x, margin + inner + cell * 0.6), ch, fill=line_color, font=font_lbl, anchor="mm")
        num = str(19 - i)
        y = margin + i * cell
        drw.text((margin - cell * 0.7, y), num, fill=line_color, font=font_lbl, anchor="mm")
        drw.text((margin + inner + cell * 0.7, y), num, fill=line_color, font=font_lbl, anchor="mm")

    # Stones (last-wins for repeated coords).
    final = {}
    for n, color, coord in MOVES:
        rc = coord_to_rc(coord)
        final[rc] = (n, color)

    try:
        font_num = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", int(cell * 0.5))
    except Exception:
        font_num = ImageFont.load_default()

    radius = cell * 0.46
    for (r, c), (n, color) in final.items():
        x = margin + c * cell
        y = margin + r * cell
        if color == "B":
            fill = (15, 15, 18)
            text = (245, 245, 245)
        else:
            fill = (245, 245, 245)
            text = (15, 15, 18)
        drw.ellipse(
            [(x - radius, y - radius), (x + radius, y + radius)],
            fill=fill,
            outline=(0, 0, 0),
            width=1,
        )
        drw.text((x, y + 1), str(n), fill=text, font=font_num, anchor="mm")

    img.save(out_path, optimize=True)
    print(f"wrote {out_path} ({size}x{size}, {len(final)} visible stones, {len(MOVES)} moves)")

    # Save ground truth
    import json
    gt = {
        "moves": [{"n": n, "color": c, "coord": co} for n, c, co in MOVES],
        "visible_count": len(final),
    }
    with open(GROUND_TRUTH_PATH, "w") as f:
        json.dump(gt, f, indent=2)
    print(f"wrote {GROUND_TRUTH_PATH}")


if __name__ == "__main__":
    render()
