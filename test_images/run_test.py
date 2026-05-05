"""Run the extractor on the synthetic kifu and report accuracy vs ground truth."""
from __future__ import annotations
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "backend"))

from kifu_extractor import extract_kifu  # noqa: E402


def main() -> int:
    img_path = os.path.join(HERE, "synthetic_kifu.png")
    gt_path = os.path.join(HERE, "ground_truth.json")
    with open(img_path, "rb") as f:
        data = f.read()
    with open(gt_path) as f:
        gt = json.load(f)

    result = extract_kifu(data, return_debug=False)
    moves = result["moves"]
    print(f"Detected {len(moves)} moves, {len(gt['moves'])} expected (visible {gt['visible_count']}).")
    print("Warnings:", result.get("warnings"))

    # Build expected final-position map (last-wins for repeated coords).
    LETTERS = "ABCDEFGHJKLMNOPQRST"
    def coord_to_rc(coord):
        return (19 - int(coord[1:]), LETTERS.index(coord[0]))

    expected_at_rc: dict[tuple[int,int], tuple[int, str]] = {}
    for m in gt["moves"]:
        expected_at_rc[coord_to_rc(m["coord"])] = (m["n"], m["color"])

    detected_at_rc = {(m["row"], m["col"]): (m["n"], m["color"]) for m in moves}

    # Position-level accuracy: did we detect a stone at every expected rc, with right color?
    pos_correct = 0
    color_correct = 0
    n_correct = 0
    n_correct_with_ocr = 0
    extra = []
    missing = []
    for rc, (en, ec) in expected_at_rc.items():
        if rc not in detected_at_rc:
            missing.append((rc, en, ec))
            continue
        pos_correct += 1
        dn, dc = detected_at_rc[rc]
        if dc == ec:
            color_correct += 1
        if dn == en:
            n_correct += 1
    for rc in detected_at_rc.keys() - expected_at_rc.keys():
        extra.append(rc)

    total_expected = len(expected_at_rc)
    print(f"Position accuracy:   {pos_correct}/{total_expected} = {pos_correct/total_expected*100:.1f}%")
    print(f"Color accuracy:      {color_correct}/{total_expected} = {color_correct/total_expected*100:.1f}%")
    print(f"Move-number accuracy:{n_correct}/{total_expected} = {n_correct/total_expected*100:.1f}%")
    print(f"Extra detections:    {len(extra)}")
    print(f"Missing detections:  {len(missing)}")
    if missing[:5]:
        print("  examples missing:", missing[:5])
    if extra[:5]:
        print("  examples extra:  ", extra[:5])

    # Compare per-stone numbers for the matched positions.
    wrong_numbers = []
    for rc, (en, ec) in expected_at_rc.items():
        if rc in detected_at_rc:
            dn, dc = detected_at_rc[rc]
            if dn != en:
                wrong_numbers.append({"rc": rc, "expected": en, "got": dn})
    if wrong_numbers:
        print(f"Wrong numbers ({len(wrong_numbers)}):")
        for w in wrong_numbers[:20]:
            print(" ", w)
    return 0


if __name__ == "__main__":
    sys.exit(main())
