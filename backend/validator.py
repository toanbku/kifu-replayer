"""Smart validation & assignment of move numbers to detected stones.

This is the secret-sauce step that pushes accuracy toward 99%. Pure OCR on
small digits is unreliable, but we have very strong structural constraints we
can lean on:

* Every visible stone has exactly one move number.
* Numbers must be distinct.
* In Go, Black plays first, so:
    - odd numbers (1, 3, 5, ...) go on BLACK stones
    - even numbers (2, 4, 6, ...) go on WHITE stones
* The set of numbers should be (approximately) 1..N where N is the total
  visible stones; in games without captures this is exact.

Algorithm:
1. For each stone, build a list of *legal* candidates (parity matches color).
2. Set up a cost matrix where rows are stones and columns are numbers in
   {1, 2, ..., max_n}; cost = -OCR_confidence (penalising low confidence) or
   a large constant if the number is not in the stone's OCR candidate list.
3. Add columns for each number that no stone listed but which has the right
   parity for SOME stone (so the matching can fall back to "no OCR support
   but we know it must be there").
4. Solve as a linear-sum-assignment (Hungarian).
5. Post-process: re-flag any assignment with no OCR support as low-confidence
   so the UI can highlight it.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linear_sum_assignment


PENALTY_NO_OCR = 1000.0    # cost if stone never produced this number from OCR
PENALTY_PARITY_BAD = 1e6   # essentially forbidden


def _parity_ok(number: int, color: str) -> bool:
    if color == "B":
        return number % 2 == 1
    if color == "W":
        return number % 2 == 0
    return False


def assign_moves(stones_with_candidates: list[dict]) -> list[dict]:
    """Assign each stone a move number.

    Inputs: list of dicts with keys row, col, cx, cy, color, candidates.
    Returns: list of dicts with row, col, cx, cy, color, number, confidence,
             ocr_supported (bool), candidates.
    Sorted by assigned move number ascending.
    """
    n = len(stones_with_candidates)
    if n == 0:
        return []

    # Determine the universe of possible numbers: at least 1..n. If any OCR
    # candidate is larger than n we extend (handles captured games where
    # max_number > visible_stones).
    max_seen = max(
        (c[0] for s in stones_with_candidates for c in s.get("candidates", [])),
        default=0,
    )
    # Make sure we always have enough columns of EACH parity for the stones
    # we have to assign. Black stones need odd numbers; white stones need
    # even numbers. The parity-aware lower bound is 2 * max(B,W).
    n_black = sum(1 for s in stones_with_candidates if s.get("color") == "B")
    n_white = sum(1 for s in stones_with_candidates if s.get("color") == "W")
    parity_min = 2 * max(n_black, n_white)
    universe_size = max(n, max_seen, parity_min)

    # Cost matrix: rows = stones, cols = numbers 1..universe_size.
    # We want to MINIMISE cost. Lower cost = preferred.
    cost = np.full((n, universe_size), PENALTY_NO_OCR, dtype=np.float64)

    for i, stone in enumerate(stones_with_candidates):
        color = stone["color"]
        for num in range(1, universe_size + 1):
            if not _parity_ok(num, color):
                cost[i, num - 1] = PENALTY_PARITY_BAD
        for num, conf in stone.get("candidates", []):
            if 1 <= num <= universe_size and _parity_ok(num, color):
                # Convert confidence (0..100) into a small cost.
                # PENALTY_NO_OCR is the baseline; OCR-supported should be lower.
                # Use a smooth gradient so 100% conf -> ~0, 0% conf -> ~1.
                cost[i, num - 1] = (100.0 - max(conf, 0.0)) / 100.0

    # If the cost matrix is non-square, linear_sum_assignment selects n cols
    # out of universe_size to minimise total cost. That's exactly what we want
    # when universe_size > n (extra capture columns just stay unused).
    if universe_size < n:
        # Pad columns with PENALTY_PARITY_BAD so there are at least n cols.
        pad = np.full((n, n - universe_size), PENALTY_PARITY_BAD)
        cost = np.hstack([cost, pad])
        universe_size = n

    row_ind, col_ind = linear_sum_assignment(cost)

    out: list[dict] = []
    for i, j in zip(row_ind, col_ind):
        stone = stones_with_candidates[i]
        number = int(j + 1)
        c = float(cost[i, j])
        ocr_supported = c < PENALTY_NO_OCR - 0.5
        if c >= PENALTY_PARITY_BAD - 1:
            # This shouldn't happen in well-detected boards. Mark as conflict.
            out.append(
                {
                    **stone,
                    "number": number,
                    "confidence": 0.0,
                    "ocr_supported": False,
                    "conflict": True,
                }
            )
            continue
        if ocr_supported:
            confidence = float(100.0 - c * 100.0)
        else:
            confidence = 0.0
        out.append(
            {
                **stone,
                "number": number,
                "confidence": confidence,
                "ocr_supported": ocr_supported,
                "conflict": False,
            }
        )

    out.sort(key=lambda s: s["number"])
    # Re-number sanity: gaps in numbers can mean captures; flag them.
    if out:
        nums = [s["number"] for s in out]
        gaps = sorted(set(range(1, max(nums) + 1)) - set(nums))
        for s in out:
            s["sequence_gaps"] = gaps
    return out
