"""Smart validation & assignment of move numbers to detected stones.

Constraints we lean on (in order of strength):

1. Each stone gets exactly one move number; numbers are unique.
2. Black plays first → ODD numbers ⇒ Black, EVEN numbers ⇒ White.
3. Each move must be played on an EMPTY intersection at its time of play
   (game rule). Captures clear opponent groups with zero liberties.
4. A move must not be suicide: after captures, the played stone's group
   must have at least one liberty.
5. Soft locality prior: consecutive moves are usually within a few grid
   spaces of each other (Manhattan distance). Used only as a tiebreaker.

Algorithm:

1. Build the universe of numbers (1..N where N = max(visible_stones,
   max_OCR_candidate, parity_minimum)).
2. Build a cost matrix: rows = stones, cols = numbers. For OCR-supplied
   candidates, cost = (100 - confidence) / 100. For parity-incompatible
   numbers, cost = PENALTY_PARITY_BAD. For numbers a stone never OCR'd,
   cost = PENALTY_NO_OCR.
3. Run Hungarian (linear_sum_assignment) → initial assignment.
4. Replay the assignment in number order. Score: count moves that land on
   an already-occupied intersection or are suicidal (these are
   game-rule violations).
5. Repair pass: for the set of stones that are currently no-OCR-supported
   (or that triggered violations), try permutations of their numbers and
   pick the one that minimises violations + (locality penalty * small).
6. Final pass: hill-climb pairwise swaps as long as they reduce
   violations. Stops at local minimum.
"""
from __future__ import annotations
import itertools
import numpy as np
from scipy.optimize import linear_sum_assignment

from go_simulator import BoardState, BLACK, WHITE


PENALTY_NO_OCR = 1000.0    # cost if stone never produced this number from OCR
PENALTY_PARITY_BAD = 1e6   # essentially forbidden

# Small locality bonus so the validator can break ties using "next move
# is usually near previous move" heuristic. Tiny — doesn't override OCR.
LOCALITY_WEIGHT = 0.001
# Repair pass tries up to this many no-OCR stones via brute permutation.
# 6! = 720 which is fine; 8! = 40320 starts to drag.
MAX_PERMUTE_K = 7


def _parity_ok(number: int, color: str) -> bool:
    if color == "B":
        return number % 2 == 1
    if color == "W":
        return number % 2 == 0
    return False


# ----------------------------------------------------------------------
# Cost matrix construction.
# ----------------------------------------------------------------------

def _build_cost_matrix(
    stones: list[dict], universe_size: int
) -> np.ndarray:
    n = len(stones)
    cost = np.full((n, universe_size), PENALTY_NO_OCR, dtype=np.float64)
    for i, stone in enumerate(stones):
        color = stone["color"]
        for num in range(1, universe_size + 1):
            if not _parity_ok(num, color):
                cost[i, num - 1] = PENALTY_PARITY_BAD
        for num, conf in stone.get("candidates", []):
            if 1 <= num <= universe_size and _parity_ok(num, color):
                # Lower cost for higher confidence; floor at 0 (perfect read).
                cost[i, num - 1] = (100.0 - max(conf, 0.0)) / 100.0
    return cost


# ----------------------------------------------------------------------
# Replay scoring: count game-rule violations + locality score.
# ----------------------------------------------------------------------

def _replay_score(
    assignment: list[tuple[dict, int]],
    board_size: int,
) -> tuple[int, float]:
    """Score the replay of a stone→number assignment.

    Returns (violations, locality_penalty).

    `violations` = number of moves that, at their time of play, would land on
    an already-occupied point or be suicide. Captures DO clear opponent
    groups, so a move that captures into an occupied-then-cleared point is
    fine. We force-apply moves so subsequent captures still happen.
    """
    in_order = sorted(assignment, key=lambda sa: sa[1])
    board = BoardState(board_size)
    violations = 0
    locality_pen = 0.0
    prev_rc: tuple[int, int] | None = None
    for stone, n in in_order:
        r, c = stone["row"], stone["col"]
        outcome = board.try_move(r, c, stone["color"])
        if not outcome.legal:
            violations += 1
        # Apply the move regardless so subsequent moves see realistic captures.
        board.play(r, c, stone["color"], force=True)
        if prev_rc is not None:
            dr = abs(r - prev_rc[0])
            dc = abs(c - prev_rc[1])
            locality_pen += dr + dc
        prev_rc = (r, c)
    return violations, locality_pen


# ----------------------------------------------------------------------
# Repair pass: permute the numbers among no-OCR stones to minimise
# replay violations.
# ----------------------------------------------------------------------

def _repair_no_ocr(
    stones_with_assignment: list[dict],
    cost: np.ndarray,
    board_size: int,
) -> list[dict]:
    """Re-permute numbers among no-OCR-supported stones to fix violations.

    Picks the no-OCR-supported stones (cost >= PENALTY_NO_OCR - 0.5 in their
    assigned cell) and tries every permutation of their numbers, scoring
    each by (replay_violations, locality_penalty). Keeps the best.
    """
    no_ocr_idx = [
        i for i, s in enumerate(stones_with_assignment)
        if not s.get("ocr_supported", True)
    ]
    if len(no_ocr_idx) < 2 or len(no_ocr_idx) > MAX_PERMUTE_K:
        return stones_with_assignment

    # Numbers currently held by no-OCR stones.
    pool_numbers = [stones_with_assignment[i]["number"] for i in no_ocr_idx]
    # Keep colors so permutation only swaps within parity class.
    by_color: dict[str, list[int]] = {"B": [], "W": []}
    for idx in no_ocr_idx:
        c = stones_with_assignment[idx]["color"]
        by_color[c].append(idx)
    # We permute Black-stone numbers among Black slots, White among White.
    # This preserves parity automatically.
    black_nums = [stones_with_assignment[i]["number"] for i in by_color["B"]]
    white_nums = [stones_with_assignment[i]["number"] for i in by_color["W"]]

    base = stones_with_assignment
    best_score = _replay_score([(s, s["number"]) for s in base], board_size)

    # Don't early-return on 0 violations: even when the assignment is already
    # legal, locality can still pick the right one when several legal options
    # exist (common when 2+ stones are no-OCR and no captures disambiguate).

    # Brute permute numbers within each color class.
    for bperm in itertools.permutations(black_nums):
        for wperm in itertools.permutations(white_nums):
            trial = [s.copy() for s in base]
            for j, idx in enumerate(by_color["B"]):
                trial[idx]["number"] = bperm[j]
            for j, idx in enumerate(by_color["W"]):
                trial[idx]["number"] = wperm[j]
            score = _replay_score([(s, s["number"]) for s in trial], board_size)
            if score < best_score:
                best_score = score
                base = trial
                if score[0] == 0:
                    return base  # found a fully-legal assignment; stop early
    return base


# ----------------------------------------------------------------------
# Hill-climb: pairwise swaps that reduce replay violations.
# ----------------------------------------------------------------------

def _hill_climb_swaps(
    stones: list[dict],
    board_size: int,
    max_iters: int = 3,
) -> list[dict]:
    """Try swapping numbers between pairs of same-color stones if it reduces
    the violation count. Stops at local min or after max_iters passes.

    We only swap stones that are already same-color (parity preserved). To
    keep this from being O(n^2) on every call, we focus on stones that are
    PART OF a violation in the current replay.
    """
    base = stones
    best_score = _replay_score([(s, s["number"]) for s in base], board_size)
    if best_score[0] == 0:
        return base
    for _ in range(max_iters):
        improved = False
        # Identify violators by replaying once and tagging.
        in_order = sorted(base, key=lambda s: s["number"])
        board = BoardState(board_size)
        violator_idxs: list[int] = []
        ordered_idx = {id(s): i for i, s in enumerate(base)}
        for s in in_order:
            outcome = board.try_move(s["row"], s["col"], s["color"])
            if not outcome.legal:
                violator_idxs.append(ordered_idx[id(s)])
            board.play(s["row"], s["col"], s["color"], force=True)
        # Try swapping each violator with every same-color stone.
        for vi in violator_idxs:
            v = base[vi]
            for j, t in enumerate(base):
                if j == vi or t["color"] != v["color"]:
                    continue
                trial = [s.copy() for s in base]
                trial[vi]["number"], trial[j]["number"] = trial[j]["number"], trial[vi]["number"]
                score = _replay_score([(s, s["number"]) for s in trial], board_size)
                if score < best_score:
                    base = trial
                    best_score = score
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return base


# ----------------------------------------------------------------------
# Main entry.
# ----------------------------------------------------------------------

def assign_moves(
    stones_with_candidates: list[dict],
    board_size: int = 19,
) -> list[dict]:
    """Assign each stone a move number.

    Inputs: list of dicts with keys row, col, cx, cy, color, candidates.
    Returns: list of dicts with row, col, cx, cy, color, number, confidence,
             ocr_supported (bool), candidates.
    Sorted by assigned move number ascending.
    """
    n = len(stones_with_candidates)
    if n == 0:
        return []

    max_seen = max(
        (c[0] for s in stones_with_candidates for c in s.get("candidates", [])),
        default=0,
    )
    n_black = sum(1 for s in stones_with_candidates if s.get("color") == "B")
    n_white = sum(1 for s in stones_with_candidates if s.get("color") == "W")
    parity_min = 2 * max(n_black, n_white)
    universe_size = max(n, max_seen, parity_min)

    cost = _build_cost_matrix(stones_with_candidates, universe_size)
    if universe_size < n:
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
            out.append({
                **stone,
                "number": number,
                "confidence": 0.0,
                "ocr_supported": False,
                "conflict": True,
            })
            continue
        if ocr_supported:
            confidence = float(100.0 - c * 100.0)
        else:
            confidence = 0.0
        out.append({
            **stone,
            "number": number,
            "confidence": confidence,
            "ocr_supported": ocr_supported,
            "conflict": False,
        })

    # Repair: permute no-OCR stones to satisfy game rules.
    out = _repair_no_ocr(out, cost, board_size)
    # Hill climb: swap pairs to fix remaining violations.
    out = _hill_climb_swaps(out, board_size)

    out.sort(key=lambda s: s["number"])
    if out:
        nums = [s["number"] for s in out]
        gaps = sorted(set(range(1, max(nums) + 1)) - set(nums))
        for s in out:
            s["sequence_gaps"] = gaps
    return out
