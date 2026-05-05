"""Minimal Go board simulator with capture detection.

We need this to validate sequences of moves: replay them in order and check
that each move is legal (lands on an empty point, has at least one liberty
after captures, and ideally doesn't violate ko). The simulator is used by
the validator to disambiguate OCR-uncertain stones via game-rule consistency.

Coordinates are (row, col) with row 0 at the top of the image and col 0 at
the left. This matches the rest of the backend.

The simulator deliberately does NOT enforce strict legality (it never raises
on illegal moves). Instead, every call returns a structured outcome the
caller can score: was the point empty?, did it capture anyone?, would it be
suicide?, would it repeat the previous board (ko)?

This makes it usable both as a hard filter and as a soft cost contributor.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

EMPTY = "."
BLACK = "B"
WHITE = "W"


def _opp(color: str) -> str:
    return WHITE if color == BLACK else BLACK


@dataclass
class MoveOutcome:
    """Result of attempting a move."""
    legal: bool                          # True if move is legal under standard rules
    on_empty: bool                       # was the target intersection empty before?
    captures: list[tuple[int, int]] = field(default_factory=list)  # opp stones removed
    suicide: bool = False                # would the played stone have 0 liberties (and not capture)
    ko: bool = False                     # would replay the previous board state
    reason: str = ""                     # human-readable note when not legal


class BoardState:
    """Mutable Go board state with capture / liberty calculation.

    Designed for replay use. Cheap to copy via .clone() so the validator can
    speculate about alternative move orders without mutating the master state.
    """

    __slots__ = ("size", "grid", "_prev_hash", "_history_hashes")

    def __init__(self, size: int = 19):
        self.size = size
        # grid[r][c] is one of EMPTY/BLACK/WHITE.
        self.grid: list[list[str]] = [[EMPTY] * size for _ in range(size)]
        self._prev_hash: Optional[int] = None
        # Track hash to detect ko (single-step) and superko if needed.
        self._history_hashes: set[int] = set()

    def clone(self) -> "BoardState":
        b = BoardState.__new__(BoardState)
        b.size = self.size
        b.grid = [row[:] for row in self.grid]
        b._prev_hash = self._prev_hash
        b._history_hashes = set(self._history_hashes)
        return b

    def at(self, r: int, c: int) -> str:
        return self.grid[r][c]

    # ------------------------------------------------------------------
    # Group / liberty helpers.
    # ------------------------------------------------------------------

    def _neighbors(self, r: int, c: int):
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                yield nr, nc

    def _flood_group(self, r: int, c: int) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
        """Return (group_stones, liberties) for the group at (r, c).

        Group = maximal connected same-color set. Liberties = empty
        neighbours of the group.
        """
        color = self.grid[r][c]
        if color == EMPTY:
            return set(), set()
        seen: set[tuple[int, int]] = {(r, c)}
        stack = [(r, c)]
        libs: set[tuple[int, int]] = set()
        while stack:
            rr, cc = stack.pop()
            for nr, nc in self._neighbors(rr, cc):
                v = self.grid[nr][nc]
                if v == EMPTY:
                    libs.add((nr, nc))
                elif v == color and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return seen, libs

    def _board_hash(self) -> int:
        # Cheap zobrist-ish hash (string-based, fine for 19x19 verification).
        return hash(tuple(tuple(row) for row in self.grid))

    # ------------------------------------------------------------------
    # Move attempt.
    # ------------------------------------------------------------------

    def try_move(self, r: int, c: int, color: str) -> MoveOutcome:
        """Compute the outcome of playing color at (r, c) WITHOUT mutating."""
        if not (0 <= r < self.size and 0 <= c < self.size):
            return MoveOutcome(legal=False, on_empty=False, reason="out_of_bounds")
        if self.grid[r][c] != EMPTY:
            return MoveOutcome(
                legal=False,
                on_empty=False,
                reason=f"point_occupied_by_{self.grid[r][c]}",
            )
        # Speculate: place the stone, find captures, check liberties on the new group.
        spec = self.clone()
        spec.grid[r][c] = color
        opp = _opp(color)
        captures: list[tuple[int, int]] = []
        for nr, nc in spec._neighbors(r, c):
            if spec.grid[nr][nc] == opp:
                stones, libs = spec._flood_group(nr, nc)
                if not libs:
                    for sr, sc in stones:
                        spec.grid[sr][sc] = EMPTY
                    captures.extend(stones)
        # Now liberties of the played stone:
        own_stones, own_libs = spec._flood_group(r, c)
        suicide = (len(own_libs) == 0)
        ko = False
        if not suicide:
            new_hash = spec._board_hash()
            if new_hash == self._prev_hash:
                ko = True
        legal = (not suicide) and (not ko)
        reason = ""
        if suicide:
            reason = "suicide"
        elif ko:
            reason = "ko"
        return MoveOutcome(
            legal=legal,
            on_empty=True,
            captures=sorted(set(captures)),
            suicide=suicide,
            ko=ko,
            reason=reason,
        )

    def play(self, r: int, c: int, color: str, force: bool = False) -> MoveOutcome:
        """Apply a move, mutating the state.

        If `force` is True, the move is applied regardless of legality (still
        with capture resolution). If False and the move is illegal, the state
        is not mutated and the outcome is returned with .legal == False.
        """
        outcome = self.try_move(r, c, color)
        if not outcome.on_empty and not force:
            return outcome
        if not outcome.legal and not force:
            return outcome
        # Apply mutation. Re-derive captures from scratch to avoid trusting
        # the speculative copy when force=True.
        if outcome.on_empty:
            self.grid[r][c] = color
        else:
            # force overwrite an occupied point
            self.grid[r][c] = color
        opp = _opp(color)
        captured: list[tuple[int, int]] = []
        for nr, nc in self._neighbors(r, c):
            if self.grid[nr][nc] == opp:
                stones, libs = self._flood_group(nr, nc)
                if not libs:
                    for sr, sc in stones:
                        self.grid[sr][sc] = EMPTY
                    captured.extend(stones)
        # Self-suicide capture (only happens under permissive rules; we still
        # apply if force=True or if explicitly suicide-permitting rules).
        own_stones, own_libs = self._flood_group(r, c)
        if not own_libs and force:
            for sr, sc in own_stones:
                self.grid[sr][sc] = EMPTY
            captured.extend(own_stones)
        # Update ko bookkeeping.
        h = self._board_hash()
        self._prev_hash = h
        self._history_hashes.add(h)
        # Re-derive outcome reflecting actual mutations.
        return MoveOutcome(
            legal=outcome.legal,
            on_empty=outcome.on_empty,
            captures=sorted(set(captured)),
            suicide=outcome.suicide,
            ko=outcome.ko,
            reason=outcome.reason,
        )

    # ------------------------------------------------------------------
    # Convenience.
    # ------------------------------------------------------------------

    def empty_at(self, r: int, c: int) -> bool:
        return 0 <= r < self.size and 0 <= c < self.size and self.grid[r][c] == EMPTY


def replay(
    moves: list[tuple[int, int, str]],
    size: int = 19,
    force_illegal: bool = True,
) -> tuple[BoardState, list[MoveOutcome]]:
    """Replay a list of (row, col, color) moves in order.

    With `force_illegal=True`, the simulator applies every move even if it's
    illegal (returning the outcome list so callers can count violations).
    With False, illegal moves leave the state untouched.
    """
    b = BoardState(size)
    outcomes: list[MoveOutcome] = []
    for r, c, color in moves:
        outcomes.append(b.play(r, c, color, force=force_illegal))
    return b, outcomes
