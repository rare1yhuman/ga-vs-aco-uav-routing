"""
Dynamic No-Fly Zone (NFZ) Module — GA vs ACO Paper

Supports three motion models for moving NFZs:
  1. StaticNFZ       — does not move (used for static trials A-C)
  2. LinearDriftNFZ  — moves at constant velocity with boundary bounce
  3. OscillatingNFZ  — pendulum-like oscillation on one axis

All classes share the interface:
    nfz.get_position(t) -> (row, col)   top-left corner at time-step t
    nfz.get_rect(t)     -> (row, col, height, width)
    nfz.cells_at(t)     -> set of (row, col) tuples occupied at time t
"""

import numpy as np
from typing import Tuple, Set, List
from grid_gen import GRID_ROWS, GRID_COLS, CELL_NFZ


# ── Base class ───────────────────────────────────────────────────────────────

class DynamicNFZ:
    """Abstract base for a rectangular NFZ that can move over time."""

    def __init__(self, nfz_id: int, row: int, col: int,
                 height: int, width: int):
        self.id     = nfz_id
        self.row0   = row       # initial top-left row
        self.col0   = col       # initial top-left col
        self.height = height
        self.width  = width

    def get_position(self, t: int) -> Tuple[int, int]:
        """Return (row, col) of top-left corner at time step t."""
        return (self.row0, self.col0)

    def get_rect(self, t: int) -> Tuple[int, int, int, int]:
        """Return (row, col, height, width) at time step t."""
        r, c = self.get_position(t)
        return (r, c, self.height, self.width)

    def cells_at(self, t: int) -> Set[Tuple[int, int]]:
        """Return the set of grid cells occupied at time step t."""
        r, c = self.get_position(t)
        cells: Set[Tuple[int, int]] = set()
        for dr in range(self.height):
            for dc in range(self.width):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    cells.add((nr, nc))
        return cells

    def __repr__(self) -> str:
        return (f"{self.__class__.__name__}(id={self.id}, "
                f"start=({self.row0},{self.col0}), "
                f"size={self.height}×{self.width})")


# ── Static NFZ ───────────────────────────────────────────────────────────────

class StaticNFZ(DynamicNFZ):
    """A non-moving NFZ (position is constant for all t)."""
    pass


# ── Linear Drift NFZ ─────────────────────────────────────────────────────────

class LinearDriftNFZ(DynamicNFZ):
    """
    NFZ moves at constant velocity (v_row, v_col) cells/timestep.
    Bounces off grid boundaries using triangle-wave reflection.
    """

    def __init__(self, nfz_id: int, row: int, col: int,
                 height: int, width: int,
                 v_row: float, v_col: float):
        super().__init__(nfz_id, row, col, height, width)
        self.v_row = v_row
        self.v_col = v_col

    def get_position(self, t: int) -> Tuple[int, int]:
        max_r = GRID_ROWS - self.height
        max_c = GRID_COLS - self.width

        r = self.row0 + self.v_row * t
        c = self.col0 + self.v_col * t

        # Bounce (triangle wave) for row
        if max_r > 0:
            period_r = max_r * 2
            r = r % period_r
            if r < 0:
                r += period_r
            if r > max_r:
                r = period_r - r
        else:
            r = 0

        # Bounce for col
        if max_c > 0:
            period_c = max_c * 2
            c = c % period_c
            if c < 0:
                c += period_c
            if c > max_c:
                c = period_c - c
        else:
            c = 0

        return (int(round(r)), int(round(c)))


# ── Oscillating NFZ ──────────────────────────────────────────────────────────

class OscillatingNFZ(DynamicNFZ):
    """
    NFZ oscillates like a pendulum on one axis (sine wave).

    Args:
        amplitude: max displacement from start position (cells)
        period:    full oscillation period in time-steps
        axis:      'row' or 'col'
    """

    def __init__(self, nfz_id: int, row: int, col: int,
                 height: int, width: int,
                 amplitude: int, period: int, axis: str = 'col'):
        super().__init__(nfz_id, row, col, height, width)
        self.amplitude = amplitude
        self.period    = period
        self.axis      = axis

    def get_position(self, t: int) -> Tuple[int, int]:
        offset = int(round(self.amplitude * np.sin(2 * np.pi * t / self.period)))
        r, c   = self.row0, self.col0

        if self.axis == 'row':
            r += offset
        else:
            c += offset

        # Clamp to valid bounds
        r = max(0, min(r, GRID_ROWS - self.height))
        c = max(0, min(c, GRID_COLS - self.width))
        return (r, c)


# ── Expanding NFZ ────────────────────────────────────────────────────────────

class ExpandingNFZ(DynamicNFZ):
    """
    NFZ that pulses in size — grows and shrinks like a weather cell.
    Centre position is fixed; size oscillates between min_size and max_size.
    """

    def __init__(self, nfz_id: int, centre_row: int, centre_col: int,
                 min_size: int, max_size: int, period: int):
        # Store centre; actual row/col depends on current size
        super().__init__(nfz_id, centre_row, centre_col, min_size, min_size)
        self.centre_row = centre_row
        self.centre_col = centre_col
        self.min_size   = min_size
        self.max_size   = max_size
        self.period     = period

    def _current_size(self, t: int) -> int:
        phase = (1 + np.sin(2 * np.pi * t / self.period)) / 2  # 0→1
        return int(round(self.min_size + phase * (self.max_size - self.min_size)))

    def get_position(self, t: int) -> Tuple[int, int]:
        s = self._current_size(t)
        r = max(0, min(self.centre_row - s // 2, GRID_ROWS - s))
        c = max(0, min(self.centre_col - s // 2, GRID_COLS - s))
        return (r, c)

    def cells_at(self, t: int) -> Set[Tuple[int, int]]:
        s        = self._current_size(t)
        r, c     = self.get_position(t)
        cells: Set[Tuple[int, int]] = set()
        for dr in range(s):
            for dc in range(s):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_ROWS and 0 <= nc < GRID_COLS:
                    cells.add((nr, nc))
        return cells


# ── Grid rendering ───────────────────────────────────────────────────────────

def render_dynamic_nfzs(base_grid: np.ndarray,
                         dynamic_nfzs: List[DynamicNFZ],
                         t: int) -> np.ndarray:
    """
    Overlay dynamic NFZs onto base_grid at time step t.

    base_grid should contain ONLY static content (depot, deliveries,
    static NFZs). Do NOT modify base_grid — returns a new array.
    """
    grid = base_grid.copy()
    for nfz in dynamic_nfzs:
        for (r, c) in nfz.cells_at(t):
            grid[r, c] = CELL_NFZ
    return grid


def build_default_dynamic_nfzs(seed: int,
                                n_linear: int = 4,
                                n_oscillating: int = 4) -> List[DynamicNFZ]:
    """
    Generate a reproducible set of dynamic NFZs for the paper's
    standard dynamic trials (D, F, H).

    Args:
        seed:         Random seed.
        n_linear:     Number of linearly-drifting NFZs.
        n_oscillating: Number of oscillating NFZs.

    Returns:
        List of DynamicNFZ objects.
    """
    rng    = np.random.default_rng(seed)
    nfzs: List[DynamicNFZ] = []
    nid    = 0

    border = 10   # keep dynamic NFZs away from borders
    h_min, h_max = 6, 12
    w_min, w_max = 6, 12

    # Linear drift NFZs
    for _ in range(n_linear):
        h  = int(rng.integers(h_min, h_max + 1))
        w  = int(rng.integers(w_min, w_max + 1))
        r  = int(rng.integers(border, GRID_ROWS - h - border))
        c  = int(rng.integers(border, GRID_COLS - w - border))
        vr = float(rng.choice([-0.5, -0.3, 0.3, 0.5]))
        vc = float(rng.choice([-0.5, -0.3, 0.3, 0.5]))
        nfzs.append(LinearDriftNFZ(nid, r, c, h, w, vr, vc))
        nid += 1

    # Oscillating NFZs
    for _ in range(n_oscillating):
        h   = int(rng.integers(h_min, h_max + 1))
        w   = int(rng.integers(w_min, w_max + 1))
        r   = int(rng.integers(border, GRID_ROWS - h - border))
        c   = int(rng.integers(border, GRID_COLS - w - border))
        amp = int(rng.integers(8, 20))
        per = int(rng.integers(20, 60))
        ax  = rng.choice(['row', 'col'])
        nfzs.append(OscillatingNFZ(nid, r, c, h, w, amp, per, str(ax)))
        nid += 1

    return nfzs


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    nfzs = build_default_dynamic_nfzs(seed=42, n_linear=4, n_oscillating=4)
    print(f"Created {len(nfzs)} dynamic NFZs:")
    for n in nfzs:
        p0 = n.get_position(0)
        p5 = n.get_position(50)
        print(f"  {n}  t=0→{p0}  t=50→{p5}")
