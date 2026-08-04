"""
Grid Generation Module — GA vs ACO Paper
Generates 200x200 grid with static and dynamic No-Fly Zones (NFZs).

Upgraded from ACO paper (100x100, 12 NFZs, 15 deliveries)
to new scale (200x200, 18 NFZs, 20-50 deliveries).
"""

import numpy as np
from collections import deque
from typing import List, Dict, Tuple, Optional

# ── Environment parameters ──────────────────────────────────────────────────
GRID_ROWS      = 200
GRID_COLS      = 200
CELL_SIZE_M    = 100        # metres per cell -> 200x200 cells = 20 km x 20 km

# ── NFZ parameters ──────────────────────────────────────────────────────────
NFZ_COUNT      = 18
NFZ_MIN_W      = 6          # cells
NFZ_MAX_W      = 16
NFZ_MIN_H      = 6
NFZ_MAX_H      = 16
BORDER_EXCLUSION = 8        # cells — NFZs cannot touch border

# ── Delivery parameters ─────────────────────────────────────────────────────
DEPOT_POS      = (10, 10)   # Fixed depot position
NUM_DELIVERIES = 20         # default; overridden per trial
MIN_POINT_SEP  = 8          # minimum separation between delivery points (cells)

# ── Cell type constants ─────────────────────────────────────────────────────
CELL_FREE     = 0
CELL_NFZ      = 1
CELL_DELIVERY = 2
CELL_DEPOT    = 3


def generate_grid(
    seed: int,
    has_nfz: bool = True,
    num_deliveries: int = NUM_DELIVERIES,
    nfz_count: int = NFZ_COUNT,
) -> Tuple[np.ndarray, List[Tuple[int, int]], List[Dict]]:
    """
    Generate a 200×200 grid with static NFZs and delivery points.

    Args:
        seed:           Random seed for reproducibility.
        has_nfz:        Whether to place NFZ blocks.
        num_deliveries: Number of delivery points to place.
        nfz_count:      Number of static NFZ blocks to place.

    Returns:
        grid:            200×200 numpy array (uint8).
        delivery_points: List of (row, col) delivery locations.
        nfz_blocks:      List of dicts with NFZ metadata.
    """
    np.random.seed(seed)
    grid = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.uint8)
    grid[DEPOT_POS] = CELL_DEPOT

    nfz_blocks: List[Dict] = []
    if has_nfz:
        nfz_blocks = _place_nfz_blocks(grid, nfz_count)

    delivery_points = _place_delivery_points(grid, num_deliveries)
    for p in delivery_points:
        grid[p] = CELL_DELIVERY

    return grid, delivery_points, nfz_blocks


# ── Internal helpers ─────────────────────────────────────────────────────────

def _place_nfz_blocks(grid: np.ndarray, nfz_count: int,
                      max_retries: int = 2000) -> List[Dict]:
    """Place rectangular NFZ blocks using rejection sampling."""
    nfz_blocks: List[Dict] = []
    total_attempts = 0

    for block_id in range(nfz_count):
        placed        = False
        block_attempts = 0

        while not placed and block_attempts < max_retries:
            block_attempts += 1
            total_attempts += 1

            w = np.random.randint(NFZ_MIN_W, NFZ_MAX_W + 1)
            h = np.random.randint(NFZ_MIN_H, NFZ_MAX_H + 1)

            max_row = GRID_ROWS - h - BORDER_EXCLUSION
            max_col = GRID_COLS - w - BORDER_EXCLUSION

            if max_row <= BORDER_EXCLUSION or max_col <= BORDER_EXCLUSION:
                continue

            r = np.random.randint(BORDER_EXCLUSION, max_row)
            c = np.random.randint(BORDER_EXCLUSION, max_col)

            if _has_overlap(r, c, w, h, nfz_blocks):
                continue

            # Depot must not be inside the block
            dr, dc = DEPOT_POS
            if r <= dr < r + h and c <= dc < c + w:
                continue

            # Connectivity check (ensure free cells remain reachable)
            temp = grid.copy()
            temp[r:r+h, c:c+w] = CELL_NFZ
            if not _is_connected(temp, DEPOT_POS):
                continue

            # Commit
            grid[r:r+h, c:c+w] = CELL_NFZ
            nfz_blocks.append(
                {'id': block_id, 'row': r, 'col': c, 'width': w, 'height': h}
            )
            placed = True

        if not placed:
            raise RuntimeError(
                f"Failed to place NFZ block {block_id} after {max_retries} tries"
            )

    print(f"[grid_gen] Placed {len(nfz_blocks)} NFZ blocks ({total_attempts} attempts)")
    return nfz_blocks


def _has_overlap(row: int, col: int, w: int, h: int,
                 existing: List[Dict]) -> bool:
    for b in existing:
        if not (row + h <= b['row'] or row >= b['row'] + b['height'] or
                col + w <= b['col'] or col >= b['col'] + b['width']):
            return True
    return False


def _is_connected(grid: np.ndarray, start: Tuple[int, int]) -> bool:
    """BFS from start; returns True if all non-NFZ cells are reachable."""
    rows, cols = grid.shape
    visited    = np.zeros((rows, cols), dtype=bool)
    queue      = deque([start])
    visited[start] = True
    reachable  = 1
    free_total = int(np.sum(grid != CELL_NFZ))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1,0),(1,0),(0,-1),(0,1)):
            nr, nc = r+dr, c+dc
            if 0 <= nr < rows and 0 <= nc < cols and not visited[nr,nc] and grid[nr,nc] != CELL_NFZ:
                visited[nr,nc] = True
                queue.append((nr,nc))
                reachable += 1

    return reachable >= free_total


def _place_delivery_points(grid: np.ndarray,
                           num: int,
                           max_attempts: int = 50_000) -> List[Tuple[int,int]]:
    """Place delivery points with minimum separation constraint."""
    points: List[Tuple[int,int]] = []
    attempts = 0

    while len(points) < num and attempts < max_attempts:
        attempts += 1
        r = np.random.randint(0, GRID_ROWS)
        c = np.random.randint(0, GRID_COLS)

        if grid[r, c] != CELL_FREE:
            continue
        if _euclidean(DEPOT_POS, (r, c)) < MIN_POINT_SEP:
            continue
        if any(_euclidean((r,c), p) < MIN_POINT_SEP for p in points):
            continue

        points.append((r, c))

    if len(points) < num:
        raise RuntimeError(
            f"[grid_gen] Only placed {len(points)}/{num} delivery points"
        )

    print(f"[grid_gen] Placed {len(points)} delivery points ({attempts} attempts)")
    return points


def _euclidean(a: Tuple[int,int], b: Tuple[int,int]) -> float:
    return float(np.hypot(a[0]-b[0], a[1]-b[1]))


# ── Public utility ───────────────────────────────────────────────────────────

def get_grid_info(grid: np.ndarray,
                  delivery_points: List[Tuple[int,int]],
                  nfz_blocks: List[Dict]) -> Dict:
    """Return a summary dict of grid statistics."""
    area_km2 = (GRID_ROWS * CELL_SIZE_M / 1000) * (GRID_COLS * CELL_SIZE_M / 1000)
    return {
        'grid_size':      f"{GRID_ROWS}x{GRID_COLS}",
        'cell_size_m':    CELL_SIZE_M,
        'total_area_km2': area_km2,
        'depot':          DEPOT_POS,
        'num_deliveries': len(delivery_points),
        'num_nfz':        len(nfz_blocks),
        'free_cells':     int(np.sum(grid == CELL_FREE)),
        'nfz_cells':      int(np.sum(grid == CELL_NFZ)),
        'nfz_coverage_%': round(100 * np.sum(grid == CELL_NFZ) / (GRID_ROWS*GRID_COLS), 2),
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== grid_gen smoke test ===")
    grid, dps, nfzs = generate_grid(seed=42)
    info = get_grid_info(grid, dps, nfzs)
    for k, v in info.items():
        print(f"  {k}: {v}")
