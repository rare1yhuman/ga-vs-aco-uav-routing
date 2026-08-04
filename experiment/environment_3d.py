"""
3D Environment Module — GA vs ACO Paper

Extends the 2D 200×200 grid to a 200×200×10 voxel space.
Altitude layers: z=0 (ground) → z=9 (100m altitude, 10m per layer).

Key concepts:
  - Buildings  : NFZs that exist from z=0 to z=z_top (can fly OVER them)
  - Air corridors: NFZs that exist ONLY at high altitude (z≥8) — simulates
                   commercial drone traffic lanes the UAV must avoid
  - Wind model : wind speed increases with altitude (exponential profile)
                 → flying high gives shorter paths but more energy drag
"""

import numpy as np
from typing import List, Dict, Tuple, Set
from grid_gen import GRID_ROWS, GRID_COLS, CELL_NFZ, CELL_FREE, CELL_DEPOT, CELL_DELIVERY

# ── Altitude parameters ──────────────────────────────────────────────────────
ALT_LAYERS    = 10         # z layers: 0..9
LAYER_HEIGHT  = 10         # metres per altitude layer
MAX_ALT_M     = ALT_LAYERS * LAYER_HEIGHT   # 100 m

# ── Wind model parameters ────────────────────────────────────────────────────
WIND_BASE_MS  = 0.0        # m/s wind at ground level (z=0)
WIND_SCALE    = 2.5        # m/s additional wind per layer (exponential)
WIND_EXP      = 1.4        # exponent — wind increases non-linearly with altitude

# Cell type (reuse from grid_gen, plus altitude-only NFZ)
CELL_AIR_CORRIDOR = 4      # high-altitude restricted zone


def wind_speed_at_layer(z: int) -> float:
    """
    Return wind speed (m/s) at altitude layer z.
    Uses a power-law profile: v(z) = WIND_SCALE * z^WIND_EXP
    z=0 → 0 m/s,  z=9 → ~27 m/s (strong headwind at max altitude).
    """
    if z <= 0:
        return WIND_BASE_MS
    return WIND_SCALE * (z ** WIND_EXP)


class Environment3D:
    """
    A 3D voxel environment: shape (GRID_ROWS, GRID_COLS, ALT_LAYERS).

    Voxel value == CELL_NFZ means the voxel is blocked.
    All other voxels are navigable.
    """

    def __init__(self, seed: int, nfz_blocks_2d: List[Dict],
                 n_buildings: int = 12, n_air_corridors: int = 3,
                 building_max_z: int = 5):
        """
        Args:
            seed:            Random seed for building/corridor placement.
            nfz_blocks_2d:   Static 2D NFZ blocks (extend vertically as buildings).
            n_buildings:     Extra standalone buildings (tall columns).
            n_air_corridors: High-altitude horizontal restricted bands.
            building_max_z:  Max altitude layer for buildings (drone can fly over).
        """
        self.seed            = seed
        self.building_max_z  = min(building_max_z, ALT_LAYERS - 1)
        self._required_altitude_cache = None
        self.voxels          = np.zeros(
            (GRID_ROWS, GRID_COLS, ALT_LAYERS), dtype=np.uint8
        )
        rng = np.random.default_rng(seed)

        # 1. Extrude 2D static NFZ blocks into vertical buildings
        for b in nfz_blocks_2d:
            r, c = b['row'], b['col']
            h, w = b['height'], b['width']
            # Each 2D NFZ becomes a building from z=0 to building_max_z
            z_top = int(rng.integers(2, self.building_max_z + 1))
            self.voxels[r:r+h, c:c+w, 0:z_top] = CELL_NFZ

        # 2. Extra random tall buildings (thin columns)
        for _ in range(n_buildings):
            r     = int(rng.integers(10, GRID_ROWS - 10))
            c     = int(rng.integers(10, GRID_COLS - 10))
            sz    = int(rng.integers(4, 10))
            z_top = int(rng.integers(3, self.building_max_z + 1))
            self.voxels[r:r+sz, c:c+sz, 0:z_top] = CELL_NFZ

        # 3. Air corridors (horizontal bands at high altitude)
        self._air_corridors: List[Dict] = []
        for _ in range(n_air_corridors):
            z     = int(rng.integers(ALT_LAYERS - 3, ALT_LAYERS))
            axis  = rng.choice(['row', 'col'])
            pos   = int(rng.integers(20, (GRID_ROWS if axis == 'row' else GRID_COLS) - 20))
            width = int(rng.integers(10, 30))
            if axis == 'row':
                self.voxels[pos:pos+width, :, z] = CELL_NFZ
            else:
                self.voxels[:, pos:pos+width, z] = CELL_NFZ
            self._air_corridors.append({'axis': axis, 'pos': pos,
                                         'width': width, 'z': z})

        print(f"[env3d] Built 3D environment: "
              f"{GRID_ROWS}×{GRID_COLS}×{ALT_LAYERS} voxels, "
              f"{n_air_corridors} air corridors, "
              f"seed={seed}")

    # ── Queries ──────────────────────────────────────────────────────────────

    def is_blocked(self, r: int, c: int, z: int) -> bool:
        if not (0 <= r < GRID_ROWS and 0 <= c < GRID_COLS and 0 <= z < ALT_LAYERS):
            return True   # out of bounds = blocked
        return bool(self.voxels[r, c, z] == CELL_NFZ)

    def free_altitude_range(self, r: int, c: int) -> Tuple[int, int]:
        """Return (min_z, max_z) of navigable layers at (r,c)."""
        free = [z for z in range(ALT_LAYERS) if self.voxels[r, c, z] != CELL_NFZ]
        if not free:
            return (-1, -1)
        return (min(free), max(free))

    def min_safe_altitude(self, r: int, c: int) -> int:
        """Minimum altitude layer at (r,c) that is not an NFZ."""
        for z in range(ALT_LAYERS):
            if self.voxels[r, c, z] != CELL_NFZ:
                return z
        return -1   # completely blocked column

    def required_altitude_map(self, proximity_radius: int = 5) -> np.ndarray:
        """
        Return a 2D map of recommended flight altitude layer per cell.

        The value is the maximum nearby building height inside a square
        proximity window, clipped to the first non-blocked layer at the cell.
        This mirrors the edge-penalty model used by the experiment runner.
        """
        if self._required_altitude_cache is not None:
            return self._required_altitude_cache

        height_map = np.zeros((GRID_ROWS, GRID_COLS), dtype=np.int32)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                for z in range(ALT_LAYERS - 1, -1, -1):
                    if self.voxels[r, c, z] == CELL_NFZ:
                        height_map[r, c] = min(z + 1, ALT_LAYERS - 1)
                        break

        from scipy.ndimage import maximum_filter
        required = maximum_filter(height_map, size=2 * proximity_radius + 1)
        for r in range(GRID_ROWS):
            for c in range(GRID_COLS):
                desired = int(required[r, c])
                chosen = -1
                for z in range(desired, ALT_LAYERS):
                    if not self.is_blocked(r, c, z):
                        chosen = z
                        break
                if chosen < 0:
                    for z in range(desired - 1, -1, -1):
                        if not self.is_blocked(r, c, z):
                            chosen = z
                            break
                required[r, c] = chosen

        self._required_altitude_cache = required
        return required

    def neighbours_3d(self, r: int, c: int, z: int) -> List[Tuple[int,int,int]]:
        """
        26-connectivity neighbour list for 3D grid navigation.
        Returns only passable (non-NFZ) neighbours within bounds.
        """
        neighbours = []
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                for dz in (-1, 0, 1):
                    if dr == 0 and dc == 0 and dz == 0:
                        continue
                    nr, nc, nz = r+dr, c+dc, z+dz
                    if not self.is_blocked(nr, nc, nz):
                        neighbours.append((nr, nc, nz))
        return neighbours

    # ── Statistics ───────────────────────────────────────────────────────────

    def stats(self) -> Dict:
        total    = GRID_ROWS * GRID_COLS * ALT_LAYERS
        blocked  = int(np.sum(self.voxels == CELL_NFZ))
        return {
            'total_voxels':    total,
            'blocked_voxels':  blocked,
            'free_voxels':     total - blocked,
            'blocked_%':       round(100 * blocked / total, 2),
            'air_corridors':   len(self._air_corridors),
        }


# ── 3D energy helper (used by energy_model.py) ──────────────────────────────

def segment_energy_3d(r1: int, c1: int, z1: int,
                       r2: int, c2: int, z2: int,
                       payload_kg: float = 2.0) -> float:
    """
    Estimate energy (Joules) to fly from voxel (r1,c1,z1) to (r2,c2,z2).

    Model:
        E = E_horizontal + E_vertical + E_wind_drag
    """
    # Physical constants (simplified quadrotor model)
    E_CRUISE_PER_M   = 0.15   # J/m horizontal cruise
    E_CLIMB_PER_M    = 0.45   # J/m ascent (fighting gravity)
    E_DESCEND_PER_M  = 0.05   # J/m descent (gravity assists)
    E_WIND_PER_MS_M  = 0.02   # extra J per (m/s × m) of horizontal travel in wind

    d_horiz = float(np.hypot((r2 - r1) * LAYER_HEIGHT,
                              (c2 - c1) * LAYER_HEIGHT))  # rough horiz dist in m
    dz      = (z2 - z1) * LAYER_HEIGHT                   # vertical displacement in m

    e_horizontal = E_CRUISE_PER_M * d_horiz
    e_vertical   = (E_CLIMB_PER_M * max(0, dz) +
                    E_DESCEND_PER_M * max(0, -dz))

    # Wind drag at average altitude
    z_avg   = (z1 + z2) / 2
    wind    = wind_speed_at_layer(int(round(z_avg)))
    e_wind  = E_WIND_PER_MS_M * wind * d_horiz

    # Payload penalty (heavier = more energy)
    payload_factor = 1.0 + 0.1 * payload_kg

    return (e_horizontal + e_vertical + e_wind) * payload_factor


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    from grid_gen import generate_grid
    _, _, nfz_blocks = generate_grid(seed=42)
    env = Environment3D(seed=42, nfz_blocks_2d=nfz_blocks)
    print("3D environment stats:", env.stats())
    print("Wind at each layer:")
    for z in range(ALT_LAYERS):
        print(f"  z={z} ({z*LAYER_HEIGHT}m): {wind_speed_at_layer(z):.1f} m/s")
    e = segment_energy_3d(10, 10, 0, 50, 80, 5)
    print(f"\nSample flight energy (ground→altitude 5): {e:.2f} J")
