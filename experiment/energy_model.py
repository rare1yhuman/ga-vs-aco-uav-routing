"""
Energy Model — GA vs ACO Paper

Computes realistic energy consumption for drone delivery missions.
Supports both 2D and 3D flight modes.

Energy components:
  E_cruise   — horizontal flight (proportional to distance)
  E_climb    — ascending altitude (costly — fighting gravity)
  E_descend  — descending altitude (cheap — gravity assists)
  E_hover    — hovering at delivery point (per stop, constant)
  E_wind     — aerodynamic drag from altitude-dependent headwind
  E_payload  — payload multiplier (heavier load = more energy)

All values are in Joules (J).
Battery capacity reference: 50 Wh = 180,000 J (typical delivery drone).
"""

import numpy as np
from typing import List, Tuple, Optional


# ── Physical parameters ──────────────────────────────────────────────────────
CELL_SIZE_M       = 100.0   # metres per grid cell
LAYER_HEIGHT_M    = 10.0    # metres per altitude layer

# Energy per unit
E_CRUISE_J_M      = 0.15    # J per metre, horizontal cruise
E_CLIMB_J_M       = 0.45    # J per metre, ascending (3× cruise)
E_DESCEND_J_M     = 0.05    # J per metre, descending
E_HOVER_J_STOP    = 120.0   # J per delivery stop (hovering ~10 seconds)
E_WIND_J_MS_M     = 0.02    # J per (m/s of wind × m of horizontal flight)
E_TURN_J          = 5.0     # J per 90° grid turn

# Wind model (must match environment_3d.py)
WIND_SCALE        = 2.5
WIND_EXP          = 1.4

# Battery
BATTERY_CAPACITY_J = 180_000.0   # 50 Wh in Joules
RESERVE_FRACTION   = 0.15         # keep 15% as safety reserve

USABLE_BATTERY_J   = BATTERY_CAPACITY_J * (1 - RESERVE_FRACTION)

# Non-linear degradation: capacity reduces as battery discharges
# Modelled as a simple Peukert-like efficiency drop
EFFICIENCY_FULL   = 1.00   # 100% efficient at full charge
EFFICIENCY_EMPTY  = 0.80   # 80% efficient when near empty


def _wind_at(z: int) -> float:
    """Wind speed (m/s) at altitude layer z."""
    if z <= 0:
        return 0.0
    return WIND_SCALE * (z ** WIND_EXP)


def _efficiency(soc: float) -> float:
    """
    Battery efficiency at state-of-charge `soc` (0.0..1.0).
    Uses linear interpolation between EFFICIENCY_EMPTY and EFFICIENCY_FULL.
    """
    soc   = max(0.0, min(1.0, soc))
    return EFFICIENCY_EMPTY + soc * (EFFICIENCY_FULL - EFFICIENCY_EMPTY)


# ─────────────────────────────────────────────────────────────────────────────

def leg_energy_2d(r1: int, c1: int, r2: int, c2: int,
                   path_length_m: Optional[float] = None,
                   payload_kg: float = 2.0,
                   current_charge_j: float = USABLE_BATTERY_J) -> float:
    """
    Estimate energy for a 2D ground-level flight leg.

    Args:
        r1, c1:           Source cell.
        r2, c2:           Destination cell.
        path_length_m:    Actual shortest-path length (metres). If None,
                          falls back to Euclidean estimate.
        payload_kg:       Package weight in kg.
        current_charge_j: Remaining battery charge (for Peukert model).

    Returns:
        Energy in Joules.
    """
    if path_length_m is None:
        path_length_m = np.hypot(r2-r1, c2-c1) * CELL_SIZE_M

    soc          = current_charge_j / BATTERY_CAPACITY_J
    eff          = _efficiency(soc)
    payload_mult = 1.0 + 0.08 * payload_kg

    e_cruise = E_CRUISE_J_M * path_length_m
    return (e_cruise * payload_mult) / eff


def leg_energy_3d(r1: int, c1: int, z1: int,
                   r2: int, c2: int, z2: int,
                   payload_kg: float = 2.0,
                   current_charge_j: float = USABLE_BATTERY_J) -> float:
    """
    Estimate energy for a 3D flight leg with altitude change and wind.

    Args:
        r1,c1,z1:         Source voxel.
        r2,c2,z2:         Destination voxel.
        payload_kg:       Package weight.
        current_charge_j: Remaining battery charge.

    Returns:
        Energy in Joules.
    """
    d_horiz_m = np.hypot(r2-r1, c2-c1) * CELL_SIZE_M
    dz_m      = (z2 - z1) * LAYER_HEIGHT_M

    soc          = current_charge_j / BATTERY_CAPACITY_J
    eff          = _efficiency(soc)
    payload_mult = 1.0 + 0.08 * payload_kg

    e_cruise  = E_CRUISE_J_M  * d_horiz_m
    e_climb   = E_CLIMB_J_M   * max(0.0, dz_m)
    e_descend = E_DESCEND_J_M * max(0.0, -dz_m)

    z_avg  = (z1 + z2) / 2.0
    wind   = _wind_at(int(round(z_avg)))
    e_wind = E_WIND_J_MS_M * wind * d_horiz_m

    total = (e_cruise + e_climb + e_descend + e_wind) * payload_mult / eff
    return total


def tour_energy_2d(delivery_points: List[Tuple[int,int]],
                    tour_order: List[int],
                    path_lengths: dict,
                    depot: Tuple[int,int],
                    payload_kg: float = 2.0) -> float:
    """
    Total energy for a complete 2D delivery tour.

    Uses a constant payload model (fixed payload_kg per leg) so that
    energy is directly proportional to distance, enabling fair comparison
    between MGA and ACO regardless of visit order.

    Args:
        delivery_points: List of (row, col) delivery positions.
        tour_order:      Permutation of indices into delivery_points.
        path_lengths:    Dict mapping (src, dst) -> metres.
        depot:           Depot (row, col).
        payload_kg:      Fixed payload for every leg (constant model).

    Returns:
        Total energy in Joules.
    """
    waypoints   = [depot] + [delivery_points[i] for i in tour_order] + [depot]
    total_e     = 0.0
    charge_left = USABLE_BATTERY_J

    for idx in range(len(waypoints) - 1):
        src, dst = waypoints[idx], waypoints[idx+1]
        dist_m   = path_lengths.get((src, dst),
                     np.hypot(dst[0]-src[0], dst[1]-src[1]) * CELL_SIZE_M)

        e_leg = leg_energy_2d(src[0], src[1], dst[0], dst[1],
                               dist_m, payload_kg, charge_left)

        # Hover energy at delivery stops (not at depot)
        if idx > 0 and idx < len(waypoints) - 2:
            e_leg += E_HOVER_J_STOP

        total_e     += e_leg
        charge_left  = max(0.0, charge_left - e_leg)

    return total_e


def is_feasible(total_energy_j: float) -> bool:
    """Return True if the tour can be completed with usable battery."""
    return total_energy_j <= USABLE_BATTERY_J


def battery_summary(total_energy_j: float) -> dict:
    """Return a dict with battery usage stats."""
    return {
        'total_energy_j':    round(total_energy_j, 1),
        'battery_used_%':    round(100 * total_energy_j / BATTERY_CAPACITY_J, 1),
        'reserve_used':      total_energy_j > USABLE_BATTERY_J,
        'feasible':          is_feasible(total_energy_j),
        'range_remaining_j': max(0.0, USABLE_BATTERY_J - total_energy_j),
    }


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print("=== Energy Model Smoke Test ===")

    # 2D leg: 50 cells (5 km), ground level
    e2d = leg_energy_2d(0, 0, 50, 0, path_length_m=5000)
    print(f"2D leg (5 km, ground): {e2d:.1f} J")

    # 3D leg: same horizontal distance but climbing to z=7
    e3d_low  = leg_energy_3d(0, 0, 0, 50, 0, 0)
    e3d_high = leg_energy_3d(0, 0, 0, 50, 0, 7)
    print(f"3D leg (stay low z=0→0): {e3d_low:.1f} J")
    print(f"3D leg (climb to z=7):   {e3d_high:.1f} J")
    print(f"3D wind penalty: {e3d_high - e3d_low:.1f} J extra for climbing")

    print(f"\nUsable battery: {USABLE_BATTERY_J:.0f} J")
    print(f"Max range (cruise only): {USABLE_BATTERY_J / E_CRUISE_J_M / 1000:.1f} km")
