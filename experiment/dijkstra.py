"""
Dijkstra Routing Module
Baseline algorithm using nearest-neighbor ordering + Dijkstra shortest paths.
"""

import networkx as nx
import numpy as np
import time
from typing import List, Tuple, Dict
from grid_gen import CELL_SIZE_M


def build_dijkstra_tour(G: nx.Graph, depot: Tuple[int, int],
                        delivery_points: List[Tuple[int, int]]) -> Dict:
    """
    Build full delivery tour using Dijkstra algorithm.

    Phase 1: Order delivery points using nearest-neighbor greedy heuristic
    Phase 2: Find shortest path between consecutive points using Dijkstra

    Args:
        G: NetworkX graph
        depot: Depot position (row, col)
        delivery_points: List of delivery point positions

    Returns:
        Dictionary with:
        - stop_order: Ordered list of points to visit
        - full_path: Complete cell-by-cell path
        - total_distance_m: Total tour length in meters
        - leg_distances_m: Distance of each leg
        - compute_time_s: Computation time in seconds
    """
    start_time = time.perf_counter()

    # Phase 1: Nearest-neighbor ordering
    stop_order = nearest_neighbor_order(G, depot, delivery_points)

    # Phase 2: Build full path using Dijkstra for each leg
    full_path = []
    leg_distances = []

    for i in range(len(stop_order) - 1):
        source = stop_order[i]
        target = stop_order[i + 1]

        # Find shortest path for this leg
        leg_path = nx.shortest_path(G, source, target, weight='weight')
        leg_distance = nx.shortest_path_length(G, source, target, weight='weight')

        # Append to full path (avoid duplicating waypoints)
        if i == 0:
            full_path.extend(leg_path)
        else:
            full_path.extend(leg_path[1:])  # Skip first node (already in path)

        leg_distances.append(leg_distance)

    compute_time = time.perf_counter() - start_time

    return {
        'stop_order': stop_order,
        'full_path': full_path,
        'total_distance_m': sum(leg_distances),
        'leg_distances_m': leg_distances,
        'compute_time_s': compute_time,
        'num_waypoints': len(full_path)
    }


def nearest_neighbor_order(G: nx.Graph, depot: Tuple[int, int],
                           delivery_points: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    """
    Order delivery points using nearest-neighbor greedy heuristic.

    Uses graph distance (shortest path length), not Euclidean distance,
    to account for NFZ obstacles.

    Args:
        G: NetworkX graph
        depot: Starting position
        delivery_points: List of delivery points to visit

    Returns:
        Ordered list: [depot, closest_point, next_closest, ..., depot]
    """
    current = depot
    unvisited = set(delivery_points)
    tour = [depot]

    while unvisited:
        # Find nearest unvisited point (by graph distance)
        nearest = min(unvisited, key=lambda p: nx.shortest_path_length(G, current, p, weight='weight'))
        tour.append(nearest)
        current = nearest
        unvisited.remove(nearest)

    # Return to depot
    tour.append(depot)

    return tour


if __name__ == '__main__':
    """Test Dijkstra routing."""
    from grid_gen import generate_grid, DEPOT_POS
    from graph_builder import build_graph, validate_graph

    print("Testing Dijkstra routing...\n")

    # Generate grid and graph
    grid, delivery_points, nfz_blocks = generate_grid(seed=42, has_nfz=True)
    print(f"Grid: {len(nfz_blocks)} NFZ blocks, {len(delivery_points)} delivery points")

    G = build_graph(grid)
    validation = validate_graph(G, DEPOT_POS, delivery_points)
    print(f"Graph: {validation['num_nodes']} nodes, {validation['num_edges']} edges, connected={validation['is_connected']}")

    # Run Dijkstra
    print("\nRunning Dijkstra...")
    result = build_dijkstra_tour(G, DEPOT_POS, delivery_points)

    print(f"\nResults:")
    print(f"  Total distance: {result['total_distance_m']:.1f} meters ({result['total_distance_m']/1000:.2f} km)")
    print(f"  Number of waypoints: {result['num_waypoints']}")
    print(f"  Compute time: {result['compute_time_s']*1000:.2f} ms")
    print(f"  Number of legs: {len(result['leg_distances_m'])}")

    print(f"\n  Stop order:")
    for i, stop in enumerate(result['stop_order']):
        if i == 0:
            print(f"    {i}: Depot {stop}")
        elif i == len(result['stop_order']) - 1:
            print(f"    {i}: Return to Depot {stop}")
        else:
            delivery_idx = delivery_points.index(stop)
            print(f"    {i}: D{delivery_idx+1} {stop}")

    print(f"\n  Leg distances:")
    for i, dist in enumerate(result['leg_distances_m']):
        print(f"    Leg {i+1}: {dist:.1f} m")

    # Verify path validity (no NFZ cells)
    nfz_violations = 0
    for cell in result['full_path']:
        if grid[cell] == 1:  # NFZ cell
            nfz_violations += 1

    print(f"\n  Path validation:")
    print(f"    NFZ violations: {nfz_violations} (should be 0)")
    print(f"    Path valid: {nfz_violations == 0}")

    # Test without NFZ
    print("\n\n" + "="*60)
    print("Testing Dijkstra without NFZ...\n")

    grid_no_nfz, delivery_points_no_nfz, _ = generate_grid(seed=42, has_nfz=False)
    G_no_nfz = build_graph(grid_no_nfz)
    result_no_nfz = build_dijkstra_tour(G_no_nfz, DEPOT_POS, delivery_points_no_nfz)

    print(f"No-NFZ Results:")
    print(f"  Total distance: {result_no_nfz['total_distance_m']:.1f} meters ({result_no_nfz['total_distance_m']/1000:.2f} km)")
    print(f"  Compute time: {result_no_nfz['compute_time_s']*1000:.2f} ms")

    print(f"\n  Comparison:")
    print(f"    With NFZ: {result['total_distance_m']:.1f} m")
    print(f"    No NFZ:   {result_no_nfz['total_distance_m']:.1f} m")
    print(f"    Overhead: {(result['total_distance_m'] - result_no_nfz['total_distance_m']):.1f} m " +
          f"({(result['total_distance_m'] / result_no_nfz['total_distance_m'] - 1)*100:.1f}% longer)")
