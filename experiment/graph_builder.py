"""
Graph Builder Module
Converts grid to NetworkX graph with 4-connectivity.
"""

import networkx as nx
import numpy as np
from typing import Tuple, List, Dict
from grid_gen import CELL_NFZ, CELL_SIZE_M


def build_graph(grid: np.ndarray) -> nx.Graph:
    """
    Convert grid to NetworkX graph.

    Args:
        grid: N×N numpy array with cell types (default 100×100)

    Returns:
        NetworkX undirected graph with:
        - Nodes: (row, col) tuples for all free cells
        - Edges: 4-connectivity (no diagonals)
        - Edge weights: 100.0 meters (uniform)
    """
    G = nx.Graph()
    rows, cols = grid.shape

    # Add nodes for all passable cells (not NFZ)
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != CELL_NFZ:
                G.add_node((r, c))

    # Add edges for 4-connected neighbors
    for r in range(rows):
        for c in range(cols):
            if grid[r, c] != CELL_NFZ:
                # Check 4 neighbors (up, down, left, right)
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc

                    # Check bounds and that neighbor is passable
                    if (0 <= nr < rows and 0 <= nc < cols and
                        grid[nr, nc] != CELL_NFZ):
                        # Add edge with uniform weight (100m per cell)
                        G.add_edge((r, c), (nr, nc), weight=CELL_SIZE_M)

    return G


def validate_graph(G: nx.Graph, depot: Tuple[int, int],
                   delivery_points: List[Tuple[int, int]]) -> Dict:
    """
    Validate graph connectivity and compute statistics.

    Args:
        G: NetworkX graph
        depot: Depot position
        delivery_points: List of delivery point positions

    Returns:
        Dictionary with validation results
    """
    results = {
        'num_nodes': G.number_of_nodes(),
        'num_edges': G.number_of_edges(),
        'is_connected': nx.is_connected(G),
        'depot_exists': depot in G.nodes,
        'all_deliveries_exist': all(p in G.nodes for p in delivery_points),
        'paths_from_depot': {}
    }

    # Check paths from depot to all delivery points
    if results['depot_exists'] and results['all_deliveries_exist']:
        for i, point in enumerate(delivery_points):
            has_path = nx.has_path(G, depot, point)
            results['paths_from_depot'][f'D{i+1}'] = has_path

        results['all_paths_exist'] = all(results['paths_from_depot'].values())
    else:
        results['all_paths_exist'] = False

    return results


def get_passable_neighbors(G: nx.Graph, node: Tuple[int, int]) -> List[Tuple[int, int]]:
    """
    Get list of passable neighbors for a node.

    Args:
        G: NetworkX graph
        node: Node position (row, col)

    Returns:
        List of neighbor positions
    """
    return list(G.neighbors(node))


if __name__ == '__main__':
    """Test graph construction."""
    from grid_gen import generate_grid, DEPOT_POS

    print("Testing graph construction...")

    # Generate grid
    grid, delivery_points, nfz_blocks = generate_grid(seed=42, has_nfz=True)
    print(f"Grid: {len(nfz_blocks)} NFZ blocks, {len(delivery_points)} delivery points")

    # Build graph
    G = build_graph(grid)
    print(f"\nGraph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # Validate
    validation = validate_graph(G, DEPOT_POS, delivery_points)

    print("\nValidation Results:")
    print(f"  Connected: {validation['is_connected']}")
    print(f"  Depot exists: {validation['depot_exists']}")
    print(f"  All deliveries exist: {validation['all_deliveries_exist']}")
    print(f"  All paths from depot: {validation['all_paths_exist']}")

    if not validation['all_paths_exist']:
        print("\n  Path check details:")
        for key, value in validation['paths_from_depot'].items():
            print(f"    {key}: {value}")

    # Test path finding
    if validation['all_paths_exist']:
        test_point = delivery_points[0]
        path_length = nx.shortest_path_length(G, DEPOT_POS, test_point, weight='weight')
        print(f"\nShortest path from depot to D1: {path_length:.1f} meters")

    # Graph statistics
    print(f"\nGraph Statistics:")
    print(f"  Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
    print(f"  Number of connected components: {nx.number_connected_components(G)}")

    # Test no-NFZ grid
    print("\n\nTesting graph without NFZ...")
    grid_no_nfz, delivery_points_no_nfz, _ = generate_grid(seed=42, has_nfz=False)
    G_no_nfz = build_graph(grid_no_nfz)
    print(f"No-NFZ graph: {G_no_nfz.number_of_nodes()} nodes, {G_no_nfz.number_of_edges()} edges")

    validation_no_nfz = validate_graph(G_no_nfz, DEPOT_POS, delivery_points_no_nfz)
    print(f"Connected: {validation_no_nfz['is_connected']}")
    print(f"All paths exist: {validation_no_nfz['all_paths_exist']}")
