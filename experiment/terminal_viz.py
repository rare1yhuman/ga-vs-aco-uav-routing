"""
Terminal Map Visualization (terminal_viz.py)
Compresses a large numpy grid (e.g. 200x200) into a smaller ANSI-colored block representation
suitable for a standard terminal window (e.g., 50x50 characters).
"""

import numpy as np
from rich.text import Text

# Block character for solid terminal rendering
BLOCK = "█"

def compress_grid(grid: np.ndarray, factor: int) -> np.ndarray:
    """Downsample the grid by 'factor' using max-pooling (if any sub-cell is blocked, block it)."""
    h, w = grid.shape
    new_h, new_w = h // factor, w // factor
    compressed = np.zeros((new_h, new_w), dtype=np.uint8)
    
    for i in range(new_h):
        for j in range(new_w):
            patch = grid[i*factor:(i+1)*factor, j*factor:(j+1)*factor]
            compressed[i, j] = np.max(patch)
            
    return compressed


def _bresenham_line(r0, c0, r1, c1):
    """Yields coordinates of a line between (r0, c0) and (r1, c1) using Bresenham's algorithm."""
    points = []
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc

    while True:
        points.append((r0, c0))
        if r0 == r1 and c0 == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r0 += sr
        if e2 < dr:
            err += dr
            c0 += sc
    return points


def render_map_to_rich(grid: np.ndarray, dps: list, depot: tuple, 
                       factor: int = 4, best_path: list = None) -> Text:
    """
    Renders the grid into a Rich Text object.
    Red blocks = NFZ
    Grey dots = Open space
    Blue triangle = Depot
    Yellow circle = Delivery point
    Green line = Best path
    """
    comp_grid = compress_grid(grid, factor)
    h, w = comp_grid.shape
    
    # Map points to compressed coordinates
    dps_comp = set((r // factor, c // factor) for (r, c) in dps)
    depot_comp = (depot[0] // factor, depot[1] // factor)
    
    # Pre-calculate path pixels if provided
    path_pixels = set()
    if best_path and len(best_path) > 1:
        # best_path is a list of node indices. 0 is depot, 1..N are dps.
        nodes = [depot] + list(dps)
        for i in range(len(best_path)):
            n1 = best_path[i]
            n2 = best_path[(i + 1) % len(best_path)]
            if n1 < len(nodes) and n2 < len(nodes):
                r0, c0 = nodes[n1][0] // factor, nodes[n1][1] // factor
                r1, c1 = nodes[n2][0] // factor, nodes[n2][1] // factor
                for pr, pc in _bresenham_line(r0, c0, r1, c1):
                    if 0 <= pr < h and 0 <= pc < w:
                        path_pixels.add((pr, pc))
    
    out_text = Text()
    
    for i in range(h):
        for j in range(w):
            if (i, j) == depot_comp:
                out_text.append("▲ ", style="bold blue")
            elif (i, j) in dps_comp:
                out_text.append("● ", style="bold yellow")
            elif comp_grid[i, j] == 1:
                out_text.append(BLOCK*2, style="red")
            elif (i, j) in path_pixels:
                out_text.append("::", style="bold green")
            else:
                out_text.append("· ", style="bright_black")
        
        if i < h - 1:
            out_text.append("\n")
            
    return out_text
