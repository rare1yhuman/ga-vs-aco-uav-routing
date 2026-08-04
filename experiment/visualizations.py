"""
Visualization Module — GA vs ACO Paper

Generates publication-quality figures for the manuscript:
1. 3D Trajectory Plot (buildings, NFZs, and drone paths)
2. Violin Plots (comparing distance and energy distributions)
3. Pareto Front (NSGA-II distance vs energy trade-off)
4. Radar/Spider Chart (multi-metric comparison across trials)

Output format: High-res PDF and PNG in <results-dir>/figures/
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Polygon
import matplotlib.colors as mcolors
from math import pi

# ── Styling Defaults ────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", context="paper", font_scale=1.5)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['axes.titlesize'] = 16
plt.rcParams['axes.labelsize'] = 14

COLORS = {
    'MGA': '#2ecc71',  # Emerald Green
    'ACO': '#e74c3c',  # Alizarin Red
    'Dijkstra': '#95a5a6', # Gray
    'NFZ': '#c0392b',
    'Depot': '#2980b9',
    'Delivery': '#f39c12'
}


def create_figure_dir(output_dir: Path) -> Path:
    fig_dir = output_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    return fig_dir


# ═══════════════════════════════════════════════════════════════════════════
# 1. Violin Plots (Distributions)
# ═══════════════════════════════════════════════════════════════════════════

def plot_violin_comparisons(master_csv: Path, raw_results_dir: Path, output_dir: Path):
    """Generate Violin plots for Distance and Energy across all trials."""
    if not master_csv.exists():
        print("[viz] Master CSV not found. Skipping violin plots.")
        return

    fig_dir = create_figure_dir(output_dir)
    
    # We need the raw data (all 50 runs) to plot distributions
    all_data = []
    
    # Parse trials dynamically from the results directory
    trials = []
    for d in raw_results_dir.iterdir():
        if d.is_dir() and d.name.startswith("trial_"):
            tid = d.name.split("_")[1]
            raw_csv = d / "raw_results.csv"
            if raw_csv.exists():
                df = pd.read_csv(raw_csv)
                df['Trial'] = tid
                all_data.append(df)
                trials.append(tid)
    
    if not all_data:
        print("[viz] No raw data found for violin plots.")
        return

    full_df = pd.concat(all_data, ignore_index=True)
    full_df = full_df[full_df['distance_m'] != float('inf')] # Drop failed runs

    # ── Distance Violin ──────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=full_df, x="Trial", y="distance_m", hue="algo",
        split=True, inner="quart", palette=[COLORS['MGA'], COLORS['ACO']],
        linewidth=1.5, order=sorted(trials)
    )
    plt.title("Tour Distance Distribution (50 runs per trial)")
    plt.ylabel("Total Distance (metres)")
    plt.xlabel("Experimental Trial")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(fig_dir / "violin_distance.pdf", dpi=300)
    plt.savefig(fig_dir / "violin_distance.png", dpi=300)
    plt.close()

    # ── Energy Violin ────────────────────────────────────────────────────────
    plt.figure(figsize=(12, 6))
    sns.violinplot(
        data=full_df, x="Trial", y="energy_j", hue="algo",
        split=True, inner="quart", palette=[COLORS['MGA'], COLORS['ACO']],
        linewidth=1.5, order=sorted(trials)
    )
    plt.title("Energy Consumption Distribution (50 runs per trial)")
    plt.ylabel("Total Energy (Joules)")
    plt.xlabel("Experimental Trial")
    plt.legend(title="Algorithm")
    plt.tight_layout()
    plt.savefig(fig_dir / "violin_energy.pdf", dpi=300)
    plt.savefig(fig_dir / "violin_energy.png", dpi=300)
    plt.close()
    
    print("[viz] Violin plots saved.")


# ═══════════════════════════════════════════════════════════════════════════
# 2. Radar/Spider Chart (Multi-metric comparison)
# ═══════════════════════════════════════════════════════════════════════════

def plot_radar_summary(master_csv: Path, output_dir: Path):
    """Plot a radar chart summarizing MGA vs ACO performance."""
    if not master_csv.exists():
        return

    df = pd.read_csv(master_csv)
    fig_dir = create_figure_dir(output_dir)

    # Average metrics across all trials (normalized where lower is better)
    metrics = ['Mean Distance', 'Best Distance', 'Mean Energy', 'Robustness (Std Dev)', 'Compute Time']
    
    # Calculate relative performance (MGA relative to ACO, lower = better, so invert for area)
    # Actually, let's normalize so 1.0 is the worst of the two, and smaller is better.
    # To make the polygon "larger is better", we invert: (worst - value) / worst
    
    mga_dist  = df['mga_mean_m'].mean()
    aco_dist  = df['aco_mean_m'].mean()
    mga_best  = df['mga_best_m'].mean()
    aco_best  = df['aco_best_m'].mean()
    mga_eng   = df['mga_energy_j'].mean()
    aco_eng   = df['aco_energy_j'].mean()
    mga_std   = df['mga_std_m'].mean()
    aco_std   = df['aco_std_m'].mean()
    mga_time  = df['mga_runtime_s'].mean()
    aco_time  = df['aco_runtime_s'].mean()

    def norm(m, a):
        # Normalize so larger value is 1.0 (outer edge), smaller is inside.
        # Since these are costs (distance, energy, time), inner = better.
        mx = max(m, a) if max(m, a) > 0 else 1.0
        return m/mx, a/mx

    m1, a1 = norm(mga_dist, aco_dist)
    m2, a2 = norm(mga_best, aco_best)
    m3, a3 = norm(mga_eng, aco_eng)
    m4, a4 = norm(mga_std, aco_std)
    m5, a5 = norm(mga_time, aco_time)

    mga_vals = [m1, m2, m3, m4, m5]
    aco_vals = [a1, a2, a3, a4, a5]

    # Close the loop
    mga_vals += mga_vals[:1]
    aco_vals += aco_vals[:1]
    
    angles = [n / float(len(metrics)) * 2 * pi for n in range(len(metrics))]
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # Draw one axe per variable + add labels
    plt.xticks(angles[:-1], metrics, color='grey', size=12)
    
    # Draw ylabels
    ax.set_rlabel_position(0)
    plt.yticks([0.25, 0.5, 0.75, 1.0], ["0.25", "0.5", "0.75", "1.0 (Worst)"], color="grey", size=10)
    plt.ylim(0, 1.1)

    # Plot data
    ax.plot(angles, mga_vals, linewidth=2, linestyle='solid', label='MGA-2opt', color=COLORS['MGA'])
    ax.fill(angles, mga_vals, COLORS['MGA'], alpha=0.1)

    ax.plot(angles, aco_vals, linewidth=2, linestyle='solid', label='ACO', color=COLORS['ACO'])
    ax.fill(angles, aco_vals, COLORS['ACO'], alpha=0.1)

    plt.title('Algorithm Profile (Smaller Area = Better Performance)', size=16, y=1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1))

    plt.tight_layout()
    plt.savefig(fig_dir / "radar_comparison.pdf", dpi=300)
    plt.savefig(fig_dir / "radar_comparison.png", dpi=300)
    plt.close()
    print("[viz] Radar chart saved.")


# ═══════════════════════════════════════════════════════════════════════════
# 3. Pareto Front (NSGA-II)
# ═══════════════════════════════════════════════════════════════════════════

def plot_pareto_front(pareto_data: list, output_dir: Path):
    """Plot NSGA-II Pareto front (distance vs energy)."""
    if not pareto_data:
        return
        
    fig_dir = create_figure_dir(output_dir)
    
    dists = [p['distance'] for p in pareto_data]
    energies = [p['energy'] for p in pareto_data]
    
    plt.figure(figsize=(10, 7))
    plt.scatter(dists, energies, c=COLORS['MGA'], s=100, alpha=0.7, edgecolor='black', label='Pareto Front Solutions')
    plt.plot(dists, energies, 'k--', alpha=0.3) # connecting line
    
    # Highlight knee point
    from nsga2 import _knee_point
    knee = _knee_point(pareto_data)
    if knee:
        plt.scatter(knee['distance'], knee['energy'], c='gold', marker='*', s=400, 
                    edgecolor='black', label='Knee Point (Best Trade-off)')
        
    plt.title("NSGA-II Pareto Front (Distance vs Energy)")
    plt.xlabel("Total Tour Distance (metres)")
    plt.ylabel("Total Energy Consumption (Joules)")
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    plt.savefig(fig_dir / "pareto_front.pdf", dpi=300)
    plt.savefig(fig_dir / "pareto_front.png", dpi=300)
    plt.close()
    print("[viz] Pareto front saved.")


# ═══════════════════════════════════════════════════════════════════════════
# 4. 3D Environment & Trajectory Plot
# ═══════════════════════════════════════════════════════════════════════════

def plot_3d_environment(env3d, best_tour_coords: list, output_dir: Path, filename="3d_trajectory"):
    """
    Plot the 3D environment with buildings (extruded NFZs) and drone flight path.
    env3d: Instance of Environment3D
    best_tour_coords: list of (x, y, z) voxel coordinates in order of flight.
    """
    fig_dir = create_figure_dir(output_dir)
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    cell_size = env3d.cell_size_m
    layer_height = env3d.layer_height_m
    
    # 1. Plot Buildings (Static NFZs)
    # To avoid plotting 10,000 individual voxels, we just draw the bounding boxes of the NFZ blocks.
    for block in env3d.nfz_blocks_2d:
        r_min, c_min, r_max, c_max = block
        
        # Convert grid coords to physical coords (meters)
        x_min = c_min * cell_size
        x_max = (c_max + 1) * cell_size
        y_min = r_min * cell_size
        y_max = (r_max + 1) * cell_size
        
        # Buildings are extruded from z=0 to some height (e.g. z=4 or 5)
        # For viz purposes, just draw a box up to z=50 meters
        z_min = 0
        z_max = 50 
        
        _plot_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color='gray', alpha=0.3)

    # 2. Plot Delivery Points
    # Extract delivery points from the tour (unique nodes)
    dps_set = set(best_tour_coords)
    dps_x = [p[1] * cell_size for p in dps_set if p != best_tour_coords[0]]
    dps_y = [p[0] * cell_size for p in dps_set if p != best_tour_coords[0]]
    dps_z = [p[2] * layer_height for p in dps_set if p != best_tour_coords[0]]
    
    ax.scatter(dps_x, dps_y, dps_z, c=COLORS['Delivery'], s=100, marker='o', edgecolors='black', label='Delivery Points')
    
    # Depot (start/end)
    depot = best_tour_coords[0]
    ax.scatter(depot[1]*cell_size, depot[0]*cell_size, depot[2]*layer_height, 
               c=COLORS['Depot'], s=200, marker='^', edgecolors='black', label='Depot')

    # 3. Plot Drone Trajectory
    path_x = [p[1] * cell_size for p in best_tour_coords]
    path_y = [p[0] * cell_size for p in best_tour_coords]
    path_z = [p[2] * layer_height for p in best_tour_coords]
    
    ax.plot(path_x, path_y, path_z, c=COLORS['MGA'], linewidth=3, linestyle='-', label='UAV Flight Path', zorder=10)

    # 4. Formatting
    ax.set_xlabel('X Distance (m)')
    ax.set_ylabel('Y Distance (m)')
    ax.set_zlabel('Altitude (m)')
    ax.set_title('3D UAV Routing in Urban Environment')
    ax.legend(loc='upper left', bbox_to_anchor=(0.0, 1.0))
    
    # Set limits based on grid size
    grid_size = env3d.grid_size
    ax.set_xlim(0, grid_size * cell_size)
    ax.set_ylim(0, grid_size * cell_size)
    ax.set_zlim(0, env3d.z_layers * layer_height)
    
    # Optimize view angle
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(fig_dir / f"{filename}.pdf", dpi=300)
    plt.savefig(fig_dir / f"{filename}.png", dpi=300)
    plt.close()
    print(f"[viz] 3D Trajectory {filename} saved.")


def _plot_3d_box(ax, x_min, x_max, y_min, y_max, z_min, z_max, color='gray', alpha=0.3):
    """Helper to draw a 3D box (building)."""
    # Define vertices
    v = np.array([
        [x_min, y_min, z_min], [x_max, y_min, z_min], [x_max, y_max, z_min], [x_min, y_max, z_min],
        [x_min, y_min, z_max], [x_max, y_min, z_max], [x_max, y_max, z_max], [x_min, y_max, z_max]
    ])
    
    # Define faces
    faces = [
        [v[0],v[1],v[2],v[3]], # bottom
        [v[4],v[5],v[6],v[7]], # top
        [v[0],v[1],v[5],v[4]], # front
        [v[2],v[3],v[7],v[6]], # back
        [v[1],v[2],v[6],v[5]], # right
        [v[0],v[3],v[7],v[4]]  # left
    ]
    
    # Plot faces
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    collection = Poly3DCollection(faces, alpha=alpha, facecolors=color, edgecolors='k', linewidths=0.5)
    ax.add_collection3d(collection)


# ═══════════════════════════════════════════════════════════════════════════
# Execute All Visualizations
# ═══════════════════════════════════════════════════════════════════════════

def generate_all_plots(results_dir="results"):
    res_path = Path(results_dir)
    print("\n" + "="*50)
    print("Generating Publication Figures")
    print("="*50)
    
    # 1. Violins
    plot_violin_comparisons(res_path / "master_results.csv", res_path, res_path)
    
    # 2. Radar
    plot_radar_summary(res_path / "master_results.csv", res_path)
    
    print(f"\n[viz] Visualizations complete. Check {res_path / 'figures'}/")


if __name__ == "__main__":
    generate_all_plots()
