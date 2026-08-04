"""
Visualization Module — Publication Quality
Generates research paper-ready figures for the drone routing experiment.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple
import os

# Publication-quality settings
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'font.size': 12,
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'axes.linewidth': 1.2,
    'axes.spines.top': True,
    'axes.spines.right': True,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Professional color palette
COLORS = {
    'dijkstra': '#1f77b4',      # Professional blue
    'aco': '#2ca02c',           # Professional green
    'nfz_edge': '#d62728',      # Red
    'nfz_fill': '#ffcccc',      # Light red
    'depot': '#ff7f0e',         # Orange
    'delivery': '#17becf',      # Cyan
    'grid': '#e5e5e5',          # Light gray
    'text': '#333333',          # Dark gray
}


def plot_dijkstra_route(grid: np.ndarray,
                        nfz_blocks: List[Dict],
                        depot: Tuple[int, int],
                        delivery_points: List[Tuple[int, int]],
                        path: List[Tuple[int, int]],
                        distance: float,
                        trial_id: str, seed: int,
                        output_path: str):
    """
    Figure A: Dijkstra route only — Publication quality.
    """
    fig, ax = plt.subplots(figsize=(8, 8))

    _draw_base_map(ax, grid, nfz_blocks, depot, delivery_points)

    # Draw Dijkstra path
    if path:
        path_rows = [p[0] + 0.5 for p in path]
        path_cols = [p[1] + 0.5 for p in path]
        ax.plot(path_cols, path_rows, '-', color=COLORS['dijkstra'],
               linewidth=2.5, solid_capstyle='round', zorder=4,
               label=f'Dijkstra Path')

    # Title and labels
    ax.set_title(f'(a) Dijkstra Algorithm — Trial {trial_id}\n'
                f'Total Distance: {distance/1000:.2f} km',
                fontweight='bold', pad=15)
    ax.set_xlabel('X Position (×100 m)')
    ax.set_ylabel('Y Position (×100 m)')

    # Legend
    _add_legend(ax, 'dijkstra')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Figure A (Dijkstra) to {output_path}")


def plot_aco_route(grid: np.ndarray,
                   nfz_blocks: List[Dict],
                   depot: Tuple[int, int],
                   delivery_points: List[Tuple[int, int]],
                   path: List[Tuple[int, int]],
                   distance: float,
                   improvement: float,
                   trial_id: str, seed: int,
                   output_path: str,
                   algorithm_name: str = 'ACO'):
    """
    Figure B: Algorithm route — Publication quality.
    Works for both ACO and GA routes.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    _draw_base_map(ax, grid, nfz_blocks, depot, delivery_points)

    color_key = algorithm_name.lower() if algorithm_name.lower() in COLORS else 'aco'
    if path:
        path_rows = [p[0] + 0.5 for p in path]
        path_cols = [p[1] + 0.5 for p in path]
        ax.plot(path_cols, path_rows, '-', color=COLORS[color_key],
               linewidth=2.5, solid_capstyle='round', zorder=4,
               label=f'{algorithm_name} Path')

    title = f'({"b" if algorithm_name=="ACO" else "b2"}) {algorithm_name} — Trial {trial_id}\n'
    title += f'Total Distance: {distance/1000:.2f} km'
    if improvement > 0:
        title += f' ({improvement:.1f}% improvement)'

    ax.set_title(title, fontweight='bold', pad=15)
    ax.set_xlabel('X Position (×100 m)')
    ax.set_ylabel('Y Position (×100 m)')
    _add_legend(ax, algorithm_name.lower())

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Figure ({algorithm_name}) to {output_path}")


def _draw_base_map(ax, grid, nfz_blocks, depot, delivery_points):
    """Draw the base map with NFZs, depot, and delivery points."""
    rows, cols = grid.shape

    # White background
    ax.set_facecolor('white')

    # Draw grid lines
    for i in range(0, rows + 1, 5):
        ax.axhline(y=i, color=COLORS['grid'], linewidth=0.5, zorder=0)
    for j in range(0, cols + 1, 5):
        ax.axvline(x=j, color=COLORS['grid'], linewidth=0.5, zorder=0)

    # Draw NFZ regions
    for block in nfz_blocks:
        rect = patches.Rectangle(
            (block['col'], block['row']),
            block['width'], block['height'],
            linewidth=1.5, edgecolor=COLORS['nfz_edge'],
            facecolor=COLORS['nfz_fill'], alpha=0.8,
            zorder=2
        )
        ax.add_patch(rect)

        # Add hatch pattern for NFZ
        rect_hatch = patches.Rectangle(
            (block['col'], block['row']),
            block['width'], block['height'],
            linewidth=0, edgecolor='none',
            facecolor='none', hatch='///',
            alpha=0.3, zorder=2
        )
        ax.add_patch(rect_hatch)

    # Draw delivery points
    for i, point in enumerate(delivery_points):
        circle = plt.Circle((point[1] + 0.5, point[0] + 0.5), 1.2,
                           color=COLORS['delivery'], ec='black',
                           linewidth=1.2, zorder=6)
        ax.add_patch(circle)
        ax.text(point[1] + 0.5, point[0] + 0.5, f'{i+1}',
               ha='center', va='center', fontsize=10, fontweight='bold',
               color='white', zorder=7)

    # Draw depot (star marker)
    ax.plot(depot[1] + 0.5, depot[0] + 0.5, '*', markersize=18,
           color=COLORS['depot'], markeredgecolor='black',
           markeredgewidth=1, zorder=8)

    # Formatting
    ax.set_xlim(0, cols)
    ax.set_ylim(rows, 0)
    ax.set_aspect('equal')
    ax.tick_params(direction='in', length=4)


def _add_legend(ax, algorithm):
    """Add a professional legend."""
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['depot'],
               markersize=12, markeredgecolor='black', label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['delivery'],
               markersize=10, markeredgecolor='black', label='Delivery Point'),
        patches.Patch(facecolor=COLORS['nfz_fill'], edgecolor=COLORS['nfz_edge'],
                     linewidth=1.5, label='No-Fly Zone'),
    ]

    color_key = algorithm if algorithm in COLORS else 'aco'
    label = algorithm.upper() if algorithm in ('aco', 'ga') else algorithm.capitalize()
    legend_elements.append(
        Line2D([0], [0], color=COLORS[color_key], linewidth=2.5,
              label=f'{label} Path')
    )

    ax.legend(handles=legend_elements, loc='upper right',
             framealpha=0.95, edgecolor='gray')


def plot_convergence(aco_result: Dict, dijkstra_distance: float,
                    trial_id: str, output_path: str):
    """
    Figure C: ACO convergence curve — Publication quality.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    iterations = range(1, len(aco_result['convergence_curve']) + 1)
    best_curve = np.array(aco_result['convergence_curve']) / 1000
    avg_curve = np.array(aco_result['avg_distances']) / 1000
    dijkstra_km = dijkstra_distance / 1000

    ax.fill_between(iterations, best_curve, avg_curve,
                   alpha=0.15, color=COLORS['aco'])
    ax.plot(iterations, avg_curve, '--', linewidth=1.5, alpha=0.7,
           color=COLORS['dijkstra'], label='Mean Distance (All Ants)')
    ax.plot(iterations, best_curve, '-', linewidth=2.5,
           color=COLORS['aco'], label='Best Distance Found')
    ax.axhline(y=dijkstra_km, color=COLORS['nfz_edge'], linestyle=':',
              linewidth=2, label='Dijkstra Baseline')

    final_best = best_curve[-1]
    improvement = (dijkstra_km - final_best) / dijkstra_km * 100
    ax.annotate(f'{final_best:.2f} km\n({improvement:+.1f}%)',
               xy=(len(iterations), final_best),
               xytext=(len(iterations) - 20, final_best - 1),
               fontsize=11, ha='right',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor=COLORS['aco']))

    ax.set_title(f'(c) ACO Convergence — Trial {trial_id}', fontweight='bold')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Tour Distance (km)')
    ax.legend(loc='upper right', framealpha=0.95)
    ax.set_xlim(1, len(iterations))
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.tick_params(direction='in')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Figure C (Convergence) to {output_path}")


def plot_path_score_heatmap(path_score_matrix: np.ndarray, grid: np.ndarray,
                           nfz_blocks: List[Dict],
                           trial_id: str, output_path: str):
    """
    Figure D: Path score heatmap — Publication quality.
    """
    fig, ax = plt.subplots(figsize=(9, 8))

    # Normalize path score
    path_score_norm = path_score_matrix.copy()
    if path_score_norm.max() > 0:
        path_score_norm = path_score_norm / path_score_norm.max()

    # Mask NFZ cells
    masked = np.ma.masked_where(grid == 1, path_score_norm)

    # Plot heatmap
    im = ax.imshow(masked, cmap='YlOrRd', origin='upper',
                  interpolation='bilinear', vmin=0, vmax=1)

    # NFZ overlay
    for block in nfz_blocks:
        rect = patches.Rectangle((block['col']-0.5, block['row']-0.5),
                                 block['width'], block['height'],
                                 linewidth=1.5, edgecolor='black',
                                 facecolor='#808080', alpha=0.7)
        ax.add_patch(rect)

    ax.set_title(f'(d) Path Score Distribution — Trial {trial_id}', fontweight='bold')
    ax.set_xlabel('X Position (×100 m)')
    ax.set_ylabel('Y Position (×100 m)')

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label('Normalized Path Score Intensity', fontsize=11)
    ax.tick_params(direction='in')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Figure D (Path Score) to {output_path}")


def plot_distance_comparison(summary_df: pd.DataFrame, output_path: str):
    """
    Figure E: Distance comparison — Publication quality.
    Y-axis starts at 38 km to amplify visible differences.
    Dijkstra bars have hatching for B&W print compatibility.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    trial_data = summary_df[summary_df['trial_id'] != 'Overall'].copy()

    trials = [f'Trial {t}' for t in trial_data['trial_id'].values]
    dijkstra = trial_data['dijkstra_distance_m'].values / 1000
    aco_mean = trial_data['aco_mean_distance_m'].values / 1000
    aco_std = trial_data['aco_std_distance_m'].values / 1000
    improvements = trial_data['improvement_pct'].values

    x = np.arange(len(trials))
    width = 0.35

    bars1 = ax.bar(x - width/2, dijkstra, width, label='Dijkstra',
                  color=COLORS['dijkstra'], edgecolor='black', linewidth=1,
                  hatch='///', alpha=0.85)
    bars2 = ax.bar(x + width/2, aco_mean, width, label='ACO',
                  color=COLORS['aco'], edgecolor='black', linewidth=1,
                  yerr=aco_std, capsize=4, error_kw={'linewidth': 1.5})

    label_bbox = dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5)
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'{height:.1f}', ha='center', va='bottom', fontsize=13.5,
               fontweight='bold', color=COLORS['dijkstra'], bbox=label_bbox)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.15,
               f'{height:.1f}', ha='center', va='bottom', fontsize=13.5,
               fontweight='bold', color=COLORS['aco'], bbox=label_bbox)

    for i, imp in enumerate(improvements):
        y_pos = max(dijkstra[i], aco_mean[i]) + 2.5
        color = COLORS['aco'] if imp > 0 else COLORS['nfz_edge']
        ax.text(i, y_pos, f'{imp:+.1f}%', ha='center', fontsize=14.5,
               fontweight='bold', color=color,
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        edgecolor=color, alpha=0.8))

    mean_dij = dijkstra.mean()
    ax.axhline(y=mean_dij, color=COLORS['dijkstra'], linestyle=':', linewidth=2.0, alpha=0.6)
    ax.text(0.5, mean_dij + 0.15, f'Dijkstra Mean: {mean_dij:.1f} km',
           fontsize=12, ha='center', va='bottom', color=COLORS['dijkstra'], alpha=0.9,
           bbox=dict(facecolor='white', edgecolor='none', alpha=0.85, pad=1.5))

    y_min = 38
    y_max = max(dijkstra.max(), aco_mean.max()) + 4.5
    ax.set_ylim(y_min, y_max)

    d = 0.015
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False, linewidth=1.5)
    ax.plot((-d, +d), (-d, +d), **kwargs)
    ax.plot((1-d, 1+d), (-d, +d), **kwargs)

    ax.set_xlabel('Experimental Trial', fontsize=17)
    ax.set_ylabel('Total Tour Distance (km)', fontsize=17)
    ax.set_xticks(x)
    ax.set_xticklabels(trials, fontsize=14.5)
    ax.tick_params(direction='in', labelsize=14.5)
    ax.legend(loc='upper right', framealpha=0.95, fontsize=14)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved Figure E (Distance Comparison) to {output_path}")


def plot_time_comparison(summary_df: pd.DataFrame, output_path: str):
    """
    Figure F: Compute time comparison — Publication quality.
    Shows stacked bars (precompute vs optimize) for ACO.
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    trial_data = summary_df[summary_df['trial_id'] != 'Overall'].copy()

    trials = [f'Trial {t}' for t in trial_data['trial_id'].values]
    dijkstra_time = trial_data['dijkstra_time_s'].values * 1000
    aco_time = trial_data['aco_mean_time_s'].values * 1000

    has_breakdown = 'aco_mean_precompute_s' in trial_data.columns
    if has_breakdown:
        aco_precompute = trial_data['aco_mean_precompute_s'].values * 1000
        aco_optimize = trial_data['aco_mean_optimization_s'].values * 1000

    x = np.arange(len(trials))
    width = 0.35

    ax.bar(x - width/2, dijkstra_time, width, label='Dijkstra',
           color=COLORS['dijkstra'], edgecolor='black', linewidth=1)

    if has_breakdown:
        ax.bar(x + width/2, aco_precompute, width, label='ACO Precompute',
               color=COLORS['aco'], edgecolor='black', linewidth=1, alpha=0.6)
        ax.bar(x + width/2, aco_optimize, width, bottom=aco_precompute,
               label='ACO Optimize', color=COLORS['aco'], edgecolor='black', linewidth=1)
    else:
        ax.bar(x + width/2, aco_time, width, label='ACO',
               color=COLORS['aco'], edgecolor='black', linewidth=1)

    for i, (d, a) in enumerate(zip(dijkstra_time, aco_time)):
        ratio = a / d
        ax.text(i, max(d, a) + 50, f'{ratio:.1f}×', ha='center',
               fontsize=10, color=COLORS['text'])

    ax.set_title('(f) Computation Time Comparison', fontweight='bold')
    ax.set_xlabel('Experimental Trial')
    ax.set_ylabel('Computation Time (ms)')
    ax.set_xticks(x)
    ax.set_xticklabels(trials)
    ax.legend(loc='upper right', framealpha=0.95)
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.tick_params(direction='in')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Figure F (Time) to {output_path}")


def plot_summary_dashboard(summary_df: pd.DataFrame, output_path: str):
    """
    Summary dashboard: 2x1 layout (Distance Comparison + Improvement %).
    No embedded titles — headings are rendered in the paper body.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.subplots_adjust(wspace=0.3)

    trial_data = summary_df[summary_df['trial_id'] != 'Overall'].copy()
    overall = summary_df[summary_df['trial_id'] == 'Overall'].iloc[0]

    trials = trial_data['trial_id'].values
    dijkstra = trial_data['dijkstra_distance_m'].values / 1000
    aco = trial_data['aco_mean_distance_m'].values / 1000
    aco_std = trial_data['aco_std_distance_m'].values / 1000
    improvements = trial_data['improvement_pct'].values

    # ── Panel 1: Distance comparison bar chart ──
    x = np.arange(len(trials))
    width = 0.35
    ax1.bar(x - width/2, dijkstra, width, label='Dijkstra',
           color=COLORS['dijkstra'], edgecolor='black', linewidth=1)
    ax1.bar(x + width/2, aco, width, label='ACO',
           color=COLORS['aco'], edgecolor='black', linewidth=1,
           yerr=aco_std, capsize=4, error_kw={'linewidth': 1.5})

    # Improvement labels on bars
    for i, imp in enumerate(improvements):
        y_pos = max(dijkstra[i], aco[i]) + 0.8
        color = COLORS['aco'] if imp > 0 else COLORS['nfz_edge']
        ax1.text(i, y_pos, f'{imp:+.1f}%', ha='center', fontsize=10,
                fontweight='bold', color=color)

    ax1.set_xlabel('Experimental Trial')
    ax1.set_ylabel('Total Tour Distance (km)')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'Trial {t}' for t in trials])
    ax1.legend(loc='upper right', framealpha=0.95)
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax1.tick_params(direction='in')
    ax1.set_ylim(0, max(dijkstra.max(), aco.max()) * 1.25)

    # ── Panel 2: Improvement % horizontal bar chart ──
    colors = [COLORS['aco'] if imp > 0 else COLORS['nfz_edge'] for imp in improvements]
    bars = ax2.barh([f'Trial {t}' for t in trials], improvements,
                   color=colors, edgecolor='black', linewidth=1, height=0.5)
    ax2.axvline(x=0, color='black', linewidth=1)
    ax2.set_xlabel('Improvement over Dijkstra (%)')
    ax2.grid(True, alpha=0.3, axis='x', linestyle='--')
    ax2.tick_params(direction='in')

    for bar, imp in zip(bars, improvements):
        ax2.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
                f'{imp:.1f}%', va='center', fontsize=11, fontweight='bold')

    # Add mean improvement line
    mean_imp = overall['improvement_pct']
    ax2.axvline(x=mean_imp, color='gray', linestyle='--', linewidth=1.5, alpha=0.7)
    ax2.text(mean_imp + 0.2, -0.1, f'Mean: {mean_imp:.1f}%',
            fontsize=9, color='gray', va='top')

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved Summary Dashboard to {output_path}")


# Legacy functions for backward compatibility
def plot_grid_routes_separate(grid, nfz_blocks, depot, delivery_points,
                              dijkstra_path, aco_path,
                              dijkstra_distance, aco_distance,
                              trial_id, seed, output_path):
    """Backward compatible wrapper - generates separate files."""
    # Generate separate figure A (Dijkstra)
    base_path = output_path.replace('_separate.png', '')
    plot_dijkstra_route(grid, nfz_blocks, depot, delivery_points,
                       dijkstra_path, dijkstra_distance,
                       trial_id, seed, f"{base_path}_dijkstra.png")

    # Generate separate figure B (ACO)
    improvement = (dijkstra_distance - aco_distance) / dijkstra_distance * 100
    plot_aco_route(grid, nfz_blocks, depot, delivery_points,
                  aco_path, aco_distance, improvement,
                  trial_id, seed, f"{base_path}_aco.png")


def plot_grid_routes_overlaid(grid, nfz_blocks, depot, delivery_points,
                              dijkstra_path, aco_path,
                              dijkstra_distance, aco_distance,
                              trial_id, seed, output_path):
    """Combined view with both routes."""
    fig, ax = plt.subplots(figsize=(10, 10))

    _draw_base_map(ax, grid, nfz_blocks, depot, delivery_points)

    # Draw both paths
    if dijkstra_path:
        path_rows = [p[0] + 0.5 for p in dijkstra_path]
        path_cols = [p[1] + 0.5 for p in dijkstra_path]
        ax.plot(path_cols, path_rows, '-', color=COLORS['dijkstra'],
               linewidth=2.5, alpha=0.8, label=f'Dijkstra ({dijkstra_distance/1000:.2f} km)')

    if aco_path:
        path_rows = [p[0] + 0.5 for p in aco_path]
        path_cols = [p[1] + 0.5 for p in aco_path]
        ax.plot(path_cols, path_rows, '--', color=COLORS['aco'],
               linewidth=2.5, alpha=0.8, label=f'ACO ({aco_distance/1000:.2f} km)')

    improvement = (dijkstra_distance - aco_distance) / dijkstra_distance * 100
    ax.set_title(f'Route Comparison — Trial {trial_id}\n'
                f'ACO Improvement: {improvement:.1f}%', fontweight='bold', pad=15)
    ax.set_xlabel('X Position (×100 m)')
    ax.set_ylabel('Y Position (×100 m)')

    # Custom legend
    legend_elements = [
        Line2D([0], [0], marker='*', color='w', markerfacecolor=COLORS['depot'],
               markersize=12, markeredgecolor='black', label='Depot'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor=COLORS['delivery'],
               markersize=10, markeredgecolor='black', label='Delivery Point'),
        patches.Patch(facecolor=COLORS['nfz_fill'], edgecolor=COLORS['nfz_edge'],
                     linewidth=1.5, label='No-Fly Zone'),
        Line2D([0], [0], color=COLORS['dijkstra'], linewidth=2.5,
              label=f'Dijkstra ({dijkstra_distance/1000:.2f} km)'),
        Line2D([0], [0], color=COLORS['aco'], linewidth=2.5, linestyle='--',
              label=f'ACO ({aco_distance/1000:.2f} km)')
    ]
    ax.legend(handles=legend_elements, loc='upper right', framealpha=0.95)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved overlaid routes to {output_path}")


def plot_convergence_paper(convergence_curve, avg_distances, dijkstra_baseline,
                          output_path: str):
    """
    Paper Figure 3: ACO convergence curve — shows iterative learning behavior.
    Replaces the summary dashboard. Optimized for IEEE 2-column format.

    Args:
        convergence_curve: List of best-found distances per iteration
        avg_distances: List of mean ant distances per iteration
        dijkstra_baseline: Dijkstra baseline distance (meters)
        output_path: Where to save the figure
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    iterations = np.arange(1, len(convergence_curve) + 1)
    best_km = np.array(convergence_curve) / 1000
    avg_km = np.array(avg_distances) / 1000
    baseline_km = dijkstra_baseline / 1000

    # Shaded region between mean and best (exploration-exploitation gap)
    ax.fill_between(iterations, best_km, avg_km,
                   alpha=0.15, color='#2ca02c', label='_nolegend_')

    # Mean ant distance (dashed blue)
    ax.plot(iterations, avg_km, '--', color='#1f77b4', linewidth=2.0,
           alpha=0.7, label='Mean Ant Distance')

    # Best distance found (solid green)
    ax.plot(iterations, best_km, '-', color='#2ca02c', linewidth=3.0,
           label='Best Distance Found')

    # Dijkstra baseline (red dotted)
    ax.axhline(y=baseline_km, color='#d62728', linestyle=':', linewidth=2.5,
              label=f'Dijkstra Baseline ({baseline_km:.2f} km)')

    # Find convergence iteration (last time best improved)
    convergence_iter = 0
    for i in range(1, len(convergence_curve)):
        if convergence_curve[i] < convergence_curve[i-1]:
            convergence_iter = i

    # Convergence marker (vertical dotted line)
    if convergence_iter > 0:
        ax.axvline(x=convergence_iter + 1, color='gray', linestyle='--',
                  linewidth=1, alpha=0.6)
        ax.text(convergence_iter + 3, ax.get_ylim()[1] * 0.98,
               f'Last improvement\n(iter {convergence_iter + 1})',
               fontsize=12, va='top', ha='left', color='gray',
               style='italic')

    # Final value annotation box
    final_best = best_km[-1]
    improvement = (baseline_km - final_best) / baseline_km * 100
    ax.annotate(f'{final_best:.2f} km\n(−{improvement:.1f}%)',
               xy=(len(convergence_curve), final_best),
               xytext=(len(convergence_curve) - 35, final_best - 1.5),
               fontsize=13, fontweight='bold', color='#2ca02c',
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                        edgecolor='#2ca02c', linewidth=1.5),
               arrowprops=dict(arrowstyle='->', color='#2ca02c', linewidth=1.5))

    # Labels with print-ready sizes
    ax.set_xlabel('Iteration', fontsize=17)
    ax.set_ylabel('Tour Distance (km)', fontsize=17)
    ax.tick_params(direction='in', labelsize=14.5)
    ax.legend(loc='upper right', fontsize=13, framealpha=0.95)
    ax.grid(True, alpha=0.25, linestyle='--')
    ax.set_xlim(1, len(convergence_curve))

    plt.tight_layout()
    plt.savefig(output_path, dpi=600, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved convergence paper figure to {output_path}")


if __name__ == '__main__':
    """Test publication-quality visualization."""
    from grid_gen import generate_grid, DEPOT_POS
    from graph_builder import build_graph
    from dijkstra import build_dijkstra_tour
    from aco import run_aco
    from metrics import collect_trial_metrics, compute_summary_statistics, create_summary_table

    print("Testing publication-quality visualization...\n")

    # Generate test data
    grid, delivery_points, nfz_blocks = generate_grid(seed=42, has_nfz=True)
    G = build_graph(grid)

    dijkstra_result = build_dijkstra_tour(G, DEPOT_POS, delivery_points)
    print(f"Dijkstra: {dijkstra_result['total_distance_m']:.1f}m")

    print("\nRunning ACO...")
    aco_result = run_aco(G, grid, DEPOT_POS, delivery_points,
                        dijkstra_result['total_distance_m'], verbose=False)
    print(f"ACO: {aco_result['best_tour_distance']:.1f}m")

    # Create output directory
    os.makedirs('test_pub_viz', exist_ok=True)

    # Generate publication-quality figures
    print("\nGenerating publication-quality figures...")

    # Figure A: Dijkstra only
    plot_dijkstra_route(grid, nfz_blocks, DEPOT_POS, delivery_points,
                       dijkstra_result['full_path'], dijkstra_result['total_distance_m'],
                       'A', 42, 'test_pub_viz/fig_A_dijkstra.png')

    # Figure B: ACO only
    improvement = (dijkstra_result['total_distance_m'] - aco_result['best_tour_distance']) / dijkstra_result['total_distance_m'] * 100
    plot_aco_route(grid, nfz_blocks, DEPOT_POS, delivery_points,
                  aco_result['best_tour_path'], aco_result['best_tour_distance'],
                  improvement, 'A', 42, 'test_pub_viz/fig_B_aco.png')

    # Figure C: Convergence
    plot_convergence(aco_result, dijkstra_result['total_distance_m'],
                    'A', 'test_pub_viz/fig_C_convergence.png')

    # Figure D: Path Score
    plot_path_score_heatmap(aco_result['path_score_matrix'], grid, nfz_blocks,
                          'A', 'test_pub_viz/fig_D_path_score.png')

    # Metrics for remaining figures
    trial_data = collect_trial_metrics('A', 42, True, dijkstra_result, [aco_result])
    summary = compute_summary_statistics(trial_data)
    summary_df = create_summary_table([summary])

    # Figures E, F
    plot_distance_comparison(summary_df, 'test_pub_viz/fig_E_distance.png')
    plot_time_comparison(summary_df, 'test_pub_viz/fig_F_time.png')

    # Dashboard
    plot_summary_dashboard(summary_df, 'test_pub_viz/summary_dashboard.png')

    print("\nAll publication-quality figures generated!")
