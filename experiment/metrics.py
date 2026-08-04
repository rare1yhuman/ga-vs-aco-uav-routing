"""
Metrics Collection Module
Collects, aggregates, and exports metrics from experiments.
"""

import pandas as pd
import numpy as np
from typing import List, Dict
import os


def collect_trial_metrics(trial_id: str, seed: int, has_nfz: bool,
                          dijkstra_result: Dict,
                          aco_results: List[Dict]) -> pd.DataFrame:
    """
    Collect metrics from a single trial.

    Args:
        trial_id: Trial identifier (e.g., "A", "B", "C", "D")
        seed: Random seed used
        has_nfz: Whether NFZs were present
        dijkstra_result: Result from dijkstra.build_dijkstra_tour()
        aco_results: List of results from multiple ACO runs

    Returns:
        DataFrame with metrics for this trial
    """
    metrics = []

    # Dijkstra metrics (single run)
    metrics.append({
        'trial_id': trial_id,
        'algorithm': 'Dijkstra',
        'run_id': 0,
        'seed': seed,
        'has_nfz': has_nfz,
        'distance_m': dijkstra_result['total_distance_m'],
        'num_waypoints': dijkstra_result['num_waypoints'],
        'compute_time_s': dijkstra_result['compute_time_s'],
        'precompute_time_s': 0.0,
        'optimization_time_s': dijkstra_result['compute_time_s'],
        'notes': 'Nearest-neighbor ordering'
    })

    # ACO metrics (multiple runs)
    for i, aco_result in enumerate(aco_results):
        metrics.append({
            'trial_id': trial_id,
            'algorithm': 'ACO',
            'run_id': i + 1,
            'seed': seed,
            'has_nfz': has_nfz,
            'distance_m': aco_result['best_tour_distance'],
            'num_waypoints': len(aco_result['best_tour_path']),
            'compute_time_s': aco_result['compute_time_s'],
            'precompute_time_s': aco_result.get('precompute_time_s', 0.0),
            'optimization_time_s': aco_result.get('optimization_time_s', 0.0),
            'notes': f"Converged at iter {aco_result['convergence_iteration']}"
        })

    return pd.DataFrame(metrics)


def compute_summary_statistics(trial_data: pd.DataFrame) -> Dict:
    """
    Compute summary statistics for a trial.

    Args:
        trial_data: DataFrame from collect_trial_metrics()

    Returns:
        Dictionary with summary statistics
    """
    dijkstra_data = trial_data[trial_data['algorithm'] == 'Dijkstra'].iloc[0]
    aco_data = trial_data[trial_data['algorithm'] == 'ACO']

    dijkstra_dist = dijkstra_data['distance_m']
    dijkstra_time = dijkstra_data['compute_time_s']

    aco_distances = aco_data['distance_m'].values
    aco_times = aco_data['compute_time_s'].values
    aco_precompute_times = aco_data['precompute_time_s'].values
    aco_optimization_times = aco_data['optimization_time_s'].values

    aco_mean_dist = np.mean(aco_distances)
    aco_std_dist = np.std(aco_distances)
    aco_min_dist = np.min(aco_distances)
    aco_max_dist = np.max(aco_distances)

    aco_mean_time = np.mean(aco_times)
    aco_mean_precompute = np.mean(aco_precompute_times)
    aco_mean_optimization = np.mean(aco_optimization_times)

    # Improvement calculation
    improvement_pct = (dijkstra_dist - aco_mean_dist) / dijkstra_dist * 100

    summary = {
        'trial_id': dijkstra_data['trial_id'],
        'seed': dijkstra_data['seed'],
        'has_nfz': dijkstra_data['has_nfz'],
        'dijkstra_distance_m': dijkstra_dist,
        'dijkstra_time_s': dijkstra_time,
        'aco_mean_distance_m': aco_mean_dist,
        'aco_std_distance_m': aco_std_dist,
        'aco_min_distance_m': aco_min_dist,
        'aco_max_distance_m': aco_max_dist,
        'aco_mean_time_s': aco_mean_time,
        'aco_mean_precompute_s': aco_mean_precompute,
        'aco_mean_optimization_s': aco_mean_optimization,
        'improvement_pct': improvement_pct,
        'num_aco_runs': len(aco_distances)
    }

    return summary


def save_trial_metrics(trial_data: pd.DataFrame, output_dir: str):
    """
    Save trial metrics to CSV.

    Args:
        trial_data: DataFrame from collect_trial_metrics()
        output_dir: Directory to save metrics.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'metrics.csv')
    trial_data.to_csv(output_path, index=False)
    print(f"  Saved metrics to {output_path}")


def create_summary_table(all_summaries: List[Dict]) -> pd.DataFrame:
    """
    Create summary table across all trials.

    Args:
        all_summaries: List of summary dicts from compute_summary_statistics()

    Returns:
        DataFrame with aggregated results
    """
    summary_df = pd.DataFrame(all_summaries)

    # Add overall statistics
    overall_row = {
        'trial_id': 'Overall',
        'seed': '-',
        'has_nfz': '-',
        'dijkstra_distance_m': summary_df['dijkstra_distance_m'].mean(),
        'dijkstra_time_s': summary_df['dijkstra_time_s'].mean(),
        'aco_mean_distance_m': summary_df['aco_mean_distance_m'].mean(),
        'aco_std_distance_m': summary_df['aco_std_distance_m'].mean(),
        'aco_min_distance_m': summary_df['aco_min_distance_m'].mean(),
        'aco_max_distance_m': summary_df['aco_max_distance_m'].mean(),
        'aco_mean_time_s': summary_df['aco_mean_time_s'].mean(),
        'aco_mean_precompute_s': summary_df['aco_mean_precompute_s'].mean(),
        'aco_mean_optimization_s': summary_df['aco_mean_optimization_s'].mean(),
        'improvement_pct': summary_df['improvement_pct'].mean(),
        'num_aco_runs': summary_df['num_aco_runs'].sum()
    }

    summary_df = pd.concat([summary_df, pd.DataFrame([overall_row])], ignore_index=True)

    return summary_df


def save_summary_table(summary_df: pd.DataFrame, output_dir: str):
    """
    Save summary table to CSV.

    Args:
        summary_df: DataFrame from create_summary_table()
        output_dir: Directory to save summary_table.csv
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, 'summary_table.csv')
    summary_df.to_csv(output_path, index=False, float_format='%.2f')
    print(f"\nSummary table saved to {output_path}")


def print_summary(summary: Dict):
    """
    Print summary statistics to console.

    Args:
        summary: Dictionary from compute_summary_statistics()
    """
    print(f"\n  Trial {summary['trial_id']} Summary:")
    print(f"    Seed: {summary['seed']}, NFZs: {summary['has_nfz']}")
    print(f"    Dijkstra: {summary['dijkstra_distance_m']:.1f}m in {summary['dijkstra_time_s']*1000:.1f}ms")
    print(f"    ACO:      {summary['aco_mean_distance_m']:.1f} ± {summary['aco_std_distance_m']:.1f}m " +
          f"(min={summary['aco_min_distance_m']:.1f}, max={summary['aco_max_distance_m']:.1f})")
    print(f"    ACO time: {summary['aco_mean_time_s']:.2f}s")
    print(f"    Improvement: {summary['improvement_pct']:.2f}%")


if __name__ == '__main__':
    """Test metrics collection."""
    from grid_gen import generate_grid, DEPOT_POS
    from graph_builder import build_graph
    from dijkstra import build_dijkstra_tour
    from aco import run_aco

    print("Testing metrics collection...\n")

    # Generate test data
    grid, delivery_points, nfz_blocks = generate_grid(seed=42, has_nfz=True)
    G = build_graph(grid)

    # Run algorithms
    dijkstra_result = build_dijkstra_tour(G, DEPOT_POS, delivery_points)
    print(f"Dijkstra: {dijkstra_result['total_distance_m']:.1f}m")

    print("\nRunning 3 ACO trials...")
    aco_results = []
    for i in range(3):
        aco_result = run_aco(G, grid, DEPOT_POS, delivery_points,
                            dijkstra_result['total_distance_m'],
                            verbose=False)
        aco_results.append(aco_result)
        print(f"  ACO run {i+1}: {aco_result['best_tour_distance']:.1f}m")

    # Collect metrics
    trial_data = collect_trial_metrics('TEST', 42, True, dijkstra_result, aco_results)
    print("\nTrial metrics:")
    print(trial_data)

    # Compute summary
    summary = compute_summary_statistics(trial_data)
    print_summary(summary)

    # Save to test directory
    save_trial_metrics(trial_data, 'test_results')

    # Test summary table
    summary_df = create_summary_table([summary])
    print("\nSummary table:")
    print(summary_df)

    save_summary_table(summary_df, 'test_results')
