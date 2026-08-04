"""
CLI Dashboard (cli_dashboard.py)

A beautiful, live Terminal User Interface (TUI) for the GA vs ACO experiments.

Usage:
    python cli_dashboard.py --demo          # Runs a visual demo showing the UI capabilities
    python cli_dashboard.py --run-trial A   # Headless execution with progress bar
"""

import time
import argparse
import random
from pathlib import Path

from rich.console import Console
from rich.layout import Layout
from rich.panel import Panel
from rich.live import Live
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn
from rich.table import Table
from rich.text import Text
import plotext as plt

# Local imports
from grid_gen import generate_grid, DEPOT_POS
from dynamic_nfz import build_default_dynamic_nfzs, render_dynamic_nfzs
from terminal_viz import render_map_to_rich

import multiprocessing as mp
from multiprocessing import Pool
import numpy as np
import pandas as pd
from pathlib import Path

# Import actual experiment logic
from run_experiment import (
    TRIALS, 
    build_trial_environment, 
    _mga_worker, 
    _aco_worker,
    DEFAULT_GA_PARAMS,
    DEFAULT_ACO_PARAMS,
    RANDOM_SEED_OFFSET,
    N_WORKERS,
    _annotate_evaluations,
    _build_dynamic_risk_matrix,
    _objective_matrix,
    _matrix_to_dist_dict,
    _load_protocol_params,
)
from statistical_tests import run_all_tests

console = Console()


def _summarize_records(records):
    finite = [r for r in records if np.isfinite(r.get('distance_m', float('inf')))]
    dists = [r['distance_m'] for r in finite]
    energies = [r['energy_j'] for r in finite if np.isfinite(r.get('energy_j', float('inf')))]
    runtimes = [r['runtime_s'] for r in finite if np.isfinite(r.get('runtime_s', 0.0))]
    return {
        'mean': float(np.mean(dists)) if dists else float('inf'),
        'std': float(np.std(dists, ddof=1)) if len(dists) > 1 else 0.0,
        'best': float(np.min(dists)) if dists else float('inf'),
        'worst': float(np.max(dists)) if dists else float('inf'),
        'valid_runs': len(finite),
        'mean_energy_j': float(np.mean(energies)) if energies else float('inf'),
        'mean_runtime_s': float(np.mean(runtimes)) if runtimes else 0.0,
        'feasible_pct': float(100 * np.mean([r.get('feasible', False) for r in records])) if records else 0.0,
        'errors': sum(1 for r in records if r.get('error')),
    }


def _save_live_trial(output_dir: Path, trial_id: str, cfg: dict, env: dict,
                     mga_results: list, aco_results: list):
    trial_dir = output_dir / f"trial_{trial_id}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    mga_records = [
        {'trial': trial_id, 'run': i, 'algo': 'MGA', 'algorithm': 'MGA', **r}
        for i, r in enumerate(mga_results)
    ]
    aco_records = [
        {'trial': trial_id, 'run': i, 'algo': 'ACO', 'algorithm': 'ACO', **r}
        for i, r in enumerate(aco_results)
    ]
    pd.concat([pd.DataFrame(mga_records), pd.DataFrame(aco_records)]).to_csv(
        trial_dir / "raw_results.csv",
        index=False,
    )
    result = {
        'trial': trial_id,
        'label': cfg['label'],
        'n_runs': len(mga_results),
        'status': 'complete',
        'dijkstra_baseline_m': env['dijkstra_dist'],
        'MGA': _summarize_records(mga_results),
        'ACO': _summarize_records(aco_results),
        'mga_distances': [r['distance_m'] for r in mga_results if np.isfinite(r.get('distance_m', float('inf')))],
        'aco_distances': [r['distance_m'] for r in aco_results if np.isfinite(r.get('distance_m', float('inf')))],
        'model_notes': {},
    }
    if (
        result['MGA']['valid_runs'] != len(mga_results)
        or result['ACO']['valid_runs'] != len(aco_results)
        or result['MGA']['errors']
        or result['ACO']['errors']
    ):
        result['status'] = 'failed'
    with open(trial_dir / "summary.json", "w") as f:
        import json
        json.dump(result, f, indent=2)
    return result


def _master_row(summary: dict):
    m, a = summary['MGA'], summary['ACO']
    return {
        'trial': summary['trial'],
        'label': summary['label'],
        'n_runs': summary['n_runs'],
        'status': summary['status'],
        'dijkstra_m': summary['dijkstra_baseline_m'],
        'mga_mean_m': round(m['mean'], 1),
        'mga_std_m': round(m['std'], 1),
        'mga_best_m': round(m['best'], 1),
        'mga_energy_j': round(m['mean_energy_j'], 0),
        'mga_runtime_s': round(m['mean_runtime_s'], 2),
        'mga_feasible_%': round(m['feasible_pct'], 1),
        'aco_mean_m': round(a['mean'], 1),
        'aco_std_m': round(a['std'], 1),
        'aco_best_m': round(a['best'], 1),
        'aco_energy_j': round(a['mean_energy_j'], 0),
        'aco_runtime_s': round(a['mean_runtime_s'], 2),
        'winner': 'MGA' if m['mean'] < a['mean'] else ('ACO' if a['mean'] < m['mean'] else 'TIE'),
        'pct_diff': round(100 * (a['mean'] - m['mean']) / max(a['mean'], 1e-6), 2),
    }

# ═══════════════════════════════════════════════════════════════════════════
# UI Layout Construction
# ═══════════════════════════════════════════════════════════════════════════

def make_layout() -> Layout:
    layout = Layout(name="root")
    
    layout.split_column(
        Layout(name="header", size=3),
        Layout(name="master_progress", size=3),
        Layout(name="main", ratio=1),
        Layout(name="footer", size=3)
    )
    
    layout["main"].split_row(
        Layout(name="left", ratio=1),
        Layout(name="right", ratio=1)
    )
    
    layout["left"].split_column(
        Layout(name="map", ratio=2),
        Layout(name="telemetry", ratio=1)
    )
    
    layout["right"].split_column(
        Layout(name="progress"),
        Layout(name="stats")
    )
    
    return layout

def generate_header() -> Panel:
    return Panel(
        Text("UAV Path Planning: Enhanced MGA vs ACO (TUI 2.0)", justify="center", style="bold white on blue"),
        style="blue"
    )

def generate_footer() -> Panel:
    return Panel(
        Text("Press Ctrl+C to exit", justify="center", style="dim"),
        style="grey39"
    )

def create_stats_table(mga_dist, mga_eng, aco_dist, aco_eng) -> Table:
    table = Table(show_header=True, header_style="bold magenta", expand=True)
    table.add_column("Algorithm")
    table.add_column("Best Distance", justify="right")
    table.add_column("Total Energy (J)", justify="right")
    
    m_style = "bold green" if mga_dist < aco_dist else "white"
    a_style = "bold green" if aco_dist < mga_dist else "white"
    
    table.add_row("MGA-Enhanced", f"[{m_style}]{mga_dist:,.1f} m[/]", f"[{m_style}]{mga_eng:,.0f}[/]")
    table.add_row("Ant Colony (ACO)", f"[{a_style}]{aco_dist:,.1f} m[/]", f"[{a_style}]{aco_eng:,.0f}[/]")
    
    return Panel(table, title="Live Statistics", border_style="magenta")

# ═══════════════════════════════════════════════════════════════════════════
# Demo Mode
# ═══════════════════════════════════════════════════════════════════════════

def run_demo():
    """Runs a simulated live dashboard to showcase the TUI."""
    console.clear()
    layout = make_layout()
    layout["header"].update(generate_header())
    layout["footer"].update(generate_footer())
    
    # Init progress bars
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
        expand=True
    )
    
    task_mga = progress.add_task("[green]MGA-Enhanced", total=50)
    task_aco = progress.add_task("[red]ACO Baseline", total=50)
    layout["progress"].update(Panel(progress, title="Experiment Progress (Runs)", border_style="blue"))
    
    # Init Map
    grid, dps, _ = generate_grid(seed=42, num_deliveries=20, nfz_count=10)
    dyn_nfzs = build_default_dynamic_nfzs(seed=42, n_linear=5, n_oscillating=5)
    
    mga_dist, aco_dist = 120000.0, 120000.0
    mock_best_path = [0] + list(range(1, len(dps) + 1))
    
    # Telemetry
    telemetry_logs = []
    def add_log(msg, color="white"):
        telemetry_logs.append(f"[{color}]{time.strftime('%H:%M:%S')} - {msg}[/]")
        if len(telemetry_logs) > 6:
            telemetry_logs.pop(0)
    
    add_log("Started TUI 2.0 Demo Mode", "bold cyan")
    
    with Live(layout, refresh_per_second=10, screen=True):
        for tick in range(500):
            # Update Map every 5 ticks
            if tick % 5 == 0:
                current_grid = render_dynamic_nfzs(grid, dyn_nfzs, t=tick//5)
                # Shuffle the path slightly to make it look alive
                if random.random() > 0.8:
                    idx1, idx2 = random.randint(1, len(dps)), random.randint(1, len(dps))
                    mock_best_path[idx1], mock_best_path[idx2] = mock_best_path[idx2], mock_best_path[idx1]
                
                map_render = render_map_to_rich(current_grid, dps, DEPOT_POS, factor=4, best_path=mock_best_path)
                legend = "[blue]▲ Depot[/]  [yellow]● Delivery[/]  [red]██ NFZ[/]  [green]:: Path[/]  [bright_black]· Clear[/]"
                layout["map"].update(Panel(map_render, title=f"Live 2D Map (t={tick//5})", subtitle=legend, border_style="green"))
            
            # Simulate progress
            if tick % 10 == 0 and not progress.finished:
                if random.random() > 0.3:
                    progress.advance(task_mga, 1)
                    mga_dist -= random.uniform(500, 1500)
                    if random.random() > 0.8:
                        add_log("MGA: Or-opt found local improvements.", "bold yellow")
                if random.random() > 0.4:
                    progress.advance(task_aco, 1)
                    aco_dist -= random.uniform(300, 1000)
                    
                mga_eng = mga_dist * 1.5
                aco_eng = aco_dist * 1.5
                layout["stats"].update(create_stats_table(mga_dist, mga_eng, aco_dist, aco_eng))
                
            log_text = "\n".join(telemetry_logs)
            layout["telemetry"].update(Panel(Text.from_markup(log_text), title="Core Telemetry Feed", border_style="yellow"))
                
            time.sleep(0.08) # Slowed down slightly for readability
            
            if progress.finished:
                break
                
    console.print("\n")
    summary_text = (
        f"[bold]MGA-Enhanced:[/] {mga_dist:,.1f} m  |  {mga_eng:,.0f} J\n"
        f"[bold]ACO Baseline:[/] {aco_dist:,.1f} m  |  {aco_eng:,.0f} J\n\n"
        f"🏆 [bold green]MGA-Enhanced won by {(aco_dist - mga_dist):,.1f} meters![/]"
    )
    console.print(Panel(summary_text, title="🏁 FINAL RESULTS SUMMARY 🏁", border_style="gold1", expand=False))
    console.print("[dim]Generating Pareto Front Scatter Plot...[/]")
    
    # Plotext Demo Chart
    print("\n")
    plt.title("Pareto Front (Simulated NSGA-II)")
    mga_x = [random.gauss(90000, 2000) for _ in range(50)]
    mga_y = [x * 1.5 + random.gauss(0, 10000) for x in mga_x]
    aco_x = [random.gauss(100000, 3000) for _ in range(50)]
    aco_y = [x * 1.5 + random.gauss(0, 12000) for x in aco_x]
    
    plt.scatter(mga_x, mga_y, label="MGA-Enhanced", color="green", marker="dot")
    plt.scatter(aco_x, aco_y, label="ACO", color="red", marker="dot")
    plt.xlabel("Distance (m)")
    plt.ylabel("Energy (J)")
    plt.show()

# ═══════════════════════════════════════════════════════════════════════════
# Live Experiment Integration
# ═══════════════════════════════════════════════════════════════════════════

def run_live_trial(trial_id: str, n_runs: int = 50, n_workers: int = N_WORKERS,
                   output_dir: Path = None, external_live=None, layout=None,
                   master_task=None, master_progress=None,
                   ga_params=None, aco_params=None):
    """Executes the real experiment but visualizes it live in the TUI."""
    if trial_id not in TRIALS:
        console.print(f"[bold red]Trial {trial_id} not found.[/]")
        return
        
    cfg = TRIALS[trial_id]
    
    # Setup environment
    env = build_trial_environment(trial_id, cfg)
    dps = env['dps']
    seeds = [RANDOM_SEED_OFFSET + r for r in range(n_runs)]
    ga_params = ga_params or DEFAULT_GA_PARAMS
    aco_params = aco_params or DEFAULT_ACO_PARAMS
    risk_matrix = _build_dynamic_risk_matrix(env) if cfg.get('dynamic') else None
    risk_weight = float(ga_params.get('dynamic_risk_weight', 0.0))
    obj_matrix = _objective_matrix(env['dist_matrix'], risk_matrix, risk_weight)
    obj_dist_dict = _matrix_to_dist_dict([DEPOT_POS] + list(dps), obj_matrix)

    mga_args = [
        (env['dist_matrix'], obj_matrix, env['dist_dict'], dps, ga_params, s, risk_matrix)
        for s in seeds
    ]
    aco_args = [
        (dps, env['dist_dict'], obj_dist_dict, env['dijkstra_dist'], aco_params, s, risk_matrix)
        for s in seeds
    ]
    
    is_standalone = external_live is None
    
    if is_standalone:
        console.clear()
        layout = make_layout()
        layout["header"].update(Panel(Text(f"LIVE EXPERIMENT: Trial {trial_id} ({cfg['label']})", justify="center", style="bold white on red"), style="red"))
        layout["footer"].update(generate_footer())
        layout["master_progress"].visible = False
    else:
        layout["header"].update(Panel(Text(f"LIVE EXPERIMENT: Trial {trial_id} ({cfg['label']})", justify="center", style="bold white on red"), style="red"))
    
    progress = Progress(
        TextColumn("[bold blue]{task.description}", justify="right"),
        BarColumn(bar_width=None),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
        expand=True
    )
    
    task_mga = progress.add_task("[green]MGA-Enhanced Runs", total=n_runs)
    task_aco = progress.add_task("[red]ACO Baseline Runs", total=n_runs)
    layout["progress"].update(Panel(progress, title="Real-Time Execution", border_style="blue"))
    
    mga_dist_best = float('inf')
    mga_eng_best = float('inf')
    aco_dist_best = float('inf')
    aco_eng_best = float('inf')
    best_path_to_draw = None
    
    # Telemetry Log
    telemetry_logs = []
    def add_log(msg, color="white"):
        telemetry_logs.append(f"[{color}]{time.strftime('%H:%M:%S')} - {msg}[/]")
        if len(telemetry_logs) > 6:
            telemetry_logs.pop(0)
    
    add_log(f"Started Trial {trial_id} on {n_workers} cores", "bold cyan")
    
    pool = Pool(n_workers)
    mga_results = []
    aco_results = []
    
    def log_mga(r): 
        mga_results.append(r)
        if r['diversity_injections'] > 0:
            add_log(f"MGA: Stagnation broken! Injected diversity.", "bold magenta")
        elif r['or_opt_improvements'] > 0:
            add_log(f"MGA: Or-opt found {r['or_opt_improvements']} local improvements.", "bold yellow")
            
    def log_aco(r): 
        aco_results.append(r)
        if r['convergence_gen'] > 300:
            add_log(f"ACO: Converged late at iteration {r['convergence_gen']}", "dim")
    
    for args in mga_args: pool.apply_async(_mga_worker, (args,), callback=log_mga)
    for args in aco_args: pool.apply_async(_aco_worker, (args,), callback=log_aco)
    pool.close()
    
    dyn_nfzs = env['dyn_nfzs']
    
    def ui_loop():
        nonlocal mga_dist_best, mga_eng_best, aco_dist_best, aco_eng_best, best_path_to_draw
        tick = 0
        while len(mga_results) < n_runs or len(aco_results) < n_runs:
            
            # Sync Progress
            progress.update(task_mga, completed=len(mga_results))
            progress.update(task_aco, completed=len(aco_results))
            if master_task and master_progress:
                master_progress.update(master_task, description=f"Trial {trial_id} ({len(mga_results)+len(aco_results)}/{n_runs*2} runs)")
            
            # Update Best Stats & Best Path
            valid_mga = [r for r in mga_results if r['distance_m'] != float('inf')]
            valid_aco = [r for r in aco_results if r['distance_m'] != float('inf')]
            
            if valid_mga:
                best = min(valid_mga, key=lambda x: x['distance_m'])
                if best['distance_m'] < mga_dist_best:
                    mga_dist_best = best['distance_m']
                    mga_eng_best = best['energy_j']
                    if mga_dist_best <= aco_dist_best:
                        best_path_to_draw = best['best_path']
                        add_log(f"MGA: New Global Best! Distance {mga_dist_best:,.0f} m", "bold green")
                        
            if valid_aco:
                best = min(valid_aco, key=lambda x: x['distance_m'])
                if best['distance_m'] < aco_dist_best:
                    aco_dist_best = best['distance_m']
                    aco_eng_best = best['energy_j']
                    if aco_dist_best < mga_dist_best:
                        best_path_to_draw = best['best_path']
                        add_log(f"ACO: New Global Best! Distance {aco_dist_best:,.0f} m", "bold red")
                
            layout["stats"].update(create_stats_table(mga_dist_best, mga_eng_best, aco_dist_best, aco_eng_best))
            
            # Update Map every 5 ticks if dynamic, or just redraw with new best_path
            if (dyn_nfzs and tick % 5 == 0) or tick % 10 == 0:
                grid_to_render = render_dynamic_nfzs(env['grid'], dyn_nfzs, t=tick//5) if dyn_nfzs else env['grid']
                map_render = render_map_to_rich(grid_to_render, dps, DEPOT_POS, factor=4, best_path=best_path_to_draw)
                title = f"Live 2D Map (t={tick//5})" if dyn_nfzs else "Static 2D Map"
                legend = "[blue]▲ Depot[/]  [yellow]● Delivery[/]  [red]██ NFZ[/]  [green]:: Path[/]  [bright_black]· Clear[/]"
                layout["map"].update(Panel(map_render, title=title, subtitle=legend, border_style="green"))
                
            # Update Telemetry
            log_text = "\n".join(telemetry_logs)
            layout["telemetry"].update(Panel(Text.from_markup(log_text), title="Core Telemetry Feed", border_style="yellow"))
            
            tick += 1
            time.sleep(0.1)
            
    if is_standalone:
        with Live(layout, refresh_per_second=10, screen=True):
            ui_loop()
    else:
        ui_loop()
            
    pool.join()
    _annotate_evaluations(mga_results, env)
    _annotate_evaluations(aco_results, env)

    valid_mga = [r for r in mga_results if np.isfinite(r.get('distance_m', float('inf')))]
    valid_aco = [r for r in aco_results if np.isfinite(r.get('distance_m', float('inf')))]
    if valid_mga:
        best = min(valid_mga, key=lambda x: x['distance_m'])
        mga_dist_best = best['distance_m']
        mga_eng_best = best['energy_j']
        best_path_to_draw = best.get('best_path', best_path_to_draw)
    if valid_aco:
        best = min(valid_aco, key=lambda x: x['distance_m'])
        aco_dist_best = best['distance_m']
        aco_eng_best = best['energy_j']
        if aco_dist_best < mga_dist_best:
            best_path_to_draw = best.get('best_path', best_path_to_draw)
    
    if is_standalone:
        console.print("\n")
    
    if mga_dist_best < aco_dist_best:
        winner_txt = f"🏆 [bold green]MGA-Enhanced won by {(aco_dist_best - mga_dist_best):,.1f} meters![/]"
    else:
        winner_txt = f"🏆 [bold red]ACO won by {(mga_dist_best - aco_dist_best):,.1f} meters![/]"

    summary_text = (
        f"Trial {trial_id} completed across {n_workers} CPU cores.\n\n"
        f"[bold]MGA-Enhanced Best:[/] {mga_dist_best:,.1f} m  |  {mga_eng_best:,.0f} J\n"
        f"[bold]ACO Baseline Best:[/] {aco_dist_best:,.1f} m  |  {aco_eng_best:,.0f} J\n\n"
        f"{winner_txt}"
    )
    
    if is_standalone:
        console.print(Panel(summary_text, title="🏁 FINAL RESULTS SUMMARY 🏁", border_style="gold1", expand=False))
        console.print("[dim]Generating Pareto Front Scatter Plot...[/]")
        
        # Pareto Plot
        print("\n")
        plt.title(f"Trial {trial_id} Final Results (Distance vs Energy)")
        mga_x = [r['distance_m'] for r in mga_results if r['distance_m'] != float('inf')]
        mga_y = [r['energy_j'] for r in mga_results if r['distance_m'] != float('inf')]
        aco_x = [r['distance_m'] for r in aco_results if r['distance_m'] != float('inf')]
        aco_y = [r['energy_j'] for r in aco_results if r['distance_m'] != float('inf')]
        
        if mga_x and aco_x:
            plt.scatter(mga_x, mga_y, label="MGA-Enhanced", color="green", marker="dot")
            plt.scatter(aco_x, aco_y, label="ACO", color="red", marker="dot")
            plt.xlabel("Distance (m)")
            plt.ylabel("Energy (J)")
            plt.show()
    summary = None
    if output_dir is not None:
        summary = _save_live_trial(Path(output_dir), trial_id, cfg, env, mga_results, aco_results)
        if is_standalone:
            console.print(f"[green]Saved live trial data to {Path(output_dir) / f'trial_{trial_id}'}[/]")
    return mga_results, aco_results, summary

def run_all_live_trials(n_runs: int = 50, n_workers: int = N_WORKERS,
                        output_dir: Path = Path("results"),
                        ga_params=None,
                        aco_params=None):
    console.clear()
    layout = make_layout()
    layout["footer"].update(generate_footer())
    
    master_progress = Progress(
        TextColumn("[bold magenta]OVERALL PROGRESS", justify="right"),
        BarColumn(bar_width=None, style="magenta"),
        "[progress.percentage]{task.percentage:>3.1f}%",
        TimeRemainingColumn(),
        expand=True
    )
    master_task = master_progress.add_task("Running...", total=len(TRIALS))
    layout["master_progress"].update(Panel(master_progress, title="Global Master Tracker", border_style="magenta"))
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    master_rows = []
    failed = []
    with Live(layout, refresh_per_second=10, screen=True) as live:
        for tid in TRIALS.keys():
            _m_res, _a_res, summary = run_live_trial(
                tid, n_runs=n_runs, n_workers=n_workers, output_dir=output_dir,
                external_live=live, layout=layout, master_task=master_task,
                master_progress=master_progress,
                ga_params=ga_params,
                aco_params=aco_params,
            )
            master_rows.append(_master_row(summary))
            if summary['status'] != 'complete':
                failed.append(tid)
            master_progress.advance(master_task, 1)
            
    pd.DataFrame(master_rows).to_csv(output_dir / "master_results.csv", index=False)
    if master_rows:
        mga_all = {}
        aco_all = {}
        for tid in TRIALS.keys():
            raw = output_dir / f"trial_{tid}" / "raw_results.csv"
            if raw.exists():
                df = pd.read_csv(raw)
                mga_all[tid] = df[df['algo'] == 'MGA']['distance_m'].replace([np.inf, -np.inf], np.nan).dropna().tolist()
                aco_all[tid] = df[df['algo'] == 'ACO']['distance_m'].replace([np.inf, -np.inf], np.nan).dropna().tolist()
        if len(mga_all) >= 2:
            run_all_tests(mga_all, aco_all, output_dir)
    if failed:
        console.print(f"\n[bold red]Live run finished with failed trials: {failed}. See per-trial raw_results.csv files.[/]")
    else:
        console.print(f"\n[bold green]Experiment Complete! Saved summaries to {output_dir / 'master_results.csv'}[/]")

# ═══════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="UAV Path Planning TUI Dashboard")
    parser.add_argument("--demo", action="store_true", help="Run the visual TUI demo")
    parser.add_argument("--run-trial", type=str, help="Run a specific trial headlessly (calls run_experiment.py)")
    parser.add_argument("--run-all", action="store_true", help="Run all paper trials sequentially in the live TUI")
    parser.add_argument("--runs", type=int, default=50, help="Runs per algorithm per trial")
    parser.add_argument("--workers", type=int, default=N_WORKERS, help="Number of CPU cores to use (default: max-1)")
    parser.add_argument("--output", default="results", help="Directory for live-run result files")
    parser.add_argument("--params", default=None, help="Parameter JSON, normally config/paper_params.json")
    
    args = parser.parse_args()
    
    if args.demo:
        run_demo()
    elif args.run_all:
        ga_params, aco_params, _source = _load_protocol_params(args.params, Path(args.output))
        console.print(f"[bold yellow]Starting Master Run (All Trials) using {args.workers} workers...[/]")
        run_all_live_trials(n_runs=args.runs, n_workers=args.workers,
                            output_dir=Path(args.output),
                            ga_params=ga_params, aco_params=aco_params)
    elif args.run_trial:
        ga_params, aco_params, _source = _load_protocol_params(args.params, Path(args.output))
        console.print(f"[bold yellow]Starting Live Run for Trial {args.run_trial} using {args.workers} workers...[/]")
        _m, _a, summary = run_live_trial(
            args.run_trial,
            n_runs=args.runs,
            n_workers=args.workers,
            output_dir=Path(args.output),
            ga_params=ga_params,
            aco_params=aco_params,
        )
        if summary is not None:
            pd.DataFrame([_master_row(summary)]).to_csv(Path(args.output) / "master_results.csv", index=False)
            console.print(f"[green]Saved single-trial summary to {Path(args.output) / 'master_results.csv'}[/]")
    else:
        console.print("[bold red]Please specify --demo, --run-trial <ID>, or --run-all. See --help for options.[/]")
