"""
Generate Publication Figures — GA vs ACO Paper
===============================================
Run this script AFTER the full experiment completes:

    python generate_figures.py --results-dir results

All figures are saved to the selected figures directory as PDF + PNG (300 dpi).

Figures produced:
  1. bar_distance_comparison    — Grouped bar: MGA vs ACO per trial + Dijkstra baseline
  2. boxplot_distance           — Box+strip plot: distribution across 50 runs
  3. convergence_trial_A        — Convergence curve: Trial A (2D Baseline)
  4. convergence_trial_E        — Convergence curve: Trial E (Dynamic NFZ)
  5. convergence_trial_F        — Convergence curve: Trial F (3D + Dynamic)
  6. radar_comparison           — Radar: multi-metric profile (smaller area = better)
  7. statistical_heatmap        — Heatmap: p-value, % improvement, Cohen's d per trial
  8. scalability_chart          — Line: runtime & distance vs problem size (20 vs 50 pts)
  9. improvement_summary        — Horizontal bar: % improvement per trial (annotated)
"""

import json
import argparse
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from pathlib import Path
from math import pi

warnings.filterwarnings('ignore', category=RuntimeWarning)

# ── Style ────────────────────────────────────────────────────────────────────
sns.set_theme(style='whitegrid', context='paper', font_scale=1.4)
plt.rcParams.update({
    'font.family':      'serif',
    'axes.titlesize':   15,
    'axes.labelsize':   13,
    'figure.dpi':       150,
    'savefig.dpi':      300,
    'savefig.bbox':     'tight',
})

MGA_COLOR  = '#27ae60'   # emerald green
ACO_COLOR  = '#e74c3c'   # alizarin red
DIJ_COLOR  = '#95a5a6'   # silver
GRID_ALPHA = 0.3

TRIAL_LABELS = {
    'A': 'A\n2D Baseline\n(seed 42)',
    'B': 'B\n2D Baseline\n(seed 7)',
    'C': 'C\n2D Baseline\n(seed 99)',
    'D': 'D\n2D Control\n(no NFZ)',
    'E': 'E\n2D Dynamic\n(seed 42)',
    'F': 'F\n2D Heavy Dyn\n(seed 7)',
    'G': 'G\n3D Static\n(seed 42)',
    'H': 'H\n3D Dynamic\n(seed 7)',
    'I': 'I\n3D Scale 50\n(seed 42)',
    'J': 'J\n3D Dyn 50\n(seed 99)',
}

RES   = Path('results')
FIGS  = RES / 'figures'


def _save(name: str):
    FIGS.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGS / f'{name}.pdf')
    plt.savefig(FIGS / f'{name}.png')
    plt.close()
    print(f'  [OK] {name}.pdf / .png')


# ═══════════════════════════════════════════════════════════════════════════
# 1. Grouped Bar Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig_grouped_bar(master: pd.DataFrame):
    df = master.copy()
    n  = len(df)
    x  = np.arange(n)
    w  = 0.30

    fig, ax = plt.subplots(figsize=(13, 6))

    ax.bar(x - w/2, df['mga_mean_m'] / 1000, w,
           color=MGA_COLOR, label='MGA-2opt', zorder=3)
    ax.bar(x + w/2, df['aco_mean_m'] / 1000, w,
           color=ACO_COLOR, label='ACO', zorder=3)
    ax.errorbar(x - w/2, df['mga_mean_m'] / 1000,
                yerr=df['mga_std_m'] / 1000,
                fmt='none', ecolor='black', capsize=4, linewidth=1.2, zorder=4)
    ax.errorbar(x + w/2, df['aco_mean_m'] / 1000,
                yerr=df['aco_std_m'] / 1000,
                fmt='none', ecolor='black', capsize=4, linewidth=1.2, zorder=4)
    ax.plot(x, df['dijkstra_m'] / 1000,
            color=DIJ_COLOR, linestyle='--', marker='D',
            linewidth=1.5, markersize=6, label='Dijkstra Baseline', zorder=5)

    ax.set_ylabel('Tour Distance (km)')
    ax.set_title('Average Tour Distance: MGA-2opt vs ACO across 8 Experimental Trials\n'
                 '(50 independent runs per trial; error bars = ±1 SD)')
    ax.set_xticks(x)
    ax.set_xticklabels([TRIAL_LABELS[t] for t in df['trial']], fontsize=10)
    ax.yaxis.grid(True, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.9)

    # annotate improvement %
    for i, row in df.iterrows():
        pct = row['pct_diff']
        ax.annotate(f'+{pct:.1f}%',
                    xy=(i, max(row['mga_mean_m'], row['aco_mean_m']) / 1000 + 1),
                    ha='center', va='bottom', fontsize=8.5, color='#2c3e50',
                    fontweight='bold')

    plt.tight_layout()
    _save('bar_distance_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# 2. Box + Strip Plot (handles zero-variance gracefully)
# ═══════════════════════════════════════════════════════════════════════════

def fig_boxplot(raw_dir: Path):
    frames = []
    for trial_dir in sorted(raw_dir.glob('trial_*')):
        tid = trial_dir.name.split('_')[1]
        csv = trial_dir / 'raw_results.csv'
        if csv.exists():
            df = pd.read_csv(csv)
            df['trial'] = tid
            frames.append(df)

    if not frames:
        print('  [SKIP] No raw data for box plot')
        return

    data = pd.concat(frames, ignore_index=True)
    data = data[data['distance_m'] < 1e10]
    data['distance_km'] = data['distance_m'] / 1000
    trial_order = sorted(data['trial'].unique())

    fig, ax = plt.subplots(figsize=(13, 6))

    palette = {'MGA': MGA_COLOR, 'ACO': ACO_COLOR}

    sns.boxplot(data=data, x='trial', y='distance_km', hue='algo',
                order=trial_order, hue_order=['MGA', 'ACO'],
                palette=palette, width=0.5, linewidth=1.2,
                flierprops={'marker': 'o', 'markersize': 3, 'alpha': 0.4},
                ax=ax)

    # Strip for individual points (jitter) — shows zero-variance clearly
    sns.stripplot(data=data, x='trial', y='distance_km', hue='algo',
                  order=trial_order, hue_order=['MGA', 'ACO'],
                  palette=palette, dodge=True, jitter=True,
                  size=2.5, alpha=0.35, ax=ax, legend=False)

    ax.set_ylabel('Tour Distance (km)')
    ax.set_xlabel('Trial')
    ax.set_title('Distribution of Tour Distances across 50 Runs\n'
                 '(MGA zero-variance = deterministic convergence)')
    ax.yaxis.grid(True, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles[:2], ['MGA-2opt', 'ACO'], framealpha=0.9)
    plt.tight_layout()
    _save('boxplot_distance')


# ═══════════════════════════════════════════════════════════════════════════
# 3. Convergence Curves (from JSON column saved by workers)
# ═══════════════════════════════════════════════════════════════════════════

def fig_convergence(raw_dir: Path, trial_id: str):
    csv = raw_dir / f'trial_{trial_id}' / 'raw_results.csv'
    if not csv.exists():
        print(f'  [SKIP] No data for convergence Trial {trial_id}')
        return

    df = pd.read_csv(csv)
    if 'convergence_curve' not in df.columns:
        print(f'  [SKIP] No convergence_curve column for Trial {trial_id}')
        return

    mga_curves, aco_curves = [], []
    for _, row in df.iterrows():
        try:
            curve = json.loads(row['convergence_curve'])
            if not curve:
                continue
            if row['algo'] == 'MGA':
                mga_curves.append([float(v) for v in curve])
            elif row['algo'] == 'ACO':
                aco_curves.append([float(v) for v in curve])
        except Exception:
            continue

    def pad_mean_std(curves):
        if not curves:
            return np.array([]), np.array([])
        max_len = max(len(c) for c in curves)
        padded  = np.array([c + [c[-1]] * (max_len - len(c)) for c in curves])
        return np.mean(padded, axis=0) / 1000, np.std(padded, axis=0) / 1000

    mga_m, mga_s = pad_mean_std(mga_curves)
    aco_m, aco_s = pad_mean_std(aco_curves)

    fig, ax = plt.subplots(figsize=(10, 6))
    if len(mga_m):
        ax.plot(mga_m, color=MGA_COLOR, linewidth=2, label='MGA-2opt')
        ax.fill_between(range(len(mga_m)), mga_m - mga_s, mga_m + mga_s,
                        color=MGA_COLOR, alpha=0.15)
    if len(aco_m):
        ax.plot(aco_m, color=ACO_COLOR, linewidth=2, label='ACO')
        ax.fill_between(range(len(aco_m)), aco_m - aco_s, aco_m + aco_s,
                        color=ACO_COLOR, alpha=0.15)

    ax.set_xlabel('Generation / Iteration')
    ax.set_ylabel('Best Tour Distance (km)')
    ax.set_title(f'Convergence Profile — Trial {trial_id}\n'
                 f'(mean ± SD across 50 runs; shaded band = variability)')
    ax.yaxis.grid(True, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)
    ax.legend(framealpha=0.9)
    plt.tight_layout()
    _save(f'convergence_trial_{trial_id}')


# ═══════════════════════════════════════════════════════════════════════════
# 4. Radar / Spider Chart
# ═══════════════════════════════════════════════════════════════════════════

def fig_radar(master: pd.DataFrame):
    df = master.copy()
    categories = ['Distance\n(shorter=better)', 'Best Distance',
                  'Energy\n(lower=better)', 'Consistency\n(lower std=better)',
                  'Speed\n(faster=better)']
    N = len(categories)
    angles = [n / N * 2 * pi for n in range(N)]
    angles += angles[:1]

    def scores(algo_prefix, other_prefix):
        """Return normalised scores 0..1 where 1.0 = worst of the pair."""
        cols = {
            'dist':  f'{algo_prefix}_mean_m',
            'best':  f'{algo_prefix}_best_m',
            'eng':   f'{algo_prefix}_energy_j',
            'std':   f'{algo_prefix}_std_m',
            'time':  f'{algo_prefix}_runtime_s',
        }
        ocols = {k: v.replace(algo_prefix, other_prefix) for k, v in cols.items()}
        v, o = [df[cols[k]].mean() for k in cols], [df[ocols[k]].mean() for k in ocols]
        result = []
        for vi, oi in zip(v, o):
            mx = max(vi, oi, 1e-9)
            result.append(vi / mx)
        return result

    mga_vals = scores('mga', 'aco') + [scores('mga', 'aco')[0]]
    aco_vals = scores('aco', 'mga') + [scores('aco', 'mga')[0]]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={'polar': True})
    ax.set_theta_offset(pi / 2)
    ax.set_theta_direction(-1)
    ax.set_thetagrids(np.degrees(angles[:-1]), categories, fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(['0.25', '0.5', '0.75', '1.0\n(worst)'], fontsize=8)

    ax.plot(angles, mga_vals, color=MGA_COLOR, linewidth=2, label='MGA-2opt')
    ax.fill(angles, mga_vals, color=MGA_COLOR, alpha=0.15)
    ax.plot(angles, aco_vals, color=ACO_COLOR, linewidth=2, label='ACO')
    ax.fill(angles, aco_vals, color=ACO_COLOR, alpha=0.15)

    ax.set_title('Algorithm Profile\n(smaller enclosed area = better overall performance)',
                 pad=25, fontsize=13)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.15), framealpha=0.9)
    plt.tight_layout()
    _save('radar_comparison')


# ═══════════════════════════════════════════════════════════════════════════
# 5. Statistical Heatmap
# ═══════════════════════════════════════════════════════════════════════════

def fig_stat_heatmap(stats_csv: Path):
    if not stats_csv.exists():
        print('  [SKIP] statistical_summary.csv not found')
        return
    df  = pd.read_csv(stats_csv).set_index('trial')

    # Cap infinite Cohen's d at ±15 for colour scaling, mark with asterisk
    def cap_cohens(val):
        if np.isinf(val):
            return -15.0 if val < 0 else 15.0
        return val

    plot_data = pd.DataFrame({
        'Improvement (%)': df['pct_improvement'],
        'p-value (Wilcoxon)': df['p_wilcoxon'],
        "Cohen's d": df['cohens_d'].apply(cap_cohens),
    })

    fig, ax = plt.subplots(figsize=(9, 6))
    im = sns.heatmap(plot_data, annot=True, fmt='.3f', cmap='RdYlGn',
                     linewidths=0.5, linecolor='white',
                     cbar_kws={'shrink': 0.8}, ax=ax)

    # Mark ±inf cells with a note
    for i, (tid, row) in enumerate(df.iterrows()):
        if np.isinf(row['cohens_d']):
            ax.text(2.5, i + 0.5, '∞', ha='center', va='center',
                    fontsize=13, color='white', fontweight='bold')

    ax.set_title("Statistical Results per Trial\n"
                 "(green = favourable; Cohen's d capped at ±15 for display)")
    ax.set_xlabel('')
    plt.tight_layout()
    _save('statistical_heatmap')


# ═══════════════════════════════════════════════════════════════════════════
# 6. Scalability Chart (20 pts → 50 pts in 3D)
# ═══════════════════════════════════════════════════════════════════════════

def fig_scalability(master: pd.DataFrame):
    # Trials G (3D 20pts) and I (3D 50pts), H (3D+Dyn 20pts) and J (3D+Dyn 50pts)
    pairs = [('G', 'I', '3D Static'), ('H', 'J', '3D + Dynamic')]
    n_pts = [20, 50]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax, (t20, t50, label) in zip([ax1, ax2], pairs):
        try:
            row20 = master[master['trial'] == t20].iloc[0]
            row50 = master[master['trial'] == t50].iloc[0]
        except IndexError:
            continue

        mga_d = [row20['mga_mean_m'] / 1000, row50['mga_mean_m'] / 1000]
        aco_d = [row20['aco_mean_m'] / 1000, row50['aco_mean_m'] / 1000]
        mga_t = [row20['mga_runtime_s'], row50['mga_runtime_s']]
        aco_t = [row20['aco_runtime_s'], row50['aco_runtime_s']]

        ax.plot(n_pts, mga_d, 'o-', color=MGA_COLOR, lw=2, ms=9, label='MGA dist (km)')
        ax.plot(n_pts, aco_d, 's-', color=ACO_COLOR, lw=2, ms=9, label='ACO dist (km)')
        ax2b = ax.twinx()
        ax2b.plot(n_pts, mga_t, 'o--', color=MGA_COLOR, lw=1.5, ms=7, alpha=0.5, label='MGA time (s)')
        ax2b.plot(n_pts, aco_t, 's--', color=ACO_COLOR, lw=1.5, ms=7, alpha=0.5, label='ACO time (s)')
        ax2b.set_ylabel('Runtime (s)', color='gray')
        ax2b.tick_params(axis='y', labelcolor='gray')

        ax.set_title(f'Scalability — {label}')
        ax.set_xlabel('Number of Delivery Points')
        ax.set_ylabel('Tour Distance (km)')
        ax.set_xticks(n_pts)
        ax.yaxis.grid(True, alpha=GRID_ALPHA)
        ax.set_axisbelow(True)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2b.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9, framealpha=0.9)

    fig.suptitle('Scalability: Distance and Runtime vs Problem Size\n'
                 '(solid lines = distance, dashed = runtime)', y=1.02)
    plt.tight_layout()
    _save('scalability_chart')


# ═══════════════════════════════════════════════════════════════════════════
# 7. Improvement Summary (Horizontal Bar)
# ═══════════════════════════════════════════════════════════════════════════

def fig_improvement_summary(master: pd.DataFrame):
    df = master.sort_values('pct_diff')

    fig, ax = plt.subplots(figsize=(9, 6))
    colors = [MGA_COLOR if v > 0 else ACO_COLOR for v in df['pct_diff']]
    bars = ax.barh(df['trial'], df['pct_diff'], color=colors, edgecolor='white', height=0.55)

    for bar, val in zip(bars, df['pct_diff']):
        ax.text(val + 0.1, bar.get_y() + bar.get_height() / 2,
                f'+{val:.1f}%', va='center', fontsize=11, fontweight='bold',
                color='#2c3e50')

    ax.axvline(0, color='black', linewidth=1)
    ax.set_xlabel('MGA Improvement over ACO (%)')
    ax.set_ylabel('Trial')
    n_trials = len(df)
    n_mga = int((df['winner'] == 'MGA').sum()) if 'winner' in df else int((df['pct_diff'] > 0).sum())
    ax.set_title('MGA-2opt Performance Advantage over ACO\n'
                 f'({n_mga}/{n_trials} trials favour MGA; see statistical table for significance)')
    ax.xaxis.grid(True, alpha=GRID_ALPHA)
    ax.set_axisbelow(True)

    mga_patch = mpatches.Patch(color=MGA_COLOR, label='MGA wins')
    ax.legend(handles=[mga_patch], framealpha=0.9)
    plt.tight_layout()
    _save('improvement_summary')


# ═══════════════════════════════════════════════════════════════════════════
# 8. Environment Map (Trial A — 2D grid with NFZs and delivery points)
# ═══════════════════════════════════════════════════════════════════════════

def fig_environment_map():
    """Generate the 2D environment map by rebuilding Trial A's grid."""
    try:
        import sys
        sys.path.insert(0, str(Path(__file__).parent))
        from grid_gen import generate_grid, DEPOT_POS, CELL_NFZ, CELL_DELIVERY, CELL_DEPOT
        import json as _json

        grid, dps, nfzs = generate_grid(seed=42, num_deliveries=20, nfz_count=18)

        # Try to overlay the best MGA path
        best_path = None
        raw_csv = RES / 'trial_A' / 'raw_results.csv'
        if raw_csv.exists():
            df = pd.read_csv(raw_csv)
            mga_row = df[df['algo'] == 'MGA'].sort_values('distance_m').iloc[0]
            try:
                path_indices = _json.loads(mga_row['best_path']) if isinstance(mga_row['best_path'], str) else mga_row['best_path']
                best_path = path_indices
            except Exception:
                pass

        rows, cols = grid.shape
        rgb = np.ones((rows, cols, 3))

        nfz_mask = grid == CELL_NFZ
        rgb[nfz_mask] = [0.8, 0.2, 0.2]          # red: NFZ

        fig, ax = plt.subplots(figsize=(9, 9))
        ax.imshow(rgb, origin='upper', interpolation='nearest')

        # Delivery points
        dp_rows = [p[0] for p in dps]
        dp_cols = [p[1] for p in dps]
        ax.scatter(dp_cols, dp_rows, c='#f39c12', s=60, zorder=5,
                   marker='o', edgecolors='white', linewidths=0.8, label='Delivery Point')

        # Depot
        ax.scatter(DEPOT_POS[1], DEPOT_POS[0], c='#2980b9', s=180, zorder=6,
                   marker='^', edgecolors='white', linewidths=1.2, label='Depot')

        ax.set_title('Trial A — 2D Environment (200×200 grid, 18 NFZs, 20 Delivery Points)\n'
                     'Red = No-Fly Zone | Orange = Delivery Point | Blue = Depot')
        ax.legend(loc='lower right', framealpha=0.9, fontsize=10)
        ax.set_xlabel('Column')
        ax.set_ylabel('Row')
        ax.xaxis.grid(False)
        ax.yaxis.grid(False)
        plt.tight_layout()
        _save('environment_map')

    except Exception as e:
        print(f'  [SKIP] environment_map: {e}')


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global RES, FIGS
    parser = argparse.ArgumentParser()
    parser.add_argument('--results-dir', default='results',
                        help='Experiment output directory containing master_results.csv')
    parser.add_argument('--figures-dir', default=None,
                        help='Figure output directory. Defaults to <results-dir>/figures')
    args = parser.parse_args()

    RES = Path(args.results_dir)
    FIGS = Path(args.figures_dir) if args.figures_dir else RES / 'figures'

    master_csv = RES / 'master_results.csv'
    stats_csv  = RES / 'statistical_summary.csv'

    if not master_csv.exists():
        print(f'[ERROR] {master_csv} not found.')
        print('        Run the experiment first: python run_experiment.py --trial all --runs 50 --output results')
        return

    master = pd.read_csv(master_csv)
    print(f'Loaded {len(master)} trials from master_results.csv')
    print(f'Generating figures → {FIGS.resolve()}\n')

    print('1/8  Grouped bar chart...')
    fig_grouped_bar(master)

    print('2/8  Box + strip plot...')
    fig_boxplot(RES)

    print('3/8  Convergence curves (A, E, H)...')
    for tid in ['A', 'E', 'H']:
        fig_convergence(RES, tid)

    print('4/8  Radar chart...')
    fig_radar(master)

    print('5/8  Statistical heatmap...')
    fig_stat_heatmap(stats_csv)

    print('6/8  Scalability chart...')
    fig_scalability(master)

    print('7/8  Improvement summary...')
    fig_improvement_summary(master)

    print('8/8  Environment map...')
    fig_environment_map()

    print(f'\nDone! {len(list(FIGS.glob("*.pdf")))} figures in {FIGS}')
    print('\nNext steps:')
    print('  mkdir -p ../manuscript/figures')
    print(f'  cp {FIGS}/*.pdf ../manuscript/figures/')
    print('  cd ../manuscript && pdflatex manuscript.tex && bibtex manuscript && pdflatex manuscript.tex && pdflatex manuscript.tex')


if __name__ == '__main__':
    main()
