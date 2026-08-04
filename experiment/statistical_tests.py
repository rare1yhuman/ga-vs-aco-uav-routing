"""
Statistical Tests — GA vs ACO Paper

Rigorous non-parametric statistical comparison between algorithms.
Run AFTER collecting all experiment results (50 runs per trial).

Tests:
    - Wilcoxon signed-rank test   (paired samples, same seed sets)
    - Mann-Whitney U test         (independent samples)
    - Cohen's d effect size       (standardised mean difference)
    - Cliff's delta               (non-parametric effect direction/strength)
    - Bootstrap 95% CI            (mean improvement interval)
    - Holm correction             (multiple-comparison correction)
    - Shapiro-Wilk normality test (to justify non-parametric choice)

Output:
    <output>/statistical_summary.csv    — p-values + effect sizes per trial
    <output>/statistical_report.txt     — human-readable interpretation
"""

import json
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from typing import List, Dict, Tuple


# ── Effect size thresholds (Cohen 1988) ─────────────────────────────────────
COHEN_SMALL  = 0.2
COHEN_MEDIUM = 0.5
COHEN_LARGE  = 0.8

ALPHA = 0.05   # significance threshold


def cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Compute Cohen's d effect size between two samples.
    
    When both samples have zero variance (std=0), returns:
      - float('inf') or float('-inf') if means differ (perfect separation)
      - 0.0 if means are also identical (true tie)
    """
    pooled_std = np.sqrt((np.std(a, ddof=1)**2 + np.std(b, ddof=1)**2) / 2)
    mean_diff = float(np.mean(a) - np.mean(b))
    if pooled_std == 0:
        if mean_diff > 0:
            return float('inf')
        elif mean_diff < 0:
            return float('-inf')
        else:
            return 0.0
    return mean_diff / pooled_std


def effect_label(d: float) -> str:
    if np.isinf(d):
        return "perfect"   # zero-variance perfect separation
    ad = abs(d)
    if ad < COHEN_SMALL:   return "negligible"
    if ad < COHEN_MEDIUM:  return "small"
    if ad < COHEN_LARGE:   return "medium"
    return "large"


def cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: negative means sample a tends to be smaller/better than b."""
    gt = 0
    lt = 0
    for av in a:
        gt += int(np.sum(av > b))
        lt += int(np.sum(av < b))
    denom = len(a) * len(b)
    return float((gt - lt) / denom) if denom else 0.0


def bootstrap_improvement_ci(a: np.ndarray, b: np.ndarray,
                             n_boot: int = 5000,
                             seed: int = 12345) -> Tuple[float, float]:
    """Bootstrap CI for percent improvement: positive means MGA better."""
    rng = np.random.default_rng(seed)
    if len(a) == 0 or len(b) == 0:
        return (float('nan'), float('nan'))
    vals = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        sb_mean = float(np.mean(sb))
        vals.append(100 * (sb_mean - float(np.mean(sa))) / max(sb_mean, 1e-9))
    return (float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5)))


def _zero_variance(x: np.ndarray) -> bool:
    return len(x) <= 1 or float(np.std(x, ddof=0)) == 0.0


def holm_bonferroni(p_values: List[float], alpha: float = ALPHA) -> Tuple[List[float], List[bool]]:
    """Return Holm-adjusted p-values and reject decisions in original order."""
    m = len(p_values)
    indexed = sorted(enumerate([1.0 if not np.isfinite(p) else float(p) for p in p_values]),
                     key=lambda x: x[1])
    adjusted = [1.0] * m
    reject = [False] * m
    prev_adj = 0.0
    still_rejecting = True
    for rank, (idx, p) in enumerate(indexed):
        adj = min(1.0, max(prev_adj, (m - rank) * p))
        adjusted[idx] = adj
        prev_adj = adj
        if still_rejecting and p <= alpha / (m - rank):
            reject[idx] = True
        else:
            still_rejecting = False
    return adjusted, reject


def run_all_tests(
    mga_results: Dict[str, List[float]],
    aco_results: Dict[str, List[float]],
    output_dir: Path,
) -> pd.DataFrame:
    """
    Run statistical tests for each trial.

    Args:
        mga_results: {trial_id: [best_distance_run1, ..., run50]}
        aco_results: {trial_id: [best_distance_run1, ..., run50]}
        output_dir:  Directory to save output files.

    Returns:
        DataFrame with one row per trial.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = []
    for trial_id in sorted(mga_results.keys()):
        mga = np.array(mga_results[trial_id])
        aco = np.array(aco_results[trial_id])

        # Normality check. Shapiro is undefined/informative for constant samples.
        if len(mga) >= 3 and not _zero_variance(mga):
            _, p_norm_mga = stats.shapiro(mga)
        else:
            p_norm_mga = 0.0
        if len(aco) >= 3 and not _zero_variance(aco):
            _, p_norm_aco = stats.shapiro(aco)
        else:
            p_norm_aco = 0.0
        both_normal   = p_norm_mga > ALPHA and p_norm_aco > ALPHA

        # Wilcoxon signed-rank (paired)
        if len(mga) == len(aco) and np.allclose(mga, aco):
            p_wilcoxon = 1.0
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _, p_wilcoxon = stats.wilcoxon(mga, aco, alternative='two-sided')
            except Exception:
                p_wilcoxon = 1.0
        if not np.isfinite(p_wilcoxon):
            p_wilcoxon = 1.0

        # Mann-Whitney U (independent)
        if np.allclose(np.mean(mga), np.mean(aco)) and _zero_variance(mga) and _zero_variance(aco):
            p_mannwhitney = 1.0
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')
                    _, p_mannwhitney = stats.mannwhitneyu(mga, aco, alternative='two-sided')
            except Exception:
                p_mannwhitney = 1.0
        if not np.isfinite(p_mannwhitney):
            p_mannwhitney = 1.0

        # Effect size
        d     = cohens_d(mga, aco)
        label = effect_label(d)
        cliff = cliffs_delta(mga, aco)
        ci_low, ci_high = bootstrap_improvement_ci(mga, aco, seed=1000 + len(records))

        # Who wins?
        mga_mean = float(np.mean(mga))
        aco_mean = float(np.mean(aco))
        winner   = "MGA" if mga_mean < aco_mean else ("ACO" if aco_mean < mga_mean else "TIE")
        pct_diff = 100 * (aco_mean - mga_mean) / aco_mean if aco_mean != 0 else 0.0

        sig_wilcoxon     = p_wilcoxon < ALPHA
        sig_mannwhitney  = p_mannwhitney < ALPHA

        record = {
            'trial':              trial_id,
            'mga_mean':           round(mga_mean, 2),
            'aco_mean':           round(aco_mean, 2),
            'mga_std':            round(float(np.std(mga, ddof=1)), 2),
            'aco_std':            round(float(np.std(aco, ddof=1)), 2),
            'mga_best':           round(float(np.min(mga)), 2),
            'aco_best':           round(float(np.min(aco)), 2),
            'winner':             winner,
            'pct_improvement':    round(pct_diff, 2),
            'p_wilcoxon':         round(p_wilcoxon, 4),
            'p_mannwhitney':      round(p_mannwhitney, 4),
            'p_wilcoxon_holm':    1.0,
            'significant_holm':   False,
            'significant_wilcoxon': sig_wilcoxon,
            'significant_mannwhitney': sig_mannwhitney,
            'cohens_d':           round(d, 3),
            'cliffs_delta':       round(cliff, 3),
            'bootstrap_ci_low_%': round(ci_low, 2),
            'bootstrap_ci_high_%': round(ci_high, 2),
            'effect_size':        label,
            'both_normal':        both_normal,
        }
        records.append(record)

    df = pd.DataFrame(records)
    if not df.empty:
        adj, reject = holm_bonferroni(df['p_wilcoxon'].to_list())
        df['p_wilcoxon_holm'] = [round(v, 4) for v in adj]
        df['significant_holm'] = reject

    # Save CSV
    csv_path = output_dir / 'statistical_summary.csv'
    df.to_csv(csv_path, index=False)
    print(f"[stats] Summary saved to {csv_path}")

    # Save human-readable report
    _write_report(df, output_dir / 'statistical_report.txt')

    return df


def _write_report(df: pd.DataFrame, path: Path):
    """Write a human-readable summary for copy-pasting into the paper."""
    lines = [
        "=" * 70,
        "STATISTICAL COMPARISON: MGA-2opt vs ACO",
        "=" * 70,
        f"Significance level: α = {ALPHA}",
        f"Tests: Wilcoxon signed-rank (paired) + Mann-Whitney U (independent)",
        f"Effect size: Cohen's d  (small={COHEN_SMALL}, medium={COHEN_MEDIUM}, large={COHEN_LARGE})",
        "",
    ]

    for _, row in df.iterrows():
        sig_str = ("✓ SIGNIFICANT" if row['significant_wilcoxon']
                   else "✗ not significant")
        lines += [
            f"Trial {row['trial']}",
            f"  MGA : {row['mga_mean']:.1f} ± {row['mga_std']:.1f} m  (best={row['mga_best']:.1f})",
            f"  ACO : {row['aco_mean']:.1f} ± {row['aco_std']:.1f} m  (best={row['aco_best']:.1f})",
            f"  Winner: {row['winner']}  ({row['pct_improvement']:+.1f}%)",
            f"  Wilcoxon p={row['p_wilcoxon']:.4f}  Holm p={row['p_wilcoxon_holm']:.4f}  MW p={row['p_mannwhitney']:.4f}  {sig_str}",
            f"  Cohen's d={row['cohens_d']:.3f} ({row['effect_size']} effect); Cliff's delta={row['cliffs_delta']:.3f}",
            f"  Bootstrap 95% CI for improvement: [{row['bootstrap_ci_low_%']:+.2f}%, {row['bootstrap_ci_high_%']:+.2f}%]",
            "",
        ]

    # Overall summary
    mga_wins = int((df['winner'] == 'MGA').sum())
    aco_wins = int((df['winner'] == 'ACO').sum())
    ties     = int((df['winner'] == 'TIE').sum())
    sig_wins = int(df['significant_holm'].sum())

    lines += [
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"MGA wins: {mga_wins}/{len(df)} trials",
        f"ACO wins: {aco_wins}/{len(df)} trials",
        f"Ties:     {ties}/{len(df)} trials",
        f"Holm-corrected statistically significant results: {sig_wins}/{len(df)} trials",
        "",
        "Paper claim (update as needed):",
        f"MGA-2opt outperforms ACO in {mga_wins} of {len(df)} trials "
        f"({sig_wins} statistically significant at α={ALPHA}).",
    ]

    path.write_text("\n".join(lines))
    print(f"[stats] Report saved to {path}")


def print_latex_table(df: pd.DataFrame):
    """Print a LaTeX table snippet ready to paste into manuscript.tex."""
    print("\n% ── LaTeX Table: Statistical Results ──────────────────────────")
    print("\\begin{table}[h]")
    print("\\centering")
    print("\\caption{Statistical Comparison of MGA-2opt vs ACO}")
    print("\\begin{tabular}{lrrrrrc}")
    print("\\hline")
    print("Trial & MGA (m) & ACO (m) & Impr.\\% & $p_{Holm}$ & Cohen's $d$ & Sig. \\\\")
    print("\\hline")
    for _, row in df.iterrows():
        sig = "\\checkmark" if row.get('significant_holm', False) else "--"
        print(f"  {row['trial']} & "
              f"{row['mga_mean']:.0f}$\\pm${row['mga_std']:.0f} & "
              f"{row['aco_mean']:.0f}$\\pm${row['aco_std']:.0f} & "
              f"{row['pct_improvement']:+.1f} & "
              f"{row['p_wilcoxon_holm']:.4f} & "
              f"{row['cohens_d']:.3f} & "
              f"{sig} \\\\")
    print("\\hline")
    print("\\end{tabular}")
    print("\\end{table}")


# ── Smoke test ───────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import tempfile, os
    rng = np.random.default_rng(0)

    # Simulate MGA slightly better than ACO
    mga_sim = {'A': list(rng.normal(8500, 200, 50)),
               'B': list(rng.normal(9100, 300, 50)),
               'E': list(rng.normal(10200, 400, 50))}
    aco_sim = {'A': list(rng.normal(9200, 250, 50)),
               'B': list(rng.normal(9050, 280, 50)),
               'E': list(rng.normal(11500, 600, 50))}

    with tempfile.TemporaryDirectory() as tmp:
        df = run_all_tests(mga_sim, aco_sim, Path(tmp))
        print("\n", df[['trial','winner','pct_improvement',
                         'p_wilcoxon','cohens_d','effect_size']].to_string(index=False))
        print_latex_table(df)
