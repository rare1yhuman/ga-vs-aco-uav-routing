"""Map-level statistical analysis for the MGA-versus-ACO paper protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest, rankdata, wilcoxon

from research_protocol import SCENARIO_FAMILIES


BOOTSTRAP_SEED = 20260729
FAMILY_ORDER = tuple(SCENARIO_FAMILIES)
DYNAMIC_FAMILIES = ("S3", "S5", "S7")
ALGORITHMS = ("MGA", "ACO")
TIMING_COLUMNS = {
    "wall_time_s",
    "cpu_time_s",
    "native_wall_time_s",
    "native_cpu_time_s",
    "history_wall_s",
    "history_cpu_s",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _holm_adjust(p_values: Sequence[float]) -> List[float]:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    count = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (count - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def _paired_wilcoxon(left: Sequence[float], right: Sequence[float]) -> float:
    left_array = np.asarray(left, dtype=float)
    right_array = np.asarray(right, dtype=float)
    finite = np.isfinite(left_array) & np.isfinite(right_array)
    left_array = left_array[finite]
    right_array = right_array[finite]
    if not len(left_array) or np.allclose(left_array, right_array):
        return 1.0
    return float(
        wilcoxon(
            left_array,
            right_array,
            zero_method="pratt",
            alternative="two-sided",
        ).pvalue
    )


def _paired_rank_biserial(differences: Sequence[float]) -> float:
    """Matched-pairs rank-biserial effect; positive values favor MGA."""
    values = np.asarray(differences, dtype=float)
    values = values[np.isfinite(values) & ~np.isclose(values, 0.0)]
    if not len(values):
        return 0.0
    ranks = rankdata(np.abs(values), method="average")
    positive = float(ranks[values > 0].sum())
    negative = float(ranks[values < 0].sum())
    return (positive - negative) / (positive + negative)


def _bootstrap_map_median(
    values: Sequence[float],
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    indices = rng.integers(0, len(array), size=(iterations, len(array)))
    estimates = np.median(array[indices], axis=1)
    return tuple(np.quantile(estimates, [0.025, 0.975]).tolist())


def _paired_metric_cube(
    frame: pd.DataFrame,
    column: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    maps = sorted(int(seed) for seed in frame["environment_seed"].unique())
    optimizers = sorted(int(seed) for seed in frame["optimizer_seed"].unique())
    cube = np.empty((len(maps), len(optimizers), 2), dtype=float)
    for map_index, environment_seed in enumerate(maps):
        map_frame = frame[frame["environment_seed"] == environment_seed]
        pivot = map_frame.pivot_table(
            index="optimizer_seed",
            columns="algo",
            values=column,
            aggfunc="first",
        ).reindex(optimizers)
        cube[map_index, :, 0] = pivot["MGA"].to_numpy(dtype=float)
        cube[map_index, :, 1] = pivot["ACO"].to_numpy(dtype=float)
    return cube, maps, optimizers


def _hierarchical_ratio_ci(
    frame: pd.DataFrame,
    column: str,
    iterations: int,
    rng: np.random.Generator,
    *,
    percent_gain: bool,
) -> Tuple[float, float]:
    """Resample maps, then paired optimizer seeds within each selected map."""
    cube, maps, optimizers = _paired_metric_cube(frame, column)
    n_maps = len(maps)
    n_optimizers = len(optimizers)
    estimates: List[np.ndarray] = []
    remaining = iterations
    while remaining:
        batch = min(1000, remaining)
        map_indices = rng.integers(0, n_maps, size=(batch, n_maps))
        optimizer_indices = rng.integers(
            0,
            n_optimizers,
            size=(batch, n_maps, n_optimizers),
        )
        sampled = cube[map_indices[:, :, None], optimizer_indices]
        map_medians = np.median(sampled, axis=2)
        if percent_gain:
            ratios = (
                100.0
                * (map_medians[:, :, 1] - map_medians[:, :, 0])
                / map_medians[:, :, 1]
            )
        else:
            ratios = map_medians[:, :, 1] / map_medians[:, :, 0]
        estimates.append(np.median(ratios, axis=1))
        remaining -= batch
    distribution = np.concatenate(estimates)
    return tuple(np.quantile(distribution, [0.025, 0.975]).tolist())


def _mcnemar_exact(frame: pd.DataFrame) -> Tuple[int, int, float]:
    paired = frame.pivot_table(
        index=["environment_seed", "optimizer_seed"],
        columns="algo",
        values="dynamic_safe",
        aggfunc="first",
    )
    mga_only = int(((paired["MGA"] == True) & (paired["ACO"] == False)).sum())
    aco_only = int(((paired["MGA"] == False) & (paired["ACO"] == True)).sum())
    discordant = mga_only + aco_only
    p_value = (
        1.0
        if discordant == 0
        else float(
            binomtest(
                min(mga_only, aco_only),
                discordant,
                0.5,
                alternative="two-sided",
            ).pvalue
        )
    )
    return mga_only, aco_only, p_value


def _json_list(value: Any) -> List[float]:
    if isinstance(value, str):
        return [float(item) for item in json.loads(value)]
    return [float(item) for item in value]


def _construction_counts(row: Any) -> np.ndarray:
    """Cumulative candidate constructions at each recorded history step.

    ACO's ``history_evaluations`` already counts ant tour constructions
    directly (250 ants x completed iterations), so it is used unmodified.
    MGA's ``history_evaluations`` instead counts fitness-cache misses,
    which understates true constructions once the run-local fitness cache
    starts absorbing repeated genotypes. MGA constructs a full
    ``pop_size`` population for the seed generation and again for every
    subsequent generation's offspring, so true cumulative constructions
    after history index ``i`` (0-based) is ``pop_size * (i + 2)``.
    ``pop_size`` is derived from the row's own validated
    ``candidate_constructions`` and ``iterations_completed`` fields rather
    than re-reading the frozen config, and the series is capped at the
    run's declared budget so the optional post-loop final-refinement
    sample never exceeds it.
    """
    raw = np.asarray(_json_list(row.history_evaluations), dtype=float)
    if row.algo != "MGA":
        return raw
    budget = float(row.candidate_constructions)
    generations = float(row.iterations_completed)
    pop_size = budget / (generations + 1.0)
    steps = np.arange(1, len(raw) + 1, dtype=float)
    return np.minimum(pop_size * (steps + 1.0), budget)


def _first_target_time(
    costs: Sequence[float],
    times: Sequence[float],
    target: float,
) -> float:
    for cost, elapsed in zip(costs, times):
        if float(cost) <= target:
            return float(elapsed)
    return float("nan")


def _validate_frame(frame: pd.DataFrame) -> None:
    required = {
        "family",
        "environment_seed",
        "optimizer_seed",
        "algo",
        "generalized_cost",
        "physical_distance_m",
        "exposure_cells",
        "dynamic_safe",
        "cpu_time_s",
        "candidate_constructions",
        "history_cost",
        "history_cpu_s",
        "history_evaluations",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if len(frame) != 700:
        raise ValueError(f"Expected 700 final rows, found {len(frame)}")
    if set(frame["family"]) != set(FAMILY_ORDER):
        raise ValueError("Final results do not contain the complete S1-S7 matrix")
    if set(frame["algo"]) != set(ALGORITHMS):
        raise ValueError("Final results must contain MGA and ACO only")
    if frame["error"].fillna("").astype(str).str.strip().ne("").any():
        raise ValueError("Final results contain software errors")
    if not (frame["candidate_constructions"] == 100000).all():
        raise ValueError("Candidate-construction budgets are not equal")


def _build_map_summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for (family, environment_seed), group in frame.groupby(
        ["family", "environment_seed"],
        sort=True,
    ):
        row: Dict[str, Any] = {
            "family": family,
            "family_label": SCENARIO_FAMILIES[family]["label"],
            "environment_seed": int(environment_seed),
            "n_deliveries": int(group["n_deliveries"].iloc[0]),
            "dynamic": bool(group["dynamic"].iloc[0]),
            "use_3d": bool(group["use_3d"].iloc[0]),
        }
        for algorithm in ALGORITHMS:
            subset = group[group["algo"] == algorithm]
            prefix = algorithm.lower()
            row[f"{prefix}_median_cost"] = float(
                subset["generalized_cost"].median()
            )
            row[f"{prefix}_median_physical_m"] = float(
                subset["physical_distance_m"].median()
            )
            row[f"{prefix}_median_exposure"] = float(
                subset["exposure_cells"].median()
            )
            row[f"{prefix}_median_cpu_s"] = float(subset["cpu_time_s"].median())
            row[f"{prefix}_median_wall_s"] = float(
                subset["wall_time_s"].median()
            )
            row[f"{prefix}_safety_rate"] = float(
                subset["dynamic_safe"].astype(bool).mean()
            )
        row["cost_gain_pct"] = 100.0 * (
            row["aco_median_cost"] - row["mga_median_cost"]
        ) / row["aco_median_cost"]
        row["physical_gain_pct"] = 100.0 * (
            row["aco_median_physical_m"] - row["mga_median_physical_m"]
        ) / row["aco_median_physical_m"]
        row["exposure_reduction"] = (
            row["aco_median_exposure"] - row["mga_median_exposure"]
        )
        row["cpu_speedup"] = (
            row["aco_median_cpu_s"] / row["mga_median_cpu_s"]
        )
        row["winner"] = (
            "MGA"
            if row["mga_median_cost"] < row["aco_median_cost"]
            else "ACO"
            if row["mga_median_cost"] > row["aco_median_cost"]
            else "TIE"
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values(
        ["family", "environment_seed"]
    ).reset_index(drop=True)


def _build_family_summary(
    frame: pd.DataFrame,
    maps: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_maps = maps[maps["family"] == family]
        raw_family = frame[frame["family"] == family]
        cost_ci = _hierarchical_ratio_ci(
            raw_family,
            "generalized_cost",
            iterations,
            rng,
            percent_gain=True,
        )
        physical_ci = _hierarchical_ratio_ci(
            raw_family,
            "physical_distance_m",
            iterations,
            rng,
            percent_gain=True,
        )
        cost_difference = (
            family_maps["aco_median_cost"]
            - family_maps["mga_median_cost"]
        ).to_numpy()
        physical_difference = (
            family_maps["aco_median_physical_m"]
            - family_maps["mga_median_physical_m"]
        ).to_numpy()
        wins = int((family_maps["winner"] == "MGA").sum())
        ties = int((family_maps["winner"] == "TIE").sum())
        losses = int((family_maps["winner"] == "ACO").sum())
        rows.append(
            {
                "family": family,
                "label": SCENARIO_FAMILIES[family]["label"],
                "maps": int(len(family_maps)),
                "mga_median_cost": float(
                    family_maps["mga_median_cost"].median()
                ),
                "aco_median_cost": float(
                    family_maps["aco_median_cost"].median()
                ),
                "median_cost_gain_pct": float(
                    family_maps["cost_gain_pct"].median()
                ),
                "gain_ci_low_pct": cost_ci[0],
                "gain_ci_high_pct": cost_ci[1],
                "cost_p_wilcoxon": _paired_wilcoxon(
                    family_maps["mga_median_cost"],
                    family_maps["aco_median_cost"],
                ),
                "cost_rank_biserial": _paired_rank_biserial(cost_difference),
                "wins": wins,
                "ties": ties,
                "losses": losses,
                "mga_median_physical_m": float(
                    family_maps["mga_median_physical_m"].median()
                ),
                "aco_median_physical_m": float(
                    family_maps["aco_median_physical_m"].median()
                ),
                "median_physical_gain_pct": float(
                    family_maps["physical_gain_pct"].median()
                ),
                "physical_ci_low_pct": physical_ci[0],
                "physical_ci_high_pct": physical_ci[1],
                "physical_p_wilcoxon": _paired_wilcoxon(
                    family_maps["mga_median_physical_m"],
                    family_maps["aco_median_physical_m"],
                ),
                "physical_rank_biserial": _paired_rank_biserial(
                    physical_difference
                ),
            }
        )
    summary = pd.DataFrame(rows)
    summary["cost_p_holm"] = _holm_adjust(summary["cost_p_wilcoxon"])
    summary["physical_p_holm"] = _holm_adjust(
        summary["physical_p_wilcoxon"]
    )
    summary["cost_significant_0_05"] = summary["cost_p_holm"] < 0.05
    return summary


def _build_speed_summary(
    maps: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        group = maps[maps["family"] == family]
        ci_low, ci_high = _bootstrap_map_median(
            group["cpu_speedup"],
            iterations,
            rng,
        )
        differences = (
            group["aco_median_cpu_s"] - group["mga_median_cpu_s"]
        )
        rows.append(
            {
                "family": family,
                "maps": int(len(group)),
                "mga_median_cpu_s": float(
                    group["mga_median_cpu_s"].median()
                ),
                "aco_median_cpu_s": float(
                    group["aco_median_cpu_s"].median()
                ),
                "median_speedup": float(group["cpu_speedup"].median()),
                "speedup_ci_low": ci_low,
                "speedup_ci_high": ci_high,
                "cpu_p_wilcoxon": _paired_wilcoxon(
                    group["mga_median_cpu_s"],
                    group["aco_median_cpu_s"],
                ),
                "cpu_rank_biserial": _paired_rank_biserial(differences),
            }
        )
    summary = pd.DataFrame(rows)
    summary["cpu_p_holm"] = _holm_adjust(summary["cpu_p_wilcoxon"])
    return summary


def _build_safety_summary(
    frame: pd.DataFrame,
    maps: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in DYNAMIC_FAMILIES:
        raw_family = frame[frame["family"] == family]
        family_maps = maps[maps["family"] == family]
        mga_only, aco_only, mcnemar_p = _mcnemar_exact(raw_family)
        safety_difference = (
            family_maps["mga_safety_rate"]
            - family_maps["aco_safety_rate"]
        )
        exposure_difference = family_maps["exposure_reduction"]
        rows.append(
            {
                "family": family,
                "maps": int(len(family_maps)),
                "mga_safety_rate": float(
                    raw_family[raw_family["algo"] == "MGA"][
                        "dynamic_safe"
                    ].astype(bool).mean()
                ),
                "aco_safety_rate": float(
                    raw_family[raw_family["algo"] == "ACO"][
                        "dynamic_safe"
                    ].astype(bool).mean()
                ),
                "median_map_safety_delta": float(
                    safety_difference.median()
                ),
                "safety_map_p_wilcoxon": _paired_wilcoxon(
                    family_maps["mga_safety_rate"],
                    family_maps["aco_safety_rate"],
                ),
                "safety_rank_biserial": _paired_rank_biserial(
                    safety_difference
                ),
                "mga_median_exposure": float(
                    family_maps["mga_median_exposure"].median()
                ),
                "aco_median_exposure": float(
                    family_maps["aco_median_exposure"].median()
                ),
                "median_exposure_reduction": float(
                    exposure_difference.median()
                ),
                "exposure_p_wilcoxon": _paired_wilcoxon(
                    family_maps["mga_median_exposure"],
                    family_maps["aco_median_exposure"],
                ),
                "exposure_rank_biserial": _paired_rank_biserial(
                    exposure_difference
                ),
                "mga_only_safe_pairs": mga_only,
                "aco_only_safe_pairs": aco_only,
                "mcnemar_descriptive_p": mcnemar_p,
            }
        )
    summary = pd.DataFrame(rows)
    summary["safety_p_holm"] = _holm_adjust(
        summary["safety_map_p_wilcoxon"]
    )
    summary["exposure_p_holm"] = _holm_adjust(
        summary["exposure_p_wilcoxon"]
    )
    return summary


def _build_scale_interaction(
    maps: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    comparisons = (
        ("static", "S4", "S6"),
        ("dynamic", "S5", "S7"),
    )
    detail_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    for label, small_family, large_family in comparisons:
        small = maps[maps["family"] == small_family].set_index(
            "environment_seed"
        )
        large = maps[maps["family"] == large_family].set_index(
            "environment_seed"
        )
        seeds = sorted(set(small.index) & set(large.index))
        differences: List[float] = []
        for seed in seeds:
            small_gain = float(small.loc[seed, "cost_gain_pct"])
            large_gain = float(large.loc[seed, "cost_gain_pct"])
            increase = large_gain - small_gain
            differences.append(increase)
            detail_rows.append(
                {
                    "comparison": label,
                    "environment_seed": int(seed),
                    "small_family": small_family,
                    "large_family": large_family,
                    "small_gain_pct": small_gain,
                    "large_gain_pct": large_gain,
                    "gain_increase_pct_points": increase,
                }
            )
        ci_low, ci_high = _bootstrap_map_median(
            differences,
            iterations,
            rng,
        )
        difference_array = np.asarray(differences, dtype=float)
        summary_rows.append(
            {
                "comparison": label,
                "small_family": small_family,
                "large_family": large_family,
                "maps": len(differences),
                "median_gain_increase_pct_points": float(
                    np.median(difference_array)
                ),
                "increase_ci_low": ci_low,
                "increase_ci_high": ci_high,
                "wins": int(np.count_nonzero(difference_array > 0)),
                "ties": int(np.count_nonzero(np.isclose(difference_array, 0))),
                "losses": int(np.count_nonzero(difference_array < 0)),
                "p_wilcoxon": _paired_wilcoxon(
                    np.zeros_like(difference_array),
                    difference_array,
                ),
                "rank_biserial": _paired_rank_biserial(difference_array),
            }
        )
    detail = pd.DataFrame(detail_rows)
    summary = pd.DataFrame(summary_rows)
    summary["p_holm"] = _holm_adjust(summary["p_wilcoxon"])
    return detail, summary


def _build_time_to_target(
    frame: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    output: List[Dict[str, Any]] = []
    for (family, environment_seed), map_frame in frame.groupby(
        ["family", "environment_seed"],
        sort=True,
    ):
        target = float(map_frame["generalized_cost"].min()) * 1.01
        for row in map_frame.itertuples():
            costs = _json_list(row.history_cost)
            cpu_times = _json_list(row.history_cpu_s)
            wall_times = _json_list(row.history_wall_s)
            target_cpu = _first_target_time(costs, cpu_times, target)
            target_wall = _first_target_time(costs, wall_times, target)
            attained = bool(np.isfinite(target_cpu))
            final_cpu = float(cpu_times[-1])
            final_wall = float(wall_times[-1])
            output.append(
                {
                    "family": family,
                    "environment_seed": int(environment_seed),
                    "optimizer_seed": int(row.optimizer_seed),
                    "algo": row.algo,
                    "target_cost": target,
                    "attained": attained,
                    "target_cpu_time_s": target_cpu,
                    "target_wall_time_s": target_wall,
                    "observed_cpu_time_s": (
                        target_cpu if attained else final_cpu
                    ),
                    "observed_wall_time_s": (
                        target_wall if attained else final_wall
                    ),
                    "final_cpu_time_s": final_cpu,
                    "final_wall_time_s": final_wall,
                }
            )
    runs = pd.DataFrame(output)
    map_summary = (
        runs.groupby(["family", "environment_seed", "algo"], sort=True)
        .agg(
            attainment_rate=("attained", "mean"),
            attained_runs=("attained", "sum"),
            median_target_cpu_s=("target_cpu_time_s", "median"),
            median_target_wall_s=("target_wall_time_s", "median"),
            median_final_cpu_s=("final_cpu_time_s", "median"),
        )
        .reset_index()
    )
    family_summary = (
        runs.groupby(["family", "algo"], sort=True)
        .agg(
            runs=("attained", "size"),
            attained_runs=("attained", "sum"),
            attainment_rate=("attained", "mean"),
            conditional_median_target_cpu_s=("target_cpu_time_s", "median"),
            conditional_median_target_wall_s=("target_wall_time_s", "median"),
            median_final_cpu_s=("final_cpu_time_s", "median"),
        )
        .reset_index()
    )
    family_summary["conditional_time_note"] = (
        "Median among attained runs; interpret with attainment_rate"
    )
    return runs, map_summary, family_summary


def _build_attainment_curves(target_runs: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for family in FAMILY_ORDER:
        family_runs = target_runs[target_runs["family"] == family]
        positive = family_runs.loc[
            family_runs["observed_cpu_time_s"] > 0,
            "observed_cpu_time_s",
        ]
        low = max(float(positive.min()), 1e-3)
        high = float(family_runs["final_cpu_time_s"].max())
        time_grid = np.unique(
            np.concatenate(([0.0], np.geomspace(low, high, 120)))
        )
        for algorithm in ALGORITHMS:
            group = family_runs[family_runs["algo"] == algorithm]
            attained_times = group["target_cpu_time_s"].to_numpy(dtype=float)
            for elapsed in time_grid:
                reached = np.isfinite(attained_times) & (
                    attained_times <= elapsed
                )
                rows.append(
                    {
                        "family": family,
                        "algo": algorithm,
                        "cpu_time_s": float(elapsed),
                        "attainment_fraction": float(reached.mean()),
                    }
                )
    return pd.DataFrame(rows)


def _build_convergence(
    frame: pd.DataFrame,
    iterations: int,
    rng: np.random.Generator,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    evaluation_grid = np.arange(1000, 100001, 1000, dtype=int)
    run_rows: List[Dict[str, Any]] = []
    for (family, environment_seed), map_frame in frame.groupby(
        ["family", "environment_seed"],
        sort=True,
    ):
        pooled_best = float(map_frame["generalized_cost"].min())
        for row in map_frame.itertuples():
            costs = np.asarray(_json_list(row.history_cost), dtype=float)
            constructions = _construction_counts(row)
            positions = np.searchsorted(
                constructions,
                evaluation_grid,
                side="right",
            ) - 1
            positions = np.clip(positions, 0, len(costs) - 1)
            regrets = 100.0 * (costs[positions] - pooled_best) / pooled_best
            for evaluation, regret in zip(evaluation_grid, regrets):
                run_rows.append(
                    {
                        "family": family,
                        "environment_seed": int(environment_seed),
                        "optimizer_seed": int(row.optimizer_seed),
                        "algo": row.algo,
                        "evaluations": int(evaluation),
                        "normalized_regret_pct": float(max(regret, 0.0)),
                    }
                )
    runs = pd.DataFrame(run_rows)
    map_curves = (
        runs.groupby(
            ["family", "environment_seed", "algo", "evaluations"],
            sort=True,
        )["normalized_regret_pct"]
        .median()
        .reset_index()
    )
    summary_rows: List[Dict[str, Any]] = []
    for (family, algorithm), group in map_curves.groupby(
        ["family", "algo"],
        sort=True,
    ):
        pivot = group.pivot(
            index="environment_seed",
            columns="evaluations",
            values="normalized_regret_pct",
        ).sort_index(axis=1)
        matrix = pivot.to_numpy(dtype=float)
        sample_indices = rng.integers(
            0,
            matrix.shape[0],
            size=(iterations, matrix.shape[0]),
        )
        boot = np.median(matrix[sample_indices], axis=1)
        medians = np.median(matrix, axis=0)
        lows = np.quantile(boot, 0.025, axis=0)
        highs = np.quantile(boot, 0.975, axis=0)
        for evaluation, median, low, high in zip(
            pivot.columns,
            medians,
            lows,
            highs,
        ):
            summary_rows.append(
                {
                    "family": family,
                    "algo": algorithm,
                    "evaluations": int(evaluation),
                    "median_regret_pct": float(median),
                    "ci_low_pct": float(low),
                    "ci_high_pct": float(high),
                }
            )
    return map_curves, pd.DataFrame(summary_rows)


def _markdown_table(frame: pd.DataFrame) -> str:
    columns = list(frame.columns)
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = []
    for values in frame.itertuples(index=False, name=None):
        formatted = []
        for value in values:
            if isinstance(value, (float, np.floating)):
                formatted.append(
                    "NA" if not np.isfinite(value) else f"{float(value):.4g}"
                )
            else:
                formatted.append(str(value))
        rows.append("| " + " | ".join(formatted) + " |")
    return "\n".join([header, separator, *rows])


def _write_report(
    path: Path,
    family: pd.DataFrame,
    speed: pd.DataFrame,
    safety: pd.DataFrame,
    scale: pd.DataFrame,
    target: pd.DataFrame,
) -> None:
    lines = [
        "# Statistical Report",
        "",
        "The independent environment map is the inferential unit. Five paired "
        "optimizer seeds are summarized within each map before family tests.",
        "",
        "Positive cost gain, physical-distance gain, speedup above one, and "
        "positive rank-biserial effects favor MGA.",
        "",
        "## Route Cost and Distance",
        "",
        _markdown_table(family),
        "",
        "## Scale Interaction",
        "",
        _markdown_table(scale),
        "",
        "## CPU Time",
        "",
        _markdown_table(speed),
        "",
        "## Dynamic Safety and Exposure",
        "",
        _markdown_table(safety),
        "",
        "McNemar values are descriptive run-level diagnostics because runs are "
        "clustered within maps. Map-level Wilcoxon tests are primary.",
        "",
        "## Time to 1% Pooled-Map Target",
        "",
        _markdown_table(target),
        "",
        "Conditional target times exclude non-attaining runs and must always "
        "be interpreted together with attainment rates.",
        "",
    ]
    path.write_text("\n".join(lines))


def _write_latex_table(path: Path, family: pd.DataFrame) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Map-level comparison under the frozen equal-budget protocol. Positive gain and speedup favor MGA.}",
        r"\label{tab:main-results}",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Family & MGA cost & ACO cost & Gain (\%) & 95\% CI & W/T/L & $p_{\mathrm{Holm}}$ & CPU ratio \\",
        r"\midrule",
    ]
    for row in family.itertuples():
        lines.append(
            f"{row.family} & {row.mga_median_cost:.0f} & "
            f"{row.aco_median_cost:.0f} & {row.median_cost_gain_pct:+.2f} & "
            f"[{row.gain_ci_low_pct:+.2f}, {row.gain_ci_high_pct:+.2f}] & "
            f"{row.wins}/{row.ties}/{row.losses} & "
            f"{row.cost_p_holm:.4f} & "
            f"{row.median_cpu_speedup:.2f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""])
    path.write_text("\n".join(lines))


def _write_supplement_table(path: Path, maps: pd.DataFrame) -> None:
    lines = [
        r"\begin{longtable}{llrrrrr}",
        r"\caption{All independent map-level results.}\label{tab:all-maps}\\",
        r"\toprule",
        r"Family & Seed & MGA cost & ACO cost & Gain (\%) & Distance gain (\%) & CPU ratio \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Family & Seed & MGA cost & ACO cost & Gain (\%) & Distance gain (\%) & CPU ratio \\",
        r"\midrule",
        r"\endhead",
    ]
    for row in maps.itertuples():
        lines.append(
            f"{row.family} & {row.environment_seed} & "
            f"{row.mga_median_cost:.0f} & {row.aco_median_cost:.0f} & "
            f"{row.cost_gain_pct:+.2f} & {row.physical_gain_pct:+.2f} & "
            f"{row.cpu_speedup:.2f}$\\times$ \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}", ""])
    path.write_text("\n".join(lines))


def _write_latex_macros(
    path: Path,
    frame: pd.DataFrame,
    family: pd.DataFrame,
    speed: pd.DataFrame,
    scale: pd.DataFrame,
) -> None:
    rows = family.set_index("family")
    speed_rows = speed.set_index("family")
    scale_rows = scale.set_index("comparison")
    large_maps = family[family["family"].isin(["S6", "S7"])]
    large_wins = int(large_maps["wins"].sum())
    large_ties = int(large_maps["ties"].sum())
    lines = [
        rf"\renewcommand{{\TotalRuns}}{{{len(frame)}}}",
        rf"\renewcommand{{\IndependentMaps}}{{{frame.groupby(['family', 'environment_seed']).ngroups}}}",
        rf"\renewcommand{{\OptimizerRepeats}}{{{frame['optimizer_seed'].nunique()}}}",
        rf"\renewcommand{{\SignificantFamilies}}{{{int(family['cost_significant_0_05'].sum())}}}",
        rf"\renewcommand{{\MedianCpuSpeedup}}{{{speed['median_speedup'].median():.2f}$\times$}}",
        rf"\renewcommand{{\LargeMapWins}}{{{large_wins}}}",
        rf"\renewcommand{{\LargeMapTies}}{{{large_ties}}}",
        rf"\renewcommand{{\SsixGain}}{{{rows.loc['S6', 'median_cost_gain_pct']:.2f}\%}}",
        rf"\renewcommand{{\SsevenGain}}{{{rows.loc['S7', 'median_cost_gain_pct']:.2f}\%}}",
        rf"\renewcommand{{\SsixPhysicalGain}}{{{rows.loc['S6', 'median_physical_gain_pct']:.2f}\%}}",
        rf"\renewcommand{{\SsevenPhysicalGain}}{{{rows.loc['S7', 'median_physical_gain_pct']:.2f}\%}}",
        rf"\renewcommand{{\SsixP}}{{{rows.loc['S6', 'cost_p_holm']:.4f}}}",
        rf"\renewcommand{{\SsevenP}}{{{rows.loc['S7', 'cost_p_holm']:.4f}}}",
        rf"\renewcommand{{\SsixSpeed}}{{{speed_rows.loc['S6', 'median_speedup']:.2f}$\times$}}",
        rf"\renewcommand{{\SsevenSpeed}}{{{speed_rows.loc['S7', 'median_speedup']:.2f}$\times$}}",
        rf"\renewcommand{{\StaticScaleIncrease}}{{{scale_rows.loc['static', 'median_gain_increase_pct_points']:.2f}}}",
        rf"\renewcommand{{\DynamicScaleIncrease}}{{{scale_rows.loc['dynamic', 'median_gain_increase_pct_points']:.2f}}}",
        "",
    ]
    path.write_text("\n".join(lines))


def analyze(
    results_dir: Path,
    *,
    bootstrap_iterations: int = 10000,
) -> Dict[str, pd.DataFrame]:
    raw_path = results_dir / "raw_results.csv"
    frame = pd.read_csv(raw_path)
    _validate_frame(frame)
    analysis_dir = results_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    maps = _build_map_summary(frame)
    family = _build_family_summary(
        frame,
        maps,
        bootstrap_iterations,
        rng,
    )
    speed = _build_speed_summary(maps, bootstrap_iterations, rng)
    family = family.merge(
        speed[
            [
                "family",
                "mga_median_cpu_s",
                "aco_median_cpu_s",
                "median_speedup",
                "speedup_ci_low",
                "speedup_ci_high",
                "cpu_p_holm",
            ]
        ],
        on="family",
        validate="one_to_one",
    )
    family = family.rename(columns={"median_speedup": "median_cpu_speedup"})
    safety = _build_safety_summary(frame, maps)
    scale_detail, scale = _build_scale_interaction(
        maps,
        bootstrap_iterations,
        rng,
    )
    target_runs, target_maps, target = _build_time_to_target(frame)
    attainment = _build_attainment_curves(target_runs)
    convergence_maps, convergence = _build_convergence(
        frame,
        bootstrap_iterations,
        rng,
    )

    outputs = {
        "map_summary.csv": maps,
        "family_summary.csv": family,
        "speed_summary.csv": speed,
        "safety_summary.csv": safety,
        "scale_interaction_maps.csv": scale_detail,
        "scale_interaction_summary.csv": scale,
        "time_to_target.csv": target_runs,
        "time_to_target_map_summary.csv": target_maps,
        "time_to_target_summary.csv": target,
        "attainment_curves.csv": attainment,
        "convergence_map_curves.csv": convergence_maps,
        "convergence_summary.csv": convergence,
    }
    for filename, output in outputs.items():
        output.to_csv(analysis_dir / filename, index=False)

    _write_report(
        analysis_dir / "statistical_report.md",
        family,
        speed,
        safety,
        scale,
        target,
    )
    _write_latex_table(
        analysis_dir / "family_results_table.tex",
        family,
    )
    _write_supplement_table(
        analysis_dir / "all_map_results_table.tex",
        maps,
    )
    _write_latex_macros(
        analysis_dir / "manuscript_values.tex",
        frame,
        family,
        speed,
        scale,
    )
    manifest = {
        "status": "complete",
        "raw_results_sha256": _sha256(raw_path),
        "rows": len(frame),
        "independent_maps": int(
            frame.groupby(["family", "environment_seed"]).ngroups
        ),
        "optimizer_repeats_per_map": int(frame["optimizer_seed"].nunique()),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_iterations": int(bootstrap_iterations),
        "independent_unit": "environment map",
        "multiple_testing": {
            "cost": "Holm across seven families",
            "physical_distance": "Holm across seven families",
            "cpu_time": "Holm across seven families",
            "dynamic_safety": "Holm across S3, S5, S7",
            "dynamic_exposure": "Holm across S3, S5, S7",
            "scale_interaction": "Holm across static and dynamic comparisons",
        },
        "timing_columns": sorted(TIMING_COLUMNS),
    }
    (analysis_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    return {
        "map": maps,
        "family": family,
        "speed": speed,
        "safety": safety,
        "scale": scale,
        "target": target,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--bootstrap", type=int, default=10000)
    args = parser.parse_args()
    outputs = analyze(args.results, bootstrap_iterations=args.bootstrap)
    print(
        "[analysis] Wrote map-level quality, scale, speed, safety, "
        f"and convergence outputs for {len(outputs['family'])} families to "
        f"{(args.results / 'analysis').resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
