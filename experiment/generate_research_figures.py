"""Generate data-first IEEE figures from validated research outputs."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd

from dynamic_nfz import render_dynamic_nfzs
from grid_gen import CELL_DEPOT, CELL_DELIVERY, CELL_NFZ
from research_protocol import SCENARIO_FAMILIES, build_research_environment


COLORS = {
    "MGA": "#0072B2",
    "ACO": "#D55E00",
    "positive": "#009E73",
    "negative": "#CC79A7",
    "neutral": "#5F6B6D",
    "grid": "#D9E0E2",
    "nfz": "#B2182B",
    "moving": "#E69F00",
}
FAMILIES = tuple(SCENARIO_FAMILIES)
ALGORITHMS = ("MGA", "ACO")


def _configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 7.5,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.8,
            "ytick.labelsize": 6.8,
            "legend.fontsize": 6.8,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.25,
            "lines.markersize": 4.0,
        }
    )


def _panel_label(axis: plt.Axes, label: str) -> None:
    axis.text(
        -0.17,
        1.04,
        label,
        transform=axis.transAxes,
        fontsize=8,
        fontweight="bold",
        va="bottom",
    )


def _save(
    fig: plt.Figure,
    output_dir: Path,
    stem: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        output_dir / f"{stem}.pdf",
        bbox_inches="tight",
        pad_inches=0.03,
    )
    fig.savefig(
        output_dir / f"{stem}.png",
        dpi=300,
        bbox_inches="tight",
        pad_inches=0.03,
    )
    plt.close(fig)


def _write_figure_data(
    frame: pd.DataFrame,
    data_dir: Path,
    filename: str,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(data_dir / filename, index=False)


def _box(
    axis: plt.Axes,
    xy: Tuple[float, float],
    width: float,
    height: float,
    text: str,
    *,
    facecolor: str = "#F7F9FA",
    edgecolor: str = "#546E7A",
    fontsize: float = 6.8,
) -> None:
    box = patches.FancyBboxPatch(
        xy,
        width,
        height,
        boxstyle="round,pad=0.012,rounding_size=0.015",
        facecolor=facecolor,
        edgecolor=edgecolor,
        linewidth=0.8,
    )
    axis.add_patch(box)
    axis.text(
        xy[0] + width / 2,
        xy[1] + height / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
    )


def _arrow(
    axis: plt.Axes,
    start: Tuple[float, float],
    end: Tuple[float, float],
) -> None:
    axis.annotate(
        "",
        xy=end,
        xytext=start,
        arrowprops={
            "arrowstyle": "-|>",
            "color": "#455A64",
            "linewidth": 0.8,
            "shrinkA": 2,
            "shrinkB": 2,
        },
    )


def _protocol_overview(
    output_dir: Path,
    data_dir: Path,
) -> None:
    # Panel (a): shared environment
    fig = plt.figure(figsize=(2.55, 2.25))
    environment_axis = fig.add_subplot(111)
    environment_axis.set_xlim(0, 1)
    environment_axis.set_ylim(0, 1)
    environment_axis.axis("off")
    _panel_label(environment_axis, "(a)")
    for coordinate in np.linspace(0.08, 0.92, 8):
        environment_axis.plot(
            [0.08, 0.92],
            [coordinate, coordinate],
            color=COLORS["grid"],
            linewidth=0.35,
        )
        environment_axis.plot(
            [coordinate, coordinate],
            [0.08, 0.92],
            color=COLORS["grid"],
            linewidth=0.35,
        )
    environment_axis.add_patch(
        patches.Rectangle(
            (0.16, 0.56),
            0.23,
            0.17,
            facecolor=COLORS["nfz"],
            alpha=0.82,
        )
    )
    environment_axis.add_patch(
        patches.Rectangle(
            (0.58, 0.23),
            0.22,
            0.15,
            facecolor=COLORS["moving"],
            alpha=0.72,
            hatch="//",
        )
    )
    environment_axis.arrow(
        0.61,
        0.43,
        0.16,
        0.08,
        width=0.006,
        head_width=0.04,
        color=COLORS["moving"],
        length_includes_head=True,
    )
    environment_axis.scatter(
        [0.12],
        [0.12],
        marker="s",
        color="#202124",
        s=28,
        zorder=4,
    )
    environment_axis.scatter(
        [0.46, 0.72, 0.86],
        [0.76, 0.62, 0.84],
        marker="o",
        color=COLORS["MGA"],
        s=24,
        zorder=4,
    )
    environment_axis.plot(
        [0.12, 0.46, 0.72, 0.86, 0.12],
        [0.12, 0.76, 0.62, 0.84, 0.12],
        color=COLORS["MGA"],
        linewidth=1.3,
    )
    environment_axis.text(
        0.5,
        0.01,
        "static NFZ + moving NFZ + required altitude",
        ha="center",
        va="bottom",
        fontsize=6.4,
    )
    _save(fig, output_dir, "fig1_problem_protocol_a")

    # Panel (b): controlled algorithm comparison
    fig = plt.figure(figsize=(2.55, 2.25))
    model_axis = fig.add_subplot(111)
    model_axis.set_xlim(0, 1)
    model_axis.set_ylim(0, 1)
    model_axis.axis("off")
    _panel_label(model_axis, "(b)")
    _box(
        model_axis,
        (0.02, 0.59),
        0.38,
        0.25,
        "MGA\npermutation\n+ local search",
        facecolor="#E6F3FA",
        edgecolor=COLORS["MGA"],
        fontsize=6.3,
    )
    _box(
        model_axis,
        (0.60, 0.59),
        0.38,
        0.25,
        "ACO\npheromone\ntour construction",
        facecolor="#FCEDE5",
        edgecolor=COLORS["ACO"],
        fontsize=6.3,
    )
    _box(
        model_axis,
        (0.31, 0.34),
        0.38,
        0.16,
        "same time-indexed\nroute objective",
        facecolor="#F3F5F6",
    )
    _box(
        model_axis,
        (0.13, 0.08),
        0.74,
        0.14,
        "100,000 constructions per run",
        facecolor="#EDF7F2",
        edgecolor=COLORS["positive"],
        fontsize=6.3,
    )
    _arrow(model_axis, (0.21, 0.59), (0.42, 0.50))
    _arrow(model_axis, (0.79, 0.59), (0.58, 0.50))
    _arrow(model_axis, (0.50, 0.34), (0.50, 0.22))
    _save(fig, output_dir, "fig1_problem_protocol_b")

    # Panel (c): held-out, paired evaluation protocol
    fig = plt.figure(figsize=(3.05, 2.25))
    protocol_axis = fig.add_subplot(111)
    protocol_axis.set_xlim(0, 1)
    protocol_axis.set_ylim(0, 1)
    protocol_axis.axis("off")
    _panel_label(protocol_axis, "(c)")
    labels = [
        ("480 calibration runs", "2 held-out maps\n24 configurations / algorithm"),
        ("freeze configuration", "disjoint final seeds\nSHA-256 recorded"),
        ("700 final runs", "7 families x 10 maps\n5 paired seeds x 2 algorithms"),
        ("map-level inference", "paired tests + Holm\nhierarchical 95% intervals"),
    ]
    y_positions = [0.76, 0.54, 0.32, 0.10]
    colors = ["#F7F9FA", "#F7F9FA", "#EDF7F2", "#EDF7F2"]
    for index, ((heading, detail), y, color) in enumerate(
        zip(labels, y_positions, colors)
    ):
        _box(
            protocol_axis,
            (0.07, y),
            0.86,
            0.15,
            f"{heading}\n{detail}",
            facecolor=color,
            fontsize=6.5,
        )
        if index < len(labels) - 1:
            _arrow(
                protocol_axis,
                (0.50, y),
                (0.50, y_positions[index + 1] + 0.15),
            )
    _save(fig, output_dir, "fig1_problem_protocol_c")

    protocol_data = pd.DataFrame(
        [
            {
                "stage": "calibration",
                "records": 480,
                "maps": 2,
                "optimizer_seeds": 5,
                "configurations_per_algorithm": 24,
            },
            {
                "stage": "final",
                "records": 700,
                "maps": 70,
                "optimizer_seeds": 5,
                "configurations_per_algorithm": 1,
            },
        ]
    )
    _write_figure_data(protocol_data, data_dir, "fig1_protocol.csv")


def _quality_and_scale(
    family: pd.DataFrame,
    scale_maps: pd.DataFrame,
    scale_summary: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    # Panel (a): family forest plot
    fig, forest = plt.subplots(figsize=(3.35, 3.1))
    ordered = family.set_index("family").loc[list(reversed(FAMILIES))]
    positions = np.arange(len(ordered))
    gains = ordered["median_cost_gain_pct"].to_numpy()
    lower = gains - ordered["gain_ci_low_pct"].to_numpy()
    upper = ordered["gain_ci_high_pct"].to_numpy() - gains
    significant = ordered["cost_p_holm"].to_numpy() < 0.05
    colors = np.where(significant, COLORS["positive"], COLORS["neutral"])
    for y, gain, low, high, color in zip(
        positions,
        gains,
        lower,
        upper,
        colors,
    ):
        forest.errorbar(
            gain,
            y,
            xerr=np.asarray([[low], [high]]),
            fmt="o",
            color=color,
            ecolor=color,
            capsize=2.4,
            linewidth=1.1,
        )
    forest.axvline(0, color="#202124", linewidth=0.8)
    forest.set_yticks(positions, ordered.index)
    forest.set_xlabel("MGA generalized-cost reduction (%)")
    forest.set_ylabel("Scenario family")
    forest.grid(axis="x", color=COLORS["grid"], linewidth=0.5)
    forest.text(
        0.98,
        0.96,
        "green: Holm $p<0.05$",
        transform=forest.transAxes,
        ha="right",
        va="top",
        fontsize=6.3,
        color=COLORS["positive"],
    )
    _panel_label(forest, "(a)")
    _save(fig, output_dir, "fig2_quality_scale_a")

    # Panel (b): scale interaction
    fig, scale_axis = plt.subplots(figsize=(3.35, 3.1))
    layout = {
        "static": (0.0, 0.9, "Static"),
        "dynamic": (1.75, 2.65, "Dynamic"),
    }
    rng = np.random.default_rng(20260729)
    for comparison, group in scale_maps.groupby("comparison", sort=False):
        left, right, title = layout[comparison]
        for row in group.itertuples():
            jitter = rng.normal(0, 0.012)
            scale_axis.plot(
                [left + jitter, right + jitter],
                [row.small_gain_pct, row.large_gain_pct],
                color="#90A4AE",
                alpha=0.65,
                linewidth=0.75,
            )
            scale_axis.scatter(
                [left + jitter, right + jitter],
                [row.small_gain_pct, row.large_gain_pct],
                color=[COLORS["neutral"], COLORS["positive"]],
                s=13,
                zorder=3,
            )
        summary = scale_summary[
            scale_summary["comparison"] == comparison
        ].iloc[0]
        scale_axis.text(
            (left + right) / 2,
            max(group["large_gain_pct"].max(), group["small_gain_pct"].max())
            + 0.28,
            (
                f"{title}: +{summary.median_gain_increase_pct_points:.2f} pp\n"
                f"Holm $p={summary.p_holm:.4f}$"
            ),
            ha="center",
            va="bottom",
            fontsize=6.5,
        )
    scale_axis.axhline(0, color="#202124", linewidth=0.8)
    scale_axis.set_xticks(
        [0.0, 0.9, 1.75, 2.65],
        ["20 stops", "50 stops", "20 stops", "50 stops"],
    )
    scale_axis.set_ylabel("MGA cost reduction (%)")
    scale_axis.set_ylim(
        min(-0.7, scale_maps[["small_gain_pct", "large_gain_pct"]].min().min() - 0.3),
        scale_maps[["small_gain_pct", "large_gain_pct"]].max().max() + 1.0,
    )
    scale_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.5)
    _panel_label(scale_axis, "(b)")
    _save(fig, output_dir, "fig2_quality_scale_b")

    _write_figure_data(
        family,
        data_dir,
        "fig2_family_effects.csv",
    )
    _write_figure_data(
        scale_maps,
        data_dir,
        "fig2_scale_pairs.csv",
    )


def _convergence_and_attainment(
    convergence: pd.DataFrame,
    attainment: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    # Panels (a)-(b): median regret to pooled map best, per family
    for family, panel, stem_suffix in (
        ("S6", "(a)", "a"),
        ("S7", "(b)", "b"),
    ):
        fig, axis = plt.subplots(figsize=(2.65, 2.45))
        family_data = convergence[convergence["family"] == family]
        for algorithm in ALGORITHMS:
            group = family_data[family_data["algo"] == algorithm]
            axis.plot(
                group["evaluations"] / 1000.0,
                group["median_regret_pct"],
                color=COLORS[algorithm],
                label=algorithm,
            )
            axis.fill_between(
                group["evaluations"] / 1000.0,
                group["ci_low_pct"],
                group["ci_high_pct"],
                color=COLORS[algorithm],
                alpha=0.13,
                linewidth=0,
            )
        axis.set_xlabel("Candidate constructions ($\\times 10^3$)")
        axis.set_ylabel("Regret to pooled map best (%)")
        axis.set_xlim(0, 100)
        axis.set_ylim(bottom=0)
        axis.grid(color=COLORS["grid"], linewidth=0.45)
        _panel_label(axis, panel)
        if family == "S6":
            axis.legend(frameon=False, loc="upper right")
        _save(fig, output_dir, f"fig3_convergence_attainment_{stem_suffix}")

    # Panel (c): target-attainment fraction
    fig, target_axis = plt.subplots(figsize=(2.95, 2.45))
    styles = {"S6": "-", "S7": "--"}
    for family in ("S6", "S7"):
        for algorithm in ALGORITHMS:
            group = attainment[
                (attainment["family"] == family)
                & (attainment["algo"] == algorithm)
            ]
            target_axis.plot(
                group["cpu_time_s"].clip(lower=0.03),
                100.0 * group["attainment_fraction"],
                color=COLORS[algorithm],
                linestyle=styles[family],
                label=f"{algorithm}-{family}",
            )
    target_axis.set_xscale("log")
    target_axis.set_xlim(left=0.03)
    target_axis.set_ylim(0, 103)
    target_axis.set_xlabel("CPU time to 1% target (s)")
    target_axis.set_ylabel("Runs attaining target (%)")
    target_axis.grid(color=COLORS["grid"], linewidth=0.45)
    target_axis.legend(frameon=False, ncol=1, loc="lower right")
    _panel_label(target_axis, "(c)")
    _save(fig, output_dir, "fig3_convergence_attainment_c")

    _write_figure_data(
        convergence[convergence["family"].isin(["S6", "S7"])],
        data_dir,
        "fig3_convergence.csv",
    )
    _write_figure_data(
        attainment[attainment["family"].isin(["S6", "S7"])],
        data_dir,
        "fig3_attainment.csv",
    )


def _runtime_advantage(
    maps: pd.DataFrame,
    speed: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    # Panel (a): map-median CPU time, MGA vs ACO
    fig, runtime_axis = plt.subplots(figsize=(2.8, 2.45))
    sizes = maps["n_deliveries"].map({20: 22, 50: 36}).to_numpy()
    colors = maps["n_deliveries"].map(
        {20: COLORS["neutral"], 50: COLORS["positive"]}
    )
    runtime_axis.scatter(
        maps["mga_median_cpu_s"],
        maps["aco_median_cpu_s"],
        s=sizes,
        c=colors,
        alpha=0.74,
        edgecolor="white",
        linewidth=0.35,
    )
    low = float(
        min(maps["mga_median_cpu_s"].min(), maps["aco_median_cpu_s"].min())
    )
    high = float(
        max(maps["mga_median_cpu_s"].max(), maps["aco_median_cpu_s"].max())
    )
    runtime_axis.plot(
        [low, high],
        [low, high],
        color="#202124",
        linewidth=0.8,
    )
    runtime_axis.set_xscale("log")
    runtime_axis.set_yscale("log")
    runtime_axis.set_xlim(low * 0.8, high * 1.2)
    runtime_axis.set_ylim(low * 0.8, high * 1.2)
    runtime_axis.xaxis.set_major_locator(FixedLocator([30, 100, 300]))
    runtime_axis.xaxis.set_major_formatter(
        FixedFormatter(["30", "100", "300"])
    )
    runtime_axis.yaxis.set_major_locator(FixedLocator([30, 100, 300]))
    runtime_axis.yaxis.set_major_formatter(
        FixedFormatter(["30", "100", "300"])
    )
    runtime_axis.xaxis.set_minor_formatter(NullFormatter())
    runtime_axis.yaxis.set_minor_formatter(NullFormatter())
    runtime_axis.set_xlabel("MGA map-median CPU time (s)")
    runtime_axis.set_ylabel("ACO map-median CPU time (s)")
    runtime_axis.grid(color=COLORS["grid"], linewidth=0.45)
    runtime_axis.legend(
        handles=[
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=COLORS["neutral"],
                label="20 stops",
            ),
            Line2D(
                [],
                [],
                marker="o",
                linestyle="",
                color=COLORS["positive"],
                label="50 stops",
            ),
        ],
        frameon=False,
        loc="upper left",
    )
    runtime_axis.text(
        0.98,
        0.04,
        f"median {maps['cpu_speedup'].median():.2f}$\\times$",
        transform=runtime_axis.transAxes,
        ha="right",
        fontsize=6.5,
    )
    _panel_label(runtime_axis, "(a)")
    _save(fig, output_dir, "fig4_runtime_advantage_a")

    # Panel (b): family-median CPU time, MGA vs ACO
    fig, cpu_axis = plt.subplots(figsize=(2.8, 2.45))
    positions = np.arange(len(speed))
    width = 0.36
    cpu_axis.bar(
        positions - width / 2,
        speed["mga_median_cpu_s"],
        width,
        color=COLORS["MGA"],
        label="MGA",
    )
    cpu_axis.bar(
        positions + width / 2,
        speed["aco_median_cpu_s"],
        width,
        color=COLORS["ACO"],
        label="ACO",
    )
    cpu_axis.set_xticks(positions, speed["family"])
    cpu_axis.set_yscale("log")
    cpu_axis.set_ylim(15, 600)
    cpu_axis.set_ylabel("Map-median CPU time (s)")
    cpu_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    cpu_axis.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(cpu_axis, "(b)")
    _save(fig, output_dir, "fig4_runtime_advantage_b")

    # Panel (c): ACO/MGA CPU ratio
    fig, speed_axis = plt.subplots(figsize=(2.5, 2.45))
    y_positions = np.arange(len(speed))
    x_error = np.vstack(
        [
            speed["median_speedup"] - speed["speedup_ci_low"],
            speed["speedup_ci_high"] - speed["median_speedup"],
        ]
    )
    speed_axis.errorbar(
        speed["median_speedup"],
        y_positions,
        xerr=x_error,
        fmt="o",
        color=COLORS["positive"],
        ecolor=COLORS["positive"],
        capsize=2.5,
        markersize=4,
        linewidth=1,
    )
    speed_axis.axvline(1, color="#202124", linewidth=0.8)
    speed_axis.set_yticks(y_positions, speed["family"])
    speed_axis.invert_yaxis()
    speed_axis.set_xlim(0.8, 3.75)
    speed_axis.set_xlabel("ACO / MGA CPU ratio")
    speed_axis.set_ylabel("Scenario family")
    speed_axis.grid(axis="x", color=COLORS["grid"], linewidth=0.45)
    speed_axis.text(
        0.04,
        0.04,
        "95% map-bootstrap CI",
        transform=speed_axis.transAxes,
        ha="left",
        fontsize=6.5,
    )
    _panel_label(speed_axis, "(c)")
    _save(fig, output_dir, "fig4_runtime_advantage_c")

    _write_figure_data(
        maps[
            [
                "family",
                "environment_seed",
                "n_deliveries",
                "mga_median_cpu_s",
                "aco_median_cpu_s",
                "cpu_speedup",
            ]
        ],
        data_dir,
        "fig4_runtime.csv",
    )
    _write_figure_data(speed, data_dir, "fig4_speed_summary.csv")


def _supplementary_safety(
    maps: pd.DataFrame,
    safety: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.15, 2.5),
        gridspec_kw={"wspace": 0.34},
    )
    safety_axis = axes[0]
    positions = np.arange(len(safety))
    width = 0.34
    safety_axis.bar(
        positions - width / 2,
        100.0 * safety["mga_safety_rate"],
        width,
        color=COLORS["MGA"],
        label="MGA",
    )
    safety_axis.bar(
        positions + width / 2,
        100.0 * safety["aco_safety_rate"],
        width,
        color=COLORS["ACO"],
        label="ACO",
    )
    safety_axis.set_xticks(positions, safety["family"])
    safety_axis.set_ylim(0, 100)
    safety_axis.set_ylabel("Zero-exposure runs (%)")
    safety_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    safety_axis.legend(frameon=False)
    _panel_label(safety_axis, "(a)")

    exposure_axis = axes[1]
    dynamic_maps = maps[maps["family"].isin(["S3", "S5", "S7"])]
    rng = np.random.default_rng(20260729)
    for index, family in enumerate(("S3", "S5", "S7")):
        values = dynamic_maps[
            dynamic_maps["family"] == family
        ]["exposure_reduction"].to_numpy()
        jitter = rng.normal(0, 0.045, size=len(values))
        exposure_axis.scatter(
            np.full(len(values), index) + jitter,
            values,
            s=18,
            color="#607D8B",
            alpha=0.75,
        )
        exposure_axis.plot(
            [index - 0.18, index + 0.18],
            [np.median(values), np.median(values)],
            color=COLORS["positive"],
            linewidth=1.7,
        )
    exposure_axis.axhline(0, color="#202124", linewidth=0.8)
    exposure_axis.set_xticks(range(3), ["S3", "S5", "S7"])
    exposure_axis.set_ylabel("ACO minus MGA exposure cells")
    exposure_axis.grid(axis="y", color=COLORS["grid"], linewidth=0.45)
    _panel_label(exposure_axis, "(b)")
    fig.subplots_adjust(left=0.08, right=0.99, top=0.94, bottom=0.2)
    _save(fig, output_dir, "supp_safety_exposure")
    _write_figure_data(safety, data_dir, "fig4_safety.csv")
    _write_figure_data(
        dynamic_maps[
            [
                "family",
                "environment_seed",
                "mga_median_exposure",
                "aco_median_exposure",
                "exposure_reduction",
            ]
        ],
        data_dir,
        "fig4_exposure.csv",
    )


def _path_cells(
    chromosome: Sequence[int],
    paths: Dict[Tuple[int, int], Sequence[Tuple[int, int]]],
) -> List[Tuple[int, int]]:
    order = [0] + [int(gene) + 1 for gene in chromosome] + [0]
    route: List[Tuple[int, int]] = []
    for source, target in zip(order, order[1:]):
        segment = list(paths[(source, target)])
        route.extend(segment if not route else segment[1:])
    return route


def _representative_s7(
    frame: pd.DataFrame,
    maps: pd.DataFrame,
) -> Tuple[int, pd.Series, pd.Series]:
    family_maps = maps[maps["family"] == "S7"].copy()
    family_median = float(family_maps["cost_gain_pct"].median())
    family_maps["distance_to_median"] = (
        family_maps["cost_gain_pct"] - family_median
    ).abs()
    selected_map = family_maps.sort_values(
        ["distance_to_median", "environment_seed"]
    ).iloc[0]
    environment_seed = int(selected_map["environment_seed"])
    map_runs = frame[
        (frame["family"] == "S7")
        & (frame["environment_seed"] == environment_seed)
    ]
    selections: Dict[str, pd.Series] = {}
    for algorithm in ALGORITHMS:
        runs = map_runs[map_runs["algo"] == algorithm].copy()
        median_cost = float(runs["generalized_cost"].median())
        runs["distance_to_median"] = (
            runs["generalized_cost"] - median_cost
        ).abs()
        selections[algorithm] = runs.sort_values(
            ["distance_to_median", "optimizer_seed"]
        ).iloc[0]
    return environment_seed, selections["MGA"], selections["ACO"]


def _plot_route_map(
    axis: plt.Axes,
    grid: np.ndarray,
    route: Sequence[Tuple[int, int]],
    dynamic_grid: np.ndarray,
    algorithm: str,
) -> None:
    display = np.zeros_like(grid, dtype=int)
    display[grid == CELL_NFZ] = 1
    moving_mask = (dynamic_grid == CELL_NFZ) & (grid != CELL_NFZ)
    display[moving_mask] = 2
    display[grid == CELL_DELIVERY] = 3
    display[grid == CELL_DEPOT] = 4
    cmap = ListedColormap(
        ["#FAFAFA", COLORS["nfz"], COLORS["moving"], "#5E9CCB", "#202124"]
    )
    axis.imshow(
        display,
        cmap=cmap,
        vmin=0,
        vmax=4,
        origin="upper",
        interpolation="nearest",
    )
    route_array = np.asarray(route)
    axis.plot(
        route_array[:, 1],
        route_array[:, 0],
        color=COLORS[algorithm],
        linewidth=0.75,
    )
    axis.set_xticks([])
    axis.set_yticks([])


def _representative_route(
    frame: pd.DataFrame,
    maps: pd.DataFrame,
    output_dir: Path,
    data_dir: Path,
) -> None:
    environment_seed, mga_run, aco_run = _representative_s7(frame, maps)
    environment = build_research_environment(
        "S7",
        SCENARIO_FAMILIES["S7"],
        environment_seed,
    )
    dynamic_grid = render_dynamic_nfzs(
        environment["grid"],
        environment["dynamic_nfzs"],
        t=400,
    )
    routes = {
        "MGA": _path_cells(
            json.loads(mga_run["best_path"]),
            environment["paths"],
        ),
        "ACO": _path_cells(
            json.loads(aco_run["best_path"]),
            environment["paths"],
        ),
    }

    corner_labels = {"MGA": "(a)", "ACO": "(b)"}
    corner_stems = {"MGA": "a", "ACO": "b"}
    for algorithm in ALGORITHMS:
        fig, axis = plt.subplots(figsize=(2.35, 2.25))
        _plot_route_map(
            axis,
            environment["grid"],
            routes[algorithm],
            dynamic_grid,
            algorithm,
        )
        axis.text(
            0.01,
            0.99,
            corner_labels[algorithm],
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=8,
            fontweight="bold",
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 1.0},
        )
        _save(
            fig,
            output_dir,
            f"fig5_representative_route_{corner_stems[algorithm]}",
        )

    # Panel (c): required-altitude profile
    fig, profile_axis = plt.subplots(figsize=(2.9, 2.25))
    required = environment["env3d"].required_altitude_map()
    profile_rows: List[Dict[str, float]] = []
    for algorithm in ALGORITHMS:
        route = routes[algorithm]
        altitudes = np.asarray(
            [
                max(0, int(required[row, column])) * 10.0
                for row, column in route
            ],
            dtype=float,
        )
        distance_km = np.arange(len(route), dtype=float) * 0.1
        profile_axis.plot(
            distance_km,
            altitudes,
            color=COLORS[algorithm],
            label=algorithm,
        )
        for distance, altitude in zip(distance_km, altitudes):
            profile_rows.append(
                {
                    "algorithm": algorithm,
                    "distance_along_path_km": distance,
                    "required_altitude_m": altitude,
                }
            )
    profile_axis.set_xlabel("Distance along realized grid path (km)")
    profile_axis.set_ylabel("Required altitude (m)")
    profile_axis.grid(color=COLORS["grid"], linewidth=0.45)
    profile_axis.legend(frameon=False)
    _panel_label(profile_axis, "(c)")
    _save(fig, output_dir, "fig5_representative_route_c")

    selected = pd.DataFrame(
        [
            {
                "family": "S7",
                "environment_seed": environment_seed,
                "algorithm": algorithm,
                "optimizer_seed": int(run["optimizer_seed"]),
                "generalized_cost": float(run["generalized_cost"]),
                "physical_distance_m": float(run["physical_distance_m"]),
                "exposure_cells": float(run["exposure_cells"]),
                "selection_rule": (
                    "map closest to S7 median cost gain; run closest to "
                    "algorithm map-median cost"
                ),
            }
            for algorithm, run in (("MGA", mga_run), ("ACO", aco_run))
        ]
    )
    _write_figure_data(selected, data_dir, "fig5_selected_runs.csv")
    _write_figure_data(
        pd.DataFrame(profile_rows),
        data_dir,
        "fig5_altitude_profiles.csv",
    )


def _supplementary_storyboard(
    frame: pd.DataFrame,
    maps: pd.DataFrame,
    output_dir: Path,
) -> None:
    environment_seed, mga_run, _ = _representative_s7(frame, maps)
    environment = build_research_environment(
        "S7",
        SCENARIO_FAMILIES["S7"],
        environment_seed,
    )
    route = _path_cells(
        json.loads(mga_run["best_path"]),
        environment["paths"],
    )
    route_array = np.asarray(route)
    cmap = ListedColormap(
        ["#FAFAFA", COLORS["nfz"], "#5E9CCB", "#202124"]
    )
    fig, axes = plt.subplots(
        1,
        4,
        figsize=(7.15, 1.95),
        sharex=True,
        sharey=True,
    )
    for axis, time_step in zip(axes, (0, 400, 800, 1200)):
        grid = render_dynamic_nfzs(
            environment["grid"],
            environment["dynamic_nfzs"],
            t=time_step,
        )
        axis.imshow(grid, cmap=cmap, vmin=0, vmax=3, origin="upper")
        axis.plot(
            route_array[:, 1],
            route_array[:, 0],
            color=COLORS["MGA"],
            linewidth=0.75,
        )
        axis.set_title(f"$t={time_step}$")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        f"S7 seed {environment_seed}: moving-NFZ storyboard and MGA route",
        y=0.99,
        fontsize=8,
    )
    fig.subplots_adjust(left=0.02, right=0.99, top=0.82, bottom=0.03, wspace=0.05)
    _save(fig, output_dir, "supp_dynamic_storyboard")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("../manuscript/figures"),
    )
    parser.add_argument(
        "--supplement-output",
        type=Path,
        default=Path("../manuscript/supplement_figures"),
    )
    args = parser.parse_args()
    _configure_style()
    frame = pd.read_csv(args.results / "raw_results.csv")
    analysis = args.results / "analysis"
    maps = pd.read_csv(analysis / "map_summary.csv")
    family = pd.read_csv(analysis / "family_summary.csv")
    speed = pd.read_csv(analysis / "speed_summary.csv")
    safety = pd.read_csv(analysis / "safety_summary.csv")
    scale_maps = pd.read_csv(analysis / "scale_interaction_maps.csv")
    scale_summary = pd.read_csv(analysis / "scale_interaction_summary.csv")
    convergence = pd.read_csv(analysis / "convergence_summary.csv")
    attainment = pd.read_csv(analysis / "attainment_curves.csv")
    data_dir = analysis / "figure_data"

    _protocol_overview(args.output, data_dir)
    _quality_and_scale(
        family,
        scale_maps,
        scale_summary,
        args.output,
        data_dir,
    )
    _convergence_and_attainment(
        convergence,
        attainment,
        args.output,
        data_dir,
    )
    _runtime_advantage(
        maps,
        speed,
        args.output,
        data_dir,
    )
    _representative_route(
        frame,
        maps,
        args.output,
        data_dir,
    )
    _supplementary_storyboard(
        frame,
        maps,
        args.supplement_output,
    )
    _supplementary_safety(
        maps,
        safety,
        args.supplement_output,
        data_dir,
    )
    print(
        "[figures] Wrote five main figures as single panels, two "
        f"supplementary figures, and source CSVs to {args.output.resolve()}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
