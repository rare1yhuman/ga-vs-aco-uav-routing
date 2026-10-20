# Statistical Report

The independent environment map is the inferential unit. Five paired optimizer seeds are summarized within each map before family tests.

Positive cost gain, physical-distance gain, speedup above one, and positive rank-biserial effects favor MGA.

## Route Cost and Distance

| family | label | maps | mga_median_cost | aco_median_cost | median_cost_gain_pct | gain_ci_low_pct | gain_ci_high_pct | cost_p_wilcoxon | cost_rank_biserial | wins | ties | losses | mga_median_physical_m | aco_median_physical_m | median_physical_gain_pct | physical_ci_low_pct | physical_ci_high_pct | physical_p_wilcoxon | physical_rank_biserial | cost_p_holm | physical_p_holm | cost_significant_0_05 | mga_median_cpu_s | aco_median_cpu_s | median_cpu_speedup | speedup_ci_low | speedup_ci_high | cpu_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 2D control | 10 | 9.81e+04 | 9.81e+04 | 0 | 0 | 0 | 1 | 0 | 0 | 10 | 0 | 9.81e+04 | 9.81e+04 | 0 | 0 | 0 | 1 | 0 | 1 | 1 | False | 25.77 | 82.89 | 3.167 | 3.069 | 3.296 | 0.01367 |
| S2 | 2D static | 10 | 1.035e+05 | 1.036e+05 | 0 | 0 | 0.09434 | 0.5 | 1 | 2 | 8 | 0 | 1.035e+05 | 1.036e+05 | 0 | 0 | 0.09434 | 0.5 | 1 | 1 | 1 | False | 26.31 | 83.9 | 3.179 | 3.061 | 3.296 | 0.01367 |
| S3 | 2D dynamic | 10 | 1.039e+05 | 1.039e+05 | 0 | 0 | 0.1984 | 0.75 | 0.3333 | 2 | 7 | 1 | 1.018e+05 | 1.023e+05 | 0 | 0 | 0.09074 | 0.5 | 1 | 1 | 1 | False | 25.93 | 82.49 | 3.118 | 3.027 | 3.274 | 0.01367 |
| S4 | 3D-aware static | 10 | 1.451e+05 | 1.451e+05 | 0 | 0 | 0 | 1 | 1 | 1 | 9 | 0 | 1.061e+05 | 1.061e+05 | 0 | 0 | 0 | 1 | 1 | 1 | 1 | False | 25.68 | 85.51 | 3.294 | 3.179 | 3.356 | 0.01367 |
| S5 | 3D-aware dynamic | 10 | 1.47e+05 | 1.478e+05 | 0.1405 | 0 | 0.6165 | 0.0625 | 1 | 5 | 5 | 0 | 1.061e+05 | 1.061e+05 | 0 | 0 | 0.7156 | 0.1875 | 0.7333 | 0.3125 | 0.9375 | False | 25.98 | 85.02 | 3.229 | 3.145 | 3.356 | 0.01367 |
| S6 | 3D-aware static scalability | 10 | 2.296e+05 | 2.35e+05 | 1.833 | 0.7878 | 2.596 | 0.003906 | 1 | 9 | 1 | 0 | 1.561e+05 | 1.621e+05 | 3.164 | 0.1951 | 5.155 | 0.01172 | 0.9333 | 0.02344 | 0.07031 | True | 127.4 | 411.4 | 3.147 | 3.011 | 3.445 | 0.01367 |
| S7 | 3D-aware dynamic scalability | 10 | 2.301e+05 | 2.396e+05 | 2.301 | 1.5 | 3.817 | 0.001953 | 1 | 10 | 0 | 0 | 1.582e+05 | 1.644e+05 | 3.166 | 1.284 | 5.254 | 0.003906 | 1 | 0.01367 | 0.02734 | True | 130.2 | 403.3 | 2.98 | 2.755 | 3.328 | 0.01367 |

## Scale Interaction

| comparison | small_family | large_family | maps | median_gain_increase_pct_points | increase_ci_low | increase_ci_high | wins | ties | losses | p_wilcoxon | rank_biserial | p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| static | S4 | S6 | 10 | 1.833 | 0.9896 | 2.364 | 9 | 1 | 0 | 0.003906 | 1 | 0.007812 |
| dynamic | S5 | S7 | 10 | 2.156 | 0.7369 | 3.249 | 9 | 0 | 1 | 0.009766 | 0.8909 | 0.009766 |

## CPU Time

| family | maps | mga_median_cpu_s | aco_median_cpu_s | median_speedup | speedup_ci_low | speedup_ci_high | cpu_p_wilcoxon | cpu_rank_biserial | cpu_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | 10 | 25.77 | 82.89 | 3.167 | 3.069 | 3.296 | 0.001953 | 1 | 0.01367 |
| S2 | 10 | 26.31 | 83.9 | 3.179 | 3.061 | 3.296 | 0.001953 | 1 | 0.01367 |
| S3 | 10 | 25.93 | 82.49 | 3.118 | 3.027 | 3.274 | 0.001953 | 1 | 0.01367 |
| S4 | 10 | 25.68 | 85.51 | 3.294 | 3.179 | 3.356 | 0.001953 | 1 | 0.01367 |
| S5 | 10 | 25.98 | 85.02 | 3.229 | 3.145 | 3.356 | 0.001953 | 1 | 0.01367 |
| S6 | 10 | 127.4 | 411.4 | 3.147 | 3.011 | 3.445 | 0.001953 | 1 | 0.01367 |
| S7 | 10 | 130.2 | 403.3 | 2.98 | 2.755 | 3.328 | 0.001953 | 1 | 0.01367 |

## Dynamic Safety and Exposure

| family | maps | mga_safety_rate | aco_safety_rate | median_map_safety_delta | safety_map_p_wilcoxon | safety_rank_biserial | mga_median_exposure | aco_median_exposure | median_exposure_reduction | exposure_p_wilcoxon | exposure_rank_biserial | mga_only_safe_pairs | aco_only_safe_pairs | mcnemar_descriptive_p | safety_p_holm | exposure_p_holm |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S3 | 10 | 0.1 | 0.24 | 0 | 0.5 | -1 | 6.5 | 7 | 0 | 1 | 0 | 0 | 7 | 0.01562 | 1 | 1 |
| S5 | 10 | 0.5 | 0.5 | 0 | 1 | 0 | 1 | 1 | 0 | 1 | 0.3333 | 0 | 0 | 1 | 1 | 1 |
| S7 | 10 | 0.2 | 0.1 | 0 | 0.75 | 0.4 | 6 | 6 | 2 | 0.209 | 0.4727 | 9 | 4 | 0.2668 | 1 | 0.627 |

McNemar values are descriptive run-level diagnostics because runs are clustered within maps. Map-level Wilcoxon tests are primary.

## Time to 1% Pooled-Map Target

| family | algo | runs | attained_runs | attainment_rate | conditional_median_target_cpu_s | conditional_median_target_wall_s | median_final_cpu_s | conditional_time_note |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| S1 | ACO | 50 | 50 | 1 | 0.8502 | 0.8522 | 84.04 | Median among attained runs; interpret with attainment_rate |
| S1 | MGA | 50 | 50 | 1 | 0.2829 | 0.283 | 25.64 | Median among attained runs; interpret with attainment_rate |
| S2 | ACO | 50 | 50 | 1 | 1.203 | 1.285 | 84.59 | Median among attained runs; interpret with attainment_rate |
| S2 | MGA | 50 | 50 | 1 | 0.2628 | 0.2817 | 25.92 | Median among attained runs; interpret with attainment_rate |
| S3 | ACO | 50 | 40 | 0.8 | 2.699 | 2.707 | 84.88 | Median among attained runs; interpret with attainment_rate |
| S3 | MGA | 50 | 50 | 1 | 0.3104 | 0.3105 | 26.15 | Median among attained runs; interpret with attainment_rate |
| S4 | ACO | 50 | 50 | 1 | 0.8637 | 0.8636 | 85.55 | Median among attained runs; interpret with attainment_rate |
| S4 | MGA | 50 | 50 | 1 | 0.2574 | 0.2584 | 25.67 | Median among attained runs; interpret with attainment_rate |
| S5 | ACO | 50 | 44 | 0.88 | 1.279 | 1.279 | 85.3 | Median among attained runs; interpret with attainment_rate |
| S5 | MGA | 50 | 50 | 1 | 0.2605 | 0.2606 | 26.04 | Median among attained runs; interpret with attainment_rate |
| S6 | ACO | 50 | 15 | 0.3 | 115.7 | 115.7 | 411.1 | Median among attained runs; interpret with attainment_rate |
| S6 | MGA | 50 | 49 | 0.98 | 3.255 | 3.256 | 126.5 | Median among attained runs; interpret with attainment_rate |
| S7 | ACO | 50 | 7 | 0.14 | 222.8 | 223 | 413.2 | Median among attained runs; interpret with attainment_rate |
| S7 | MGA | 50 | 49 | 0.98 | 3.52 | 3.52 | 129.6 | Median among attained runs; interpret with attainment_rate |

Conditional target times exclude non-attaining runs and must always be interpreted together with attainment rates.
