# TUI 2.0: The Live Experiment Dashboard

Welcome to the **Terminal User Interface (TUI)** for the UAV Path Planning experiment. This dashboard turns your standard command line into a highly visual, live-updating mission control center.

Instead of waiting blindly for background logs, you can watch the MGA and ACO algorithms battle in real-time.

## Features

- **Live Trajectory Snake:** Watch the algorithms draw their best flight paths in real-time (`::`) routing through the No-Fly Zones.
- **Dynamic NFZ Animation:** If you run a dynamic trial (like Trial E or H), the red No-Fly Zone blocks will visually move across the terminal map as the simulation time ticks forward.
- **Algorithmic Telemetry Feed:** A scrolling log window showing live internal events from your CPU cores (e.g., watching the MGA trigger *Adaptive Mutation* or *Diversity Injection*).
- **Dual Progress Race:** Side-by-side progress bars racing to complete 50 runs.
- **Live Scoreboard:** Real-time updates of the best distance and energy found so far.
- **Terminal Pareto Front:** Powered by `plotext`, it draws a full scatter plot of the distance vs. energy trade-off right in your terminal window when the run finishes.

---

## Usage & Commands

You can run the dashboard from inside the `experiment/` directory:

```bash
cd "/home/fat32/orbit/Work/FinalYear/New Paper/GA_vs_ACO_Paper/experiment"
```

### 1. The Visual Demo Mode
Want to see how gorgeous the software looks without waiting 15 minutes for actual computations? Run the demo mode. It simulates a fast-paced race and animates the dynamic NFZs perfectly.

```bash
python cli_dashboard.py --demo
```

### 2. Running a Single Trial
To run real number-crunching for a specific trial and visualize it in real time:

```bash
python cli_dashboard.py --run-trial E --runs 50 --workers 4 --params config/paper_params.json --output /tmp/ga_tui_trial_E
```
Use `--workers 11` only when the machine is otherwise idle.

### 3. "Safe Mode" (Single CPU)
If you already have the main `run_experiment.py` script running in the background (using 11 cores), you can still watch the live dashboard by restricting it to just **1 core**. It will run slower, but your system will stay perfectly responsive!

```bash
# Safely watch dynamic Trial E compute using only 1 worker
python cli_dashboard.py --run-trial E --runs 50 --workers 1 --params config/paper_params.json --output /tmp/ga_tui_trial_E
```

### 4. Running a Specific Trial with N CPUs
You can dedicate a specific amount of processing power to the dashboard:

```bash
# Run Trial A using exactly 4 CPU cores and 5 runs for a quick check
python cli_dashboard.py --run-trial A --runs 5 --workers 4 --params config/paper_params.json --output /tmp/ga_tui_A_check
```

### 5. Running All Trials Visually
This runs the full paper trial matrix through the TUI and saves the same high-level
summary schema as the headless runner:

```bash
python cli_dashboard.py --run-all --runs 50 --workers 11 --params config/paper_params.json --output results
```

For paper data, prefer the headless runner first because it is easier to audit:

```bash
python run_experiment.py --trial all --runs 50 --workers 11 --params config/paper_params.json --output results --clean-output
```

## Saved Outputs

Real TUI runs write:

- `trial_<ID>/raw_results.csv`
- `trial_<ID>/summary.json`
- `master_results.csv`

Demo mode (`--demo`) is visual only and does not write paper results. For temporary TUI checks, write to `/tmp/...`. Keep `results/` as the only persistent paper result folder.

---

## Interpreting the Map

The 200x200 3D spatial grid is compressed into a crisp 50x50 block ANSI map:

- `▲` (Blue Triangle): **The Drone Depot** (Start/End point)
- `●` (Yellow Circle): **Delivery Points**
- `██` (Red Blocks): **No-Fly Zones (Buildings/Obstacles)**
- `::` (Bright Green Line): **The Current Best Flight Path** found by the algorithms.
- `·` (Grey Dots): Open airspace

Enjoy watching your algorithms work!
