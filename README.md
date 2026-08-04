# GA vs ACO: UAV Route Planning

Code for a research study comparing a Memetic Genetic Algorithm (MGA) against
Ant Colony Optimization (ACO) for dynamic, 3D-aware UAV route planning —
routing under altitude layers, no-fly zones, and moving obstacles, evaluated
under an equal compute/wall-clock budget.

Authors: Rishi Mandal, Shilpi Khanna, Santripti Bajpai
Dept. of IT, Shri Ramswaroop Memorial College of Engineering and Management,
Lucknow.

## Layout

```
experiment/
  ga.py, memetic_ga.py, ga_reference.py, memetic_ga_reference.py  # GA / MGA
  aco.py                                                          # ACO
  environment_3d.py, grid_gen.py, dynamic_nfz.py, energy_model.py # world model
  graph_builder.py, dijkstra.py, nsga2.py                         # routing/support
  metrics.py, statistical_tests.py                                # evaluation
  run_research_experiment.py, calibrate_protocol_v2.py            # experiment drivers
  validate_research_results.py, analyze_research_results.py       # post-processing
  generate_research_figures.py                                    # figure generation
  config/                                                          # protocol/param JSON
  tests/                                                           # pytest suite
requirements.txt
Makefile
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Running

```bash
make validate   # sanity-check a results directory
make analyze    # statistical analysis (bootstrap CIs, significance tests)
make figures    # regenerate result figures
```

or invoke the scripts directly, e.g.:

```bash
python experiment/run_research_experiment.py --help
```

Run the test suite with:

```bash
pytest experiment/tests
```

## Scope of this repo

This repository holds the algorithm implementations, experiment pipeline, and
tests only. Raw experiment results, calibration data, and the manuscript are
regenerable/maintained separately and are not tracked here.
