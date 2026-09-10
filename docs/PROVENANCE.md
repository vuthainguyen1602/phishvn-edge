# Provenance

This repository is an independently versioned extraction of PhishVN Edge.
Source workspace base commit: `fc41059` in the original PhishVN repository.
The software article includes subsequent local manuscript edits.

- `scripts/export_edge.py` and `scripts/bench_edge_ort.py` originate from
  `scripts/studies/future_edge/`, with standalone error-handling and portability fixes.
- `scripts/make_edge_assets.py` is the standalone aggregate renderer. The original
  prediction-based renderer depends on private test splits and model files and is
  not the renderer shipped here.
- `results/pilot.csv` transcribes the three populated rows in
  `papers/future_edge/sections/06_results.tex`: Jetson Orin Nano pilot, 2026-07-22,
  CPU batch 1, 5,000 timed iterations, 30-second throughput window; GPU host latency
  from trtexec. Telemetry uses VDD_IN at 500 ms. Marginal energy subtracts idle power.
- `results/confusion.json` transcribes the counts displayed in
  `papers/future_edge/figures/fig_confusion.pdf`, with 8,941 test records per model.

Raw device logs, fitted research models, and private URL/SMS records are not included.
The pilot cannot be independently remeasured from these aggregates alone. The demo
is synthetic, and must not be substituted for the reported detector evaluation.
The CPU harness reports total board energy per inference when telemetry is available;
it does not estimate marginal energy without an independently measured idle baseline.

Local release validation used Python 3.14 on macOS/Apple Silicon; the exact
installed versions are recorded in `requirements-tested-macos-py314.txt`.
That file is an environment record, not a cross-platform lockfile. CI uses
Python 3.11 on Linux and installs the compatible ranges in `requirements.txt`.
