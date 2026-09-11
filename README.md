# PhishVN Edge

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22702706.svg)](https://doi.org/10.5281/zenodo.22702706)

Standalone tools for exporting scikit-learn detectors to ONNX and benchmarking
CPU and TensorRT GPU inference on Linux edge devices.

## Predict a URL with the real model

The repository now includes the trained **21-feature PhishVN HistGB detector**,
its offline URL feature extractor, and CPU/GPU inference commands:

```bash
python -m pip install -r requirements-predict.txt
python scripts/predict_url.py --url 'https://example.com'
```

Run these in a Python environment from the repository root. For setup and actual
Jetson GPU execution, follow [real-model instructions](docs/REAL_MODEL.md).
Read the [model card](artifacts/url_histgb21/README.md) for validation and limits.

## Quick start

On Linux or macOS with Python 3.11 or newer, clone and run the bundled synthetic
ONNX model. No training or export dependencies are required:

```bash
git clone https://github.com/vuthainguyen1602/phishvn-edge.git
cd phishvn-edge
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-demo.txt
make quickstart
```

The approximately 8 KB [dummy model](examples/artifacts/dummy.onnx) ships in Git
and release source archives. `make quickstart` writes `runs/quickstart.json` with
latency, throughput, and memory measurements. The timed throughput window is one
second; installation time depends on network and platform wheel availability.
The target is a first run in under five minutes on a supported workstation.
This synthetic classifier is **not a phishing detector** and does not reproduce
the historical Jetson pilot. See its [model card](examples/artifacts/README.md).

To exercise training and export as well, install the full dependencies:

```bash
python -m pip install -r requirements.txt
make demo
```

For your own trusted scikit-learn model with 9 float input features:

```bash
python scripts/export_edge.py --model models/your-model.joblib --dim 9 --out models/your-model.onnx
python scripts/bench_edge_ort.py --model models/your-model.onnx --dim 9 > runs/your-model.json
```

Create `runs/` first. Supply the actual feature dimension and preserve your
training feature order. The benchmark uses random numeric inputs; it measures
model inference, excluding URL parsing, feature extraction, and service latency.
Joblib files must come from a trusted source because loading them executes Python code.

## Reproduce the reported assets

```bash
make assets
```

This regenerates the confusion figure and the benchmark table into `results/`,
from the aggregate snapshots already in that directory. The snapshots were
transcribed from the original study outputs; the command does not rerun the
hardware experiment or recompute predictions from private test records. See
[provenance](docs/PROVENANCE.md). The manuscript that reports these numbers is
maintained outside this repository.

## Scope and limitations

- The software includes real URL inference on CPU and validated TensorRT FP32 GPU execution.
- [Jetson instructions](docs/JETSON_DEPLOY.md) describe on-device CPU benchmarking
  and the prerequisites for using an independently prepared TensorRT engine.
- TensorRT conversion is implemented for the bundled numeric binary HistGB model.
  A scoring API, SMS forwarding, and gateway integrations are not implemented.
- Historical Jetson pilot values are reference results, not performance promises.
  FP16 detection accuracy, long thermal soaks, and external power validation remain pending.
- Releases are archived to Zenodo, which mints the software DOI; `.zenodo.json` holds the
  record metadata. Cite the concept DOI [10.5281/zenodo.22702706](https://doi.org/10.5281/zenodo.22702706)
  for the software, or [10.5281/zenodo.22702707](https://doi.org/10.5281/zenodo.22702707) for the
  archived `v0.2.1` release specifically.

## Development

```bash
make test
```

The integration check trains the synthetic model, exports it, compares sklearn
and ONNX predictions, and validates benchmark JSON. CI runs this on Linux.

## License and attribution

Code retains the original project's [MIT license](LICENSE).
Dataset and manuscript rights are not broadened by the code license. This
repository contains no raw dataset.

Extracted from the [PhishVN project](https://github.com/vuthainguyen1602/phishvn).
See [CITATION.cff](CITATION.cff) for software attribution.
