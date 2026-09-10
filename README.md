# PhishVN Edge

Standalone tools for exporting scikit-learn detectors to ONNX, benchmarking CPU
inference on Linux edge devices, and preparing the PhishVN Edge software article.

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

## Reproduce the article assets

```bash
make assets
make paper
```

`make assets` regenerates the confusion figure and benchmark table from the
aggregate snapshots in `results/`. These snapshots were transcribed from the
original study outputs; this operation does not rerun the hardware experiment
or recompute predictions from private test records. See [provenance](docs/PROVENANCE.md).

`make paper` requires pdfLaTeX and common LaTeX packages (natbib, booktabs,
tabularx, enumitem, caption, xcolor, hyperref, xurl). The Elsevier class and
its sources are bundled under `paper/`. Read the [compiled article](paper/main.pdf).

## Scope and limitations

- The runnable software covers scikit-learn export and ONNX Runtime CPU inference.
- [Jetson instructions](docs/JETSON_DEPLOY.md) describe on-device CPU benchmarking
  and the prerequisites for using an independently prepared TensorRT engine.
- TensorRT engine conversion, a scoring API, SMS forwarding, and gateway
  integrations are not implemented in this repository.
- Historical Jetson pilot values are reference results, not performance promises.
  FP16 detection accuracy, long thermal soaks, and external power validation remain pending.
- A public reproducible capsule and a software archive DOI have not been created.

## Development

```bash
make test
```

The integration check trains the synthetic model, exports it, compares sklearn
and ONNX predictions, and validates benchmark JSON. CI runs this on Linux.

## License and attribution

Code retains the original project's [MIT license](LICENSE).
Elsevier class files retain LPPL 1.3 or later; see [third-party notices](THIRD_PARTY_NOTICES.md).
The manuscript is an unpublished author draft. Dataset and manuscript rights
are not broadened by the code license. This repository contains no raw dataset.

Extracted from the [PhishVN project](https://github.com/vuthainguyen1602/phishvn).
See [CITATION.cff](CITATION.cff) for software attribution.
