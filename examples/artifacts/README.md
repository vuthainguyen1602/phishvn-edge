# Synthetic dummy model

`dummy.onnx` is a bundled RandomForestClassifier trained only on synthetic numeric
data from scikit-learn `make_classification`: 256 samples, 9 features, 5 informative
features, seed 42. The forest has 8 trees, maximum depth 4, seed 42. Labels 0 and 1
are artificial; no URL, message, person, or research training record is included.
The artifact is MIT-licensed and intended only to test installation and inference.
It must not be used to assess phishing detection quality or deployment safety.

Input: float32 tensor `input`, shape `[batch_size, 9]`.
Outputs: class labels and class-probability maps. ONNX IR 10; main opset 9,
`ai.onnx.ml` opset 1. The artifact is self-contained (no external tensor files).
`manifest.json` records its SHA-256 and size.

Rebuild from the repository root with the full requirements installed:

```bash
python examples/train_demo.py
python scripts/export_edge.py --model models/demo.joblib --dim 9 --out models/demo.onnx --bench 100
```

The bundled file was produced with the versions recorded in
`docs/requirements-tested-macos-py314.txt`. Rebuilding under other dependency
versions can change binary serialization; validate predictions before replacing it.

To run the bundled file without training: `make quickstart`.
The model is included in GitHub release source archives and attached separately
to release v0.1.1. A Zenodo deposit has not yet been created.
