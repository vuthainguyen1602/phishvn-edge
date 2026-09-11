# PhishVN URL HistGB: real research model

This is a trained binary URL classifier, not the synthetic quickstart model.
The source is the PhishVN `url_hgb21.joblib` HistGradientBoostingClassifier:
88 boosting iterations, 21 lexical URL features, classes benign (0) / phishing (1).
Code and exported model artifacts are provided under MIT. Dataset attribution:
https://doi.org/10.17632/b97hxbxtpd.4 . Raw training/test records are not bundled.

## Contract

Features follow `features.json`; extraction uses `scripts/url_features.py` and
an offline tldextract 5.3.1 suffix snapshot. HTTP/HTTPS schemes are excluded from
lexical features to avoid the original collection's scheme/label confound.
Only finite float32 numeric inputs are supported. Missing-value and categorical
inference are not supported by the tensor graph. A score strictly above 0.5 is
classified as phishing. Scores are not calibrated probabilities of real-world harm.

- `cpu.onnx`: ONNX tree ensemble followed by sigmoid, for ONNX Runtime CPU.
- `gpu.onnx`: the same trees expressed with tensor operations for TensorRT.
  The export fixes float32 split boundaries by rounding thresholds down when
  nearest rounding would incorrectly include an additional float32 value.
- GPU execution uses a locally built FP32 TensorRT engine, with TF32 disabled.
  FP16/INT8 have not been validated and are not the defaults.
- `manifest.json` records SHA-256 hashes. The trusted original joblib is not
  needed for inference. Re-export uses `scripts/export_histgb.py` and sklearn 1.4.2.

## Validation and limitations

Validated on 8,941 held-out PhishVN temporal URL records on the second Jetson Orin
Nano Super. CPU and TensorRT GPU predictions agree with the original sklearn
model given the same float32 feature vectors. The full URL-to-feature-to-score
path obtains accuracy 0.848339, phishing F1 0.854694, and ROC-AUC 0.963277 on this
split. See the aggregate JSON reports in `results/` for actual errors and timing.

The historical archive uses float64 features. Converting to float32 changes five
labels relative to that original evaluation. The current offline suffix snapshot
also differs from the archived features on three URLs (domain length, suffix
length, subdomain count), affecting two labels relative to the archived float32
features. These are documented preprocessing/version effects; new results must
not be substituted silently into the historical article tables.

Historical ONNX exports were replaced because they did not preserve the chosen
float32 sklearn contract. Six reserved/example-URL fixtures test extraction and
reference scores in CI. The full private test set remains outside this repo.

This research baseline misses some phishing URLs and flags some benign URLs.
It does not inspect page content, fetch URLs, check reputation, or block traffic.
Performance on future attacks or different datasets has not been established.
