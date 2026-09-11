# Use the real URL detector

## CPU: any supported Linux/macOS workstation or Jetson

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-predict.txt
python scripts/predict_url.py --url 'https://example.com'
```

The model ships in `artifacts/url_histgb21/`. No training or download of private
records is required. The command emits JSON with a label, phishing score, and
threshold. It only parses the supplied string; it never visits the URL.

## GPU: tested Jetson configuration

Validated on Orin Nano Super, L4T 36.5.0, Python 3.10, TensorRT 10.3, CUDA 12.6.
The Jetson must have the NVIDIA repository matching its installed L4T version.
Install the following runtime components if missing:

- `libnvinfer-bin`, `python3-libnvinfer`
- `cuda-cudart-12-6`, `libcublas-12-6`, `libcudla-12-6`
- `nvidia-l4t-dla-compiler` matching the installed L4T release

Do not upgrade the board's BSP just to install a compiler from a different L4T
release. The tested DLA compiler package was `36.5.0-20260115194252`. Although this
model uses GPU rather than DLA, the TensorRT plugin library requires these shared
libraries to load.

Use a separate environment exposing the system TensorRT Python bindings:

```bash
python3 -m venv --system-site-packages .venv-gpu
source .venv-gpu/bin/activate
pip install -r requirements-predict.txt
export LD_LIBRARY_PATH=/usr/local/cuda-12.6/targets/aarch64-linux/lib:${LD_LIBRARY_PATH:-}
python scripts/build_gpu.py
python scripts/predict_url.py --runtime gpu --engine models/url_hgb21_fp32.engine --url 'https://example.com'
```

`build_gpu.py` builds a batch-one FP32 engine with TF32 disabled. Engines must be
rebuilt locally for the device and TensorRT version; they are not portable release
artifacts. The GPU runner uses TensorRT 10 tensor addresses and CUDA runtime
buffers, including transfers and synchronization:
https://docs.nvidia.com/deeplearning/tensorrt/10.x.x/inference-library/python-api-docs.html

## Validate your own model conversion

`scripts/validate_real_runtime.py` accepts an NPZ containing finite float32 `X`,
binary `y`, and `reference_scores` from sklearn for the same X. It may also carry
a boolean `is_test` mask, and it should: parity is checked over every row in the
file, while accuracy is read only on the masked rows. The two answer different
questions, and held-out-ness matters to the second one only, so restricting
parity to a test split discards evidence for nothing. Without the mask both fall
back to the whole file. The report states `n_parity` and `n_accuracy` separately.
It checks all scores/labels, then benchmarks real feature rows. It refuses to pass conversion
with any changed label or a maximum score error at least 1e-5. The private study
vectors are not included; release reports contain aggregates only.

See the model card for preprocessing drift relative to historical article tables.
The article PDF remains the v0.1.0 draft, not a report of this new GPU evaluation.
