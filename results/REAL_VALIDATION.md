# Real-model validation on the second Jetson

`jetson_real_cpu.json` and `jetson_real_gpu.json` were measured on the same
Jetson Orin Nano Super on 2026-09-11. TensorRT 10.3.0, CUDA 12.6, L4T 36.5.0;
ONNX Runtime 1.23.2, numpy 1.26.4, scikit-learn 1.4.2, tldextract 5.3.1.

Both runtimes use 8,941 real temporal-test URL feature vectors and compare against
sklearn predictions on the same float32 vectors. GPU inference actually executes
through TensorRT and CUDA, with no CPU fallback. All labels agree; maximum score
errors are below 1.2e-7. The reports use newly extracted features, so their accuracy
must be distinguished from archived preprocessing in `real_model_validation.json`.

Latency uses 200 warm-up calls and 1,000 timed batch-one calls; throughput uses
one 30-second window over test feature vectors. It excludes URL parsing and feature
extraction. GPU timing includes host/device transfers and stream synchronization.
These are single-run pilot measurements on a machine with existing background
services; clocks/power mode were not locked and no background services were stopped.
They do not establish a thermal steady state or deployment-wide performance.

CPU was faster for this small model: median about 0.040 ms / 22,861 inferences/s,
versus GPU 0.164 ms / 5,998 inferences/s. GPU correctness is established, but no GPU
speedup is claimed. The older `pilot.csv` is an independent historical experiment.

Only aggregate reports are released. Test URLs, labels paired with individual
records, source joblib files, and device-specific TensorRT engines are excluded.
