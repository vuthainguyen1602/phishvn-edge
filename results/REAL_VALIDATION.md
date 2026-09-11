# Real-model validation on the second Jetson

`jetson_real_cpu.json` and `jetson_real_gpu.json` were re-measured on the same
Jetson Orin Nano Super on 2026-09-11. TensorRT 10.3.0, CUDA 12.6, L4T 36.5.0;
ONNX Runtime 1.23.2, numpy 2.2.6, tldextract 5.3.1 on the device. The reference
scores were produced off-device with scikit-learn 1.4.2 and numpy 1.26.4.

Parity and accuracy are read over different rows, because they are different
questions. Parity — does this runtime reproduce the host model's scores — runs
over all 53,116 corpus feature vectors, since held-out-ness is irrelevant to it
and every vector is evidence. Accuracy runs over the 8,941 held-out temporal-test
rows only. An earlier run gave both the test split; that under-powered parity by
a factor of six and is superseded by these reports. Both runtimes agree with
sklearn on every label across the whole corpus; maximum score errors are 1.5e-7
(CPU) and 9.7e-8 (GPU). GPU inference actually executes through TensorRT and
CUDA, with no CPU fallback.

Latency uses 200 warm-up calls and 1,000 timed batch-one calls; throughput uses
one 30-second window over test feature vectors. It excludes URL parsing and
feature extraction. GPU timing includes host/device transfers and stream
synchronization. These are single-run pilot measurements on a machine with
existing background services; clocks/power mode were not locked and no background
services were stopped. They do not establish a thermal steady state or
deployment-wide performance, and they move between runs: the CPU throughput here
is 21,275/s where an earlier run of the same command reported 22,861/s.

CPU was faster for this small model: median about 0.041 ms / 21,275 inferences/s,
versus GPU 0.163 ms / 5,971 inferences/s. GPU correctness is established, but no
GPU speedup is claimed. The older `pilot.csv` is an independent historical
experiment.

`real_model_validation.json` is an earlier aggregate from the same detector under
the archived preprocessing rather than the extractor in `scripts/`: it records
5 label changes from float64 to float32 and 2 from raw to archived features, on
the same 8,941 test rows. It is kept because those counts are the evidence that
the conversion is lossless at the label level, and it is not comparable with the
reports above, which use newly extracted features.

Only aggregate reports are released. Test URLs, labels paired with individual
records, source joblib files, and device-specific TensorRT engines are excluded.
