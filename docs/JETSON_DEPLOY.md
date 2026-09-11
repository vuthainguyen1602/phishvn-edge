# Jetson edge benchmarks

Clone https://github.com/vuthainguyen1602/phishvn-edge on the device and follow
README setup instructions. Use an ONNX Runtime wheel compatible with the device's
Python, Linux/aarch64, and JetPack environment. The synthetic demo runs on the CPU.

```bash
make demo
python scripts/bench_edge_ort.py --model models/demo.onnx --dim 9 --lat-iters 5000 --sustain-secs 30 > runs/jetson-demo.json
```

If `tegrastats` is on PATH, the harness captures board power and temperature at
500 ms intervals. Without it, `power` is empty. Record JetPack/runtime versions,
power mode, cooling, and clock configuration alongside your measurements.

## TensorRT

The real HistGB detector now has a validated FP32 TensorRT path. Follow
[REAL_MODEL.md](REAL_MODEL.md) for runtime packages, local engine construction,
and GPU URL inference. The historical FP16 row remains a separate experiment;
this release validates FP32 with TF32 disabled.

## Deployment boundary

This toolkit benchmarks models; it does not install collectors, background timers,
SMS forwarding, DNS filters, or a network scoring service. Such integrations are
future applications that require separate implementation and end-to-end evaluation.
