#!/usr/bin/env python3
"""Edge benchmark for the P7 measurement campaign (ONNX Runtime, CPU EP).

Reports: median/p95/p99 batch-1 latency, sustained throughput over a fixed
wall-clock window, peak RSS, and (on Jetson) average power + temperature parsed
from a concurrent tegrastats capture.
"""
import argparse, json, os, re, resource, statistics, subprocess, sys, tempfile, time

import numpy as np
import onnxruntime as ort


def parse_tegrastats(lines):
    pw, temps = [], []
    for ln in lines:
        m = re.search(r"VDD_IN (\d+)mW", ln)
        if m:
            pw.append(int(m.group(1)))
        t = re.findall(r"(?:cpu|CPU|tj)@([0-9.]+)C", ln)
        if t:
            temps.append(max(float(x) for x in t))
    return pw, temps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--dim", type=int, required=True)
    ap.add_argument("--lat-iters", type=int, default=5000)
    ap.add_argument("--sustain-secs", type=int, default=30)
    args = ap.parse_args()

    if args.dim <= 0 or args.lat_iters <= 0 or args.sustain_secs <= 0:
        ap.error("--dim, --lat-iters, and --sustain-secs must be positive")

    sess = ort.InferenceSession(args.model, providers=["CPUExecutionProvider"])
    name = sess.get_inputs()[0].name
    X = np.random.rand(1, args.dim).astype(np.float32)
    for _ in range(200):
        sess.run(None, {name: X})

    # batch-1 latency distribution
    ts = []
    for _ in range(args.lat_iters):
        t0 = time.perf_counter()
        sess.run(None, {name: X})
        ts.append((time.perf_counter() - t0) * 1000)
    ts.sort()
    lat = {
        "median_ms": round(statistics.median(ts), 4),
        "p95_ms": round(ts[int(0.95 * len(ts))], 4),
        "p99_ms": round(ts[int(0.99 * len(ts))], 4),
    }

    # sustained throughput with concurrent tegrastats (if available)
    tegra = None
    telemetry = tempfile.TemporaryFile(mode="w+t")
    try:
        tegra = subprocess.Popen(["tegrastats", "--interval", "500"],
                                 stdout=telemetry, text=True)
    except FileNotFoundError:
        pass
    n, t0 = 0, time.perf_counter()
    while time.perf_counter() - t0 < args.sustain_secs:
        sess.run(None, {name: X})
        n += 1
    elapsed = time.perf_counter() - t0
    power = {}
    if tegra:
        tegra.terminate()
        tegra.wait(timeout=5)
        telemetry.seek(0)
        out = telemetry.read().splitlines()
        pw, temps = parse_tegrastats(out)
        if pw:
            power = {"avg_power_mw": round(sum(pw) / len(pw)),
                     "max_power_mw": max(pw),
                     "max_temp_c": max(temps) if temps else None,
                     "energy_per_inf_mj": round(sum(pw) / len(pw) * elapsed / n, 4)}

    telemetry.close()
    rss_divisor = 1024 * 1024 if sys.platform == "darwin" else 1024
    print(json.dumps({
        "model": os.path.basename(args.model),
        "size_mb": round(os.path.getsize(args.model) / 1e6, 3),
        "latency_batch1": lat,
        "sustained_throughput_inf_s": round(n / elapsed, 1),
        "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / rss_divisor, 1),
        "power": power,
        "device": open("/proc/device-tree/model").read().strip("\x00") if os.path.exists("/proc/device-tree/model") else "unknown",
    }, indent=2))


if __name__ == "__main__":
    main()
