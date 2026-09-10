"""End-to-end regression checks without private models or datasets."""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]

class WorkflowTest(unittest.TestCase):
    def test_export_predictions_and_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            def run(script, *args):
                return subprocess.run([sys.executable, str(ROOT / script), *args],
                                      cwd=tmp, check=True, capture_output=True, text=True)
            run('examples/train_demo.py')
            run('scripts/export_edge.py', '--model', 'models/demo.joblib', '--dim', '9',
                '--out', 'models/demo.onnx', '--bench', '5')
            model = joblib.load(Path(tmp) / 'models/demo.joblib')
            session = ort.InferenceSession(str(Path(tmp) / 'models/demo.onnx'),
                                           providers=['CPUExecutionProvider'])
            X = np.random.default_rng(0).normal(size=(32, 9)).astype(np.float32)
            labels, probability_maps = session.run(None, {session.get_inputs()[0].name: X})
            np.testing.assert_array_equal(labels, model.predict(X))
            np.testing.assert_allclose([v[1] for v in probability_maps],
                                       model.predict_proba(X)[:, 1], atol=1e-6)
            result = run('scripts/bench_edge_ort.py', '--model', 'models/demo.onnx',
                         '--dim', '9', '--lat-iters', '20', '--sustain-secs', '1')
            measurement = json.loads(result.stdout)
            self.assertGreater(measurement['sustained_throughput_inf_s'], 0)
            self.assertGreater(measurement['peak_rss_mb'], 0)
            latency = measurement['latency_batch1']
            self.assertLessEqual(latency['median_ms'], latency['p95_ms'])
            self.assertLessEqual(latency['p95_ms'], latency['p99_ms'])

    def test_export_failure_is_nonzero(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'invalid.joblib'
            joblib.dump({'not_a_model': True}, path)
            result = subprocess.run([sys.executable, str(ROOT / 'scripts/export_edge.py'),
                                     '--model', str(path), '--dim', '9'], capture_output=True)
            self.assertNotEqual(result.returncode, 0)
