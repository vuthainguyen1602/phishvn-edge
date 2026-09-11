PYTHON ?= python3
.PHONY: quickstart demo assets test
demo:
	$(PYTHON) examples/train_demo.py
	$(PYTHON) scripts/export_edge.py --model models/demo.joblib --dim 9 --out models/demo.onnx --bench 100
	mkdir -p runs
	$(PYTHON) scripts/bench_edge_ort.py --model models/demo.onnx --dim 9 --lat-iters 100 --sustain-secs 1 > runs/demo.json
assets:
	$(PYTHON) scripts/make_edge_assets.py
test:
	$(PYTHON) -m unittest discover -s tests -v

quickstart:
	mkdir -p runs
	$(PYTHON) scripts/bench_edge_ort.py --model examples/artifacts/dummy.onnx --dim 9 --lat-iters 100 --sustain-secs 1 > runs/quickstart.json
	@echo "Benchmark saved to runs/quickstart.json"
