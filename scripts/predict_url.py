#!/usr/bin/env python3
"""Offline URL scoring with the released HistGB model; no page is fetched."""
import argparse
import json
from pathlib import Path
from urllib.parse import urlsplit
import numpy as np
from url_features import extract

ROOT = Path(__file__).resolve().parents[1]

class CpuPredictor:
    def __init__(self, model):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(model), providers=['CPUExecutionProvider'])
    def predict(self, X):
        return self.session.run(None, {'features': np.asarray(X,dtype=np.float32)})[0].reshape(-1)

class GpuPredictor:
    """TensorRT 10 execution with CUDA runtime buffers; a batch-one engine is required."""
    def __init__(self, model):
        import ctypes as c
        import ctypes.util
        import tensorrt as trt
        self.c=c
        library=c.util.find_library('cudart') or '/usr/local/cuda-12.6/targets/aarch64-linux/lib/libcudart.so.12'
        self.cuda=c.CDLL(library)
        self.cuda.cudaMalloc.argtypes=[c.POINTER(c.c_void_p),c.c_size_t]
        self.cuda.cudaFree.argtypes=[c.c_void_p]
        self.cuda.cudaStreamCreate.argtypes=[c.POINTER(c.c_void_p)]
        self.cuda.cudaStreamSynchronize.argtypes=[c.c_void_p]
        self.cuda.cudaStreamDestroy.argtypes=[c.c_void_p]
        self.cuda.cudaMemcpyAsync.argtypes=[c.c_void_p,c.c_void_p,c.c_size_t,c.c_int,c.c_void_p]
        self.logger=trt.Logger(trt.Logger.WARNING)
        self.runtime=trt.Runtime(self.logger)
        self.engine=self.runtime.deserialize_cuda_engine(Path(model).read_bytes())
        if self.engine is None:raise RuntimeError('Cannot load TensorRT engine')
        self.context=self.engine.create_execution_context()
        if not self.context.set_input_shape('features',(1,21)):raise ValueError('Expected batch-one, 21-feature engine')
        self.stream=c.c_void_p();self.check(self.cuda.cudaStreamCreate(c.byref(self.stream)))
        self.buffers={}
        for name in ['features','phishing_score']:
            shape=tuple(self.context.get_tensor_shape(name));dtype=trt.nptype(self.engine.get_tensor_dtype(name))
            host=np.empty(shape,dtype=dtype);device=c.c_void_p()
            self.check(self.cuda.cudaMalloc(c.byref(device),host.nbytes))
            if not self.context.set_tensor_address(name,device.value):raise RuntimeError('Tensor binding failed')
            self.buffers[name]=(host,device)
    def check(self, status):
        if status != 0:raise RuntimeError(f'CUDA runtime error {status}')
    def predict(self, X):
        scores=[]
        for row in np.asarray(X,dtype=np.float32):
            host,device=self.buffers['features'];host[:]=row
            self.check(self.cuda.cudaMemcpyAsync(device,host.ctypes.data,host.nbytes,1,self.stream))
            if not self.context.execute_async_v3(self.stream.value):raise RuntimeError('TensorRT inference failed')
            out,ptr=self.buffers['phishing_score']
            self.check(self.cuda.cudaMemcpyAsync(out.ctypes.data,ptr,out.nbytes,2,self.stream))
            self.check(self.cuda.cudaStreamSynchronize(self.stream))
            scores.append(float(out.reshape(-1)[0]))
        return np.asarray(scores)
    def close(self):
        for _,device in self.buffers.values():self.check(self.cuda.cudaFree(device))
        self.buffers.clear();self.check(self.cuda.cudaStreamDestroy(self.stream))


def vector(url, features):
    value=url.strip()
    if not value or any(ch.isspace() for ch in value):raise ValueError('URL must be nonempty and contain no whitespace')
    parsed=urlsplit(value if '://' in value else 'http://'+value)
    if parsed.scheme.lower() not in {'http','https'} or not parsed.hostname:raise ValueError('Expected an HTTP(S) URL or hostname')
    _ = parsed.port  # Reject malformed ports and scheme-like non-HTTP input.
    row=extract(value)
    X=np.array([[row[f] for f in features]],dtype=np.float32)
    if not np.isfinite(X).all():raise ValueError('Features must be finite')
    return X


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--url',required=True);p.add_argument('--runtime',choices=['cpu','gpu'],default='cpu')
    p.add_argument('--engine',type=Path,help='TensorRT engine built locally for this Jetson')
    a=p.parse_args();meta=json.loads((ROOT/'artifacts/url_histgb21/features.json').read_text())
    try:X=vector(a.url,meta['features'])
    except ValueError as e:p.error(str(e))
    if a.runtime=='gpu' and not a.engine:p.error('--engine is required for GPU inference')
    model=CpuPredictor(ROOT/'artifacts/url_histgb21/cpu.onnx') if a.runtime=='cpu' else GpuPredictor(a.engine)
    try:score=float(model.predict(X)[0])
    finally:
        if a.runtime=='gpu':model.close()
    print(json.dumps({'model':'PhishVN URL HistGB (21 features)','runtime':a.runtime,
        'label':'phishing' if score>meta['threshold'] else 'benign','phishing_score':score,
        'threshold':meta['threshold'],'note':'Research model score, not a calibrated safety guarantee; URL was not fetched.'},indent=2))

if __name__=='__main__':main()
