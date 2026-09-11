"""Validate a runtime against locally held reference vectors, then benchmark.

The NPZ must contain finite float32 X, binary y, and sklearn reference_scores.
Only aggregate metrics are written; no URLs or per-record scores are published.
"""
import argparse,json,time,platform,statistics
from pathlib import Path
import numpy as np
from predict_url import CpuPredictor,GpuPredictor


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--model',required=True)
    p.add_argument('--runtime',choices=['cpu','gpu'],required=True);p.add_argument('--out',required=True)
    p.add_argument('--seconds',type=int,default=30);p.add_argument('--iterations',type=int,default=1000)
    a=p.parse_args()
    if a.seconds<=0 or a.iterations<=0:p.error('Benchmark durations must be positive')
    data=np.load(a.data);X=data['X'];y=data['y'];ref=data['reference_scores']
    if X.dtype!=np.float32 or not np.isfinite(X).all():raise ValueError('Expected finite float32 X')
    model=CpuPredictor(a.model) if a.runtime=='cpu' else GpuPredictor(a.model)
    try:
        scores=model.predict(X);pred=scores>.5
        mismatches=int(np.sum(pred!=(ref>.5)))
        error=float(np.max(abs(scores-ref)))
        tp=int(np.sum(pred & (y==1)));fp=int(np.sum(pred & (y==0)))
        fn=int(np.sum(~pred & (y==1)));tn=int(np.sum(~pred & (y==0)))
        report={'runtime':a.runtime,'n_test':len(y),'max_score_error':error,'label_disagreements':mismatches,
            'accuracy':float(np.mean(pred==y)),'f1':2*tp/(2*tp+fp+fn),'confusion_matrix':[[tn,fp],[fn,tp]],
            'validation_passed':mismatches==0 and error<1e-5,'platform':platform.machine()}
        if not report['validation_passed']:
            Path(a.out).write_text(json.dumps(report,indent=2)+'\n');raise RuntimeError('Runtime parity failed')
        for i in range(200):model.predict(X[i%len(X):i%len(X)+1])
        times=[]
        for i in range(a.iterations):
            row=X[i%len(X):i%len(X)+1];start=time.perf_counter();model.predict(row)
            times.append((time.perf_counter()-start)*1000)
        n=0;start=time.perf_counter()
        while time.perf_counter()-start<a.seconds:
            i=n%len(X);model.predict(X[i:i+1]);n+=1
        elapsed=time.perf_counter()-start
        report['benchmark']={'median_ms':statistics.median(times),'p95_ms':float(np.percentile(times,95)),
            'throughput_inf_s':n/elapsed,'window_seconds':elapsed,'timed_iterations':a.iterations,
            'scope':'batch-one model inference; GPU includes host/device copies and synchronization; excludes URL features',
            'precision':'FP32; GPU engine built with TF32 disabled'}
        Path(a.out).write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
    finally:
        if a.runtime=='gpu':model.close()

if __name__=='__main__':main()
