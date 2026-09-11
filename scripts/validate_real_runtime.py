"""Validate a runtime against locally held reference vectors, then benchmark.

The NPZ must contain finite float32 X, binary y, and sklearn reference_scores.
It may also contain a boolean is_test mask, and it should: parity and accuracy
are different questions over different rows. Parity asks whether this runtime
reproduces the host model's scores, which every available vector helps answer;
accuracy asks how the model generalises and may only be read on held-out rows.
Without the mask both fall back to the whole file, which is what the first run
did, and it under-powered the parity check to the size of the test split.
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
    mask=data['is_test'].astype(bool) if 'is_test' in data.files else np.ones(len(y),dtype=bool)
    if mask.shape!=y.shape:raise ValueError('is_test must be one flag per row')
    if not mask.any():raise ValueError('is_test selects no rows; accuracy would be undefined')
    model=CpuPredictor(a.model) if a.runtime=='cpu' else GpuPredictor(a.model)
    try:
        scores=model.predict(X);pred=scores>.5
        mismatches=int(np.sum(pred!=(ref>.5)))
        error=float(np.max(abs(scores-ref)))
        hp,hy=pred[mask],y[mask]
        tp=int(np.sum(hp & (hy==1)));fp=int(np.sum(hp & (hy==0)))
        fn=int(np.sum(~hp & (hy==1)));tn=int(np.sum(~hp & (hy==0)))
        report={'runtime':a.runtime,'n_parity':int(len(y)),'n_accuracy':int(mask.sum()),
            'accuracy_rows':'held-out test split' if not mask.all() else 'all rows in the file',
            'n_test':int(mask.sum()),
            'max_score_error':error,'label_disagreements':mismatches,
            'accuracy':float(np.mean(hp==hy)),'f1':2*tp/(2*tp+fp+fn),'confusion_matrix':[[tn,fp],[fn,tp]],
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
