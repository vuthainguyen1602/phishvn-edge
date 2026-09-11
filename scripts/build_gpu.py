"""Build a local TensorRT 10 FP32 engine; serialized engines are device-specific."""
import argparse,os,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=ROOT/'models/url_hgb21_fp32.engine')
    p.add_argument('--trtexec',default=shutil.which('trtexec') or '/usr/src/tensorrt/bin/trtexec')
    a=p.parse_args();a.out.parent.mkdir(parents=True,exist_ok=True)
    env=os.environ.copy();cuda=Path('/usr/local/cuda-12.6/targets/aarch64-linux/lib')
    if cuda.exists():env['LD_LIBRARY_PATH']=str(cuda)+':'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([a.trtexec,'--onnx='+str(ROOT/'artifacts/url_histgb21/gpu.onnx'),
        '--minShapes=features:1x21','--optShapes=features:1x21','--maxShapes=features:1x21',
        '--saveEngine='+str(a.out),'--noTF32','--skipInference'],env=env,check=True)
