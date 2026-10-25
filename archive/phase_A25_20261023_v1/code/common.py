"""A25 runtime support: independent ledgers, immutable sources, no figures."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS']: os.environ[k]='1'
os.environ['CUDA_VISIBLE_DEVICES']=''
from pathlib import Path
import hashlib,json,time
from datetime import datetime
RUN=Path(__file__).resolve().parents[1]
BASE=RUN.parents[1]
A17=BASE/'runs/phase_A17R_20261021_v1'
A18=BASE/'runs/phase_A18_20261021_v1'
A24=BASE/'runs/phase_A24_20261021_v1/recovery_E_wsl_20261023_v2'
A9=Path('/home/shu/codex_runs/phase_A9_20261017_v1')
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp')
    tmp.write_text(json.dumps(x,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def verify():
    lock=read(RUN/'source_lock.json')
    for p,h in lock['files'].items():assert sha(p)==h, 'Frozen source changed: '+p
    return lock
def deadline(start,seconds):
    if time.monotonic()-start>seconds:raise TimeoutError(f'A25 module limit {seconds}s')
    if (RUN/'STOP').exists():raise RuntimeError('A25 manual STOP')
def status(module,**kwargs):
    v=dict(time=now(),pid=os.getpid(),CPU_threads=1,SPH_admitted=False,new_figures=0,**kwargs)
    put(RUN/module/'status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
