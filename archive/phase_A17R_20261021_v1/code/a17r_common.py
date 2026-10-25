"""Paths and immutable analysis definitions for the bounded A17 repair."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
from pathlib import Path
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,json
RUN=Path('/home/shu/codex_runs/phase_A17R_20261021_v1')
BASE=Path('DRIVE_ROOT/wsl/2026-09-07/target-nature-e-you-are-a/unified_project')
META=next(Path('DRIVE_ROOT/脑心同查/datasets/ptb-xl/1.0.3').rglob('ptbxl_database.csv'))
DATA=META.parent
PW=Path('/home/shu/codex_runs/phase_A10_20261019_v1/C1_sign_orbit/seed_20261020/fold_0/best.pt')
FM=Path('DRIVE_ROOT/脑心同查/models/ecg-fm/584219ea492cdeef2e19ffbdf9c6ecc874ba427e/mimic_iv_ecg_physionet_pretrained.pt')
FM_SHA='4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e'
LEADS=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
ARMS=['D','D_RR','D_P','D_RR_P','D_FM','D_RR_FM','D_RR_P_FM']
CS=[.01,.1,1.]; SEED=20261021; MAX_FITS=30; PLANNED_FITS=28
def now():return datetime.now(ZoneInfo('Asia/Shanghai')).isoformat(timespec='seconds')
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
 return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def write(p,v):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');t.replace(p)
def status(stage,**kw):
 v=dict(run_id=RUN.name,updated_at=now(),pid=os.getpid(),stage=stage,**kw);write(RUN/'status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
def check_freeze():
 pin=read(RUN/'source_lock.json')
 for p,h in pin['files'].items():assert sha(p)==h,f'Frozen source changed: {p}'
 return pin
