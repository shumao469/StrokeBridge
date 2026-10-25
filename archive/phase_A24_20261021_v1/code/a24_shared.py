"""Pinned inputs and common analysis helpers for A24. No import-time inference."""
from pathlib import Path
from datetime import datetime
import hashlib,json,os,time,sys
P=Path(__file__).resolve().parents[1];U=P.parents[1];A=U/'runs/phase_A17R_20261021_v1';B=U/'runs/phase_A18_20261021_v1';V=U/'manuscript_NC_review_v2_20261021'
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='2'
os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
VENDOR=P/'runtime/vendor'
if VENDOR.exists():sys.path.insert(0,str(VENDOR))
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');tmp.replace(p)
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def csv(p):
    import pandas as pd
    return pd.read_csv(p,float_precision='round_trip')
def cohort():
    d=csv(A/'analysis_dataset.csv').sort_values('ecg_id').reset_index(drop=True)
    assert len(d)==21002 and d.ecg_id.is_unique and d.patient_id.nunique()==18260 and d.groupby('patient_id').role.nunique().max()==1
    assert d.role.eq('train').sum()==14709 and d.role.eq('calibrate').sum()==2097 and d.role.eq('evaluate').sum()==2095
    return d
def prior_integrity():
    prior=read(V/'analysis/freeze_before_fitting.json')
    for path,h in prior['sources'].items():assert sha(path)==h,path
def freeze(module,extra=()):
    dest=P/module/'source_lock.json';assert not dest.exists(),'Module already frozen; no implicit retry'
    dependencies={'extraction':['extract_controls.py'],'negative':['negative_controls.py','extract_controls.py'],'comparisons':['fit_and_audit.py']}
    files=[P/'protocol.txt',Path(__file__),*[P/'code'/n for n in dependencies.get(module,[])],A/'analysis_dataset.csv',A/'embeddings.npz',A/'waveform_audit.csv',B/'factor_definitions.json',*extra]
    lock={str(p):sha(p) for p in files};put(dest,{'time':now(),'files':lock,'post_hoc_development':True});return lock
def verify(lock):
    for path,h in lock.items():assert sha(path)==h,path
