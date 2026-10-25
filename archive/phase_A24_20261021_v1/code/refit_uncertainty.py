"""One bounded paired patient full-pipeline refit job. No neural inference."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='2'
from pathlib import Path
from datetime import datetime
import json,time,hashlib,warnings
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
from threadpoolctl import threadpool_limits
P=Path(__file__).resolve().parents[1];U=P.parents[1];A=U/'runs/phase_A17R_20261021_v1';B=U/'runs/phase_A18_20261021_v1';V=U/'manuscript_NC_review_v2_20261021'
START=time.monotonic();OUT=P/'refit';OUT.mkdir(exist_ok=True)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
    q=OUT/n;t=q.with_suffix(q.suffix+'.tmp');t.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8');t.replace(q)
def deadline():
    if time.monotonic()-START>7200:raise TimeoutError('R 2-hour limit; no retry')
def status(**v):put('status.json',{'time':datetime.now().astimezone().isoformat(timespec='seconds'),'pid':os.getpid(),'wall_seconds':time.monotonic()-START,**v})
def fit(name,model,x,y):
    deadline();f=OUT/'fit_ledger.json';rows=json.loads(f.read_text()) if f.exists() else [];assert len(rows)<400
    row={'attempt':len(rows)+1,'name':name,'status':'started','n':len(y),'AF':int(sum(y))};rows.append(row);put('fit_ledger.json',rows)
    try:
        with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
            warnings.simplefilter('always');model.fit(x,y)
        row['warnings']=[str(w.message) for w in ws]
        if any(issubclass(w.category,ConvergenceWarning) for w in ws):raise RuntimeError('Nonconvergence; no retry')
        deadline();row['status']='complete'
    except Exception as e:row['status']='failed_no_retry';row['error']=str(e);raise
    finally:rows[-1]=row;put('fit_ledger.json',rows)
    return model
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def metric(y,p,t):
    z=p>=t
    return {'AUROC':float(roc_auc_score(y,p)),'AP':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p)),'sensitivity':float(np.mean(z[y==1])),'specificity':float(np.mean(~z[y==0])),'threshold':t}
def main():
    assert not (OUT/'fit_ledger.json').exists(),'Existing ledger: do not repeat'
    files=[P/'protocol.txt',Path(__file__),A/'analysis_dataset.csv',A/'embeddings.npz',B/'factor_definitions.json',V/'analysis/freeze_before_fitting.json',V/'analysis/models_frozen_before_readout.json']
    lock={str(f):sha(f) for f in files};put('freeze_before_fitting.json',{'time':datetime.now().astimezone().isoformat(),'files':lock,'max_fits':400,'replicates':100,'no_hyperparameter_selection':True})
    prior=json.loads((V/'analysis/freeze_before_fitting.json').read_text(encoding='utf-8'))
    for path,h in prior['sources'].items():assert sha(path)==h,path
    d=pd.read_csv(A/'analysis_dataset.csv',float_precision='round_trip');assert len(d)==21002 and d.patient_id.nunique()==18260 and d.groupby('patient_id').role.nunique().max()==1
    fd=json.loads((B/'factor_definitions.json').read_text(encoding='utf-8'));z=np.load(A/'embeddings.npz');ix={int(v):i for i,v in enumerate(z['ecg_id'])}
    emb=z['embedding'][[ix[int(v)] for v in d.ecg_id]];fm=[f'FM_{i:03d}' for i in range(768)];d=pd.concat([d.reset_index(drop=True),pd.DataFrame(emb,columns=fm)],axis=1)
    for c in fd['factors']['V']:
        if c.startswith('V_present__'):d[c]=(d[c.split('__',1)[1]].fillna(0)>0).astype(float)
        elif c.startswith('V_missing__'):d[c]=(~np.isfinite(d[c.split('__',1)[1]])).astype(float)
    common=fd['base']+fm+fd['factors']['V']+fd['factors']['N'];cols={'FM_VN':common,'FM_VNM':common+fd['factors']['M']}
    xx={a:d[c].replace([np.inf,-np.inf],np.nan).to_numpy(float) for a,c in cols.items()};y=d.label.to_numpy();rng=np.random.default_rng(2026102401)
    groups={}
    for role in ['train','calibrate','evaluate']:
        ids=np.flatnonzero(d.role.eq(role));ps=d.iloc[ids].patient_id.to_numpy();patients=np.unique(ps);groups[role]=(patients,{v:ids[np.flatnonzero(ps==v)] for v in patients})
    rows=[]
    for rep in range(100):
        sample={}
        for role,(patients,mapping) in groups.items():sample[role]=np.concatenate([mapping[v] for v in rng.choice(patients,len(patients),replace=True)])
        tr,ca,te=(sample[r] for r in ['train','calibrate','evaluate']);assert all(np.unique(y[s]).size==2 for s in sample.values())
        for arm in cols:
            model=make_pipeline(SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2500,solver='lbfgs',random_state=2026102401))
            fit(f'{rep}/{arm}/classifier',model,xx[arm][tr],y[tr]);raw=model.predict_proba(xx[arm][ca])[:,1]
            cal=fit(f'{rep}/{arm}/calibrator',LogisticRegression(C=1,max_iter=1000,random_state=2026102401),logit(raw),y[ca]);cp=cal.predict_proba(logit(raw))[:,1]
            fpr,tpr,threshold=roc_curve(y[ca],cp);valid=np.flatnonzero((tpr>=.95)&np.isfinite(threshold));j=min(valid,key=lambda j:(fpr[j],-threshold[j]));t=float(threshold[j])
            pp=cal.predict_proba(logit(model.predict_proba(xx[arm][te])[:,1]))[:,1];assert np.isfinite(pp).all()
            rows.append({'replicate':rep,'arm':arm,'evaluation_ECGs':len(te),'evaluation_AF':int(sum(y[te])),**metric(y[te],pp,t)})
        pd.DataFrame(rows).to_csv(OUT/'replicate_metrics.csv',index=False);status(state='running',completed_replicates=rep+1,fits=4*(rep+1),max_fits=400)
        print('REFIT',rep+1,'/100',flush=True)
    results=pd.DataFrame(rows);diff=[]
    for met in ['AUROC','AP','Brier','sensitivity','specificity']:
        pivot=results.pivot(index='replicate',columns='arm',values=met);v=pivot.FM_VNM-pivot.FM_VN
        for r,val in v.items():diff.append({'replicate':r,'metric':met,'delta':val})
    delta=pd.DataFrame(diff);delta.to_csv(OUT/'paired_refit_differences.csv',index=False)
    summary=[]
    for met,g in delta.groupby('metric'):
        v=g.delta.to_numpy();q=np.quantile(v,[.05,.25,.5,.75,.95]);summary.append({'metric':met,'p05':q[0],'q25':q[1],'median':q[2],'q75':q[3],'p95':q[4],'positive_fraction':float(np.mean(v>0)),'replicates':100,'interpretation':'Exploratory paired training/calibration/evaluation patient refit distribution,not a confirmatory CI; selection recipe fixed'})
    pd.DataFrame(summary).to_csv(OUT/'refit_distribution_summary.csv',index=False)
    for f,h in lock.items():assert sha(f)==h,f
    put('completion.json',{'state':'complete','fits':400,'max_fits':400,'replicates':100,'neural_epochs':0,'SPH_predictions':0,'original_thresholds_changed':False,'independent_validation':False,'wall_seconds':time.monotonic()-START})
    status(state='complete',completed_replicates=100,fits=400,max_fits=400)
if __name__=='__main__':
    try:main()
    except Exception as e:status(state='stopped_no_retry',error=str(e));raise
