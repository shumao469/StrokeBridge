"""Frozen point comparisons after verified extraction; max22 fits, no search."""
from a24_shared import *
import numpy as np,pandas as pd,joblib,warnings
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import roc_curve,roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits
OUT=P/'comparisons';OUT.mkdir(exist_ok=True);(OUT/'models').mkdir(exist_ok=True);START=time.monotonic()
def deadline():
    if time.monotonic()-START>2700:raise TimeoutError('Comparison45-minute limit')
def status(**v):put(OUT/'status.json',{'time':now(),'pid':os.getpid(),'wall_seconds':time.monotonic()-START,**v})
def fit(name,model,x,y):
    deadline();f=OUT/'fit_ledger.json';rows=read(f) if f.exists() else [];assert len(rows)<22
    r={'attempt':len(rows)+1,'name':name,'status':'started','n':len(y),'AF':int(sum(y))};rows.append(r);put(f,rows)
    try:
        with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=2):
            warnings.simplefilter('always');model.fit(x,y)
        r['warnings']=[str(w.message) for w in ws]
        if any(issubclass(w.category,ConvergenceWarning) for w in ws):raise RuntimeError('Nonconvergence: no retry')
        deadline();r['status']='complete'
    except Exception as e:r['status']='failed_no_retry';r['error']=str(e);raise
    finally:rows[-1]=r;put(f,rows)
    return model
def logit(p):p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def decorate(d,fd):
    d=d.copy()
    for c in fd['factors']['V']:
        if c.startswith('V_present__'):d[c]=(d[c.split('__',1)[1]].fillna(0)>0).astype(float)
        elif c.startswith('V_missing__'):d[c]=(~np.isfinite(d[c.split('__',1)[1]])).astype(float)
    return d
def metric(y,p,t):
    z=p>=t;tp=int(sum(z&(y==1)));fp=int(sum(z&(y==0)));fn=int(sum(~z&(y==1)));tn=int(sum(~z&(y==0)))
    return {'AUROC':float(roc_auc_score(y,p)),'AP':float(average_precision_score(y,p)),'Brier':float(brier_score_loss(y,p)),'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'PPV':tp/max(tp+fp,1),'TP':tp,'FP':fp,'FN':fn,'TN':tn}
def main():
    assert not (OUT/'fit_ledger.json').exists(),'Existing ledger; do not repeat'
    prior_integrity();E_ok=(P/'extraction/completion.json').exists() and read(P/'extraction/completion.json')['state']=='complete'
    assert read(P/'negative/completion.json')['state']=='complete'
    inputs=[P/'negative'/n for n in ['current_candidates.csv','displaced_morphology.csv','anchor_morphology.csv','completion.json']]
    if E_ok:inputs.extend([P/'extraction/common5s_embeddings.npz',P/'extraction/completion.json'])
    lock=freeze('comparisons',inputs);d=cohort();fd=read(B/'factor_definitions.json');M=fd['factors']['M'];VN=fd['factors']['V']+fd['factors']['N'];base=fd['base']
    original=decorate(d,fd);designs={};cols={}
    if E_ok:
        z=np.load(P/'extraction/common5s_embeddings.npz');assert np.array_equal(d.ecg_id,z['ecg_id'])
        for enc in ['E1','E2']:
            names=[f'{enc}_{i:03d}' for i in range(768)];frame=pd.concat([original,pd.DataFrame(z[enc],columns=names)],axis=1)
            for suffix,add in [('',[]),('_VN',VN),('_VNM',VN+M)]:
                name=enc+suffix;designs[name]=frame;cols[name]=base+names+add
    current=csv(inputs[0]);assert np.array_equal(current.ecg_id,d.ecg_id)
    frame=d.drop(columns=[c for c in current.columns if c!='ecg_id' and c in d]).merge(current,on='ecg_id',validate='one_to_one');frame=decorate(frame,fd)
    designs['N_VN']=frame;cols['N_VN']=base+VN;designs['N_current']=frame;cols['N_current']=base+VN+M
    admission=[]
    for name,file in [('N_displaced',inputs[1]),('N_anchor',inputs[2])]:
        control=csv(file);assert np.array_equal(control.ecg_id,d.ecg_id);new=frame.copy();ok=True
        for c in M:
            old=frame[c].to_numpy(float);value=control[c].to_numpy(float);finite=np.isfinite(old)
            if np.any(~np.isfinite(value[finite])):ok=False
            new[c]=np.where(finite,value,np.nan)
        admission.append({'arm':name,'admitted':ok,'rule':'Current V/N and current-M missingness fixed; new missing finite entries stop only this control'})
        if ok:designs[name]=new;cols[name]=base+VN+M
    rng=np.random.default_rng(2026102402);permuted=frame.copy()
    for role in ['train','select','calibrate','evaluate']:
        for c in M:
            ix=np.flatnonzero(frame.role.eq(role).to_numpy()&np.isfinite(frame[c].to_numpy(float)));permuted.loc[ix,c]=rng.permutation(frame.loc[ix,c].to_numpy())
    designs['N_permuted']=permuted;cols['N_permuted']=base+VN+M
    put(OUT/'control_admission.json',admission)
    train=d.role.eq('train').to_numpy();cal=d.role.eq('calibrate').to_numpy();test=d.role.eq('evaluate').to_numpy();y=d.label.to_numpy()
    packs={};x={}
    for name,frame in designs.items():
        x[name]=frame[cols[name]].replace([np.inf,-np.inf],np.nan).to_numpy(float)
        model=make_pipeline(SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True),StandardScaler(),LogisticRegression(C=.1,max_iter=2500,solver='lbfgs',random_state=2026102402))
        fit(name+'/classifier',model,x[name][train],y[train]);raw=model.predict_proba(x[name][cal])[:,1]
        calibrator=fit(name+'/calibrator',LogisticRegression(C=1,max_iter=1000,random_state=2026102402),logit(raw),y[cal]);cp=calibrator.predict_proba(logit(raw))[:,1]
        fpr,tpr,th=roc_curve(y[cal],cp);eligible=np.flatnonzero((tpr>=.95)&np.isfinite(th));j=min(eligible,key=lambda j:(fpr[j],-th[j]));threshold=float(th[j])
        pack={'classifier':model,'calibrator':calibrator,'threshold':threshold,'columns':cols[name],'scope':'Viewed exposure-stratified development; concurrent AF'};joblib.dump(pack,OUT/'models'/f'{name}.joblib');packs[name]=pack
        status(state='fitting',model=name,fits=len(read(OUT/'fit_ledger.json')),max_fits=22)
    put(OUT/'models_frozen_before_evaluation.json',{'time':now(),'models':{p.name:sha(p) for p in (OUT/'models').glob('*.joblib')},'thresholds':{n:q['threshold'] for n,q in packs.items()},'fits':len(read(OUT/'fit_ledger.json'))})
    pred=d.loc[test,['ecg_id','patient_id','label']].reset_index(drop=True).copy();rows=[]
    for name,pack in packs.items():
        prob=pack['calibrator'].predict_proba(logit(pack['classifier'].predict_proba(x[name][test])[:,1]))[:,1];assert np.isfinite(prob).all();pred[name]=prob
        rows.append({'arm':name,'n':len(pred),'patients':pred.patient_id.nunique(),'AF':int(sum(pred.label)),'threshold':pack['threshold'],**metric(pred.label.to_numpy(),prob,pack['threshold'])})
    pred.to_csv(OUT/'evaluation_predictions.csv',index=False);pd.DataFrame(rows).to_csv(OUT/'model_metrics.csv',index=False)
    # Exact decimal replay,never loosen the metric tolerance.
    replay=csv(OUT/'evaluation_predictions.csv');met=pd.DataFrame(rows).set_index('arm')
    for name in packs:
        for k,v in metric(replay.label.to_numpy(),replay[name].to_numpy(),packs[name]['threshold']).items():assert np.isclose(v,met.loc[name,k],rtol=0,atol=1e-12),(name,k)
    pairs=[('E1_VN','E1'),('E1_VNM','E1_VN'),('E2_VN','E2'),('E2_VNM','E2_VN'),('E2','E1'),('N_current','N_VN'),('N_displaced','N_current'),('N_permuted','N_current'),('N_anchor','N_current')]
    pairs=[(a,b) for a,b in pairs if a in packs and b in packs];names=list(packs);yy=pred.label.to_numpy();pp=pred[names].to_numpy();patients=np.unique(pred.patient_id);groups={v:np.flatnonzero(pred.patient_id.to_numpy()==v) for v in patients};rng=np.random.default_rng(2026102403)
    def values(ix):return np.array([[roc_auc_score(yy[ix],pp[ix,j]),average_precision_score(yy[ix],pp[ix,j]),brier_score_loss(yy[ix],pp[ix,j])] for j in range(len(names))])
    point=values(np.arange(len(pred)));bags=[]
    for r in range(1000):
        if r%100==0:deadline();status(state='statistics',draw=r,models=len(names))
        ix=np.concatenate([groups[v] for v in rng.choice(patients,len(patients),replace=True)]);assert np.unique(yy[ix]).size==2;bags.append(values(ix))
    bags=np.stack(bags);ci=np.quantile(bags,[.025,.975],axis=0);absolute=[];paired=[]
    for j,name in enumerate(names):
        for k,m in enumerate(['AUROC','AP','Brier']):absolute.append({'arm':name,'metric':m,'estimate':point[j,k],'lower':ci[0,j,k],'upper':ci[1,j,k],'confidence':.95,'scope':'Exploratory fixed-model patient-cluster interval'})
    for a,b in pairs:
        ai,bi=names.index(a),names.index(b);q=np.quantile(bags[:,ai]-bags[:,bi],[.025,.975],axis=0)
        for k,m in enumerate(['AUROC','AP','Brier']):paired.append({'method':a,'baseline':b,'metric':m,'delta':point[ai,k]-point[bi,k],'lower':q[0,k],'upper':q[1,k],'confidence':.95,'scope':'Exploratory,not confirmatory or causal'})
    pd.DataFrame(absolute).to_csv(OUT/'absolute_intervals.csv',index=False);pd.DataFrame(paired).to_csv(OUT/'paired_intervals.csv',index=False)
    verify(lock);status(state='complete',fits=len(read(OUT/'fit_ledger.json')),models=len(packs),SPH_predictions=0)
    put(OUT/'completion.json',{'state':'complete','fits':len(read(OUT/'fit_ledger.json')),'max_fits':22,'models':len(packs),'encoder_comparison_complete':E_ok,'bootstrap_draws':1000,'metric_replay':'PASS','eligible_control_modules':admission,'independent_validation':False,'NC_submission_ready':False,'wall_seconds':time.monotonic()-START})
if __name__=='__main__':
    try:main()
    except Exception as e:status(state='stopped_no_retry',error=str(e));raise
