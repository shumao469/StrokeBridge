"""Prespecified V/N-fixed morphology reliability experiment; no encoder inference."""
from common import *
import sys,traceback,warnings
import numpy as np,pandas as pd,joblib
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
OUT=RUN/'missingness';START=time.monotonic();MAX=5400;SEED=2026102305;ARMS=['D0','D1','D2','D3']
def check():deadline(START,MAX)
def metric(y,p,t):
    assert np.isfinite(p).all() and np.unique(y).size==2
    z=p>=t;tp=int(sum(z&(y==1)));fp=int(sum(z&(y==0)));fn=int(sum(~z&(y==1)));tn=int(sum(~z&(y==0)))
    return dict(AUROC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),TP=tp,FP=fp,FN=fn,TN=tn)
def logit(p):
    p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def fit(name,model,x,y,weights=None):
    check();path=OUT/'fit_ledger.json';rows=read(path) if path.exists() else [];assert len(rows)<88
    record=dict(attempt=len(rows)+1,name=name,state='started',rows=len(y),AF=int(sum(y)),time=now());rows.append(record);put(path,rows)
    try:
        with warnings.catch_warnings(record=True) as ws,threadpool_limits(limits=1):
            warnings.simplefilter('always');model.fit(x,y,sample_weight=weights)
        record['warnings']=[str(v.message) for v in ws]
        if any(issubclass(w.category,ConvergenceWarning) for w in ws):raise RuntimeError('Nonconvergence: no retry')
        assert np.isfinite(model.coef_).all();check();record['state']='complete'
    except Exception as e:record['state']='failed_no_retry';record['error']=repr(e);raise
    finally:rows[-1]=record;put(path,rows)
    return model
def decorate(d,fd):
    d=d.copy()
    for c in fd['factors']['V']:
        if c.startswith('V_present__'):d[c]=(d[c.split('__',1)[1]].fillna(0)>0).astype(float)
        elif c.startswith('V_missing__'):d[c]=(~np.isfinite(d[c.split('__',1)[1]])).astype(float)
    return d
def design(arm,x,conf,base_size,pack,dropout=0.,rng=None):
    x=x[:,:base_size] if arm=='D0' else x
    z=pack['scaler'].transform(pack['imputer'].transform(x))
    if arm in ['D2','D3']:z[:,base_size:]*=np.nan_to_num(np.clip(conf,0,1),nan=0)[:,None]
    if dropout and arm!='D0':
        assert rng is not None;mask=rng.random(z[:,base_size:].shape)<dropout;z[:,base_size:][mask]=0.
    assert np.isfinite(z).all();return z
def train(arm,x,conf,y,tr,cal,base_size,tag,rng):
    check();xx=x[tr,:base_size] if arm=='D0' else x[tr]
    imputer=SimpleImputer(strategy='median',add_indicator=False,keep_empty_features=True);scaler=StandardScaler()
    ap=OUT/'preprocessor_ledger.json';aux=read(ap) if ap.exists() else [];assert len(aux)<44
    aux.append(dict(name=tag,imputer_fits=1,scaler_fits=1,training_rows=len(tr),train_only=True,state='reserved_before_transformer_fit'));put(ap,aux)
    xt=imputer.fit_transform(xx);scaler.fit(xt);aux[-1]['state']='complete';put(ap,aux)
    pack=dict(imputer=imputer,scaler=scaler,arm=arm,base_size=base_size)
    xt=design(arm,x[tr],conf[tr],base_size,pack);yy=y[tr];weights=None
    if arm=='D3':
        z1=design(arm,x[tr],conf[tr],base_size,pack,.2,rng);z2=design(arm,x[tr],conf[tr],base_size,pack,.4,rng)
        xt=np.concatenate([xt,z1,z2]);yy=np.tile(yy,3);weights=np.full(len(yy),1/3.)
    clf=fit(tag+'/classifier',LogisticRegression(C=.1,max_iter=2500,solver='lbfgs',random_state=SEED),xt,yy,weights)
    xc=design(arm,x[cal],conf[cal],base_size,pack);raw=clf.predict_proba(xc)[:,1]
    calibrator=fit(tag+'/calibrator',LogisticRegression(C=1,max_iter=1000,random_state=SEED),logit(raw),y[cal])
    cp=calibrator.predict_proba(logit(raw))[:,1];fpr,tpr,th=roc_curve(y[cal],cp)
    eligible=np.flatnonzero((tpr>=.95)&np.isfinite(th));assert len(eligible)
    j=min(eligible,key=lambda k:(fpr[k],-th[k]));pack.update(classifier=clf,calibrator=calibrator,threshold=float(th[j]))
    return pack
def predict(pack,x,conf,dropout=0.,rng=None):
    z=design(pack['arm'],x,conf,pack['base_size'],pack,dropout,rng)
    return pack['calibrator'].predict_proba(logit(pack['classifier'].predict_proba(z)[:,1]))[:,1]
def sample_patients(d,idx,rng):
    patients=d.patient_id.to_numpy();p=np.unique(patients[idx]);groups={v:idx[patients[idx]==v] for v in p}
    return np.concatenate([groups[v] for v in rng.choice(p,len(p),replace=True)])
def diagnostics(d,fd):
    # Missing mechanism categories here are observation patterns, NOT clinical no-P diagnoses.
    rows=[]
    for (role,label),g in d.groupby(['role','label']):
        for col in fd['factors']['M']+['p_candidate_confidence']:
            values=pd.to_numeric(g[col],errors='coerce').to_numpy(float)
            rows.append(dict(role=role,label=int(label),feature=col,ECGs=len(g),patients=g.patient_id.nunique(),missing=int((~np.isfinite(values)).sum()),missing_fraction=float((~np.isfinite(values)).mean()),detector_zero_candidates=int((g.p_candidate_count.fillna(0)==0).sum())))
    pd.DataFrame(rows).to_csv(OUT/'missingness_by_role_and_AF.csv',index=False)
    clinical=Path('DRIVE_ROOT/脑心同查/datasets/shareedb/1.0.0/metadata_only/info.txt')
    if clinical.exists():
        info=pd.read_csv(clinical,sep=r'\\t|\t',engine='python');event=info.iloc[:,-1].astype(str).str.strip()
        put(OUT/'brain_heart_public_feasibility.json',dict(source=str(clinical),SHA256=sha(clinical),records=len(info),columns=list(info.columns),event_counts=event.value_counts().to_dict(),event_time_available=False,death_censoring_verified=False,new_stroke_fits=0,paired_brain_imaging_available=False,decision='Descriptive feasibility only; 3 strokes in official SHAREE description cannot support new high-dimensional stroke risk modelling'))
    prior=pd.read_csv(BASE/'A24_verified_review_20261023/E_paired_intervals_verified.csv');prior.to_csv(OUT/'verified_prior_paired_increments.csv',index=False)
    comparison=[dict(stage='historical_CODE',same_cohort_as_A24=False,native_500Hz_same=False,time_support='different source',interpretation='Not paired algorithm progress'),dict(stage='A17R',same_cohort_as_A24=True,native_500Hz_same=True,time_support='6s E1',interpretation='Label/QC/native preprocessing repair; not comparable to older CODE as causal gain'),dict(stage='A24_R',same_cohort_as_A24=True,native_500Hz_same=True,time_support='historical6s E1',interpretation='100 full patient refits; weak stable M increment'),dict(stage='A24_E',same_cohort_as_A24=True,native_500Hz_same=True,time_support='nominal5s E1/E2 with different filtering context; covariates6s',interpretation='Both encoders source-exposed; matched M beyond VN intervals cross zero')]
    pd.DataFrame(comparison).to_csv(OUT/'cross_stage_comparability.csv',index=False)
def selftest():
    z=np.arange(60,dtype=float).reshape(12,5);z[2,4]=np.nan
    pp=dict(imputer=SimpleImputer(keep_empty_features=True),scaler=StandardScaler())
    pp['scaler'].fit(pp['imputer'].fit_transform(z[:8]));cc=np.ones(12)
    d=design('D3',z,cc,3,pp,1.,np.random.default_rng(1));base=design('D3',z,cc,3,pp)
    assert np.array_equal(d[:,:3],base[:,:3]) and np.all(d[:,3:]==0)
    q=design('D2',z,np.zeros(12),3,pp);assert np.all(q[:,3:]==0)
    assert np.isclose(np.full(36,1/3).sum(),12)
    return dict(VN_unchanged_under_M_intervention='PASS',zero_confidence_attenuation='PASS',augmentation_weight='PASS',no_real_fits=True)
def main():
    assert not (OUT/'started.json').exists(),'No automatic restart'
    put(OUT/'started.json',dict(time=now(),pid=os.getpid()));verify();put(OUT/'mock_tests.json',selftest())
    assert read(BASE/'A24_verified_review_20261023/verification.json')['state']=='PASS'
    d=pd.read_csv(A17/'analysis_dataset.csv',float_precision='round_trip').sort_values('ecg_id').reset_index(drop=True)
    assert len(d)==21002 and d.patient_id.nunique()==18260 and d.ecg_id.is_unique and d.groupby('patient_id').role.nunique().max()==1
    fd=read(A18/'factor_definitions.json');d=decorate(d,fd);diagnostics(d,fd)
    z=np.load(A24/'extraction/common5s_embeddings.npz');assert np.array_equal(z['ecg_id'],d.ecg_id)
    embed=z['E1'];assert embed.shape==(21002,768) and np.isfinite(embed).all()
    bcols=fd['base'];vncols=fd['factors']['V']+fd['factors']['N'];mcols=fd['factors']['M']
    x=np.concatenate([d[bcols].to_numpy(float),embed,d[vncols].to_numpy(float),d[mcols].to_numpy(float)],axis=1);x[~np.isfinite(x)]=np.nan
    conf=d.p_candidate_confidence.to_numpy(float);y=d.label.to_numpy(int);base_size=len(bcols)+768+len(vncols)
    roles={r:np.flatnonzero(d.role.eq(r).to_numpy()) for r in ['train','calibrate','evaluate']}
    assert len(roles['train'])==14709 and len(roles['calibrate'])==2097 and len(roles['evaluate'])==2095
    rows=[];packs={};pred=d.loc[roles['evaluate'],['ecg_id','patient_id','label']].reset_index(drop=True)
    for arm in ARMS:
        check();rng=np.random.default_rng(SEED);pack=train(arm,x,conf,y,roles['train'],roles['calibrate'],base_size,'point/'+arm,rng);packs[arm]=pack
        joblib.dump(pack,OUT/f'{arm}.joblib');put(OUT/'thresholds_frozen.json',{a:p['threshold'] for a,p in packs.items()})
        pp=predict(pack,x[roles['evaluate']],conf[roles['evaluate']]);pred[arm]=pp
        rows.append(dict(arm=arm,threshold=pack['threshold'],**metric(y[roles['evaluate']],pp,pack['threshold'])))
        pd.DataFrame(rows).to_csv(OUT/'point_metrics.csv',index=False);pred.to_csv(OUT/'point_predictions.csv',index=False)
        status('missingness',state='point_fitting',arm=arm,fits=len(read(OUT/'fit_ledger.json')),max_fits=88)
    # Pre-specified sensitivity: mask M only, retain the observed V/N channels exactly.
    stress=[]
    for rate in [0.,.2,.4,.6,.8]:
      for seed in range(5):
       for arm in ARMS:
        check();pp=predict(packs[arm],x[roles['evaluate']],conf[roles['evaluate']],rate,np.random.default_rng(SEED+seed))
        stress.append(dict(arm=arm,rate=rate,seed=seed,mean_abs_prediction_drift=float(np.mean(abs(pp-pred[arm].to_numpy()))),**metric(y[roles['evaluate']],pp,packs[arm]['threshold'])))
    pd.DataFrame(stress).to_csv(OUT/'mask_intervention_stress.csv',index=False)
    # Complete downstream refits: train-only imputation/scaling, classifier, role-specific calibration/threshold, evaluation resampling.
    refits=[]
    for rep in range(20):
        check();rng=np.random.default_rng(SEED+1000+rep)
        sampled={role:sample_patients(d,ix,rng) for role,ix in roles.items()};summary={}
        for arm in ['D0','D3']:
            pack=train(arm,x,conf,y,sampled['train'],sampled['calibrate'],base_size,f'refit_{rep:02d}/'+arm,np.random.default_rng(SEED+2000+rep))
            pp=predict(pack,x[sampled['evaluate']],conf[sampled['evaluate']]);summary[arm]=metric(y[sampled['evaluate']],pp,pack['threshold'])
        for m in ['AUROC','AP','Brier']:refits.append(dict(replicate=rep,metric=m,D0=summary['D0'][m],D3=summary['D3'][m],delta=summary['D3'][m]-summary['D0'][m]))
        pd.DataFrame(refits).to_csv(OUT/'full_refit_paired.csv',index=False);status('missingness',state='patient_refits',completed=rep+1,total=20,fits=len(read(OUT/'fit_ledger.json')))
    r=pd.DataFrame(refits);summ=[]
    for m,g in r.groupby('metric'):summ.append(dict(metric=m,n=len(g),median=float(g.delta.median()),p05=float(g.delta.quantile(.05)),p95=float(g.delta.quantile(.95)),positive=int((g.delta>0).sum()),interpretation='20 full downstream refits, empirical stability distribution, not a precise confidence interval'))
    pd.DataFrame(summ).to_csv(OUT/'full_refit_stability.csv',index=False)
    bags=[];yy=pred.label.to_numpy();patients=pred.patient_id.to_numpy();unique=np.unique(patients);groups={v:np.flatnonzero(patients==v) for v in unique};rng=np.random.default_rng(SEED+3000)
    for i in range(1000):
        if i%100==0:check();status('missingness',state='fixed_model_patient_bootstrap',draw=i,fits=88)
        ix=np.concatenate([groups[v] for v in rng.choice(unique,len(unique),replace=True)]);assert np.unique(yy[ix]).size==2
        for method,base in [('D1','D0'),('D3','D0'),('D3','D1')]:
            a=metric(yy[ix],pred[method].to_numpy()[ix],packs[method]['threshold']);b=metric(yy[ix],pred[base].to_numpy()[ix],packs[base]['threshold'])
            for m in ['AUROC','AP','Brier']:bags.append(dict(draw=i,method=method,baseline=base,metric=m,delta=a[m]-b[m]))
    bag=pd.DataFrame(bags);ci=[];points=pd.DataFrame(rows).set_index('arm')
    for (a,b,m),g in bag.groupby(['method','baseline','metric']):ci.append(dict(method=a,baseline=b,metric=m,delta=float(points.loc[a,m]-points.loc[b,m]),lower=float(g.delta.quantile(.025)),upper=float(g.delta.quantile(.975)),scope='Exploratory fixed-model patient-cluster interval; viewed evaluation; no multiplicity correction'))
    pd.DataFrame(ci).to_csv(OUT/'paired_fixed_model_intervals.csv',index=False)
    # Reload saved numerical outputs, use original strict 1e-12 probability/metric replay tolerance.
    replay=pd.read_csv(OUT/'point_predictions.csv',float_precision='round_trip');errs=[]
    for arm,pack in packs.items():
        saved=joblib.load(OUT/f'{arm}.joblib');prob=predict(saved,x[roles['evaluate']],conf[roles['evaluate']]);error=float(np.max(abs(prob-replay[arm].to_numpy())));assert error<=1e-12
        mm=metric(replay.label.to_numpy(),replay[arm].to_numpy(),pack['threshold'])
        for k,v in mm.items():assert abs(v-points.loc[arm,k])<=1e-12
        errs.append(dict(arm=arm,probability_error=error,metric_replay='PASS'))
    pd.DataFrame(errs).to_csv(OUT/'saved_model_replay.csv',index=False)
    verify();ledger=read(OUT/'fit_ledger.json');assert len(ledger)==88 and all(v['state']=='complete' for v in ledger)
    aux=read(OUT/'preprocessor_ledger.json');assert len(aux)==44
    put(OUT/'completion.json',dict(time=now(),state='complete',fits=88,max_fits=88,imputer_fits=44,scaler_fits=44,neural_epochs=0,new_encoder_forwards=0,full_patient_refits=20,fixed_model_bootstrap=1000,metric_probability_replay='PASS',independent_validation=False,clinical_future_risk_evidence=False,NC_submission_ready=False,SPH_admitted=False,new_figures=0,wall_seconds=time.monotonic()-START))
    status('missingness',state='complete',fits=88)
if __name__=='__main__':
    try:main()
    except Exception as e:status('missingness',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
