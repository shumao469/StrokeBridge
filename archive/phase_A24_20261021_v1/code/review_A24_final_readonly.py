"""Fresh zero-fit A24 review: source, patient roles, saved-model probabilities and intervals."""
from pathlib import Path
import json,sys,time,traceback
import execute_E_wsl_20261023_v2 as x
e=x.e;P=e.P;REC=x.REC;U=e.shared.U;OUT=U/'A24_verified_review_20261023'
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
from threadpoolctl import threadpool_limits

def main():
    start=time.monotonic();OUT.mkdir(exist_ok=True)
    assert not (OUT/'verification.json').exists(),'Do not replace a completed review receipt'
    audit=e.read(REC/'audit/completion.json')
    assert audit['state']=='E_independent_zero_fit_audit_PASS'
    assert e.read(REC/'audit/checkpoint_replay.json')['state']=='PASS'
    for module in ('extraction','comparisons'):e.verify(e.read(REC/module/'source_lock.json')['files'])
    e.prior_integrity()
    for root,n in [(P/'refit',400),(P/'recovery_N_v1/comparisons',10),(REC/'comparisons',12)]:
        rows=e.read(root/'fit_ledger.json');assert len(rows)==n and all(r['status']=='complete' for r in rows)
    d=e.shared.cohort();test=d[d.role.eq('evaluate')].reset_index(drop=True)
    roles=d.groupby('role').agg(ECGs=('ecg_id','size'),patients=('patient_id','nunique'),AF=('label','sum')).reset_index()
    roles.to_csv(OUT/'cohort_denominators.csv',index=False)
    assert d.groupby('patient_id').role.nunique().max()==1
    z=np.load(REC/'extraction/common5s_embeddings.npz')
    assert np.array_equal(z['ecg_id'],d.ecg_id)
    for enc in ('E1','E2'):assert z[enc].shape==(21002,768) and np.isfinite(z[enc]).all()
    # Independently reconstruct the V columns from their frozen factor definitions.
    fd=e.read(e.shared.B/'factor_definitions.json');frame=test.copy()
    for name in fd['factors']['V']:
        if name.startswith('V_present__'):
            source=name.split('__',1)[1];frame[name]=(frame[source].fillna(0)>0).astype(float)
        elif name.startswith('V_missing__'):
            source=name.split('__',1)[1];frame[name]=(~np.isfinite(frame[source])).astype(float)
        else:assert name in frame.columns,('Frozen V field not available',name)
    pred=e.shared.csv(REC/'comparisons/evaluation_predictions.csv')
    met=e.shared.csv(REC/'comparisons/model_metrics.csv').set_index('arm')
    frozen=e.read(REC/'comparisons/models_frozen_before_evaluation.json')
    for c in ('ecg_id','patient_id','label'):assert np.array_equal(pred[c],test[c])
    assert len(pred)==2095 and pred.patient_id.nunique()==1822 and pred.label.sum()==150
    replay=[];mask=d.role.eq('evaluate').to_numpy()
    with threadpool_limits(limits=2):
        for arm,row in met.iterrows():
            enc=arm.split('_',1)[0];names=[f'{enc}_{j:03d}' for j in range(768)]
            inputs=pd.concat([frame,pd.DataFrame(z[enc][mask],columns=names)],axis=1)
            file=REC/'comparisons/models'/(arm+'.joblib')
            assert e.sha(file)==frozen['models'][file.name]
            pack=joblib.load(file);assert pack['threshold']==row.threshold==frozen['thresholds'][arm]
            a=inputs[pack['columns']].replace([np.inf,-np.inf],np.nan).to_numpy(float)
            raw=np.clip(pack['classifier'].predict_proba(a)[:,1],1e-6,1-1e-6)
            probability=pack['calibrator'].predict_proba(np.log(raw/(1-raw)).reshape(-1,1))[:,1]
            saved=pred[arm].to_numpy();err=float(np.max(np.abs(saved-probability)))
            assert err<=1e-12,(arm,err)
            yy=pred.label.to_numpy();label=saved>=pack['threshold']
            tp=int(sum(label&(yy==1)));fp=int(sum(label&(yy==0)));fn=int(sum(~label&(yy==1)));tn=int(sum(~label&(yy==0)))
            metric={'AUROC':roc_auc_score(yy,saved),'AP':average_precision_score(yy,saved),'Brier':brier_score_loss(yy,saved),
                    'TP':tp,'FP':fp,'FN':fn,'TN':tn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'PPV':tp/(tp+fp)}
            for name,value in metric.items():assert np.isclose(value,row[name],rtol=0,atol=1e-12),(arm,name)
            replay.append({'arm':arm,'saved_model_probability_max_abs_error':err,'threshold':pack['threshold'],**metric})
    pd.DataFrame(replay).to_csv(OUT/'saved_model_probability_and_metric_replay.csv',index=False)
    intervals=e.shared.csv(REC/'comparisons/paired_intervals.csv')
    for r in intervals.itertuples():
        assert np.isclose(r.delta,met.loc[r.method,r.metric]-met.loc[r.baseline,r.metric],rtol=0,atol=1e-12)
        assert r.lower<=r.upper and r.confidence==.95
    met.reset_index().to_csv(OUT/'E_metrics_verified.csv',index=False)
    intervals.to_csv(OUT/'E_paired_intervals_verified.csv',index=False)
    # Check all21 stored chunks against the complete embedding arrays without any encoder forward.
    offset=0
    for r in e.read(REC/'extraction/checkpoints/manifest.json'):
        assert r['start']==offset and e.sha(r['file'])==r['SHA256'];part=np.load(r['file']);end=r['end_exclusive']
        for key in ('ecg_id','E1','E2'):assert np.array_equal(part[key],z[key][offset:end])
        offset=end
    assert offset==21002
    result={'time':e.now(),'state':'PASS','scientific_completion_time':e.read(REC/'numerical_completion.json')['time'],
        'ECGs':21002,'patients':18260,'evaluation_ECGs':2095,'evaluation_patients':1822,'evaluation_AF_ECGs':150,
        'patient_role_overlap':0,'source_hashes':'PASS','saved_model_probability_replay':'PASS',
        'maximum_probability_replay_error':max(r['saved_model_probability_max_abs_error'] for r in replay),
        'metric_replay':'PASS','paired_point_delta_replay':'PASS','checkpoint_array_replay':'PASS',
        'R_fits':400,'N_fits':10,'E_fits':12,'total_fits':422,'new_review_fits':0,'new_encoder_forwards':0,
        'saved_classifier_probability_replays':6,'new_figures':0,'new_PPTX':0,
        'independent_unexposed_validation':False,'future_risk_evidence':False,'NC_submission_ready':False,
        'ready_to_organize_technical_manuscript':True,'wall_seconds':time.monotonic()-start}
    e.put(OUT/'verification.json',result);print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':
    try:main()
    except Exception as exc:
        e.put(OUT/'verification_failure.json',{'time':e.now(),'state':'FAIL','error':str(exc),'traceback':traceback.format_exc(),'fits':0})
        raise
