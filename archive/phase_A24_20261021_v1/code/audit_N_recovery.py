"""Independent zero-fit numerical/provenance closeout of the approved N recovery."""
from a24_shared import *
import numpy as np,pandas as pd,joblib
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss

REC=P/'recovery_N_v1';OUT=REC/'audit'

def main():
    assert not (OUT/'completion.json').exists(),'Do not overwrite an existing audit'
    completed=read(REC/'completion.json');assert completed['state']=='N_recovery_complete_E_still_ineligible'
    launch=read(REC/'launch.json');assert sha(P/'code/recover_N_once.py')==launch['code_SHA256'];assert sha(REC/'protocol.txt')==launch['protocol_SHA256']
    for name in ['negative_controls.py','fit_and_audit.py']:
        transform=read(REC/(name+'.transformations.json'))
        assert sha(transform['original_path'])==transform['original_SHA256']
        assert sha(REC/(name+'.frozen.txt'))==transform['transformed_SHA256']
    put(OUT/'audit_started.json',{'time':now(),'audit_code_SHA256':sha(__file__),'audit_plan_SHA256':sha(REC/'audit_plan.txt'),'fits':0,'inference_calls':0})
    before=read(REC/'negative/first32_replay_PASS.json');assert before['records']==32 and before['original_tolerances_unchanged'] is True
    for module in ['negative','comparisons']:verify(read(REC/module/'source_lock.json')['files'])
    verify(read(P/'refit/freeze_before_fitting.json')['files']);prior_integrity()
    assert read(P/'negative/status.json')['state']=='stopped_no_retry'
    assert not (P/'extraction/started.json').exists() and not (P/'comparisons/fit_ledger.json').exists()
    rledger=read(P/'refit/fit_ledger.json');nledger=read(REC/'comparisons/fit_ledger.json')
    assert len(rledger)==400 and len(nledger)<=10 and len(rledger)+len(nledger)<=422
    assert all(q['status']=='complete' for q in rledger+nledger)
    assert len(nledger)==completed['N_fits']
    original=cohort();current=csv(REC/'negative/current_candidates.csv');assert np.array_equal(original.ecg_id,current.ecg_id)
    assert original.groupby('patient_id').role.nunique().max()==1
    roles=[]
    for role,g in original.groupby('role'):roles.append({'role':role,'ECGs':len(g),'patients':g.patient_id.nunique(),'AF':int(g.label.sum())})
    pd.DataFrame(roles).to_csv(OUT/'patient_denominators.csv',index=False)
    replay=[]
    for c in current.columns:
        if c=='ecg_id' or c not in original:continue
        a=original[c].to_numpy(float);b=current[c].to_numpy(float);finite=np.isfinite(a)&np.isfinite(b)
        exact=np.isclose(a,b,rtol=0,atol=0,equal_nan=True);near=np.isclose(a,b,rtol=1e-5,atol=1e-6,equal_nan=True)
        replay.append({'feature':c,'records':len(a),'exact_equal':int(exact.sum()),'within_original_continuous_tolerance':int(near.sum()),'finite_mask_changes':int((np.isfinite(a)!=np.isfinite(b)).sum()),'max_abs_finite_difference':float(np.max(np.abs(a[finite]-b[finite]))) if finite.any() else None})
    pd.DataFrame(replay).to_csv(OUT/'full_candidate_replay_descriptive.csv',index=False)
    counts=read(REC/'negative/completion.json');assert counts['count_agreement']>=.995 and counts['records']==len(original)
    factors=read(B/'factor_definitions.json');M=factors['factors']['M'];controls=[]
    for name in ['displaced_morphology','anchor_morphology']:
        data=csv(REC/'negative'/(name+'.csv'));assert np.array_equal(data.ecg_id,current.ecg_id)
        for c in M:
            a=current[c].to_numpy(float);b=data[c].to_numpy(float);finite=np.isfinite(a)
            controls.append({'control':name,'feature':c,'original_finite':int(finite.sum()),'new_missing_at_original_finite':int((finite&~np.isfinite(b)).sum()),'changed_at_original_finite':int((finite&~np.isclose(a,b,rtol=0,atol=0,equal_nan=True)).sum()),'unchanged_at_original_finite':int((finite&np.isclose(a,b,rtol=0,atol=0,equal_nan=True)).sum())})
    pd.DataFrame(controls).to_csv(OUT/'control_changes_and_missingness.csv',index=False)
    pred=csv(REC/'comparisons/evaluation_predictions.csv');test=original[original.role.eq('evaluate')].reset_index(drop=True)
    for c in ['ecg_id','patient_id','label']:assert np.array_equal(test[c],pred[c])
    met=csv(REC/'comparisons/model_metrics.csv').set_index('arm');frozen=read(REC/'comparisons/models_frozen_before_evaluation.json')
    yy=pred.label.to_numpy();recalc=[]
    for arm,row in met.iterrows():
        model=REC/'comparisons/models'/(arm+'.joblib');assert sha(model)==frozen['models'][model.name]
        pack=joblib.load(model);assert pack['threshold']==frozen['thresholds'][arm]==row.threshold
        assert len(set(pack['columns']))==len(pack['columns'])
        pp=pred[arm].to_numpy();assert np.isfinite(pp).all() and ((pp>=0)&(pp<=1)).all();zz=pp>=row.threshold
        tp=int(sum(zz&(yy==1)));fp=int(sum(zz&(yy==0)));fn=int(sum(~zz&(yy==1)));tn=int(sum(~zz&(yy==0)))
        values={'AUROC':roc_auc_score(yy,pp),'AP':average_precision_score(yy,pp),'Brier':brier_score_loss(yy,pp),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'PPV':tp/max(tp+fp,1)}
        for k,v in values.items():assert np.isclose(v,row[k],rtol=0,atol=1e-12),(arm,k)
        recalc.append({'arm':arm,**values,'metric_replay':'PASS'})
    pd.DataFrame(recalc).to_csv(OUT/'independent_metric_replay.csv',index=False)
    paired=csv(REC/'comparisons/paired_intervals.csv')
    for row in paired.to_dict('records'):
        assert np.isclose(row['delta'],met.loc[row['method'],row['metric']]-met.loc[row['baseline'],row['metric']],rtol=0,atol=1e-12)
        assert row['confidence']==.95 and row['lower']<=row['upper']
    audit={'time':now(),'state':'zero_fit_N_recovery_closeout_PASS','N_recovery_state':completed['state'],'records':len(current),'patients':int(original.patient_id.nunique()),'evaluation_ECGs':len(pred),'evaluation_patients':int(pred.patient_id.nunique()),'evaluation_AF':int(pred.label.sum()),'R_fits':400,'N_fits':len(nledger),'E_fits':0,'total_fits':400+len(nledger),'global_limit':422,'new_audit_fits':0,'new_audit_inference_calls':0,'recovery_wall_seconds':read(REC/'status.json')['wall_seconds'],'count_agreement':counts['count_agreement'],'candidate_replay_columns':len(replay),'numerical_replay':'PASS','source_locks':'PASS','patient_role_overlap':0,'control_admission':read(REC/'comparisons/control_admission.json'),'metrics':met.reset_index().to_dict('records'),'paired_intervals':paired.to_dict('records'),'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'independent_validation':False,'future_risk_prediction':False,'full_A24_figure_gate':False,'NC_submission_ready':False,'E_recovery_authorized':False,'E_weight_loaded':False,'original_N_failure_preserved':True,'background_worker':False,'interpretation_limits':['Concurrent AF, viewed development; no independent unseen validation.','Detector V/N and missingness are preserved in controls; no target-domain P identity reference.','Displacement retains candidate duration/geometry; a shifted window is not confirmed no-P.','Featurewise permutation does not preserve joint morphology covariance.','Skipped controls are admission failures, not zero-effect scientific findings.','Fixed-model pointwise95% intervals do not include fitting/model-selection uncertainty.','R100 empirical refit distribution is a different uncertainty analysis, not a confirmatory95%CI.']}
    put(OUT/'completion.json',audit)
    print(json.dumps({'state':audit['state'],'N_fits':len(nledger),'total_fits':audit['total_fits'],'count_agreement':audit['count_agreement'],'models':list(met.index),'AP_intervals':paired[paired.metric.eq('AP')].to_dict('records')},ensure_ascii=False))

if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except Exception as e:put(OUT/'failed_audit_no_refit.json',{'time':now(),'error':str(e),'fits':0,'inference':0});raise
