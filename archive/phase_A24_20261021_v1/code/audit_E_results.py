"""Zero-fit,zero-network independent audit after successful E-only completion."""
from a24_shared import *
import numpy as np,pandas as pd,joblib
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss
REC=P/'recovery_E_tail_v1';OUT=REC/'audit'
def main():
    assert not (OUT/'completion.json').exists();done=read(REC/'science_completion.json');assert done['state']=='E_science_complete_audit_pending'
    verify(read(REC/'extraction/source_lock.json')['files']);verify(read(REC/'comparisons/source_lock.json')['files']);prior_integrity()
    for f in ['extract_controls.py','fit_and_audit.py']:
        q=read(REC/(f+'.transformations.json'));assert sha(q['original'])==q['original_SHA256'] and sha(q['frozen'])==q['frozen_SHA256']
    weight=read(REC/'weight_completion.json');assert Path(weight['weight']).stat().st_size==372518672 and sha(weight['weight'])==weight['SHA256']=='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
    preprocessing=read(REC/'extraction/preprocessing_admission.json');preflight=read(REC/'extraction/encoder_preflight.json');assert preprocessing['state']==preflight['state']=='PASS'
    assert preprocessing['records']==32 and preflight['all_frozen'] and preflight['strict_state_load'] and preflight['ECGFM_upstream_parity']<=1e-6 and preflight['HuBERT_batch_error']<=1e-5
    extraction=read(REC/'extraction/completion.json');assert extraction['P_network_ECGs']==0 and extraction['N_features_reused']
    r=read(P/'refit/fit_ledger.json');n=read(P/'recovery_N_v1/comparisons/fit_ledger.json');e=read(REC/'comparisons/fit_ledger.json');assert len(r)==400 and len(n)==10 and len(e)==12 and all(q['status']=='complete' for q in r+n+e)
    assert all(q['name'].startswith(('E1','E2')) for q in e) and len(r+n+e)==422
    d=cohort();assert d.groupby('patient_id').role.nunique().max()==1
    z=np.load(REC/'extraction/common5s_embeddings.npz');assert np.array_equal(z['ecg_id'],d.ecg_id)
    for enc in ['E1','E2']:assert z[enc].shape==(21002,768) and np.isfinite(z[enc]).all()
    pred=csv(REC/'comparisons/evaluation_predictions.csv');test=d[d.role.eq('evaluate')].reset_index(drop=True)
    for c in ['ecg_id','patient_id','label']:assert np.array_equal(pred[c],test[c])
    met=csv(REC/'comparisons/model_metrics.csv').set_index('arm');frozen=read(REC/'comparisons/models_frozen_before_evaluation.json');assert set(met.index)=={'E1','E1_VN','E1_VNM','E2','E2_VN','E2_VNM'}
    yy=pred.label.to_numpy();replay=[]
    for arm,row in met.iterrows():
        file=REC/'comparisons/models'/(arm+'.joblib');assert sha(file)==frozen['models'][file.name];pack=joblib.load(file);assert pack['threshold']==row.threshold==frozen['thresholds'][arm]
        p=pred[arm].to_numpy();assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all();v=p>=row.threshold
        tp=int(sum(v&(yy==1)));fp=int(sum(v&(yy==0)));fn=int(sum(~v&(yy==1)));tn=int(sum(~v&(yy==0)))
        metrics={'AUROC':roc_auc_score(yy,p),'AP':average_precision_score(yy,p),'Brier':brier_score_loss(yy,p),'TP':tp,'FP':fp,'FN':fn,'TN':tn,'sensitivity':tp/(tp+fn),'specificity':tn/(tn+fp),'PPV':tp/max(tp+fp,1)}
        for key,value in metrics.items():assert np.isclose(value,row[key],rtol=0,atol=1e-12),(arm,key)
        replay.append({'arm':arm,**metrics,'independent_metric_replay':'PASS'})
    pd.DataFrame(replay).to_csv(OUT/'independent_metric_replay.csv',index=False)
    pairs=csv(REC/'comparisons/paired_intervals.csv')
    for q in pairs.to_dict('records'):
        assert np.isclose(q['delta'],met.loc[q['method'],q['metric']]-met.loc[q['baseline'],q['metric']],rtol=0,atol=1e-12) and q['confidence']==.95 and q['lower']<=q['upper']
    result={'time':now(),'state':'E_independent_zero_fit_audit_PASS','R_fits':400,'N_fits':10,'E_fits':12,'total_fits':422,'max_fits':422,'records':21002,'patients':18260,'evaluation_ECGs':2095,'evaluation_patients':1822,'evaluation_AF_ECGs':150,'patient_role_overlap':0,'asset_gate':'PASS','preprocessing_gate':'PASS','encoder_preflight':'PASS','metric_replay':'PASS','source_hashes':'PASS','P_network_forwards_in_E':0,'R_N_rerun':False,'new_audit_fits':0,'new_audit_network':0,'new_audit_inference':0,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'independent_unexposed_validation':False,'future_risk_prediction':False,'NC_submission_ready':False,'eligible_for_result_figure_generation':True,'metrics':met.reset_index().to_dict('records'),'paired_intervals':pairs.to_dict('records'),'scope':'Viewed source-exposed concurrent AF; nominal5s matched,preprocessing context differs; exploratory pointwise95%fixed-model patient-cluster intervals.'}
    put(OUT/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':
    OUT.mkdir(exist_ok=True)
    try:main()
    except Exception as e:put(OUT/'audit_failure.json',{'time':now(),'state':'audit_failed_no_refit','error':str(e),'fits':0,'inference':0});raise
