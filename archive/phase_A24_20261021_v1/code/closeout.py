"""Read-only closeout of the bounded partial run; never starts or retries a worker."""
from a24_shared import *
import pandas as pd,numpy as np
def main():
    r=read(P/'refit/completion.json');assert r['state']=='complete' and r['fits']==400
    ledger=read(P/'refit/fit_ledger.json');assert len(ledger)==400 and all(q['status']=='complete' for q in ledger)
    verify(read(P/'refit/freeze_before_fitting.json')['files']);prior_integrity()
    metrics=csv(P/'refit/replicate_metrics.csv');diff=csv(P/'refit/paired_refit_differences.csv')
    assert len(metrics)==200 and metrics.replicate.nunique()==100 and not metrics.duplicated(['replicate','arm']).any()
    for metric in diff.metric.unique():
        pivot=metrics.pivot(index='replicate',columns='arm',values=metric);expected=pivot.FM_VNM-pivot.FM_VN;actual=diff[diff.metric.eq(metric)].set_index('replicate').delta
        assert np.allclose(expected.loc[actual.index],actual,rtol=0,atol=1e-12),metric
    diag=read(P/'diagnostic/precision_comparison.json');assert diag['fits']==0 and diag['network_calls']==4
    assert not (P/'comparisons/fit_ledger.json').exists() and not (P/'extraction/started.json').exists()
    state={'time':now(),'run':P.name,'state':'bounded_partial_complete_stopped_awaiting_explicit_recovery','fits_completed':400,'max_fits':422,'R_replicates_completed':100,'R_fits':400,'R_replay':'PASS','N_fits':0,'E_fits':0,'N_state':'stopped at first32 replay gate on ECG1 candidate confidence; no cohort rerun','N_diagnostic':'On ECG1,CuDNN TF32 False gives confidence difference -0.0000363886356; True matches the historical value exactly. Does not certify the rest of the cohort. Original tolerance unchanged.','E_state':'Official weight incomplete96,509,823/372,518,672bytes; never loaded; no unapproved retry','new_network_training_epochs':0,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'original_models_and_thresholds_changed':False,'background_worker':False,'scheduled_automation':False,'new_figure_job_started':False,'new_PPTX_delivered':False,'NC_submission_ready':False,'independent_validation':False,'branches':{'A':'public technical work; R complete, E/N require separately bounded recovery','B':'hospital and independent target-domain expert truth waiting'},'refit_summary':csv(P/'refit/refit_distribution_summary.csv').to_dict('records'),'protocol':'runs/'+P.name+'/protocol.txt','index':'runs/'+P.name+'/index.html','recovery_authorization_received':False}
    put(P/'partial_completion_receipt.json',state);put(U/'PHASE_A24_STATUS.json',state)
    print(json.dumps({'state':state['state'],'fits':400,'R_replicates':100,'refit_AP':state['refit_summary'][0],'background_worker':False,'figures_started':False},ensure_ascii=False))
if __name__=='__main__':main()
