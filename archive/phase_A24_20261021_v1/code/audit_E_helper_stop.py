"""Audit the helper binding interruption. Zero new trained-model inference/fits."""
from a24_shared import *
from datetime import timedelta
REC=P/'recovery_E_cpu8h_corrected_v1'
def main():
    assert not (REC/'stop_audit_receipt.json').exists()
    launch=read(REC/'science_launch.json');assert sha(P/'code/execute_E_cpu8h_corrected_once.py')==launch['code_SHA256'] and sha(REC/'protocol.txt')==launch['protocol_SHA256']
    assert read(REC/'status.json')['state']=='stopped_no_retry';child=read(REC/'extract_controls.py.stop.json');assert "name 'now' is not defined" in child['error']
    verify(read(REC/'extraction/source_lock.json')['files']);prior_integrity()
    assert not (REC/'extraction/common5s_embeddings.npz').exists() and not (REC/'extraction/preprocessing_admission.json').exists() and not (REC/'comparisons/fit_ledger.json').exists()
    assert len(read(P/'refit/fit_ledger.json'))==400 and len(read(P/'recovery_N_v1/comparisons/fit_ledger.json'))==10
    mock=read(P/'software_contracts_E_helper_v1/completion.json');assert mock['state']=='mock_only_helper_contracts_PASS' and mock['real_encoder_forwards']==mock['fits']==0
    verify(read(P/'software_contracts_E_helper_v1/source_lock.json')['files'])
    deadline=(datetime.fromisoformat(launch['time'])+timedelta(hours=8)).isoformat()
    result={'time':now(),'state':'corrective_run_stopped_on_missing_now_helper_binding','cause':'Implementation error in newly added timing callback namespace,not an observed numerical/source/cohort failure. No automatic retry.','source_audit':'PASS','mock_only_helper_fix_contracts':'PASS','mock_test_real_encoder_forwards':0,'mock_test_fits':0,'mock_test_ECG_reads':0,'original8h_start':launch['time'],'original8h_deadline_not_extended':deadline,'full_real_ECG_encoder_inference':0,'E_fits':0,'R_fits':400,'N_fits':10,'total_A24_fits':410,'max_A24_fits':422,'E_fits_reserved_unspent':12,'R_N_rerun':False,'audit_inference':0,'audit_fits':0,'audit_network':0,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'two_branches_preserved':True,'full_result_figure_gate':False,'new_PPTX_delivered':False,'NC_submission_ready':False,'background_worker':False,'next_authority':'Await explicit approval to bind the helper and recover before the SAME original8hdeadline after mock tests;no new scientific/numeric/fit budget.'}
    put(REC/'stop_audit_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
