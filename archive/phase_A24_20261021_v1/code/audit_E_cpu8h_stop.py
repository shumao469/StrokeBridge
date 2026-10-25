"""Read-only audit of the interrupted8h launch; no retries or model work."""
from a24_shared import *
REC=P/'recovery_E_cpu8h_v1'
def main():
    assert not (REC/'stop_audit_receipt.json').exists()
    launch=read(REC/'science_launch.json');assert sha(P/'code/execute_E_cpu8h_once.py')==launch['code_SHA256'] and sha(REC/'protocol.txt')==launch['protocol_SHA256']
    stop=read(REC/'status.json');guard=read(REC/'preflight_deadline_guard.json');assert stop['state']=='stopped_no_retry' and guard['state']=='own_preflight_child_stopped_no_retry'
    verify(read(REC/'extraction/source_lock.json')['files']);prior_integrity()
    for name in ['extract_controls.py','fit_and_audit.py']:
        q=read(REC/(name+'.transformations.json'));assert sha(q['prior_CPU_source'])==q['prior_CPU_source_SHA256'] and sha(q['original'])==q['original_SHA256'] and sha(q['frozen'])==q['frozen_SHA256']
    assert not (REC/'extraction/encoder_preflight.json').exists() and not (REC/'extraction/preprocessing_admission.json').exists() and not (REC/'extraction/common5s_embeddings.npz').exists() and not (REC/'comparisons/fit_ledger.json').exists()
    r=read(P/'refit/fit_ledger.json');n=read(P/'recovery_N_v1/comparisons/fit_ledger.json');assert len(r)==400 and len(n)==10 and all(q['status']=='complete' for q in r+n)
    result={'time':now(),'state':'CPU8h_interrupted_by_retained180second_preflight_guard','local_zero_fit_audit':'PASS','extraction_cap_authorized_seconds':28800,'retained_preflight_cap_seconds':180,'guard_worker_created':guard['worker_created'],'guard_deadline':guard['deadline'],'guard_action_time':guard['time'],'source_lock_completed_time':read(REC/'extraction/source_lock.json')['time'],'numeric_failure_observed':False,'numeric_preflight_completed':False,'actual_model_initialization_progress':'not fully recorded before interruption; do not claim complete load or zero synthetic calls','full_real_ECG_encoder_inference':0,'E_fits':0,'R_fits':400,'N_fits':10,'total_A24_fits':410,'max_A24_fits':422,'E_fits_reserved_unspent':12,'R_N_rerun':False,'audit_fits':0,'audit_inference':0,'audit_network':0,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'two_branches_preserved':True,'full_result_figure_gate':False,'new_PPTX_delivered':False,'NC_submission_ready':False,'background_worker':False,'next_authority':'Await explicit approval of one corrective run with hashing/initialization/strict preflight/bulk all inside8h and no separate180s resource subcap; no scientific/numerical/fit budget changes.'}
    put(REC/'stop_audit_receipt.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
