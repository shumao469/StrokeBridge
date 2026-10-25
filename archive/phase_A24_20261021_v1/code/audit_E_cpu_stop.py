"""Local zero-fit audit of successful asset/preprocessing and failed time admission."""
from a24_shared import *
REC=P/'recovery_E_cpu_v1';TAIL=P/'recovery_E_tail_v1'
def main():
    assert not (REC/'stop_audit_receipt.json').exists()
    launch=read(REC/'science_launch.json');assert launch['code_SHA256']==sha(P/'code/execute_E_cpu_once.py') and launch['protocol_SHA256']==sha(REC/'protocol.txt')
    assert read(REC/'status.json')['state']=='stopped_no_retry';verify(read(REC/'extraction/source_lock.json')['files']);prior_integrity()
    for name in ['extract_controls.py','fit_and_audit.py']:
        q=read(REC/(name+'.transformations.json'));assert sha(q['original'])==q['original_SHA256'] and sha(q['frozen'])==q['frozen_SHA256']
    weight=read(TAIL/'weight_completion.json');assert Path(weight['weight']).stat().st_size==372518672 and sha(weight['weight'])==weight['SHA256']=='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
    for file,h in [('model.safetensors.partial','efedb093538fe92106abb3d5a21b49b42731aabc241c502cfafd075110daf74a'),('model.safetensors.A24_E_recovery.partial','fbe9556c414d27af4f6027f248cc3b9913a9255709cfb359e9c664068ca24411')]:assert sha(Path(weight['weight']).parent/file)==h
    ep=read(REC/'extraction/encoder_preflight.json');pp=read(REC/'extraction/preprocessing_admission.json');bench=read(REC/'extraction/CPU_benchmark.json')
    assert ep['state']==pp['state']=='PASS' and ep['device']=='cpu' and ep['strict_state_load'] and ep['all_frozen']
    assert ep['ECGFM_upstream_parity']<=1e-6 and ep['HuBERT_repeat_exact'] and ep['HuBERT_batch_error']<=1e-5
    assert pp['records']==32 and all(q['E1_max_abs_error']==q['E2_max_abs_error']==0 for q in pp['first32_sorted_ids'])
    assert bench['threads']==2 and bench['GPU_forward_calls']==0 and not bench['admitted_for_bulk'] and bench['preflight_wall_seconds']<180
    assert bench['projected_full_extraction_seconds']>bench['admission_projection_limit_seconds']
    assert not (REC/'extraction/common5s_embeddings.npz').exists() and not (REC/'extraction/completion.json').exists() and not (REC/'comparisons/fit_ledger.json').exists()
    r=read(P/'refit/fit_ledger.json');n=read(P/'recovery_N_v1/comparisons/fit_ledger.json');assert len(r)==400 and len(n)==10 and all(q['status']=='complete' for q in r+n)
    verify(read(P/'refit/freeze_before_fitting.json')['files']);verify(read(P/'recovery_N_v1/negative/source_lock.json')['files']);verify(read(P/'recovery_N_v1/comparisons/source_lock.json')['files'])
    q={'time':now(),'state':'asset_CPU_numerical_preprocessing_PASS_time_admission_FAILED_stopped','local_zero_fit_audit':'PASS','asset_size':372518672,'asset_SHA256':weight['SHA256'],'asset_path':weight['weight'],'tail_additional_bytes':688130,'tail_logical_requests':1,'tail_wall_seconds':weight['wall_seconds'],'previous_partials_preserved':True,'GPU_wait_stopped_before_inference':True,'encoder_preflight_device':'cpu','strict_load_frozen':'PASS','ECGFM_upstream_parity':ep['ECGFM_upstream_parity'],'HuBERT_repeat_exact':ep['HuBERT_repeat_exact'],'HuBERT_batch_error':ep['HuBERT_batch_error'],'native_preprocessing_records':32,'native_preprocessing_exact_parity':'PASS','CPU_threads':2,'preflight_wall_seconds':bench['preflight_wall_seconds'],'projected_extraction_seconds':bench['projected_full_extraction_seconds'],'projection_limit_seconds':6900,'hard_extraction_limit_seconds':7200,'time_admission':False,'full_real_record_encoder_inference':0,'synthetic_preflight_example_forwards':{'E1':14,'E2':16},'E_classifier_and_calibrator_fits':0,'R_fits':400,'N_fits':10,'total_A24_fits':410,'max_A24_fits':422,'E_fits_reserved_unspent':12,'R_N_rerun':False,'audit_network':0,'audit_inference':0,'audit_fits':0,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'two_branches_preserved':True,'hospital_expert':'waiting','independent_validation':False,'future_risk_prediction':False,'full_result_figure_gate':False,'new_PPTX_delivered':False,'NC_submission_ready':False,'background_worker':False}
    put(REC/'stop_audit_receipt.json',q);print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':main()
