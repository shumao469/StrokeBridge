"""Local-only stop audit. No network, model load, inference or fitting."""
from a24_shared import *
import struct
REC=P/'recovery_E_v1';ROOT=Path('DRIVE_ROOT/脑心同查/models/hubert-ecg-base/8ffa1ad3593366ca45f5351d75317d4ff8f14465')

def main():
    assert not (REC/'stop_audit_receipt.json').exists(),'Keep existing audit receipt'
    launch=read(REC/'download_launch.json');assert launch['code_SHA256']==sha(P/'code/restore_E_weight_once.py') and launch['protocol_SHA256']==sha(REC/'protocol.txt')
    req=read(REC/'request_started.json');head=read(REC/'response_headers.json');progress=read(REC/'download_progress.json');stop=read(REC/'download_stop.json')
    assert stop['state']=='stopped_no_retry' and req['logical_request']==1
    old=ROOT/'model.safetensors.partial';part=ROOT/'model.safetensors.A24_E_recovery.partial'
    oldhash=sha(old);parthash=sha(part);assert old.stat().st_size==96509823 and oldhash=='efedb093538fe92106abb3d5a21b49b42731aabc241c502cfafd075110daf74a'
    meta=read(P/'source_audit/HuBERT_hf_api.json');asset=next(q for q in meta['siblings'] if q['rfilename']=='model.safetensors')
    assert meta['sha']=='8ffa1ad3593366ca45f5351d75317d4ff8f14465' and asset['size']==372518672 and asset['lfs']['sha256']=='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
    prefix=hashlib.sha256()
    with part.open('rb') as f:
        left=old.stat().st_size
        while left:
            block=f.read(min(left,8*1024*1024));assert block;prefix.update(block);left-=len(block)
        f.seek(0);hlen=struct.unpack('<Q',f.read(8))[0];assert hlen<1024*1024
        header=json.loads(f.read(hlen));ends=[v['data_offsets'][1] for k,v in header.items() if k!='__metadata__'];header_expected=8+hlen+max(ends)
    assert prefix.hexdigest()==oldhash and header_expected==asset['size']
    received=part.stat().st_size-old.stat().st_size;missing=asset['size']-part.stat().st_size
    assert received==progress['received_additional_bytes'] and 0<missing and received<=276100000 and stop['wall_seconds']<900
    assert head['status']==206 and head['Content-Range']=='bytes 96509823-372518671/372518672'
    assert not (ROOT/'model.safetensors').exists() and not (REC/'weight_completion.json').exists()
    assert not (P/'extraction/started.json').exists() and not (REC/'extraction/started.json').exists()
    for path in [P/'comparisons/fit_ledger.json',REC/'comparisons/fit_ledger.json']:assert not path.exists()
    rledger=read(P/'refit/fit_ledger.json');nledger=read(P/'recovery_N_v1/comparisons/fit_ledger.json')
    assert len(rledger)==400 and len(nledger)==10 and all(q['status']=='complete' for q in rledger+nledger)
    verify(read(P/'refit/freeze_before_fitting.json')['files']);verify(read(P/'recovery_N_v1/negative/source_lock.json')['files']);verify(read(P/'recovery_N_v1/comparisons/source_lock.json')['files']);prior_integrity()
    result={'time':now(),'state':'authorized_E_resume_failed_short_body_stopped','local_zero_fit_audit':'PASS','R_fits':400,'N_fits':10,'E_fits':0,'total_A24_fits':410,'max_A24_fits':422,'reserved_E_fits_unspent':12,'logical_official_requests':1,'official_redirects':read(REC/'redirects.json'),'HTTP_status':head['status'],'response_header_expected_additional_bytes':int(head['Content-Length']),'actual_additional_network_bytes':received,'max_additional_bytes':276100000,'resume_file_bytes':part.stat().st_size,'expected_full_bytes':asset['size'],'missing_bytes':missing,'acquisition_wall_seconds':stop['wall_seconds'],'max_acquisition_seconds':900,'original_partial_bytes':old.stat().st_size,'original_partial_SHA256':oldhash,'recovery_partial_SHA256_for_provenance_only':parthash,'prefix_copy_SHA256_matches':True,'safetensors_header_expected_full_bytes':header_expected,'tensor_entries':len(ends),'official_complete_size_pass':False,'official_full_SHA_pass':False,'preprocessing_admission':'not executed because asset gate failed','E_weight_loaded':False,'E_inference_ECGs':0,'new_audit_network_requests':0,'new_audit_inference_calls':0,'new_audit_fits':0,'R_N_rerun':False,'raw_data_changed':False,'SPH_admitted':False,'SPH_predictions':0,'SPH94_deleted':False,'two_branches_preserved':True,'full_A24_figure_gate':False,'new_PPTX_delivered':False,'NC_submission_ready':False,'independent_validation':False,'background_worker':False,'cause_statement':'Observed complete-looking206headers but an incomplete body. Local evidence does not identify whether source,CDN,proxy or transport caused the short body. No retry authorized.'}
    put(REC/'stop_audit_receipt.json',result);print(json.dumps(result,ensure_ascii=False))

if __name__=='__main__':main()
