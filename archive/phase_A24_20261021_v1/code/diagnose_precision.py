from a24_shared import *
import torch,numpy as np,wfdb
from extract_controls import definitions,LEADS
O=P/'diagnostic';O.mkdir(exist_ok=True);START=time.monotonic()
def main():
    assert not (O/'started.json').exists(),'No repeat of diagnostic'
    plan=P/'diagnostic_only_first_failure.txt';pw=P/'runtime/P_C1_frozen.pt';assert sha(pw)=='98557608ee9c581bc0e0ee4f1f0578ad4c14f961952357b5461782f349b341d4'
    put(O/'started.json',{'time':now(),'plan_SHA256':sha(plan),'code_SHA256':sha(__file__),'max_network_calls':4,'max_seconds':300,'fits':0,'N_remains_stopped':True})
    prior_integrity();d=cohort();row=d[d.ecg_id.eq(1)].iloc[0];au=csv(A/'waveform_audit.csv').set_index('ecg_id').loc[1]
    meta=next(k for k in read(A/'source_lock.json')['files'] if k.endswith('/ptbxl_database.csv'));file=Path(meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')).parent/row.filename_hr
    assert sha(file.with_suffix('.hea'))==au.header_SHA256 and sha(file.with_suffix('.dat'))==au.data_SHA256
    r=wfdb.rdrecord(str(file));names=[v.upper() for v in r.sig_name];x=r.p_signal[:,[names.index(n) for n in LEADS]]
    torch.set_num_threads(2);torch.cuda.set_device(1);torch.cuda.set_per_process_memory_fraction(.25,1);s=definitions();m=s['Net']();m.load_state_dict(torch.load(pw,map_location='cpu',weights_only=True),strict=True);m=m.to('cuda:1').eval().requires_grad_(False)
    phys,norm,_=s['prepare_raw'](x);rr,_=s['r_features'](phys);xx=np.stack([norm.T,np.gradient(norm.T,axis=-1)*5],axis=1);t=torch.from_numpy(xx).to('cuda:1');out=[]
    for tf32 in [False,True]:
        torch.backends.cudnn.allow_tf32=tf32;torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.benchmark=False
        with torch.inference_mode():prob=((m(t)+m(-t))/2).softmax(1).cpu().numpy()
        f=s['p_features'](phys,norm,prob,rr);keys=['p_candidate_confidence','p_candidate_count','p_count_II','p_count_V1','p_axis_paired_beats']
        out.append({'cudnn_TF32':tf32,'observed':{k:float(f[k]) for k in keys},'historical':{k:float(row[k]) for k in keys},'confidence_difference':float(f['p_candidate_confidence']-row.p_candidate_confidence),'within_original_confidence_tolerance':bool(np.isclose(f['p_candidate_confidence'],row.p_candidate_confidence,atol=1e-6,rtol=1e-5,equal_nan=True))})
        assert time.monotonic()-START<=300
    put(O/'precision_comparison.json',{'time':now(),'ecg_id':1,'torch':torch.__version__,'cuda':torch.version.cuda,'device':torch.cuda.get_device_name(1),'P_SHA256':sha(pw),'raw_header_SHA256':au.header_SHA256,'raw_data_SHA256':au.data_SHA256,'results':out,'network_calls':4,'fits':0,'N_remains_stopped':True,'source_of_historical_difference_not_proven_by_one_record':True})
    print(json.dumps(out,ensure_ascii=False))
if __name__=='__main__':main()
