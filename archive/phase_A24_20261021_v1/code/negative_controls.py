"""Preallocated N module independent of the second encoder acquisition gate."""
from a24_shared import *
import numpy as np,pandas as pd,torch,wfdb,subprocess
from extract_controls import definitions,LEADS
OUT=P/'negative';OUT.mkdir(exist_ok=True);START=time.monotonic()
def status(**v):put(OUT/'status.json',{'time':now(),'pid':os.getpid(),'wall_seconds':time.monotonic()-START,**v})
def main():
    assert not (OUT/'started.json').exists(),'No implicit retry'
    prior_integrity();pw=P/'runtime/P_C1_frozen.pt';expected='98557608ee9c581bc0e0ee4f1f0578ad4c14f961952357b5461782f349b341d4';assert sha(pw)==expected
    lock=freeze('negative',[pw,A/'code/a17r_features.py']);put(OUT/'started.json',{'time':now(),'P_SHA256':expected,'E2_not_loaded':True})
    state=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True);line=next(r for r in state.splitlines() if r.strip().startswith('1,'));_,free,util=[int(v.strip()) for v in line.split(',')];assert free>=16384 and util<=10
    torch.set_num_threads(2);torch.cuda.set_device(1);torch.cuda.set_per_process_memory_fraction(.25,1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    s=definitions();model=s['Net']();model.load_state_dict(torch.load(pw,map_location='cpu',weights_only=True),strict=True);model=model.to('cuda:1').eval().requires_grad_(False)
    d=cohort();audit=csv(A/'waveform_audit.csv').set_index('ecg_id');fd=read(B/'factor_definitions.json');M=fd['factors']['M'];sl=read(A/'source_lock.json')['files'];meta=next(k for k in sl if k.endswith('/ptbxl_database.csv'));root=Path(meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')).parent
    morph=s['morphology'];current=[];shift=[];anchor=[];mismatch=[]
    for k,row in enumerate(d.to_dict('records')):
        if k%64==0:
            if time.monotonic()-START>7200:raise TimeoutError('N extraction2-hour cap')
            status(state='running',processed=k,total=len(d),E2_loaded=False)
        file=root/row['filename_hr'];ar=audit.loc[row['ecg_id']];assert sha(file.with_suffix('.hea'))==ar.header_SHA256 and sha(file.with_suffix('.dat'))==ar.data_SHA256
        rec=wfdb.rdrecord(str(file));assert rec.fs==500 and rec.sig_len==5000 and set(rec.units)=={'mV'};names=[n.upper() for n in rec.sig_name];assert set(names)==set(LEADS)
        x=rec.p_signal[:,[names.index(n) for n in LEADS]];assert np.isfinite(x).all();phys,norm,_=s['prepare_raw'](x);rr,_=s['r_features'](phys);xx=np.stack([norm.T,np.gradient(norm.T,axis=-1)*5],axis=1)
        with torch.inference_mode():t=torch.from_numpy(xx).to('cuda:1');prob=((model(t)+model(-t))/2).softmax(1).cpu().numpy()
        s['morphology']=morph;cur=s['p_features'](phys,norm,prob,rr);cur={'ecg_id':row['ecg_id'],**cur};current.append(cur)
        bad=[c for c in cur if c!='ecg_id' and c in row and not np.isclose(cur[c],row[c],atol=1e-6,rtol=1e-5,equal_nan=True)]
        if k<32 and bad:raise RuntimeError(f'First32 candidate replay failed at {row["ecg_id"]}: {bad}')
        if any(not np.isclose(cur[c],row[c],atol=0,rtol=0,equal_nan=True) for c in ['p_candidate_count','p_count_II','p_count_V1','p_axis_paired_beats']):mismatch.append(row['ecg_id'])
        def displaced(z,c):
            cc=dict(c,a=c['a']+200,b=c['b']+200,pk=c['pk']+200);assert 0<=cc['a']<cc['b']<len(z);return morph(z,cc)
        s['morphology']=displaced;sh=s['p_features'](phys,norm,prob,rr);s['morphology']=morph;fake=np.zeros_like(prob);fake[:,0,:]=1
        for r in rr:fake[:,0,r-100:r-39]=0;fake[:,1,r-100:r-39]=1
        an=s['p_features'](phys,norm,fake,rr);shift.append({'ecg_id':row['ecg_id'],**{c:sh[c] for c in M}});anchor.append({'ecg_id':row['ecg_id'],**{c:an[c] for c in M}})
        if (k+1)%256==0:pd.DataFrame(current).to_csv(OUT/'current_candidates.partial.csv',index=False)
    assert len(mismatch)/len(d)<=.005,'Count agreement below99.5%: no relaxation'
    for name,rows in [('current_candidates',current),('displaced_morphology',shift),('anchor_morphology',anchor)]:pd.DataFrame(rows).to_csv(OUT/(name+'.csv'),index=False)
    verify(lock);put(OUT/'completion.json',{'state':'complete','records':len(d),'patients':int(d.patient_id.nunique()),'count_replay_mismatches':mismatch,'count_agreement':1-len(mismatch)/len(d),'P_network_training_epochs':0,'E2_loaded':False,'SPH_predictions':0,'wall_seconds':time.monotonic()-START});status(state='complete',processed=len(d),total=len(d))
if __name__=='__main__':
    try:main()
    except Exception as e:status(state='stopped_no_retry',error=str(e));raise
