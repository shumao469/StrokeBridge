"""Common-time frozen encoders and matched physical-window controls. Zero training."""
from a24_shared import *
import ast,types,importlib.util,subprocess
import numpy as np,pandas as pd,torch,wfdb
from torch import nn
from scipy.signal import butter,sosfiltfilt,find_peaks,resample,decimate
from scipy.ndimage import uniform_filter1d
OUT=P/'extraction';OUT.mkdir(exist_ok=True);START=time.monotonic();LEADS=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
def status(**v):put(OUT/'status.json',{'time':now(),'pid':os.getpid(),'wall_seconds':time.monotonic()-START,**v})
def checktime():
    if time.monotonic()-START>7200:raise TimeoutError('Extraction 2-hour cap')
def definitions():
    # Execute only the preserved inference definitions,without platform-bound top-level paths.
    f=A/'code/a17r_features.py';tree=ast.parse(f.read_text(encoding='utf-8'))
    names={'Block','Net','components','med','morphology','r_features','p_features','prepare_raw'}
    nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names]
    scope={'np':np,'pd':pd,'torch':torch,'nn':nn,'butter':butter,'sosfiltfilt':sosfiltfilt,'find_peaks':find_peaks,'uniform_filter1d':uniform_filter1d,'LEADS':LEADS}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(f),'exec'),scope);return scope
def module_file(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def load_models(rt):
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True)
    row=next(x for x in gpu.strip().splitlines() if x.strip().startswith('1,'));_,free,util=[int(v.strip()) for v in row.split(',')]
    assert free>=16384 and util<=10,'GPU1 not idle/admissible; do not interfere'
    torch.set_num_threads(2);torch.cuda.set_device(1);torch.cuda.set_per_process_memory_fraction(.25,1)
    torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
    s=definitions();pm=s['Net']();pm.load_state_dict(torch.load(rt['P_checkpoint'],map_location='cpu',weights_only=True),strict=True);pm=pm.to('cuda:1').eval().requires_grad_(False)
    stub=types.ModuleType('acquire_a3');stub.digest=sha;stub.SHA='4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e';sys.modules['acquire_a3']=stub
    mod=module_file('a24_ecgfm',A/'code/ecgfm_frozen.py');mod.UPSTREAM=Path(rt['E1_upstream'])
    ef=mod.FrozenECGFM().to('cuda:1').eval().requires_grad_(False)
    hm=module_file('a24_hubert',P/'source_audit/hf_hubert_ecg.py');cfg=hm.HuBERTECGConfig.from_json_file(str(P/'source_audit/HuBERT_config.json'));cfg._attn_implementation='eager'
    hf=hm.HuBERTECG(cfg)
    from safetensors.torch import load_file
    hf.load_state_dict(load_file(rt['E2_weight'],device='cpu'),strict=True);hf=hf.to('cuda:1').eval().requires_grad_(False)
    assert not any(p.requires_grad for m in [ef,hf,pm] for p in m.parameters())
    with torch.inference_mode():
        inp=torch.linspace(-1,1,12*2500,device='cuda:1').reshape(1,12,2500);parity=ef.upstream_parity(inp);assert parity<=1e-6,parity
        hb=torch.linspace(-1,1,6000,device='cuda:1').reshape(1,-1);v=hf(hb).last_hidden_state.mean(1)
        assert v.shape==(1,768) and torch.isfinite(v).all();assert torch.equal(v,hf(hb).last_hidden_state.mean(1))
        batch_error=float((v-hf(hb.repeat(2,1)).last_hidden_state.mean(1)[0]).abs().max());assert batch_error<=1e-5,batch_error
    put(OUT/'encoder_preflight.json',{'state':'PASS','device':'cuda:1','ECGFM_upstream_parity':parity,'HuBERT_repeat_exact':True,'HuBERT_batch_error':batch_error,'strict_state_load':True,'all_frozen':True,'E1_load':ef.load_audit,'E2_parameters':sum(p.numel() for p in hf.parameters()),'E2_exposure':'PTB-XL source-family pretraining exposure; not independent unexposed validation'})
    return s,pm,ef,hf
def hubert_preprocessor():
    # Use unchanged BioSPPy tools/utils without importing unrelated optional plotting modules.
    bio=types.ModuleType('biosppy');bio.__path__=[str(VENDOR/'biosppy')];sys.modules['biosppy']=bio
    sig=types.ModuleType('biosppy.signals');sig.__path__=[str(VENDOR/'biosppy/signals')];sys.modules['biosppy.signals']=sig
    bio.utils=module_file('biosppy.utils',VENDOR/'biosppy/utils.py')
    tools=module_file('biosppy.signals.tools',VENDOR/'biosppy/signals/tools.py');filter_signal=tools.filter_signal
    file=P/'source_audit/repo_hubert_ecg__utils.py';tree=ast.parse(file.read_text(encoding='utf-8'))
    nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['apply_filter','scaling','ecg_preprocessing']]
    scope={'np':np,'resample':resample,'filter_signal':filter_signal};exec(compile(ast.Module(body=nodes,type_ignores=[]),str(file),'exec'),scope)
    def run(x):
        a=scope['ecg_preprocessing'](x.T,500);assert a.shape==(12,5000)
        return decimate(a[:,1250:3750].reshape(-1),5).astype('float32')
    return run
def fm5(x):
    a=x[1250:3750].T.copy();a-=a.mean(-1,keepdims=True);a/=np.maximum(a.std(-1,keepdims=True),1e-8)
    a=sosfiltfilt(butter(4,40,btype='lowpass',fs=500,output='sos'),a,axis=-1);a-=a.mean(-1,keepdims=True);a/=np.maximum(a.std(-1,keepdims=True),1e-8)
    return a.astype('float32')
def main():
    assert not (OUT/'started.json').exists(),'Do not implicitly repeat extraction'
    prior_integrity();rt=read(P/'runtime/completion.json');assert sha(rt['E2_weight'])==rt['E2_SHA256'] and sha(rt['P_checkpoint'])==rt['P_SHA256']
    extra=[Path(rt['E2_weight']),Path(rt['P_checkpoint']),P/'source_audit/hf_hubert_ecg.py',P/'source_audit/repo_hubert_ecg__utils.py',P/'source_audit/repo_hubert_ecg__dataset.py',P/'source_audit/HuBERT_config.json',P/'runtime/completion.json']
    extra.extend(p for p in Path(rt['E1_upstream']).rglob('*.py'))
    extra.extend([VENDOR/'biosppy/utils.py',VENDOR/'biosppy/signals/tools.py'])
    lock=freeze('extraction',extra);put(OUT/'started.json',{'time':now(),'protocol_SHA256':sha(P/'protocol.txt')})
    s,pm,ef,hf=load_models(rt);hp=hubert_preprocessor();d=cohort();audit=csv(A/'waveform_audit.csv').set_index('ecg_id');fd=read(B/'factor_definitions.json');M=fd['factors']['M']
    sl=read(A/'source_lock.json')['files'];meta=next(k for k in sl if k.endswith('/ptbxl_database.csv'));root=Path(meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')).parent
    orig_morph=s['morphology'];emb1=[];emb2=[];current=[];shift=[];anchor=[];mismatch=[];N_admitted=True
    reuse=(P/'negative/completion.json').exists()
    if reuse:
        assert read(P/'negative/completion.json')['state']=='complete'
        for n in ['current_candidates.csv','displaced_morphology.csv','anchor_morphology.csv']:assert (P/'negative'/n).exists()
    batch=[];ids=[]
    def encode():
        if not batch:return
        with torch.inference_mode():
            e1=ef(torch.from_numpy(np.stack([q[0] for q in batch])).to('cuda:1')).cpu().numpy()
            e2=hf(torch.from_numpy(np.stack([q[1] for q in batch])).to('cuda:1')).last_hidden_state.mean(1).cpu().numpy()
        assert e1.shape==e2.shape==(len(batch),768) and np.isfinite(e1).all() and np.isfinite(e2).all();emb1.extend(e1);emb2.extend(e2);batch.clear()
    for k,row in enumerate(d.to_dict('records')):
        if k%64==0:checktime();status(state='running',processed=k,total=len(d),control_admission=N_admitted)
        file=root/row['filename_hr'];ar=audit.loc[row['ecg_id']]
        assert sha(file.with_suffix('.hea'))==ar.header_SHA256 and sha(file.with_suffix('.dat'))==ar.data_SHA256,'Frozen raw waveform changed'
        rec=wfdb.rdrecord(str(file));assert rec.fs==500 and rec.sig_len==5000 and set(rec.units)=={'mV'}
        names=[n.upper() for n in rec.sig_name];assert set(names)==set(LEADS);x=rec.p_signal[:,[names.index(n) for n in LEADS]];assert np.isfinite(x).all()
        if reuse:
            batch.append((fm5(x),hp(x)));ids.append(row['ecg_id'])
            if len(batch)==4:encode()
            continue
        phys,norm,_=s['prepare_raw'](x);rr,rf=s['r_features'](phys);xx=np.stack([norm.T,np.gradient(norm.T,axis=-1)*5],axis=1)
        with torch.inference_mode():t=torch.from_numpy(xx).to('cuda:1');prob=((pm(t)+pm(-t))/2).softmax(1).cpu().numpy()
        s['morphology']=orig_morph;cur=s['p_features'](phys,norm,prob,rr);cur={'ecg_id':row['ecg_id'],**cur};current.append(cur)
        bad=[key for key in cur if key!='ecg_id' and key in row and not np.isclose(cur[key],row[key],atol=1e-6,rtol=1e-5,equal_nan=True)]
        if k<32 and bad:raise RuntimeError(f'First32 candidate replay failed at {row["ecg_id"]}: {bad}')
        if any(not np.isclose(cur[c],row[c],equal_nan=True,atol=0,rtol=0) for c in ['p_candidate_count','p_count_II','p_count_V1','p_axis_paired_beats']):mismatch.append(row['ecg_id'])
        def displaced(z,c):
            cc=dict(c,a=c['a']+200,b=c['b']+200,pk=c['pk']+200);assert 0<=cc['a']<cc['b']<len(z);return orig_morph(z,cc)
        s['morphology']=displaced;sh=s['p_features'](phys,norm,prob,rr)
        s['morphology']=orig_morph;fake=np.zeros_like(prob);fake[:,0,:]=1
        for r in rr:fake[:,0,r-100:r-39]=0;fake[:,1,r-100:r-39]=1
        an=s['p_features'](phys,norm,fake,rr)
        shift.append({'ecg_id':row['ecg_id'],**{c:sh[c] for c in M}});anchor.append({'ecg_id':row['ecg_id'],**{c:an[c] for c in M}})
        batch.append((fm5(x),hp(x)));ids.append(row['ecg_id'])
        if len(batch)==4:encode()
        if (k+1)%256==0:
            pd.DataFrame(current).to_csv(OUT/'current_candidates.partial.csv',index=False)
    encode();assert len(ids)==len(d) and len(mismatch)/len(d)<=.005,'Full candidate count replay agreement below99.5%'
    np.savez_compressed(OUT/'common5s_embeddings.npz',ecg_id=ids,E1=np.asarray(emb1),E2=np.asarray(emb2))
    if reuse:
        import shutil
        for n in ['current_candidates.csv','displaced_morphology.csv','anchor_morphology.csv']:shutil.copy2(P/'negative'/n,OUT/n)
    else:
        pd.DataFrame(current).to_csv(OUT/'current_candidates.csv',index=False);pd.DataFrame(shift).to_csv(OUT/'displaced_morphology.csv',index=False);pd.DataFrame(anchor).to_csv(OUT/'anchor_morphology.csv',index=False)
    verify(lock);put(OUT/'completion.json',{'state':'complete','records':len(ids),'patients':int(d.patient_id.nunique()),'encoder_forwards':2*len(ids),'P_network_ECGs':len(ids),'neural_training_epochs':0,'count_replay_mismatches':mismatch,'count_agreement':1-len(mismatch)/len(d),'SPH_predictions':0,'independent_validation':False,'wall_seconds':time.monotonic()-START})
    status(state='complete',processed=len(d),total=len(d))
if __name__=='__main__':
    try:main()
    except Exception as e:status(state='stopped_no_retry',error=str(e));raise
