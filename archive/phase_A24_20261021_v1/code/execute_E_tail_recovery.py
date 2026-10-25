"""Parent-enforced E-only execution. Preserves R/N and original science sources."""
from a24_shared import *
import subprocess,traceback,ast
REC=P/'recovery_E_tail_v1';NR=P/'recovery_N_v1';START=time.monotonic()
def state(**v):
    q={'time':now(),'pid':os.getpid(),'R_fits':400,'N_fits':10,'E_max_fits':12,'A24_max_fits':422,'R_N_rerun':False,'SPH_admitted':False,'SPH_predictions':0,'NC_submission_ready':False,**v};put(REC/'status.json',q);put(U/'PHASE_A24_STATUS.json',q)
def transform(name):
    f=P/'code'/name;s=f.read_text(encoding='utf-8');changes=[]
    def one(a,b):
        nonlocal s
        assert s.count(a)==1,(name,a,s.count(a));s=s.replace(a,b,1);changes.append({'before':a,'after':b})
    if name=='extract_controls.py':
        one("OUT=P/'extraction'","OUT=E_ROOT/'extraction'")
        one("rt=read(P/'runtime/completion.json')","rt=read(E_ROOT/'runtime_completion.json')")
        one("P/'runtime/completion.json'","E_ROOT/'runtime_completion.json'")
        before="s=definitions();pm=s['Net']();pm.load_state_dict(torch.load(rt['P_checkpoint'],map_location='cpu',weights_only=True),strict=True);pm=pm.to('cuda:1').eval().requires_grad_(False)"
        one(before,'s=definitions();pm=None  # Reuse N: no P-network load or forward')
        one('for m in [ef,hf,pm]','for m in [ef,hf]')
        count=s.count("P/'negative");assert count>=4;s=s.replace("P/'negative","N_ROOT/'negative");changes.append({'negative_inputs_only':count})
        one("reuse=(N_ROOT/'negative/completion.json').exists()","reuse=(N_ROOT/'negative/completion.json').exists();assert reuse,'N must be reused; never rerun'")
        one('hp=hubert_preprocessor();d=cohort()','hp=hubert_preprocessor();PREPROCESS_ADMISSION(s,hp,fm5);d=cohort()')
        one("'P_network_ECGs':len(ids)","'P_network_ECGs':0,'N_features_reused':True")
    else:
        one("OUT=P/'comparisons'","OUT=E_ROOT/'comparisons'")
        count=s.count("P/'extraction");assert count>=4;s=s.replace("P/'extraction","E_ROOT/'extraction");changes.append({'E_input_path_substitutions':count})
        count=s.count("P/'negative");assert count>=2;s=s.replace("P/'negative","N_ROOT/'negative");changes.append({'N_input_path_substitutions':count})
        a=s.index('    current=csv(inputs[0]);');b=s.index('    train=d.role.eq(',a)
        removed=s[a:b];assert "designs['N_current']" in removed and "designs['N_permuted']" in removed
        s=s[:a]+"    assert E_ok and set(designs)=={'E1','E1_VN','E1_VNM','E2','E2_VN','E2_VNM'};admission=[]\n"+s[b:];changes.append({'removed_N_design_block':True,'removed_text_SHA256':hashlib.sha256(removed.encode()).hexdigest()})
        one('assert len(rows)<22','assert len(rows)<12');s=s.replace('max_fits=22','max_fits=12').replace("'max_fits':22","'max_fits':12")
    ast.parse(s);dst=REC/(name+'.frozen.txt');assert not dst.exists();dst.write_text(s,encoding='utf-8')
    put(REC/(name+'.transformations.json'),{'original':str(f),'original_SHA256':sha(f),'frozen':str(dst),'frozen_SHA256':sha(dst),'changes':changes,'scientific_recipe_unchanged':True})

def freeze_E(module,extra=()):
    f=REC/module/'source_lock.json';assert not f.exists()
    inputs=[REC/'protocol.txt',P/'recovery_E_v1/protocol.txt',P/'protocol.txt',Path(__file__),P/'code/a24_shared.py',P/'code/extract_controls.py',P/'code/fit_and_audit.py',*REC.glob('*.frozen.txt'),*REC.glob('*.transformations.json'),REC/'weight_completion.json',REC/'runtime_completion.json',A/'analysis_dataset.csv',A/'embeddings.npz',A/'waveform_audit.csv',B/'factor_definitions.json',NR/'negative/completion.json',NR/'audit/completion.json',NR/'negative/current_candidates.csv',NR/'negative/displaced_morphology.csv',NR/'negative/anchor_morphology.csv',*extra]
    lock={str(p):sha(p) for p in inputs};put(f,{'time':now(),'files':lock,'E_only':True,'R_N_reused':True});return lock

def preprocessing_admission(defs,hp,fm5):
    import numpy as np,wfdb
    from scipy.signal import butter,sosfiltfilt,resample,decimate
    from biosppy.signals.tools import filter_signal
    source=(A/'code/a17r_features.py').read_text(encoding='utf-8');tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='prepare_raw')
    # Independent literal reuse of A17R preprocessing,only declared6s->5s crop.
    text=ast.get_source_segment(source,node);assert text.count('x[1000:4000]')==1;text=text.replace('x[1000:4000]','x[1250:3750]')
    ref={'np':np,'butter':butter,'sosfiltfilt':sosfiltfilt};exec(compile(text,str(A/'code/a17r_features.py')+'[declared5s]','exec'),ref)
    d=cohort().iloc[:32];audit=csv(A/'waveform_audit.csv').set_index('ecg_id');meta=next(k for k in read(A/'source_lock.json')['files'] if k.endswith('/ptbxl_database.csv'));root=Path(meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')).parent
    rows=[];leads=['I','II','III','AVR','AVL','AVF','V1','V2','V3','V4','V5','V6']
    for row in d.to_dict('records'):
        file=root/row['filename_hr'];ar=audit.loc[row['ecg_id']];assert sha(file.with_suffix('.hea'))==ar.header_SHA256 and sha(file.with_suffix('.dat'))==ar.data_SHA256
        rec=wfdb.rdrecord(str(file));names=[q.upper() for q in rec.sig_name];assert rec.fs==500 and rec.sig_len==5000 and set(rec.units)=={'mV'} and set(names)==set(leads)
        x=rec.p_signal[:,[names.index(q) for q in leads]];assert np.isfinite(x).all()
        e1=fm5(x);r1=ref['prepare_raw'](x)[2];assert e1.shape==(12,2500) and np.isfinite(e1).all() and np.array_equal(e1,r1)
        filtered=filter_signal(signal=x.T,ftype='FIR',band='bandpass',order=150,frequency=[.05,47],sampling_rate=500)[0]
        rr=resample(filtered,5000,axis=1);scaled=2*(rr-rr.min(axis=1)[:,None])/(rr.max(axis=1)-rr.min(axis=1)+1e-8)[:,None]-1
        r2=decimate(scaled[:,1250:3750].reshape(-1),5).astype('float32');e2=hp(x)
        assert e2.shape==(6000,) and np.isfinite(e2).all() and np.array_equal(e2,r2)
        rows.append({'ecg_id':int(row['ecg_id']),'E1_max_abs_error':float(np.max(abs(e1-r1))),'E2_max_abs_error':float(np.max(abs(e2-r2)))})
    put(REC/'extraction/preprocessing_admission.json',{'time':now(),'state':'PASS','records':32,'first32_sorted_ids':rows,'native_fs':500,'unit':'mV','lead_order':leads,'nominal_raw_crop_seconds':[2.5,7.5],'E1_shape':[12,2500],'E2_shape':[6000],'E2_official_default_crop':'first5s; A24center5s adaptation declared in original protocol','context_difference':'E1filters5s crop;E2filters/scales full10s before crop;pipeline comparison not architecture-only','P_network_forwards':0})

def module_worker(name):
    file=REC/(name+'.frozen.txt');scope={'__name__':'a24_E_only_module','__file__':str(P/'code'/name),'E_ROOT':REC,'N_ROOT':NR,'PREPROCESS_ADMISSION':preprocessing_admission}
    exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),scope);scope['freeze']=freeze_E;scope['main']()

def resource_wait():
    started=time.monotonic();consecutive=0
    while time.monotonic()-started<600:
        rows=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).splitlines();row=next(q for q in rows if q.strip().startswith('1,'));_,free,util=[int(q.strip()) for q in row.split(',')]
        consecutive=consecutive+1 if free>=16384 and util<=10 else 0
        state(state='resource_preflight_wait',GPU1_free_MiB=free,GPU1_utilization=util,consecutive_idle_checks=consecutive,wall_seconds=time.monotonic()-START,background_worker=True)
        if consecutive>=3:return
        time.sleep(5)
    raise TimeoutError('GPU1not stably idle within600s;no interference/no inference retry')

def main():
    assert not (REC/'science_launch.json').exists(),'One E science launch only'
    weight=read(REC/'weight_completion.json');assert weight['state']=='official_asset_size_SHA_PASS' and sha(weight['weight'])==weight['SHA256']
    assert read(NR/'audit/completion.json')['state']=='zero_fit_N_recovery_closeout_PASS'
    assert len(read(P/'refit/fit_ledger.json'))==400 and len(read(NR/'comparisons/fit_ledger.json'))==10
    prior_integrity();verify(read(NR/'negative/source_lock.json')['files']);verify(read(NR/'comparisons/source_lock.json')['files'])
    upstream=P/'runtime/ecgfm_upstream';native=Path(r'\\wsl.localhost\Ubuntu-22.04\home\shu\codex_runs\phase_A17R_20261021_v1\upstream');copies=[]
    for f in upstream.rglob('*.py'):
        src=native/f.relative_to(upstream);assert sha(src)==sha(f),str(f);copies.append({'source':str(src),'copy':str(f),'SHA256':sha(f)})
    put(REC/'source_mirrors.json',copies)
    pw=P/'runtime/P_C1_frozen.pt';assert sha(pw)=='98557608ee9c581bc0e0ee4f1f0578ad4c14f961952357b5461782f349b341d4'
    put(REC/'runtime_completion.json',{'state':'asset_acquired_preflight_pending','E2_weight':weight['weight'],'E2_SHA256':weight['SHA256'],'P_checkpoint':str(pw),'P_SHA256':sha(pw),'E1_upstream':str(upstream),'vendor':str(VENDOR),'global_environment_modified':False,'source_files':len(copies),'P_forward_not_authorized':True})
    transform('extract_controls.py');transform('fit_and_audit.py')
    put(REC/'science_launch.json',{'time':now(),'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(REC/'protocol.txt'),'max_E_fits':12,'A24_already_spent':410,'max_science_seconds':10800,'max_resource_wait_seconds':600})
    resource_wait()
    for name,cap in [('extract_controls.py',7200),('fit_and_audit.py',2700)]:
        assert time.monotonic()-START<10800;state(state='E_extraction' if name.startswith('extract') else 'E_only_fitting_statistics',wall_seconds=time.monotonic()-START,background_worker=True)
        with (REC/(name+'.log')).open('w',encoding='utf-8') as log:
            done=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module',name],stdout=log,stderr=subprocess.STDOUT,timeout=min(cap,10800-(time.monotonic()-START)),creationflags=subprocess.CREATE_NO_WINDOW)
        assert done.returncode==0,f'{name} failed;no retry'
    ledger=read(REC/'comparisons/fit_ledger.json');assert len(ledger)==12 and all(q['status']=='complete' and q['name'].startswith(('E1','E2')) for q in ledger)
    assert read(REC/'extraction/completion.json')['P_network_ECGs']==0
    put(REC/'science_completion.json',{'time':now(),'state':'E_science_complete_audit_pending','E_fits':12,'R_fits':400,'N_fits':10,'total_fits':422,'network_training_epochs':0,'R_N_rerun':False,'wall_seconds':time.monotonic()-START,'full_figure_gate':False})
    state(state='E_science_complete_audit_pending',E_fits=12,total_fits=422,wall_seconds=time.monotonic()-START,background_worker=False)
if __name__=='__main__':
    try:module_worker(sys.argv[sys.argv.index('--module')+1]) if '--module' in sys.argv else main()
    except Exception as e:
        if '--module' in sys.argv:put(REC/(sys.argv[-1]+'.stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'traceback':traceback.format_exc()})
        else:state(state='stopped_no_retry',error=str(e),traceback=traceback.format_exc(),background_worker=False)
        raise
