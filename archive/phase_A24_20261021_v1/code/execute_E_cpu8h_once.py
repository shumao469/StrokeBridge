"""One authorized CPU8h E continuation. No changes to scientific recipes."""
from a24_shared import *
import importlib.util,subprocess,shutil,traceback,ast
BASE=P/'recovery_E_cpu_v1';REC=P/'recovery_E_cpu8h_v1';START=time.monotonic()
spec=importlib.util.spec_from_file_location('CPU_base',P/'code/execute_E_cpu_once.py');base=importlib.util.module_from_spec(spec);spec.loader.exec_module(base)
base.REC=REC;base.START=START;base.port.REC=REC;base.port.START=START
PORT=base.port

def state(**v):PORT.state(CPU_threads=2,device='cpu',max_extraction_seconds=28800,**v)
def preprocessing8h(defs,hp,fm5):
    # Literal original CPU timing formula. Only authorized wall/admission limits change.
    source=(P/'code/execute_E_cpu_once.py').read_text(encoding='utf-8');tree=ast.parse(source)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preprocessing_cpu');text=ast.get_source_segment(source,node)
    assert text.count('6900')==4 and text.count('7200')==1
    text=text.replace('6900','28500').replace('7200','28800')
    scope={'time':time,'port':PORT,'read':read,'REC':REC,'START':START,'put':put};exec(compile(text,str(P/'code/execute_E_cpu_once.py')+'[authorized8h limits]','exec'),scope)
    scope['preprocessing_cpu'](defs,hp,fm5)

def checkpoint(ids,e1,e2):
    import numpy as np
    n=len(e1)
    if n%1024 and n!=21002:return
    out=REC/'extraction/checkpoints';out.mkdir(exist_ok=True);manifest=out/'manifest.json';rows=read(manifest) if manifest.exists() else [];start=rows[-1]['end_exclusive'] if rows else 0
    assert start<n and n-start<=1024 and len(e2)==n and len(ids)==n
    file=out/f'embeddings_{start:05d}_{n:05d}.npz';assert not file.exists()
    np.savez_compressed(file,ecg_id=np.asarray(ids[start:n]),E1=np.asarray(e1[start:n]),E2=np.asarray(e2[start:n]));rows.append({'start':start,'end_exclusive':n,'records':n-start,'file':str(file),'SHA256':sha(file)});put(manifest,rows)

def make_sources():
    for name in ['extract_controls.py','fit_and_audit.py']:
        old=BASE/(name+'.frozen.txt');original=old.read_text(encoding='utf-8');s=original;changes=[]
        if name=='extract_controls.py':
            assert s.count('if time.monotonic()-START>7200:')==1;s=s.replace('if time.monotonic()-START>7200:','if time.monotonic()-START>28800:');changes.append({'wall_cap_seconds':[7200,28800]})
            assert s.count("'Extraction 2-hour cap'")==1;s=s.replace("'Extraction 2-hour cap'","'Extraction 8-hour cap'")
            line='emb1.extend(e1);emb2.extend(e2);batch.clear()';assert s.count(line)==1;s=s.replace(line,line+';CHECKPOINT_EMBEDDINGS(ids,emb1,emb2)');changes.append({'derived_checkpoint_hook':True,'scientific_values_unchanged':True})
        ast.parse(s);new=REC/(name+'.frozen.txt');assert not new.exists();new.write_text(s,encoding='utf-8')
        meta=read(BASE/(name+'.transformations.json'));meta.update({'prior_CPU_source':str(old),'prior_CPU_source_SHA256':sha(old),'frozen':str(new),'frozen_SHA256':sha(new),'authorized_8h_changes':changes,'protocol8h_SHA256':sha(REC/'protocol.txt')});put(REC/(name+'.transformations.json'),meta)

def freeze8h(module,extra=()):
    return PORT.freeze_E(module,[Path(__file__),P/'code/execute_E_cpu_once.py',BASE/'protocol.txt',BASE/'stop_audit_receipt.json',BASE/'extraction/source_lock.json',BASE/'extraction/encoder_preflight.json',BASE/'extraction/preprocessing_admission.json',*extra])

def worker(name):
    file=REC/(name+'.frozen.txt');scope={'__name__':'a24_CPU8h_E_only','__file__':str(P/'code'/name),'E_ROOT':REC,'N_ROOT':P/'recovery_N_v1','PREPROCESS_ADMISSION':preprocessing8h,'CPU_BENCHMARK':base.cpu_benchmark,'CHECKPOINT_EMBEDDINGS':checkpoint}
    exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),scope);scope['freeze']=freeze8h;scope['main']()

def independent_audit():
    source=P/'code/audit_E_results.py';spec=importlib.util.spec_from_file_location('E_audit8h',source);audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit);audit.REC=REC;audit.OUT=REC/'audit';audit.OUT.mkdir(exist_ok=True)
    put(REC/'audit/audit_launch.json',{'time':now(),'source':str(source),'SHA256':sha(source),'root_substitution_only':True,'fits':0,'inference':0,'network':0})
    audit.main()
    import numpy as np
    z=np.load(REC/'extraction/common5s_embeddings.npz');rows=read(REC/'extraction/checkpoints/manifest.json');offset=0
    for row in rows:
        assert row['start']==offset and sha(row['file'])==row['SHA256'];part=np.load(row['file']);end=row['end_exclusive']
        for key in ['ecg_id','E1','E2']:assert np.array_equal(part[key],z[key][offset:end]),(key,offset,end)
        offset=end
    assert offset==21002;put(REC/'audit/checkpoint_replay.json',{'time':now(),'state':'PASS','chunks':len(rows),'records':offset,'exact_array_replay':True,'fits':0,'inference':0})

def main():
    assert not (REC/'science_launch.json').exists(),'One new authorized8hlaunch only'
    assert read(BASE/'stop_audit_receipt.json')['state']=='asset_CPU_numerical_preprocessing_PASS_time_admission_FAILED_stopped'
    verify(read(BASE/'extraction/source_lock.json')['files']);prior_integrity()
    assert not (BASE/'comparisons/fit_ledger.json').exists() and not (P/'recovery_E_tail_v1/comparisons/fit_ledger.json').exists()
    assert len(read(P/'refit/fit_ledger.json'))==400 and len(read(P/'recovery_N_v1/comparisons/fit_ledger.json'))==10
    weight=read(BASE/'weight_completion.json');assert Path(weight['weight']).stat().st_size==372518672 and sha(weight['weight'])==weight['SHA256']
    assert shutil.disk_usage(REC).free>5*1024**3,'InsufficientDspace:stop'
    for name in ['weight_completion.json','runtime_completion.json','source_mirrors.json']:shutil.copy2(BASE/name,REC/name)
    make_sources();put(REC/'science_launch.json',{'time':now(),'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(REC/'protocol.txt'),'CPU_threads':2,'GPU_allocation':False,'max_extraction_seconds':28800,'max_comparison_seconds':2700,'max_science_seconds':32400,'max_E_fits':12,'prior_fits':410,'global_limit':422,'automatic_postfit_zero_fit_audit':True})
    for name,cap in [('extract_controls.py',28800),('fit_and_audit.py',2700)]:
        state(state='CPU8h_preflight_then_extraction' if name.startswith('extract') else 'CPU_E_only_fitting_statistics',wall_seconds=time.monotonic()-START,background_worker=True)
        with (REC/(name+'.log')).open('w',encoding='utf-8') as log:
            q=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module',name],stdout=log,stderr=subprocess.STDOUT,timeout=min(cap,32400-(time.monotonic()-START)),creationflags=subprocess.CREATE_NO_WINDOW)
        assert q.returncode==0,f'{name} failed,no retry';assert shutil.disk_usage(REC).free>5*1024**3
    ledger=read(REC/'comparisons/fit_ledger.json');assert len(ledger)==12 and all(q['status']=='complete' and q['name'].startswith(('E1','E2')) for q in ledger)
    put(REC/'science_completion.json',{'time':now(),'state':'E_science_complete_audit_pending','device':'cpu','CPU_threads':2,'E_fits':12,'R_fits':400,'N_fits':10,'total_fits':422,'R_N_rerun':False,'wall_seconds':time.monotonic()-START,'full_figure_gate':False})
    state(state='E_science_complete_zero_fit_audit_running',E_fits=12,total_fits=422,background_worker=True)
    independent_audit()
    put(REC/'numerical_completion.json',{'time':now(),'state':'E_audit_PASS_figure_generation_required','fits':422,'E_fits':12,'network_training_epochs':0,'R_N_rerun':False,'eligible_for_result_figures':True,'SPH_predictions':0,'NC_submission_ready':False,'figures_delivered':False,'PPTX_delivered':False})
    state(state='E_audit_PASS_figure_generation_required',E_fits=12,total_fits=422,wall_seconds=time.monotonic()-START,background_worker=False,next_action='Generate/render/inspectNRCfiguresandeditablePPTX;do not mark complete before actual delivery')
if __name__=='__main__':
    try:worker(sys.argv[sys.argv.index('--module')+1]) if '--module' in sys.argv else main()
    except Exception as e:
        if '--module' in sys.argv:put(REC/(sys.argv[-1]+'.stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'traceback':traceback.format_exc()})
        else:state(state='stopped_no_retry',error=str(e),traceback=traceback.format_exc(),background_worker=False)
        raise
