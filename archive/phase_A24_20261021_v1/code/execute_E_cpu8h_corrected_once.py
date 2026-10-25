"""One user-approved corrective launch with a single inclusive8h extraction cap."""
from a24_shared import *
import importlib.util,subprocess,shutil,traceback,ast
REC=P/'recovery_E_cpu8h_corrected_v1';OLD=P/'recovery_E_cpu8h_v1';START=time.monotonic();WALL_START=now()
spec=importlib.util.spec_from_file_location('E8_previous',P/'code/execute_E_cpu8h_once.py');prev=importlib.util.module_from_spec(spec);spec.loader.exec_module(prev)
prev.REC=REC;prev.START=START;prev.base.REC=REC;prev.base.START=START;prev.PORT.REC=REC;prev.PORT.START=START

def state(**v):prev.PORT.state(CPU_threads=2,device='cpu',max_extraction_seconds=28800,unified_clock_includes_hashing_initialization_preflight=True,**v)
def timing_benchmark(ef,hf):
    source=(P/'code/execute_E_cpu_once.py').read_text(encoding='utf-8');tree=ast.parse(source);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cpu_benchmark');text=ast.get_source_segment(source,node)
    old="assert time.monotonic()-START<180,'CPUpreflight180second limit'";assert text.count(old)==1;text=text.replace(old,"assert time.monotonic()-START<28800,'Unified8hour extraction limit'")
    scope={'time':time,'START':START,'REC':REC,'put':put,'BENCH_START':None};exec(compile(text,str(P/'code/execute_E_cpu_once.py')+'[authorized unified8h]','exec'),scope);scope['cpu_benchmark'](ef,hf)
def preprocess_unified(defs,hp,fm5):
    begin=time.monotonic();prev.PORT.preprocessing_admission(defs,hp,fm5);elapsed=time.monotonic()-START;preproc=time.monotonic()-begin
    q=read(REC/'extraction/CPU_benchmark.json');worst=max(r['seconds_per_ECG'] for r in q['synthetic_timing_batches']);projected=21002*(worst+preproc/32)*1.5+elapsed+180
    q.update({'native32_preprocessing_seconds':preproc,'preflight_wall_seconds':elapsed,'projected_full_extraction_seconds':projected,'projection_role':'diagnostic only,not an independent resource gate','hard_extraction_cap_seconds':28800,'projection_safety_factor':1.5,'admitted_for_bulk':True,'parent_unified_deadline_enforced':True,'independent180second_preflight_cap':False});put(REC/'extraction/CPU_benchmark.json',q)
    assert elapsed<28800,'Unified8hour extraction limit'
def freeze_corrected(module,extra=()):
    return prev.freeze8h(module,[Path(__file__),OLD/'protocol.txt',OLD/'stop_audit_receipt.json',OLD/'extraction/source_lock.json',*extra])
def worker(name):
    file=REC/(name+'.frozen.txt');scope={'__name__':'a24_CPU8h_corrected_E_only','__file__':str(P/'code'/name),'E_ROOT':REC,'N_ROOT':P/'recovery_N_v1','PREPROCESS_ADMISSION':preprocess_unified,'CPU_BENCHMARK':timing_benchmark,'CHECKPOINT_EMBEDDINGS':prev.checkpoint}
    exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),scope);scope['freeze']=freeze_corrected;scope['main']()
def main():
    assert not (REC/'science_launch.json').exists(),'Exactly one corrective launch'
    put(REC/'science_launch.json',{'time':WALL_START,'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(REC/'protocol.txt'),'CPU_threads':2,'GPU_allocation':False,'max_inclusive_extraction_seconds':28800,'max_comparison_seconds':2700,'max_science_seconds':32400,'independent_preflight_subcap':False,'prior_fits':410,'max_E_fits':12,'A24_global_limit':422,'automatic_zero_fit_postflight_audit':True})
    state(state='corrective_integrity_verification_inside8hour_clock',wall_seconds=time.monotonic()-START,background_worker=True)
    assert read(OLD/'stop_audit_receipt.json')['state']=='CPU8h_interrupted_by_retained180second_preflight_guard'
    verify(read(OLD/'extraction/source_lock.json')['files']);prior_integrity()
    assert len(read(P/'refit/fit_ledger.json'))==400 and len(read(P/'recovery_N_v1/comparisons/fit_ledger.json'))==10
    for root in [OLD,P/'recovery_E_cpu_v1',P/'recovery_E_tail_v1']:assert not (root/'comparisons/fit_ledger.json').exists()
    weight=read(OLD/'weight_completion.json');assert Path(weight['weight']).stat().st_size==372518672 and sha(weight['weight'])==weight['SHA256']
    assert shutil.disk_usage(REC).free>5*1024**3
    for name in ['weight_completion.json','runtime_completion.json','source_mirrors.json']:shutil.copy2(OLD/name,REC/name)
    for name in ['extract_controls.py','fit_and_audit.py']:
        old=OLD/(name+'.frozen.txt');new=REC/(name+'.frozen.txt');assert not new.exists();shutil.copy2(old,new);assert sha(old)==sha(new)
        meta=read(OLD/(name+'.transformations.json'));meta.update({'previous8h_source':str(old),'previous8h_source_SHA256':sha(old),'frozen':str(new),'frozen_SHA256':sha(new),'corrective_change':'scientific source byte-identical; only CPU timing hook/parent unified wall-clock corrected','corrective_protocol_SHA256':sha(REC/'protocol.txt')});put(REC/(name+'.transformations.json'),meta)
    remaining=28800-(time.monotonic()-START);assert remaining>0
    state(state='CPU8h_corrected_preflight_then_extraction',wall_seconds=time.monotonic()-START,remaining_inclusive_extraction_seconds=remaining,background_worker=True)
    with (REC/'extract_controls.py.log').open('w',encoding='utf-8') as log:
        q=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module','extract_controls.py'],stdout=log,stderr=subprocess.STDOUT,timeout=remaining,creationflags=subprocess.CREATE_NO_WINDOW)
    assert q.returncode==0,'Extraction/preflight failed,no retry';assert time.monotonic()-START<28800,'Unified8hour extraction deadline';assert shutil.disk_usage(REC).free>5*1024**3
    state(state='CPU_E_only_fitting_statistics',wall_seconds=time.monotonic()-START,background_worker=True)
    with (REC/'fit_and_audit.py.log').open('w',encoding='utf-8') as log:
        q=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module','fit_and_audit.py'],stdout=log,stderr=subprocess.STDOUT,timeout=min(2700,32400-(time.monotonic()-START)),creationflags=subprocess.CREATE_NO_WINDOW)
    assert q.returncode==0,'Efit/statistics failed,no retry'
    ledger=read(REC/'comparisons/fit_ledger.json');assert len(ledger)==12 and all(q['status']=='complete' and q['name'].startswith(('E1','E2')) for q in ledger)
    put(REC/'science_completion.json',{'time':now(),'state':'E_science_complete_audit_pending','device':'cpu','CPU_threads':2,'E_fits':12,'R_fits':400,'N_fits':10,'total_fits':422,'R_N_rerun':False,'wall_seconds':time.monotonic()-START,'full_figure_gate':False})
    state(state='E_science_complete_zero_fit_audit_running',E_fits=12,total_fits=422,background_worker=True);prev.independent_audit()
    put(REC/'numerical_completion.json',{'time':now(),'state':'E_audit_PASS_figure_generation_required','fits':422,'E_fits':12,'network_training_epochs':0,'R_N_rerun':False,'eligible_for_result_figures':True,'SPH_predictions':0,'NC_submission_ready':False,'figures_delivered':False,'PPTX_delivered':False})
    state(state='E_audit_PASS_figure_generation_required',E_fits=12,total_fits=422,wall_seconds=time.monotonic()-START,background_worker=False,next_action='Generate/render/inspectNRCscientificfiguresandnativeeditablePPTX;notcompletebeforeactualdelivery')
if __name__=='__main__':
    try:worker(sys.argv[sys.argv.index('--module')+1]) if '--module' in sys.argv else main()
    except Exception as e:
        if '--module' in sys.argv:put(REC/(sys.argv[-1]+'.stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'traceback':traceback.format_exc()})
        else:state(state='stopped_no_retry',error=str(e),traceback=traceback.format_exc(),background_worker=False)
        raise
