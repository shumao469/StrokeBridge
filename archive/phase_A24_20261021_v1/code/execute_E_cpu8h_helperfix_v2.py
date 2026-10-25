"""Single authorized E-only background continuation with explicit tested callbacks."""
from a24_shared import *
import importlib.util, subprocess, shutil, traceback, sys

REC=P/'recovery_E_cpu8h_helperfix_v2'
OLD=P/'recovery_E_cpu8h_corrected_v1'
PREVIOUS8=P/'recovery_E_cpu8h_v1'
START=time.monotonic()
WALL_START=now()
EXTRACT_LIMIT=28800
POST_LIMIT=2700
SCIENCE_END=START+EXTRACT_LIMIT+POST_LIMIT

spec=importlib.util.spec_from_file_location('E8_helperfix_v2_base',P/'code/execute_E_cpu8h_corrected_once.py')
old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)

def set_anchor(anchor):
    global START,SCIENCE_END
    START=float(anchor);SCIENCE_END=START+EXTRACT_LIMIT+POST_LIMIT
    old.REC=REC;old.START=START
    old.prev.REC=REC;old.prev.START=START
    old.prev.base.REC=REC;old.prev.base.START=START
    old.prev.PORT.REC=REC;old.prev.PORT.START=START

set_anchor(START)

def state(**values):
    record={'time':now(),'pid':os.getpid(),'CPU_threads':2,'device':'cpu','R_fits':400,'N_fits':10,
            'E_max_fits':12,'A24_max_fits':422,'R_N_rerun':False,'SPH_admitted':False,
            'SPH_predictions':0,'SPH94_deleted':False,'NC_submission_ready':False,
            'extraction_deadline_seconds':EXTRACT_LIMIT,'fit_and_audit_deadline_seconds':POST_LIMIT,
            'background_worker':True,**values}
    put(REC/'status.json',record);put(U/'PHASE_A24_STATUS.json',record)

def cpu_benchmark(ef,hf):
    from E_helper_contracts import run_timing_benchmark
    run_timing_benchmark(ef,hf,REC,START,writer=put)

def preprocessing_admission(defs,hp,fm5):
    begin=time.monotonic()
    old.prev.PORT.preprocessing_admission(defs,hp,fm5)
    elapsed=time.monotonic()-START;native=time.monotonic()-begin
    q=read(REC/'extraction/CPU_benchmark.json')
    worst=max(r['seconds_per_ECG'] for r in q['synthetic_timing_batches'])
    projected=21002*(worst+native/32)*1.5+elapsed+180
    q.update({'native32_preprocessing_seconds':native,'preflight_wall_seconds_inclusive_from_parent_start':elapsed,
              'projected_full_extraction_seconds':projected,
              'projection_role':'diagnostic only; no projection admission gate',
              'hard_extraction_deadline_seconds':EXTRACT_LIMIT,'admitted_for_bulk':True,
              'parent_fixed_deadline_enforced':True,'independent_preflight_subcap':False})
    put(REC/'extraction/CPU_benchmark.json',q)
    assert time.monotonic()-START<EXTRACT_LIMIT,'fixed extraction deadline reached during native preprocessing'

def freeze_current(module,extra=()):
    required=[Path(__file__),P/'code/E_helper_contracts.py',P/'code/execute_E_cpu_once.py',
              P/'code/execute_E_cpu8h_corrected_once.py',REC/'protocol.txt',
              P/'software_contracts_E_helper_v1/completion.json',P/'software_contracts_E_helper_v1/source_lock.json',
              P/'software_contracts_E_checkpoint_v1/completion.json',P/'software_contracts_E_checkpoint_v1/source_lock.json',
              OLD/'stop_audit_receipt.json',OLD/'science_launch.json',OLD/'extract_controls.py.stop.json',
              P/'A24_20261023_恢复窗口状态核验.txt',*extra]
    return old.freeze_corrected(module,required)

def worker(module,anchor):
    set_anchor(anchor)
    if module=='independent_audit':
        old.prev.independent_audit()
        return
    allowed={'extract_controls.py','fit_and_audit.py'}
    assert module in allowed
    frozen=REC/(module+'.frozen.txt')
    scope={'__name__':'a24_A24_E_helperfix_v2','__file__':str(P/'code'/module),
           'E_ROOT':REC,'N_ROOT':P/'recovery_N_v1',
           'PREPROCESS_ADMISSION':preprocessing_admission,'CPU_BENCHMARK':cpu_benchmark,
           'CHECKPOINT_EMBEDDINGS':old.prev.checkpoint}
    exec(compile(frozen.read_text(encoding='utf-8'),str(frozen),'exec'),scope)
    scope['freeze']=freeze_current
    scope['main']()

def run_child(module,timeout):
    cmd=[sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module',module,'--start-mono',repr(START)]
    with (REC/(module+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT,timeout=max(.01,timeout),
                              creationflags=subprocess.CREATE_NO_WINDOW)
    assert result.returncode==0,f'{module} failed; no retry'

def main():
    REC.mkdir(parents=True,exist_ok=True)
    assert not (REC/'science_launch.json').exists(),'one launch per authorized run root'
    put(REC/'science_launch.json',{'time':WALL_START,'pid':os.getpid(),'code_SHA256':sha(__file__),
       'protocol_SHA256':sha(REC/'protocol.txt'),'CPU_threads':2,'GPU_allocation':False,
       'fixed_extraction_deadline_seconds':EXTRACT_LIMIT,'fit_and_audit_deadline_seconds':POST_LIMIT,
       'prior_R_N_fits':410,'max_E_fits':12,'A24_global_max_fits':422,'background_launch':True,
       'mock_contract_receipts_verified':False,'automatic_postfit_zero_fit_audit':True})
    state(state='prelaunch_integrity_and_source_verification',wall_seconds=0)
    assert read(OLD/'stop_audit_receipt.json')['state']=='corrective_run_stopped_on_missing_now_helper_binding'
    assert read(P/'software_contracts_E_helper_v1/completion.json')['state']=='mock_only_helper_contracts_PASS'
    assert read(P/'software_contracts_E_checkpoint_v1/completion.json')['state']=='mock_only_checkpoint_contract_PASS'
    verify(read(P/'software_contracts_E_helper_v1/source_lock.json')['files'])
    verify(read(P/'software_contracts_E_checkpoint_v1/source_lock.json')['files'])
    verify(read(PREVIOUS8/'extraction/source_lock.json')['files'])
    prior_integrity()
    r=read(P/'refit/fit_ledger.json');n=read(P/'recovery_N_v1/comparisons/fit_ledger.json')
    assert len(r)==400 and all(x['status']=='complete' for x in r)
    assert len(n)==10 and all(x['status']=='complete' for x in n)
    for root in [P/'recovery_E_cpu_v1',P/'recovery_E_tail_v1',PREVIOUS8,OLD]:
        assert not (root/'comparisons/fit_ledger.json').exists(),str(root)
    weight=read(OLD/'weight_completion.json')
    assert Path(weight['weight']).stat().st_size==372518672
    assert sha(weight['weight'])==weight['SHA256']=='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
    assert shutil.disk_usage(REC).free>5*1024**3
    for name in ['weight_completion.json','runtime_completion.json','source_mirrors.json']:
        shutil.copy2(OLD/name,REC/name)
    for name in ['extract_controls.py','fit_and_audit.py']:
        oldfile=OLD/(name+'.frozen.txt');dest=REC/(name+'.frozen.txt')
        assert not dest.exists();shutil.copy2(oldfile,dest);assert sha(oldfile)==sha(dest)
        meta=read(OLD/(name+'.transformations.json'))
        meta.update({'previous_frozen_file':str(oldfile),'previous_SHA256':sha(oldfile),
                     'new_frozen_file':str(dest),'new_SHA256':sha(dest),
                     'scientific_source_unchanged':True,'authorized_callback':'tested E_helper_contracts.run_timing_benchmark',
                     'new_protocol_SHA256':sha(REC/'protocol.txt')})
        put(REC/(name+'.transformations.json'),meta)
    put(REC/'science_launch.json',{'time':WALL_START,'pid':os.getpid(),'code_SHA256':sha(__file__),
       'protocol_SHA256':sha(REC/'protocol.txt'),'CPU_threads':2,'GPU_allocation':False,
       'fixed_extraction_deadline_seconds':EXTRACT_LIMIT,'fit_and_audit_deadline_seconds':POST_LIMIT,
       'prior_R_N_fits':410,'max_E_fits':12,'A24_global_max_fits':422,'background_launch':True,
       'mock_contract_receipts_verified':True,'automatic_postfit_zero_fit_audit':True})
    remaining=EXTRACT_LIMIT-(time.monotonic()-START)
    assert remaining>0,'integrity verification exhausted the fixed extraction window'
    state(state='E_preflight_then_extraction',elapsed_inclusive_seconds=time.monotonic()-START,
          remaining_extraction_seconds=remaining)
    run_child('extract_controls.py',remaining)
    assert time.monotonic()-START<EXTRACT_LIMIT,'fixed extraction deadline expired'
    assert shutil.disk_usage(REC).free>5*1024**3,'D drive reserve below 5 GiB'
    fit_audit_start=time.monotonic()
    state(state='E_only_fitting_then_independent_audit',extraction_seconds=fit_audit_start-START,
          fit_and_audit_cap_seconds=POST_LIMIT)
    run_child('fit_and_audit.py',POST_LIMIT)
    ledger=read(REC/'comparisons/fit_ledger.json')
    assert len(ledger)==12 and all(x['status']=='complete' and x['name'].startswith(('E1','E2')) for x in ledger)
    run_child('independent_audit',POST_LIMIT-(time.monotonic()-fit_audit_start))
    assert time.monotonic()-fit_audit_start<=POST_LIMIT,'combined fit/audit 45-minute cap expired'
    receipt=read(REC/'audit/completion.json')
    assert receipt['state']=='E_independent_zero_fit_audit_PASS' and receipt['eligible_for_result_figure_generation'] is True
    assert read(REC/'audit/checkpoint_replay.json')['state']=='PASS'
    put(REC/'science_completion.json',{'time':now(),'state':'E_audit_PASS_figure_generation_required',
       'R_fits':400,'N_fits':10,'E_fits':12,'total_fits':422,'max_fits':422,
       'fit_and_audit_seconds':time.monotonic()-fit_audit_start,'R_N_rerun':False,
       'full_result_figures_generated':False,'PPTX_generated':False})
    state(state='E_audit_PASS_figure_generation_required',E_fits=12,total_fits=422,
          fit_and_audit_seconds=time.monotonic()-fit_audit_start,background_worker=False,
          figure_generation='pending; only now is the result gate open',NC_submission_ready=False)

if __name__=='__main__':
    if '--module' in sys.argv:
        module=sys.argv[sys.argv.index('--module')+1]
        anchor=float(sys.argv[sys.argv.index('--start-mono')+1]) if '--start-mono' in sys.argv else START
        try:worker(module,anchor)
        except Exception as exc:
            put(REC/(module+'.stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(exc),
                'traceback':traceback.format_exc(),'fits_before_stop':len(read(REC/'comparisons/fit_ledger.json')) if (REC/'comparisons/fit_ledger.json').exists() else 0})
            raise
    else:
        try:main()
        except Exception as exc:
            state(state='stopped_no_retry',error=str(exc),traceback=traceback.format_exc(),background_worker=False,
                  elapsed_seconds=time.monotonic()-START)
            raise
