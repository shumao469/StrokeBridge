"""Prepared correction for the stopped WSL launch; never starts science without --run."""
from pathlib import Path
import ast, builtins, dis, importlib.util, json, subprocess, sys, time, traceback

ENTRY=Path(__file__).resolve()
spec=importlib.util.spec_from_file_location('a24_wsl_v1_preserved',ENTRY.with_name('execute_E_wsl_20261023.py'))
e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
e.REC=e.P/'recovery_E_wsl_20261023_v2'
REC=e.REC
original_function=e.function_from
original_freeze=e.source_freeze

def checked_function(path,name,scope,replacements=()):
    scope.update(ast=ast)
    callback=original_function(path,name,scope,replacements)
    unresolved=sorted({i.argval for i in dis.get_instructions(callback)
        if i.opname=='LOAD_GLOBAL' and i.argval not in callback.__globals__ and not hasattr(builtins,i.argval)})
    assert not unresolved,(name,'Unbound callback globals',unresolved)
    return callback
e.function_from=checked_function
e.source_freeze=lambda module,extra=():original_freeze(module,[ENTRY,*extra])

def child(module,anchor,timeout):
    assert timeout>0,'Fixed deadline reached'
    with (REC/(module+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u','-B',str(ENTRY),'--module',module,
            '--anchor',repr(anchor)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    assert result.returncode==0,module+' failed; stop, no retry'
e.child=child

def prepare():
    assert not (REC/'science_launch.json').exists()
    e.environment_preflight()
    e.seal_preparation()
    # Run the actual preserved native32 admission with no encoder/model initialization.
    mock_root=REC/'native32_precheck';(mock_root/'extraction').mkdir(parents=True)
    ns=dict(vars(e.shared));ns.update(REC=mock_root,START=time.monotonic())
    admit=checked_function(e.CODE/'execute_E_tail_recovery.py','preprocessing_admission',ns,
        [("meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')",'meta')])
    scope={'__name__':'preprocessing_only','__file__':str(e.CODE/'extract_controls.py'),
           'E_ROOT':REC/'environment_mock','N_ROOT':e.P/'recovery_N_v1','linux_path':e.linux_path}
    exec(compile((REC/'extract_controls.py.frozen.txt').read_text(encoding='utf-8'),'preprocessing_only','exec'),scope)
    hp=scope['hubert_preprocessor']();admit({},hp,scope['fm5'])
    result=e.read(mock_root/'extraction/preprocessing_admission.json')
    assert result['state']=='PASS' and result['records']==32
    assert all(x['E1_max_abs_error']==x['E2_max_abs_error']==0 for x in result['first32_sorted_ids'])
    # Model admission already passed in the preserved stopped run; retain, do not relabel it.
    prior=e.read(e.P/'recovery_E_wsl_20261023_v1/extraction/encoder_preflight.json')
    assert prior['state']=='PASS' and prior['strict_state_load'] and prior['all_frozen']
    e.put(REC/'callback_contract.json',{'time':e.now(),'state':'PASS',
        'dependency_check':'all LOAD_GLOBAL bindings checked before callback invocation',
        'native32_preprocessing':'both encoders exact array equality','ECG_reads':32,
        'encoder_forwards':0,'model_loads':0,'fits':0,
        'prior_encoder_preflight':'reference from stopped v1 only; repeat strict gates inside new authorized window',
        'entry_script':str(ENTRY),'child_entry_script':str(ENTRY),'fixed_window_launched':False})
    lock=e.read(REC/'preparation_lock.json')
    for path in (ENTRY,REC/'callback_contract.json',mock_root/'extraction/preprocessing_admission.json'):
        lock['files'][str(path)]=e.sha(path)
    e.put(REC/'preparation_lock.json',lock)
    print('CORRECTED_WSL_FULL_PRECHECK_PASS_NO_SCIENTIFIC_LAUNCH',flush=True)

if __name__=='__main__':
    if '--prepare' in sys.argv:
        prepare()
    elif '--module' in sys.argv:
        module=sys.argv[sys.argv.index('--module')+1]
        try:e.worker(module,float(sys.argv[sys.argv.index('--anchor')+1]))
        except Exception as exc:
            e.put(REC/(module+'.stop.json'),{'time':e.now(),'error':str(exc),
                'traceback':traceback.format_exc(),'state':'stopped_no_retry'})
            raise
    elif '--run' in sys.argv:
        try:
            assert e.read(REC/'callback_contract.json')['state']=='PASS'
            if '--retain-original-deadline' in sys.argv:
                original=e.read(e.P/'recovery_E_wsl_20261023_v1/science_launch.json')
                e.EXTRACT_LIMIT=original['extraction_deadline_epoch']-time.time()
                assert 0<e.EXTRACT_LIMIT<=28800,'Original extraction window has expired'
            e.main()
        except Exception as exc:
            e.state(state='stopped_no_retry',error=str(exc),traceback=traceback.format_exc(),background_worker=False)
            raise
    else:
        raise SystemExit('Use --prepare for repair validation; --run requires a new scientific-window authorization.')
