"""A24 platform repair and one bounded WSL E execution; scientific recipes retained."""
from pathlib import Path
import os, sys, re, json, time, ast, hashlib, shutil, subprocess, traceback, importlib.util

CODE = Path(__file__).resolve().parent
P = CODE.parent
REC = P / 'recovery_E_wsl_20261023_v1'
OLD = P / 'recovery_E_cpu8h_helperfix_v2'
NATIVE = Path('/home/shu/codex_runs/a24_E_wsl_20261023_v1')
SAFE = NATIVE / 'vendor'
EXTRACT_LIMIT, POST_LIMIT = 28800, 2700
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ[key] = '2'
os.environ.update(CUDA_VISIBLE_DEVICES='', HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                  TOKENIZERS_PARALLELISM='false', PYTHONDONTWRITEBYTECODE='1')

def linux_path(value):
    s = str(value)
    if s.startswith('\\\\wsl.localhost\\Ubuntu-22.04\\'):
        return Path('/' + s.split('Ubuntu-22.04\\',1)[1].replace('\\','/'))
    match = re.match(r'^([A-Za-z]):[\\/](.*)$', s)
    return Path('/mnt')/match[1].lower()/match[2].replace('\\','/') if match else Path(s)

import a24_shared as shared
while str(shared.VENDOR) in sys.path:
    sys.path.remove(str(shared.VENDOR))
sys.path.insert(0, str(SAFE))
_sha = shared.sha
shared.sha = lambda p: _sha(linux_path(p))
shared.VENDOR = SAFE
from a24_shared import read, put, now, sha, verify, prior_integrity

def function_from(path, name, scope, replacements=()):
    source = Path(path).read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    body = ast.get_source_segment(source,node)
    for before, after in replacements:
        assert body.count(before)==1, (name,before)
        body = body.replace(before,after)
    exec(compile(body,str(path)+'[platform callback]', 'exec'),scope)
    return scope[name]

def make_sources():
    changes = [('mod.FrozenECGFM()', 'mod.FrozenECGFM(checkpoint=linux_path(mod.WEIGHTS))'),
               ("Path(meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')).parent", 'linux_path(meta).parent')]
    for name in ('extract_controls.py','fit_and_audit.py'):
        src = OLD/(name+'.frozen.txt'); text = src.read_text(encoding='utf-8')
        applied = []
        for before,after in changes if name.startswith('extract') else []:
            assert text.count(before)==1, before
            text=text.replace(before,after); applied.append({'before':before,'after':after,'role':'path only'})
        ast.parse(text)
        dest=REC/(name+'.frozen.txt')
        assert not dest.exists(), 'No replacement of prepared sources'
        dest.write_text(text,encoding='utf-8')
        meta=read(OLD/(name+'.transformations.json'))
        meta.update(frozen=str(dest),frozen_SHA256=sha(dest),previous_frozen_file=str(src),
                    previous_SHA256=sha(src),wsl_platform_changes=applied,
                    scientific_recipe_unchanged=True)
        put(REC/(name+'.transformations.json'),meta)

def environment_preflight():
    """Imports and source contracts only; no model load, ECG read, inference or fit."""
    import importlib.metadata as md
    import numpy, pandas, torch, wfdb, scipy, sklearn, joblib, tokenizers, safetensors
    from omegaconf import OmegaConf
    import transformers, huggingface_hub
    from transformers import HubertConfig, HubertModel
    from safetensors.torch import load_file
    from threadpoolctl import threadpool_info
    assert transformers.__version__=='4.57.1'
    assert huggingface_hub.__version__=='0.36.0'
    assert str(SAFE) in transformers.__file__ and str(SAFE) in huggingface_hub.__file__
    assert 'DRIVE_ROOT/' not in tokenizers.__file__ and 'DRIVE_ROOT/' not in safetensors.__file__
    torch.set_num_threads(2); assert torch.get_num_threads()==2
    assert linux_path('DRIVE_ROOT/example')==Path('DRIVE_ROOT/example')
    assert linux_path(r'DRIVE_ROOT/example')==Path('DRIVE_ROOT/example')
    assert linux_path(r'\\wsl.localhost\Ubuntu-22.04\home\shu\x')==Path('/home/shu/x')
    for name in ('extract_controls.py','fit_and_audit.py'):
        ast.parse((OLD/(name+'.frozen.txt')).read_text(encoding='utf-8'))
    # Execute the tested storage callback with synthetic arrays, isolated from scientific outputs.
    scope=dict(vars(shared));scope['REC']=REC/'environment_mock'
    (scope['REC']/'extraction').mkdir(parents=True,exist_ok=True)
    callback=function_from(CODE/'execute_E_cpu8h_once.py','checkpoint',scope)
    ids=numpy.arange(1024)+99000000; a=numpy.zeros((1024,768),dtype='float32')
    callback(ids,a,a)
    rows=read(scope['REC']/'extraction/checkpoints/manifest.json')
    part=numpy.load(rows[0]['file'])
    assert numpy.array_equal(part['ecg_id'],ids) and numpy.array_equal(part['E1'],a)
    # Exercise OS timeout and exit handling without models/data.
    child=subprocess.run([sys.executable,'-c','raise SystemExit(0)'],timeout=10)
    assert child.returncode==0
    try:
        subprocess.run([sys.executable,'-c','import time; time.sleep(2)'],timeout=.05,
                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        pass
    else:
        raise AssertionError('Subprocess timeout guard did not fire')
    result={'time':now(),'state':'WSL_environment_mock_preflight_PASS','python':sys.executable,
            'packages':{p:md.version(p) for p in ('torch','numpy','pandas','wfdb','scipy','scikit-learn',
                        'joblib','transformers','huggingface-hub','tokenizers','safetensors')},
            'CPU_threads':2,'torch_threads':torch.get_num_threads(),'native_extension_paths':
            {'tokenizers':tokenizers.__file__,'safetensors':safetensors.__file__},
            'threadpools':threadpool_info(),'ECG_reads':0,'model_loads':0,'encoder_forwards':0,'fits':0,
            'checkpoint_mock_PASS':True,'child_exit_and_timeout_PASS':True,
            'scientific_preflight':'strict load/parity/repeat/batch/native32 retained inside new fixed window'}
    put(REC/'environment_preflight.json',result)
    make_sources()
    rt=read(OLD/'runtime_completion.json')
    for key in ('E2_weight','P_checkpoint','E1_upstream','vendor'):
        rt[key]=str(linux_path(rt[key]))
    rt['vendor']=str(SAFE)
    put(REC/'runtime_completion.json',rt)
    weight=read(OLD/'weight_completion.json');weight['weight']=str(linux_path(weight['weight']))
    put(REC/'weight_completion.json',weight)
    put(REC/'preparation_lock.json',{'time':now(),'files':{str(f):sha(f) for f in
        [Path(__file__),REC/'protocol.txt',REC/'environment_preflight.json',REC/'runtime_completion.json',
         REC/'weight_completion.json',*REC.glob('*.frozen.txt'),*REC.glob('*.transformations.json')]}})
    print(json.dumps(result,ensure_ascii=False),flush=True)

def state(**values):
    value={'time':now(),'pid':os.getpid(),'CPU_threads':2,'device':'cpu','R_fits':400,'N_fits':10,
           'E_max_fits':12,'A24_max_fits':422,'R_N_rerun':False,'SPH_admitted':False,
           'SPH_predictions':0,'SPH94_deleted':False,'NC_submission_ready':False,
           'background_worker':True,'current_run':str(REC.relative_to(shared.U)),**values}
    put(REC/'status.json',value);put(shared.U/'PHASE_A24_STATUS.json',value)

def seal_preparation():
    import tarfile
    members={}
    with tarfile.open(REC/'vendor_pure.tar.gz','r:gz') as archive:
        for member in archive:
            if not member.isfile():continue
            path=SAFE/member.name
            assert path.is_relative_to(SAFE)
            expected=hashlib.sha256(archive.extractfile(member).read()).hexdigest()
            assert sha(path)==expected,str(path)
            members[str(path)]=expected
    put(REC/'environment_source_lock.json',{'time':now(),'files':members,
        'archive_SHA256':sha(REC/'vendor_pure.tar.gz'),'all_extracted_members_match':True})
    # Compile both workers and bind every extraction callback before real execution.
    ns=dict(vars(shared));ns.update(REC=REC,START=time.monotonic())
    function_from(CODE/'execute_E_tail_recovery.py','preprocessing_admission',dict(ns),
        [("meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')",'meta')])
    for name in ('extract_controls.py','fit_and_audit.py'):
        scope={'__name__':'a24_mock_import_only','__file__':str(CODE/name),'E_ROOT':REC/'environment_mock',
               'N_ROOT':P/'recovery_N_v1','linux_path':linux_path}
        exec(compile((REC/(name+'.frozen.txt')).read_text(encoding='utf-8'),name,'exec'),scope)
        assert callable(scope['main'])
        if name.startswith('extract'):
            hp=scope['hubert_preprocessor']()
            import numpy as np
            sample=np.tile(np.sin(np.arange(5000)[:,None]/30),(1,12))
            assert hp(sample).shape==(6000,) and scope['fm5'](sample).shape==(12,2500)
    put(REC/'preparation_sealed.json',{'time':now(),'state':'PASS','vendor_files':len(members),
        'worker_imports_PASS':True,'synthetic_preprocessors_PASS':True,'ECG_reads':0,
        'model_loads':0,'fits':0})
    files=[Path(__file__),REC/'protocol.txt',REC/'environment_preflight.json',REC/'runtime_completion.json',
           REC/'weight_completion.json',REC/'environment_source_lock.json',REC/'preparation_sealed.json',
           *REC.glob('*.frozen.txt'),*REC.glob('*.transformations.json')]
    put(REC/'preparation_lock.json',{'time':now(),'files':{str(f):sha(f) for f in files}})
    print('PREPARATION_SEALED_PASS',len(members),flush=True)

def source_freeze(module,extra=()):
    dest=REC/module/'source_lock.json';assert not dest.exists()
    files=[P/'protocol.txt',REC/'protocol.txt',Path(__file__),CODE/'a24_shared.py',
           CODE/'E_helper_contracts.py',CODE/'execute_E_cpu8h_once.py',CODE/'execute_E_tail_recovery.py',
           REC/'environment_preflight.json',REC/'preparation_lock.json',REC/'runtime_completion.json',
           REC/'weight_completion.json',*REC.glob('*.frozen.txt'),*REC.glob('*.transformations.json'),
           shared.A/'analysis_dataset.csv',shared.A/'embeddings.npz',shared.A/'waveform_audit.csv',
           shared.B/'factor_definitions.json',P/'refit/fit_ledger.json',
           P/'recovery_N_v1/comparisons/fit_ledger.json',P/'recovery_N_v1/negative/completion.json',
           P/'recovery_N_v1/audit/completion.json',*extra]
    lock={str(p):sha(p) for p in files}
    vendor_lock=read(REC/'environment_source_lock.json')['files'];verify(vendor_lock);lock.update(vendor_lock)
    put(dest,{'time':now(),'files':lock,'platform':'WSL','R_N_reused':True});return lock

def worker(module,anchor):
    if module=='independent_audit':
        independent_audit();return
    assert module in ('extract_controls.py','fit_and_audit.py')
    scope={'__name__':'a24_wsl_E_only','__file__':str(CODE/module),'E_ROOT':REC,
           'N_ROOT':P/'recovery_N_v1','linux_path':linux_path}
    exec(compile((REC/(module+'.frozen.txt')).read_text(encoding='utf-8'),str(CODE/module),'exec'),scope)
    scope['freeze']=source_freeze
    if module.startswith('extract'):
        from E_helper_contracts import run_timing_benchmark
        scope['CPU_BENCHMARK']=lambda ef,hf:run_timing_benchmark(ef,hf,REC,anchor,writer=put)
        ns=dict(vars(shared));ns.update(REC=REC,START=anchor)
        scope['CHECKPOINT_EMBEDDINGS']=function_from(CODE/'execute_E_cpu8h_once.py','checkpoint',dict(ns))
        admit=function_from(CODE/'execute_E_tail_recovery.py','preprocessing_admission',dict(ns),
              [("meta.replace('DRIVE_ROOT/','DRIVE_ROOT/')",'meta')])
        def preflight(defs,hp,fm5):
            begin=time.monotonic();admit(defs,hp,fm5)
            native=time.monotonic()-begin;elapsed=time.monotonic()-anchor
            q=read(REC/'extraction/CPU_benchmark.json')
            worst=max(r['seconds_per_ECG'] for r in q['synthetic_timing_batches'])
            q.update(native32_preprocessing_seconds=native,preflight_wall_seconds=elapsed,
                     projected_full_extraction_seconds=21002*(worst+native/32)*1.5+elapsed+180,
                     hard_extraction_deadline_seconds=EXTRACT_LIMIT,admitted_for_bulk=True,
                     projection_role='diagnostic only; parent fixed deadline is enforced')
            put(REC/'extraction/CPU_benchmark.json',q)
            assert elapsed<EXTRACT_LIMIT
        scope['PREPROCESS_ADMISSION']=preflight
    scope['main']()

def independent_audit():
    spec=importlib.util.spec_from_file_location('a24_wsl_independent_audit',CODE/'audit_E_results.py')
    audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)
    audit.REC=REC;audit.OUT=REC/'audit';audit.OUT.mkdir(exist_ok=True)
    audit.main()
    import numpy as np
    z=np.load(REC/'extraction/common5s_embeddings.npz');offset=0
    rows=read(REC/'extraction/checkpoints/manifest.json')
    for row in rows:
        assert row['start']==offset and sha(row['file'])==row['SHA256']
        part=np.load(row['file']);end=row['end_exclusive']
        for key in ('ecg_id','E1','E2'):
            assert np.array_equal(part[key],z[key][offset:end])
        offset=end
    assert offset==21002
    put(REC/'audit/checkpoint_replay.json',{'time':now(),'state':'PASS','chunks':len(rows),
        'records':offset,'exact_array_replay':True,'fits':0,'encoder_forwards':0})

def child(module,anchor,timeout):
    assert timeout>0,'Fixed deadline reached'
    with (REC/(module+'.log')).open('w',encoding='utf-8') as log:
        result=subprocess.run([sys.executable,'-u','-B',str(Path(__file__)),
            '--module',module,'--anchor',repr(anchor)],stdout=log,stderr=subprocess.STDOUT,timeout=timeout)
    assert result.returncode==0,module+' failed; stop, no retry'

def main():
    started=time.monotonic()
    assert not (REC/'science_launch.json').exists(),'One authorized launch only'
    assert read(REC/'environment_preflight.json')['state']=='WSL_environment_mock_preflight_PASS'
    assert read(REC/'preparation_sealed.json')['state']=='PASS'
    verify(read(REC/'preparation_lock.json')['files'])
    put(REC/'science_launch.json',{'time':now(),'pid':os.getpid(),'CPU_threads':2,
        'max_inclusive_extraction_seconds':EXTRACT_LIMIT,'max_fit_and_audit_seconds':POST_LIMIT,
        'R_prior_fits':400,'N_prior_fits':10,'E_max_fits':12,'global_fit_limit':422,
        'R_N_rerun':False,'one_launch_no_retry':True,'background':True,
        'extraction_deadline_epoch':time.time()+EXTRACT_LIMIT,
        'authorization':'2026-10-23 user: repair WSL/preflight then one new fixed background window'})
    state(state='WSL_source_and_asset_verification',run_started=now())
    prior_integrity()
    for location in ('refit','recovery_N_v1/comparisons'):
        ledger=read(P/location/'fit_ledger.json')
        assert len(ledger)==(400 if location=='refit' else 10)
        assert all(x['status']=='complete' for x in ledger)
    for f in P.glob('recovery_E*/comparisons/fit_ledger.json'):
        assert len(read(f))==0, 'Earlier E attempt already consumed budget: '+str(f)
    weight=read(REC/'weight_completion.json')
    assert Path(weight['weight']).stat().st_size==372518672
    assert sha(weight['weight'])==weight['SHA256']=='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
    assert shutil.disk_usage(REC).free>5*1024**3
    state(state='WSL_E_strict_preflight_then_extraction',elapsed_seconds=time.monotonic()-started)
    child('extract_controls.py',started,EXTRACT_LIMIT-(time.monotonic()-started))
    assert time.monotonic()-started<EXTRACT_LIMIT
    post=time.monotonic()
    state(state='WSL_E_only_fitting_then_audit',extraction_seconds=post-started)
    child('fit_and_audit.py',started,POST_LIMIT)
    e=read(REC/'comparisons/fit_ledger.json')
    assert len(e)==12 and all(x['status']=='complete' for x in e)
    # The existing independent audit explicitly requires this state before execution.
    put(REC/'science_completion.json',{'time':now(),'state':'E_science_complete_audit_pending',
        'E_fits':12,'R_fits':400,'N_fits':10,'total_fits':422})
    child('independent_audit',started,POST_LIMIT-(time.monotonic()-post))
    assert time.monotonic()-post<=POST_LIMIT
    audit=read(REC/'audit/completion.json')
    assert audit['state']=='E_independent_zero_fit_audit_PASS' and audit['eligible_for_result_figure_generation']
    assert read(REC/'audit/checkpoint_replay.json')['state']=='PASS'
    put(REC/'numerical_completion.json',{'time':now(),'state':'E_audit_PASS_figure_generation_required',
        'E_fits':12,'total_fits':422,'extraction_seconds':post-started,'fit_and_audit_seconds':time.monotonic()-post,
        'eligible_for_result_figures':True,'NC_submission_ready':False})
    state(state='E_audit_PASS_figure_generation_required',E_fits=12,total_fits=422,background_worker=False)

if __name__=='__main__':
    if '--prepare' in sys.argv:
        environment_preflight()
    elif '--seal' in sys.argv:
        seal_preparation()
    elif '--module' in sys.argv:
        try:worker(sys.argv[sys.argv.index('--module')+1],float(sys.argv[sys.argv.index('--anchor')+1]))
        except Exception as exc:
            put(REC/(sys.argv[sys.argv.index('--module')+1]+'.stop.json'),
                {'time':now(),'error':str(exc),'traceback':traceback.format_exc(),'state':'stopped_no_retry'})
            raise
    else:
        try:main()
        except Exception as exc:
            state(state='stopped_no_retry',error=str(exc),traceback=traceback.format_exc(),background_worker=False)
            raise
