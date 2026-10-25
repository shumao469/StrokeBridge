"""User-authorized CPU preflight; conditional E-only execution, no GPU/network."""
from a24_shared import *
import importlib.util,subprocess,shutil,traceback
TAIL=P/'recovery_E_tail_v1';REC=P/'recovery_E_cpu_v1';START=time.monotonic()
spec=importlib.util.spec_from_file_location('E_port',P/'code/execute_E_tail_recovery.py');port=importlib.util.module_from_spec(spec);spec.loader.exec_module(port);port.REC=REC;port.START=START
BENCH_START=None
def cpu_benchmark(ef,hf):
    import torch,numpy as np
    global BENCH_START
    BENCH_START=START
    assert torch.get_num_threads()==2 and all(p.device.type=='cpu' and not p.requires_grad for m in [ef,hf] for p in m.parameters())
    x=torch.linspace(-1,1,12*2500).reshape(1,12,2500).repeat(4,1,1);y=torch.linspace(-1,1,6000).reshape(1,-1).repeat(4,1)
    rows=[]
    with torch.inference_mode():
        for k in range(3):
            begin=time.monotonic();a=ef(x);b=hf(y).last_hidden_state.mean(1);elapsed=time.monotonic()-begin
            assert a.shape==b.shape==(4,768) and torch.isfinite(a).all() and torch.isfinite(b).all();rows.append({'batch':k,'ECGs':4,'joint_encoder_seconds':elapsed,'seconds_per_ECG':elapsed/4})
            assert time.monotonic()-START<180,'CPUpreflight180second limit'
    put(REC/'extraction/CPU_benchmark.json',{'time':now(),'threads':2,'device':'cpu','synthetic_timing_batches':rows,'GPU_forward_calls':0,'timing_encoder_batch_calls':6,'timing_encoder_example_forwards':24,'admitted_for_bulk':False})
def preprocessing_cpu(defs,hp,fm5):
    begin=time.monotonic();port.preprocessing_admission(defs,hp,fm5);preprocess_seconds=time.monotonic()-begin
    q=read(REC/'extraction/CPU_benchmark.json');worst=max(r['seconds_per_ECG'] for r in q['synthetic_timing_batches']);elapsed=time.monotonic()-START
    projected=21002*(worst+preprocess_seconds/32)*1.5+elapsed+180
    q.update({'native32_preprocessing_seconds':preprocess_seconds,'preflight_wall_seconds':elapsed,'projected_full_extraction_seconds':projected,'admission_projection_limit_seconds':6900,'hard_extraction_cap_seconds':7200,'projection_safety_factor':1.5,'admitted_for_bulk':bool(projected<=6900 and elapsed<=180)});put(REC/'extraction/CPU_benchmark.json',q)
    assert elapsed<=180,'CPUpreflight180second limit';assert projected<=6900,f'ProjectedCPUextraction{projected:.1f}s exceeds6900s admission;stop,no full run'
def freeze_cpu(module,extra=()):return port.freeze_E(module,[Path(__file__),P/'code/execute_E_tail_recovery.py',TAIL/'protocol.txt',TAIL/'status.json',*extra])
def make_sources():
    port.transform('extract_controls.py');port.transform('fit_and_audit.py')
    file=REC/'extract_controls.py.frozen.txt';s=file.read_text(encoding='utf-8');meta=read(REC/'extract_controls.py.transformations.json');changes=[]
    a=s.index('    gpu=subprocess.check_output(');b=s.index('    torch.backends.cuda.matmul.allow_tf32=False;',a)
    removed=s[a:b];assert 'free>=16384 and util<=10' in removed and 'torch.cuda.set_per_process_memory_fraction' in removed
    s=s[:a]+'    torch.set_num_threads(2)\n'+s[b:];changes.append({'removed_GPU_allocation_gate_in_CPU_only_source_SHA256':hashlib.sha256(removed.encode()).hexdigest()})
    count=s.count("'cuda:1'");assert count>=5;s=s.replace("'cuda:1'","'cpu'");changes.append({'device_substitutions':count,'from':'cuda:1','to':'cpu'})
    before="    put(OUT/'encoder_preflight.json',";assert s.count(before)==1;s=s.replace(before,"    CPU_BENCHMARK(ef,hf)\n"+before,1);changes.append({'CPU_synthetic_timing_hook':True})
    file.write_text(s,encoding='utf-8');meta.update({'frozen_SHA256':sha(file),'additional_CPU_changes':changes,'CPU_protocol_SHA256':sha(REC/'protocol.txt')});put(REC/'extract_controls.py.transformations.json',meta)
def worker(name):
    file=REC/(name+'.frozen.txt');scope={'__name__':'a24_CPU_E_only','__file__':str(P/'code'/name),'E_ROOT':REC,'N_ROOT':P/'recovery_N_v1','PREPROCESS_ADMISSION':preprocessing_cpu,'CPU_BENCHMARK':cpu_benchmark}
    exec(compile(file.read_text(encoding='utf-8'),str(file),'exec'),scope);scope['freeze']=freeze_cpu;scope['main']()
def main():
    assert not (REC/'science_launch.json').exists(),'One CPUlaunch only'
    assert read(TAIL/'status.json')['state']=='stopped_no_retry' and not (TAIL/'extraction/started.json').exists() and not (TAIL/'comparisons/fit_ledger.json').exists()
    weight=read(TAIL/'weight_completion.json');assert sha(weight['weight'])==weight['SHA256'] and Path(weight['weight']).stat().st_size==372518672
    assert len(read(P/'refit/fit_ledger.json'))==400 and len(read(P/'recovery_N_v1/comparisons/fit_ledger.json'))==10;prior_integrity()
    for name in ['weight_completion.json','runtime_completion.json','source_mirrors.json']:shutil.copy2(TAIL/name,REC/name)
    make_sources();put(REC/'science_launch.json',{'time':now(),'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(REC/'protocol.txt'),'CPU_threads':2,'GPU_allocation':False,'max_extraction_seconds':7200,'max_comparison_seconds':2700,'max_E_fits':12,'prior_fits':410,'global_limit':422})
    for name,cap in [('extract_controls.py',7200),('fit_and_audit.py',2700)]:
        port.state(state='CPU_preflight_then_conditional_extraction' if name.startswith('extract') else 'CPU_E_only_fitting_statistics',background_worker=True)
        with (REC/(name+'.log')).open('w',encoding='utf-8') as log:
            q=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--module',name],stdout=log,stderr=subprocess.STDOUT,timeout=min(cap,10800-(time.monotonic()-START)),creationflags=subprocess.CREATE_NO_WINDOW)
        assert q.returncode==0,f'{name} failed,no retry'
    ledger=read(REC/'comparisons/fit_ledger.json');assert len(ledger)==12 and all(q['status']=='complete' and q['name'].startswith(('E1','E2')) for q in ledger)
    put(REC/'science_completion.json',{'time':now(),'state':'E_science_complete_audit_pending','device':'cpu','CPU_threads':2,'E_fits':12,'R_fits':400,'N_fits':10,'total_fits':422,'R_N_rerun':False,'wall_seconds':time.monotonic()-START,'full_figure_gate':False})
    port.state(state='E_science_complete_audit_pending',E_fits=12,total_fits=422,background_worker=False)
if __name__=='__main__':
    try:worker(sys.argv[sys.argv.index('--module')+1]) if '--module' in sys.argv else main()
    except Exception as e:
        if '--module' in sys.argv:put(REC/(sys.argv[-1]+'.stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'traceback':traceback.format_exc()})
        else:port.state(state='stopped_no_retry',error=str(e),traceback=traceback.format_exc(),background_worker=False)
        raise
