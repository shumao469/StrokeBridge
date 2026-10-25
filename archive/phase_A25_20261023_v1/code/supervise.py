"""One background window; independent M/D tasks; never restarts WSL or a failed job."""
from common import *
import subprocess,sys,signal,traceback
START=time.monotonic();CAP=10800;children={};logs=[]
def update(state,**kw):
    v=dict(time=now(),state=state,pid=os.getpid(),window_seconds=CAP,wall_seconds=time.monotonic()-START,children={k:dict(pid=p.pid,returncode=p.poll()) for k,p in children.items()},max_new_A25_fits=88,prior_A24_fits=422,R_N_E_rerun=False,RDB_admitted=False,SPH_admitted=False,hospital_branch='waiting',expert_reference='waiting',new_figures=0,new_PPTX=0,**kw)
    put(RUN/'background_status.json',v);put(BASE/'PHASE_A25_STATUS.json',v)
def stop_child(p):
    if p.poll() is None:
        os.killpg(p.pid,signal.SIGTERM)
        try:p.wait(timeout=10)
        except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait(timeout=10)
def report():
    m=read(RUN/'measurement/completion.json') if (RUN/'measurement/completion.json').exists() else read(RUN/'measurement/status.json')
    d=read(RUN/'missingness/completion.json') if (RUN/'missingness/completion.json').exists() else read(RUN/'missingness/status.json')
    ledger=read(RUN/'missingness/fit_ledger.json') if (RUN/'missingness/fit_ledger.json').exists() else []
    result=dict(time=now(),state='complete' if m.get('state')=='complete' and d.get('state')=='complete' else 'bounded_finished_with_stopped_modules',M=m,D=d,A25_fit_attempts=len(ledger),A25_fit_cap=88,A24_fits_preserved=422,RDB_admitted=False,SPH_admitted=False,NC_submission_ready=False,clinical_future_risk_evidence=False,new_figures=0,new_PPTX=0,source_gate='See scientific_admission.json',hospital_branch='waiting',expert_reference='waiting')
    put(RUN/'completion.json',result)
    lines=['A25 bounded execution summary',now(),f'M state: {m.get("state")}; engineering gate: {m.get("engineering_gate_pass")}',f'D state: {d.get("state")}; fits attempted: {len(ledger)}/88','RDB scientific gate remains closed; SPH remains excluded. A24 R/N/E not rerun.','Current-AF recognition / development measurement only. No new longitudinal stroke or brain-heart evidence. No figures.']
    (RUN/'执行结果摘要.txt').write_text('\n'.join(lines),encoding='utf-8')
    return result
def main():
    assert not (RUN/'background_started.json').exists(),'No automatic restart'
    assert read(RUN/'preflight.json')['state']=='PASS';verify()
    put(RUN/'background_started.json',dict(time=now(),pid=os.getpid(),window_seconds=CAP,deadline_unix=time.time()+CAP,code_SHA256=sha(__file__)))
    for name,script in [('measurement','measurement.py'),('missingness','missingness.py')]:
        out=(RUN/(name+'_stdout.log')).open('w');err=(RUN/(name+'_stderr.log')).open('w');logs.extend([out,err])
        children[name]=subprocess.Popen([sys.executable,'-u',str(RUN/'code'/script)],cwd=str(RUN/'code'),stdout=out,stderr=err,start_new_session=True,env={**os.environ,'OMP_NUM_THREADS':'1','MKL_NUM_THREADS':'1','OPENBLAS_NUM_THREADS':'1','CUDA_VISIBLE_DEVICES':''})
    update('running')
    while any(p.poll() is None for p in children.values()):
        if time.monotonic()-START>CAP or (RUN/'STOP').exists():
            for p in children.values():stop_child(p)
            update('STOPPED_AT_PARENT_BOUND');return
        for name,p in children.items():
            limit=7200 if name=='measurement' else 5400
            if p.poll() is None and time.monotonic()-START>limit:
                stop_child(p);status(name,state='FAIL_STOP',error='Supervisor enforced module time cap')
        update('running');time.sleep(2)
    verify();result=report();update(result['state'],A25_fit_attempts=result['A25_fit_attempts'])
if __name__=='__main__':
    try:main()
    except Exception as e:
        for p in children.values():stop_child(p)
        update('FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
    finally:
        for f in logs:f.close()
