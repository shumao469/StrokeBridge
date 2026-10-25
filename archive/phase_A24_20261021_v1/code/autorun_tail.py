"""Bounded stage coordinator. Never retries a failed scientific or acquisition module."""
from a24_shared import *
import subprocess
START=time.monotonic();TOTAL=6*3600
def state(**v):
    record={'time':now(),'pid':os.getpid(),'run':P.name,'max_fits':422,'SPH_admitted':False,'SPH_predictions':0,'branches':{'A':'public technical development','B':'hospital/expert evidence waiting'},'independent_validation':False,'NC_submission_ready':False,'wall_seconds':time.monotonic()-START,**v}
    put(P/'autorun_status.json',record);put(U/'PHASE_A24_STATUS.json',record)
def run(file,seconds):
    left=TOTAL-(time.monotonic()-START);assert left>0
    state(state='running',stage=file)
    log=(P/(file.replace('.py','')+'.log')).open('w',encoding='utf-8')
    with log:
        result=subprocess.run([sys.executable,'-X','utf8','-B',str(P/'code'/file)],cwd=P,stdout=log,stderr=subprocess.STDOUT,timeout=min(seconds,left))
    return result.returncode
def main():
    assert not (P/'autorun_launch.json').exists(),'Coordinator already launched; no duplicate'
    put(P/'autorun_launch.json',{'time':now(),'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(P/'protocol.txt'),'max_seconds':TOTAL})
    state(state='waiting',stage='existing_preallocated_R_refits')
    while not (P/'refit/completion.json').exists():
        s=read(P/'refit/status.json') if (P/'refit/status.json').exists() else {}
        if s.get('state')=='stopped_no_retry':raise RuntimeError('R stopped; preserve outputs, no retry')
        if time.monotonic()-START>7200:raise TimeoutError('R queue wait cap')
        time.sleep(5)
    assert read(P/'refit/completion.json')['fits']==400
    code=run('negative_controls.py',7200)
    if code:raise RuntimeError('N stopped; no retry; no full figure gate')
    # E needs a complete verified acquisition receipt,including any separately approved recovery.
    if (P/'runtime/completion.json').exists():
        code=run('extract_controls.py',7200)
        if code:put(P/'encoder_closed_not_admitted.json',{'time':now(),'reason':'Encoder-specific preflight/extraction failed; no retry','no_independent_validation':True})
    else:put(P/'encoder_closed_not_admitted.json',{'time':now(),'reason':'Official weight incomplete; original download stopped. No unapproved recovery or inference.','no_independent_validation':True})
    code=run('fit_and_audit.py',2700)
    if code:raise RuntimeError('Point comparison stopped; no retry')
    point=read(P/'comparisons/completion.json');fits=400+point['fits'];assert fits<=422
    receipt={'time':now(),'numerical_gate':True,'R_complete':True,'N_complete':True,'E_complete':point['encoder_comparison_complete'],'fits':fits,'max_fits':422,'post_hoc_development':True,'clinical_or_independent_validation':False,'NC_submission_ready':False}
    put(P/'analysis_verified.json',receipt)
    # If all requested scientific modules pass,trigger the separately authored figure job.
    builder=P/'code/build_verified_figures.py'
    if point['encoder_comparison_complete'] and builder.exists():
        code=run(builder.name,2700)
        if code:state(state='analyses_complete_figure_delivery_stopped',stage='figure_QA',fits=fits);return
        state(state='bounded_execution_complete_not_submission_ready',stage='complete',fits=fits)
    else:state(state='verified_partial_results_encoder_or_figure_gate_pending',stage='awaiting_new_authority_or_completed_encoder_gate',fits=fits,figure_job_started=False,E_complete=point['encoder_comparison_complete'])
if __name__=='__main__':
    try:main()
    except Exception as e:state(state='stopped_no_retry',stage='preserve_partial_outputs',error=str(e));raise
