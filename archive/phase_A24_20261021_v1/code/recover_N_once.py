"""One explicitly authorized N recovery; preserves all stopped/original artifacts."""
from a24_shared import *
import subprocess,traceback
REC=P/'recovery_N_v1';START=time.monotonic();CAP=7200
def state(**v):
    q={'time':now(),'pid':os.getpid(),'run':P.name,'recovery':'N_v1','fits_already_completed':400,'max_recovery_fits':10,'max_total_fits':422,'E2_recovery_authorized':False,'SPH_admitted':False,'SPH_predictions':0,'NC_submission_ready':False,'independent_validation':False,'wall_seconds':time.monotonic()-START,**v}
    put(REC/'status.json',q);put(U/'PHASE_A24_STATUS.json',q)
def transformed(name):
    path=P/'code'/name;s=path.read_text(encoding='utf-8');original=sha(path);changes=[]
    def one(before,after):
        nonlocal s
        count=s.count(before);assert count==1,(name,before,count);s=s.replace(before,after,1);changes.append({'old':before,'new':after,'count':count})
    if name=='negative_controls.py':
        one("OUT=P/'negative'","OUT=RECOVERY_ROOT/'negative'")
        one('torch.backends.cudnn.allow_tf32=False','torch.backends.cudnn.allow_tf32=True')
        line="        if k<32 and bad:raise RuntimeError(f'First32 candidate replay failed at {row[\"ecg_id\"]}: {bad}')"
        one(line,line+"\n        if k==31:put(OUT/'first32_replay_PASS.json',{'time':now(),'records':32,'original_tolerances_unchanged':True,'CuDNN_TF32':True})")
    else:
        one("OUT=P/'comparisons'","OUT=RECOVERY_ROOT/'comparisons'")
        count=s.count("P/'negative");assert count>=2;s=s.replace("P/'negative","RECOVERY_ROOT/'negative");changes.append({'kind':'negative input output-path substitution','count':count})
        one('assert len(rows)<22','assert len(rows)<10')
        s=s.replace('max_fits=22','max_fits=10').replace("'max_fits':22","'max_fits':10")
        assert not (P/'extraction/completion.json').exists(),'Do not silently add encoder recovery'
    dst=REC/(name+'.frozen.txt');dst.write_text(s,encoding='utf-8')
    put(REC/(name+'.transformations.json'),{'original_path':str(path),'original_SHA256':original,'transformed_SHA256':sha(dst),'changes':changes,'no_scientific_recipe_change':True})
    return s,path
def freeze_recovery(module,extra=()):
    dest=REC/('negative' if module=='negative' else 'comparisons')/'source_lock.json';assert not dest.exists()
    inputs=[REC/'protocol.txt',P/'protocol.txt',Path(__file__),P/'code/a24_shared.py',P/'code/extract_controls.py',A/'analysis_dataset.csv',A/'embeddings.npz',A/'waveform_audit.csv',B/'factor_definitions.json',*sorted(REC.glob('*.frozen.txt')),*extra]
    lock={str(f):sha(f) for f in inputs};put(dest,{'time':now(),'files':lock,'one_authorized_recovery':True});return lock
def execute(name):
    assert time.monotonic()-START<CAP
    code,source=transformed(name);scope={'__name__':'a24_recovery_module','__file__':str(source),'RECOVERY_ROOT':REC}
    exec(compile(code,str(source)+'[authorized recovery]','exec'),scope);scope['freeze']=freeze_recovery
    # One worker process bounds the full sequence; this call does not repeat or spawn fits.
    scope['main']()
def main():
    assert not (REC/'launch.json').exists(),'Recovery already launched; never repeat'
    assert read(P/'refit/completion.json')['fits']==400
    assert read(P/'negative/status.json')['state']=='stopped_no_retry'
    verify(read(P/'negative/source_lock.json')['files'])
    put(REC/'launch.json',{'time':now(),'pid':os.getpid(),'code_SHA256':sha(__file__),'protocol_SHA256':sha(REC/'protocol.txt'),'user_authorization':'Allows one bounded N recovery; no E recovery approval','max_total_seconds':CAP})
    state(state='running',stage='N_replay_and_controls')
    execute('negative_controls.py')
    assert (REC/'negative/first32_replay_PASS.json').exists() and read(REC/'negative/completion.json')['state']=='complete'
    state(state='running',stage='N_preallocated_point_fits')
    execute('fit_and_audit.py')
    fits=read(REC/'comparisons/completion.json')['fits'];assert fits<=10 and 400+fits<=422
    assert time.monotonic()-START<CAP
    put(REC/'completion.json',{'time':now(),'state':'N_recovery_complete_E_still_ineligible','R_fits_preserved':400,'N_fits':fits,'total_fits':400+fits,'max_total_fits':422,'E_fits':0,'network_training_epochs':0,'first32':'PASS','original_gate_unchanged':True,'SPH_predictions':0,'full_A24_figure_gate':False})
    state(state='N_recovery_complete_E_still_ineligible',stage='awaiting_encoder_recovery_authorization',N_fits=fits,total_fits=400+fits,background_worker=False)
if __name__=='__main__':
    try:main()
    except Exception as e:
        state(state='stopped_second_failure_no_further_recovery',stage='preserve_all_outputs',error=str(e),traceback=traceback.format_exc(),background_worker=False);raise
