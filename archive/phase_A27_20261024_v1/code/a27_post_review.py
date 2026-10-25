"""One reserved post-training replay; never trains or changes any scientific choice."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import json,time,gzip,traceback,math
from pathlib import Path
import torch
import a27_method as m
RUN=m.RUN;OUT=RUN/'post_review';P=m.read(RUN/'review_protocol.json')
def check():
    assert time.time()-m.read(RUN/'attribution/started.json')['unix_time']<m.B['total_seconds'],'Original6h cap'
    done=list((RUN/'models').glob('*/*/*/training_done.json'))
    if len(done)==30:assert time.time()-max(m.read(p)['unix_time'] for p in done)<1800,'Shared original post-training30min cap'
def main():
    assert not (OUT/'started.json').exists();OUT.mkdir(exist_ok=True);m.put(OUT/'started.json',dict(time=m.now(),pid=os.getpid(),protocol_SHA256=m.sha(RUN/'review_protocol.json'),code_SHA256=m.sha(__file__)))
    while not (RUN/'completion.json').exists():
        check();assert not (RUN/'stop.json').exists(),'Method stopped; do not retry';time.sleep(15)
    t=time.monotonic();m.verify();d=m.prepare();signals,ann,roles,ids,x,y,anchors,partial,targets,labels=d;assert roles==m.read(RUN/'patient_roles.json');assert len(list((RUN/'models').glob('*/*/*/attempt.json')))==30;assert sum(len(m.read(p)) for p in (RUN/'models').glob('*/*/*/history.json'))==360
    checked=[];signals_used=0
    for j in m.jobs():
        check();assert time.monotonic()-t<P['max_seconds'];path=m.path(j);done=m.read(path/'done.json');assert m.sha(path/'final.pt')==done['checkpoint_SHA256'];model=m.old.A9.Net();model.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu'));model.eval();r=roles[j['fold']]
        design=m.infer(model,signals,ann,r['design'],anchors);choice=m.design_choice(design);saved=m.read(path/'design_selection.json');assert choice==saved,'Design selection replay';cc=choice['choice'];cal=m.select(m.infer(model,signals,ann,r['calibration'],anchors),cc['mode'],cc['threshold']);outer=m.select(m.infer(model,signals,ann,r['outer'],anchors),cc['mode'],cc['threshold']);signals_used+=2*(len(r['design'])+len(r['calibration'])+len(r['outer']));assert signals_used<=P['max_lead_signals']
        for norm in [False,True]:
            for ninety in [False,True]:
                policy=('normalised' if norm else 'absolute')+('_wave90_secondary' if ninety else '_ALL_primary');recal=m.calibrate(cal,norm,ninety)
                with gzip.open(path/(policy+'.json.gz'),'rt') as f:prior=json.load(f)
                prev=prior['calibration'];assert recal['infinite']==prev['infinite'] and recal['n_eligible']==prev['n_eligible'] and recal['k']==prev['k']
                if not recal['infinite']:assert abs(recal['q']-prev['q'])<=P['quantile_absolute_tolerance']
                fresh=m.results(outer,signals,labels,recal,j,policy);assert fresh['counts']==prior['counts'] and fresh['patients']==prior['patients'] and fresh['waves']==prior['waves'],'Exact count/coverage replay'
                checked.append(dict(**j,policy=policy,state='PASS'))
        m.put(OUT/'status.json',dict(time=m.now(),state='running',models_checked=j['index']+1,lead_signals=signals_used))
    m.verify();gates=m.read(RUN/'analysis/gates.json');assert gates['strict_primary_not_replaced_by_wave90'];m.put(OUT/'contract_review.json',dict(time=m.now(),state='source_selection_quantile_counts_coverage_replay_PASS',models=30,policies_checked=len(checked),lead_signals=signals_used,new_fits=0,new_epochs=0,quantile_tolerance=1e-12,exact_integer_and_boolean_replay=True,measurement_gate_pass=gates['measurement_gate_pass'],clinical_noP_validated=False,NC_submission_ready=False,AF_morphology_gain=False,SPH_admitted=False,RDB_scored=False,new_figures=0,wall_seconds=time.monotonic()-t,details=checked))
    m.put(RUN/'status.json',dict(time=m.now(),state='bounded_complete_replay_reviewed',completed_jobs=30,epochs=360,review='post_review/contract_review.json',effectiveness_gate='analysis/gates.json',measurement_gate_pass=gates['measurement_gate_pass'],AF_fits=0,SPH_admitted=False,RDB_scored=False,new_figures=0))
    project=RUN.parents[1]/'PROJECT_STATE.json';autorun=RUN.parents[1]/'AUTORUN_STATE.json'
    if project.exists():
        state=m.read(project)
        if state.get('current_work_package')=='runs/phase_A27_20261024_v1':
            state['as_of']=m.now();state['A27'].update(state='bounded_complete_replay_reviewed',measurement_gate_pass=gates['measurement_gate_pass'],completed_jobs=30,completed_epochs=360,background_worker=False);m.put(project,state)
            if autorun.exists():
                state=m.read(autorun);state.update(coordinator_pid=None,A27_state='bounded_complete_replay_reviewed',current_worker='A27 complete;30jobs360epochs,replay review complete; effectiveness gate is separate; no AF fits/SPH/RDB/figures');m.put(autorun,state)
if __name__=='__main__':
    try:main()
    except Exception:m.put(OUT/'stop.json',dict(time=m.now(),state='REVIEW_FAIL_STOP',traceback=traceback.format_exc(),no_automatic_retry=True));raise
