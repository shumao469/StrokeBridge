"""One zero-fit review; never interpret absent labels as certified no-P."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,traceback,math
from pathlib import Path
import numpy as np
import pandas as pd
import a26_method as m
RUN=m.RUN;OUT=RUN/'post_review'
def bounded():
    if (RUN/'coordinator_started.json').exists():assert time.time()-m.read(RUN/'coordinator_started.json')['unix_time']<18000,'Fixed overall5h deadline'
def review():
    assert not (OUT/'started.json').exists(),'No review retry';OUT.mkdir(exist_ok=True);m.put(OUT/'started.json',dict(time=m.now(),pid=os.getpid(),addendum_SHA256=m.sha(RUN/'audit_addendum.txt'),code_SHA256=m.sha(Path(__file__))))
    while not (RUN/'completion.json').exists():
        bounded();assert not (RUN/'stop.json').exists(),'Coordinator stopped, no review';time.sleep(15)
    t=time.monotonic();m.verify_lock();z=np.load(m.INPUT/'data.npz');labels=z['labels'];d=m.data();signals,ann,roles,ids,x,y,ra,targets=d;rows=[];replays=[]
    assert len(list((RUN/'models').glob('*/*/*/attempt.json')))==30
    assert sum(len(pd.read_csv(p)) for p in (RUN/'models').glob('*/*/*/history.csv'))==360
    for j in m.jobs():
        bounded();assert time.monotonic()-t<1800,'Review30min cap';path=m.jp(j);done=m.read(path/'done.json');assert done['epochs']==12 and m.sha(path/'final.pt')==done['checkpoint_SHA256']
        model=m.A9.Net();model.load_state_dict(m.torch.load(path/'final.pt',map_location='cpu',weights_only=True));model.eval();outer=m.infer(model,signals,ann,roles[j['fold']]['outer'],ra);cal=m.read(path/'calibration.json');q=float('inf') if cal['patient_q_infinite'] else cal['patient_q_samples']
        for policy,qq in [('nominal',0),('patient',q)]:
            frozen=pd.read_csv(path/f'{policy}_counts.csv');fresh=[]
            for r in outer:
                p=r['patient'];l=r['li'];pred=m.accepted(r['raw'],qq);pairs=m.A9.match(r['ref'],pred);matched={k for i,k,iou in pairs};unknown=0;known_fp=0
                for k,v in enumerate(pred):
                    if k in matched:continue
                    if (labels[p,l,v['onset']:v['offset']+1]==-100).any():unknown+=1
                    else:known_fp+=1
                nm=len(pairs);n=len(pred);valid_ms=int(np.sum(labels[p,l,1000:4000]!=-100))*2
                fresh.append(dict(patient=p,lead=r['lead'],reference=len(r['ref']),predicted=n,matched=nm))
                rows.append(dict(**j,policy=policy,patient=p,lead=r['lead'],reference=len(r['ref']),predicted=n,matched=nm,unknown_unmatched=unknown,known_region_unmatched=known_fp,fully_unknown_pair=bool(np.all(labels[p,l]==-100)),labelled_central_ms=valid_ms,precision_lower=nm/n if n else None,precision_upper=(nm+unknown)/n if n else None,precision_scorable=nm/(n-unknown) if n>unknown else None))
            fresh=pd.DataFrame(fresh)
            assert fresh[['patient','lead','reference','predicted','matched']].equals(frozen[['patient','lead','reference','predicted','matched']]),'Frozen count replay failure'
            replays.append(dict(**j,policy=policy,count_replay='PASS'))
    table=pd.DataFrame(rows);table.to_csv(OUT/'partial_reference_patient_counts.csv',index=False);summary=[]
    for keys,group in table.groupby(['arm','seed','policy','lead']):
        n=int(group.predicted.sum());nm=int(group.matched.sum());u=int(group.unknown_unmatched.sum());nr=int(group.reference.sum())
        summary.append(dict(arm=keys[0],seed=int(keys[1]),policy=keys[2],lead=keys[3],predicted=n,matched=nm,unknown_unmatched=u,known_region_unmatched=int(group.known_region_unmatched.sum()),precision_lower=nm/n if n else None,precision_upper=(nm+u)/n if n else None,precision_scorable=nm/(n-u) if n>u else None,recall=nm/nr if nr else None,fully_unknown_pairs=int(group.fully_unknown_pair.sum())))
    pd.DataFrame(summary).to_csv(OUT/'summary.csv',index=False);gates=m.read(RUN/'analysis/gates.json');m.put(OUT/'contract_review.json',dict(time=m.now(),state='zero_fit_contract_replay_pass_with_reference_limits',count_replays=replays,partial_reference_identification_bounds='summary.csv',clinical_noP_validated=False,measurement_engineering_pass=gates['measurement_engineering_pass'],AF_morphology_gain_established=False,NC_submission_ready=False,RDB_scored=False,SPH_admitted=False,model_fits=0,new_epochs=0,new_figures=0,wall_seconds=time.monotonic()-t))
    m.put(RUN/'status.json',dict(time=m.now(),state='bounded_complete_reviewed',completed_jobs=30,completed_epochs=360,contract_review='post_review/contract_review.json',engineering_gate='analysis/gates.json',SPH_admitted=False,RDB_scored=False,new_figures=0))
if __name__=='__main__':
    try:review()
    except Exception:m.put(OUT/'stop.json',dict(time=m.now(),state='REVIEW_FAIL_STOP',traceback=traceback.format_exc(),no_automatic_retry=True));raise
