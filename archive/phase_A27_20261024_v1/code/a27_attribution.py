"""Frozen A26 candidate decomposition. No optimiser, relabelling, or encoder fitting."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,json,time,gzip,hashlib,traceback,importlib.util
from pathlib import Path
import numpy as np,pandas as pd,torch
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];OLD=BASE/'runs/phase_A26_20261024_v1';OUT=RUN/'attribution'
sys.path.insert(0,str(OLD/'code'));import a26_method as old
torch.set_num_threads(1)
B=json.loads((RUN/'budget.json').read_text());MODES=['continuous','ordered','union'];TARGETS=[.6,.7,.8,.9]
def put(p,v):old.put(p,v)
def read(p):return old.read(p)
def sha(p):return old.sha(p)
def grid(records,lead,mode):
    vals=[p['confidence'] for r in records if r['lead']==lead for p in r['raw'] if mode=='union' or p['kind']==mode]
    return sorted(set([0.,1.]+([float(v) for v in np.quantile(vals,np.linspace(0,1,63))] if vals else [])))
def proposals(r,mode,threshold):
    return old.nms([p for p in r['raw'] if (mode=='union' or p['kind']==mode) and p['confidence']>=threshold and max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100])
def evaluate(records,labels,mode,thresholds):
    rows=[]
    for r in records:
        pred=proposals(r,mode,thresholds[r['lead']]);pairs=old.A9.match(r['ref'],pred);matched={j for i,j,iou in pairs};known=unknown=0;by={k:dict(known_unmatched=0,unknown_unmatched=0,matched=0,predicted=0) for k in ['continuous','ordered']}
        for j,p in enumerate(pred):
            v=by[p['kind']];v['predicted']+=1
            if j in matched:v['matched']+=1
            elif (labels[r['patient'],r['li'],p['onset']:p['offset']+1]==-100).any():unknown+=1;v['unknown_unmatched']+=1
            else:known+=1;v['known_unmatched']+=1
        rows.append(dict(patient=r['patient'],lead=r['lead'],reference=len(r['ref']),predicted=len(pred),matched=len(pairs),known_unmatched=known,unknown_unmatched=unknown,by_surviving_source=by))
    return rows
def totals(rows):
    nr=sum(r['reference'] for r in rows);n=sum(r['predicted'] for r in rows);nm=sum(r['matched'] for r in rows);u=sum(r['unknown_unmatched'] for r in rows)
    return dict(reference=nr,predicted=n,matched=nm,recall=nm/nr if nr else None,precision_lower=nm/n if n else None,precision_upper=(nm+u)/n if n else None,known_unmatched=sum(r['known_unmatched'] for r in rows),unknown_unmatched=u,source={k:{f:sum(r['by_surviving_source'][k][f] for r in rows) for f in ['known_unmatched','unknown_unmatched','matched','predicted']} for k in ['continuous','ordered']})
def main():
    assert not (OUT/'started.json').exists(),'One invocation only';OUT.mkdir(exist_ok=True);t=time.monotonic();put(OUT/'started.json',dict(time=old.now(),pid=os.getpid(),unix_time=time.time(),code_SHA256=sha(__file__),protocol_SHA256=sha(RUN/'protocol.txt')))
    old.verify_lock();z=np.load(old.INPUT/'data.npz');labels=z['labels'];d=old.data();signals,ann,roles,ids,x,y,ra,targets=d
    audit=read(BASE/'runs/phase_A7b_20261017_v1/annotation_audit.json');counts={str(v):int((labels==v).sum()) for v in [-100,0,1,2,3]};assert counts==audit['class_samples']
    assert np.all(labels[np.all(labels==-100,axis=2)]==-100)
    put(OUT/'label_provenance.json',dict(source='A7b annotation bracket/QRS-gap structural mask; no clinical absence certification',class_samples=counts,admitted_leads=audit['admitted_leads'],quarantined_leads=audit['quarantined_leads'],unknown_label=-100,background_label=0,clinical_noP=False,source_builder_SHA256=sha(BASE/'runs/phase_A7b_20261017_v1/a7b_annotation_audit.py')))
    fixed=[];matched=[];replays=[];signals_used=0
    for j in old.jobs():
        assert time.monotonic()-t<B['zero_fit_seconds'];assert not (RUN/'STOP_NEW_WORK').exists();path=old.jp(j);receipt=read(path/'done.json');assert sha(path/'final.pt')==receipt['checkpoint_SHA256']
        model=old.A9.Net();model.load_state_dict(torch.load(path/'final.pt',weights_only=True,map_location='cpu'));model.eval();role=roles[j['fold']];cal=old.infer(model,signals,ann,role['selection']+role['calibration'],ra);outer=old.infer(model,signals,ann,role['outer'],ra);signals_used+=2*(len(role['selection'])+len(role['calibration'])+len(role['outer']));assert signals_used<=3840
        for r in outer:
            pred=old.accepted(r['raw'],0);nm=len(old.A9.match(r['ref'],pred));saved=pd.read_csv(path/'nominal_counts.csv');row=saved[(saved.patient==r['patient'])&(saved.lead==r['lead'])];assert len(row)==1 and int(row.iloc[0].reference)==len(r['ref']) and int(row.iloc[0].predicted)==len(pred) and int(row.iloc[0].matched)==nm
        replays.append(dict(**j,state='PASS'))
        cache=OUT/'cache'/f"job_{j['index']:02d}.json.gz";cache.parent.mkdir(exist_ok=True)
        with gzip.open(cache,'wt',encoding='utf8') as f:json.dump(dict(job=j,calibration=cal,outer=outer),f,allow_nan=False)
        for mode in MODES:
            baseline=evaluate(outer,labels,mode,dict(ii=.5,v1=.5))
            for lead in ['ii','v1']:fixed.append(dict(**j,mode=mode,lead=lead,policy='fixed_0.5',**totals([r for r in baseline if r['lead']==lead])))
            for lead in ['ii','v1']:
                curve=[]
                for ct in grid(cal,lead,mode):
                    rr=evaluate([r for r in cal if r['lead']==lead],labels,mode,{lead:ct});curve.append(dict(threshold=ct,**totals(rr)))
                for target in TARGETS:
                    feasible=[v for v in curve if v['recall'] is not None and target<=v['recall']<=target+.02 and v['precision_lower'] is not None]
                    if not feasible:matched.append(dict(**j,mode=mode,lead=lead,target=target,feasible=False,reason='No calibration point in frozen coverage band'));continue
                    choice=max(feasible,key=lambda v:(v['precision_lower'],v['threshold']));res=totals(evaluate([r for r in outer if r['lead']==lead],labels,mode,{lead:choice['threshold']}));matched.append(dict(**j,mode=mode,lead=lead,target=target,feasible=True,threshold=choice['threshold'],calibration=choice,outer=res))
        put(OUT/'status.json',dict(time=old.now(),state='running',completed_models=len(replays),max_models=30,lead_signals=signals_used,model_fits=0,epochs=0));print('N0 model',j['index'],'complete',flush=True)
    put(OUT/'fixed_source_decomposition.json',fixed);put(OUT/'calibration_selected_operating_points.json',matched)
    comparisons=[]
    for seed in old.B['seeds']:
        for mode in MODES:
            for target in TARGETS:
                for lead in ['ii','v1']:
                    paired=[]
                    for fold in range(5):
                        aa=[v for v in matched if v['arm']=='C0_complete' and v['seed']==seed and v['fold']==fold and v['mode']==mode and v['target']==target and v['lead']==lead]
                        bb=[v for v in matched if v['arm']=='C2_ordered_partial' and v['seed']==seed and v['fold']==fold and v['mode']==mode and v['target']==target and v['lead']==lead]
                        a,b=aa[0],bb[0]
                        if a['feasible'] and b['feasible']:paired.append(dict(fold=fold,C0=a['outer'],C2=b['outer'],outer_coverage_matched=abs(a['outer']['recall']-b['outer']['recall'])<=.02))
                    comparisons.append(dict(seed=seed,mode=mode,target=target,lead=lead,fold_pairs=paired,selection='calibration only; unmatched outer comparisons cannot claim equal-coverage superiority'))
    put(OUT/'fair_comparisons.json',comparisons)
    mechanism=[]
    for seed in old.B['seeds']:
        a=[v for v in fixed if v['arm']=='C0_complete' and v['seed']==seed and v['lead']=='v1' and v['mode']=='union'];b=[v for v in fixed if v['arm']=='C2_ordered_partial' and v['seed']==seed and v['lead']=='v1' and v['mode']=='union']
        mechanism.append(dict(seed=seed,C0_known_unmatched=sum(v['known_unmatched'] for v in a),C2_known_unmatched=sum(v['known_unmatched'] for v in b),by_source_C0={k:sum(v['source'][k]['known_unmatched'] for v in a) for k in ['continuous','ordered']},by_source_C2={k:sum(v['source'][k]['known_unmatched'] for v in b) for k in ['continuous','ordered']}))
    supported=any(v['C2_known_unmatched']>v['C0_known_unmatched'] for v in mechanism);put(OUT/'completion.json',dict(time=old.now(),state='PASS' if supported else 'STOP_UNSUPPORTED_REPAIR_HYPOTHESIS',replayed_models=len(replays),lead_signals=signals_used,sign_orbit_passes=signals_used*2,model_fits=0,epochs=0,mechanism=mechanism,repair_hypothesis_supported=supported,label_background_reference_only=True,clinical_noP_certified=False,training_gate_pass=supported,wall_seconds=time.monotonic()-t))
if __name__=='__main__':
    try:main()
    except Exception:put(OUT/'stop.json',dict(time=old.now(),state='FAIL_STOP',traceback=traceback.format_exc(),automatic_retry=False));raise
