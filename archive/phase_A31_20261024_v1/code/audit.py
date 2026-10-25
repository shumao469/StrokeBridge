"""A31 bounded diagnostic only. No loss/gradient/optimiser/training calls."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,math,hashlib,traceback,subprocess,importlib.util
from pathlib import Path
import numpy as np
import torch
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A30=BASE/'runs/phase_A30_20261024_v1'
sys.path.insert(0,str(A30/'code'));import method as q
sys.path.insert(0,str(Path(__file__).resolve().parent))
e=q.e;m=q.m;d=q.d;torch.set_num_threads(1);torch.set_grad_enabled(False)
read=q.read;put=q.put;sha=q.sha;now=q.now;gz=q.gz;ungz=q.ungz;B=read(RUN/'budget.json')
def load(name,path):
    s=importlib.util.spec_from_file_location(name,path);v=importlib.util.module_from_spec(s);s.loader.exec_module(v);return v
tr=load('a31_provenance_trace',RUN/'code/trace.py');ind=load('a31_independent_geometry',q.A29/'code/review.py')

def check():
    assert not (RUN/'STOP').exists(),'A31 owned worker/coordinator stop'
    assert time.time()<read(RUN/'started.json')['deadline_unix'],'A31 fixed7200s deadline'

def verify():
    for p,h in read(RUN/'source_lock.json').items():check();assert sha(p)==h,f'Frozen source changed:{p}'

def size():
    n=0
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        try:n+=p.stat().st_size
        except FileNotFoundError:
            if p.suffix!='.tmp':raise
    return n

def supports(ref,r):
    return r is not None and 10<=ref[2]-ref[0]<=100 and r-150<=ref[0]<=ref[2]<=r-11

def bare(p):return {k:v for k,v in p.items() if k not in ['wave_R','expert_beat']}

def support_key(p):
    dur=p['offset']-p['onset'];return (p['confidence']*math.sqrt(dur/100),p['confidence'],dur,-p['uid'])

def select(r,choice,anchors,policy):
    values=[bare(p) for p in r['raw'] if (choice['mode']=='union' or p['kind']==choice['mode']) and p['confidence']>=choice['threshold']]
    if policy=='O1_ordered_restriction':values=[p for p in values if p['kind']=='ordered']
    if policy=='C2_continuous_restriction':values=[p for p in values if p['kind']=='continuous']
    if policy!='S3_support_first':ps,ledger=d.merge(values,anchors,support_rank=True)
    else:
        groups={};unresolved=[]
        for p in values:
            r0=d.wave_anchor(p,anchors)
            if r0 is None:unresolved.append(dict(p,rejection='waveform_beat_unresolved'));continue
            groups.setdefault(r0,[]).append(dict(p,wave_R=r0))
        ps=[max(v,key=support_key) for r0,v in sorted(groups.items())];ledger=dict(counterfactual='support_order_before_NMS',unresolved=unresolved,groups=[dict(wave_R=r0,winner_uid=max(v,key=support_key)['uid'],raw_count=len(v)) for r0,v in sorted(groups.items())])
    decorated=d.decorate(ps,anchors,r['qrs']);assert len({p['wave_R'] for p in decorated})==len(decorated)
    return dict(r,selected=decorated,merger=ledger)

def stage(ref,ps,qrs):
    return dict(reference=len(ref),predicted=len(ps),matched=len(d.point_pairs(ref,ps,qrs)),identity_matched=len(d.point_pairs(ref,ps,qrs,True)))

def assigned(r):
    s=d.scores(r['ref'],r['selected'],r['qrs'])['strict']
    if not math.isfinite(s):return d.identity_pairs(r['ref'],r['selected'],r['qrs'])
    if not r['ref'] or not r['selected']:return []
    return d.assign([[1/(1+d.distance(z,p)) if d.compatible(z,p,r['qrs']) and d.distance(z,p)<=s else 0 for p in r['selected']] for z in r['ref']])

def audit_record(r,choice,anchors):
    assert e.clean(d.decorate([bare(p) for p in r['raw']],anchors,r['qrs']))==r['raw'],'Cached waveform-R/expert identity'
    factual=select(r,choice,anchors,'F0_factual');assert factual['selected']==r['selected'] and factual['merger']==r['merger'],'Exact frozen merger replay'
    raw=r['raw'];source=[p for p in raw if choice['mode']=='union' or p['kind']==choice['mode']];accepted=[p for p in source if p['confidence']>=choice['threshold']];resolved=[p for p in accepted if p['wave_R'] is not None];groups={}
    for p in resolved:groups.setdefault(p['wave_R'],[]).append(p)
    local=[p for rr,pp in sorted(groups.items()) for p in d.nms(pp)];stages=[dict(name=name,**stage(r['ref'],ps,r['qrs'])) for name,ps in [('raw',raw),('source',source),('confidence',accepted),('resolved_R',resolved),('within_R_NMS',local),('final',r['selected'])]];nn=[v['matched'] for v in stages];assert all(a>=b for a,b in zip(nn,nn[1:]))
    pairs=dict(assigned(r));events=[]
    for i,z in enumerate(r['ref']):
        beat=d.next_expert(z[2],r['qrs']);expert_R=r['qrs'][beat] if beat is not None else None;waves=[int(a) for a in anchors if beat is not None and d.map_expert(int(a),r['qrs'])==beat];same=[p for p in raw if beat is not None and p['expert_beat']==beat];admitted=[p for p in accepted if beat is not None and p['expert_beat']==beat];good=[p for p in same if d.iou(z,p)>=.5];agood=[p for p in admitted if d.iou(z,p)>=.5];p=r['selected'][pairs[i]] if i in pairs else None;selected_good=p is not None and d.iou(z,p)>=.5
        err=lambda v:max(abs(v['onset']-z[0]),abs(v['offset']-z[2]))
        best=min(admitted,key=err) if admitted else None;rawbest=min(same,key=err) if same else None;ep=supports(z,expert_R);wp=any(supports(z,a) for a in waves)
        if beat is None:reason='reference_identity_unresolved'
        elif not ep:reason='expert_R_window_unrepresentable'
        elif not wp:reason='waveform_R_or_window_support_failure'
        elif agood and not selected_good:reason='admitted_good_candidate_lost_in_selection'
        elif good and not agood:reason='raw_good_candidate_filtered_by_source_or_confidence'
        elif not good:reason='no_good_raw_same_beat_boundary_proposal'
        else:reason='point_match_available_check_endpoint_tail_separately'
        event=dict(patient=r['patient'],record=r['record'],lead=r['lead'],reference_index=i,reference=z,reference_expert_beat=beat,expert_R=expert_R,waveform_Rs=waves,expert_R_supported=ep,any_waveform_R_supported=wp,selected_R_supported=supports(z,p['wave_R']) if p else False,selected=p,selected_good_IoU=selected_good,selected_endpoint_error_samples=err(p) if p else None,selected_normalised_residual=d.distance(z,p) if p else None,missing_same_beat_assignment=p is None,raw_good_uids=[p['uid'] for p in good],admitted_good_uids=[p['uid'] for p in agood],best_raw_same_beat=rawbest,best_admitted_same_beat=best,best_admitted_endpoint_error_samples=err(best) if best else None,admitted_endpoint_dominates_selected=best is not None and p is not None and err(best)<err(p),same_R_endpoint_alternative_better=bool(p) and any(v['wave_R']==p['wave_R'] and err(v)<err(p) for v in admitted),reason=reason,boundary_training_candidate=beat is not None and ep and wp and not good,expert_beat_multiple_reference_P=beat is not None and sum(d.next_expert(v[2],r['qrs'])==beat for v in r['ref'])>1)
        events.append(event)
    return dict(patient=r['patient'],record=r['record'],lead=r['lead'],role=r['role'],stages=stages,branch_raw={k:stage(r['ref'],[p for p in raw if p['kind']==k],r['qrs']) for k in ['ordered','continuous']},branch_admitted={k:stage(r['ref'],[p for p in accepted if p['kind']==k],r['qrs']) for k in ['ordered','continuous']},selected_branch_counts={k:sum(p['kind']==k for p in r['selected']) for k in ['ordered','continuous']},events=events)

def patient_extremes(records):
    out=[]
    for pat in sorted({r['patient'] for r in records}):
        rr=[r for r in records if r['patient']==pat and r['ref']]
        if not rr:continue
        ss=[d.scores(r['ref'],r['selected'],r['qrs']) for r in rr];out.append(dict(patient=pat,strict=max(v['strict'] for v in ss),missing=sum(v['missing'] for v in ss)))
    return sorted(out,key=lambda x:(-x['strict'],-x['missing'],x['patient']))

def trace_patients(j,cache,anchors,signals,ann,out):
    if j['arm']!='C2_balanced_tail':return dict(models=0,lead_signals=0,sign_orbit_passes=0)
    cal=[p['patient'] for p in patient_extremes(cache['calibration'])[:3]]
    if any(r['patient']==12 for r in cache['calibration']) and 12 not in cal:cal.append(12)
    outer=[p['patient'] for p in patient_extremes(cache['outer'])[:2]];chosen=[('calibration',p) for p in cal]+[('outer',p) for p in outer];assert len(chosen)<=6
    model=m.old.A9.Net();model.load_state_dict(torch.load(q.jobpath(j)/'final.pt',weights_only=True,map_location='cpu'));model.eval();records=[]
    for role,pat in chosen:
        check()
        with torch.inference_mode():pr=m.old.fwd(model,torch.from_numpy(m.old.A9.channels(signals[pat,:,[1,6]].copy()))).softmax(1).numpy()
        assert pr.shape==(2,4,5000) and np.isfinite(pr).all()
        for k,li in enumerate([1,6]):
            r=next(r for r in cache[role] if r['patient']==pat and r['li']==li);raw=d.posterior_decode(pr[k],signals[pat,:,li],anchors[pat]);assert e.clean(d.decorate(raw,anchors[pat],r['qrs']))==r['raw'],'Frozen trace raw cache replay'
            cont,ledger=tr.continuous(pr[k],signals[pat,:,li],anchors[pat],d);expected=[p for p in raw if p['kind']=='continuous'];assert e.clean(cont)==e.clean([{k:v for k,v in p.items() if k!='uid'} for p in expected]),'Independent continuous provenance output'
            it=iter(expected)
            for v in ledger:
                if v['discard'] is not None:continue
                p=next(it);v['uid']=p['uid'];v['reference_overlap']=[dict(reference=z,grown_overlap_samples=max(0,min(v['grown'][1],z[2])-max(v['grown'][0],z[0])),final_IoU=d.iou(z,p)) for z in r['ref'] if min(v['grown'][1],z[2])>=max(v['grown'][0],z[0])]
            records.append(dict(patient=pat,record=r['record'],lead=r['lead'],role=role,trace=ledger))
    gz(out/'provenance_trace.json.gz',dict(job=j,selection='outcome_enriched_diagnostic_not_validation',records=records));return dict(models=1,lead_signals=len(records),sign_orbit_passes=2*len(records))

def audit_job(j,data):
    check();out=RUN/'jobs'/f"job_{j['index']:02d}";assert not out.exists(),'No A31 audit-job retry';out.mkdir(parents=True);put(out/'attempt.json',dict(time=now(),job=j));signals,labels,ann,anchors=data;cache=ungz(q.jobpath(j)/'prediction_cache.json.gz');old=ungz(q.jobpath(j)/'result.json.gz');choice=old['choice'];diags={};policies={p:{} for p in B['policies']}
    for role in ['design','calibration','outer']:
        diags[role]=[audit_record(r,choice,anchors[r['patient']]) for r in cache[role]]
        for policy in B['policies']:policies[policy][role]=[select(r,choice,anchors[r['patient']],policy) for r in cache[role]]
        gz(out/(role+'_diagnostics.json.gz'),diags[role])
    for policy,rr in policies.items():
        check();cc=e.calibrate(rr['calibration']);result=e.evaluate(rr['outer'],cc,signals,labels);design={}
        for lead in ['ii','v1']:
            rows=[r for r in rr['design'] if r['lead']==lead];nr=sum(len(r['ref']) for r in rows);nm=sum(len(d.point_pairs(r['ref'],r['selected'],r['qrs'])) for r in rows);design[lead]=dict(reference=nr,matched=nm,recall=nm/max(nr,1))
        if policy=='F0_factual':
            assert e.clean(cc)==old['calibration'],'Exact ALL/component calibration replay'
            for key in ['records','patients','widths','component_widths_DIAGNOSTIC']:assert e.clean(result[key])==old[key],f'Exact frozen A30 output:{key}'
            prior=read(q.jobpath(j)/'design.json')
            for lead in ['ii','v1']:assert design[lead]['matched']==prior['by_lead'][lead]['matched'] and design[lead]['reference']==prior['by_lead'][lead]['reference']
        gz(out/(policy+'.json.gz'),dict(job=j,policy=policy,diagnostic_not_promoted=policy!='F0_factual',design=design,design_feasible=all(v['recall']>=.8 for v in design.values()),choice=choice,**result))
    extreme=[]
    for role in ['calibration','outer']:
        ranked=patient_extremes(cache[role])
        for p in ranked[:3]:
            events=[v for r in diags[role] if r['patient']==p['patient'] for v in r['events']]
            key=lambda v:(float('inf') if v['selected_normalised_residual'] is None else v['selected_normalised_residual'],-v['reference'][1])
            worst=max(events,key=key);extreme.append(dict(role=role,patient=p['patient'],patient_strict=p['strict'],missing=p['missing'],worst_event=worst))
    put(out/'extreme_patients.json',extreme);trace=trace_patients(j,cache,anchors,signals,ann,out)
    put(out/'done.json',dict(time=now(),job=j,cached_lead_records=sum(len(v) for v in cache.values() if isinstance(v,list)),policies=4,original_model_replay=True,new_fits=0,new_epochs=0,trace=trace))

def preflight():
    tests=d.preflight()
    def p(a,b,c,uid):return dict(onset=a,offset=b,peak=(a+b)//2,confidence=c,uid=uid,kind='ordered',source_R=1200,scale=5.,box=[a,a,b,b])
    rr=dict(raw=d.decorate([p(1100,1110,.9,0),p(1096,1114,.8,1)],[1200],[1200]),qrs=[1200],ref=[(1096,1105,1114)])
    choice=dict(mode='union',threshold=.5);a=select(rr,choice,[1200],'F0_factual');b=select(rr,choice,[1200],'S3_support_first');assert a['selected'][0]['uid']==0 and b['selected'][0]['uid']==1
    assert supports((1040,1050,1060),1150) and not supports((1040,1050,1060),1200)
    assert not select(rr,dict(mode='ordered',threshold=.5),[1200],'C2_continuous_restriction')['selected']
    prob=np.zeros((4,5000));prob[1]=.01;prob[0]=.99;prob[1,1300:1351]=.9;prob[0,1300:1351]=.1;z=np.sin(np.arange(5000)/7);raw=d.posterior_decode(prob,z,[1400]);cont,trace=tr.continuous(prob,z,[1400],d);assert cont==[{k:v for k,v in p.items() if k!='uid'} for p in raw if p['kind']=='continuous']
    assert ind.cardinality([[True,True],[True,False]])==2
    return tests+['support_NMS_noncommutation','expert_vs_shifted_R_support','restriction_never_enables_forbidden_source','continuous_instrumentation_matches_decoder','independent_augmenting_paths']

def worker(w):
    verify();z=np.load(m.old.INPUT/'data.npz');signals=z['signals'];labels=z['labels'];ann=read(m.old.INPUT/'pwave_annotations.json');anchors=[m.old.anchors(v) for v in signals];data=(signals,labels,ann,anchors)
    for j in q.jobs():
        if j['index']%2==w:
            audit_job(j,data);put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='running',last_job=j['index']))
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))

def freeze():
    assert read(A30/'completion.json')['state']=='bounded_training_complete_review_PASS' and read(A30/'post_review/contract_review.json')['state']=='PASS'
    locks=read(A30/'source_lock.json');extra=[RUN/'protocol.txt',RUN/'budget.json',RUN/'code/audit.py',RUN/'code/trace.py',RUN/'code/finish.py',A30/'source_lock.json',A30/'completion.json',A30/'post_review/contract_review.json',A30/'analysis/summary.json',A30/'analysis/gates.json',q.A29/'reference_admission.json']
    for j in q.jobs():extra +=[q.jobpath(j)/f for f in ['prediction_cache.json.gz','result.json.gz','final.pt','done.json','design.json','training_done.json','history.json','sampling_ledger.json.gz']]
    for p in extra:check();locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),protocol_SHA256=sha(RUN/'protocol.txt'),budget=B,old_results_unchanged=True));put(RUN/'reference_admission.json',dict(time=now(),inherited_evidence=str(A30/'reference_admission.json'),new_official_search=False,qualified_new_sources=0,SPH_admitted=False,RDB_scored=False,hospital_branch='waiting',expert_reference='waiting'))

def main():
    assert not (RUN/'started.json').exists(),'One A31 invocation';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+7200,pid=os.getpid(),max_seconds=7200));procs=[]
    try:
        freeze();tests=preflight();roles=read(A30/'patient_roles.json')
        for r in roles:
            groups=[set(r[k]) for k in ['train','design','calibration','outer']];assert list(map(len,groups))==[124,12,24,40] and all(not a&b for i,a in enumerate(groups) for b in groups[:i])
        assert sorted(p for r in roles for p in r['outer'])==list(range(200));put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests+['source_hashes','original_patient_roles'],new_fits=0,real_model_forwards=0))
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check();assert not any(p.poll() not in [None,0] for p,_ in procs),'Owned worker failed';done=[read(p) for p in (RUN/'jobs').glob('*/done.json')];trace_models=sum(v['trace']['models'] for v in done);signals=sum(v['trace']['lead_signals'] for v in done)
            assert len(done)<=30 and trace_models<=10 and signals<=120 and size()<=B['max_output_bytes']
            put(RUN/'status.json',dict(time=now(),state='running_zero_fit_alignment_audit',completed_cache_models=len(done),completed_policies=sum(v['policies'] for v in done),trace_models=trace_models,trace_lead_signals=signals,new_fits=0,new_epochs=0,SPH_admitted=False,new_figures=0,seconds_remaining=origin+7200-time.time()));time.sleep(min(5,max(.05,origin+7200-time.time())))
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==30;import finish;finish.main();verify();check();done=[read(p) for p in (RUN/'jobs').glob('*/done.json')];assert sum(v['cached_lead_records'] for v in done)==4560 and size()<=B['max_output_bytes']
        put(RUN/'completion.json',dict(time=now(),state='zero_fit_alignment_audit_complete_review_PASS',models_replayed=30,cached_lead_records=4560,policies_evaluated=120,frozen_trace_models=sum(v['trace']['models'] for v in done),trace_lead_signals=sum(v['trace']['lead_signals'] for v in done),new_fits=0,new_epochs=0,source_hashes_unchanged=True,wall_seconds=time.time()-origin,automatic_retraining=False,SPH_admitted=False,RDB_scored=False,new_figures=0,NC_submission_ready=False,hospital_branch='waiting',expert_reference='waiting'));put(RUN/'status.json',read(RUN/'completion.json'))
    except Exception:
        put(RUN/'stop.json',dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),no_retry=True));(RUN/'STOP').touch()
        for p,_ in procs:
            if p.poll() is None:p.terminate()
        put(RUN/'status.json',read(RUN/'stop.json'));raise
    finally:
        for p,log in procs:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            log.close()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,choices=[0,1]);ap.add_argument('--mock-preflight',action='store_true');args=ap.parse_args()
    if args.mock_preflight:print(json.dumps(dict(state='PASS',tests=preflight(),new_fits=0,real_model_forwards=0)))
    else:
        try:main() if args.worker is None else worker(args.worker)
        except Exception:
            if args.worker is not None:(RUN/'STOP').touch()
            raise
