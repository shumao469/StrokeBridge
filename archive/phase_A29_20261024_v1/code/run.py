"""One bounded A29 invocation. Frozen forwards and deterministic decoding only."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,time,json,gzip,math,traceback,subprocess
from pathlib import Path
import numpy as np
import torch
import decoder as d
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A27=BASE/'runs/phase_A27_20261024_v1';A28=BASE/'runs/phase_A28_20261024_v1'
sys.path.insert(0,str(A27/'code'));import a27_method as m
torch.set_num_threads(1);torch.set_grad_enabled(False)
B=json.loads((RUN/'budget.json').read_text());read=m.read;sha=m.sha;now=m.now

def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(tuple,list)):return [clean(x) for x in v]
    if isinstance(v,np.integer):return int(v)
    if isinstance(v,np.bool_):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if math.isfinite(v) else None
    return v

def put(p,v):m.put(p,clean(v))

def gz(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(p,'wt',encoding='utf8') as f:json.dump(clean(v),f,ensure_ascii=False,allow_nan=False)

def ungz(p):
    with gzip.open(p,'rt',encoding='utf8') as f:return json.load(f)

def check():
    assert not (RUN/'STOP').exists(),'Owned worker/coordinator requested stop'
    assert time.time()<read(RUN/'started.json')['deadline_unix'],'A29 fixed7200s deadline'

def verify():
    for p,h in read(RUN/'source_lock.json').items():check();assert sha(p)==h,f'Frozen source changed:{p}'

def output_size():
    total=0
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        try:total+=p.stat().st_size
        except FileNotFoundError:
            if p.suffix!='.tmp':raise
    return total

def capture_old(prob,z,anchors,qrs):
    pp=m.old.raw_proposals(prob,z,anchors);pr=prob[1].astype('float64');lod=(np.log(np.maximum(pr,1e-12))-np.log(np.maximum(1-pr,1e-12)))/5;cs=np.r_[0,np.cumsum(lod)];rs=[]
    for r in anchors:
        if not 1000<r<4200:continue
        ss,ee=d.legal_pairs(r,len(pr));sc=cs[ee+1]-cs[ss]-math.log(len(ss));i=int(sc.argmax());a,b=int(ss[i]),int(ee[i]);pk=d.peak(z,a,b)
        if 1000<=pk<4000:rs.append((int(r),a,b))
    assert [(p['onset'],p['offset']) for p in pp if p['kind']=='ordered']==[(a,b) for r,a,b in rs]
    rit=iter(rs)
    for uid,p in enumerate(pp):
        p['uid']=uid;p['legacy_score']=p['confidence'];p['confidence']=min(p['confidence'],float(prob[1,p['onset']:p['offset']+1].mean())) if p['kind']=='ordered' else p['confidence'];p['scale']=max(5.,.5*max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2]));p['source_R']=next(rit)[0] if p['kind']=='ordered' else None;p['expert_beat']=d.map_expert(p['source_R'],qrs) if p['kind']=='ordered' else d.next_expert(p['offset'],qrs)
    return pp

def calibrate(records,legacy_q=None):
    ledger=[]
    for pat in sorted({r['patient'] for r in records}):
        rr=[r for r in records if r['patient']==pat];nr=sum(len(r['ref']) for r in rr)
        if not nr:continue
        rows=[dict(lead=r['lead'],record=r['record'],reference_count=len(r['ref']),**d.scores(r['ref'],r['selected'],r['qrs'])) for r in rr if r['ref']]
        ledger.append(dict(patient=pat,reference=nr,missing=sum(r['missing'] for r in rows),strict=max(r['strict'] for r in rows),component=max(r['component'] for r in rows),rows=rows))
    strict=d.quantile([r['strict'] for r in ledger]);component=d.quantile([r['component'] for r in ledger]);used=legacy_q if legacy_q is not None else strict
    return dict(strict_q=strict,component_q=component,used_q=used,used_legacy_q=legacy_q is not None,n_reference_positive=len(ledger),k=math.ceil((len(ledger)+1)*.95),missing_patient_count=sum(r['missing']>0 for r in ledger),infinite_strict_patient_count=sum(not math.isfinite(r['strict']) for r in ledger),infinite_component_patient_count=sum(not math.isfinite(r['component']) for r in ledger),all_patients_retained=True,component_is_diagnostic_not_primary=True,ledger=ledger)

def metrics(ref,pp,qrs,signal,labels,patient,li):
    pairs=d.point_pairs(ref,pp,qrs);ipairs=d.point_pairs(ref,pp,qrs,True);mi={i for i,j in pairs};mj={j for i,j in pairs};unknown=sum(j not in mj and np.any(labels[patient,li,p['onset']:p['offset']+1]==-100) for j,p in enumerate(pp))
    return dict(reference=len(ref),predicted=len(pp),matched=len(pairs),identity_matched=len(ipairs),unknown_unmatched=unknown,polarity={kind:dict(reference=sum(m.old.A9.polarity(signal,a,b)==kind for a,pk,b in ref),matched=sum(i in mi and m.old.A9.polarity(signal,a,b)==kind for i,(a,pk,b) in enumerate(ref))) for kind in ['positive','negative','biphasic']})

def evaluate(records,cal,signals,labels):
    rec=[];widths=[];local_widths=[]
    for r in records:
        pp=r['selected'];expanded=d.expand(pp,cal['used_q']);reported=[p for p in expanded if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100]
        component=d.expand(pp,cal['component_q']);creported=[p for p in component if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100]
        nr=len(r['ref']);before=len(d.box_pairs(r['ref'],expanded,r['qrs']));after=len(d.box_pairs(r['ref'],reported,r['qrs']));compbefore=len(d.box_pairs(r['ref'],component,r['qrs']));compafter=len(d.box_pairs(r['ref'],creported,r['qrs']))
        for ps,target in [(reported,widths),(creported,local_widths)]:
            target.extend(dict(patient=r['patient'],lead=r['lead'],onset_ms=2*(p['box'][1]-p['box'][0]),offset_ms=2*(p['box'][3]-p['box'][2])) for p in ps)
        score=d.scores(r['ref'],pp,r['qrs']);s=signals[r['patient'],:,r['li']]
        tables={name:metrics(r['ref'],values,r['qrs'],s,labels,r['patient'],r['li']) for name,values in [('raw',r['raw']),('selected',pp),('reported',reported),('component_reported_DIAGNOSTIC',creported)]}
        pol=[m.old.A9.polarity(s,a,b) for a,pk,b in r['ref']];pairids={i for i,j in d.point_pairs(r['ref'],reported,r['qrs'])}
        rec.append(dict(patient=r['patient'],lead=r['lead'],record=r['record'],reference=nr,reference_pol=[dict(kind=k,matched=i in pairids) for i,k in enumerate(pol)],tables=tables,identity_score=score,strict_unique_covered=before,strict_unique_reported=after,component_unique_covered_DIAGNOSTIC=compbefore,component_unique_reported_DIAGNOSTIC=compafter,merger=r['merger'],selected=pp))
    patients=[]
    for p in sorted({r['patient'] for r in rec}):
        rr=[r for r in rec if r['patient']==p];nr=sum(r['reference'] for r in rr)
        if not nr:continue
        patients.append(dict(patient=p,reference=nr,missing=sum(r['identity_score']['missing'] for r in rr),strict_all=all(r['strict_unique_covered']==r['reference'] for r in rr),strict_reported_all=all(r['strict_unique_reported']==r['reference'] for r in rr),component_all_DIAGNOSTIC=all(r['component_unique_covered_DIAGNOSTIC']==r['reference'] for r in rr),component_reported_all_DIAGNOSTIC=all(r['component_unique_reported_DIAGNOSTIC']==r['reference'] for r in rr)))
    return dict(calibration=cal,records=rec,patients=patients,widths=widths,component_widths_DIAGNOSTIC=local_widths)

def replay(j,cache,signals,labels):
    path=m.path(j);cal=m.select(cache['calibration'],cache['mode'],cache['threshold']);outer=m.select(cache['outer'],cache['mode'],cache['threshold']);replayed=[];primary=None
    for norm in [False,True]:
        for wave90 in [False,True]:
            check();policy=('normalised' if norm else 'absolute')+('_wave90_secondary' if wave90 else '_ALL_primary');fresh=m.calibrate(cal,norm,wave90);old=ungz(path/(policy+'.json.gz'));oc=old['calibration'];assert fresh['infinite']==oc['infinite'] and fresh['n_eligible']==oc['n_eligible'] and fresh['k']==oc['k']
            if not fresh['infinite']:assert abs(fresh['q']-oc['q'])<=1e-12
            rr=m.results(outer,signals,labels,fresh,j,policy);assert rr['counts']==old['counts'] and rr['waves']==old['waves'] and rr['patients']==old['patients'],'Original policy replay'
            replayed.append(policy)
            if norm and not wave90:primary=float('inf') if fresh['infinite'] else fresh['q']
    return replayed,primary

def job(j,data):
    check();out=RUN/'jobs'/f"job_{j['index']:02d}";assert not out.exists(),'No job retry';out.mkdir(parents=True);put(out/'attempt.json',dict(time=now(),job=j))
    signals,ann,roles,ids,x,y,anchors,partial,targets,labels=data;cache=ungz(A28/'jobs'/f"job_{j['index']:02d}"/'candidate_cache.json.gz');assert cache['job']==j
    assert sha(m.path(j)/'final.pt')==read(m.path(j)/'done.json')['checkpoint_SHA256'];model=m.old.A9.Net();model.load_state_dict(torch.load(m.path(j)/'final.pt',map_location='cpu',weights_only=True));model.eval();variants={name:dict(calibration=[],outer=[]) for name in B['policies']};lead_signals=0
    for role in ['calibration','outer']:
        grouped={}
        for r in cache[role]:grouped.setdefault(r['patient'],[]).append(r)
        assert sorted(grouped)==sorted(roles[j['fold']][role])
        for pat,rows in grouped.items():
            check()
            with torch.inference_mode():pr=m.old.fwd(model,torch.from_numpy(m.old.A9.channels(signals[pat,:,[1,6]].copy()))).softmax(1).numpy()
            assert pr.shape==(2,4,5000) and np.isfinite(pr).all() and np.all((pr>=0)&(pr<=1))
            for k,li in enumerate([1,6]):
                r=next(v for v in rows if v['li']==li);oldraw=capture_old(pr[k],signals[pat,:,li],anchors[pat],r['qrs']);assert clean(oldraw)==r['raw'],'Exact raw confidence/geometry/kernel replay';lead_signals+=1
                newraw=d.posterior_decode(pr[k],signals[pat,:,li],anchors[pat]);legacy=[p for p in oldraw if (cache['mode']=='union' or p['kind']==cache['mode']) and p['confidence']>=cache['threshold']];legacy=d.nms(legacy)
                for policy in B['policies']:
                    raw=newraw if policy=='B3_boundary' else oldraw
                    accepted=[p for p in raw if (cache['mode']=='union' or p['kind']==cache['mode']) and p['confidence']>=cache['threshold']]
                    if policy in ['L0_legacy','I1_identity']:sel=legacy;merger=dict(policy='original_global_NMS',unresolved=[])
                    else:sel,merger=d.merge(accepted,anchors[pat],support_rank=policy=='B3_boundary')
                    assert all(p['confidence']>=cache['threshold'] for p in sel),'Confidence gate changed'
                    rr=dict(r,raw=d.decorate(raw,anchors[pat],r['qrs']),selected=d.decorate(sel,anchors[pat],r['qrs']),merger=merger)
                    variants[policy][role].append(rr)
    assert lead_signals==128
    original,legacy_q=replay(j,cache,signals,labels);gz(out/'candidates.json.gz',variants)
    for policy,rr in variants.items():
        check();cal=calibrate(rr['calibration'],legacy_q if policy=='L0_legacy' else None);result=evaluate(rr['outer'],cal,signals,labels);gz(out/(policy+'.json.gz'),dict(job=j,policy=policy,source_policy=cache['mode'],confidence_threshold=cache['threshold'],**result))
    put(out/'done.json',dict(time=now(),job=j,checkpoint_SHA256=sha(m.path(j)/'final.pt'),models_loaded=1,lead_signals=lead_signals,sign_orbit_passes=2*lead_signals,original_policies_replayed=original,decoder_policies=len(B['policies']),new_fits=0,new_epochs=0))

def worker(w):
    verify();data=m.prepare()
    for j in m.jobs():
        if j['index']%2==w:
            job(j,data);put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='running',last_job=j['index']))
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))

def aggregate():
    check();summary=[];extreme=[]
    for arm in m.ARMS:
        for seed in m.B['seeds']:
            for policy in B['policies']:
                outs=[ungz(RUN/'jobs'/f"job_{j['index']:02d}"/(policy+'.json.gz')) for j in m.jobs() if j['arm']==arm and j['seed']==seed];patients=[p for v in outs for p in v['patients']];assert len({p['patient'] for p in patients})==len(patients)
                for lead in ['ii','v1']:
                    rec=[r for v in outs for r in v['records'] if r['lead']==lead];w=[x[k] for v in outs for x in v['widths'] if x['lead']==lead for k in ['onset_ms','offset_ms']];cw=[x[k] for v in outs for x in v['component_widths_DIAGNOSTIC'] if x['lead']==lead for k in ['onset_ms','offset_ms']]
                    def stage(name):
                        rows=[r['tables'][name] for r in rec];nr=sum(r['reference'] for r in rows);nm=sum(r['matched'] for r in rows);npred=sum(r['predicted'] for r in rows);unknown=sum(r['unknown_unmatched'] for r in rows)
                        return dict(reference=nr,matched=nm,predicted=npred,identity_matched=sum(r['identity_matched'] for r in rows),recall=nm/nr if nr else None,precision_lower=nm/npred if npred else None,precision_upper=(nm+unknown)/npred if npred else None,polarity={kind:dict(reference=sum(r['polarity'][kind]['reference'] for r in rows),matched=sum(r['polarity'][kind]['matched'] for r in rows),recall=sum(r['polarity'][kind]['matched'] for r in rows)/max(1,sum(r['polarity'][kind]['reference'] for r in rows)),patients=len({r['patient'] for r in rec if any(v['kind']==kind for v in r['reference_pol'])})) for kind in ['positive','negative','biphasic']})
                    summary.append(dict(arm=arm,seed=seed,policy=policy,lead=lead,stages={name:stage(name) for name in ['raw','selected','reported','component_reported_DIAGNOSTIC']},reference_positive_patients=len(patients),missing_identity_waves=sum(r['identity_score']['missing'] for r in rec),strict_patient_joint_before=sum(p['strict_all'] for p in patients)/len(patients),strict_patient_joint_reported=sum(p['strict_reported_all'] for p in patients)/len(patients),component_patient_joint_before_DIAGNOSTIC=sum(p['component_all_DIAGNOSTIC'] for p in patients)/len(patients),component_patient_joint_reported_DIAGNOSTIC=sum(p['component_reported_all_DIAGNOSTIC'] for p in patients)/len(patients),median_width_ms=float(np.median(w)) if w else None,p95_width_ms=float(np.quantile(w,.95)) if w else None,component_median_width_ms_DIAGNOSTIC=float(np.median(cw)) if cw else None,strict_q_finite_folds=sum(v['calibration']['strict_q'] is not None for v in outs),original_design_feasible_folds=sum(read(m.path(j)/'design_selection.json')['design_gate_pass'] for j in m.jobs() if j['arm']==arm and j['seed']==seed),calibration_patients_retained=[v['calibration']['n_reference_positive'] for v in outs]))
                for v in outs:
                    fold=v['job']['fold'];cal=v['calibration'];top=sorted(cal['ledger'],key=lambda p:float('inf') if p['strict'] is None else p['strict'],reverse=True)[:3]
                    extreme.extend(dict(arm=arm,seed=seed,policy=policy,fold=fold,**p) for p in top)
    put(RUN/'analysis/summary.json',summary);put(RUN/'analysis/extreme_patients.json',extreme);gates=[];fair=[]
    get=lambda a,s,p,l:next(v for v in summary if v['arm']==a and v['seed']==s and v['policy']==p and v['lead']==l)
    for policy in B['policies']:
        for seed in m.B['seeds']:
            checks={}
            for lead in ['ii','v1']:
                a=get(m.ARMS[0],seed,policy,lead);b=get(m.ARMS[2],seed,policy,lead);am=a['stages']['reported'];bm=b['stages']['reported'];legacy=get(m.ARMS[2],seed,'L0_legacy',lead)['stages']['reported']
                checks[lead+'_strict_useful']=b['strict_patient_joint_before']>=.95 and b['strict_patient_joint_reported']>=.95 and bm['recall']>=.8 and b['median_width_ms'] is not None and b['median_width_ms']<=120 and b['p95_width_ms']<=200 and b['original_design_feasible_folds']==5
                checks[lead+'_noninferior_C0']=am['precision_lower'] is not None and bm['precision_lower'] is not None and bm['recall']>=am['recall']-.02 and bm['precision_lower']>=am['precision_lower']-.02
                fair.append(dict(seed=seed,policy=policy,lead=lead,comparison='C2 vs C0',both_recall_at_least80=am['recall']>=.8 and bm['recall']>=.8,matched_coverage=am['recall']>=.8 and bm['recall']>=.8 and abs(am['recall']-bm['recall'])<=.02,advantage_claim=False))
                fair.append(dict(seed=seed,policy=policy,lead=lead,comparison='C2 vs C2 L0',both_recall_at_least80=legacy['recall']>=.8 and bm['recall']>=.8,matched_coverage=legacy['recall']>=.8 and bm['recall']>=.8 and abs(legacy['recall']-bm['recall'])<=.02,advantage_claim=False))
            a=get(m.ARMS[0],seed,policy,'v1')['stages']['reported'];b=get(m.ARMS[2],seed,policy,'v1')['stages']['reported']
            for kind in ['negative','biphasic']:checks[kind+'_gain5pp']=a['predicted']>0 and b['predicted']>0 and b['polarity'][kind]['patients']>=10 and b['polarity'][kind]['recall']>=a['polarity'][kind]['recall']+.05
            gates.append(dict(policy=policy,seed=seed,checks=checks,pass_all=all(checks.values())))
    put(RUN/'analysis/gates.json',dict(entries=gates,any_policy_both_seed_pass=any(all(v['pass_all'] for v in gates if v['policy']==p) for p in B['policies']),no_posthoc_policy_promotion=True,original_ALL_primary_preserved=True));put(RUN/'analysis/matched_coverage.json',fair)
    put(RUN/'analysis/decision.json',dict(time=now(),state='zero_fit_decoder_repair_complete',automatic_retraining=False,new_fits=0,new_epochs=0,identity_borrowing_forbidden=True,candidate_reuse_forbidden=True,extreme_patients_retained=True,independent_source_admission=0,NC_submission_ready=False,clinical_noP_validated=False,AF_morphology_increment=False,brain_stroke_risk_validated=False,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting',next_step='Review all four fixed policies. A separate training/independent-reference protocol is required; no automatic fit or gate relaxation.'))

def freeze():
    assert read(A28/'completion.json')['source_hashes_unchanged'];locks=read(A28/'source_lock.json')
    extra=[RUN/'protocol.txt',RUN/'budget.json',Path(__file__).resolve(),RUN/'code/decoder.py',A28/'source_lock.json',A28/'completion.json',A28/'reference_admission.json',A28/'analysis/decision.json']
    for j in m.jobs():extra+=[A28/'jobs'/f"job_{j['index']:02d}"/'candidate_cache.json.gz',A28/'jobs'/f"job_{j['index']:02d}"/'done.json']
    for p in extra:check();locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),protocol_SHA256=sha(RUN/'protocol.txt'),budget=B,old_results_unchanged=True));put(RUN/'reference_admission.json',dict(time=now(),inherited_from=str(A28/'reference_admission.json'),evidence_date=read(A28/'reference_admission.json')['time'],new_source_search=False,new_downloads=0,new_admissions=0,SPH_admitted=False,RDB_scored=False,hospital_branch='waiting',expert_reference='waiting'))

def main():
    assert not (RUN/'started.json').exists(),'One A29 invocation only';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['max_seconds'],pid=os.getpid(),max_seconds=B['max_seconds']));procs=[]
    try:
        freeze();tests=d.preflight();roles=read(A27/'patient_roles.json')
        for r in roles:
            groups=[set(r[k]) for k in ['train','design','calibration','outer']];assert list(map(len,groups))==[124,12,24,40] and all(not a&b for i,a in enumerate(groups) for b in groups[:i])
        assert sorted(p for r in roles for p in r['outer'])==list(range(200));put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests+['frozen_source_hashes','disjoint_roles','outer_partition'],real_model_forwards=0,new_fits=0))
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check();assert not any(p.poll() not in [None,0] for p,_ in procs),'Owned worker failure';done=list((RUN/'jobs').glob('*/done.json'));rows=[read(p) for p in done];assert len(rows)<=30 and sum(p['lead_signals'] for p in rows)<=3840;assert output_size()<=B['max_output_bytes'],'Output2GiB cap'
            put(RUN/'status.json',dict(time=now(),state='running_zero_fit_decoder_repair',completed_models=len(rows),completed_policies=sum(p['decoder_policies'] for p in rows),new_fits=0,new_epochs=0,seconds_remaining=origin+7200-time.time(),SPH_admitted=False,new_figures=0));time.sleep(min(5,max(.05,origin+7200-time.time())))
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==30,'Incomplete execution';aggregate();verify();check();assert output_size()<=B['max_output_bytes'];put(RUN/'completion.json',dict(time=now(),state='zero_fit_decoder_repair_complete',models=30,lead_signals=3840,sign_orbit_passes=7680,original_policies_replayed=120,decoder_policies=120,new_fits=0,new_epochs=0,source_hashes_unchanged=True,wall_seconds=time.time()-origin,automatic_retraining=False,NC_submission_ready=False,SPH_admitted=False,new_figures=0));put(RUN/'status.json',read(RUN/'completion.json'))
    except Exception:
        put(RUN/'stop.json',dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),no_retry=True));(RUN/'STOP').touch()
        for p,_ in procs:
            if p.poll() is None:p.terminate()
        raise
    finally:
        for p,log in procs:
            if p.poll() is None:
                try:p.wait(timeout=10)
                except subprocess.TimeoutExpired:p.kill();p.wait(timeout=5)
            log.close()

if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,choices=[0,1]);ap.add_argument('--mock-preflight',action='store_true');args=ap.parse_args()
    if args.mock_preflight:print(json.dumps(dict(state='PASS',tests=d.preflight(),model_forwards=0,new_fits=0)))
    else:
        try:main() if args.worker is None else worker(args.worker)
        except Exception:
            if args.worker is not None:(RUN/'STOP').touch()
            raise
