"""A28 bounded inference-only mechanistic audit. No training or policy repair."""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,math,gzip,traceback,hashlib,subprocess
from pathlib import Path
import numpy as np
import torch
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A27=BASE/'runs/phase_A27_20261024_v1'
sys.path.insert(0,str(A27/'code'));import a27_method as m
import geometry as g
torch.set_num_threads(1);torch.set_grad_enabled(False)
B=json.loads((RUN/'budget.json').read_text());sha=m.sha;read=m.read;now=m.now
def clean(v):
    if isinstance(v,dict):return {str(k):clean(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return [clean(x) for x in v]
    if isinstance(v,(np.integer,)):return int(v)
    if isinstance(v,(np.bool_,)):return bool(v)
    if isinstance(v,(float,np.floating)):return float(v) if math.isfinite(v) else None
    return v
def put(path,v):m.put(path,clean(v))
def dumpgz(path,v):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,'wt',encoding='utf8') as f:json.dump(clean(v),f,ensure_ascii=False,allow_nan=False)
def check():
    assert not (RUN/'STOP_AUDIT').exists(),'Audit stopped by peer/coordinator'
    assert time.time()<read(RUN/'started.json')['deadline_unix'],'A28 original120min cap'
def verify():
    for p,h in read(RUN/'source_lock.json').items():check();assert sha(p)==h,f'Source changed:{p}'
def recover_anchors(prob,signal,anchors):
    p=prob[1].astype('float64');lod=(np.log(np.maximum(p,1e-12))-np.log(np.maximum(1-p,1e-12)))/5;cs=np.r_[0,np.cumsum(lod)];out=[]
    for r in anchors:
        if not 1000<r<4200:continue
        ss,ee=m.old.legal_pairs(r);scores=cs[ee+1]-cs[ss]-math.log(len(ss));i=int(scores.argmax());a,b=int(ss[i]),int(ee[i]);pk=a+int(np.argmax(abs(signal[a:b+1]-np.linspace(signal[a],signal[b],b-a+1))))
        if 1000<=pk<4000:out.append((int(r),a,b))
    return out
def infer(model,signals,ann,patients,anchors,role):
    records=[]
    with torch.inference_mode():
        for pat in patients:
            check();prob=m.old.fwd(model,torch.from_numpy(m.old.A9.channels(signals[pat,:,[1,6]].copy()))).softmax(1).numpy();assert np.isfinite(prob).all()
            for k,li in enumerate([1,6]):
                z=signals[pat,:,li];pp=m.old.raw_proposals(prob[k],z,anchors[pat]);rr=recover_anchors(prob[k],z,anchors[pat]);ordered=[p for p in pp if p['kind']=='ordered'];assert [(p['onset'],p['offset']) for p in ordered]==[(a,b) for r,a,b in rr],'Generating R-anchor geometry mismatch';it=iter(rr);qrs=ann[pat]['leads'][li]['qrs']
                for uid,p in enumerate(pp):
                    assert 0<=p['onset']<=p['peak']<=p['offset']<5000
                    p['uid']=uid;p['legacy_score']=p['confidence'];p['confidence']=min(p['confidence'],float(prob[k,1,p['onset']:p['offset']+1].mean())) if p['kind']=='ordered' else p['confidence'];p['scale']=max(5.,.5*max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2]));p['source_R']=next(it)[0] if p['kind']=='ordered' else None;p['expert_beat']=g.anchor_beat(p['source_R'],qrs) if p['kind']=='ordered' else g.next_beat(p['offset'],qrs)
                records.append(dict(patient=int(pat),li=li,lead='ii' if li==1 else 'v1',role=role,record=ann[pat]['record'],qrs=qrs,ref=m.old.references(ann,pat,li),raw=pp))
    return records
def point_stage(r,pred,signals,name):
    legacy=m.old.A9.match(r['ref'],pred);optimal=g.optimal(r['ref'],pred);inclusive=g.optimal(r['ref'],pred,True);assert len(optimal)>=len(legacy),'Optimal assignment invariant';matched={i for i,j,iou in optimal};by={}
    for kind in ['positive','negative','biphasic']:
        ids=[i for i,(a,pk,b) in enumerate(r['ref']) if m.old.A9.polarity(signals[r['patient'],:,r['li']],a,b)==kind];by[kind]=dict(reference=len(ids),matched=sum(i in matched for i in ids))
    errors=[dict(onset_ms=2*(pred[j]['onset']-r['ref'][i][0]),offset_ms=2*(pred[j]['offset']-r['ref'][i][2]),peak_ms=2*(pred[j]['peak']-r['ref'][i][1])) for i,j,overlap in optimal]
    return dict(stage=name,reference=len(r['ref']),predicted=len(pred),greedy_matched=len(legacy),optimal_matched=len(optimal),inclusive_matched=len(inclusive),polarity=by,matched_boundary_errors_ms=errors)
def localisation_failures(r):
    rows=[];raw=r['raw'];matched={i for i,j,overlap in g.optimal(r['ref'],raw)}
    for i,ref in enumerate(r['ref']):
        if i in matched:continue
        near=[p for p in raw if abs(p['peak']-ref[1])<=75 or min(p['offset'],ref[2])>=max(p['onset'],ref[0])];beat=g.next_beat(ref[2],r['qrs']);compatible=[p for p in near if beat is not None and p['expert_beat']==beat];wrong=[p for p in near if beat is not None and p['expert_beat'] is not None and p['expert_beat']!=beat]
        feasible=[p for p in raw if g.iou(ref,p)>=.5]
        if feasible:reason='one_to_one_assignment_conflict'
        elif not near:reason='no_local_raw_candidate'
        elif compatible:reason='same_beat_boundary_or_localisation_error'
        elif wrong and len(wrong)==len(near):reason='different_resolvable_beat_candidates'
        else:reason='near_candidate_identity_unresolved'
        best=min(near,key=lambda p:abs(p['peak']-ref[1])) if near else None
        rows.append(dict(reference_index=i,reference=list(ref),reason=reason,expert_beat=beat,near_candidates=len(near),wrong_beat_candidates=len(wrong),nearest=best))
    return rows
def record_audit(r,signals,labels,mode,ct,cal):
    raw=r['raw'];source=[p for p in raw if mode=='union' or p['kind']==mode];scored=[p for p in source if p['confidence']>=ct];selected,suppressed=g.nms_trace(scored);expected=m.old.nms(scored);assert [v['uid'] for v in selected]==[v['uid'] for v in expected],'NMS replay';q=float('inf') if cal['infinite'] else cal['q'];boxes=g.expanded(selected,q,True);reported=[p for p in boxes if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100]
    stages=[point_stage(r,pred,signals,name) for name,pred in [('raw',raw),('source_filter',source),('confidence_filter',scored),('NMS',selected),('width_rejection',reported)]];counts=[v['optimal_matched'] for v in stages];assert all(a>=b for a,b in zip(counts,counts[1:])),'Monotone maximum-cardinality waterfall'
    tie_selected,_=g.nms_trace(list(reversed(scored)));tie_change=len(g.optimal(r['ref'],tie_selected))-len(g.optimal(r['ref'],selected));byuid={p['uid']:p for p in scored};harm=[]
    for pair in suppressed:
        victim=byuid[pair['victim']];winner=byuid[pair['winner']]
        refs=[i for i,ref in enumerate(r['ref']) if g.iou(ref,victim)>=.5 and g.iou(ref,winner)<.5 and not any(g.iou(ref,p)>=.5 for p in selected)]
        if refs:harm.append(dict(**pair,reference_edges_lost=refs))
    union=[any(p['box'][0]<=ref[0]<=p['box'][1] and p['box'][2]<=ref[2]<=p['box'][3] for p in boxes) for ref in r['ref']];union_report=[any(p['box'][0]<=ref[0]<=p['box'][1] and p['box'][2]<=ref[2]<=p['box'][3] for p in reported) for ref in r['ref']];one=g.boxes_match(r['ref'],boxes);one_report=g.boxes_match(r['ref'],reported);assert len(one)<=sum(union) and len(one_report)<=sum(union_report)
    residuals=[]
    for i,ref in enumerate(r['ref']):
        score,distance,uid=g.residual(ref,selected,True);beat=g.next_beat(ref[2],r['qrs']);same,_,_=g.residual(ref,selected,True,True,beat);candidate=next((p for p in selected if p['uid']==uid),None)
        if beat is not None:assert same>=score,'Constrained residual invariant'
        residuals.append(dict(reference_index=i,reference=list(ref),polarity=m.old.A9.polarity(signals[r['patient'],:,r['li']],ref[0],ref[2]),normalised_residual=score,residual_infinite=not math.isfinite(score),raw_distance_samples=distance,raw_distance_ms=2*distance,candidate=candidate,reference_beat=beat,same_beat_residual=same if beat is not None else None,same_beat_residual_infinite=beat is not None and not math.isfinite(same),reference_fully_inside_admitted_mask=bool(np.all(labels[r['patient'],r['li'],ref[0]:ref[2]+1]!=-100))))
    if q>10:assert len(reported)==0,'Scale-floor forced-rejection arithmetic'
    return dict(patient=r['patient'],record=r['record'],lead=r['lead'],role=r['role'],stages=stages,raw_localisation_failures=localisation_failures(r),suppression_pairs=suppressed,harmful_suppression=harm,tie_reverse_match_change=tie_change,ordered_unmatched_R=sum(p['kind']=='ordered' and p['expert_beat'] is None for p in raw),legacy_mass_ge_half_common_below_half=sum(p['kind']=='ordered' and p['legacy_score']>=.5 and p['confidence']<.5 for p in raw),reference_count=len(r['ref']),union_covered=sum(union),one_to_one_box_covered=len(one),union_reported=sum(union_report),one_to_one_reported=len(one_report),normalised_residuals=residuals,forced_all_reject_from_q=q>10)
def extremes(records,audits,role,q):
    eligible=m.eligibility(records);by={}
    for r in audits:
        if not eligible[r['patient']]:continue
        for v in r['normalised_residuals']:by.setdefault(r['patient'],[]).append(dict(v,lead=r['lead'],record=r['record'],qrs_context=next(x['qrs'] for x in records if x['patient']==r['patient'] and x['lead']==r['lead'])))
    patients=[]
    for pat,events in by.items():
        worst=max(events,key=lambda v:float('inf') if v['residual_infinite'] else v['normalised_residual']);score=float('inf') if worst['residual_infinite'] else worst['normalised_residual'];patients.append(dict(patient=pat,role=role,max_residual=score,infinite=not math.isfinite(score),reference_waves=len(events),worst_event=worst))
    patients.sort(key=lambda v:v['max_residual'],reverse=True)
    if role=='calibration':
        computed=m.old.quantile([v['max_residual'] for v in patients]);assert math.isfinite(computed)==math.isfinite(q)
        if math.isfinite(q):assert abs(computed-q)<=1e-12,'Explicit extreme patient q replay'
    return patients[:3],dict(eligible_reference_patients=len(patients),top_score=patients[0]['max_residual'] if patients else None,second_score=patients[1]['max_residual'] if len(patients)>1 else None,k=math.ceil((len(patients)+1)*.95),k_is_maximum=bool(patients) and math.ceil((len(patients)+1)*.95)==len(patients))
def patient_box_summary(audits,role):
    out=[]
    for pat in sorted({r['patient'] for r in audits}):
        rows=[r for r in audits if r['patient']==pat];n=sum(r['reference_count'] for r in rows)
        if not n:continue
        union=all(r['union_covered']==r['reference_count'] for r in rows);unique=all(r['one_to_one_box_covered']==r['reference_count'] for r in rows);rep=all(r['union_reported']==r['reference_count'] for r in rows);repone=all(r['one_to_one_reported']==r['reference_count'] for r in rows)
        out.append(dict(patient=pat,role=role,reference=n,union_all=union,one_to_one_all=unique,union_reported_all=rep,one_to_one_reported_all=repone,union_pass_but_identity_assignment_fails=union and not unique))
    return out
def structured_first_epoch(j,d):
    if j['arm']!='C2_null_ordered':return None
    signals,ann,roles,ids,x,y,anchors,partial,targets,labels=d;r=roles[j['fold']];ix=np.array([i for i,(p,l) in enumerate(ids) if p in r['train']]);rng=np.random.default_rng(j['seed']+100*j['fold']);order=rng.permutation(ix);positions={k:dict(samples=0,with_structured_target=0,positive_targets=0,null_targets=0) for k in range(8)}
    for start in range(0,len(order),8):
        batch=order[start:start+8];all_targets=[(b,t) for b,i in enumerate(batch) for t in targets[i]];kept=all_targets[:24]
        for b,i in enumerate(batch):
            v=positions[b];v['samples']+=1;vv=[t for bb,t in kept if bb==b];v['with_structured_target']+=bool(vv);v['positive_targets']+=sum(t[1] is not None for t in vv);v['null_targets']+=sum(t[1] is None for t in vv)
    return dict(epoch=1,positions=positions,scope='structured-loss term only; frame supervision remains on all admitted samples',gradients=0,optimizer_steps=0)
def audit_job(j,d):
    check();p=RUN/'jobs'/f"job_{j['index']:02d}";assert not p.exists(),'No audit-job retry';p.mkdir(parents=True);put(p/'attempt.json',dict(job=j,time=now()));signals,ann,roles,ids,x,y,anchors,partial,targets,labels=d;oldpath=m.path(j);done=read(oldpath/'done.json');assert sha(oldpath/'final.pt')==done['checkpoint_SHA256'];model=m.old.A9.Net();model.load_state_dict(torch.load(oldpath/'final.pt',weights_only=True,map_location='cpu'));model.eval();r=roles[j['fold']];cal=infer(model,signals,ann,r['calibration'],anchors,'calibration');outer=infer(model,signals,ann,r['outer'],anchors,'outer');choice=read(oldpath/'design_selection.json')['choice'];cal_selected=m.select(cal,choice['mode'],choice['threshold']);outer_selected=m.select(outer,choice['mode'],choice['threshold']);replay=[];primary=None
    for norm in [False,True]:
        for ninety in [False,True]:
            check();policy=('normalised' if norm else 'absolute')+('_wave90_secondary' if ninety else '_ALL_primary');cc=m.calibrate(cal_selected,norm,ninety)
            with gzip.open(oldpath/(policy+'.json.gz'),'rt') as f:prior=json.load(f)
            prev=prior['calibration'];assert cc['infinite']==prev['infinite'] and cc['n_eligible']==prev['n_eligible'] and cc['k']==prev['k']
            if not cc['infinite']:assert abs(cc['q']-prev['q'])<=1e-12
            fresh=m.results(outer_selected,signals,labels,cc,j,policy);assert fresh['counts']==prior['counts'] and fresh['patients']==prior['patients'] and fresh['waves']==prior['waves'],'Frozen A27 count/coverage replay'
            replay.append(dict(policy=policy,state='PASS'))
            if norm and not ninety:primary=cc
    dumpgz(p/'candidate_cache.json.gz',dict(job=j,calibration=cal,outer=outer,mode=choice['mode'],threshold=choice['threshold']))
    audits={}
    for name,records in [('calibration',cal),('outer',outer)]:
        check();audits[name]=[record_audit(v,signals,labels,choice['mode'],choice['threshold'],primary) for v in records];dumpgz(p/f'{name}_diagnostics.json.gz',audits[name])
    q=float('inf') if primary['infinite'] else primary['q'];ext=[];influence={}
    for name,records in [('calibration',cal_selected),('outer',outer_selected)]:
        ee,ii=extremes(records,audits[name],name,q);ext+=ee;influence[name]=ii
    pairs=patient_box_summary(audits['calibration'],'calibration')+patient_box_summary(audits['outer'],'outer');put(p/'extreme_patients.json',ext);put(p/'boundary_target_identity.json',pairs);put(p/'calibration_influence.json',influence);put(p/'structured_target_first_epoch.json',structured_first_epoch(j,d))
    put(p/'done.json',dict(job=j,time=now(),model_loads=1,lead_signals=2*(len(r['calibration'])+len(r['outer'])),sign_orbit_passes=4*(len(r['calibration'])+len(r['outer'])),replays=replay,normalised_q=q,design_gate_pass=read(oldpath/'design_selection.json')['design_gate_pass'],new_fits=0,new_epochs=0))
def worker(w):
    verify();d=m.prepare()
    for j in m.jobs():
        if j['index']%2==w:audit_job(j,d);put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='running',last_job=j['index']))
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))
def source_review():
    a9=BASE/'runs/phase_A9_20261017_v1';isp=read(a9/'ISP_schema_audit.json');rdb=read(BASE/'runs/phase_A26_20261024_v1/scientific_admission.json');rangecheck=read(BASE/'runs/phase_A26_20261024_v1/range_audit/completion.json');partial=BASE/'runs/phase_A26_20261024_v1/source/ECGData.zip.partial'
    matrix=[dict(source='LUDB',strict_independent_admitted=False,scope='repeatedly viewed development',reasons=['A27 training source; not external','No independent clinical absence adjudication']),dict(source='RDB/Chapman',strict_independent_admitted=False,scope='potential future global positive-P only',observed_parent_raw_partial_bytes=partial.stat().st_size if partial.exists() else None,sample_numeric_matches=rangecheck['sample_count'],all_parent_waveforms_verified=False,patients='2398 parent groups,not certified2398 patients',reasons=['All2399 parent numeric identities not verified','Global reference not V1/noP gold','Duplicate waveform has discordant annotation versions']),dict(source='ISP',strict_independent_admitted=False,scope='sealed public candidate,no predictions',records=isp['records'],sampling_rates=isp['sampling_rates'],units=isp['units'],patient_ID_fields=isp['patient_ID_fields'],lead_specific_expert_boundaries=isp['lead_specific_expert_boundaries'],reasons=isp['reasons']),dict(source='QTDB',strict_independent_admitted=False,scope='previously viewed sparse positive-boundary engineering only',reasons=['Reference incompleteness/parent-patient grouping require endpoint audit','Not uniform V1 or clinical absence gold']),dict(source='BUT-PDB',strict_independent_admitted=False,scope='previously viewed P-peak engineering only',reasons=['Peak-only references cannot establish onset/offset boundaries','Parent/patient aliases and clinical absence unresolved'])]
    put(RUN/'reference_admission.json',dict(time=now(),review_completed=True,qualified_strict_independent_sources=0,new_external_predictions=0,new_downloads=0,SPH_admitted=False,component_training='A27 fresh LUDB-only P networks; not the older ECG-FM family',entries=matrix,decision='No source released merely because internal replay passes; new provenance/reference evidence required'))
def aggregate():
    check();records=[];ext=[];identities=[];supp=[]
    for j in m.jobs():
        check();p=RUN/'jobs'/f"job_{j['index']:02d}";assert read(p/'done.json')['new_fits']==0
        with gzip.open(p/'outer_diagnostics.json.gz','rt') as f:rr=json.load(f)
        for r in rr:
            for s in r['stages']:records.append(dict(arm=j['arm'],seed=j['seed'],fold=j['fold'],patient=r['patient'],lead=r['lead'],**s))
            supp.append(dict(arm=j['arm'],seed=j['seed'],fold=j['fold'],patient=r['patient'],lead=r['lead'],suppressed=len(r['suppression_pairs']),cross_source=sum(v['different_source'] for v in r['suppression_pairs']),harmful_suppression=len(r['harmful_suppression']),tie_reverse_match_change=r['tie_reverse_match_change'],unmatched_R=r['ordered_unmatched_R'],greedy_suboptimal_stages=sum(s['optimal_matched']>s['greedy_matched'] for s in r['stages']),localisation_reasons=[v['reason'] for v in r['raw_localisation_failures']]))
        ext+=[dict(arm=j['arm'],seed=j['seed'],fold=j['fold'],**r) for r in read(p/'extreme_patients.json')]
        identities+=[dict(arm=j['arm'],seed=j['seed'],fold=j['fold'],**r) for r in read(p/'boundary_target_identity.json') if r['role']=='outer']
    summary=[]
    for arm in m.ARMS:
        for seed in m.B['seeds']:
            for lead in ['ii','v1']:
                for stage in ['raw','source_filter','confidence_filter','NMS','width_rejection']:
                    rr=[r for r in records if r['arm']==arm and r['seed']==seed and r['lead']==lead and r['stage']==stage];nr=sum(v['reference'] for v in rr);nm=sum(v['optimal_matched'] for v in rr);n=sum(v['predicted'] for v in rr);errs=[e for v in rr for e in v['matched_boundary_errors_ms']]
                    summary.append(dict(arm=arm,seed=seed,lead=lead,stage=stage,reference=nr,predicted=n,optimal_matched=nm,greedy_matched=sum(v['greedy_matched'] for v in rr),inclusive_matched=sum(v['inclusive_matched'] for v in rr),recall=nm/max(nr,1),annotation_precision=nm/n if n else None,onset_MAE_matched_ms=float(np.mean([abs(e['onset_ms']) for e in errs])) if errs else None,offset_MAE_matched_ms=float(np.mean([abs(e['offset_ms']) for e in errs])) if errs else None,polarity={k:dict(reference=sum(v['polarity'][k]['reference'] for v in rr),matched=sum(v['polarity'][k]['matched'] for v in rr)) for k in ['positive','negative','biphasic']}))
    put(RUN/'analysis/stage_summary.json',summary);put(RUN/'analysis/suppression_assignment_summary.json',supp);put(RUN/'analysis/extreme_patients.json',ext);put(RUN/'analysis/union_vs_unique_patient_coverage.json',identities)
    c2=[v for v in identities if v['arm']=='C2_null_ordered'];reuse=sum(v['union_pass_but_identity_assignment_fails'] for v in c2);matching=sum(v['greedy_suboptimal_stages'] for v in supp);harm=sum(v['harmful_suppression'] for v in supp);raw={s:next(v for v in summary if v['arm']=='C2_null_ordered' and v['seed']==s and v['lead']=='v1' and v['stage']=='raw')['recall'] for s in m.B['seeds']}
    decision=dict(time=now(),state='audit_complete_not_effectiveness_pass',new_fits=0,new_epochs=0,raw_C2_V1_recall=raw,greedy_suboptimal_record_stages=matching,harmful_suppression_events=harm,patient_union_pass_but_unique_assignment_fails_C2_both_seeds=reuse,patient_counts_between_seeds_not_independent=True,retrain_automatically=False,recommendation='Repair measurement/assignment/uncertainty contract before considering any separately authorised training; do not blindly add epochs',causal_strength='Mechanistic accounting/counterfactual metric sensitivity,not randomised proof of a training cause',strict_independent_sources_admitted=0,clinical_noP_validated=False,NC_submission_ready=False,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting');put(RUN/'analysis/decision.json',decision)
def main():
    assert not (RUN/'started.json').exists(),'One A28 invocation only';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['max_seconds'],pid=os.getpid(),max_seconds=7200));tests=g.preflight();m.verify();locks=read(A27/'method_source_lock.json')
    extra=[RUN/'protocol.txt',RUN/'budget.json',Path(__file__).resolve(),RUN/'code/geometry.py',A27/'patient_roles.json',A27/'post_review/contract_review.json',BASE/'runs/phase_A9_20261017_v1/ISP_schema_audit.json',BASE/'runs/phase_A9_20261017_v1/checkpoint_exposure_audit.json',BASE/'runs/phase_A26_20261024_v1/scientific_admission.json']
    for j in m.jobs():extra+=[m.path(j)/'final.pt',m.path(j)/'done.json',m.path(j)/'design_selection.json']+list(m.path(j).glob('*.json.gz'))
    for p in extra:check();locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),budget=B,original_A27_unchanged=True));put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests,real_model_forwards=0,new_fits=0));source_review();procs=[]
    try:
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check();assert not any(p.poll() not in [None,0] for p,_ in procs),'Worker failure';done=list((RUN/'jobs').glob('*/done.json'));used=sum(read(p)['lead_signals'] for p in done);assert len(done)<=30 and used<=3840;size=sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file());assert size<=B['max_output_bytes'];put(RUN/'status.json',dict(time=now(),state='running_zero_fit_audit',completed_models=len(done),max_models=30,completed_lead_signals=used,new_fits=0,new_epochs=0,AF_fits=0,SPH_admitted=False,new_figures=0,seconds_remaining=max(0,origin+7200-time.time())));time.sleep(min(10,max(.1,origin+7200-time.time())))
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==30;aggregate();verify();check();put(RUN/'completion.json',dict(time=now(),state='zero_fit_audit_complete',models=30,lead_signals=3840,sign_orbit_passes=7680,policies_replayed=120,new_fits=0,new_epochs=0,source_hashes_unchanged=True,wall_seconds=time.time()-origin,decision='analysis/decision.json',no_automatic_retraining=True,NC_submission_ready=False,SPH_admitted=False,new_figures=0));put(RUN/'status.json',read(RUN/'completion.json'))
    except Exception:
        (RUN/'STOP_AUDIT').write_text(traceback.format_exc())
        for p,_ in procs:
            if p.poll() is None:p.terminate()
        put(RUN/'stop.json',dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),no_retry=True));raise
    finally:
        for p,log in procs:
            if p.poll() is None:p.wait(timeout=30)
            log.close()
if __name__=='__main__':
    import argparse
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,choices=[0,1]);args=ap.parse_args()
    try:main() if args.worker is None else worker(args.worker)
    except Exception:
        if args.worker is not None:(RUN/'STOP_AUDIT').write_text(traceback.format_exc())
        elif not (RUN/'stop.json').exists():put(RUN/'stop.json',dict(time=now(),state='PREFLIGHT_FAIL_STOP',traceback=traceback.format_exc(),no_retry=True))
        raise
