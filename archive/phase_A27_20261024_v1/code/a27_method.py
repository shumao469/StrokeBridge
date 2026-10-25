"""Reference-aware partial supervision; fixed-design patient set calibration.
No clinical absence labels, AF fits, pretrained encoder use, or outer-set selection.
"""
import os
for key in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[key]='1'
import sys,json,time,math,traceback,hashlib,subprocess,gzip
from pathlib import Path
import numpy as np,pandas as pd,torch
from torch import nn
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];OLD=BASE/'runs/phase_A26_20261024_v1'
sys.path.insert(0,str(OLD/'code'));import a26_method as old
torch.set_num_threads(1);B=json.loads((RUN/'budget.json').read_text());ARMS=B['arms']
put=old.put;read=old.read;sha=old.sha;now=old.now
def jobs():return [dict(index=i,arm=a,seed=s,fold=f) for i,(f,s,a) in enumerate((f,s,a) for f in B['folds'] for s in B['seeds'] for a in ARMS)]
def path(j):return RUN/'models'/j['arm']/str(j['seed'])/str(j['fold'])
def check(stage):
    assert not (RUN/'STOP_NEW_WORK').exists(),'Peer failure'
    origin=read(RUN/'attribution/started.json')['unix_time'];assert time.time()-origin<B['total_seconds'],'A27 overall6h deadline'
    if stage=='train':assert time.time()-read(RUN/'coordinator_started.json')['unix_time']<B['training_seconds'],'Training4h cap'
def verify():
    for p,h in read(RUN/'method_source_lock.json').items():assert sha(p)==h,f'Source changed:{p}'
def roles(original):
    out=[]
    for r in original:
        design=sorted(r['train'],key=lambda p:hashlib.sha256(f"A27-design:{r['fold']}:{p}".encode()).hexdigest())[:12]
        out.append(dict(fold=r['fold'],train=[p for p in r['train'] if p not in design],design=design,calibration=r['selection']+r['calibration'],outer=r['outer']))
    return out
def prepare():
    signals,ann,rr,ids,x,y,anchors,positive=old.data();rs=roles(rr);partial=np.zeros(y.shape,bool);targets=[]
    for i,(p,l) in enumerate(ids):
        for a,pk,b in old.references(ann,p,l):
            for edge in [a,b]:partial[i,max(0,edge-2):min(5000,edge+3)]=True
        partial[i]&=(y[i]==0)|(y[i]==1)
        null=[]
        for r in anchors[p]:
            lo=max(0,int(r)-150);hi=max(0,int(r)-10)
            if lo<hi and 1000<=lo<hi<=4000 and np.all(y[i,lo:hi]!=-100) and not np.any(y[i,lo:hi]==1):null.append((int(r),None,None))
        targets.append(positive[i][:3]+null[:3])
    full_labels=np.load(old.INPUT/'data.npz')['labels']
    return signals,ann,rs,ids,x,y,anchors,partial,targets,full_labels
def frame_loss(logits,y,w,amb):
    valid=y.ne(-100);lp=logits.log_softmax(1);exact=-lp.gather(1,y.clamp_min(0)[:,None]).squeeze(1);raw=torch.where(amb,-torch.logsumexp(lp[:,:2],1),exact);ww=w*valid
    ce=(raw*ww).sum()/ww.sum().clamp_min(1);known=valid&~amb;p=lp[:,1].exp()*known;g=y.eq(1)&known;dice=1-(2*(p*g).sum(1)+1)/(p.sum(1)+g.sum(1)+1);use=known.any(1)
    return ce+(dice[use].mean() if use.any() else logits.sum()*0)
def interval_loss(logits,targets):
    lp=logits.log_softmax(1);lod=(lp[:,1]-torch.logsumexp(lp[:,[0,2,3]],1))/5;loss=[]
    for batch,(r,a,b) in targets:
        ss,ee=old.legal_pairs(r);s=torch.as_tensor(ss);e=torch.as_tensor(ee);cs=nn.functional.pad(lod[batch].cumsum(0),(1,0));score=cs[e+1]-cs[s]-math.log(len(ss));z=torch.logsumexp(torch.cat([score,score.new_zeros(1)]),0)
        if a is None:loss.append(z)
        else:
            good=(abs(ss-a)<=2)&(abs(ee-b)<=2)
            if good.any():loss.append(z-torch.logsumexp(score[torch.as_tensor(good)],0))
    return torch.stack(loss).mean() if loss else logits.sum()*0
def infer(model,signals,ann,patients,anchors):
    out=[]
    with torch.inference_mode():
        for p in patients:
            check('audit');pr=old.fwd(model,torch.from_numpy(old.A9.channels(signals[p,:,[1,6]].copy()))).softmax(1).numpy()
            for k,l in enumerate([1,6]):
                pp=old.raw_proposals(pr[k],signals[p,:,l],anchors[p])
                for v in pp:
                    v['legacy_score']=v['confidence'];v['confidence']=min(v['confidence'],float(pr[k,1,v['onset']:v['offset']+1].mean())) if v['kind']=='ordered' else v['confidence'];v['scale']=max(5.,.5*max(v['box'][1]-v['box'][0],v['box'][3]-v['box'][2]))
                out.append(dict(patient=int(p),li=l,lead='ii' if l==1 else 'v1',ref=old.references(ann,p,l),raw=pp))
    return out
def select(records,mode,ct):
    return [dict(r,selected=old.nms([v for v in r['raw'] if (mode=='union' or mode==v['kind']) and v['confidence']>=ct])) for r in records]
def detection(records):
    out={}
    for lead in ['ii','v1']:
        rr=[r for r in records if r['lead']==lead];n=sum(len(r['selected']) for r in rr);nr=sum(len(r['ref']) for r in rr);nm=sum(len(old.A9.match(r['ref'],r['selected'])) for r in rr);out[lead]=dict(reference=nr,predicted=n,matched=nm,recall=nm/nr if nr else 0,precision=nm/n if n else 0)
    return out
def design_choice(records):
    choices=[]
    for mode in ['continuous','ordered','union']:
        vals=[v['confidence'] for r in records for v in r['raw'] if mode=='union' or mode==v['kind']];grid=sorted(set([0.,1.]+([float(v) for v in np.quantile(vals,np.linspace(0,1,63))] if vals else [])))
        for ct in grid:
            scores=detection(select(records,mode,ct));nr=sum(v['reference'] for v in scores.values());nm=sum(v['matched'] for v in scores.values());n=sum(v['predicted'] for v in scores.values());choices.append(dict(mode=mode,threshold=ct,precision=nm/max(n,1),recall=nm/max(nr,1),both_leads_feasible=all(v['recall']>=.8 for v in scores.values()),by_lead=scores))
    ok=[v for v in choices if v['both_leads_feasible']];choice=max(ok,key=lambda v:(v['precision'],v['threshold'],v['mode'])) if ok else next(v for v in choices if v['mode']=='union' and v['threshold']==0)
    return dict(choice=choice,design_gate_pass=bool(ok),selected_on='design12 only; no calibration or outer selection',tested_options=len(choices))
def residual(ref,proposals,normalised):
    a,pk,b=ref
    return min((max(v['box'][0]-a,a-v['box'][1],v['box'][2]-b,b-v['box'][3],0)/(v['scale'] if normalised else 1) for v in proposals),default=float('inf'))
def eligibility(records):
    return {p:all(any(r['selected'] for r in records if r['patient']==p and r['lead']==lead) for lead in ['ii','v1']) for p in sorted({r['patient'] for r in records})}
def calibrate(records,normalised,wave90=False):
    eligible=eligibility(records);by={}
    for r in records:
        if not eligible[r['patient']]:continue
        for ref in r['ref']:by.setdefault(r['patient'],[]).append(residual(ref,r['selected'],normalised))
    values=[sorted(v)[math.ceil(.9*len(v))-1] if wave90 else max(v) for p,v in sorted(by.items())];q=old.quantile(values)
    return dict(q=old.safe_num(q),infinite=not np.isfinite(q),n_eligible=len(values),k=math.ceil((len(values)+1)*.95),calibration_eligible_patients=[p for p in by],normalised=normalised,target='patient_at_least90pct_waves' if wave90 else 'patient_ALL_waves',alpha=.05,formal_selection_after_width_rejection_claim=False)
def results(records,signals,labels,cal,j,policy):
    q=float('inf') if cal['infinite'] else cal['q'];normalised=cal['normalised'];waves=[];counts=[];widths=[];pat=[];eligible=eligibility(records)
    for r in records:
        allset=[];reported=[]
        for v in r['selected']:
            scale=v['scale'] if normalised else 1.;a,b,c,d=v['box'];box=[a-q*scale,b+q*scale,c-q*scale,d+q*scale];z=dict(v,box=box)
            allset.append(z)
            if np.isfinite(q) and max(box[1]-box[0],box[3]-box[2])<=100:reported.append(z);widths.append(dict(patient=r['patient'],lead=r['lead'],onset_ms=2*(box[1]-box[0]),offset_ms=2*(box[3]-box[2])))
        pairs=old.A9.match(r['ref'],reported);mi={i for i,k,iou in pairs};mj={k for i,k,iou in pairs};u=0
        for k,v in enumerate(reported):
            if k not in mj and (labels[r['patient'],r['li'],v['onset']:v['offset']+1]==-100).any():u+=1
        counts.append(dict(patient=r['patient'],lead=r['lead'],reference=len(r['ref']),predicted=len(reported),matched=len(pairs),preselected=len(r['selected']),unknown_unmatched=u,known_unmatched=len(reported)-len(pairs)-u,eligible=eligible[r['patient']],calibration_usable=not cal['infinite']))
        for i,ref in enumerate(r['ref']):
            a,pk,b=ref;rawcov=bool(eligible[r['patient']] and np.isfinite(q) and any(v['box'][0]<=a<=v['box'][1] and v['box'][2]<=b<=v['box'][3] for v in allset));repcov=bool(any(v['box'][0]<=a<=v['box'][1] and v['box'][2]<=b<=v['box'][3] for v in reported))
            waves.append(dict(patient=r['patient'],lead=r['lead'],peak=pk,polarity=old.A9.polarity(signals[r['patient'],:,r['li']],a,b),matched=i in mi,raw_covered=rawcov,reported_covered=repcov))
    for p in sorted({r['patient'] for r in records}):
        w=[v for v in waves if v['patient']==p]
        if w:pat.append(dict(patient=p,eligible=eligible[p],reference=len(w),raw_all=all(v['raw_covered'] for v in w),reported_all=all(v['reported_covered'] for v in w),raw90=sum(v['raw_covered'] for v in w)>=math.ceil(.9*len(w)),reported90=sum(v['reported_covered'] for v in w)>=math.ceil(.9*len(w))))
    return dict(job=j,policy=policy,calibration=cal,counts=counts,waves=waves,patients=pat,widths=widths)
def train_job(j,d):
    check('train');p=path(j);assert not p.exists(),'No job retry/overwrite';p.mkdir(parents=True);put(p/'attempt.json',dict(**j,time=now()));signals,ann,roles,ids,x,y,anchors,partial,targets,labels=d;r=roles[j['fold']];ixtrain=np.array([i for i,(pat,l) in enumerate(ids) if pat in r['train']]);seed=j['seed']+100*j['fold'];rng=np.random.default_rng(seed);torch.manual_seed(seed);model=old.A9.Net();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    weights,_,meta=old.A9.weight_maps(signals,ann,x,y,np.array([v[0] for v in ids]),np.array([v[1] for v in ids]),ixtrain);put(p/'training_contract.json',dict(roles=r,weights=meta,partial_background_is_clinical_noP=False,fresh_initialisation=True));hist=[];start=time.monotonic()
    for epoch in range(12):
        check('train');model.train();order=rng.permutation(ixtrain);losses=[]
        for begin in range(0,len(order),8):
            check('train');ix=order[begin:begin+8];xx=torch.from_numpy(old.A9.augment(x[ix],rng));yy=torch.from_numpy(y[ix]);amb=torch.from_numpy(partial[ix]) if j['arm']!=ARMS[0] else torch.zeros_like(yy,dtype=torch.bool);lg=old.fwd(model,xx);loss=frame_loss(lg,yy,torch.from_numpy(weights[ix]),amb)
            if j['arm']==ARMS[2]:loss=loss+.1*interval_loss(lg,[(b,t) for b,i in enumerate(ix) for t in targets[i]][:24])
            assert torch.isfinite(loss);opt.zero_grad(set_to_none=True);loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(gn);opt.step();losses.append(float(loss.detach()))
        hist.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),wall_seconds=time.monotonic()-start));put(p/'history.json',hist);put(RUN/f"worker_{j['index']%2}.json",dict(time=now(),pid=os.getpid(),state='training',job=j,epoch=epoch+1))
    torch.save(model.state_dict(),p/'final.pt');put(p/'training_done.json',dict(time=now(),unix_time=time.time(),epochs=12));model.eval();design=infer(model,signals,ann,r['design'],anchors);choice=design_choice(design);put(p/'design_selection.json',choice);mode=choice['choice']['mode'];ct=choice['choice']['threshold'];cr=select(infer(model,signals,ann,r['calibration'],anchors),mode,ct);outer=select(infer(model,signals,ann,r['outer'],anchors),mode,ct)
    for norm in [False,True]:
        for ninety in [False,True]:
            policy=('normalised' if norm else 'absolute')+('_wave90_secondary' if ninety else '_ALL_primary');cal=calibrate(cr,norm,ninety);out=results(outer,signals,labels,cal,j,policy)
            with gzip.open(p/(policy+'.json.gz'),'wt',encoding='utf8') as f:json.dump(out,f,allow_nan=False)
    put(p/'done.json',dict(**j,time=now(),epochs=12,wall_seconds=time.monotonic()-start,checkpoint_SHA256=sha(p/'final.pt'),design_gate_pass=choice['design_gate_pass'],AF_fits=0,clinical_admission=False))
def preflight():
    assert not (RUN/'preflight.json').exists();t=time.monotonic();d=prepare();signals,ann,rr,ids,x,y,anchors,partial,targets,labels=d
    assert x.shape==(391,2,5000) and not np.any(partial&(y==-100)) and not np.any(partial&np.isin(y,[2,3]))
    for r in rr:
        groups=[set(r[k]) for k in ['train','design','calibration','outer']];assert list(map(len,groups))==[124,12,24,40] and all(not a&b for i,a in enumerate(groups) for b in groups[:i])
    assert sorted(p for r in rr for p in r['outer'])==list(range(200));put(RUN/'patient_roles.json',rr)
    lg=torch.zeros(1,4,128,requires_grad=True);yy=torch.zeros(1,128,dtype=torch.long);yy[:,:10]=-100;val=frame_loss(lg,yy,torch.ones_like(yy,dtype=torch.float32),torch.zeros_like(yy,dtype=torch.bool));val.backward();assert lg.grad[:,:,:10].abs().max()==0 and lg.grad[:,1,30]>0,'Known background must penalise P'
    lg2=torch.zeros(1,4,5000,requires_grad=True);v=interval_loss(lg2,[(0,(1600,None,None))]);v.backward();assert torch.isfinite(v) and lg2.grad[0,1,1500]>0,'Reference-null interval should suppress P'
    assert math.isinf(old.quantile(list(range(12)))) and old.quantile(list(range(24)))==23
    assert residual((10,20,30),[dict(box=[8,12,28,32],scale=5)],True)==0
    torch.manual_seed(29);model=old.A9.Net();xx=torch.from_numpy(x[:8]);yy=torch.from_numpy(y[:8]);ww=torch.ones_like(yy,dtype=torch.float32);amb=torch.from_numpy(partial[:8]);times=[]
    for _ in range(3):
        tt=time.monotonic();model.zero_grad(set_to_none=True);lg=old.fwd(model,xx);v=frame_loss(lg,yy,ww,amb)+.1*interval_loss(lg,[(b,t) for b in range(8) for t in targets[b]][:24]);v.backward();times.append(time.monotonic()-tt)
    estimate=max(times)*math.ceil(248/8)*360/2*1.5+1200;assert estimate<B['training_seconds'],'CPU timing gate'
    assert sha(RUN/'protocol.txt')==read(RUN/'attribution/started.json')['protocol_SHA256'],'Original N0 protocol must not be rewritten'
    old.verify_lock();locks=read(OLD/'source_lock.json');locks.update({str(p):sha(p) for p in [RUN/'protocol.txt',RUN/'method_addendum.txt',RUN/'budget.json',Path(__file__).resolve(),RUN/'code/a27_attribution.py',RUN/'attribution/completion.json',RUN/'attribution/label_provenance.json',BASE/'runs/phase_A7b_20261017_v1/a7b_annotation_audit.py',BASE/'runs/phase_A7b_20261017_v1/annotation_audit.json']});put(RUN/'method_source_lock.json',locks);put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=['patient_role_disjointness','unknown_gradient_zero','admitted_background_P_negative_constraint','hard_QRS_T_not_ambiguous','null_interval_gradient','finite_patient_quantile','normalised_geometry','CPU_backward_timing'],optimizer_steps=0,backward_seconds=times,conservative_seconds=estimate,wall_seconds=time.monotonic()-t));return d
def aggregate():
    verify();rows=[]
    for arm in ARMS:
        for seed in B['seeds']:
            paths=[path(j) for j in jobs() if j['arm']==arm and j['seed']==seed]
            for policy in ['absolute_ALL_primary','normalised_ALL_primary','absolute_wave90_secondary','normalised_wave90_secondary']:
                outs=[]
                for p in paths:
                    with gzip.open(p/(policy+'.json.gz'),'rt') as f:outs.append(json.load(f))
                pat=[r for v in outs for r in v['patients']];usable=all(not v['calibration']['infinite'] for v in outs);choice_ok=all(read(p/'done.json')['design_gate_pass'] for p in paths)
                for lead in ['ii','v1']:
                    counts=[r for v in outs for r in v['counts'] if r['lead']==lead];wave=[r for v in outs for r in v['waves'] if r['lead']==lead];width=[w[k] for v in outs for w in v['widths'] if w['lead']==lead for k in ['onset_ms','offset_ms']];n=sum(r['predicted'] for r in counts);nr=sum(r['reference'] for r in counts);nm=sum(r['matched'] for r in counts);u=sum(r['unknown_unmatched'] for r in counts)
                    row=dict(arm=arm,seed=seed,policy=policy,lead=lead,reference=nr,predicted=n,matched=nm,recall=nm/max(nr,1),precision_lower=nm/n if n else None,precision_upper=(nm+u)/n if n else None,patient_joint_raw=sum(v['raw_all'] for v in pat)/len(pat),patient_joint_reported=sum(v['reported_all'] for v in pat)/len(pat),patient90_raw=sum(v['raw90'] for v in pat)/len(pat),patient90_reported=sum(v['reported90'] for v in pat)/len(pat),reference_positive_patients=len(pat),eligible_patients=sum(v['eligible'] for v in pat),median_width_ms=float(np.median(width)) if width else None,p95_width_ms=float(np.quantile(width,.95)) if width else None,all_folds_calibration_finite=usable,all_folds_design_feasible=choice_ok)
                    for kind in ['negative','biphasic']:
                        vv=[v for v in wave if v['polarity']==kind];row[kind+'_recall']=sum(v['matched'] for v in vv)/len(vv) if vv else None;row[kind+'_events']=len(vv);row[kind+'_patients']=len({v['patient'] for v in vv})
                    rows.append(row)
    put(RUN/'analysis/summary.json',rows);gates=[]
    for seed in B['seeds']:
        get=lambda a,l:next(v for v in rows if v['arm']==a and v['seed']==seed and v['lead']==l and v['policy']=='normalised_ALL_primary');checks={}
        for lead in ['ii','v1']:
            a,b=get(ARMS[0],lead),get(ARMS[2],lead);checks[lead+'_useful_strict_coverage']=b['all_folds_calibration_finite'] and b['all_folds_design_feasible'] and b['patient_joint_raw']>=.95 and b['patient_joint_reported']>=.95 and b['recall']>=.8 and b['median_width_ms'] is not None and b['median_width_ms']<=120 and b['p95_width_ms']<=200
            checks[lead+'_matched_coverage']=a['recall']>=.8 and b['recall']>=.8 and abs(a['recall']-b['recall'])<=.02
            checks[lead+'_noninferior']=a['precision_lower'] is not None and b['precision_lower'] is not None and a['predicted']>0 and b['predicted']>0 and b['recall']>=a['recall']-.02 and b['precision_lower']>=a['precision_lower']-.02
        a,b=get(ARMS[0],'v1'),get(ARMS[2],'v1')
        for k in ['negative','biphasic']:checks[k+'_gain5pp']=a[k+'_recall'] is not None and b[k+'_recall'] is not None and a['predicted']>0 and b['predicted']>0 and b[k+'_patients']>=10 and b[k+'_recall']>=a[k+'_recall']+.05
        gates.append(dict(seed=seed,checks=checks,pass_all=all(checks.values())))
    put(RUN/'analysis/gates.json',dict(seeds=gates,measurement_gate_pass=all(v['pass_all'] for v in gates),strict_primary_not_replaced_by_wave90=True,clinical_noP_validated=False,AF_morphology_increment=False,NC_submission_ready=False,SPH_admitted=False,RDB_scored=False,new_figures=0))
def worker(w):
    verify();d=prepare()
    for j in jobs():
        if j['index']%2==w:train_job(j,d)
    put(RUN/f'worker_{w}.json',dict(time=now(),state='finished',pid=os.getpid()))
def main():
    assert not (RUN/'coordinator_started.json').exists(),'One bounded invocation';origin=read(RUN/'attribution/started.json')['unix_time']
    while not (RUN/'attribution/completion.json').exists():
        assert time.time()-origin<B['zero_fit_seconds'] and not (RUN/'attribution/stop.json').exists(),'N0 failure or45min cap';time.sleep(10)
    assert read(RUN/'attribution/completion.json')['training_gate_pass'],'N0 scientific gate not passed';preflight();verify();put(RUN/'method_protocol_frozen.json',dict(time=now(),code_lock_SHA256=sha(RUN/'method_source_lock.json'),budget=B,zero_fit_gate='PASS',scientific_scope='development only'))
    put(RUN/'coordinator_started.json',dict(time=now(),unix_time=time.time(),pid=os.getpid(),overall_origin_unix=origin,max_jobs=30,max_epochs=360));procs=[]
    try:
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check('audit');assert not any(p.poll() not in [None,0] for p,_ in procs),'Worker failure';attempts=list((RUN/'models').glob('*/*/*/attempt.json'));hist=list((RUN/'models').glob('*/*/*/history.json'));epochs=sum(len(read(p)) for p in hist);done=list((RUN/'models').glob('*/*/*/done.json'));assert len(attempts)<=30 and epochs<=360
            training_done=list((RUN/'models').glob('*/*/*/training_done.json'))
            if len(training_done)==30:assert time.time()-max(read(p)['unix_time'] for p in training_done)<B['analysis_seconds'],'Post-training30min cap'
            put(RUN/'status.json',dict(time=now(),state='running',stage='patient_isolated_method',completed_jobs=len(done),attempts=len(attempts),epochs=epochs,max_jobs=30,max_epochs=360,AF_fits=0,FM_forwards=0,SPH_admitted=False,RDB_scored=False,new_figures=0));time.sleep(10)
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'models').glob('*/*/*/done.json')))==30;aggregate();verify()
        for j in jobs():assert read(path(j)/'done.json')['checkpoint_SHA256']==sha(path(j)/'final.pt')
        put(RUN/'completion.json',dict(time=now(),state='bounded_complete_internal_contract_checked',jobs=30,epochs=360,source_and_checkpoint_SHA='PASS',AF_fits=0,FM_forwards=0,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting',effectiveness_gate='analysis/gates.json',independent_statistical_review='not yet completed'))
        put(RUN/'status.json',dict(time=now(),state='bounded_complete_internal_contract_checked',completed_jobs=30,epochs=360,effectiveness_gate='analysis/gates.json',AF_fits=0,SPH_admitted=False,RDB_scored=False,new_figures=0))
    except Exception:
        (RUN/'STOP_NEW_WORK').write_text(traceback.format_exc())
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
        if args.worker is not None:(RUN/'STOP_NEW_WORK').write_text(traceback.format_exc())
        elif not (RUN/'stop.json').exists():put(RUN/'stop.json',dict(time=now(),state='PRE_TRAIN_FAIL_STOP',traceback=traceback.format_exc(),no_retry=True))
        raise
