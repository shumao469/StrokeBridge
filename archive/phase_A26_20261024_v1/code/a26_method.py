"""A26: partial-reference training and patient-max set coverage. No clinical no-P gold."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']: os.environ[k]='1'
import sys,json,time,math,hashlib,traceback,subprocess,importlib.util,argparse
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from scipy.signal import find_peaks
from scipy.ndimage import uniform_filter1d
RUN=Path(__file__).resolve().parents[1]; BASE=RUN.parents[1]
LEGACY=BASE/'runs/phase_A9_20261017_v1/code/a9_method.py'
spec=importlib.util.spec_from_file_location('a9_frozen',LEGACY); A9=importlib.util.module_from_spec(spec);spec.loader.exec_module(A9)
torch.set_num_threads(1); torch.backends.cudnn.allow_tf32=False;torch.backends.cuda.matmul.allow_tf32=False
INPUT=Path('/home/shu/codex_runs/phase_A9_20261017_v1/input')
B=json.loads((RUN/'budget.json').read_text()); ARMS=B['arms']
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def read(p):return json.loads(Path(p).read_text())
def put(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False));tmp.replace(p)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def fwd(model,x):return (model(x)+model(-x))/2
def frame_loss(logits,y,w,partial):
    valid=y.ne(-100);safe=y.clamp_min(0);lp=logits.log_softmax(1)
    exact=-lp.gather(1,safe[:,None]).squeeze(1)
    if partial:exact=torch.where(y.eq(0),-torch.logsumexp(lp[:,:2],dim=1),exact)
    ww=w*valid;ce=(exact*ww).sum()/ww.sum().clamp_min(1)
    known=valid & (y.ne(0) if partial else torch.ones_like(valid));p=lp[:,1].exp()*known;g=y.eq(1)*known
    dice=1-(2*(p*g).sum(1)+1)/(p.sum(1)+g.sum(1)+1)
    use=known.any(1);return ce+(dice[use].mean() if use.any() else p.sum()*0)
def legal_pairs(r,length=5000):
    a=np.arange(max(0,int(r)-150),max(0,int(r)-10));s,e=np.meshgrid(a,a,indexing='ij');ok=(e-s>=10)&(e-s<=100)&(e<length)
    return s[ok].astype(np.int64),e[ok].astype(np.int64)
def anchors(signal):
    energy=uniform_filter1d(np.mean(np.gradient(signal[:,[1,6]],axis=0)**2,axis=1),size=25)
    med=np.median(energy); mad=np.median(abs(energy-med));peaks,_=find_peaks(energy,distance=125,prominence=max(3*mad,1e-7))
    out=[]
    for p in peaks:
        a=max(0,p-20);b=min(len(signal),p+21);local=np.max(abs(signal[a:b][:,[1,6]]),axis=1);out.append(a+int(local.argmax()))
    return np.asarray(sorted(set(out)),dtype=int)
def interval_loss(logits,targets):
    lp=logits.log_softmax(1);lod=(lp[:,1]-torch.logsumexp(lp[:,[0,2,3]],1))/5
    losses=[]
    for batch,(r,a,b) in targets:
        ss,ee=legal_pairs(r,logits.shape[-1]);good=(abs(ss-a)<=2)&(abs(ee-b)<=2)
        if not good.any():continue
        s=torch.as_tensor(ss);e=torch.as_tensor(ee);cs=nn.functional.pad(lod[batch].cumsum(0),(1,0));score=cs[e+1]-cs[s]-math.log(len(s))
        losses.append(torch.logsumexp(torch.cat([score,score.new_zeros(1)]),0)-torch.logsumexp(score[torch.as_tensor(good)],0))
    return torch.stack(losses).mean() if losses else logits.sum()*0
def data():
    z=np.load(INPUT/'data.npz');signals=z['signals'];labels=z['labels'];ann=read(INPUT/'pwave_annotations.json');roles=read(INPUT/'patient_roles.json')
    ids=[(p,l) for p in range(200) for l in [1,6] if (labels[p,l]!=-100).any()]
    x=A9.channels(np.stack([signals[p,:,l] for p,l in ids]));y=np.stack([labels[p,l] for p,l in ids]).astype('int64');ra=[anchors(v) for v in signals]
    target=[]
    for p,l in ids:
        t=[]
        for a,pk,b in ann[p]['leads'][l]['p']:
            if not (1000<=pk<4000 and np.all(labels[p,l,a:b+1]!=-100)):continue
            rs=[r for r in ra[p] if r-150<=a<=b<=r-11 and 10<=b-a<=100]
            if rs:t.append((int(min(rs,key=lambda r:abs(r-b-40))),a,b))
        target.append(t)
    return signals,ann,roles,ids,x,y,ra,target
def jobs():return [dict(index=i,arm=a,seed=s,fold=f) for i,(f,s,a) in enumerate((f,s,a) for f in B['folds'] for s in B['seeds'] for a in ARMS)]
def jp(j):return RUN/'models'/j['arm']/str(j['seed'])/str(j['fold'])
def references(ann,p,l):return [tuple(v) for v in ann[p]['leads'][l]['p'] if 1000<=v[1]<4000]
def raw_proposals(prob,z,rpeaks):
    out=[];p=prob[1].astype('float64');lod=(np.log(np.maximum(p,1e-12))-np.log(np.maximum(1-p,1e-12)))/5;cs=np.r_[0,np.cumsum(lod)]
    for r in rpeaks:
        if not 1000<r<4200:continue
        ss,ee=legal_pairs(r);score=cs[ee+1]-cs[ss]-math.log(len(ss));mx=max(0,float(score.max()));w=np.exp(score-mx);null=math.exp(-mx);tot=w.sum()+null;pn=float(w.sum()/tot);w/=w.sum();i=int(score.argmax())
        def bounds(v):
            marginal=np.bincount(v,weights=w,minlength=5000);cdf=np.cumsum(marginal)
            return [int(np.searchsorted(cdf,.0125)),int(min(4999,np.searchsorted(cdf,.9875)))]
        a,b=int(ss[i]),int(ee[i]);pk=a+int(np.argmax(abs(z[a:b+1]-np.linspace(z[a],z[b],b-a+1))))
        if 1000<=pk<4000:out.append(dict(onset=a,offset=b,peak=pk,confidence=pn,box=bounds(ss)+bounds(ee),kind='ordered'))
    mask=p>=.5;diff=np.diff(np.r_[False,mask,False].astype(int))
    for a,b in zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)):
        b-=1
        if 10<=b-a<=100:
            pk=a+int(np.argmax(abs(z[a:b+1]-np.linspace(z[a],z[b],b-a+1))))
            if 1000<=pk<4000:out.append(dict(onset=int(a),offset=int(b),peak=int(pk),confidence=float(p[a:b+1].mean()),box=[int(a),int(a),int(b),int(b)],kind='continuous'))
    return out
def nms(proposals):
    out=[]
    for p in sorted(proposals,key=lambda x:x['confidence'],reverse=True):
        if all(max(0,min(p['offset'],v['offset'])-max(p['onset'],v['onset']))/max(max(p['offset'],v['offset'])-min(p['onset'],v['onset']),1)<.5 for v in out):out.append(p)
    return out
def residual(ref,proposals):
    a,pk,b=ref
    return min((max(v['box'][0]-a,a-v['box'][1],v['box'][2]-b,b-v['box'][3],0) for v in proposals),default=float('inf'))
def quantile(scores):
    n=len(scores);k=math.ceil((n+1)*.95)
    return float(sorted(scores)[k-1]) if n and k<=n else float('inf')
def safe_num(v):return float(v) if np.isfinite(v) else None
def accepted(proposals,q=0):
    if not np.isfinite(q):return []
    return nms([v for v in proposals if v['confidence']>=.5 and max(v['box'][1]-v['box'][0]+2*q,v['box'][3]-v['box'][2]+2*q)<=100])
def infer(model,signals,ann,patients,rpeaks):
    out=[]
    with torch.inference_mode():
        for p in patients:
            xx=torch.from_numpy(A9.channels(signals[p,:,[1,6]].copy()));probs=fwd(model,xx).softmax(1).numpy()
            for j,l in enumerate([1,6]):out.append(dict(patient=int(p),li=l,lead='ii' if l==1 else 'v1',ref=references(ann,p,l),raw=raw_proposals(probs[j],signals[p,:,l],rpeaks[p])))
    return out
def patient_scores(records):
    by={}
    for r in records:
        for ref in r['ref']:by.setdefault(r['patient'],[]).append(residual(ref,r['raw']))
    return [max(v) for k,v in sorted(by.items())],by
def count_tables(records,signals,q,job):
    counts=[];waves=[];patient=[];widths=[]
    for r in records:
        pred=accepted(r['raw'],q);pairs=A9.match(r['ref'],pred);matched={i for i,j,k in pairs};raw_rs=[];reported=[]
        for i,ref in enumerate(r['ref']):
            a,pk,b=ref;rr=residual(ref,r['raw']);ar=residual(ref,pred);raw_rs.append(rr);reported.append(ar)
            waves.append(dict(**job,patient=r['patient'],lead=r['lead'],peak=pk,polarity=A9.polarity(signals[r['patient'],:,r['li']],a,b),matched=i in matched,raw_nominal_covered=rr==0,patient_calibrated_set_covered=bool(np.isfinite(rr) and rr<=q),reported_set_covered=bool(ar<=q and np.isfinite(q)),raw_residual_samples=safe_num(rr)))
        nr=len(r['ref']);nm=len(pairs);npred=len(pred)
        counts.append(dict(**job,patient=r['patient'],lead=r['lead'],reference=nr,predicted=npred,matched=nm,raw_proposals=len(r['raw']),false_outputs=npred-nm,rejected_raw_proposals=len(r['raw'])-len(pred)))
        for v in pred:widths.append(dict(**job,patient=r['patient'],lead=r['lead'],onset_width_ms=2*(v['box'][1]-v['box'][0]+2*q),offset_width_ms=2*(v['box'][3]-v['box'][2]+2*q)))
    for p in sorted({r['patient'] for r in records}):
        w=[v for v in waves if v['patient']==p]
        if w:patient.append(dict(**job,patient=p,n_reference=len(w),raw_nominal_joint=all(v['raw_nominal_covered'] for v in w),patient_calibrated_joint=all(v['patient_calibrated_set_covered'] for v in w),reported_joint=all(v['reported_set_covered'] for v in w)))
    return counts,waves,patient,widths
def stopcheck(stage):
    assert not (RUN/'STOP_NEW_WORK').exists(),'Peer failure stops this worker'
    start=read(RUN/'coordinator_started.json')['unix_time'];cap=B['training_seconds'] if stage=='train' else B['total_seconds'];assert time.time()-start<cap,'Wall-clock deadline reached'
def train(j,d):
    signals,ann,roles,ids,x,y,ra,targets=d;path=jp(j);assert not path.exists(),'No automatic retry or overwrite';path.mkdir(parents=True)
    put(path/'attempt.json',dict(**j,time=now()));role=roles[j['fold']];tr=np.array([i for i,(p,l) in enumerate(ids) if p in role['train']]);cal=role['selection']+role['calibration'];seed=j['seed']+100*j['fold'];rng=np.random.default_rng(seed);torch.manual_seed(seed)
    model=A9.Net();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    patients=np.array([p for p,l in ids]);leads=np.array([l for p,l in ids]);weights,_,wm=A9.weight_maps(signals,ann,x,y,patients,leads,tr);put(path/'train_weights.json',wm);history=[];t=time.monotonic()
    for epoch in range(B['epochs_per_job']):
        stopcheck('train');model.train();losses=[];order=rng.permutation(tr)
        for first in range(0,len(order),8):
            stopcheck('train');ix=order[first:first+8];xx=torch.from_numpy(A9.augment(x[ix],rng));yy=torch.from_numpy(y[ix]);ww=torch.from_numpy(weights[ix]);logits=fwd(model,xx)
            loss=frame_loss(logits,yy,ww,j['arm']!=ARMS[0])
            if j['arm']==ARMS[2]:loss=loss+.1*interval_loss(logits,[(b,t) for b,i in enumerate(ix) for t in targets[i]][:24])
            assert torch.isfinite(loss),'Nonfinite loss';opt.zero_grad(set_to_none=True);loss.backward();norm=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(norm),'Nonfinite gradient';opt.step();losses.append(float(loss.detach()))
        history.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),elapsed_seconds=time.monotonic()-t));pd.DataFrame(history).to_csv(path/'history.csv',index=False)
        put(RUN/f"worker_{j['index']%2}.json",dict(time=now(),pid=os.getpid(),state='training',job=j,epoch=epoch+1))
    torch.save(model.state_dict(),path/'final.pt');model.eval();stopcheck('audit');cr=infer(model,signals,ann,cal,ra);scores,by=patient_scores(cr);q=quantile(scores);pooled=quantile([v for vv in by.values() for v in vv]);outer=infer(model,signals,ann,role['outer'],ra)
    put(path/'calibration.json',dict(calibration_patients=cal,n_eligible=len(scores),k=math.ceil((len(scores)+1)*.95),patient_q_samples=safe_num(q),patient_q_infinite=not np.isfinite(q),wave_diagnostic_q_samples=safe_num(pooled),patient_scores=[dict(patient=p,score=safe_num(max(v)),infinite=not np.isfinite(max(v))) for p,v in sorted(by.items())],no_conditional_coverage_claim=True))
    for policy,qq in [('nominal',0),('wave_diagnostic',pooled),('patient',q)]:
        for name,rows in zip(['counts','waves','patients','widths'],count_tables(outer,signals,qq,j)):pd.DataFrame(rows).to_csv(path/f'{policy}_{name}.csv',index=False)
    put(path/'done.json',dict(**j,time=now(),epochs=12,wall_seconds=time.monotonic()-t,checkpoint_SHA256=sha(path/'final.pt'),clinical_admission=False))
def preflight():
    assert not (RUN/'preflight.json').exists(),'One preflight only';t=time.monotonic();torch.manual_seed(73)
    logits=torch.randn(2,4,64,requires_grad=True);y=torch.zeros(2,64,dtype=torch.long);y[:,:10]=-100;y[:,20:30]=1;y[:,40:50]=2;w=torch.ones(2,64)
    v=frame_loss(logits,y,w,True);v.backward();assert logits.grad[:,:,:10].abs().max()==0
    # Adding mass within {background,P} at an ambiguous position cannot increase its set loss.
    x1=torch.tensor([1.,0.,-2.,-3.]);x2=x1.clone();x2[1]+=1
    assert -torch.logsumexp(x2.log_softmax(0)[:2],0)<-torch.logsumexp(x1.log_softmax(0)[:2],0)
    toy=torch.randn(1,4,5000,requires_grad=True);iv=interval_loss(toy,[(0,(1600,1490,1530))]);iv.backward();assert torch.isfinite(iv) and toy.grad.abs().sum()>0
    assert quantile(list(range(24)))==23 and math.isinf(quantile(list(range(12))))
    ref=(20,30,40);pp=[dict(box=[18,22,38,42])];assert residual(ref,pp)==0 and math.isinf(residual(ref,[]))
    d=data();signals,ann,roles,ids,x,y,ra,targets=d
    assert x.shape==(391,2,5000) and np.isfinite(x).all()  # Nine entirely unlabelled lead/patient pairs are not fabricated as negatives.
    for role in roles:
        sets=[set(role[k]) for k in ['train','selection','calibration','outer']];assert [len(s) for s in sets]==[136,12,12,40] and all(not a&b for i,a in enumerate(sets) for b in sets[:i])
    outer=[p for role in roles for p in role['outer']];assert sorted(outer)==list(range(200))
    model=A9.Net();xx=torch.from_numpy(x[:8]);yy=torch.from_numpy(y[:8]);ww=torch.ones_like(yy,dtype=torch.float32);model.eval()
    with torch.inference_mode():assert torch.allclose(fwd(model,xx),fwd(model,-xx),atol=1e-7,rtol=0)
    timings=[]
    for _ in range(3):
        a=time.monotonic();model.zero_grad(set_to_none=True);lg=fwd(model,xx);ll=frame_loss(lg,yy,ww,True)+.1*interval_loss(lg,[(b,t) for b in range(8) for t in targets[b]][:24]);ll.backward();timings.append(time.monotonic()-a)
    steps=math.ceil(272/8)*12*30;estimate=max(timings)*steps/2*1.5+900
    assert estimate<B['training_seconds'],f'Timing gate failed:{estimate}'
    locks=[LEGACY,RUN/'protocol.txt',RUN/'budget.json',Path(__file__).resolve(),INPUT/'data.npz',INPUT/'pwave_annotations.json',INPUT/'patient_roles.json',RUN/'metadata/completion.json',RUN/'range_audit/completion.json',RUN/'reference_audit/completion.json',RUN/'metadata/Diagnostics_v2.xlsx',BASE/'runs/phase_A25_20261023_v1/data/dataset.zip',BASE/'runs/phase_A25_20261023_v1/data/rdb_chapman_mapping.xlsx']
    lock={str(p):sha(p) for p in locks};put(RUN/'source_lock.json',lock)
    put(RUN/'preflight.json',dict(state='PASS',time=now(),tests=['unknown_gradient_zero','partial_set_loss_monotonic','ordered_interval_gradient','finite_sample_patient_quantile','empty_prediction_is_infinite_residual','patient_disjointness','outer_partition','sign_invariance','CPU_real_backward_timing'],backward_seconds=timings,conservative_training_seconds=estimate,optimizer_steps=0,model_fits=0,parameters=sum(v.numel() for v in model.parameters()),elapsed_seconds=time.monotonic()-t))
    return d
def verify_lock():
    for p,h in read(RUN/'source_lock.json').items():assert sha(p)==h,f'Source changed:{p}'
def worker(w):
    verify_lock();d=data()
    for j in jobs():
        if j['index']%2==w:train(j,d)
    put(RUN/f'worker_{w}.json',dict(state='finished',time=now(),pid=os.getpid()))
def wilson(k,n):
    if not n:return [None,None]
    z=1.95996398454;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d;return [c-h,c+h]
def analyse():
    verify_lock();rows=[]
    for a in ARMS:
        for seed in B['seeds']:
            paths=[jp(j) for j in jobs() if j['arm']==a and j['seed']==seed]
            for policy in ['nominal','wave_diagnostic','patient']:
                cc=pd.concat([pd.read_csv(p/f'{policy}_counts.csv') for p in paths]);ww=pd.concat([pd.read_csv(p/f'{policy}_waves.csv') for p in paths]);pp=pd.concat([pd.read_csv(p/f'{policy}_patients.csv') for p in paths]);width=[]
                for p in paths:
                    if (p/f'{policy}_widths.csv').stat().st_size>1:width.append(pd.read_csv(p/f'{policy}_widths.csv'))
                wd=pd.concat(width) if width else pd.DataFrame();pdout=RUN/'analysis'/f'{a}_{seed}_{policy}';pdout.parent.mkdir(exist_ok=True)
                cc.to_csv(str(pdout)+'_counts.csv',index=False);ww.to_csv(str(pdout)+'_waves.csv',index=False);pp.to_csv(str(pdout)+'_patients.csv',index=False)
                for lead in ['ii','v1']:
                    c=cc[cc.lead.eq(lead)];w=ww[ww.lead.eq(lead)];nr=int(c.reference.sum());nm=int(c.matched.sum());npred=int(c.predicted.sum());row=dict(arm=a,seed=seed,policy=policy,lead=lead,reference=nr,predicted=npred,matched=nm,recall=nm/max(nr,1),precision=nm/max(npred,1),patient_joint_coverage=float(pp.patient_calibrated_joint.mean()),reported_patient_joint_coverage=float(pp.reported_joint.mean()),patient_coverage_Wilson95=wilson(int(pp.patient_calibrated_joint.sum()),len(pp)),raw_nominal_patient_joint_coverage=float(pp.raw_nominal_joint.mean()),distinct_patients=int(c.patient.nunique()))
                    for kind in ['negative','biphasic']:
                        v=w[w.polarity.eq(kind)];row[kind+'_recall']=float(v.matched.mean()) if len(v) else None;row[kind+'_events']=len(v);row[kind+'_patients']=int(v.patient.nunique())
                    vals=wd.loc[wd.lead.eq(lead),['onset_width_ms','offset_width_ms']].to_numpy().ravel() if len(wd) else np.array([]);row['median_reported_width_ms']=float(np.median(vals)) if len(vals) else None;row['p95_reported_width_ms']=float(np.quantile(vals,.95)) if len(vals) else None;rows.append(row)
    pd.DataFrame(rows).to_csv(RUN/'analysis/summary.csv',index=False);put(RUN/'analysis/summary.json',rows)
    gates=[]
    for seed in B['seeds']:
        def get(a,l):return next(v for v in rows if v['arm']==a and v['seed']==seed and v['policy']=='patient' and v['lead']==l)
        base=get(ARMS[0],'v1');new=get(ARMS[2],'v1');g={}
        for k in ['negative','biphasic']:g[k+'_gain_5pp']=new[k+'_recall'] is not None and base[k+'_recall'] is not None and new[k+'_recall']-base[k+'_recall']>=.05;g[k+'_enough_patients']=new[k+'_patients']>=10
        for lead in ['ii','v1']:
            b,n=get(ARMS[0],lead),get(ARMS[2],lead);g[lead+'_noninferior']=n['recall']>=b['recall']-.02 and n['precision']>=b['precision']-.02;g[lead+'_useful_calibration']=n['patient_joint_coverage']>=.95 and n['median_reported_width_ms'] is not None and n['median_reported_width_ms']<=120 and n['p95_reported_width_ms']<=200 and n['recall']>=.8
        gates.append(dict(seed=seed,checks=g,pass_all=all(g.values())))
    put(RUN/'analysis/gates.json',dict(seeds=gates,measurement_engineering_pass=all(g['pass_all'] for g in gates),clinical_noP_validated=False,AF_morphology_increment=False,future_stroke_prediction=False,NC_submission_ready=False,RDB_scored=False,SPH_admitted=False,new_figures=0))
def main():
    assert not (RUN/'coordinator_started.json').exists(),'No automatic rerun';preflight();verify_lock()
    put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),budget=B,preflight='PASS',RDB_scoring=False))
    put(RUN/'coordinator_started.json',dict(time=now(),unix_time=time.time(),pid=os.getpid(),max_jobs=30,max_epochs=360));procs=[]
    try:
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=i,pid=p.pid) for i,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            assert not any(p.poll() not in [None,0] for p,_ in procs),'Worker failure';stopcheck('audit')
            done=list((RUN/'models').glob('*/*/*/done.json'));epochs=sum(len(pd.read_csv(p)) for p in (RUN/'models').glob('*/*/*/history.csv'));attempts=len(list((RUN/'models').glob('*/*/*/attempt.json')));assert epochs<=360 and attempts<=30
            put(RUN/'status.json',dict(time=now(),state='running',completed_jobs=len(done),training_attempts=attempts,completed_epochs=epochs,max_jobs=30,max_epochs=360,SPH_admitted=False,RDB_scored=False,new_figures=0));time.sleep(10)
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'models').glob('*/*/*/done.json')))==30
        analyse();put(RUN/'completion.json',dict(time=now(),state='bounded_complete_pending_external_review',jobs=30,epochs=360,AF_fits=0,encoder_forwards=0,RDB_scored=False,SPH_admitted=False,new_figures=0,hospital_branch='waiting',results='analysis/gates.json'))
    except Exception:
        (RUN/'STOP_NEW_WORK').write_text(traceback.format_exc())
        for p,_ in procs:
            if p.poll() is None:p.terminate()
        put(RUN/'stop.json',dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),automatic_retry=False));raise
    finally:
        for p,log in procs:
            if p.poll() is None:p.wait(timeout=30)
            log.close()
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--worker',type=int,choices=[0,1]);args=ap.parse_args()
    try: main() if args.worker is None else worker(args.worker)
    except Exception:
        if args.worker is not None:(RUN/'STOP_NEW_WORK').write_text(traceback.format_exc())
        elif not (RUN/'stop.json').exists():put(RUN/'stop.json',dict(time=now(),state='PREFLIGHT_FAIL_STOP',traceback=traceback.format_exc(),automatic_retry=False))
        raise
