"""One frozen A32 invocation. Owned workers only; no retry or automatic next phase."""
import sys,os,time,math,hashlib,traceback,subprocess,shutil
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
import numpy as np
import torch
from torch import nn
import a32_method as q
RUN=q.RUN;B=q.B;m=q.m;d=q.d;e=q.e
read=q.read;put=q.put;sha=q.sha;now=q.now
torch.set_num_threads(1);torch.set_grad_enabled(True)

def check(stage=None):
    assert not (RUN/'STOP').exists(),'A32 requested STOP'
    start=read(RUN/'started.json');assert time.time()<start['deadline_unix'],'A32 overall7200s deadline'
    if stage=='preflight':assert time.time()-start['unix_time']<B['preflight_seconds'],'A32 preflight600s cap'
    if stage in ['workers','analysis']:
        p=RUN/('training_started.json' if stage=='workers' else 'analysis_started.json');lim=B['worker_seconds'] if stage=='workers' else B['analysis_seconds'];assert time.time()-read(p)['unix_time']<lim,f'A32 {stage} time cap'

def verify():
    for path,h in read(RUN/'source_lock.json').items():check();assert sha(path)==h,f'Frozen source changed: {path}'

def output_size():
    total=0
    for p in RUN.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:
            if p.suffix!='.tmp':raise
    return total

def freeze():
    assert read(q.A31/'completion.json')['source_hashes_unchanged'] and read(q.A31/'post_review/contract_review.json')['state']=='PASS'
    locks=read(q.A31/'source_lock.json');extra=[RUN/'protocol.txt',RUN/'budget.json',*sorted((RUN/'code').glob('*.py')),q.A31/'source_lock.json',q.A31/'completion.json',q.A31/'post_review/contract_review.json',q.A31/'analysis/decision.json',q.A31/'post_review/解释补充_窗口截断与重训边界.txt']
    for p in extra:check('preflight');locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),protocol_SHA256=sha(RUN/'protocol.txt'),source_lock_SHA256=sha(RUN/'source_lock.json'),budget=B,previous_records_unchanged=True))
    put(RUN/'reference_admission.json',dict(time=now(),inherited_from=str(q.A31/'reference_admission.json'),new_admissions=0,SPH_admitted=False,RDB_scored=False,LUDB_repeatedly_viewed=True,hospital_branch='waiting',expert_reference='waiting',no_new_search_or_download=True))

def preflight():
    check('preflight');tests=q.synthetic_tests();data=q.prepare();base,targets,coverage=data;signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base
    assert x.shape==(391,2,5000) and np.isfinite(x).all() and not np.any(partial&(y==-100)) and not np.any(partial&np.isin(y,[2,3]))
    for role in roles:
        sets=[set(role[k]) for k in ['train','design','calibration','outer']];assert list(map(len,sets))==[124,12,24,40] and all(not a&b for i,a in enumerate(sets) for b in sets[:i])
        batchlist=list(q.batches(ids,role['train']));assert len(batchlist)==31 and sorted(i for b in batchlist for i in b)==sorted(i for i,(p,l) in enumerate(ids) if p in role['train'])
    assert sorted(p for r in roles for p in r['outer'])==list(range(200));put(RUN/'patient_roles.json',roles);put(RUN/'target_support.json',coverage)
    train=np.array([i for i,(p,l) in enumerate(ids) if p in roles[0]['train']]);weights,_,weightmeta=m.old.A9.weight_maps(signals,ann,x,y,np.array([a for a,l in ids]),np.array([l for a,l in ids]),train)
    batches=list(q.batches(ids,roles[0]['train']));rank=sorted(batches,key=lambda b:sum(len(targets[int(i)]['positive'])+len(targets[int(i)]['null']) for i in b),reverse=True);timings=[];torch.manual_seed(3200);model=m.old.A9.Net();before=q.tensor_sha(model)
    for arm in B['arms']:
        for repeat,batch in enumerate(rank[:3]):
            check('preflight');assert all(ids[int(i)][0] in roles[0]['train'] for i in batch);model.zero_grad(set_to_none=True);t=time.monotonic();xx=torch.from_numpy(x[batch]);lg=m.old.fwd(model,xx);loss,meta=q.patient_loss(lg,torch.from_numpy(y[batch]),torch.from_numpy(weights[batch]),torch.from_numpy(partial[batch]),arm,legacy,batch,targets,ids,signals,q.choice(dict(seed=B['seeds'][0],fold=0))['threshold']);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());timings.append(dict(arm=arm,seconds=time.monotonic()-t,patients=sorted({ids[i][0] for i in batch}),leads=len(batch),supervision=meta))
            if arm==B['arms'][1]:assert meta['positive']==sum(len(targets[int(i)]['positive']) for i in batch) and meta['endpoint_terms']==meta['positive']
    assert before==q.tensor_sha(model),'Zero optimizer steps in preflight'
    with torch.inference_mode():
        a=m.old.fwd(model,torch.from_numpy(x[rank[0]]));b=m.old.fwd(model,-torch.from_numpy(x[rank[0]]));assert torch.allclose(a,b,atol=1e-7,rtol=0)
    estimate=max(v['seconds'] for v in timings)*B['max_optimizer_steps']/2*1.5+600
    assert estimate<B['worker_seconds'],f'CPU resource preflight failed:{estimate:.1f}s >=5400s; no fits allowed'
    assert shutil.disk_usage(RUN).free>=4*1024**3,'Require4GiB output space'
    put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests+['original_disjoint_patient_roles','200_outer_patient_partition','391_admitted_leads','31_complete_patient_batches_per_fold','train_only_weights_and_real_backward','finite_gradients','all_supported_positives_each_batch','zero_optimizer_steps_unchanged_weights','sign_invariance','CPU_resource_gate','free_disk'],new_fits=0,optimizer_steps=0,real_backward_batches=len(timings),backward_timings=timings,conservative_worker_seconds=estimate,estimated_total_minutes=[max(20,estimate/60*.65),min(110,estimate/60+15)],parameters=sum(p.numel() for p in model.parameters()),all_extreme_patients_retained=True,calibration_outer_targets_used_in_loss=False))
    return data

def infer(model,signals,ann,patients,anchors,choice,role):
    rows={p:[] for p in B['policies']}
    with torch.inference_mode():
        for pat in patients:
            check('workers');prob=m.old.fwd(model,torch.from_numpy(m.old.A9.channels(signals[pat,:,[1,6]].copy()))).softmax(1).numpy();assert prob.shape==(2,4,5000) and np.isfinite(prob).all()
            for k,li in enumerate([1,6]):
                for policy in B['policies']:
                    if policy=='L0_legacy_decoder':
                        raw=d.posterior_decode(prob[k],signals[pat,:,li],anchors[pat]);accepted=[p for p in raw if (choice['mode']=='union' or p['kind']==choice['mode']) and p['confidence']>=choice['threshold']];selected,ledger=d.merge(accepted,anchors[pat],support_rank=True);decorate=d.decorate
                    else:
                        raw=q.decode(prob[k],signals[pat,:,li],anchors[pat]);accepted=[p for p in raw if p['confidence']>=choice['threshold']];selected,ledger=q.merge(accepted,anchors[pat]);decorate=q.decorate
                    qs=ann[pat]['leads'][li]['qrs'];rows[policy].append(dict(patient=int(pat),record=ann[pat]['record'],li=li,lead='ii' if li==1 else 'v1',role=role,qrs=qs,ref=m.old.references(ann,pat,li),raw=decorate(raw,anchors[pat],qs),selected=decorate(selected,anchors[pat],qs),merger=ledger))
    return rows

def train(j,data):
    check('workers');p=q.jobpath(j);assert not p.exists(),'A32 no job retry/overwrite';p.mkdir(parents=True);put(p/'attempt.json',dict(time=now(),job=j));base,targets,coverage=data;signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base;role=roles[j['fold']];indices=np.array([i for i,(pat,l) in enumerate(ids) if pat in role['train']]);seed=j['seed']+100*j['fold'];rng=np.random.default_rng(seed);torch.manual_seed(seed);model=m.old.A9.Net();initial=q.tensor_sha(model);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);weights,_,wm=m.old.A9.weight_maps(signals,ann,x,y,np.array([a for a,l in ids]),np.array([l for a,l in ids]),indices);choice=q.choice(j)
    put(p/'training_contract.json',dict(job=j,roles=role,train_leads=len(indices),initial_tensor_SHA256=initial,train_only_weights=wm,epochs=8,patients_per_batch=4,control='Fresh paired legacy objective on SAME whole-patient batches; NOT A30 exact replay',choice=choice))
    hist=[];ledger=[];orderhash=hashlib.sha256();aughash=hashlib.sha256();steps=0;origin=time.monotonic()
    for epoch in range(B['epochs_per_fit']):
        model.train();order=rng.permutation(np.array(role['train'],dtype=np.int64));orderhash.update(order.tobytes());losses=[];totals=dict(positive=0,null=0,outside_old=0,endpoint_terms=0,report_missing=0);visited=[]
        for batch in q.batches(ids,order):
            check('workers');xx=m.old.A9.augment(x[batch],rng);aughash.update(xx.tobytes());visited.extend(map(int,batch));opt.zero_grad(set_to_none=True);lg=m.old.fwd(model,torch.from_numpy(xx));loss,meta=q.patient_loss(lg,torch.from_numpy(y[batch]),torch.from_numpy(weights[batch]),torch.from_numpy(partial[batch]),j['arm'],legacy,batch,targets,ids,signals,choice['threshold']);assert torch.isfinite(loss),'Nonfinite A32 loss';loss.backward();gn=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(gn),'Nonfinite A32 gradient';opt.step();steps+=1;losses.append(float(loss.detach()))
            for key in totals:totals[key]+=meta[key]
        assert sorted(visited)==sorted(map(int,indices)) and len(visited)==len(set(visited))
        if j['arm']==B['arms'][1]:assert totals['positive']==sum(len(targets[int(i)]['positive']) for i in indices) and totals['endpoint_terms']==totals['positive']
        hist.append(dict(epoch=epoch+1,loss=float(np.mean(losses)),optimizer_steps=steps,wall_seconds=time.monotonic()-origin,supervision=totals));ledger.append(dict(epoch=epoch+1,patient_order=list(map(int,order)),lead_items=visited,supervision=totals));put(p/'history.json',hist);put(RUN/f"worker_{j['index']%2}.json",dict(time=now(),pid=os.getpid(),state='training',job=j,epoch=epoch+1,optimizer_steps=steps))
    torch.save(model.state_dict(),p/'final.pt');put(p/'training_done.json',dict(time=now(),job=j,epochs=8,optimizer_steps=steps,initial_tensor_SHA256=initial,order_SHA256=orderhash.hexdigest(),augmentation_SHA256=aughash.hexdigest(),checkpoint_SHA256=sha(p/'final.pt')));q.gz(p/'sampling_ledger.json.gz',ledger);model.eval();caches={policy:{} for policy in B['policies']}
    for rname in ['design','calibration','outer']:
        rr=infer(model,signals,ann,role[rname],anchors,choice,rname)
        for policy in B['policies']:caches[policy][rname]=rr[policy]
    q.gz(p/'prediction_cache.json.gz',dict(job=j,policies=caches))
    for policy,cache in caches.items():
        feasible=True;designcounts={}
        for lead in ['ii','v1']:
            rr=[r for r in cache['design'] if r['lead']==lead];nr=sum(len(r['ref']) for r in rr);nm=sum(len(d.point_pairs(r['ref'],r['selected'],r['qrs'])) for r in rr);npred=sum(len(r['selected']) for r in rr);designcounts[lead]=dict(reference=nr,matched=nm,predicted=npred,recall=nm/max(1,nr));feasible=feasible and nm/max(1,nr)>=.8
        cc=e.calibrate(cache['calibration']);out=e.evaluate(cache['outer'],cc,signals,labels);q.gz(p/(policy+'.json.gz'),dict(job=j,policy=policy,choice=choice,design_feasible=feasible,design_counts=designcounts,**out))
    put(p/'done.json',dict(time=now(),job=j,new_fit=1,epochs=8,optimizer_steps=steps,inference_lead_signals=152,policy_evaluations=2,AF_fits=0,checkpoint_SHA256=sha(p/'final.pt'),wall_seconds=time.monotonic()-origin))

def worker(w):
    verify();data=q.prepare()
    for j in q.jobs():
        if j['index']%2==w:train(j,data)
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))

def main():
    assert not (RUN/'started.json').exists(),'One A32 invocation only';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['overall_seconds'],pid=os.getpid(),max_seconds=B['overall_seconds']));procs=[]
    try:
        freeze();preflight();verify();put(RUN/'training_started.json',dict(time=now(),unix_time=time.time(),deadline_unix=min(origin+7200,time.time()+5400)))
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check('workers');assert not any(p.poll() not in [None,0] for p,_ in procs),'Owned A32 worker failed';attempts=list((RUN/'jobs').glob('*/attempt.json'));hist=[read(p) for p in (RUN/'jobs').glob('*/history.json')];steps=sum(v[-1]['optimizer_steps'] for v in hist if v);epochs=sum(len(v) for v in hist);done=list((RUN/'jobs').glob('*/done.json'));assert len(attempts)<=B['max_fits'] and steps<=B['max_optimizer_steps'] and epochs<=B['max_epochs'] and output_size()<=B['max_output_bytes'],'A32 frozen budget excess'
            put(RUN/'status.json',dict(time=now(),state='running_bounded_patient_training',fit_attempts=len(attempts),completed_fits=len(done),completed_epochs=epochs,optimizer_steps=steps,max_fits=20,max_epochs=160,AF_fits=0,SPH_admitted=False,new_figures=0,seconds_remaining=origin+7200-time.time()));time.sleep(10)
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==20,'Incomplete A32 training';put(RUN/'analysis_started.json',dict(time=now(),unix_time=time.time()));put(RUN/'status.json',dict(time=now(),state='analysis_independent_review',completed_fits=20,completed_epochs=160))
        import a32_review as review
        review.analyse();review.review();verify();check('analysis');receipts=[read(q.jobpath(j)/'done.json') for j in q.jobs()];assert sum(v['inference_lead_signals'] for v in receipts)==3040 and sum(v['optimizer_steps'] for v in receipts)<=4960 and output_size()<=B['max_output_bytes']
        complete=dict(time=now(),state='bounded_patient_training_complete_review_PASS',fits=20,epochs=160,optimizer_steps=sum(v['optimizer_steps'] for v in receipts),inference_lead_signals=3040,policy_evaluations=40,source_hashes_unchanged=True,wall_seconds=time.time()-origin,measurement_gate_pass=read(RUN/'analysis/gates.json')['both_seed_primary_pass'],NC_submission_ready=False,AF_fits=0,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting',automatic_followon_training=False);put(RUN/'completion.json',complete);put(RUN/'status.json',complete)
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
    if args.mock_preflight:print(dict(state='PASS',tests=q.synthetic_tests(),fits=0,optimizer_steps=0))
    else:
        try:main() if args.worker is None else worker(args.worker)
        except Exception:
            if args.worker is not None:(RUN/'STOP').touch()
            raise
