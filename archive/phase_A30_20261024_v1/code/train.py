"""Bounded A30 coordinator/worker. No new training before frozen preflight PASS."""
import sys,os,time,math,json,hashlib,traceback,subprocess,shutil
from pathlib import Path
import numpy as np
import torch
from torch import nn
import method as q
sys.path.insert(0,str(Path(__file__).resolve().parent))
RUN=q.RUN;B=q.B;m=q.m;d=q.d;e=q.e
put=q.put;read=q.read;sha=q.sha;now=q.now
torch.set_num_threads(1);torch.set_grad_enabled(True)

def check(stage=None):
    assert not (RUN/'STOP').exists(),'A30 peer/coordinator stop'
    elapsed=time.time()-read(RUN/'started.json')['unix_time'];assert elapsed<B['overall_seconds'],'A30 fixed overall deadline'
    if stage=='preflight':assert elapsed<B['preflight_seconds'],'Preflight600s cap'
    if stage=='workers':assert time.time()-read(RUN/'training_started.json')['unix_time']<B['worker_seconds'],'Workers5400s cap'
    if stage=='analysis':assert time.time()-read(RUN/'analysis_started.json')['unix_time']<B['analysis_seconds'],'Analysis1200s cap'

def verify():
    for p,h in read(RUN/'source_lock.json').items():check();assert sha(p)==h,f'Frozen source changed:{p}'

def size():
    total=0
    for p in RUN.rglob('*'):
        if not p.is_file():continue
        try:total+=p.stat().st_size
        except FileNotFoundError:
            if p.suffix!='.tmp':raise
    return total

def frozen_choice(j):
    p=m.path(dict(j,arm='C2_null_ordered'))/'design_selection.json'
    c=read(p)['choice'];assert c['mode'] in ['union','ordered','continuous'] and 0<=c['threshold']<=1
    return dict(mode=c['mode'],threshold=c['threshold'],source=str(p),shared_across_A30_arms=True,selected_in_A30=False)

def freeze():
    assert read(q.A29/'post_review/contract_review.json')['state']=='PASS'
    locks=read(q.A29/'source_lock.json');extra=[RUN/'protocol.txt',RUN/'budget.json',RUN/'code/method.py',Path(__file__),RUN/'code/analyse.py',RUN/'code/review.py',q.A29/'source_lock.json',q.A29/'code/review.py',q.A29/'post_review/contract_review.json',q.A29/'post_review/retraining_decision.json',q.A29/'analysis/summary.json']
    for seed in B['seeds']:
        for fold in B['folds']:extra+=[m.path(dict(arm='C2_null_ordered',seed=seed,fold=fold))/x for x in ['history.json','design_selection.json','done.json']]
    for p in extra:check('preflight');locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),protocol_SHA256=sha(RUN/'protocol.txt'),source_lock_SHA256=sha(RUN/'source_lock.json'),budget=B,old_results_unchanged=True))
    put(RUN/'reference_admission.json',dict(time=now(),inherited_evidence=str(q.A29/'reference_admission.json'),new_official_search=False,new_admissions=0,SPH_admitted=False,RDB_scored=False,hospital_branch='waiting',expert_reference='waiting'))

def preflight():
    check('preflight');tests=q.synthetic_preflight();data=q.prepare();base,pools,coverage=data;signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base
    assert x.shape==(391,2,5000) and np.isfinite(x).all() and not np.any(partial&(y==-100)) and not np.any(partial&np.isin(y,[2,3]))
    for role in roles:
        sets=[set(role[k]) for k in ['train','design','calibration','outer']];assert list(map(len,sets))==[124,12,24,40] and all(not a&b for i,a in enumerate(sets) for b in sets[:i])
    assert sorted(p for role in roles for p in role['outer'])==list(range(200));put(RUN/'patient_roles.json',roles);put(RUN/'target_support.json',coverage)
    train=np.array([i for i,(p,l) in enumerate(ids) if p in roles[0]['train']]);batch=train[:8];assert all(ids[i][0] in roles[0]['train'] for i in batch)
    torch.manual_seed(300);model=m.old.A9.Net();initial=q.tensor_sha(model);xx=torch.from_numpy(x[batch]);yy=torch.from_numpy(y[batch]);amb=torch.from_numpy(partial[batch]);weights=torch.ones_like(yy,dtype=torch.float32);picked=q.chosen(pools,batch,B['seeds'][0],0,0);timings=[]
    with torch.inference_mode():
        a=m.old.fwd(model,xx);b=m.old.fwd(model,-xx);assert torch.allclose(a,b,atol=1e-7,rtol=0)
    for arm in B['arms']:
        for repeat in range(3):
            check('preflight');model.zero_grad(set_to_none=True);t=time.monotonic();lg=m.old.fwd(model,xx);loss,meta=q.loss(lg,yy,weights,amb,arm,legacy,batch,picked);assert torch.isfinite(loss);loss.backward();assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters());timings.append(dict(arm=arm,seconds=time.monotonic()-t))
    assert initial==q.tensor_sha(model),'Preflight must not update weights'
    conservative=max(v['seconds'] for v in timings)*B['max_optimizer_steps']/2*1.5+600
    assert conservative<B['worker_seconds'],f'CPU timing gate:{conservative:.1f}s >5400s; do not start fits'
    assert shutil.disk_usage(RUN).free>=4*1024**3,'Require4GiB free output volume'
    put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests+['original_disjoint_patient_roles','outer_partition200','all391_lead_inputs','full_pool_legacy_prefix_identity','sign_invariance','train_only_real_backward','finite_gradients','no_preflight_weight_update','CPU_conservative_timing','free_output_disk'],real_backward_batches=len(timings),optimizer_steps=0,new_fits=0,parameters=sum(p.numel() for p in model.parameters()),backward_timings=timings,conservative_worker_seconds=conservative,estimated_compute_minutes=conservative/60,all_extreme_patients_retained=True));return data

def infer(model,signals,ann,patients,anchors,choice,role):
    rows=[]
    with torch.inference_mode():
        for pat in patients:
            check('workers');prob=m.old.fwd(model,torch.from_numpy(m.old.A9.channels(signals[pat,:,[1,6]].copy()))).softmax(1).numpy()
            assert prob.shape==(2,4,5000) and np.isfinite(prob).all() and np.all((prob>=0)&(prob<=1))
            for k,li in enumerate([1,6]):
                pp=d.posterior_decode(prob[k],signals[pat,:,li],anchors[pat]);accepted=[p for p in pp if (choice['mode']=='union' or p['kind']==choice['mode']) and p['confidence']>=choice['threshold']];selected,ledger=d.merge(accepted,anchors[pat],support_rank=True)
                assert len({p['wave_R'] for p in selected})==len(selected) and all(p['confidence']>=choice['threshold'] for p in selected)
                qs=ann[pat]['leads'][li]['qrs'];rows.append(dict(patient=int(pat),record=ann[pat]['record'],li=li,lead='ii' if li==1 else 'v1',role=role,qrs=qs,ref=m.old.references(ann,pat,li),raw=d.decorate(pp,anchors[pat],qs),selected=d.decorate(selected,anchors[pat],qs),merger=ledger))
    return rows

def train(j,data):
    check('workers');p=q.jobpath(j);assert not p.exists(),'No fit retry/overwrite';p.mkdir(parents=True);put(p/'attempt.json',dict(time=now(),job=j));base,pools,coverage=data;signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base;role=roles[j['fold']];indices=np.array([i for i,(pat,l) in enumerate(ids) if pat in role['train']]);assert len(indices)<=248;seed=j['seed']+100*j['fold'];rng=np.random.default_rng(seed);torch.manual_seed(seed);model=m.old.A9.Net();initial=q.tensor_sha(model);opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
    weights,_,meta=m.old.A9.weight_maps(signals,ann,x,y,np.array([a for a,l in ids]),np.array([l for a,l in ids]),indices);put(p/'training_contract.json',dict(job=j,roles=role,train_lead_items=len(indices),initial_tensor_SHA256=initial,train_only_weights=meta,control='A30 C0 reproduces A27 C2 legacy structured objective; not A27 C0_complete',fresh_initialisation=True,epochs=8))
    hist=[];sampling=[];augmentation=hashlib.sha256();orderhash=hashlib.sha256();steps=0;origin=time.monotonic()
    for epoch in range(B['epochs_per_fit']):
        model.train();order=rng.permutation(indices);orderhash.update(order.tobytes());losses=[];tail_values=[];positions={b:dict(items=0,eligible=0,represented=0,positive=0,null=0) for b in range(8)};target_rows=[]
        for start in range(0,len(order),8):
            check('workers');batch=order[start:start+8];xx=m.old.A9.augment(x[batch],rng);augmentation.update(xx.tobytes());yy=torch.from_numpy(y[batch]);amb=torch.from_numpy(partial[batch]);picked=q.chosen(pools,batch,j['seed'],j['fold'],epoch)
            if j['arm']==B['arms'][0]:actual=[(b,t,'positive' if t[1] is not None else 'null',int(batch[b]),None) for b,t in [(b,t) for b,i in enumerate(batch) for t in legacy[int(i)]][:24]]
            else:actual=picked
            for b,i in enumerate(batch):
                pos=positions[b];pos['items']+=1;pos['eligible']+=any(pools[int(i)].values());v=[a for a in actual if a[0]==b];pos['represented']+=bool(v);pos['positive']+=sum(a[2]=='positive' for a in v);pos['null']+=sum(a[2]=='null' for a in v)
                if j['arm']!=B['arms'][0] and any(pools[int(i)].values()):assert v,'Every eligible lead item needs a structural target'
            target_rows.extend(dict(item=i,kind=kind,pool_index=k,anchor=t[0],onset=t[1],offset=t[2]) for b,t,kind,i,k in actual)
            opt.zero_grad(set_to_none=True);lg=m.old.fwd(model,torch.from_numpy(xx));value,details=q.loss(lg,yy,torch.from_numpy(weights[batch]),amb,j['arm'],legacy,batch,picked);assert torch.isfinite(value),'Nonfinite train loss';value.backward();grad=nn.utils.clip_grad_norm_(model.parameters(),1);assert torch.isfinite(grad),'Nonfinite gradient';opt.step();steps+=1;losses.append(float(value.detach()));tail_values.append(details['tail'])
        row=dict(epoch=epoch+1,loss=float(np.mean(losses)),tail_diagnostic=float(np.mean(tail_values)),optimizer_steps=steps,wall_seconds=time.monotonic()-origin);hist.append(row);sampling.append(dict(epoch=epoch+1,positions=positions,targets=target_rows))
        if j['arm']==B['arms'][0]:
            old=read(m.path(dict(j,arm='C2_null_ordered'))/'history.json')[epoch];assert abs(row['loss']-old['loss'])<=1e-6,'C0 legacy first8 training-loss replay'
        put(p/'history.json',hist);put(RUN/f"worker_{j['index']%2}.json",dict(time=now(),pid=os.getpid(),state='training',job=j,epoch=epoch+1,optimizer_steps=steps))
    check('workers');torch.save(model.state_dict(),p/'final.pt');put(p/'training_done.json',dict(time=now(),job=j,epochs=8,optimizer_steps=steps,initial_tensor_SHA256=initial,order_SHA256=orderhash.hexdigest(),augmentation_SHA256=augmentation.hexdigest(),checkpoint_SHA256=sha(p/'final.pt')));q.gz(p/'sampling_ledger.json.gz',sampling);model.eval();choice=frozen_choice(j)
    design=infer(model,signals,ann,role['design'],anchors,choice,'design');by={}
    for lead in ['ii','v1']:
        rr=[r for r in design if r['lead']==lead];nr=sum(len(r['ref']) for r in rr);nm=sum(len(d.point_pairs(r['ref'],r['selected'],r['qrs'])) for r in rr);npred=sum(len(r['selected']) for r in rr);by[lead]=dict(reference=nr,matched=nm,predicted=npred,recall=nm/max(nr,1),precision=nm/npred if npred else None)
    feasible=all(v['recall']>=.8 for v in by.values());put(p/'design.json',dict(choice=choice,by_lead=by,design_feasible=feasible,threshold_searches=0));cal=infer(model,signals,ann,role['calibration'],anchors,choice,'calibration');outer=infer(model,signals,ann,role['outer'],anchors,choice,'outer');q.gz(p/'prediction_cache.json.gz',dict(job=j,design=design,calibration=cal,outer=outer));cc=e.calibrate(cal);result=e.evaluate(outer,cc,signals,labels);q.gz(p/'result.json.gz',dict(job=j,design_feasible=feasible,choice=choice,**result))
    put(p/'done.json',dict(time=now(),job=j,epochs=8,optimizer_steps=steps,inference_lead_signals=2*(len(role['design'])+len(role['calibration'])+len(role['outer'])),new_fit=1,AF_fits=0,checkpoint_SHA256=sha(p/'final.pt'),wall_seconds=time.monotonic()-origin))

def worker(w):
    verify();data=q.prepare()
    for j in q.jobs():
        if j['index']%2==w:train(j,data)
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))

def main():
    assert not (RUN/'started.json').exists(),'One bounded A30 invocation';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['overall_seconds'],pid=os.getpid(),max_seconds=B['overall_seconds']));procs=[]
    try:
        freeze();preflight();verify();put(RUN/'training_started.json',dict(time=now(),unix_time=time.time(),deadline_unix=min(origin+7200,time.time()+5400)));logs=[]
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check('workers');assert not any(p.poll() not in [None,0] for p,_ in procs),'Owned worker failed';attempts=list((RUN/'jobs').glob('*/attempt.json'));hist=[read(p) for p in (RUN/'jobs').glob('*/history.json')];steps=sum(v[-1]['optimizer_steps'] for v in hist if v);epochs=sum(len(v) for v in hist);done=list((RUN/'jobs').glob('*/done.json'))
            assert len(attempts)<=30 and steps<=7440 and epochs<=240 and size()<=B['max_output_bytes'],'Frozen compute/output budget';put(RUN/'status.json',dict(time=now(),state='running_bounded_training',fit_attempts=len(attempts),completed_fits=len(done),completed_epochs=epochs,optimizer_steps=steps,max_fits=30,max_epochs=240,AF_fits=0,SPH_admitted=False,new_figures=0,seconds_remaining=origin+7200-time.time()));time.sleep(min(10,max(.05,origin+7200-time.time())))
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==30,'Incomplete bounded training';put(RUN/'analysis_started.json',dict(time=now(),unix_time=time.time()));put(RUN/'status.json',dict(time=now(),state='analysis_review',completed_fits=30,completed_epochs=240))
        import analyse,review
        analyse.main();review.main();verify();check('analysis');receipts=[read(q.jobpath(j)/'done.json') for j in q.jobs()];assert sum(r['inference_lead_signals'] for r in receipts)==4560 and sum(r['optimizer_steps'] for r in receipts)<=7440 and size()<=B['max_output_bytes'];put(RUN/'completion.json',dict(time=now(),state='bounded_training_complete_review_PASS',fits=30,epochs=240,optimizer_steps=sum(r['optimizer_steps'] for r in receipts),inference_lead_signals=4560,sign_orbit_passes=9120,source_hashes_unchanged=True,wall_seconds=time.time()-origin,measurement_gate_pass=read(RUN/'analysis/gates.json')['both_seed_primary_pass'],NC_submission_ready=False,AF_fits=0,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting',automatic_followon_training=False));put(RUN/'status.json',read(RUN/'completion.json'))
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
    if args.mock_preflight:print(json.dumps(dict(state='PASS',tests=q.synthetic_preflight(),optimizer_steps=0)))
    else:
        try:main() if args.worker is None else worker(args.worker)
        except Exception:
            if args.worker is not None:(RUN/'STOP').touch()
            raise
