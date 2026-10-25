"""Frozen-cache audit only: no network instance, checkpoint load, forward or fit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,time,math,json,traceback,subprocess,shutil,importlib.util
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A32=BASE/'runs/phase_A32_20261024_v1'
sys.path.insert(0,str(A32/'code'));import a32_method as q
q.torch.set_num_threads(1);q.torch.set_grad_enabled(False)
d=q.d;e=q.e;read=q.read;put=q.put;sha=q.sha;gz=q.gz;ungz=q.ungz;now=q.now
B=read(RUN/'budget.json');FACTORS=B['factorial_policies'];REPLAYS=B['policies'][:2]

def jobs():return q.jobs()
def path(j):return RUN/'jobs'/f"job_{j['index']:02d}"
def check():
    assert not (RUN/'STOP').exists(),'A33 STOP'
    assert time.time()<read(RUN/'started.json')['deadline_unix'],'A33 fixed7200s deadline'
def verify():
    for p,h in read(RUN/'source_lock.json').items():check();assert sha(p)==h,f'Frozen source changed:{p}'
def size():
    total=0
    for p in RUN.rglob('*'):
        try:
            if p.is_file():total+=p.stat().st_size
        except FileNotFoundError:
            if p.suffix!='.tmp':raise
    return total
def load_data():return q.m.prepare()
def strip(p):return {k:v for k,v in p.items() if k not in ['expert_beat','wave_R','uid']}

def components(oldraw,newraw,policy):
    assert policy in FACTORS;wide=policy in ['F10_wide_clipped','F11_wide_unclipped'];unclipped=policy in ['F01_old_unclipped','F11_wide_unclipped'];values=[];origins=[];extra=[]
    for branch,source,pp in [('ordered','W1' if wide else 'L0',newraw if wide else oldraw),('continuous','W1' if unclipped else 'L0',newraw if unclipped else oldraw)]:
        for p in pp:
            if p['kind']!=branch:continue
            if branch=='ordered' and not 1000<p['source_R']<4200:extra.append(dict(source=source,candidate=p));continue
            values.append(dict(strip(p),uid=len(values)));origins.append(dict(uid=len(values)-1,origin_policy=source,origin_uid=p['uid'],kind=branch))
    assert len({p['uid'] for p in values})==len(values)
    return values,origins,extra

def staged(pp,anchors,cutoff):
    accepted=[p for p in pp if p['confidence']>=cutoff];groups={}
    for p in accepted:
        r=q.waveform_anchor(p,anchors)
        if r is not None:groups.setdefault(r,[]).append(dict(p,wave_R=r))
    nms=[p for r,v in sorted(groups.items()) for p in d.nms(v)];selected,merger=q.merge(accepted,anchors)
    assert len({p['wave_R'] for p in selected})==len(selected)
    return dict(raw=pp,threshold=accepted,nms=nms,selected=selected),merger

def factorial(oldrow,newrow,anchors,policy,cutoff):
    for key in ['patient','record','li','lead','role','qrs','ref']:assert oldrow[key]==newrow[key],f'Cache reference/role mismatch:{key}'
    raw,origins,extra=components(oldrow['raw'],newrow['raw'],policy);stages,merger=staged(raw,anchors,cutoff);qs=oldrow['qrs'];decorated={k:q.decorate(v,anchors,qs) for k,v in stages.items()}
    row={k:oldrow[k] for k in ['patient','record','li','lead','role','qrs','ref']};row.update(raw=decorated['raw'],selected=decorated['selected'],merger=merger)
    return row,decorated,dict(origins=origins,excluded_extra_R=extra)

def oracle(ref,ps,qs,kind):
    slots=sorted({p['wave_R'] for p in ps if p.get('wave_R') is not None});edges=[]
    for r in ref:
        row=[]
        for slot in slots:
            good=False
            for p in ps:
                if p.get('wave_R')!=slot or (kind!='point_geometry' and not d.compatible(r,p,qs)):continue
                pt=d.iou(r,p)>=.5;loc=max(abs(r[0]-p['onset']),abs(r[2]-p['offset']))<=50
                if (kind in ['point','point_geometry'] and pt) or (kind=='local200' and loc) or (kind=='point_local200' and pt and loc):good=True;break
            row.append(good)
        edges.append(row)
    matched=len(d.assign(np.array(edges,dtype=float))) if ref and slots else 0
    return dict(matched=matched,slots=slots,edges=edges)

def diagnose(row,stages,extra):
    ref=row['ref'];qs=row['qrs'];rs={stage:{kind:oracle(ref,ps,qs,kind) for kind in ['point','point_geometry','local200','point_local200']} for stage,ps in stages.items()};matched={i for i,k in d.point_pairs(ref,stages['selected'],qs,True)};fail=[]
    def good(r,p):return d.compatible(r,p,qs) and p.get('wave_R') is not None and d.iou(r,p)>=.5
    for i,r in enumerate(ref):
        if i in matched:continue
        if not any(d.iou(r,p)>=.5 for p in stages['raw']):why='raw_geometry_absent_or_inaccurate'
        elif not any(good(r,p) for p in stages['raw']):why='waveform_identity_failure'
        elif not any(good(r,p) for p in stages['threshold']):why='numeric_threshold_loss'
        elif not any(good(r,p) for p in stages['nms']):why='local_NMS_loss'
        elif not any(good(r,p) for p in stages['selected']):why='same_R_competition_loss'
        else:why='slot_capacity_matching_collision'
        fail.append(dict(reference_index=i,reference=r,reason=why))
    assert len(fail)+len(matched)==len(ref)
    return dict(patient=row['patient'],record=row['record'],lead=row['lead'],role=row['role'],reference=len(ref),stages=rs,failures=fail,unresolved_raw=sum(p['wave_R'] is None for p in stages['raw']),extra_ordered_R_candidates=extra['excluded_extra_R'],origin_map=extra['origins'])

def calibration_traces(records,cal,policy):
    result=[];rank=sorted(cal['ledger'],key=lambda r:float('inf') if not math.isfinite(r['strict']) else r['strict'],reverse=True);top={r['patient'] for r in rank[:3]}|{r['patient'] for r in rank if not math.isfinite(r['strict'])}
    for patient in sorted(top):
        entry=next(r for r in rank if r['patient']==patient);rows=[]
        for row in [r for r in records if r['patient']==patient and r['ref']]:
            waves=[]
            for ref in row['ref']:
                same=[p for p in row['raw'] if p['wave_R'] is not None and d.compatible(ref,p,row['qrs'])];wins=[p for p in row['selected'] if d.compatible(ref,p,row['qrs'])]
                def detail(p):
                    absolute=max(abs(ref[0]-p['onset']),abs(ref[2]-p['offset']));return dict(candidate=p,absolute_max_endpoint_error_ms=2*absolute,scale_samples=p['scale'],normalized_residual=absolute/p['scale'],point_match=d.iou(ref,p)>=.5,support_score=p['confidence']*math.sqrt((p['offset']-p['onset'])/100))
                waves.append(dict(reference=ref,selected=[detail(p) for p in wins],same_beat_raw=[detail(p) for p in same],missing_same_beat=not wins))
            rows.append(dict(lead=row['lead'],waves=waves))
        result.append(dict(policy=policy,patient=patient,record=next(r['record'] for r in records if r['patient']==patient),patient_strict=entry['strict'],fold_q=cal['strict_q'],missing=entry['missing'],rows=rows))
    return result

def evaluate(rows,design,signals,labels,j,policy,choice):
    feasible=True;dc={}
    for lead in ['ii','v1']:
        rr=[r for r in design if r['lead']==lead];nr=sum(len(r['ref']) for r in rr);nm=sum(len(d.point_pairs(r['ref'],r['selected'],r['qrs'])) for r in rr);npred=sum(len(r['selected']) for r in rr);dc[lead]=dict(reference=nr,matched=nm,predicted=npred,recall=nm/max(1,nr));feasible=feasible and nm/max(1,nr)>=.8
    cal=e.calibrate(rows['calibration']);out=e.evaluate(rows['outer'],cal,signals,labels)
    return dict(job=j,policy=policy,choice=choice,design_feasible=feasible,design_counts=dc,**out)

def job(j,data):
    check();p=path(j);assert not p.exists(),'A33 no overwrite/retry';p.mkdir(parents=True);put(p/'attempt.json',dict(time=now(),job=j));signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=data;oldpath=q.jobpath(j);cache=ungz(oldpath/'prediction_cache.json.gz');assert cache['job']==j;choice=ungz(oldpath/'W1_window_aware.json.gz')['choice'];allrows={};diagnostics=[];traces=[];origin=time.monotonic()
    for name,original in zip(REPLAYS,['L0_legacy_decoder','W1_window_aware']):
        src=cache['policies'][original];rebuilt={}
        for role in ['design','calibration','outer']:
            rebuilt[role]=[]
            for row in src[role]:
                check();aa=anchors[row['patient']];raw=[dict(strip(v),uid=v['uid']) for v in row['raw']]
                if name==REPLAYS[0]:
                    accepted=[v for v in raw if (choice['mode']=='union' or v['kind']==choice['mode']) and v['confidence']>=choice['threshold']];selected,merger=d.merge(accepted,aa,support_rank=True);selected=d.decorate(selected,aa,row['qrs'])
                else:
                    accepted=[v for v in raw if v['confidence']>=choice['threshold']];selected,merger=q.merge(accepted,aa);selected=q.decorate(selected,aa,row['qrs'])
                assert e.clean(selected)==row['selected'],'A32 exact factual selected replay';rebuilt[role].append(dict(row,selected=selected))
        out=evaluate(rebuilt,rebuilt['design'],signals,labels,j,name,choice);stored=ungz(oldpath/(original+'.json.gz'))
        for key in ['calibration','records','patients','widths','component_widths_DIAGNOSTIC','design_feasible','design_counts']:assert e.clean(out[key])==stored[key],f'A32 exact factual result replay:{key}'
        allrows[name]=rebuilt;gz(p/('result_'+name+'.json.gz'),out);traces.extend(calibration_traces(rebuilt['calibration'],out['calibration'],name))
    for policy in FACTORS:
        values={}
        for role in ['design','calibration','outer']:
            oo=cache['policies']['L0_legacy_decoder'][role];nn=cache['policies']['W1_window_aware'][role];assert len(oo)==len(nn)==2*len(roles[j['fold']][role]);values[role]=[]
            for oldrow,newrow in zip(oo,nn):
                check();row,stages,meta=factorial(oldrow,newrow,anchors[oldrow['patient']],policy,choice['threshold']);values[role].append(row)
                if role!='design':diagnostics.append(dict(policy=policy,**diagnose(row,stages,meta)))
        out=evaluate(values,values['design'],signals,labels,j,policy,dict(choice,mode='union',common_ordered_R_domain=[1000,4200],no_selection_in_A33=True));allrows[policy]=values;gz(p/('result_'+policy+'.json.gz'),out);traces.extend(calibration_traces(values['calibration'],out['calibration'],policy))
    gz(p/'candidate_cache.json.gz',dict(job=j,policies=allrows));gz(p/'oracle_diagnostics.json.gz',diagnostics);gz(p/'calibration_traces.json.gz',traces);put(p/'done.json',dict(time=now(),job=j,cached_lead_records=152,factual_replays=2,factorial_evaluations=4,new_fits=0,model_forwards=0,epochs=0,wall_seconds=time.monotonic()-origin))

def synthetic_tests():
    tests=d.preflight()
    def prop(a,b,r,uid=0):return dict(onset=a,offset=b,peak=(a+b)//2,confidence=.9,box=[a,a,b,b],scale=5.,kind='ordered',source_R=r,wave_R=r,expert_beat=0,uid=uid)
    rr=[(40,50,60),(70,80,90)];pp=[prop(40,60,150),prop(70,90,150,1)];orac=oracle(rr,pp,[150],'point');assert orac['matched']==1 and len(d.point_pairs(rr,pp,[150],True))==2,'R-slot oracle must forbid two outputs'
    a=[prop(1300,1340,1400)];c=dict(a[0],kind='continuous',source_R=None);a.append(c);b=[dict(a[0],onset=1200,offset=1240),dict(c,onset=1250,offset=1290)]
    for policy in FACTORS:
        vv,origin,extra=components(a,b,policy);assert len(vv)==2 and len({v['uid'] for v in vv})==2 and all('expert_beat' not in v and 'wave_R' not in v for v in vv);ss,_=staged(vv,[1400],0);assert len(ss['selected'])<=1
    z=prop(4250,4280,4400);vv,_,extra=components([z],[z],FACTORS[0]);assert vv==[] and len(extra)==1
    return tests+['oracle_R_capacity_not_raw_candidate_count','factorial_component_uid_and_label_stripping','common_identity_and_one_R_winner','extra_ordered_R_logged_not_patient_removed']

def freeze():
    assert read(A32/'completion.json')['state']=='bounded_patient_training_complete_review_PASS' and read(A32/'post_review/contract_review.json')['state']=='PASS';locks=read(A32/'source_lock.json');locks.update(read(A32/'recoveries/R1_batch_accounting/source_lock.json'));extra=[RUN/'protocol.txt',RUN/'budget.json',RUN/'independent_sources.json',*sorted((RUN/'code').glob('*.py')),A32/'completion.json',A32/'post_review/contract_review.json',A32/'patient_roles.json',A32/'source_lock.json',A32/'recoveries/R1_batch_accounting/source_lock.json']
    for j in jobs():
        jp=q.jobpath(j);done=read(jp/'done.json');assert done['checkpoint_SHA256']==sha(jp/'final.pt');extra.extend(jp/name for name in ['prediction_cache.json.gz','L0_legacy_decoder.json.gz','W1_window_aware.json.gz','final.pt','done.json','training_done.json','training_contract.json'])
    for p in extra:check();locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),protocol_SHA256=sha(RUN/'protocol.txt'),budget=B,old_results_unchanged=True));put(RUN/'reference_admission.json',dict(time=now(),inherited_from=str(A32/'reference_admission.json'),new_independent_sources=0,SPH_admitted=False,RDB_scored=False,official_refresh=str(RUN/'independent_sources.json'),hospital_branch='waiting',expert_reference='waiting'))

def preflight():
    origin=read(RUN/'started.json')['unix_time'];check();assert time.time()-origin<B['preflight_seconds'];tests=synthetic_tests();data=load_data();signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=data
    for role in roles:
        sets=[set(role[k]) for k in ['train','design','calibration','outer']];assert list(map(len,sets))==[124,12,24,40] and all(not a&b for i,a in enumerate(sets) for b in sets[:i])
    assert sorted(p for role in roles for p in role['outer'])==list(range(200)) and len(ids)==391 and len(anchors)==200
    first=ungz(q.jobpath(jobs()[0])/'prediction_cache.json.gz');timings=[]
    for policy in FACTORS:
        start=time.monotonic()
        for old,new in zip(first['policies']['L0_legacy_decoder']['design'],first['policies']['W1_window_aware']['design']):
            row,stages,meta=factorial(old,new,anchors[old['patient']],policy,0.);diagnose(row,stages,meta)
        timings.append(time.monotonic()-start)
    assert time.time()-origin<B['preflight_seconds'] and shutil.disk_usage(RUN).free>=4*1024**3
    from brain_heart_contract import mock_tests
    contract=mock_tests();put(RUN/'data_contract_preflight.json',dict(time=now(),state='PASS',tests=contract,risk_fits=0,real_patient_risk_outputs=0))
    estimate=max(timings)*152/24*20/2*4+600;assert estimate<6000,'Conservative cache-audit timing gate'
    put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests+['patient_roles_disjoint','outer_partition200','391_historical_lead_items_unchanged','full_waveform_anchors200','cache_only_factorial_timing','risk_contract_mock','free_disk'],new_fits=0,model_forwards=0,checkpoint_loads=0,design_cache_factorial_timings=timings,conservative_seconds=estimate));return data

def worker(w):
    verify();data=load_data()
    for j in jobs():
        if j['index']%2==w:job(j,data)
    put(RUN/f'worker_{w}.json',dict(time=now(),pid=os.getpid(),state='finished'))

def main():
    assert not (RUN/'started.json').exists(),'Only one bounded A33 invocation';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['max_seconds'],pid=os.getpid(),max_seconds=B['max_seconds']));procs=[]
    try:
        freeze();preflight();verify();put(RUN/'execution_started.json',dict(time=now(),unix_time=time.time()))
        for w in range(2):
            log=(RUN/f'worker_{w}.log').open('w');p=subprocess.Popen([sys.executable,'-B','-u',__file__,'--worker',str(w)],stdout=log,stderr=subprocess.STDOUT);procs.append((p,log))
        put(RUN/'worker_handles.json',[dict(worker=w,pid=p.pid) for w,(p,_) in enumerate(procs)])
        while any(p.poll() is None for p,_ in procs):
            check();assert not any(p.poll() not in [None,0] for p,_ in procs),'Owned A33 worker failed';done=[read(p) for p in (RUN/'jobs').glob('*/done.json')];assert len(done)<=20 and size()<=B['max_output_bytes'],'A33 cache/output budget'
            put(RUN/'status.json',dict(time=now(),state='running_zero_fit_factorial_audit',completed_cache_jobs=len(done),completed_factual_replays=sum(x['factual_replays'] for x in done),completed_factorial_evaluations=sum(x['factorial_evaluations'] for x in done),max_cache_jobs=20,new_fits=0,epochs=0,model_forwards=0,SPH_admitted=False,new_figures=0,seconds_remaining=origin+7200-time.time()));time.sleep(5)
        assert all(p.returncode==0 for p,_ in procs) and len(list((RUN/'jobs').glob('*/done.json')))==20;put(RUN/'status.json',dict(time=now(),state='aggregation_independent_review',completed_cache_jobs=20,new_fits=0));import a33_analysis,a33_review
        a33_analysis.main();a33_review.main();verify();check();assert size()<=B['max_output_bytes'];result=dict(time=now(),state='zero_fit_factorial_audit_complete_review_PASS',cache_jobs=20,cached_lead_records=3040,factual_replays=40,factorial_evaluations=80,new_fits=0,epochs=0,model_forwards=0,source_hashes_unchanged=True,wall_seconds=time.time()-origin,no_policy_promoted=True,automatic_followon_training=False,SPH_admitted=False,RDB_scored=False,new_figures=0,NC_submission_ready=False,hospital_branch='waiting_data_contract_only',expert_reference='waiting');put(RUN/'completion.json',result);put(RUN/'status.json',result)
    except Exception:
        fail=dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),no_retry=True);put(RUN/'stop.json',fail);(RUN/'STOP').touch();put(RUN/'status.json',fail)
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
    if args.mock_preflight:print(json.dumps(dict(state='PASS',tests=synthetic_tests(),fits=0,model_forwards=0)))
    else:
        try:main() if args.worker is None else worker(args.worker)
        except Exception:
            if args.worker is not None:(RUN/'STOP').touch()
            raise
