"""Necessary-condition gate BEFORE endpoint-only fitting; no model loaded/created."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,time,math,traceback,shutil
from pathlib import Path
import numpy as np
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A33=BASE/'runs/phase_A33_20261024_v1'
sys.path.insert(0,str(A33/'code'));import a33_audit as prior
q=prior.q;d=q.d;read=q.read;put=q.put;ungz=q.ungz;gz=q.gz;sha=q.sha;now=q.now
B=read(RUN/'budget.json');STAGES=['raw_identity','selected_identity','threshold_identity','ideal_confidence_identity']

def jobs():return [j for j in q.jobs() if j['arm']=='C1_patient_report_aligned']
def path(j):return RUN/'jobs'/f"job_{j['index']:02d}"
def check():
    assert not (RUN/'STOP').exists(),'A34 STOP'
    start=read(RUN/'started.json');assert time.time()<start['deadline_unix'],'A34 fixed total deadline';assert time.time()-start['unix_time']<B['preflight_seconds'],'A34 preflight600s cap'
def verify():
    for f,h in read(RUN/'source_lock.json').items():check();assert sha(f)==h,f'A34 source mismatch:{f}'
def upper(p):
    if p['kind']=='continuous':return 1.
    assert p['kind']=='ordered' and p.get('non_null_mass') is not None
    value=float(p['non_null_mass']);assert 0<=value<=1 and p['confidence']<=value+1e-12
    return value
def pool(row,stage,cut):
    if stage=='selected_identity':return row['selected']
    if stage=='raw_identity':return row['raw']
    if stage=='threshold_identity':return [p for p in row['raw'] if p['confidence']>=cut]
    assert stage=='ideal_confidence_identity';return [p for p in row['raw'] if upper(p)>=cut]
def graph(ref,ps,qs):
    slots=sorted({p['wave_R'] for p in ps if p.get('wave_R') is not None});edges=[[any(p.get('wave_R')==r and d.compatible(t,p,qs) for p in ps) for r in slots] for t in ref];n=len(d.assign(np.array(edges,dtype=float))) if ref and slots else 0
    return dict(slots=slots,edges=edges,matched=n)
def grouped(records,stage):
    patients=[]
    for p in sorted({r['patient'] for r in records}):
        rr=[r for r in records if r['patient']==p];nr=sum(len(r['reference']) for r in rr)
        if nr==0:continue
        nm=sum(r['stages'][stage]['matched'] for r in rr);patients.append(dict(patient=p,record=rr[0]['record'],reference=nr,matched=nm,missing=nr-nm))
    return patients

def job(j):
    check();out=path(j);assert not out.exists(),'A34 no overwrite/retry';out.mkdir(parents=True);source=prior.path(j);cache=ungz(source/'candidate_cache.json.gz');data=cache['policies']['F01_old_unclipped'];result=ungz(source/'result_F01_old_unclipped.json.gz');cut=result['choice']['threshold'];rows=[]
    for role in ['calibration','outer']:
        for row in data[role]:
            check();stages={stage:graph(row['ref'],pool(row,stage,cut),row['qrs']) for stage in STAGES}
            assert stages['raw_identity']['matched']>=stages['ideal_confidence_identity']['matched']>=stages['threshold_identity']['matched']
            assert stages['threshold_identity']['slots']==stages['selected_identity']['slots'],'Fixed threshold identity R slots must survive one-winner selection'
            rows.append(dict(role=role,patient=row['patient'],record=row['record'],lead=row['lead'],reference=row['ref'],qrs=row['qrs'],list_candidates=row['raw'],stages=stages))
    summary=dict(job=j,cutoff=cut,calibration={},outer={})
    for stage in STAGES:
        cal=grouped([r for r in rows if r['role']=='calibration'],stage);n=len(cal);k=math.ceil((n+1)*.95);miss=sum(p['missing']>0 for p in cal);qi=(k>n or miss>n-k);outer=grouped([r for r in rows if r['role']=='outer'],stage)
        summary['calibration'][stage]=dict(n=n,k=k,infinite_patients=miss,quantile_infinite=qi,patients=cal)
        summary['outer'][stage]=dict(positive_patients=len(outer),covered_patients=sum(p['missing']==0 for p in outer),patients=outer)
    assert len(rows)==128
    gz(out/'input_data.json.gz',dict(job=j,choice=result['choice'],calibration=data['calibration'],outer=data['outer']));gz(out/'identity_graphs.json.gz',rows);put(out/'summary.json',summary);put(out/'done.json',dict(time=now(),job=j,cached_lead_records=128,identity_graphs=512,fits=0,epochs=0,model_forwards=0,checkpoint_loads=0));return summary

def freeze():
    assert read(A33/'completion.json')['source_hashes_unchanged'] and read(A33/'post_review/contract_review.json')['state']=='PASS';locks=read(A33/'source_lock.json');extra=[RUN/'budget.json',RUN/'protocol.txt',*sorted((RUN/'code').glob('*.py')),A33/'source_lock.json',A33/'completion.json',A33/'post_review/contract_review.json',A33/'analysis/necessary_condition_audit.json']
    for j in jobs():extra.extend(prior.path(j)/name for name in ['candidate_cache.json.gz','result_F01_old_unclipped.json.gz','done.json'])
    for p in extra:check();locks[str(p)]=sha(p)
    put(RUN/'source_lock.json',locks);verify();put(RUN/'protocol_frozen.json',dict(time=now(),source_lock_SHA256=sha(RUN/'source_lock.json'),protocol_SHA256=sha(RUN/'protocol.txt'),budget=B,selection_semantics='same_formula_not_frozen_winner',training_budget_released=0))

def synthetic():
    def p(r,kind='ordered',pn=.4,confidence=.3):return dict(kind=kind,wave_R=r,expert_beat=0,confidence=confidence,non_null_mass=pn)
    refs=[[40,50,60],[70,80,90]];rr=dict(raw=[p(150),p(150)],selected=[p(150)])
    assert graph(refs,rr['raw'],[150])['matched']==1
    assert pool(rr,'ideal_confidence_identity',.7)==[]
    assert len(pool(dict(raw=[p(150,'continuous')]),'ideal_confidence_identity',.7))==1
    assert math.ceil(23*.95)==22 and 3>22-22
    return ['R_slot_capacity_one','ordered_PN_absolute_confidence_upper','continuous_optimistic_upper_one','finite_sample_max_infinite_gate']

def aggregate(results):
    seeds=[];badfolds=[];all_baseline_reject=True
    for seed in [2026102701,2026102702]:
        rr=[r for r in results if r['job']['seed']==seed];assert len(rr)==5;positive=sum(r['outer']['ideal_confidence_identity']['positive_patients'] for r in rr);blocked=0;combined=0
        for r in rr:
            c=r['calibration']['ideal_confidence_identity'];o=r['outer']['ideal_confidence_identity']
            if c['quantile_infinite']:
                blocked+=o['positive_patients'];badfolds.append(dict(job=r['job'],cutoff=r['cutoff'],n=c['n'],k=c['k'],infinite_patients=c['infinite_patients'],missing_patients=[p for p in c['patients'] if p['missing']>0],blocked_outer_positive_patients=o['positive_patients']))
            else:combined+=o['covered_patients']
            parent=ungz(prior.path(r['job'])/'result_F01_old_unclipped.json.gz');all_baseline_reject=all_baseline_reject and not parent['widths']
        assert positive==176;seeds.append(dict(seed=seed,positive_patients=positive,calibration_blocked_outer_patients=blocked,calibration_induced_patient_ceiling=(positive-blocked)/positive,combined_ideal_patient_ceiling=combined/positive,admissible=combined/positive>=.95))
    eligible=all(r['admissible'] for r in seeds);out=dict(time=now(),seeds=seeds,blocked_folds=badfolds,training_admissible=eligible,actual_fits=0,automatic_training=False,paired_C0_all_reject=all_baseline_reject,matched_coverage_advantage_estimable=False,upper_bound_is_NOT_model_performance=True)
    put(RUN/'analysis/feasibility.json',out);return out

def closeout(feas):
    lines=['A34最终端点单模块可行性门控与收尾','本轮只读缓存和数学必要条件；没有实例化/加载/训练网络。','保持选择公式而非强制原winner：允许端点改变后按原式重算meanP，但ordered置信度仍<=原PN、continuous<=1；原阈值和身份不变。']
    for r in feas['seeds']:lines.append(f"seed{r['seed']}: calibration-blocked outer positives={r['calibration_blocked_outer_patients']}/{r['positive_patients']}; even-all-other-folds-perfect ceiling={r['calibration_induced_patient_ceiling']:.6f}; combined optimistic identity ceiling={r['combined_ideal_patient_ceiling']:.6f}")
    for b in feas['blocked_folds']:
        lines.append(f"blocked seed/fold={b['job']['seed']}/{b['job']['fold']}; cutoff={b['cutoff']}; n/k={b['n']}/{b['k']}; unavoidable missing cal patients={b['infinite_patients']}; outer positives={b['blocked_outer_positive_patients']}")
        for p in b['missing_patients']:lines.append(f" {p['record']}: identity bound {p['matched']}/{p['reference']} reference P")
    lines+=['训练必要条件通过='+str(feas['training_admissible']),'若必要条件未通过：科学性停止、释放拟合预算0；不以失败训练浪费计算或事后换检测器/改PN/阈值。','此不证明任何可能的ECG方法都不可行；结论只适用于冻结父模型、候选来源、PN、身份和验收条件下的坐标修正。','固定C0原F01全拒绝，因此完整匹配覆盖/最终精度非劣结论本来也不能从相对全拒绝改善中获得。','可立即收尾的是已完成的工程/方法审计、负向消融及证据缺口；不能把它写成NC级可靠测量/独立泛化/脑心未来风险成功。','本收尾周期不自动开启A35或更多模型训练。若以后增加新的检测置信、候选生成或独立来源，必须重新立项，而不是偷换纯端点模块。','脑心未来风险仅已有数据合同与合成单测；实际事件、死亡及有效随访不足，医院/专家分支继续等待。','SPH/RDB不放行、不评分、不删极端、不放宽门槛、不画图。'];(RUN/'A34可行性与收尾结论.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')

def main():
    assert not (RUN/'started.json').exists(),'One A34 bounded invocation';origin=time.time();put(RUN/'started.json',dict(time=now(),unix_time=origin,deadline_unix=origin+B['max_seconds'],pid=os.getpid(),max_seconds=B['max_seconds']))
    try:
        tests=synthetic();freeze();roles=read(q.RUN/'patient_roles.json')
        for role in roles:
            sets=[set(role[k]) for k in ['train','design','calibration','outer']];assert list(map(len,sets))==[124,12,24,40] and all(not x&y for i,x in enumerate(sets) for y in sets[:i])
        assert sorted(p for r in roles for p in r['outer'])==list(range(200)) and shutil.disk_usage(RUN).free>=2*1024**3
        results=[]
        for j in jobs():
            results.append(job(j));put(RUN/'status.json',dict(time=now(),state='running_zero_fit_necessary_condition_preflight',completed_parent_models=len(results),max_parent_models=10,fits=0,epochs=0,model_forwards=0))
        feas=aggregate(results);import a34_review
        a34_review.main();verify();check();assert sum(p.stat().st_size for p in RUN.rglob('*') if p.is_file())<=B['max_output_bytes'];closeout(feas)
        state='SCIENTIFIC_STOP_ENDPOINT_ONLY_NOT_ADMISSIBLE' if not feas['training_admissible'] else 'FEASIBILITY_PASS_TRAINING_NOT_RELEASED_PENDING_EXECUTOR_FREEZE'
        put(RUN/'preflight.json',dict(time=now(),state='NOT_ADMISSIBLE' if not feas['training_admissible'] else 'PASS_NECESSARY_ONLY',tests=tests+['roles_disjoint','all_patients_retained','independent_R_graph_matching','conservative_confidence_upper_bounds','exact_finite_sample_infinity_gate'],training_admissible=feas['training_admissible'],fits=0,model_forwards=0,source_hashes_unchanged=True))
        put(RUN/'training_stop.json',dict(time=now(),state=state,released_fits=0,reason='Existing identity/PN/cutoff imply an infinite calibration fold and reported patient ceiling below95%' if not feas['training_admissible'] else 'Necessary condition only; no unreviewed executor is permitted',no_automatic_next_phase=True))
        completion=dict(time=now(),state=state,execution_review='PASS',training_admissible=feas['training_admissible'],parent_models=10,cached_lead_records=1280,identity_graphs=5120,fits=0,epochs=0,model_forwards=0,checkpoint_loads=0,source_hashes_unchanged=True,wall_seconds=time.time()-origin,closeout_cycle_complete=True,automatic_next_phase=False,NC_submission_ready=False,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting');put(RUN/'completion.json',completion);put(RUN/'status.json',completion)
    except Exception:
        failure=dict(time=now(),state='FAIL_STOP',traceback=traceback.format_exc(),no_retry=True);put(RUN/'stop.json',failure);(RUN/'STOP').touch();put(RUN/'status.json',failure);raise

if __name__=='__main__':
    if '--mock-preflight' in sys.argv:print(dict(state='PASS',tests=synthetic(),fits=0,model_forwards=0))
    else:main()
