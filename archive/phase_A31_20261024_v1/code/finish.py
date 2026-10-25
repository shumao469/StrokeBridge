"""Fixed aggregate and independent geometry audit; no policy selection."""
import math
import numpy as np
import audit as a
RUN=a.RUN;B=a.B;q=a.q;d=a.d;v=a.ind

def independent_selected(r,choice,policy):
    ps=[p for p in r['raw'] if (choice['mode']=='union' or choice['mode']==p['kind']) and p['confidence']>=choice['threshold'] and p['wave_R'] is not None]
    if policy=='O1_ordered_restriction':ps=[p for p in ps if p['kind']=='ordered']
    if policy=='C2_continuous_restriction':ps=[p for p in ps if p['kind']=='continuous']
    groups={}
    for p in ps:groups.setdefault(p['wave_R'],[]).append(p)
    out=[]
    for rr,items in sorted(groups.items()):
        kept=[]
        if policy=='S3_support_first':kept=items
        else:
            for p in sorted(items,key=lambda z:z['confidence'],reverse=True):
                def ov(t):
                    return max(0,min(p['offset'],t['offset'])-max(p['onset'],t['onset']))/max(max(p['offset'],t['offset'])-min(p['onset'],t['onset']),1)
                if all(ov(t)<.5 for t in kept):kept.append(p)
        key=lambda z:(z['confidence']*math.sqrt((z['offset']-z['onset'])/100),z['confidence'],z['offset']-z['onset'],-z['uid'])
        out.append(max(kept,key=key))
    return out

def review():
    counts=dict(outer_records=0,calibration_records=0,patient_entries=0);extreme=[];trace_rows=[]
    for j in q.jobs():
        a.check();p=RUN/'jobs'/f"job_{j['index']:02d}";cache=a.ungz(q.jobpath(j)/'prediction_cache.json.gz');original=a.ungz(q.jobpath(j)/'result.json.gz')
        for policy in B['policies']:
            out=a.ungz(p/(policy+'.json.gz'));c=out['calibration'];expected={r['patient'] for r in cache['calibration'] if r['ref']};assert {z['patient'] for z in c['ledger']}==expected
            for pat in c['ledger']:
                rows=[r for r in cache['calibration'] if r['patient']==pat['patient'] and r['ref']];scores=[];components=[];missing=0
                for r in rows:
                    ps=independent_selected(r,out['choice'],policy);ref=r['ref'];qs=r['qrs'];edges=[[v.allowed(z,t,qs) for t in ps] for z in ref];n=v.cardinality(edges);missing+=len(ref)-n
                    if n<len(ref):score=float('inf')
                    else:
                        vals=sorted({v.dist(z,t) for z in ref for t in ps if v.allowed(z,t,qs)});score=next(x for x in vals if v.cardinality([[edges[i][k] and v.dist(z,t)<=x for k,t in enumerate(ps)] for i,z in enumerate(ref)])==len(ref))
                    row=next(z for z in pat['rows'] if z['lead']==r['lead']);assert v.same(row['strict'],score);assert len(row['pairs'])==n and len({i for i,k in row['pairs']})==n and len({k for i,k in row['pairs']})==n;assert all(v.allowed(ref[i],ps[k],qs) for i,k in row['pairs']);comp=max((v.dist(ref[i],ps[k]) for i,k in row['pairs']),default=float('inf'));assert v.same(row['component'],comp);components.append(comp);scores.append(score);counts['calibration_records']+=1
                assert pat['missing']==missing and v.same(pat['strict'],max(scores)) and v.same(pat['component'],max(components))
            n=len(c['ledger']);k=math.ceil((n+1)*.95);assert c['n_reference_positive']==n and c['k']==k
            for key in ['strict','component']:
                ss=sorted(float('inf') if z[key] is None else z[key] for z in c['ledger']);qq=ss[k-1] if k<=n else float('inf');assert v.same(c[key+'_q'],qq)
            assert c['used_q']==c['strict_q']
            for r,stored in zip(cache['outer'],out['records']):
                ps=independent_selected(r,out['choice'],policy);assert ps==stored['selected'],'Independent policy reconstruction';assert len({z['wave_R'] for z in ps})==len(ps);ref=r['ref'];qs=r['qrs'];expanded=v.expand(ps,c['used_q']);reported=[z for z in expanded if max(z['box'][1]-z['box'][0],z['box'][3]-z['box'][2])<=100]
                assert v.boundary_count(ref,expanded,qs)==stored['strict_unique_covered'] and v.boundary_count(ref,reported,qs)==stored['strict_unique_reported']
                for name,pp in [('raw',r['raw']),('selected',ps),('reported',reported)]:
                    assert stored['tables'][name]['matched']==v.point_count(ref,pp,qs) and stored['tables'][name]['identity_matched']==v.point_count(ref,pp,qs,True)
                counts['outer_records']+=1
            for pat in out['patients']:
                rows=[r for r in out['records'] if r['patient']==pat['patient']];assert pat['strict_all']==all(r['strict_unique_covered']==r['reference'] for r in rows) and pat['strict_reported_all']==all(r['strict_unique_reported']==r['reference'] for r in rows);counts['patient_entries']+=1
        extreme.extend(dict(job=j,**z) for z in a.read(p/'extreme_patients.json'))
        if (p/'provenance_trace.json.gz').exists():
            obj=a.ungz(p/'provenance_trace.json.gz');trace_rows.extend(dict(job=j,**r) for r in obj['records'])
    a.put(RUN/'post_review/contract_review.json',dict(time=a.now(),state='PASS',independent_matcher='augmenting paths',independent_policy_reconstruction=True,all_calibration_patients_retained=True,extreme_patients_retained=True,source_hashes_unchanged=True,new_fits=0,new_epochs=0,**counts))
    a.put(RUN/'analysis/extreme_patients.json',extreme);a.gz(RUN/'analysis/provenance_cases.json.gz',trace_rows)
    return extreme,trace_rows

def aggregate(extreme,trace_rows):
    summary=[];stages=[];events=[];branches=[]
    for j in q.jobs():
        p=RUN/'jobs'/f"job_{j['index']:02d}"
        for r in a.ungz(p/'outer_diagnostics.json.gz'):
            for z in r['stages']:stages.append(dict(job=j,patient=r['patient'],lead=r['lead'],**z))
            events.extend(dict(job=j,**z) for z in r['events']);branches.append(dict(job=j,patient=r['patient'],lead=r['lead'],raw=r['branch_raw'],admitted=r['branch_admitted'],selected=r['selected_branch_counts']))
    for arm in q.B['arms']:
        for seed in q.B['seeds']:
            for policy in B['policies']:
                outs=[a.ungz(RUN/'jobs'/f"job_{j['index']:02d}"/(policy+'.json.gz')) for j in q.jobs() if j['arm']==arm and j['seed']==seed];patients=[p for o in outs for p in o['patients']];assert len({p['patient'] for p in patients})==len(patients)
                for lead in ['ii','v1']:
                    rr=[r for o in outs for r in o['records'] if r['lead']==lead];widths=[w[k] for o in outs for w in o['widths'] if w['lead']==lead for k in ['onset_ms','offset_ms']]
                    def stage(name):
                        tt=[r['tables'][name] for r in rr];nr=sum(t['reference'] for t in tt);nm=sum(t['matched'] for t in tt);nn=sum(t['predicted'] for t in tt)
                        return dict(reference=nr,matched=nm,predicted=nn,recall=nm/max(nr,1),annotation_precision=nm/nn if nn else None,negative_recall=sum(t['polarity']['negative']['matched'] for t in tt)/max(1,sum(t['polarity']['negative']['reference'] for t in tt)),biphasic_recall=sum(t['polarity']['biphasic']['matched'] for t in tt)/max(1,sum(t['polarity']['biphasic']['reference'] for t in tt)),negative_patients=len({r['patient'] for r in rr if any(p['kind']=='negative' for p in r['reference_pol'])}),biphasic_patients=len({r['patient'] for r in rr if any(p['kind']=='biphasic' for p in r['reference_pol'])}))
                    summary.append(dict(arm=arm,seed=seed,policy=policy,lead=lead,selected=stage('selected'),reported=stage('reported'),patient_reference=len(patients),patient_ALL_before=sum(p['strict_all'] for p in patients)/len(patients),patient_ALL_reported=sum(p['strict_reported_all'] for p in patients)/len(patients),median_width_ms=float(np.median(widths)) if widths else None,p95_width_ms=float(np.quantile(widths,.95)) if widths else None,design_feasible_folds=sum(o['design_feasible'] for o in outs),calibration_q=[o['calibration']['strict_q'] for o in outs],diagnostic_not_promoted=policy!='F0_factual'))
    a.put(RUN/'analysis/summary.json',summary);a.gz(RUN/'analysis/stage_records.json.gz',stages);a.gz(RUN/'analysis/reference_events.json.gz',events);a.gz(RUN/'analysis/branch_records.json.gz',branches)
    triage=[]
    for seed in q.B['seeds']:
        rows=[]
        for fold in q.B['folds']:
            candidates=[z for z in extreme if z['job']['arm']=='C2_balanced_tail' and z['job']['seed']==seed and z['job']['fold']==fold and z['role']=='calibration'];assert len(candidates)==3
            z=candidates[0];rows.append(dict(fold=fold,patient=z['patient'],patient_strict=z['patient_strict'],reason=z['worst_event']['reason'],branch=z['worst_event']['selected']['kind'] if z['worst_event']['selected'] else None,whole_patient_training_candidate=z['worst_event']['boundary_training_candidate'],event=z['worst_event']))
        triage.append(dict(seed=seed,eligible_folds=sum(z['whole_patient_training_candidate'] for z in rows),minimum_required=3,rows=rows))
    recommend=all(z['eligible_folds']>=3 for z in triage)
    trace=[z for r in trace_rows for z in r['trace']];accepted=[z for z in trace if z['discard'] is None]
    a.put(RUN/'analysis/provenance_summary.json',dict(model_count=10,lead_case_count=len(trace_rows),grown_components=len(trace),accepted_components=len(accepted),clipped_accepted=sum(z['clipping_changed'] for z in accepted),generating_vs_assigned_R_different=sum(z['generating_R']!=z['assigned_R'] for z in accepted),clipping_discarded_duration=sum(z['discard']=='duration' and z['clipping_changed'] for z in trace),outcome_enriched_not_population_rates=True,new_fits=0))
    get=lambda arm,seed,policy,lead:next(z for z in summary if z['arm']==arm and z['seed']==seed and z['policy']==policy and z['lead']==lead);gates=[]
    for policy in B['policies']:
        for seed in q.B['seeds']:
            checks={}
            for lead in ['ii','v1']:
                baseline=get(q.B['arms'][0],seed,policy,lead);b=get(q.B['arms'][2],seed,policy,lead);aa=baseline['reported'];bb=b['reported'];checks[lead+'_strict_useful']=b['patient_ALL_before']>=.95 and b['patient_ALL_reported']>=.95 and bb['recall']>=.8 and b['median_width_ms'] is not None and b['median_width_ms']<=120 and b['p95_width_ms']<=200 and b['design_feasible_folds']==5;checks[lead+'_noninferior']=aa['annotation_precision'] is not None and bb['annotation_precision'] is not None and bb['recall']>=aa['recall']-.02 and bb['annotation_precision']>=aa['annotation_precision']-.02;checks[lead+'_matched_coverage']=aa['recall']>=.8 and bb['recall']>=.8 and abs(aa['recall']-bb['recall'])<=.02
            aa=get(q.B['arms'][0],seed,policy,'v1')['reported'];bb=get(q.B['arms'][2],seed,policy,'v1')['reported']
            for kind in ['negative','biphasic']:checks[kind+'_gain5pp']=aa['predicted']>0 and bb['predicted']>0 and bb[kind+'_patients']>=10 and bb[kind+'_recall']>=aa[kind+'_recall']+.05
            gates.append(dict(policy=policy,seed=seed,checks=checks,all_pass=all(checks.values()),not_promoted=True))
    a.put(RUN/'analysis/gates.json',dict(entries=gates,strict_original_gates_preserved=True,all_policies_diagnostic_not_promoted=True))
    decision=dict(time=a.now(),state='audit_complete_not_effectiveness_pass',whole_patient_pilot_recommended_by_frozen_triage=recommend,triage=triage,decision='separately_budgeted_whole_patient_pilot_candidate' if recommend else 'prioritise_selection_score_anchor_window_contract_before_more_training',automatic_retraining=False,new_fits=0,new_epochs=0,SPH_admitted=False,RDB_scored=False,new_figures=0,independent_sources_admitted=0,NC_submission_ready=False,hospital_branch='waiting',expert_reference='waiting');a.put(RUN/'analysis/decision.json',decision);report(summary,decision)

def report(summary,decision):
    text=['A31零拟合训练目标—解码—最终报告失配审计','全部旧结果保留；本轮没有重训、阈值搜索、来源放行或图件。','F0原政策；O1/C2仅限制原已准入源；S3同一原阈值下把支持量排序移到NMS之前。','所有反事实只用于诊断，不能据评价集选最优后宣称独立方法验证。','严格覆盖同时考察II/V1全部参考P；极端患者保留，缺失记无限残差。','连续候选共享帧级监督，不能称其完全未训练；直接端点尾部损失作用于有序后验。']
    for r in summary:
        if r['arm']!='C2_balanced_tail' or r['lead']!='v1':continue
        text.append(f"seed{r['seed']} {r['policy']}: selected_recall={r['selected']['recall']:.4f}; annotation_precision={r['selected']['annotation_precision']}; reported_recall={r['reported']['recall']:.4f}; patient_ALL_before/after={r['patient_ALL_before']:.4f}/{r['patient_ALL_reported']:.4f}; width_median_ms={r['median_width_ms']}")
    for z in decision['triage']:text.append(f"种子{z['seed']}：符合预冻结整患者边界监督候选条件{z['eligible_folds']}/5折；要求每种子至少3折。")
    text+=['冻结分流建议='+decision['decision'],'此分流是机制诊断启发式，不是统计成功检验或重训一定有效的证明。','所有反事实的未达标、无法匹配覆盖及无输出均保留；不删除患者或放宽任何可靠测量门。','详细分阶段、分支、窗口、替代候选及极端病例见analysis及jobs；受限溯源是富集诊断样本，不能算独立验证。','独立来源准入仍无新增；SPH/RDB不评分；医院/专家等待；不支持未来AF、脑损伤或卒中风险预测结论。','NC投稿就绪仍为否；任何下一轮训练需另行预算，本轮不会自动启动。']
    (RUN/'A31审计结果与下一步.txt').write_text('\n'.join(text)+'\n',encoding='utf8')

def main():
    a.check();extreme,trace=review();aggregate(extreme,trace);a.verify();a.check()
