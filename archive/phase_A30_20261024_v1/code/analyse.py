"""All fixed arms, both seeds and failures. No model promotion or tuning."""
import math
import numpy as np
import method as q
RUN=q.RUN;B=q.B

def main():
    import train as t
    t.check('analysis');summary=[];patient_tables={}
    for arm in B['arms']:
        for seed in B['seeds']:
            outs=[q.ungz(q.jobpath(j)/'result.json.gz') for j in q.jobs() if j['arm']==arm and j['seed']==seed];patients=[p for v in outs for p in v['patients']];assert len({p['patient'] for p in patients})==len(patients)
            for lead in ['ii','v1']:
                rec=[r for v in outs for r in v['records'] if r['lead']==lead];width=[w[k] for v in outs for w in v['widths'] if w['lead']==lead for k in ['onset_ms','offset_ms']]
                def stage(name):
                    rr=[r['tables'][name] for r in rec];nr=sum(v['reference'] for v in rr);nm=sum(v['matched'] for v in rr);npred=sum(v['predicted'] for v in rr);nu=sum(v['unknown_unmatched'] for v in rr)
                    return dict(reference=nr,matched=nm,predicted=npred,recall=nm/max(nr,1),precision_lower=nm/npred if npred else None,precision_upper=(nm+nu)/npred if npred else None,identity_matched=sum(v['identity_matched'] for v in rr),polarity={kind:dict(reference=sum(v['polarity'][kind]['reference'] for v in rr),matched=sum(v['polarity'][kind]['matched'] for v in rr),recall=sum(v['polarity'][kind]['matched'] for v in rr)/max(1,sum(v['polarity'][kind]['reference'] for v in rr)),patients=len({r['patient'] for r in rec if any(x['kind']==kind for x in r['reference_pol'])})) for kind in ['positive','negative','biphasic']})
                summary.append(dict(arm=arm,seed=seed,lead=lead,stages={k:stage(k) for k in ['raw','selected','reported','component_reported_DIAGNOSTIC']},reference_positive_patients=len(patients),strict_patient_joint_before=sum(v['strict_all'] for v in patients)/len(patients),strict_patient_joint_reported=sum(v['strict_reported_all'] for v in patients)/len(patients),median_width_ms=float(np.median(width)) if width else None,p95_width_ms=float(np.quantile(width,.95)) if width else None,finite_strict_calibration_folds=sum(v['calibration']['strict_q'] is not None for v in outs),design_feasible_folds=sum(v['design_feasible'] for v in outs),missing_same_beat=sum(v['identity_score']['missing'] for v in rec),calibration_n=[v['calibration']['n_reference_positive'] for v in outs],calibration_q=[v['calibration']['strict_q'] for v in outs]))
                for stage_name in ['selected','reported']:
                    table={}
                    for r in rec:
                        tab=r['tables'][stage_name];table[r['patient']]=np.array([tab['matched'],tab['reference'],tab['predicted']])
                    patient_tables[(arm,seed,lead,stage_name)]=table
    q.put(RUN/'analysis/summary.json',summary);get=lambda a,s,l:next(r for r in summary if r['arm']==a and r['seed']==s and r['lead']==l);gates=[];fair=[];pairwise=[]
    for seed in B['seeds']:
        checks={}
        for lead in ['ii','v1']:
            a=get(B['arms'][0],seed,lead);b=get(B['arms'][2],seed,lead);aa=a['stages']['reported'];bb=b['stages']['reported']
            checks[lead+'_useful_strict']=b['strict_patient_joint_before']>=.95 and b['strict_patient_joint_reported']>=.95 and bb['recall']>=.8 and b['median_width_ms'] is not None and b['median_width_ms']<=120 and b['p95_width_ms']<=200 and b['design_feasible_folds']==5
            checks[lead+'_noninferior']=aa['precision_lower'] is not None and bb['precision_lower'] is not None and bb['recall']>=aa['recall']-.02 and bb['precision_lower']>=aa['precision_lower']-.02
            matched=aa['recall']>=.8 and bb['recall']>=.8 and abs(aa['recall']-bb['recall'])<=.02;checks[lead+'_matched_coverage']=matched
            fair.append(dict(seed=seed,lead=lead,compared=['C2','C0'],recall_C0=aa['recall'],recall_C2=bb['recall'],matched_coverage=matched,advantage_claim=False,interpretation='No matched-coverage advantage established; inspect gates and uncertainty'))
        a=get(B['arms'][0],seed,'v1')['stages']['reported'];b=get(B['arms'][2],seed,'v1')['stages']['reported']
        for kind in ['negative','biphasic']:checks[kind+'_gain5pp']=a['predicted']>0 and b['predicted']>0 and b['polarity'][kind]['patients']>=10 and b['polarity'][kind]['recall']>=a['polarity'][kind]['recall']+.05
        gates.append(dict(seed=seed,checks=checks,pass_all=all(checks.values())))
        for lead in ['ii','v1']:
            for stage in ['selected','reported']:
                for before,after in [(B['arms'][0],B['arms'][1]),(B['arms'][1],B['arms'][2]),(B['arms'][0],B['arms'][2])]:
                    t.check('analysis');a=patient_tables[(before,seed,lead,stage)];b=patient_tables[(after,seed,lead,stage)];ids=sorted(a);assert ids==sorted(b);aa=np.stack([a[i] for i in ids]);bb=np.stack([b[i] for i in ids]);rng=np.random.default_rng(seed+len(pairwise));idx=rng.integers(0,len(ids),(1000,len(ids)));ra=aa[idx].sum(1);rb=bb[idx].sum(1);dr=rb[:,0]/np.maximum(rb[:,1],1)-ra[:,0]/np.maximum(ra[:,1],1);valid=(ra[:,2]>0)&(rb[:,2]>0);dp=rb[valid,0]/rb[valid,2]-ra[valid,0]/ra[valid,2];precision_defined=(aa[:,2].sum()>0 and bb[:,2].sum()>0)
                    pairwise.append(dict(seed=seed,lead=lead,stage=stage,before=before,after=after,patients=len(ids),bootstrap=1000,precision_defined_bootstrap=int(valid.sum()),recall_delta=float(bb[:,0].sum()/max(bb[:,1].sum(),1)-aa[:,0].sum()/max(aa[:,1].sum(),1)),recall_CI95=np.quantile(dr,[.025,.975]).tolist(),precision_delta=float(bb[:,0].sum()/bb[:,2].sum()-aa[:,0].sum()/aa[:,2].sum()) if precision_defined else None,precision_CI95=np.quantile(dp,[.025,.975]).tolist() if precision_defined and len(dp) else None,descriptive_only=True))
    q.put(RUN/'analysis/gates.json',dict(seeds=gates,both_seed_primary_pass=all(v['pass_all'] for v in gates),all_wave_target_preserved=True,clinical_noP_validated=False,AF_increment_validated=False,NC_submission_ready=False));q.put(RUN/'analysis/matched_coverage.json',fair);q.put(RUN/'analysis/patient_bootstrap.json',pairwise)
    old=q.read(q.A29/'analysis/summary.json');q.put(RUN/'analysis/historical_C0_context.json',dict(source=str(q.A29/'analysis/summary.json'),context_only_not_matched_training=True,rows=[v for v in old if v['arm']=='C0_complete' and v['policy']=='B3_boundary']))
    q.put(RUN/'analysis/decision.json',dict(time=q.now(),state='bounded_mechanistic_training_finished',measurement_gate_pass=all(v['pass_all'] for v in gates),automatic_followon_training=False,independent_sources_admitted=0,NC_submission_ready=False,AF_increment_validated=False,brain_stroke_risk_validated=False,SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting',expert_reference='waiting'))
    write_report(summary,gates)

def write_report(summary,gates):
    passed=all(v['pass_all'] for v in gates)
    lines=['A30固定小预算机制重训结果（自动生成，需结合核验记录阅读）','30作业×8epoch=240epoch；五折、双种子；所有失败/不合格保留。','C0是A27 C2旧结构化分配机制复现，不是重命名A27 C0。','旧帧级C0作为历史背景，不作匹配训练因果对照。','以下点召回/精度均先区分区间拒绝前与最终报告，不能互换。']
    for r in summary:
        if r['lead']!='v1':continue
        s=r['stages']['selected'];p=r['stages']['reported'];lines.append(f"{r['arm']} seed{r['seed']}: raw={r['stages']['raw']['recall']:.4f}; selected={s['recall']:.4f}; selected_precision={s['precision_lower']}; negative={s['polarity']['negative']['recall']:.4f}; biphasic={s['polarity']['biphasic']['recall']:.4f}; reported_recall={p['recall']:.4f}; patient_ALL_before/after={r['strict_patient_joint_before']:.4f}/{r['strict_patient_joint_reported']:.4f}; width_median_ms={r['median_width_ms']}; width_p95_ms={r['p95_width_ms']}; q={r['calibration_q']}")
    lines+=['两种子全部主门通过='+str(passed),'通过工程/统计核验不等于可靠性达标；未通过的门不得删除或事后修改。','不删极端患者；缺失记为无限严格残差；不跨心搏借框、不复用候选。','本研究是已反复查看LUDB上的开发检验，不是独立确认队列。','严格独立来源准入仍0；SPH/RDB不评分；医院及专家资料继续等待。','AF识别/P测量不能替代未来AF、脑损伤或卒中风险预测证据。','无新图件、无AF拟合；不自动追加训练；NC投稿就绪仍为否。']
    (RUN/'A30结果与下一步.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
