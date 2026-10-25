"""All fixed cells, all failed gates, R-capacity ceilings. No policy selection."""
import math
from collections import Counter
import numpy as np
import a33_audit as a
RUN=a.RUN;B=a.B;q=a.q

def main():
    summary=[];tables={};diagnostics={};failure_rows=[]
    for policy in B['policies']:
        for arm in B['arms']:
            for seed in B['seeds']:
                a.check();jj=[j for j in a.jobs() if j['arm']==arm and j['seed']==seed];outs=[a.ungz(a.path(j)/('result_'+policy+'.json.gz')) for j in jj];patients=[p for o in outs for p in o['patients']];assert len(patients)==176 and len({p['patient'] for p in patients})==176
                for lead in ['ii','v1']:
                    rec=[r for o in outs for r in o['records'] if r['lead']==lead];assert len(rec)==200 and len({r['patient'] for r in rec})==200;width=[w[k] for o in outs for w in o['widths'] if w['lead']==lead for k in ['onset_ms','offset_ms']]
                    def stage(name):
                        rr=[r['tables'][name] for r in rec];nr=sum(r['reference'] for r in rr);nm=sum(r['matched'] for r in rr);pred=sum(r['predicted'] for r in rr);unknown=sum(r['unknown_unmatched'] for r in rr)
                        return dict(reference=nr,matched=nm,predicted=pred,recall=nm/max(1,nr),precision_lower=nm/pred if pred else None,precision_upper=(nm+unknown)/pred if pred else None,identity_matched=sum(r['identity_matched'] for r in rr),polarity={k:dict(reference=sum(r['polarity'][k]['reference'] for r in rr),matched=sum(r['polarity'][k]['matched'] for r in rr),recall=sum(r['polarity'][k]['matched'] for r in rr)/max(1,sum(r['polarity'][k]['reference'] for r in rr)),patients=len({r['patient'] for r in rec if any(v['kind']==k for v in r['reference_pol'])})) for k in ['positive','negative','biphasic']})
                    summary.append(dict(policy=policy,arm=arm,seed=seed,lead=lead,stages={k:stage(k) for k in ['raw','selected','reported','component_reported_DIAGNOSTIC']},reference_positive_patients=176,strict_patient_joint_before=sum(p['strict_all'] for p in patients)/176,strict_patient_joint_reported=sum(p['strict_reported_all'] for p in patients)/176,median_width_ms=float(np.median(width)) if width else None,p95_width_ms=float(np.quantile(width,.95)) if width else None,design_feasible_folds=sum(o['design_feasible'] for o in outs),calibration_q=[o['calibration']['strict_q'] for o in outs],calibration_n=[o['calibration']['n_reference_positive'] for o in outs],missing_same_beat=sum(r['identity_score']['missing'] for r in rec)))
                    for st in ['selected','reported']:tables[(policy,arm,seed,lead,st)]={r['patient']:np.array([r['tables'][st][k] for k in ['matched','reference','predicted']]) for r in rec}
                if policy in a.FACTORS:
                    dd=[r for j in jj for r in a.ungz(a.path(j)/'oracle_diagnostics.json.gz') if r['policy']==policy and r['role']=='outer'];assert len(dd)==400 and len({(r['patient'],r['lead']) for r in dd})==400;diagnostics[(policy,arm,seed)]=dd
                    for lead in ['ii','v1']:
                        rr=[r for r in dd if r['lead']==lead];counts=Counter(f['reason'] for r in rr for f in r['failures']);failure_rows.append(dict(policy=policy,arm=arm,seed=seed,lead=lead,reference=sum(r['reference'] for r in rr),identity_point_misses=sum(counts.values()),reasons=dict(counts),unresolved_raw=sum(r['unresolved_raw'] for r in rr),extra_ordered_R_candidates=sum(len(r['extra_ordered_R_candidates']) for r in rr)))
    a.put(RUN/'analysis/summary.json',summary);get=lambda p,arm,s,l:next(r for r in summary if r['policy']==p and r['arm']==arm and r['seed']==s and r['lead']==l);gates=[];fair=[]
    for policy in B['policies']:
        for seed in B['seeds']:
            checks={}
            for lead in ['ii','v1']:
                c=get(policy,B['arms'][0],seed,lead);n=get(policy,B['arms'][1],seed,lead);cc=c['stages']['reported'];nn=n['stages']['reported'];checks[lead+'_useful_strict']=n['strict_patient_joint_before']>=.95 and n['strict_patient_joint_reported']>=.95 and nn['recall']>=.8 and n['median_width_ms'] is not None and n['median_width_ms']<=120 and n['p95_width_ms']<=200 and n['design_feasible_folds']==5;checks[lead+'_noninferior']=cc['precision_lower'] is not None and nn['precision_lower'] is not None and nn['recall']>=cc['recall']-.02 and nn['precision_lower']>=cc['precision_lower']-.02;matched=cc['recall']>=.8 and nn['recall']>=.8 and abs(cc['recall']-nn['recall'])<=.02;checks[lead+'_matched_coverage']=matched;fair.append(dict(policy=policy,seed=seed,lead=lead,C0_recall=cc['recall'],C1_recall=nn['recall'],matched_coverage=matched,advantage_claim=False))
            c=get(policy,B['arms'][0],seed,'v1')['stages']['reported'];n=get(policy,B['arms'][1],seed,'v1')['stages']['reported']
            for kind in ['negative','biphasic']:checks[kind+'_gain5pp']=c['predicted']>0 and n['predicted']>0 and n['polarity'][kind]['patients']>=10 and n['polarity'][kind]['recall']>=c['polarity'][kind]['recall']+.05
            gates.append(dict(policy=policy,seed=seed,checks=checks,pass_all=all(checks.values())))
    a.put(RUN/'analysis/gates.json',dict(entries=gates,policy_both_seed_pass={p:all(r['pass_all'] for r in gates if r['policy']==p) for p in B['policies']},new_policy_promoted=False,NC_submission_ready=False));a.put(RUN/'analysis/matched_coverage.json',fair);a.put(RUN/'analysis/failure_decomposition.json',failure_rows)
    ceilings=[];necessary=[]
    for (policy,arm,seed),rr in diagnostics.items():
        for stage in ['raw','threshold','nms','selected']:
            for kind in ['point','point_geometry','local200','point_local200']:
                bypat={}
                for r in rr:bypat.setdefault(r['patient'],[]).append(r)
                pats=[rs for rs in bypat.values() if sum(r['reference'] for r in rs)>0];assert len(pats)==176;joint=sum(all(r['stages'][stage][kind]['matched']==r['reference'] for r in rs) for rs in pats)/176
                for lead in ['ii','v1']:
                    vals=[r for r in rr if r['lead']==lead];nr=sum(r['reference'] for r in vals);nm=sum(r['stages'][stage][kind]['matched'] for r in vals);ceilings.append(dict(policy=policy,arm=arm,seed=seed,lead=lead,stage=stage,criterion=kind,reference=nr,matched_upper=nm,recall_upper=nm/max(1,nr),joint_patient_upper=joint,positive_patients=176,oracle_NOT_deployable=True))
        raw=[r for r in ceilings if r['policy']==policy and r['arm']==arm and r['seed']==seed and r['stage']=='raw'];pointok=all(r['recall_upper']>=.8 for r in raw if r['criterion']=='point_geometry');localok=all(r['joint_patient_upper']>=.95 for r in raw if r['criterion']=='local200');necessary.append(dict(policy=policy,arm=arm,seed=seed,geometric_point_80_possible=pointok,local200_patient95_possible=localok,selection_alone_cannot_suffice=not(pointok and localok),passing_bound_not_sufficient=True))
    a.put(RUN/'analysis/oracle_ceilings.json',ceilings);a.put(RUN/'analysis/necessary_condition_audit.json',necessary)
    comparisons=[]
    for seed in B['seeds']:
        for lead in ['ii','v1']:
            for st in ['selected','reported']:
                pairs=[('training_same_policy',(p,B['arms'][0]),(p,B['arms'][1])) for p in a.FACTORS]
                for arm in B['arms']:
                    pairs.extend((name,(before,arm),(after,arm)) for name,before,after in [('unclipping_old_ordered','F00_old_clipped','F01_old_unclipped'),('expansion_clipped','F00_old_clipped','F10_wide_clipped'),('expansion_unclipped','F01_old_unclipped','F11_wide_unclipped'),('unclipping_wide_ordered','F10_wide_clipped','F11_wide_unclipped')])
                for name,(p0,a0),(p1,a1) in pairs:
                    a.check();t0=tables[(p0,a0,seed,lead,st)];t1=tables[(p1,a1,seed,lead,st)];ids=sorted(t0);assert ids==sorted(t1);x=np.stack([t0[i] for i in ids]);y=np.stack([t1[i] for i in ids]);rng=np.random.default_rng(seed+len(comparisons));idx=rng.integers(0,len(ids),(1000,len(ids)));xx=x[idx].sum(1);yy=y[idx].sum(1);dr=yy[:,0]/np.maximum(1,yy[:,1])-xx[:,0]/np.maximum(1,xx[:,1]);valid=(xx[:,2]>0)&(yy[:,2]>0);dp=yy[valid,0]/yy[valid,2]-xx[valid,0]/xx[valid,2];defined=x[:,2].sum()>0 and y[:,2].sum()>0
                    comparisons.append(dict(name=name,before=[p0,a0],after=[p1,a1],seed=seed,lead=lead,stage=st,patients=200,bootstrap=1000,descriptive_only=True,recall_delta=float(y[:,0].sum()/max(1,y[:,1].sum())-x[:,0].sum()/max(1,x[:,1].sum())),recall_CI95=np.quantile(dr,[.025,.975]).tolist(),precision_delta=float(y[:,0].sum()/y[:,2].sum()-x[:,0].sum()/x[:,2].sum()) if defined else None,precision_CI95=np.quantile(dp,[.025,.975]).tolist() if defined and len(dp) else None))
    assert len(comparisons)==96;a.put(RUN/'analysis/patient_bootstrap.json',comparisons)
    a.put(RUN/'analysis/decision.json',dict(time=a.now(),state='zero_fit_mechanism_audit_complete',new_policy_promoted=False,automatic_retraining=False,fits=0,epochs=0,model_forwards=0,independent_sources_admitted=0,clinical_noP_validated=False,brain_heart_future_risk_validated=False,NC_submission_ready=False,next_step='Review fixed factorial and capacity ceilings; only a separate approved mechanism-specific protocol may train. No evidence-free repeated classifier fits.',SPH_admitted=False,RDB_scored=False,new_figures=0,hospital_branch='waiting_data_contract_only',expert_reference='waiting'))
    lines=['A33零拟合候选竞争审计：结果与下一步','20原模型缓存、40原政策重放+80固定因子评价；无新增拟合、无新网络推理。','F00旧有序+截断连续；F01旧有序+未截断连续；F10扩展有序+截断连续；F11扩展有序+未截断连续。','四格同union/原数值阈值/心搏身份/NMS与选择/共同ordered R域1000..4200；不是把F00冒充历史L0。','原A32 L0/W1另行精确重放；所有校准患者/极端值保留。']
    for r in summary:
        if r['lead']!='v1' or r['policy'] not in a.FACTORS:continue
        s=r['stages']['selected'];z=r['stages']['reported'];lines.append(f"{r['policy']} {r['arm']} seed{r['seed']}: selected recall/precision={s['recall']:.4f}/{s['precision_lower']}; negative/biphasic={s['polarity']['negative']['recall']:.4f}/{s['polarity']['biphasic']['recall']:.4f}; reported recall={z['recall']:.4f}; patient ALL before/after={r['strict_patient_joint_before']:.4f}/{r['strict_patient_joint_reported']:.4f}; median/p95 width={r['median_width_ms']}/{r['p95_width_ms']}; q={r['calibration_q']}")
    for r in necessary:lines.append(f"{r['policy']} {r['arm']} seed{r['seed']}: raw geometry80 feasible={r['geometric_point_80_possible']}; raw optimistic local200 patient95 feasible={r['local200_patient95_possible']}; selection alone insufficient={r['selection_alone_cannot_suffice']}")
    lines+=['Oracle只用于诊断上限；不作为可部署预测/校准/临床证据，也不按其删除患者。','先区分候选不足、阈值/NMS/同R竞争损失、端点误差与scale歧义失配；不直接追加epoch。','全部门与失败见analysis/gates.json；双种子同患者，不合并扩充样本。','原始/阈值后/NMS后/最终R槽图均独立复核；只有核验PASS不等于效能通过。','独立来源准入仍0；SPH/RDB封存，无图。脑心风险仅数据合同/mock测试，无真实风险拟合或输出。','医院/专家资料继续等待。NC投稿就绪仍否，未自动放行任何新策略或重训。'];(RUN/'A33审计结果与下一步.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')
