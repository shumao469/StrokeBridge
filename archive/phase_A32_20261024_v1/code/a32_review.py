"""Report every fixed comparison and independently review matching, calibration and pairing."""
import math,importlib.util
import numpy as np
import a32_method as q
RUN=q.RUN;B=q.B
spec=importlib.util.spec_from_file_location('a32_independent_matching',q.old.A29/'code/review.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def analyse():
    import a32_run as t
    summary=[];tables={}
    for policy in B['policies']:
        for arm in B['arms']:
            for seed in B['seeds']:
                t.check('analysis');outs=[q.ungz(q.jobpath(j)/(policy+'.json.gz')) for j in q.jobs() if j['arm']==arm and j['seed']==seed];patients=[p for o in outs for p in o['patients']];assert len(patients)==176 and len({p['patient'] for p in patients})==176
                for lead in ['ii','v1']:
                    rec=[r for o in outs for r in o['records'] if r['lead']==lead];width=[w[k] for o in outs for w in o['widths'] if w['lead']==lead for k in ['onset_ms','offset_ms']]
                    def stage(name):
                        rr=[r['tables'][name] for r in rec];nr=sum(a['reference'] for a in rr);nm=sum(a['matched'] for a in rr);npred=sum(a['predicted'] for a in rr);nu=sum(a['unknown_unmatched'] for a in rr)
                        return dict(reference=nr,matched=nm,predicted=npred,recall=nm/max(1,nr),precision_lower=nm/npred if npred else None,precision_upper=(nm+nu)/npred if npred else None,identity_matched=sum(a['identity_matched'] for a in rr),polarity={kind:dict(reference=sum(a['polarity'][kind]['reference'] for a in rr),matched=sum(a['polarity'][kind]['matched'] for a in rr),recall=sum(a['polarity'][kind]['matched'] for a in rr)/max(1,sum(a['polarity'][kind]['reference'] for a in rr)),patients=len({r['patient'] for r in rec if any(a['kind']==kind for a in r['reference_pol'])})) for kind in ['positive','negative','biphasic']})
                    summary.append(dict(policy=policy,arm=arm,seed=seed,lead=lead,stages={k:stage(k) for k in ['raw','selected','reported','component_reported_DIAGNOSTIC']},reference_positive_patients=len(patients),strict_patient_joint_before=sum(p['strict_all'] for p in patients)/len(patients),strict_patient_joint_reported=sum(p['strict_reported_all'] for p in patients)/len(patients),median_width_ms=float(np.median(width)) if width else None,p95_width_ms=float(np.quantile(width,.95)) if width else None,finite_calibration_folds=sum(o['calibration']['strict_q'] is not None for o in outs),calibration_q=[o['calibration']['strict_q'] for o in outs],calibration_n=[o['calibration']['n_reference_positive'] for o in outs],design_feasible_folds=sum(o['design_feasible'] for o in outs),missing_same_beat=sum(r['identity_score']['missing'] for r in rec)))
                    for stage_name in ['selected','reported']:tables[(policy,arm,seed,lead,stage_name)]={r['patient']:np.array([r['tables'][stage_name][key] for key in ['matched','reference','predicted']]) for r in rec}
    q.put(RUN/'analysis/summary.json',summary);get=lambda p,a,s,l:next(r for r in summary if r['policy']==p and r['arm']==a and r['seed']==s and r['lead']==l);gates=[];fair=[];bootstrap=[]
    for seed in B['seeds']:
        checks={}
        for lead in ['ii','v1']:
            a=get(B['primary_policy'],B['arms'][0],seed,lead);b=get(B['primary_policy'],B['arms'][1],seed,lead);aa=a['stages']['reported'];bb=b['stages']['reported']
            checks[lead+'_useful_strict']=b['strict_patient_joint_before']>=.95 and b['strict_patient_joint_reported']>=.95 and bb['recall']>=.8 and b['median_width_ms'] is not None and b['median_width_ms']<=120 and b['p95_width_ms']<=200 and b['design_feasible_folds']==5
            checks[lead+'_noninferior']=aa['precision_lower'] is not None and bb['precision_lower'] is not None and bb['recall']>=aa['recall']-.02 and bb['precision_lower']>=aa['precision_lower']-.02
            matched=aa['recall']>=.8 and bb['recall']>=.8 and abs(aa['recall']-bb['recall'])<=.02;checks[lead+'_matched_coverage']=matched;fair.append(dict(seed=seed,lead=lead,policy=B['primary_policy'],C0_reported_recall=aa['recall'],C1_reported_recall=bb['recall'],matched_coverage=matched,advantage_claim=False,interpretation='Descriptive paired comparison only; nonmatched coverage is not a fairness pass'))
        a=get(B['primary_policy'],B['arms'][0],seed,'v1')['stages']['reported'];b=get(B['primary_policy'],B['arms'][1],seed,'v1')['stages']['reported']
        for kind in ['negative','biphasic']:checks[kind+'_gain5pp']=a['predicted']>0 and b['predicted']>0 and b['polarity'][kind]['patients']>=10 and b['polarity'][kind]['recall']>=a['polarity'][kind]['recall']+.05
        gates.append(dict(seed=seed,checks=checks,pass_all=all(checks.values())))
        for lead in ['ii','v1']:
            for stage in ['selected','reported']:
                comparisons=[('training_same_W1',(B['primary_policy'],B['arms'][0]),(B['primary_policy'],B['arms'][1]))]+[('decoder_same_weights',('L0_legacy_decoder',arm),('W1_window_aware',arm)) for arm in B['arms']]
                for name,(pa,aa),(pb,ab) in comparisons:
                    t.check('analysis');a=tables[(pa,aa,seed,lead,stage)];b=tables[(pb,ab,seed,lead,stage)];ids=sorted(a);assert ids==sorted(b) and len(ids)==200;av=np.stack([a[i] for i in ids]);bv=np.stack([b[i] for i in ids]);rng=np.random.default_rng(seed+len(bootstrap));idx=rng.integers(0,len(ids),(1000,len(ids)));ax=av[idx].sum(1);bx=bv[idx].sum(1);dr=bx[:,0]/np.maximum(1,bx[:,1])-ax[:,0]/np.maximum(1,ax[:,1]);valid=(ax[:,2]>0)&(bx[:,2]>0);dp=bx[valid,0]/bx[valid,2]-ax[valid,0]/ax[valid,2];pd=av[:,2].sum()>0 and bv[:,2].sum()>0
                    bootstrap.append(dict(name=name,before=[pa,aa],after=[pb,ab],seed=seed,lead=lead,stage=stage,patients=200,bootstrap=1000,descriptive_only=True,recall_delta=float(bv[:,0].sum()/max(1,bv[:,1].sum())-av[:,0].sum()/max(1,av[:,1].sum())),recall_CI95=np.quantile(dr,[.025,.975]).tolist(),precision_delta=float(bv[:,0].sum()/bv[:,2].sum()-av[:,0].sum()/av[:,2].sum()) if pd else None,precision_CI95=np.quantile(dp,[.025,.975]).tolist() if pd and len(dp) else None))
    q.put(RUN/'analysis/gates.json',dict(seeds=gates,both_seed_primary_pass=all(v['pass_all'] for v in gates),all_patient_references_preserved=True,clinical_noP_validated=False,AF_increment_validated=False,brain_stroke_risk_validated=False,NC_submission_ready=False));q.put(RUN/'analysis/matched_coverage.json',fair);q.put(RUN/'analysis/patient_bootstrap.json',bootstrap)
    q.put(RUN/'analysis/decision.json',dict(time=q.now(),measurement_gate_pass=all(v['pass_all'] for v in gates),independent_sources_admitted=0,automatic_followon_training=False,SPH_admitted=False,RDB_scored=False,new_figures=0,NC_submission_ready=False,hospital_branch='waiting',expert_reference='waiting'))
    lines=['A32整患者多心搏、窗外表征及报告一致性试验（自动结果，须结合独立核验）','20配对拟合、160epoch；双种子使用同一批患者，不能合并为两队列。','C0是相同整患者批次上的旧目标新拟合，非A30精确复现；C1为新监督。','W1是预先声明的新输入/表征政策；L0保留原窗口/模式。二者不追溯更改A31。']
    for r in summary:
        if r['lead']!='v1':continue
        s=r['stages']['selected'];z=r['stages']['reported'];lines.append(f"{r['policy']} {r['arm']} seed{r['seed']}: selected_recall={s['recall']:.4f}; selected_precision={s['precision_lower']}; reported_recall={z['recall']:.4f}; reported_precision={z['precision_lower']}; negative={z['polarity']['negative']['recall']:.4f}; biphasic={z['polarity']['biphasic']['recall']:.4f}; patient_ALL_before/after={r['strict_patient_joint_before']:.4f}/{r['strict_patient_joint_reported']:.4f}; median/p95_width_ms={r['median_width_ms']}/{r['p95_width_ms']}; q={r['calibration_q']}")
    lines+=['两种子主门同时通过='+str(all(v['pass_all'] for v in gates)),'未删极端患者；缺失为检出失败/无限严格分数；不可用跨心搏候选或重复框代替。','低样本患者校准的极端分位数限制保留；不把扩大区间造成的集合覆盖当成可靠测量。','零新增独立来源；SPH和RDB不评分；医院及专家资料继续等待。','P波测量/AF识别不是未来AF、脑损伤或卒中风险预测；NC投稿就绪仍否。','结果失败不自动追加训练；无新图。'];(RUN/'A32结果与下一步.txt').write_text('\n'.join(lines)+'\n',encoding='utf8')

def review():
    import a32_run as t
    counts=dict(epochs=0,optimizer_steps=0,calibration_records=0,outer_records=0,patient_entries=0);pairs={};extremes=[];roles=q.read(RUN/'patient_roles.json');support=q.read(RUN/'target_support.json')
    for j in q.jobs():
        t.check('analysis');p=q.jobpath(j);done=q.read(p/'done.json');trained=q.read(p/'training_done.json');hist=q.read(p/'history.json');contract=q.read(p/'training_contract.json');cache=q.ungz(p/'prediction_cache.json.gz');sampling=q.ungz(p/'sampling_ledger.json.gz');assert contract['roles']==roles[j['fold']] and len(hist)==8 and done['epochs']==8 and done['optimizer_steps']==248 and done['inference_lead_signals']==152 and q.sha(p/'final.pt')==trained['checkpoint_SHA256']==done['checkpoint_SHA256'];counts['epochs']+=8;counts['optimizer_steps']+=done['optimizer_steps'];pairs.setdefault((j['seed'],j['fold']),[]).append((trained['initial_tensor_SHA256'],trained['order_SHA256'],trained['augmentation_SHA256'],contract['train_only_weights']))
        for row in sampling:
            assert sorted(row['patient_order'])==sorted(contract['roles']['train']) and len(row['lead_items'])==len(set(row['lead_items']))==contract['train_leads']
            if j['arm']==B['arms'][1]:
                expected=sum(r['supported'] for r in support if r['patient'] in contract['roles']['train']);assert row['supervision']['positive']==expected==row['supervision']['endpoint_terms'];assert row['supervision']['outside_old']==sum(r['outside_old'] for r in support if r['patient'] in contract['roles']['train'])
        for policy in B['policies']:
            out=q.ungz(p/(policy+'.json.gz'));data=cache['policies'][policy];c=out['calibration'];expected={r['patient'] for r in data['calibration'] if r['ref']};assert {r['patient'] for r in c['ledger']}==expected and c['n_reference_positive']==len(expected)
            for row in c['ledger']:
                scores=[];missing=0
                for r in [a for a in data['calibration'] if a['patient']==row['patient'] and a['ref']]:
                    ref=r['ref'];ps=r['selected'];qs=r['qrs'];edges=[[v.allowed(a,p,qs) for p in ps] for a in ref];n=v.cardinality(edges);missing+=len(ref)-n
                    if n<len(ref):score=float('inf')
                    else:
                        vals=sorted({v.dist(a,p) for a in ref for p in ps if v.allowed(a,p,qs)});score=next(x for x in vals if v.cardinality([[edges[i][k] and v.dist(a,p)<=x for k,p in enumerate(ps)] for i,a in enumerate(ref)])==len(ref))
                    stored=next(a for a in row['rows'] if a['lead']==r['lead']);assert v.same(stored['strict'],score);assert len(stored['pairs'])==n and len({i for i,k in stored['pairs']})==n and len({k for i,k in stored['pairs']})==n and all(v.allowed(ref[i],ps[k],qs) for i,k in stored['pairs']);scores.append(score);counts['calibration_records']+=1
                assert row['missing']==missing and v.same(row['strict'],max(scores))
            k=math.ceil((len(c['ledger'])+1)*.95);vals=sorted(float('inf') if r['strict'] is None else r['strict'] for r in c['ledger']);quant=vals[k-1] if k<=len(vals) else float('inf');assert c['k']==k and v.same(c['strict_q'],quant) and c['used_q']==c['strict_q']
            for r,stored in zip(data['outer'],out['records']):
                assert r['patient']==stored['patient'] and r['selected']==stored['selected'];ref=r['ref'];ps=r['selected'];qs=r['qrs'];assert len({p['wave_R'] for p in ps})==len(ps) and all(p['wave_R'] is not None and p['confidence']>=out['choice']['threshold'] for p in ps)
                exp=v.expand(ps,c['used_q']);rep=[p for p in exp if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100];assert v.boundary_count(ref,exp,qs)==stored['strict_unique_covered'] and v.boundary_count(ref,rep,qs)==stored['strict_unique_reported']
                for name,values in [('raw',r['raw']),('selected',ps),('reported',rep)]:assert stored['tables'][name]['matched']==v.point_count(ref,values,qs) and stored['tables'][name]['identity_matched']==v.point_count(ref,values,qs,True)
                counts['outer_records']+=1
            assert {r['patient'] for r in data['outer']}==set(contract['roles']['outer'])
            for pat in out['patients']:
                rr=[r for r in out['records'] if r['patient']==pat['patient']];assert pat['strict_all']==all(r['strict_unique_covered']==r['reference'] for r in rr) and pat['strict_reported_all']==all(r['strict_unique_reported']==r['reference'] for r in rr);counts['patient_entries']+=1
            if 12 in expected:extremes.append(dict(job=j,policy=policy,record='data/13',entry=next(r for r in c['ledger'] if r['patient']==12)))
    assert counts['epochs']==160 and counts['optimizer_steps']==4960 and all(len(x)==2 and x[0]==x[1] for x in pairs.values()) and extremes;t.verify();t.check('analysis')
    q.put(RUN/'post_review/contract_review.json',dict(time=q.now(),state='PASS',independent_matcher='augmenting_paths',paired_initialization_order_augmentation_weights=True,all_supported_targets_each_epoch=True,extreme_patients_retained=True,strict_missing_infinity=True,unique_same_beat=True,source_hashes_unchanged=True,new_review_fits=0,**counts));q.put(RUN/'post_review/extreme_patients.json',extremes)
