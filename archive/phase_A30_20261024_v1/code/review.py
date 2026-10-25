"""A30 final zero-fit review; independent matching borrowed as a pure library."""
import math,importlib.util
from pathlib import Path
import method as q
RUN=q.RUN
spec=importlib.util.spec_from_file_location('a29_independent_matching',q.A29/'code/review.py');v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)

def main():
    import train as t
    t.check('analysis');counts=dict(outer_records=0,calibration_records=0,patient_entries=0,epochs=0,optimizer_steps=0);groups={};extremes=[]
    for j in q.jobs():
        t.check('analysis');p=q.jobpath(j);done=q.read(p/'done.json');hist=q.read(p/'history.json');contract=q.read(p/'training_contract.json');trained=q.read(p/'training_done.json');cache=q.ungz(p/'prediction_cache.json.gz');out=q.ungz(p/'result.json.gz');sampling=q.ungz(p/'sampling_ledger.json.gz')
        assert done['epochs']==8 and len(hist)==8 and done['checkpoint_SHA256']==q.sha(p/'final.pt') and trained['checkpoint_SHA256']==done['checkpoint_SHA256'];counts['epochs']+=len(hist);counts['optimizer_steps']+=done['optimizer_steps'];assert done['inference_lead_signals']==152
        key=(j['seed'],j['fold']);groups.setdefault(key,[]).append((trained['initial_tensor_SHA256'],trained['order_SHA256'],trained['augmentation_SHA256'],contract['train_only_weights']))
        if j['arm']==q.B['arms'][0]:
            old=q.read(q.m.path(dict(j,arm='C2_null_ordered'))/'history.json');assert all(abs(a['loss']-b['loss'])<=1e-6 for a,b in zip(hist,old))
        else:
            for epoch in sampling:
                assert all(pos['represented']==pos['eligible'] for pos in epoch['positions'].values()),'Balanced eligible lead item missing target'
        c=out['calibration'];expected={r['patient'] for r in cache['calibration'] if r['ref']};assert {a['patient'] for a in c['ledger']}==expected and c['n_reference_positive']==len(expected)
        for a in c['ledger']:
            rows=[r for r in cache['calibration'] if r['patient']==a['patient'] and r['ref']];stricts=[];components=[];missing=0
            for r in rows:
                ref=r['ref'];ps=r['selected'];qs=r['qrs'];edges=[[v.allowed(z,p,qs) for p in ps] for z in ref];n=v.cardinality(edges);missing+=len(ref)-n
                if n<len(ref):score=float('inf')
                else:
                    values=sorted({v.dist(z,p) for z in ref for p in ps if v.allowed(z,p,qs)});score=next(x for x in values if v.cardinality([[edges[i][k] and v.dist(z,p)<=x for k,p in enumerate(ps)] for i,z in enumerate(ref)])==len(ref))
                rr=next(z for z in a['rows'] if z['lead']==r['lead']);assert v.same(rr['strict'],score);assert len(rr['pairs'])==n and len({i for i,k in rr['pairs']})==n and len({k for i,k in rr['pairs']})==n;assert all(v.allowed(ref[i],ps[k],qs) for i,k in rr['pairs']);component=max((v.dist(ref[i],ps[k]) for i,k in rr['pairs']),default=float('inf'));assert v.same(rr['component'],component);components.append(component);stricts.append(score);counts['calibration_records']+=1
            assert a['missing']==missing and v.same(a['strict'],max(stricts)) and v.same(a['component'],max(components))
        k=math.ceil((len(c['ledger'])+1)*.95);vals=sorted(float('inf') if a['strict'] is None else a['strict'] for a in c['ledger']);qq=vals[k-1] if k<=len(vals) else float('inf');assert c['k']==k and v.same(c['strict_q'],qq) and c['used_q']==c['strict_q']
        component_vals=sorted(float('inf') if a['component'] is None else a['component'] for a in c['ledger']);component_q=component_vals[k-1] if k<=len(component_vals) else float('inf');assert v.same(c['component_q'],component_q)
        for r,stored in zip(cache['outer'],out['records']):
            assert r['patient']==stored['patient'] and r['selected']==stored['selected'];ps=r['selected'];ref=r['ref'];qs=r['qrs'];assert len({p['wave_R'] for p in ps})==len(ps) and all(p['wave_R'] is not None for p in ps)
            assert all(p['confidence']>=out['choice']['threshold'] and (out['choice']['mode']=='union' or p['kind']==out['choice']['mode']) for p in ps)
            exp=v.expand(ps,c['used_q']);rep=[p for p in exp if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100]
            assert v.boundary_count(ref,exp,qs)==stored['strict_unique_covered'] and v.boundary_count(ref,rep,qs)==stored['strict_unique_reported']
            for name,pp in [('raw',r['raw']),('selected',ps),('reported',rep)]:assert stored['tables'][name]['matched']==v.point_count(ref,pp,qs) and stored['tables'][name]['identity_matched']==v.point_count(ref,pp,qs,True)
            counts['outer_records']+=1
        for pat in out['patients']:
            rr=[a for a in out['records'] if a['patient']==pat['patient']];assert pat['strict_all']==all(a['strict_unique_covered']==a['reference'] for a in rr) and pat['strict_reported_all']==all(a['strict_unique_reported']==a['reference'] for a in rr);counts['patient_entries']+=1
        if 12 in expected:extremes.append(dict(job=j,record='data/13',entry=next(a for a in c['ledger'] if a['patient']==12)))
    assert counts['epochs']==240 and counts['optimizer_steps']<=7440
    assert all(len(rows)==3 and all(row==rows[0] for row in rows) for rows in groups.values()),'Paired initialisation/order/augmentation/weights mismatch'
    assert len(extremes)>0;t.verify();t.check('analysis')
    q.put(RUN/'post_review/contract_review.json',dict(time=q.now(),state='PASS',independent_matcher='augmenting paths',paired_initialisation_order_augmentation_weights=True,extreme_patients_retained=True,calibration_all_reference_patients_retained=True,one_to_one_same_beat=True,source_hashes_unchanged=True,new_review_fits=0,**counts));q.put(RUN/'post_review/data13_trace.json',extremes)
