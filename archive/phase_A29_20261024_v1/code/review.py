"""Read-only numerical/identity review; independent augmenting-path matcher."""
import sys,math,time
from pathlib import Path
import run as r
RUN=r.RUN

def beat(offset,qs):
    for i,q in enumerate(qs):
        if q>offset:return i if q-offset<=250 else None
    return None

def allowed(ref,p,qs):
    b=beat(ref[2],qs);w=p.get('wave_R')
    if w is None or not qs:return False
    k=min(range(len(qs)),key=lambda i:abs(qs[i]-w));pb=k if abs(qs[k]-w)<=50 else None
    assert pb==p['expert_beat']
    return b is not None and pb==b

def cardinality(edges):
    owners={}
    def dfs(i,seen):
        for j,yes in enumerate(edges[i]):
            if not yes or j in seen:continue
            seen.add(j)
            if j not in owners or dfs(owners[j],seen):owners[j]=i;return True
        return False
    return sum(dfs(i,set()) for i in range(len(edges)))

def dist(ref,p):
    a,b,c,d=p['box'];return max(a-ref[0],ref[0]-b,c-ref[2],ref[2]-d,0)/p['scale']

def boundary_count(ref,ps,qs):
    return cardinality([[allowed(t,p,qs) and p['box'][0]<=t[0]<=p['box'][1] and p['box'][2]<=t[2]<=p['box'][3] for p in ps] for t in ref])

def point_count(ref,ps,qs,identity=False):
    def edge(t,p):
        ov=max(0,min(t[2],p['offset'])-max(t[0],p['onset']))/max(max(t[2],p['offset'])-min(t[0],p['onset']),1)
        return ov>=.5 and (not identity or allowed(t,p,qs))
    return cardinality([[edge(t,p) for p in ps] for t in ref])

def expand(ps,q):
    if q is None:return []
    return [dict(p,box=[p['box'][0]-q*p['scale'],p['box'][1]+q*p['scale'],p['box'][2]-q*p['scale'],p['box'][3]+q*p['scale']]) for p in ps]

def same(a,b):
    if a is None:return math.isinf(b)
    return math.isfinite(b) and abs(a-b)<=1e-12

def main():
    r.check();r.verify();frozen=r.read(RUN/'protocol_frozen.json');assert r.sha(RUN/'source_lock.json')==frozen['source_lock_SHA256'];complete=r.read(RUN/'completion.json');assert complete['models']==30 and complete['decoder_policies']==120 and complete['new_fits']==0
    recordchecks=calchecks=patientchecks=0;traces=[];denominators=[]
    for j in r.m.jobs():
        r.check();path=RUN/'jobs'/f"job_{j['index']:02d}";v=r.ungz(path/'candidates.json.gz');cache=r.ungz(r.A28/'jobs'/f"job_{j['index']:02d}"/'candidate_cache.json.gz');done=r.read(path/'done.json');assert done['lead_signals']==128 and len(done['original_policies_replayed'])==4
        for role in ['calibration','outer']:
            assert [x['selected'] for x in v['L0_legacy'][role]]==[x['selected'] for x in v['I1_identity'][role]]
        for policy in r.B['policies']:
            out=r.ungz(path/(policy+'.json.gz'));cal=out['calibration'];ledger=cal['ledger'];expected={x['patient'] for x in cache['calibration'] if x['ref']};assert {x['patient'] for x in ledger}==expected and cal['n_reference_positive']==len(expected)
            denominators.append(dict(job=j['index'],policy=policy,n=len(expected),missing_patients=cal['missing_patient_count'],strict_q=cal['strict_q'],component_q=cal['component_q']))
            for a,b in zip(v[policy]['calibration'],cache['calibration']):assert a['patient']==b['patient'] and a['ref']==b['ref'],'Reference/patient deletion'
            for x in ledger:
                rs=[a for a in v[policy]['calibration'] if a['patient']==x['patient'] and a['ref']];stricts=[];cs=[];missing=0
                for a in rs:
                    ref=a['ref'];pp=a['selected'];qs=a['qrs'];edges=[[allowed(t,p,qs) for p in pp] for t in ref];n=cardinality(edges);missing+=len(ref)-n
                    if n<len(ref):strict=float('inf')
                    else:
                        values=sorted({dist(t,p) for t in ref for p in pp if allowed(t,p,qs)})
                        strict=next(q for q in values if cardinality([[edges[i][k] and dist(t,p)<=q for k,p in enumerate(pp)] for i,t in enumerate(ref)])==len(ref))
                    sr=next(a2 for a2 in x['rows'] if a2['lead']==a['lead']);assert same(sr['strict'],strict);assert len(sr['pairs'])==n
                    assert len({i for i,k in sr['pairs']})==n and len({k for i,k in sr['pairs']})==n
                    errs=[]
                    for i,k in sr['pairs']:assert allowed(ref[i],pp[k],qs);errs.append(dist(ref[i],pp[k]))
                    comp=max(errs,default=float('inf'));assert same(sr['component'],comp);stricts.append(strict);cs.append(comp);calchecks+=1
                assert x['missing']==missing and same(x['strict'],max(stricts)) and same(x['component'],max(cs))
            k=math.ceil((len(ledger)+1)*.95);assert k==cal['k']
            for name in ['strict','component']:
                vals=sorted(float('inf') if a[name] is None else a[name] for a in ledger);q=vals[k-1] if k<=len(vals) else float('inf');assert same(cal[name+'_q'],q)
            if policy!='L0_legacy':assert cal['used_q']==cal['strict_q']
            bypat={}
            for a,stored in zip(v[policy]['outer'],out['records']):
                assert a['patient']==stored['patient'] and a['lead']==stored['lead'] and a['selected']==stored['selected'];ps=a['selected'];qs=a['qrs'];ref=a['ref']
                assert all(p['confidence']>=cache['threshold'] and (cache['mode']=='union' or cache['mode']==p['kind']) for p in ps)
                if policy in ['M2_beat_merge','B3_boundary']:assert len({p['wave_R'] for p in ps})==len(ps) and all(p['wave_R'] is not None for p in ps)
                expanded=expand(ps,cal['used_q']);reported=[p for p in expanded if max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=100]
                assert boundary_count(ref,expanded,qs)==stored['strict_unique_covered'] and boundary_count(ref,reported,qs)==stored['strict_unique_reported']
                for name,pp in [('raw',a['raw']),('selected',ps),('reported',reported)]:
                    table=stored['tables'][name];assert table['matched']==point_count(ref,pp,qs) and table['identity_matched']==point_count(ref,pp,qs,True)
                assert all(2*max(p['box'][1]-p['box'][0],p['box'][3]-p['box'][2])<=200+1e-12 for p in reported)
                bypat.setdefault(a['patient'],[]).append(stored);recordchecks+=1
            for p in out['patients']:
                rows=bypat[p['patient']];assert p['reference']==sum(x['reference'] for x in rows)
                assert p['strict_all']==all(x['strict_unique_covered']==x['reference'] for x in rows) and p['strict_reported_all']==all(x['strict_unique_reported']==x['reference'] for x in rows);patientchecks+=1
            if j['index']==8:traces.append(dict(policy=policy,used_q=cal['used_q'],n=cal['n_reference_positive'],data13=next(x for x in ledger if x['patient']==12)))
    r.verify();r.check();r.put(RUN/'post_review/contract_review.json',dict(time=r.now(),state='PASS',review_code_SHA256=r.sha(__file__),matcher='independent augmenting paths, not scipy Hungarian',outer_lead_records_checked=recordchecks,calibration_lead_records_checked=calchecks,patient_entries_checked=patientchecks,original_replays=120,policies=120,same_beat_and_distinct_assignment=True,all_reference_positive_calibration_patients_retained=True,extreme_data13_retained=True,source_hashes_unchanged=True,new_fits=0,new_epochs=0,real_model_forwards=0,effectiveness_pass=r.read(RUN/'analysis/gates.json')['any_policy_both_seed_pass'],NC_submission_ready=False))
    r.put(RUN/'post_review/calibration_denominators.json',denominators);r.put(RUN/'post_review/data13_trace.json',traces);print('PASS independent zero-fit review',recordchecks,calchecks,patientchecks)

if __name__=='__main__':main()
