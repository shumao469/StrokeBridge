"""Fixed, inference-only decoder. No annotation argument is accepted here."""
import math
import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.optimize import linear_sum_assignment

def iou(ref,p):
    return max(0,min(ref[2],p['offset'])-max(ref[0],p['onset']))/max(max(ref[2],p['offset'])-min(ref[0],p['onset']),1)

def assign(score):
    a=np.asarray(score,float)
    if a.size==0:return []
    assert a.ndim==2 and np.isfinite(a).all() and np.all((a>=0)&(a<=1))
    ok=a>0;reward=ok*(max(a.shape)+1+a)
    ii,jj=linear_sum_assignment(reward,maximize=True)
    return [(int(i),int(j)) for i,j in zip(ii,jj) if ok[i,j]]

def next_expert(offset,qrs):
    j=next((i for i,r in enumerate(qrs) if r>offset),None)
    return j if j is not None and qrs[j]-offset<=250 else None

def map_expert(anchor,qrs):
    if anchor is None or not qrs:return None
    j=min(range(len(qrs)),key=lambda i:abs(qrs[i]-anchor))
    return j if abs(qrs[j]-anchor)<=50 else None

def wave_anchor(p,anchors):
    if p['kind']=='ordered':return p['source_R']
    r=next((int(r) for r in anchors if r>p['offset']),None)
    return r if r is not None and r-p['offset']<=250 and r-150<=p['peak']<=r-10 else None

def nms(pp):
    out=[]
    for p in sorted(pp,key=lambda v:v['confidence'],reverse=True):
        if all(iou((p['onset'],p['peak'],p['offset']),v)<.5 for v in out):out.append(p)
    return out

def merge(pp,anchors,support_rank=False):
    groups={};unresolved=[]
    for p in pp:
        r=wave_anchor(p,anchors)
        if r is None:unresolved.append(dict(p,rejection='waveform_beat_unresolved'));continue
        groups.setdefault(r,[]).append(dict(p,wave_R=r))
    selected=[];ledger=[]
    for r,values in sorted(groups.items()):
        kept=nms(values)
        def key(v):
            dur=v['offset']-v['onset']
            return (v['confidence']*math.sqrt(dur/100) if support_rank else v['confidence'],v['confidence'],dur,-v['uid'])
        winner=max(kept,key=key);selected.append(winner)
        ledger.append(dict(wave_R=r,raw_count=len(values),after_local_NMS=len(kept),winner_uid=winner['uid'],suppressed_uids=[v['uid'] for v in values if v['uid']!=winner['uid']]))
    assert len({p['wave_R'] for p in selected})==len(selected)
    return selected,dict(unresolved=unresolved,groups=ledger)

def legal_pairs(r,n):
    a=np.arange(max(0,int(r)-150),max(0,int(r)-10));s,e=np.meshgrid(a,a,indexing='ij');ok=(e-s>=10)&(e-s<=100)&(e<n)
    return s[ok].astype(np.int64),e[ok].astype(np.int64)

def peak(z,a,b):
    return a+int(np.argmax(abs(z[a:b+1]-np.linspace(z[a],z[b],b-a+1))))

def posterior_decode(prob,z,anchors):
    p=prob[1].astype('float64');lod=(np.log(np.maximum(p,1e-12))-np.log(np.maximum(1-p,1e-12)))/5;cs=np.r_[0,np.cumsum(lod)];out=[]
    for r in anchors:
        if not 1000<int(r)<4200:continue
        ss,ee=legal_pairs(r,len(p));score=cs[ee+1]-cs[ss]-math.log(len(ss));mx=max(0,float(score.max()));ww=np.exp(score-mx);pn=float(ww.sum()/(ww.sum()+math.exp(-mx)));ww/=ww.sum()
        def qs(v):
            cdf=np.cumsum(np.bincount(v,weights=ww,minlength=len(p)))
            return [int(min(len(p)-1,np.searchsorted(cdf,q))) for q in [.25,.5,.75]]
        aq,bq=qs(ss),qs(ee);a,b=aq[1],bq[1]
        assert 10<=b-a<=100 and int(r)-150<=a<=b<=int(r)-11,'Median geometry'
        pk=peak(z,a,b)
        if 1000<=pk<4000:
            out.append(dict(onset=a,offset=b,peak=pk,confidence=min(pn,float(p[a:b+1].mean())),legacy_score=pn,non_null_mass=pn,box=[a,a,b,b],scale=max(5.,.5*max(aq[2]-aq[0],bq[2]-bq[0])),kind='ordered',source_R=int(r),boundary_rule='marginal_median',endpoint_IQR=[aq,bq]))
    sm=uniform_filter1d(prob.astype('float64'),size=5,axis=-1,mode='nearest');hard=(sm[2]>sm[1])|(sm[3]>sm[1]);mask=(sm[1]>=.5)&~hard
    diff=np.diff(np.r_[False,mask,False].astype(int));runs=list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)-1));joined=[]
    for a,b in runs:
        if joined and a-joined[-1][1]-1<=10 and not hard[joined[-1][1]+1:a].any():joined[-1]=(joined[-1][0],b)
        else:joined.append((a,b))
    seen=set()
    for a,b in joined:
        while a>0 and sm[1,a-1]>=.2 and not hard[a-1]:a-=1
        while b<len(p)-1 and sm[1,b+1]>=.2 and not hard[b+1]:b+=1
        pk=peak(z,a,b)
        r=next((int(r) for r in anchors if r>pk),None)
        if r is None:continue
        a=max(a,r-150);b=min(b,r-11)
        if not 10<=b-a<=100:continue
        pk=peak(z,a,b)
        if not 1000<=pk<4000 or (a,b,r) in seen:continue
        seen.add((a,b,r));out.append(dict(onset=int(a),offset=int(b),peak=int(pk),confidence=float(p[a:b+1].mean()),box=[int(a),int(a),int(b),int(b)],scale=5.,kind='continuous',source_R=None,boundary_rule='fixed_hysteresis'))
    for uid,v in enumerate(out):
        v['uid']=uid;assert 0<=v['onset']<=v['peak']<=v['offset']<len(p) and 0<=v['confidence']<=1
    return out

def decorate(pp,anchors,qrs):
    return [dict(p,wave_R=wave_anchor(p,anchors),expert_beat=map_expert(wave_anchor(p,anchors),qrs)) for p in pp]

def compatible(ref,p,qrs):
    beat=next_expert(ref[2],qrs)
    return beat is not None and p.get('expert_beat')==beat

def point_pairs(ref,pp,qrs,identity=False):
    return assign([[iou(r,p) if iou(r,p)>=.5 and (not identity or compatible(r,p,qrs)) else 0 for p in pp] for r in ref]) if ref and pp else []

def box_pairs(ref,pp,qrs):
    if not ref or not pp:return []
    return assign([[float(compatible(r,p,qrs) and p['box'][0]<=r[0]<=p['box'][1] and p['box'][2]<=r[2]<=p['box'][3]) for p in pp] for r in ref])

def distance(r,p):
    a,b,c,d=p['box'];return max(a-r[0],r[0]-b,c-r[2],r[2]-d,0)/p['scale']

def identity_pairs(ref,pp,qrs):
    if not ref or not pp:return []
    return assign([[1/(1+distance(r,p)) if compatible(r,p,qrs) else 0 for p in pp] for r in ref])

def scores(ref,pp,qrs):
    if not ref:return dict(strict=0.,component=0.,missing=0,pairs=[],errors=[])
    pairs=identity_pairs(ref,pp,qrs);err=[distance(ref[i],pp[j]) for i,j in pairs];component=max(err,default=float('inf'))
    if len(pairs)!=len(ref):strict=float('inf')
    else:
        vals=sorted({distance(r,p) for r in ref for p in pp if compatible(r,p,qrs)});lo,hi=0,len(vals)-1
        while lo<hi:
            mid=(lo+hi)//2;edges=[[float(compatible(r,p,qrs) and distance(r,p)<=vals[mid]) for p in pp] for r in ref]
            if len(assign(edges))==len(ref):hi=mid
            else:lo=mid+1
        strict=vals[lo]
    return dict(strict=strict,component=component,missing=len(ref)-len(pairs),pairs=pairs,errors=err)

def quantile(values):
    n=len(values);k=math.ceil((n+1)*.95)
    return float(sorted(values)[k-1]) if n and k<=n else float('inf')

def expand(pp,q):
    if not math.isfinite(q):return []
    out=[]
    for p in pp:
        a,b,c,d=p['box'];s=p['scale'];out.append(dict(p,box=[a-q*s,b+q*s,c-q*s,d+q*s]))
    return out

def preflight():
    def prop(a,b,r,score=.9,uid=0):return dict(onset=a,peak=(a+b)//2,offset=b,box=[a,a,b,b],scale=5.,kind='ordered',source_R=r,confidence=score,uid=uid)
    qs=[100,300];ref=[(40,50,60),(240,250,260)];old=decorate([prop(40,60,100)],[100,300],qs)
    ss=scores(ref,old,qs);assert math.isinf(ss['strict']) and ss['missing']==1 and ss['component']==0
    broad=dict(old[0],box=[0,400,0,400]);assert len(box_pairs(ref,[broad],qs))==1
    # Different waveform identities cannot suppress each other even with full overlap.
    pp=[prop(40,60,100,.9,0),prop(40,60,105,.8,1)];assert len(nms(pp))==1 and len(merge(pp,[100,105])[0])==2
    same=[prop(40,60,100,.9,0),prop(41,61,100,.8,1)];assert len(merge(same,[100])[0])==1
    assert len(assign([[.9,.8],[.8,0]]))==2
    two=decorate([prop(45,65,100),prop(240,260,300,uid=1)],[100,300],qs);assert scores(ref,two,qs)['strict']==1
    assert quantile(list(range(24)))==23 and math.isinf(quantile(list(range(12))))
    assert math.isinf(quantile([0.]*20+[float('inf')])) # Do not delete extreme/missing patient.
    assert len(expand(old,float('inf')))==0 and expand(old,10)[0]['box'][1]-expand(old,10)[0]['box'][0]==100
    z=np.sin(np.arange(5000)/7);pr=np.zeros((4,5000));pr[0]=.95;pr[1]=.05;pr[0,1300:1351]=.1;pr[1,1300:1351]=.9;pr[2,1330:1334]=.99;pr[1,1330:1334]=.01
    props=posterior_decode(pr,z,[1400]);assert props and all(10<=v['offset']-v['onset']<=100 for v in props)
    assert all(not (p['onset']<1331 and p['offset']>1332) for p in props if p['kind']=='continuous')
    assert all(v['confidence']>=.9 for v in [p for p in props if p['confidence']>=.9])
    return ['missing_identity_is_infinite_not_cross_beat_error','no_candidate_reuse','cross_R_suppression_forbidden','same_R_single_slot','maximum_cardinality','minimax_residual','finite_patient_quantile','extreme_patient_retained','width_and_infinite_q','posterior_median_legal','hard_QRS_hysteresis_barrier','original_confidence_cutoff']
