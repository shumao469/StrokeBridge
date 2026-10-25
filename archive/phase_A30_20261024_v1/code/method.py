"""Paired training mechanisms; no checkpoint selection or threshold fitting."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,math,hashlib
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch import nn
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A29=BASE/'runs/phase_A29_20261024_v1'
sys.path.insert(0,str(A29/'code'));import run as e
m=e.m;d=e.d;torch.set_num_threads(1);torch.set_grad_enabled(True)
B=e.read(RUN/'budget.json');read=e.read;sha=e.sha;put=e.put;gz=e.gz;ungz=e.ungz;now=e.now

def jobs():
    return [dict(index=i,arm=a,seed=s,fold=f) for i,(f,s,a) in enumerate((f,s,a) for f in B['folds'] for s in B['seeds'] for a in B['arms'])]

def jobpath(j):return RUN/'jobs'/f"job_{j['index']:02d}"

def tensor_sha(model):
    h=hashlib.sha256()
    for name,x in model.state_dict().items():h.update(name.encode());h.update(x.detach().cpu().numpy().tobytes())
    return h.hexdigest()

def prepare():
    base=m.prepare();signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base;pools=[];coverage=[]
    for i,(p,l) in enumerate(ids):
        pos=[];nul=[];refs=m.old.references(ann,p,l)
        for a,pk,b in refs:
            if not np.all(y[i,a:b+1]!=-100):continue
            rr=[int(r) for r in anchors[p] if r-150<=a<=b<=r-11 and 10<=b-a<=100]
            if rr:pos.append((min(rr,key=lambda r:abs(r-b-40)),a,b))
        for r in anchors[p]:
            lo=max(0,int(r)-150);hi=max(0,int(r)-10)
            if lo<hi and 1000<=lo<hi<=4000 and np.all(y[i,lo:hi]!=-100) and not np.any(y[i,lo:hi]==1):nul.append((int(r),None,None))
        assert pos[:3]+nul[:3]==legacy[i],'Legacy training target pool identity'
        pools.append(dict(positive=pos,null=nul));coverage.append(dict(patient=p,li=l,reference=len(refs),positive_targets=len(pos),null_targets=len(nul)))
    return base,pools,coverage

@lru_cache(maxsize=128)
def pairs(anchor):
    a,b=m.old.legal_pairs(anchor)
    return torch.from_numpy(a),torch.from_numpy(b)

def terms(logits,target):
    anchor,a,b=target;ss,ee=pairs(anchor);lp=logits.log_softmax(0);lod=(lp[1]-torch.logsumexp(lp[[0,2,3]],0))/5;cs=nn.functional.pad(lod.cumsum(0),(1,0));score=cs[ee+1]-cs[ss]-math.log(len(ss));z=torch.logsumexp(torch.cat([score,score.new_zeros(1)]),0)
    if a is None:return z,logits.sum()*0
    good=(abs(ss-a)<=2)&(abs(ee-b)<=2);assert good.any(),'Positive target legal support'
    struct=z-torch.logsumexp(score[good],0)
    cost=torch.maximum(abs(ss-a),abs(ee-b)).to(score.dtype)/10
    tail=torch.logsumexp(score.log_softmax(0)+cost,0)
    return struct,tail

def chosen(pools,indices,seed,fold,epoch):
    out=[]
    for batch,i in enumerate(indices):
        for kind in ['positive','null']:
            pool=pools[int(i)][kind]
            if not pool:continue
            origin=int(hashlib.sha256(f"A30:{seed}:{fold}:{int(i)}:{kind}".encode()).hexdigest()[:16],16)%len(pool)
            index=(origin+epoch)%len(pool);out.append((batch,pool[index],kind,int(i),index))
    return out

def balanced(logits,selected):
    by={};tails={}
    for batch,t,kind,i,k in selected:
        a,b=terms(logits[batch],t);by.setdefault(batch,[]).append(a)
        if kind=='positive':tails.setdefault(batch,[]).append(b)
    zero=logits.sum()*0
    struct=torch.stack([torch.stack(v).mean() for v in by.values()]).mean() if by else zero
    tail=torch.stack([torch.stack(v).mean() for v in tails.values()]).mean() if tails else zero
    return struct,tail

def loss(logits,yy,weights,amb,arm,legacy,indices,selected):
    frame=m.frame_loss(logits,yy,weights,amb)
    if arm==B['arms'][0]:
        targets=[(batch,t) for batch,i in enumerate(indices) for t in legacy[int(i)]][:24]
        return frame+.1*m.interval_loss(logits,targets),dict(targets=len(targets),represented=len({a for a,t in targets}),tail=0.)
    struct,tail=balanced(logits,selected);value=frame+.1*struct
    if arm==B['arms'][2]:value=value+.05*tail
    return value,dict(targets=len(selected),represented=len({a for a,t,kind,i,k in selected}),tail=float(tail.detach()))

def synthetic_preflight():
    tests=d.preflight();torch.manual_seed(300);fake=[dict(positive=[(1400,1300,1340),(1800,1700,1740)],null=[(2200,None,None)]) for i in range(8)]
    picked=chosen(fake,np.arange(8),9,0,0);assert {a for a,t,k,i,j in picked}==set(range(8)) and len(picked)==16
    assert picked==chosen(fake,np.arange(8),9,0,0)
    nxt=chosen(fake,np.arange(8),9,0,1);assert all(a[4]!=b[4] for a,b in zip(picked,nxt) if a[2]=='positive')
    lg=torch.zeros(8,4,5000,requires_grad=True);s,t=balanced(lg,picked);assert torch.isfinite(s+t) and t>=0;(s+.05*t).backward();assert torch.isfinite(lg.grad).all() and all(lg.grad[b].abs().sum()>0 for b in range(8))
    null=chosen([dict(positive=[],null=[(1400,None,None)])],np.array([0]),0,0,0);nl=torch.zeros(1,4,5000,requires_grad=True);ns,nt=balanced(nl,null);assert nt==0;ns.backward();assert nl.grad[0,1,1300]>0
    aligned=torch.zeros(4,5000);aligned[1]= -8;aligned[1,1300:1341]=8
    shifted=torch.zeros_like(aligned);shifted[1]= -8;shifted[1,1250:1291]=8
    assert terms(aligned,(1400,1300,1340))[1]<terms(shifted,(1400,1300,1340))[1]
    z=torch.randn(2,4,5000,requires_grad=True);ts=[(0,(1400,1300,1340)),(1,(1400,None,None))];manual=torch.stack([terms(z[b],t)[0] for b,t in ts]).mean();old=m.interval_loss(z,ts);assert torch.allclose(manual,old,atol=1e-6,rtol=0)
    fl=torch.zeros(1,4,128,requires_grad=True);yy=torch.zeros(1,128,dtype=torch.long);yy[:,:10]=-100;fv=m.frame_loss(fl,yy,torch.ones(1,128),torch.zeros(1,128,dtype=torch.bool));fv.backward();assert fl.grad[:,:,:10].abs().max()==0
    return tests+['all8_items_positive_null_represented','deterministic_round_robin','late_batch_positions_receive_gradients','supported_null_has_no_endpoint_loss','tail_risk_aligned_lower_than_shifted','legacy_struct_loss_equivalence','frame_unknown_gradient_zero']
