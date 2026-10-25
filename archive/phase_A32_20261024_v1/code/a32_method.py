"""Patient-complete target supervision and a separately declared window-aware decoder."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import sys,math,hashlib,importlib.util
from pathlib import Path
from functools import lru_cache
import numpy as np
import torch
from torch import nn
from scipy.ndimage import uniform_filter1d
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A30=BASE/'runs/phase_A30_20261024_v1';A31=BASE/'runs/phase_A31_20261024_v1'
spec=importlib.util.spec_from_file_location('a32_frozen_a30_library',A30/'code/method.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
m=old.m;d=old.d;e=old.e;read=old.read;put=old.put;sha=old.sha;gz=old.gz;ungz=old.ungz;now=old.now
B=read(RUN/'budget.json');torch.set_num_threads(1);torch.set_grad_enabled(True)
tensor_sha=old.tensor_sha
def jobs():return [dict(index=i,fold=f,seed=s,arm=a) for i,(f,s,a) in enumerate((f,s,a) for f in B['folds'] for s in B['seeds'] for a in B['arms'])]
def jobpath(j):return RUN/'jobs'/f"job_{j['index']:02d}"
def choice(j):
    p=m.path(dict(j,arm='C2_null_ordered'))/'design_selection.json';c=read(p)['choice']
    return dict(mode=c['mode'],threshold=c['threshold'],source=str(p),no_search=True)

def cell(r,anchors,n=5000):
    prev=max([int(v) for v in anchors if v<r],default=-11)
    return max(0,prev+11,int(r)-350),min(n-1,int(r)-11),max(0,int(r)-250)

@lru_cache(maxsize=4096)
def pair_arrays(r,anchors,n=5000):
    lo,hi,endlo=cell(r,anchors,n);v=np.arange(lo,hi+1);a,b=np.meshgrid(v,v,indexing='ij');ok=(b-a>=10)&(b-a<=100)&(b>=endlo)
    return a[ok].astype(np.int64),b[ok].astype(np.int64)

def target_anchor(a,b,anchors):
    r=next((int(v) for v in anchors if v>b),None)
    if r is None or r-b>250:return None
    lo,hi,endlo=cell(r,tuple(anchors))
    return r if lo<=a<=b<=hi and b>=endlo and 10<=b-a<=100 else None

def waveform_anchor(p,anchors):
    if p['kind']=='ordered':return p['source_R']
    r=next((int(v) for v in anchors if v>p['offset']),None)
    if r is None or r-p['offset']>250 or any(p['onset']<=v<=p['offset'] for v in anchors):return None
    return r

def decode(prob,z,anchors):
    """No annotation argument: extended ordered lattice plus UNCLIPPED continuous spans."""
    anchors=tuple(map(int,anchors));p=prob[1].astype('float64');lod=(np.log(np.maximum(p,1e-12))-np.log(np.maximum(1-p,1e-12)))/5;cs=np.r_[0,np.cumsum(lod)];out=[]
    for r in anchors:
        if not 1000<r<4500:continue
        ss,ee=pair_arrays(r,anchors,len(p))
        if not len(ss):continue
        score=cs[ee+1]-cs[ss]-math.log(len(ss));mx=max(0,float(score.max()));ww=np.exp(score-mx);pn=float(ww.sum()/(ww.sum()+math.exp(-mx)));ww/=ww.sum()
        def quant(v):
            cdf=np.cumsum(np.bincount(v,weights=ww,minlength=len(p)));return [int(min(len(p)-1,np.searchsorted(cdf,q))) for q in [.25,.5,.75]]
        aq,bq=quant(ss),quant(ee);a,b=aq[1],bq[1];pk=d.peak(z,a,b)
        assert 10<=b-a<=100 and target_anchor(a,b,anchors)==r,'Extended posterior median geometry'
        if 1000<=pk<4000:out.append(dict(onset=a,offset=b,peak=pk,confidence=min(pn,float(p[a:b+1].mean())),non_null_mass=pn,box=[a,a,b,b],scale=max(5.,.5*max(aq[2]-aq[0],bq[2]-bq[0])),kind='ordered',source_R=r,boundary_rule='extended_marginal_median',endpoint_IQR=[aq,bq]))
    sm=uniform_filter1d(prob.astype('float64'),size=5,axis=-1,mode='nearest');hard=(sm[2]>sm[1])|(sm[3]>sm[1]);mask=(sm[1]>=.5)&~hard;diff=np.diff(np.r_[False,mask,False].astype(int));runs=list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)-1));joined=[]
    for a,b in runs:
        if joined and a-joined[-1][1]-1<=10 and not hard[joined[-1][1]+1:a].any():joined[-1]=(joined[-1][0],b)
        else:joined.append((a,b))
    seen=set()
    for a,b in joined:
        while a>0 and sm[1,a-1]>=.2 and not hard[a-1]:a-=1
        while b<len(p)-1 and sm[1,b+1]>=.2 and not hard[b+1]:b+=1
        if not 10<=b-a<=100:continue
        pk=d.peak(z,a,b)
        if not 1000<=pk<4000 or (a,b) in seen:continue
        seen.add((a,b));out.append(dict(onset=int(a),offset=int(b),peak=int(pk),confidence=float(p[a:b+1].mean()),box=[int(a),int(a),int(b),int(b)],scale=5.,kind='continuous',source_R=None,boundary_rule='unclipped_fixed_hysteresis'))
    for i,pv in enumerate(out):pv['uid']=i;assert 0<=pv['onset']<=pv['peak']<=pv['offset']<len(p) and 0<=pv['confidence']<=1
    return out

def merge(pp,anchors):
    groups={};unresolved=[]
    for p in pp:
        r=waveform_anchor(p,anchors)
        if r is None:unresolved.append(dict(p,rejection='waveform_beat_unresolved'));continue
        groups.setdefault(r,[]).append(dict(p,wave_R=r))
    selected=[];ledger=[]
    for r,values in sorted(groups.items()):
        kept=d.nms(values);winner=max(kept,key=lambda v:(v['confidence']*math.sqrt((v['offset']-v['onset'])/100),v['confidence'],v['offset']-v['onset'],-v['uid']));selected.append(winner);ledger.append(dict(wave_R=r,raw_count=len(values),after_local_NMS=len(kept),winner_uid=winner['uid']))
    assert len({p['wave_R'] for p in selected})==len(selected)
    return selected,dict(groups=ledger,unresolved=unresolved)

def decorate(pp,anchors,qrs):return [dict(p,wave_R=waveform_anchor(p,anchors),expert_beat=d.map_expert(waveform_anchor(p,anchors),qrs)) for p in pp]

def prepare():
    base,oldpools,oldcoverage=old.prepare();signals,ann,roles,ids,x,y,anchors,partial,legacy,labels=base;targets=[];coverage=[]
    for i,(p,li) in enumerate(ids):
        refs=m.old.references(ann,p,li);positive=[];null=[];aa=tuple(map(int,anchors[p]))
        for k,(a,pk,b) in enumerate(refs):
            if not np.all(y[i,a:b+1]!=-100):continue
            left=max(0,a-100,int((refs[k-1][2]+a)/2)+1 if k else 0);right=min(4999,b+100,int((b+refs[k+1][0])/2) if k+1<len(refs) else 4999)
            r=target_anchor(a,b,aa);outside=r is None or not(r-150<=a<=b<=r-11)
            positive.append(dict(onset=a,peak=pk,offset=b,R=r,left=left,right=right,known=np.flatnonzero(y[i,left:right+1]!=-100).tolist(),outside_old=outside))
        for r in aa:
            if not 1000<r<4500:continue
            lo,hi,_=cell(r,aa)
            if 1000<=lo<=hi<4000 and np.all(y[i,lo:hi+1]!=-100) and not np.any(y[i,lo:hi+1]==1) and len(pair_arrays(r,aa)[0]):null.append(r)
        targets.append(dict(positive=positive,null=null,anchors=aa));coverage.append(dict(patient=p,li=li,reference=len(refs),supported=len(positive),ordered_supported=sum(v['R'] is not None for v in positive),outside_old=sum(v['outside_old'] for v in positive),missing_ordered_identity=sum(v['R'] is None for v in positive),supported_null=len(null)))
    return base,targets,coverage

def batches(ids,patients):
    for start in range(0,len(patients),B['patients_per_batch']):
        pats=list(map(int,patients[start:start+B['patients_per_batch']]));ii=np.array([i for p in pats for i,(pat,li) in enumerate(ids) if pat==p],dtype=np.int64)
        assert 0<len(ii)<=8 and {ids[i][0] for i in ii}==set(pats)
        yield ii

def posterior(logits,r,anchors):
    a,b=pair_arrays(int(r),tuple(anchors),logits.shape[-1]);ss=torch.from_numpy(a);ee=torch.from_numpy(b);lp=logits.log_softmax(0);lod=(lp[1]-torch.logsumexp(lp[[0,2,3]],0))/5;cs=nn.functional.pad(lod.cumsum(0),(1,0));score=cs[ee+1]-cs[ss]-math.log(len(ss));z=torch.logsumexp(torch.cat([score,score.new_zeros(1)]),0)
    return ss,ee,score,z

def ordered_terms(cache,a=None,b=None):
    ss,ee,score,z=cache
    if a is None:return z,None
    good=(abs(ss-a)<=2)&(abs(ee-b)<=2);assert bool(good.any()),'Supported full target must have legal endpoint pair'
    struct=z-torch.logsumexp(score[good],0);cost=torch.maximum(abs(ss-a),abs(ee-b)).to(score.dtype)/10;tail=torch.logsumexp(score.log_softmax(0)+cost,0)
    return struct,tail

def transition_tail(pr,t):
    # Training-only reference-local region; does not enter decode, selection or calibration.
    sm=nn.functional.avg_pool1d(nn.functional.pad(pr[None,None],(2,2),mode='replicate'),5,1)[0,0];left,right=t['left'],t['right'];v=sm[left:right+1];prev=sm[max(0,left-1):right] if left else torch.cat([sm[:1],sm[:right]])
    nxt=sm[left+1:right+2] if right+1<len(sm) else torch.cat([sm[left+1:],sm[-1:]])
    idx=torch.arange(left,right+1,device=pr.device);on=nn.functional.logsigmoid(8*(v-prev-.05))+v.clamp_min(1e-8).log();off=nn.functional.logsigmoid(8*(v-nxt-.05))+v.clamp_min(1e-8).log();known=torch.tensor(t.get('known',list(range(right-left+1))),dtype=torch.long);idx=idx[known];on=on[known];off=off[known];out=[]
    for logp,end in [(on,t['onset']),(off,t['offset'])]:out.append(torch.logsumexp(logp.log_softmax(0)+(idx-end).abs().to(pr.dtype)/10,0))
    return torch.logsumexp(torch.stack(out),0)-math.log(2)

def differentiable_conf(pr,p,cache):
    mean=pr[p['onset']:p['offset']+1].mean()
    if p['kind']=='ordered':
        _,_,scores,z=cache[p['wave_R']];pn=(torch.logsumexp(scores,0)-z).exp();return torch.minimum(mean,pn)
    return mean

def patient_loss(logits,yy,weights,amb,arm,legacy,indices,targets,ids,signals,cutoff):
    frame=m.frame_loss(logits,yy,weights,amb);zero=logits.sum()*0
    if arm==B['arms'][0]:
        ts=[(batch,t) for batch,i in enumerate(indices) for t in legacy[int(i)]][:24]
        return frame+.1*m.interval_loss(logits,ts),dict(positive=sum(t[1] is not None for b,t in ts),null=sum(t[1] is None for b,t in ts),outside_old=0,endpoint_terms=0,report_missing=0)
    by={};meta=dict(positive=0,null=0,outside_old=0,endpoint_terms=0,report_missing=0);probs=logits.softmax(1)
    for k,i in enumerate(indices):
        pat,li=ids[int(i)];tt=targets[int(i)];pr=probs[k,1];structpos=[];structnull=[];tails=[];align=[];cache={}
        def get(r):
            if r not in cache:cache[r]=posterior(logits[k],r,tt['anchors'])
            return cache[r]
        raw=decode(probs[k].detach().numpy(),signals[pat,:,li],tt['anchors']);selected,_=merge([v for v in raw if v['confidence']>=cutoff],tt['anchors']);selected={p['wave_R']:p for p in selected}
        for t in tt['positive']:
            a,b,r=t['onset'],t['offset'],t['R'];risk=[transition_tail(pr,t)];meta['positive']+=1;meta['outside_old']+=t['outside_old'];meta['endpoint_terms']+=1
            if r is not None:
                s,tail=ordered_terms(get(r),a,b);structpos.append(s);risk.append(tail)
            tails.append(torch.logsumexp(torch.stack(risk),0)-math.log(len(risk)));winner=selected.get(r) if r is not None else None;targetmean=pr[a:b+1].mean()
            if winner is None:align.append(-targetmean.clamp_min(1e-8).log());meta['report_missing']+=1
            else:
                conf=differentiable_conf(pr,winner,{winner['wave_R']:get(winner['wave_R'])}) if winner['kind']=='ordered' else differentiable_conf(pr,winner,{})
                score=conf*math.sqrt((winner['offset']-winner['onset'])/100);truth=targetmean*math.sqrt((b-a)/100);cost=.2*min(1,max(abs(a-winner['onset']),abs(b-winner['offset']))/50);align.append(nn.functional.relu(score-truth+cost))
        for r in tt['null']:
            s,_=ordered_terms(get(r));structnull.append(s);meta['null']+=1
            if r in selected:
                p=selected[r];conf=differentiable_conf(pr,p,{r:get(r)}) if p['kind']=='ordered' else differentiable_conf(pr,p,{})
                align.append(-(1-conf).clamp_min(1e-8).log())
        groups=[torch.stack(v).mean() for v in [structpos,structnull] if v];item=by.setdefault(pat,dict(struct=[],tails=[],align=[]))
        if groups:item['struct'].append(torch.stack(groups).mean())
        item['tails'].extend(tails);item['align'].extend(align)
    def patientmean(key,entropic=False):
        vv=[]
        for entry in by.values():
            ts=entry[key]
            if ts:vv.append(torch.logsumexp(torch.stack(ts),0)-math.log(len(ts)) if entropic else torch.stack(ts).mean())
        return torch.stack(vv).mean() if vv else zero
    return frame+.1*patientmean('struct')+.05*patientmean('tails',True)+.1*patientmean('align'),meta

def synthetic_tests():
    tests=d.preflight();torch.manual_seed(32)
    anchors=(2500,3127,3700);pr=np.zeros((4,5000));pr[0]=.98;pr[1]=.02;pr[0,2952:2980]=.05;pr[1,2952:2980]=.95;z=np.sin(np.arange(5000)/5)
    out=decode(pr,z,anchors);cont=[v for v in out if v['kind']=='continuous'];assert any(v['onset']<2977 and v['offset']>=2979 for v in cont),'Unclipped regression'
    pp,_=merge(out,anchors);assert len({v['wave_R'] for v in pp})==len(pp);assert target_anchor(2964,2982,anchors)==3127
    fake=[dict(positive=[dict(onset=1300,peak=1320,offset=1340,R=1400,left=1200,right=1440,outside_old=False),dict(onset=2964,peak=2973,offset=2982,R=3127,left=2864,right=3082,outside_old=True)],null=[],anchors=(1400,2500,3127,3700)) for _ in range(2)]
    lg=torch.zeros(2,4,5000,requires_grad=True);yy=torch.zeros(2,5000,dtype=torch.long);sig=np.zeros((1,5000,12));val,meta=patient_loss(lg,yy,torch.ones_like(yy,dtype=torch.float32),torch.zeros_like(yy,dtype=torch.bool),B['arms'][1],[[],[]],np.array([0,1]),fake,[(0,1),(0,6)],sig,0.)
    assert meta['positive']==4 and meta['outside_old']==2 and meta['endpoint_terms']==4;val.backward();assert torch.isfinite(lg.grad).all() and all(lg.grad[k,:,2950:3000].abs().sum()>0 for k in [0,1])
    nl=torch.zeros(4,5000,requires_grad=True);s,t=ordered_terms(posterior(nl,1400,(1400,2500)));assert t is None;s.backward();assert nl.grad[1,1300]>0
    f=torch.zeros(1,4,128,requires_grad=True);lab=torch.zeros(1,128,dtype=torch.long);lab[:,:10]=-100;m.frame_loss(f,lab,torch.ones(1,128),torch.zeros(1,128,dtype=torch.bool)).backward();assert f.grad[:,:,:10].abs().max()==0
    assert list(batches([(0,1),(0,6),(1,1),(1,6)],[1,0]))[0].tolist()==[2,3,0,1]
    assert waveform_anchor(dict(kind='continuous',onset=1350,offset=1450),[1400,1800]) is None
    return tests+['unclipped_data130_geometry_regression','extended_ordered_support','both_leads_all_positive_endpoints_no_prefix','late_outside_old_window_gradient','null_has_no_endpoint_target','unknown_frame_gradient_zero','whole_patient_batching','continuous_R_crossing_rejected']
