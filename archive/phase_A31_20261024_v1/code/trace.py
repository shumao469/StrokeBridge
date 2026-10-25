"""Instrumented continuous decoding; never changes the frozen decision rule."""
import numpy as np
from scipy.ndimage import uniform_filter1d

def continuous(prob,z,anchors,decoder):
    p=prob[1].astype('float64');sm=uniform_filter1d(prob.astype('float64'),size=5,axis=-1,mode='nearest');hard=(sm[2]>sm[1])|(sm[3]>sm[1]);mask=(sm[1]>=.5)&~hard
    diff=np.diff(np.r_[False,mask,False].astype(int));runs=list(zip(np.flatnonzero(diff==1),np.flatnonzero(diff==-1)-1));joined=[]
    for a,b in runs:
        if joined and a-joined[-1][1]-1<=10 and not hard[joined[-1][1]+1:a].any():joined[-1]=(joined[-1][0],b)
        else:joined.append((a,b))
    seen=set();out=[];ledger=[]
    for initial_a,initial_b in joined:
        a,b=initial_a,initial_b
        while a>0 and sm[1,a-1]>=.2 and not hard[a-1]:a-=1
        while b<len(p)-1 and sm[1,b+1]>=.2 and not hard[b+1]:b+=1
        pk=decoder.peak(z,a,b);r=next((int(r) for r in anchors if r>pk),None);row=dict(joined=[int(initial_a),int(initial_b)],grown=[int(a),int(b)],initial_peak=int(pk),generating_R=r)
        if r is None:row['discard']='no_R_after_initial_peak';ledger.append(row);continue
        a=max(a,r-150);b=min(b,r-11);row['clipped']=[int(a),int(b)];row['clipping_changed']=row['grown']!=row['clipped']
        if not 10<=b-a<=100:row['discard']='duration';ledger.append(row);continue
        pk=decoder.peak(z,a,b)
        if not 1000<=pk<4000:row['discard']='noncentral';ledger.append(row);continue
        if (a,b,r) in seen:row['discard']='duplicate_span_R';ledger.append(row);continue
        seen.add((a,b,r));proposal=dict(onset=int(a),offset=int(b),peak=int(pk),confidence=float(p[a:b+1].mean()),box=[int(a),int(a),int(b),int(b)],scale=5.,kind='continuous',source_R=None,boundary_rule='fixed_hysteresis');out.append(proposal);row.update(discard=None,final=proposal,assigned_R=decoder.wave_anchor(proposal,anchors));ledger.append(row)
    return out,ledger
