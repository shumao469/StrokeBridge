"""Ordered P-interval versus explicit null; no training or hidden threshold search.

This is a fixed structured inference prototype, not a new clinical P/no-P truth.
"""
import numpy as np
from scipy.special import logsumexp, expit

def posterior(p,r,z):
    p=np.asarray(p,float);z=np.asarray(z,float)
    assert p.shape==z.shape and p.ndim==1 and np.isfinite(p).all() and np.isfinite(z).all()
    lo=max(0,int(r)-150);hi=min(len(p)-1,int(r)-11)
    if hi-lo<10:return None
    aa,dd=np.meshgrid(np.arange(lo,hi-9),np.arange(10,101),indexing='ij')
    bb=aa+dd;ok=bb<=hi;a=aa[ok];b=bb[ok]
    q=np.clip(p,1e-6,1-1e-6);cs=np.r_[0.,np.cumsum(np.log(q/(1-q)))]
    scores=(cs[b+1]-cs[a])/5.-np.log(len(a))
    ls=logsumexp(scores);mass=float(expit(ls));weights=np.exp(scores-ls)
    j=int(np.argmax(scores));on=int(a[j]);off=int(b[j]);w=z[on:off+1]-np.linspace(z[on],z[off],off-on+1)
    peak=on+int(np.argmax(abs(w)))
    def summary(v):
        order=np.argsort(v);cum=np.cumsum(weights[order]);qs=v[order][np.minimum(np.searchsorted(cum,[.025,.975]),len(v)-1)]
        mean=float(weights@v);var=float(weights@((v-mean)**2));return mean,var,[int(i) for i in qs]
    am,av,aci=summary(a);bm,bv,bci=summary(b)
    dur=2.*(b-a);dm=float(weights@dur);dv=float(weights@((dur-dm)**2))
    return dict(onset=on,offset=off,peak=peak,confidence=mass,onset_mean_sample=am,onset_variance_sample2=av,offset_mean_sample=bm,offset_variance_sample2=bv,onset_nominal95=aci,offset_nominal95=bci,duration_mean_ms=dm,duration_variance_ms2=dv,conditional_entropy=float(-(weights*np.log(np.maximum(weights,1e-300))).sum()),legal_intervals=len(a),reference='model-derived, uncalibrated; null is abstention, not certified no P')

def selftest():
    p=np.full(500,.01);z=np.zeros(500)
    assert posterior(p,350,z)['confidence']<.05
    p[250:291]=.99;z[250:291]=np.sin(np.linspace(0,np.pi,41))
    v=posterior(p,350,z);w=posterior(p,350,-z)
    assert v['confidence']>.9 and abs(v['onset']-250)<=2 and abs(v['offset']-290)<=2
    for k in ['onset','offset','confidence','duration_mean_ms']:assert v[k]==w[k]
    assert 10<=v['offset']-v['onset']<=100 and v['offset']<340
    assert v['onset_nominal95'][0]<=v['onset_mean_sample']<=v['onset_nominal95'][1]
    assert posterior(np.full(10,.5),5,np.zeros(10)) is None
    try:posterior(np.full(20,np.nan),15,np.zeros(20))
    except AssertionError:pass
    else:raise AssertionError('Nonfinite admission')
    return {'finite_and_bounds':'PASS','null_rejection':'PASS','positive_boundaries':'PASS','sign_invariance':'PASS','empty_window':'PASS','clinical_no_P_certified':False}
