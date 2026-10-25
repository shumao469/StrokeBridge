"""Patient-isolated development comparison of frozen delineator inference rules."""
from common import *
import numpy as np,pandas as pd,torch,importlib.util,sys,traceback
from scipy.signal import butter,sosfiltfilt,find_peaks
from scipy.ndimage import uniform_filter1d
from intervals import posterior,selftest
torch.set_num_threads(1);torch.set_num_interop_threads(1)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
START=time.monotonic();MAX=7200;SEEDS=[20261020,20261021];GRID=[.25,.5,.65,.75,.85,.95]
OUT=RUN/'measurement'
def check():deadline(START,MAX)
def rpeaks(x):
    band=sosfiltfilt(butter(3,[5,20],btype='bandpass',fs=500,output='sos'),x,axis=0)
    scale=np.maximum(np.quantile(abs(band),.95,axis=0),1e-8)
    energy=uniform_filter1d(np.median((band/scale)**2,axis=1),size=35)
    e=energy[1000:4000];base=float(np.median(e));spread=float(np.quantile(e,.95)-base)
    pk,_=find_peaks(energy,distance=120,height=base+.25*spread,prominence=.15*spread)
    return pk[(pk>=1000)&(pk<4000)]
def anchored(pred,rr):
    out=[];used=set()
    for r in rr:
        eligible=[(j,p) for j,p in enumerate(pred) if j not in used and 20<=r-p['peak']<=150 and p['offset']<r-10]
        if eligible:
            j,p=max(eligible,key=lambda jp:(jp[1]['confidence'],-abs(r-jp[1]['peak']-75)))
            used.add(j);out.append(p)
    return out
def evaluate(records,ann,signals,labels,api,threshold=0.):
    rows=[];errors=[];pol=[]
    for q in records:
        pat,li=q['patient'],q['li'];ref=[tuple(v) for v in ann[pat]['leads'][li]['p'] if 1000<=v[1]<4000]
        pred=[v for v in q['pred'] if v['confidence']>threshold];pairs=api.match(ref,pred);matched={i for i,j,k in pairs};used={j for i,j,k in pairs}
        fp=unknown=0
        for j,p in enumerate(pred):
            if j not in used:
                if (labels[pat,li,p['onset']:p['offset']+1]!=-100).all():fp+=1
                else:unknown+=1
        de=[]
        for i,j,iou in pairs:
            a,pk,b=ref[i];pr=pred[j];v=2*((pr['offset']-pr['onset'])-(b-a));de.append(abs(v))
            errors.append(dict(patient=pat,lead=li,onset_error_ms=2*(pr['onset']-a),offset_error_ms=2*(pr['offset']-b),duration_error_ms=v))
        for i,(a,pk,b) in enumerate(ref):pol.append(dict(patient=pat,lead=li,polarity=api.polarity(signals[pat,:,li],a,b),matched=i in matched))
        nr=len(ref);nm=len(pairs);npred=len(pred);fail=sum(v>20 for v in de)
        rows.append(dict(patient=pat,lead=li,reference=nr,predicted=npred,matched=nm,reference_background_FP=fp,unknown_outputs=unknown,error_events=nr+npred-2*nm+fail,duration_failures=fail))
    return pd.DataFrame(rows),pd.DataFrame(errors),pd.DataFrame(pol)
def point(frames):
    c,e,p=frames;nr=int(c.reference.sum());n=int(c.predicted.sum());m=int(c.matched.sum())
    return dict(reference=nr,predicted=n,matched=m,coverage=m/max(nr,1),precision_conservative=m/max(n,1),reference_background_FP=int(c.reference_background_FP.sum()),unknown_outputs=int(c.unknown_outputs.sum()),unknown_output_fraction=float(c.unknown_outputs.sum()/max(n,1)),error_events_per_reference=float(c.error_events.sum()/max(nr,1)),duration_MAE_ms=float(e.duration_error_ms.abs().mean()) if len(e) else None)
def main():
    assert not (OUT/'started.json').exists(),'No automatic restart'
    put(OUT/'started.json',dict(time=now(),pid=os.getpid()));verify();put(OUT/'mock_tests.json',selftest())
    src=BASE/'runs/phase_A9_20261017_v1/code/a9_method.py'
    spec=importlib.util.spec_from_file_location('a25_pinned_a9',src);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api)
    # The imported module sets 2 intra-op threads; restore the shared 1-thread contract.
    torch.set_num_threads(1)
    zz=np.load(A9/'input/data.npz');signals=zz['signals'];labels=zz['labels'];ann=read(A9/'input/pwave_annotations.json');roles=read(A9/'input/patient_roles.json')
    assert signals.shape==(200,5000,12) and labels.shape==(200,12,5000)
    assert sorted(sum([f['outer'] for f in roles],[]))==list(range(200))
    for f in roles:
        sets=[set(f[k]) for k in ['train','selection','calibration','outer']]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:])
    raw=api.channels(np.stack([signals[0,:,1],signals[0,:,6],signals[1,:,1],signals[1,:,6]]))
    model=api.Net().eval().requires_grad_(False);model.load_state_dict(torch.load(A9/'C1_sign_orbit/seed_20261020/fold_0/best.pt',map_location='cpu',weights_only=True))
    tick=time.monotonic()
    with torch.inference_mode():
        x=torch.from_numpy(raw);prob=((model(x)+model(-x))/2).softmax(1).numpy()
        mirrored=((model(-x)+model(x))/2).softmax(1).numpy()
    parity=float(np.max(abs(prob-mirrored)));assert parity<=1e-7
    for j in range(4):
        for r in rpeaks(signals[j//2]):posterior(prob[j,1],r,signals[j//2,:,1 if j%2==0 else 6])
    elapsed=time.monotonic()-tick;projection=elapsed/4*1040*1.8
    put(OUT/'preflight.json',dict(state='PASS' if projection<MAX-(time.monotonic()-START) else 'TIMING_STOP',seconds_per_4_leads=elapsed,conservative_projected_seconds=projection,sign_parity=parity,threads=torch.get_num_threads(),clinical_no_P_reference=False))
    if projection>MAX-(time.monotonic()-START):raise TimeoutError('Projected 1040 lead evaluations exceed fixed budget')
    allcounts=[];allerrors=[];allpol=[];choices=[];predictions=[]
    for seed in SEEDS:
      for fold,f in enumerate(roles):
        check();model.load_state_dict(torch.load(A9/f'C1_sign_orbit/seed_{seed}/fold_{fold}/best.pt',map_location='cpu',weights_only=True));model.eval()
        def predict(pats):
            outputs={k:[] for k in ['continuous','hard_R','ordered_R']}
            for pat in pats:
                check();r=rpeaks(signals[pat]);xx=torch.from_numpy(api.channels(signals[pat,:,[1,6]]))
                with torch.inference_mode():pb=((model(xx)+model(-xx))/2).softmax(1).numpy()
                assert np.isfinite(pb).all()
                for j,li in enumerate([1,6]):
                    sig=signals[pat,:,li];hard=[p for p in api.components(pb[j],sig) if 1000<=p['peak']<4000]
                    soft=[posterior(pb[j,1],ri,sig) for ri in r];soft=[v for v in soft if v is not None and 1000<=v['peak']<4000]
                    for name,pr in [('continuous',hard),('hard_R',anchored(hard,r)),('ordered_R',soft)]:outputs[name].append(dict(patient=pat,li=li,pred=pr))
            return outputs
        cal=predict(f['calibration']);thresholds={}
        for li in [1,6]:
            gold=point(evaluate([q for q in cal['hard_R'] if q['li']==li],ann,signals,labels,api))
            vals=[]
            for t in GRID:
                v=point(evaluate([q for q in cal['ordered_R'] if q['li']==li],ann,signals,labels,api,t));vals.append(dict(threshold=t,**v))
            ok=[v for v in vals if v['coverage']>=gold['coverage']-.02]
            picked=min(ok,key=lambda v:(v['error_events_per_reference'],-v['coverage'],-v['threshold'])) if ok else vals[0]
            thresholds[li]=picked['threshold'];choices.append(dict(seed=seed,fold=fold,lead=li,calibration_feasible=bool(ok),hard_reference=gold,grid=vals,chosen=picked))
        out=predict(f['outer'])
        for name,records in out.items():
            for li in [1,6]:
                rec=[q for q in records if q['li']==li];t=thresholds[li] if name=='ordered_R' else 0.
                frames=evaluate(rec,ann,signals,labels,api,t)
                for target,frame in zip([allcounts,allerrors,allpol],frames):
                    frame['seed']=seed;frame['fold']=fold;frame['arm']=name;target.append(frame)
                if name=='ordered_R':predictions.extend([dict(seed=seed,fold=fold,threshold=t,**q) for q in rec])
        put(OUT/'calibration_choices.json',choices)
        status('measurement',state='running',seed=seed,fold=fold,elapsed_seconds=time.monotonic()-START)
    c=pd.concat(allcounts,ignore_index=True);e=pd.concat(allerrors,ignore_index=True);p=pd.concat(allpol,ignore_index=True)
    c.to_csv(OUT/'patient_counts.csv',index=False);e.to_csv(OUT/'boundary_errors.csv',index=False);p.to_csv(OUT/'polarity_recall.csv',index=False);put(OUT/'outer_interval_outputs.json',predictions)
    summary=[];polarity=[]
    for (seed,arm,li),cc in c.groupby(['seed','arm','lead']):
        ee=e[(e.seed==seed)&(e.arm==arm)&(e.lead==li)];pp=p[(p.seed==seed)&(p.arm==arm)&(p.lead==li)]
        summary.append(dict(seed=int(seed),arm=arm,lead=int(li),**point((cc,ee,pp))))
        for typ,g in pp.groupby('polarity'):polarity.append(dict(seed=int(seed),arm=arm,lead=int(li),polarity=typ,reference=len(g),recall=float(g.matched.mean())))
    ss=pd.DataFrame(summary);ps=pd.DataFrame(polarity);ss.to_csv(OUT/'summary.csv',index=False);ps.to_csv(OUT/'polarity_summary.csv',index=False)
    gates=[]
    for seed in SEEDS:
        checks={}
        for li in [1,6]:
            a=ss[(ss.seed==seed)&(ss.arm=='ordered_R')&(ss.lead==li)].iloc[0];b=ss[(ss.seed==seed)&(ss.arm=='hard_R')&(ss.lead==li)].iloc[0]
            checks[f'lead{li}_coverage']=bool(a.coverage>=b.coverage-.02)
            checks[f'lead{li}_precision']=bool(a.precision_conservative>=b.precision_conservative-.02)
            checks[f'lead{li}_duration']=bool(pd.notna(a.duration_MAE_ms) and a.duration_MAE_ms<=20 and pd.notna(b.duration_MAE_ms) and a.duration_MAE_ms<=b.duration_MAE_ms+2)
        for typ in ['negative','biphasic']:
            a=ps[(ps.seed==seed)&(ps.arm=='ordered_R')&(ps.lead==6)&(ps.polarity==typ)];b=ps[(ps.seed==seed)&(ps.arm=='hard_R')&(ps.lead==6)&(ps.polarity==typ)]
            checks[typ+'_gain5pp']=bool(len(a)==1 and len(b)==1 and a.iloc[0].reference>=20 and a.iloc[0].recall>=b.iloc[0].recall+.05)
        gates.append(dict(seed=seed,checks=checks,pass_all=all(checks.values())))
    verify();put(OUT/'completion.json',dict(time=now(),state='complete',engineering_gate_pass=all(g['pass_all'] for g in gates),gates=gates,neural_epochs=0,supervised_fits=0,independent_external=False,clinical_no_P_certified=False,new_full_PTB_extraction_allowed=False,NC_submission_ready=False,wall_seconds=time.monotonic()-START))
    status('measurement',state='complete',engineering_gate_pass=all(g['pass_all'] for g in gates))
if __name__=='__main__':
    try:main()
    except Exception as e:status('measurement',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
