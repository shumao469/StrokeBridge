"""Native 500 Hz frozen P/ECG-FM feature extraction and physical morphology."""
from a17r_common import *
import ast,sys,importlib.util,time,traceback
import numpy as np,pandas as pd,torch,wfdb
from torch import nn
from scipy.signal import butter,sosfiltfilt,find_peaks
from scipy.ndimage import uniform_filter1d
torch.set_num_threads(2)
class Block(nn.Module):
 def __init__(self,a,b):super().__init__();self.net=nn.Sequential(nn.Conv1d(a,b,7,padding=3),nn.GroupNorm(4,b),nn.GELU(),nn.Conv1d(b,b,7,padding=3),nn.GroupNorm(4,b),nn.GELU())
 def forward(self,x):return self.net(x)
class Net(nn.Module):
 def __init__(self):
  super().__init__();self.a=Block(2,16);self.b=Block(16,32);self.c=Block(32,64);self.d=Block(96,32);self.e=Block(48,16);self.head=nn.Conv1d(16,4,1)
 def forward(self,x):
  a=self.a(x);b=self.b(nn.functional.avg_pool1d(a,2));c=self.c(nn.functional.avg_pool1d(b,2));d=self.d(torch.cat([nn.functional.interpolate(c,size=b.shape[-1],mode='linear',align_corners=False),b],1));e=self.e(torch.cat([nn.functional.interpolate(d,size=a.shape[-1],mode='linear',align_corners=False),a],1));return self.head(e)
def components(prob,z,lead):
 mask=prob[1]>.15 if lead==1 else prob.argmax(0)==1
 edges=np.diff(np.r_[False,mask,False].astype(int));out=[]
 for a,b in zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)):
  b-=1
  if not 10<=b-a<=100:continue
  zz=z[a:b+1]-np.linspace(z[a],z[b],b-a+1);pk=a+int(np.argmax(abs(zz)))
  if 1000<=pk<4000:out.append(dict(a=int(a),b=int(b),pk=pk,conf=float(prob[1,a:b+1].mean())))
 return out
def med(v):return float(np.median(v)) if len(v) else np.nan
def morphology(z,c):
 a,b=c['a'],c['b'];w=z[a:b+1]-np.linspace(z[a],z[b],b-a+1);pos=max(float(w.max()),0);neg=max(float(-w.min()),0)
 typ='biphasic' if min(pos,neg)>.2*max(pos,neg) else ('positive' if pos>=neg else 'negative')
 edges=np.diff(np.r_[False,w<0,False].astype(int));spans=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)))
 # Last negative lobe must reach the final 20% of the candidate, not an initial negative lobe.
 terminal=[(s,e) for s,e in spans if e>=.8*len(w)]
 force=area=np.nan
 if terminal:
  s,e=terminal[-1];force=float(w[s:e].min()*10*(e-s)/500);area=float(w[s:e].sum()*10/500)
 elif pos>0:force=area=0.
 return dict(duration=(b-a)*2.,amp_pos=pos,amp_neg=-neg,net_area=float(np.trapz(w,dx=.002)),polarity=typ,ptf=force,terminal_area=area)
def r_features(z):
 band=sosfiltfilt(butter(3,[5,20],btype='bandpass',fs=500,output='sos'),z,axis=0)
 scale=np.maximum(np.quantile(abs(band),.95,axis=0),1e-8);energy=uniform_filter1d(np.median((band/scale)**2,axis=1),size=35)
 e=energy[1000:4000];base=float(np.median(e));spread=float(np.quantile(e,.95)-base)
 peaks,_=find_peaks(energy,distance=120,height=base+.25*spread,prominence=.15*spread)
 peaks=peaks[(peaks>=1000)&(peaks<4000)];rr=np.diff(peaks)/500.;rr=rr[(rr>=.28)&(rr<=2.)]
 vals=dict(rr_n=len(rr),r_count=len(peaks),rr_mean=med(rr),rr_sd=float(np.std(rr)) if len(rr) else np.nan,rr_cv=float(np.std(rr)/np.mean(rr)) if len(rr) else np.nan,hr_median=60/med(rr) if len(rr) else np.nan,rr_rmssd=float(np.sqrt(np.mean(np.diff(rr)**2))) if len(rr)>1 else np.nan,rr_pnn50=float(np.mean(abs(np.diff(rr))>.05)) if len(rr)>1 else np.nan)
 return peaks,vals
def p_features(physical,norm,prob,rpeaks):
 comps=[components(prob[i],norm[:,i],i) for i in range(12)];matches={};allm=[]
 for li,cc in enumerate(comps):
  for ri,r in enumerate(rpeaks):
   eligible=[c for c in cc if 20<=r-c['pk']<=150 and c['b']<r-10]
   if eligible:
    c=max(eligible,key=lambda q:(q['conf'],-abs(r-q['pk']-75)));m=morphology(physical[:,li],c);matches[(li,ri)]=(c,m);allm.append(m)
 out=dict(p_candidate_count=sum(map(len,comps)),p_matched_leadbeats=len(matches),p_matched_fraction=len(matches)/max(12*len(rpeaks),1),p_candidate_confidence=med([c['conf'] for cc in comps for c in cc]),p_duration_all_ms=med([m['duration'] for m in allm]))
 for li,name in [(1,'II'),(6,'V1')]:
  mm=[m for (l,r),(c,m) in matches.items() if l==li];out['p_duration_'+name+'_ms']=med([m['duration'] for m in mm]);out['p_count_'+name]=len(mm)
  for pol in ['positive','negative','biphasic']:out['p_'+name+'_'+pol+'_fraction']=float(np.mean([m['polarity']==pol for m in mm])) if mm else np.nan
 for li,name in enumerate(LEADS):
  mm=[m for (l,r),(c,m) in matches.items() if l==li];out['p_amp_'+name+'_mV']=med([m['amp_pos']+m['amp_neg'] for m in mm])
 v=[m for (l,r),(c,m) in matches.items() if l==6];out['p_terminal_force_V1_mm_s']=med([m['ptf'] for m in v if np.isfinite(m['ptf'])]);out['p_terminal_area_V1_mm_s']=med([m['terminal_area'] for m in v if np.isfinite(m['terminal_area'])])
 axes=[]
 for ri in range(len(rpeaks)):
  if (0,ri) in matches and (1,ri) in matches:
   ai=matches[(0,ri)][1]['net_area'];aii=matches[(1,ri)][1]['net_area'];yy=(2*aii-ai)/np.sqrt(3)
   if np.hypot(ai,yy)>1e-7:axes.append(float(np.arctan2(yy,ai)))
 out['p_axis_sin']=float(np.mean(np.sin(axes))) if axes else np.nan;out['p_axis_cos']=float(np.mean(np.cos(axes))) if axes else np.nan;out['p_axis_paired_beats']=len(axes)
 return out
def prepare_raw(x):
 phys=sosfiltfilt(butter(3,[.5,40],btype='bandpass',fs=500,output='sos'),x,axis=0)
 zz=phys-np.median(phys,axis=0);scale=np.quantile(zz,.95,axis=0)-np.quantile(zz,.05,axis=0);norm=np.clip(zz/np.maximum(scale,1e-8),-8,8).astype('float32')
 fm=x[1000:4000].T.copy();fm-=fm.mean(-1,keepdims=True);fm/=np.maximum(fm.std(-1,keepdims=True),1e-8)
 fm=sosfiltfilt(butter(4,40,btype='lowpass',fs=500,output='sos'),fm,axis=-1);fm-=fm.mean(-1,keepdims=True);fm/=np.maximum(fm.std(-1,keepdims=True),1e-8)
 return phys,norm,fm.astype('float32')
def load_fm(device):
 import ecgfm_frozen
 ecgfm_frozen.UPSTREAM=RUN/'upstream';m=ecgfm_frozen.FrozenECGFM(checkpoint=FM).to(device).eval();assert all(not p.requires_grad for p in m.parameters());return m
def labels():
 d=pd.read_csv(META);table=pd.read_csv(DATA/'scp_statements.csv',index_col=0);rh=set(table[table.rhythm.eq(1)].index);rows=[]
 assert d.patient_id.notna().all() and d.ecg_id.is_unique and d.groupby('patient_id').strat_fold.nunique().max()==1
 for r in d.to_dict('records'):
  code=ast.literal_eval(r['scp_codes']);keys=set(code);af='AFIB' in keys;sin=bool(keys&{'SR','STACH','SARRH','SBRAD'});aftext=any(s in str(r['report']).lower() for s in ['vorhofflimmern','atrial fibrillation']);reason=''
  if af and ('AFLT' in keys or sin):reason='mixed_AF_flutter_or_sinus'
  elif not af and aftext:reason='text_AF_without_AFIB'
  elif not af and not keys&rh:reason='no_rhythm_reference'
  role='train' if r['strat_fold']<=7 else {8:'select',9:'calibrate',10:'evaluate'}[r['strat_fold']]
  age=r['age'];top=bool(pd.notna(age) and age>89);age=90. if top else age;age=np.nan if pd.notna(age) and not 0<=age<=90 else age
  rows.append(dict(ecg_id=int(r['ecg_id']),patient_id=int(r['patient_id']),fold=int(r['strat_fold']),role=role,filename_hr=r['filename_hr'],label=int(af),label_eligible=not bool(reason),label_exclusion=reason,AFIB_value=code.get('AFIB',np.nan),AFLT='AFLT' in keys,SR=sin,age=age,age_topcoded=int(top),sex=r['sex'] if r['sex'] in [0,1] else np.nan))
 out=pd.DataFrame(rows);out.to_csv(RUN/'metadata_roles_labels.csv',index=False)
 write(RUN/'label_audit.json',dict(records=len(out),patients=int(out.patient_id.nunique()),eligible=int(out.label_eligible.sum()),eligible_AF=int(out.loc[out.label_eligible,'label'].sum()),exclusions=out.loc[~out.label_eligible,'label_exclusion'].value_counts().to_dict(),role_counts=out[out.label_eligible].groupby('role').agg(records=('ecg_id','size'),AF=('label','sum'),patients=('patient_id','nunique')).reset_index().to_dict('records'),patient_role_overlap=0,diagnosis='concurrent AF reference; official code presence with explicit conflict quarantine'))
 return out
def extract():
 check_freeze();start=time.monotonic();d=labels();manifest={}
 for ln in (DATA/'SHA256SUMS.txt').read_text().splitlines():
  h,n=ln.split(maxsplit=1);manifest[n.lstrip('*').strip().removeprefix('./')]=h
 chunks=RUN/'features';chunks.mkdir(exist_ok=True);device='cuda:0';pm=Net().to(device);pm.load_state_dict(torch.load(PW,map_location=device,weights_only=True));pm.eval().requires_grad_(False);fm=load_fm('cuda:1')
 write(RUN/'encoder_audit.json',dict(P_frozen=True,P_checkpoint_SHA256=sha(PW),FM_frozen=True,FM_SHA256=sha(FM),FM_load=fm.load_audit,FM_source_exposure='PTB-XL part of PhysioNet2021 pretraining; this is not pretraining-unseen validation'))
 with torch.inference_mode():parity=fm.upstream_parity(torch.zeros(1,12,3000,device='cuda:1'))
 assert parity<1e-6;write(RUN/'upstream_parity.json',dict(max_error=parity))
 rows=[];raws=[];ids=[];batch=12
 for offset in range(0,len(d),batch):
  if time.monotonic()-start>6*3600:raise TimeoutError('6-hour extraction bound')
  dest=chunks/f'batch_{offset:06d}.npz';csv=chunks/f'batch_{offset:06d}.csv';audit=chunks/f'batch_{offset:06d}_audit.csv'
  if dest.exists() and csv.exists() and audit.exists():continue
  features=[];inputs=[];auditrows=[];chunkids=[]
  for r in d.iloc[offset:offset+batch].to_dict('records'):
   p=DATA/r['filename_hr'];hashes={}
   for ext in ['.hea','.dat']:
    name=r['filename_hr']+ext;h=sha(DATA/name);assert h==manifest[name],f'Provider hash mismatch {name}';hashes[ext]=h
   rec=wfdb.rdrecord(str(p));assert rec.fs==500 and rec.sig_len==5000 and rec.n_sig==12 and set(rec.units)=={'mV'}
   names=[n.upper() for n in rec.sig_name];assert sorted(names)==sorted(LEADS);x=rec.p_signal[:,[names.index(n) for n in LEADS]]
   qc=''
   if not np.isfinite(x).all():qc='nonfinite'
   elif any(np.ptp(x[:,li])<1e-6 for li in [0,1,6,7,8,9,10,11]):qc='constant_required_lead'
   auditrows.append(dict(ecg_id=r['ecg_id'],QC_exclusion=qc,header_SHA256=hashes['.hea'],data_SHA256=hashes['.dat'],fs=500,n=5000,unit='mV'))
   if qc:continue
   phys,norm,ff=prepare_raw(x);rr,rf=r_features(phys)
   xx=np.stack([norm.T,np.gradient(norm.T,axis=-1)*5],axis=1)
   with torch.inference_mode():
    t=torch.from_numpy(xx).to(device);prob=((pm(t)+pm(-t))/2).softmax(1).cpu().numpy()
   feat=dict(ecg_id=r['ecg_id'],**rf,**p_features(phys,norm,prob,rr));feat['quality_range_mV']=float(np.median(np.ptp(x,axis=0)));feat['quality_baseline_wander']=float(np.median(np.std(x-phys,axis=0)));feat['quality_highfreq_ratio']=float(np.var(np.diff(x,axis=0))/max(np.var(x),1e-12));features.append(feat);inputs.append(ff);chunkids.append(r['ecg_id'])
  if inputs:
   with torch.inference_mode():emb=fm(torch.from_numpy(np.stack(inputs)).to('cuda:1')).cpu().numpy()
   assert emb.shape==(len(inputs),768) and np.isfinite(emb).all()
  else:emb=np.empty((0,768),dtype='float32')
  pd.DataFrame(features).to_csv(csv,index=False);pd.DataFrame(auditrows).to_csv(audit,index=False);tmp=dest.with_suffix('.tmp.npz');np.savez_compressed(tmp,ecg_id=chunkids,embedding=emb);tmp.replace(dest)
  if offset%(batch*20)==0:status('native500_feature_extraction',state='running',processed=min(offset+batch,len(d)),total=len(d),fits_started=0)
 allf=pd.concat([pd.read_csv(p) for p in sorted(chunks.glob('batch_*.csv')) if not p.name.endswith('_audit.csv')],ignore_index=True);aa=pd.concat([pd.read_csv(p) for p in sorted(chunks.glob('*_audit.csv'))],ignore_index=True)
 assert len(aa)==len(d) and aa.ecg_id.is_unique and allf.ecg_id.is_unique
 ee={}
 for p in sorted(chunks.glob('batch_*.npz')):
  z=np.load(p)
  for key,v in zip(z['ecg_id'],z['embedding']):assert int(key) not in ee;ee[int(key)]=v
 allf=allf.sort_values('ecg_id').reset_index(drop=True);assert set(ee)==set(allf.ecg_id)
 np.savez_compressed(RUN/'embeddings.npz',ecg_id=allf.ecg_id.to_numpy(),embedding=np.stack([ee[int(i)] for i in allf.ecg_id]))
 aa.to_csv(RUN/'waveform_audit.csv',index=False);allf.to_csv(RUN/'physical_and_candidate_features.csv',index=False)
 out=d.merge(allf,on='ecg_id',how='inner',validate='one_to_one');out=out[out.label_eligible].copy();out.to_csv(RUN/'analysis_dataset.csv',index=False)
 write(RUN/'features_complete.json',dict(records_total=len(d),waveform_pass=len(allf),analysis_records=len(out),analysis_patients=int(out.patient_id.nunique()),analysis_AF=int(out.label.sum()),provider_hashes_verified=2*len(aa),native_fs=500,feature_SHA256=sha(RUN/'physical_and_candidate_features.csv'),embedding_SHA256=sha(RUN/'embeddings.npz'),analysis_SHA256=sha(RUN/'analysis_dataset.csv'),P_network_training_epochs=0,scope='current AF diagnosis; candidates not verified P waves'))
 status('features_complete',state='awaiting_bounded_fits',records=len(out),fits_started=0)
