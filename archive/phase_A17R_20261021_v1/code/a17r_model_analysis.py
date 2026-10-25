"""Frozen 28-fit evaluation: fold8 selection, fold9 calibration, fold10 readout."""
from a17r_common import *
import warnings,time,traceback,platform
import numpy as np,pandas as pd,joblib
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score,average_precision_score,brier_score_loss,roc_curve,log_loss
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
def fit_budget(name,model,x,y):
 p=RUN/'fit_ledger.json';ledger=read(p) if p.exists() else []
 assert not any(z['name']==name for z in ledger),'Never implicitly refit an existing attempt'
 assert len(ledger)<MAX_FITS
 row=dict(number=len(ledger)+1,name=name,started_at=now(),status='started',rows=len(y),AF=int(np.sum(y)))
 ledger.append(row);write(p,ledger);status('bounded_classifier_fits',state='running',attempts=len(ledger),limit=MAX_FITS,current=name)
 try:
  with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
   warnings.simplefilter('always');model.fit(x,y)
  row['warnings']=[str(q.message) for q in caught];row['converged']=not any(issubclass(q.category,ConvergenceWarning) for q in caught)
  if not row['converged']:raise RuntimeError('Non-converged classifier: stop; no budget-free retry')
  row.update(status='complete',finished_at=now());ledger[-1]=row;write(p,ledger);return model
 except Exception:
  row.update(status='failed_no_retry',finished_at=now(),error=traceback.format_exc());ledger[-1]=row;write(p,ledger);raise
def logit(p):
 p=np.clip(p,1e-6,1-1e-6);return np.log(p/(1-p)).reshape(-1,1)
def score(y,p,threshold=.5):
 q=p>=threshold;tp=int(np.sum(q&(y==1)));tn=int(np.sum(~q&(y==0)));fp=int(np.sum(q&(y==0)));fn=int(np.sum(~q&(y==1)))
 return dict(n=len(y),AF=int(np.sum(y)),prevalence=float(np.mean(y)),AUROC=float(roc_auc_score(y,p)),AP=float(average_precision_score(y,p)),Brier=float(brier_score_loss(y,p)),log_loss=float(log_loss(y,p,labels=[0,1])),mean_probability=float(np.mean(p)),threshold=float(threshold),sensitivity=tp/max(tp+fn,1),specificity=tn/max(tn+fp,1),PPV=tp/max(tp+fp,1),TP=tp,TN=tn,FP=fp,FN=fn)
def patient_bootstrap(d,pairs):
 rng=np.random.default_rng(SEED);patients=np.unique(d.patient_id);groups={v:np.flatnonzero(d.patient_id.to_numpy()==v) for v in patients};allrows=[];result=[];y=d.label.to_numpy()
 # Resample patient clusters; retain all ECGs of each selected patient.
 point={}
 for a,b in pairs:
  pa=d[a].to_numpy();pb=d[b].to_numpy();point[(a,b)]=np.array([average_precision_score(y,pa)-average_precision_score(y,pb),roc_auc_score(y,pa)-roc_auc_score(y,pb),np.mean((pa-y)**2-(pb-y)**2)])
 bags={p:[] for p in pairs}
 for _ in range(2000):
  ix=np.concatenate([groups[p] for p in rng.choice(patients,len(patients),replace=True)]);yy=y[ix]
  if np.unique(yy).size<2:continue
  for a,b in pairs:
   pa=d[a].to_numpy()[ix];pb=d[b].to_numpy()[ix];bags[(a,b)].append([average_precision_score(yy,pa)-average_precision_score(yy,pb),roc_auc_score(yy,pa)-roc_auc_score(yy,pb),np.mean((pa-yy)**2-(pb-yy)**2)])
 for a,b in pairs:
  vv=np.array(bags[(a,b)]);q=np.quantile(vv,[.025,.975,.0125,.9875],axis=0)
  for k,metric in enumerate(['AP','AUROC','Brier']):result.append(dict(method=a,baseline=b,metric=metric,delta=float(point[(a,b)][k]),lower95=float(q[0,k]),upper95=float(q[1,k]),lower97_5=float(q[2,k]),upper97_5=float(q[3,k]),replicates=len(vv),patients=len(patients),scope='fixed-model clustered patient; selection/refit uncertainty excluded'))
 return pd.DataFrame(result)
def figures(pred,metrics,paired):
 out=RUN/'figures';out.mkdir(exist_ok=True)
 plt.rcParams.update({'font.size':10,'font.family':'DejaVu Sans','axes.spines.right':False,'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none'})
 colors=['#777777','#31688e','#b98e33','#d57932','#7d5fa6','#318e8b','#a73c60']
 def save(fig,name,caption):
  fig.subplots_adjust(bottom=.23,top=.81);fig.text(.06,.025,caption,fontsize=8,va='bottom')
  for ext in ['png','svg','pdf']:fig.savefig(out/f'{name}.{ext}',dpi=220,facecolor='white')
  plt.close(fig)
 from sklearn.metrics import precision_recall_curve
 fig,axs=plt.subplots(1,2,figsize=(12,5.5));fig.suptitle('A17R patient-isolated concurrent AF recognition: viewed fold 10')
 for i,a in enumerate(ARMS):
  yy=pred.label.to_numpy();p=pred[a].to_numpy();f,t,_=roc_curve(yy,p);prec,rec,_=precision_recall_curve(yy,p)
  axs[0].plot(f,t,c=colors[i],lw=1.4,ls='--' if i%2 else '-',label=f'{a} ({roc_auc_score(yy,p):.3f})');axs[1].plot(rec,prec,c=colors[i],lw=1.4,ls='--' if i%2 else '-',label=f'{a} ({average_precision_score(yy,p):.3f})')
 axs[0].plot([0,1],[0,1],c='#aaaaaa',lw=.8);axs[1].axhline(pred.label.mean(),c='#aaaaaa',lw=.8)
 for ax in axs:ax.set(xlim=(0,1),ylim=(0,1));ax.legend(frameon=False,fontsize=7)
 axs[0].set(xlabel='False positive rate',ylabel='Sensitivity',title='ROC (AUROC)');axs[1].set(xlabel='Recall',ylabel='Precision',title='Precision-recall (AP)')
 save(fig,'Figure1_ROC_PR','Same evaluation ECGs across seven frozen feature groups; native 500 Hz; calibration/thresholds from fold 9.\nPTB-XL previously viewed and ECG-FM source exposed; concurrent diagnosis, not incident AF or stroke prediction.')
 fig,axs=plt.subplots(1,3,figsize=(12,4.8));fig.suptitle('A17R paired increment from candidate atrial morphology')
 for ax,metric,title in zip(axs,['AP','AUROC','Brier'],['AP difference (higher better)','AUROC difference (higher better)','Brier difference (lower better)']):
  for i,(a,b) in enumerate([('D_RR_P','D_RR'),('D_RR_P_FM','D_RR_FM')]):
   r=paired[paired.method.eq(a)&paired.baseline.eq(b)&paired.metric.eq(metric)].iloc[0];lo=r.lower97_5 if metric=='AP' else r.lower95;hi=r.upper97_5 if metric=='AP' else r.upper95;ax.plot([lo,hi],[i,i],color=colors[3+i*3],lw=2);ax.scatter(r.delta,i,color=colors[3+i*3],marker=['o','s'][i]);ax.text(.02,.94-i*.11,f'{r.delta:+.3f} [{lo:+.3f}, {hi:+.3f}]',transform=ax.transAxes,fontsize=8)
  ax.axvline(0,color='#777',lw=.8,ls='--');ax.set_yticks([0,1],['P beyond RR','P beyond RR+FM'],fontsize=8);ax.set_ylim(-.5,1.7);ax.set_title(title,fontsize=10)
 fig.subplots_adjust(wspace=.65)
 save(fig,'Figure2_paired_increment','2,000 patient-cluster bootstrap replicates; fixed-model intervals. AP: 97.5% intervals for two primary comparisons.\nAUROC/Brier: descriptive 95% intervals; no implication of equivalence from an interval crossing zero.')
 cal=[];fig,ax=plt.subplots(figsize=(7,5.2));fig.suptitle('A17R calibration on viewed evaluation patients')
 for i,a in enumerate(ARMS):
  tmp=pd.DataFrame({'p':pred[a],'y':pred.label});tmp['bin']=pd.qcut(tmp.p,q=8,duplicates='drop');g=tmp.groupby('bin',observed=True).agg(probability=('p','mean'),observed_AF=('y','mean'),n=('y','size')).reset_index(drop=True);g['arm']=a;cal.append(g);ax.plot(g.probability,g.observed_AF,color=colors[i],marker=['o','s','^','D','v','P','x'][i],ms=3,label=a)
 ax.plot([0,1],[0,1],color='#777',ls='--',lw=.8);ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean calibrated probability',ylabel='Observed AF proportion');ax.legend(frameon=False,fontsize=7)
 save(fig,'Figure3_calibration','Eight descriptive equal-count bins per model; fold 9 logistic calibration, fold 10 readout.\nNatural selected cohort prevalence retained; no longitudinal risk claim.')
 pd.concat(cal,ignore_index=True).to_csv(RUN/'calibration_bins.csv',index=False)
def run():
 start=time.monotonic();check_freeze();lock=read(RUN/'analysis_source_lock.json');assert all(sha(p)==h for p,h in lock.items())
 assert not (RUN/'fit_ledger.json').exists(),'No automatic fits after existing budget ledger'
 d=pd.read_csv(RUN/'analysis_dataset.csv');z=np.load(RUN/'embeddings.npz');loc={int(i):v for i,v in zip(z['ecg_id'],z['embedding'])};em=np.stack([loc[int(i)] for i in d.ecg_id]);fmcols=[f'FM_{i:03d}' for i in range(768)];df=pd.concat([d.reset_index(drop=True),pd.DataFrame(em,columns=fmcols)],axis=1)
 rr=[c for c in d if c.startswith(('rr_','hr_','quality_')) or c=='r_count'];pcols=[c for c in d if c.startswith('p_')];demo=['age','sex','age_topcoded']
 families={'D':demo,'D_RR':demo+rr,'D_P':demo+pcols,'D_RR_P':demo+rr+pcols,'D_FM':demo+fmcols,'D_RR_FM':demo+rr+fmcols,'D_RR_P_FM':demo+rr+pcols+fmcols}
 write(RUN/'feature_families.json',families)
 masks={r:df.role.eq(r).to_numpy() for r in ['train','select','calibrate','evaluate']};sets={r:set(df.loc[m,'patient_id']) for r,m in masks.items()}
 assert all(not a&b for i,a in enumerate(sets.values()) for b in list(sets.values())[:i]);assert all(df.loc[m,'label'].nunique()==2 for m in masks.values())
 y=df.label.to_numpy();choices=[];packs={};thresholds={};allpred=df[['ecg_id','patient_id','fold','role','label','AFLT','SR','age','sex']].copy();models=RUN/'models';models.mkdir(exist_ok=True)
 # No evaluation-score access in candidate selection or calibration loop.
 for arm in ARMS:
  cols=families[arm];xx=df[cols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float);best=None
  for c in CS:
   model=make_pipeline(SimpleImputer(strategy='median',add_indicator=True,keep_empty_features=True),StandardScaler(),LogisticRegression(C=c,max_iter=2500,solver='lbfgs',random_state=SEED))
   fit_budget(f'{arm}/C={c}',model,xx[masks['train']],y[masks['train']]);ps=model.predict_proba(xx[masks['select']])[:,1];ap=float(average_precision_score(y[masks['select']],ps));choices.append(dict(arm=arm,C=c,selection_AP=ap,training_records=int(masks['train'].sum()),selection_records=int(masks['select'].sum())))
   if best is None or ap>best[0]:best=(ap,c,model)
  _,c,model=best;raw=model.predict_proba(xx)[:,1];cal=LogisticRegression(C=1.,max_iter=1000,random_state=SEED);fit_budget(f'{arm}/Platt_fold9',cal,logit(raw[masks['calibrate']]),y[masks['calibrate']]);prob=cal.predict_proba(logit(raw))[:,1];fpr,tpr,tt=roc_curve(y[masks['calibrate']],prob[masks['calibrate']]);valid=np.flatnonzero((tpr>=.95)&np.isfinite(tt));idx=min(valid,key=lambda i:(fpr[i],-tt[i]));th=float(tt[idx]);thresholds[arm]=dict(C=c,threshold=th,selected_on='fold8 AP',calibrated_on='fold9',threshold_policy='fold9 sensitivity >=.95, minimum FPR then largest threshold',calibration_sensitivity=float(tpr[idx]),calibration_specificity=float(1-fpr[idx]))
  pack=dict(feature_columns=cols,classifier=model,calibrator=cal,threshold=th,feature_families=arm,source_lock=read(RUN/'source_lock.json'),target='concurrent AF recognition',folds=dict(train=list(range(1,8)),select=8,calibrate=9,evaluate=10),P_frozen=True,FM_frozen=True)
  joblib.dump(pack,models/f'{arm}.joblib');packs[arm]=pack;allpred[arm+'__raw']=raw;allpred[arm]=prob
  if time.monotonic()-start>12*3600:raise TimeoutError('12-hour phase bound')
 pd.DataFrame(choices).to_csv(RUN/'candidate_selection.csv',index=False);write(RUN/'selection_calibration_thresholds.json',thresholds);assert len(read(RUN/'fit_ledger.json'))==28
 allpred.to_csv(RUN/'role_predictions.csv',index=False);write(RUN/'models_frozen_before_evaluation.json',dict(time=now(),model_SHA256={p.name:sha(p) for p in models.glob('*.joblib')},threshold_SHA256=sha(RUN/'selection_calibration_thresholds.json'),fits=28))
 test=allpred[allpred.role.eq('evaluate')].copy();test.to_csv(RUN/'evaluation_predictions.csv',index=False);metrics=[];strata=[];bins=[]
 for arm in ARMS:
  yy=test.label.to_numpy();pp=test[arm].to_numpy();th=thresholds[arm]['threshold']
  for policy,col in [('raw',arm+'__raw'),('calibrated',arm)]:metrics.append(dict(arm=arm,policy=policy,**score(yy,test[col].to_numpy(),th if policy=='calibrated' else .5)))
  strata_groups={'sinus_vs_AF':test[(test.label.eq(1))|test.SR.eq(True)],'AFLT_only_negatives':test[test.AFLT.eq(True)&test.label.eq(0)],'all_nonAF':test[test.label.eq(0)],'first_ECG_per_patient':test.sort_values('ecg_id').drop_duplicates('patient_id')}
  for name,g in strata_groups.items():
   if g.empty:continue
   if g.label.nunique()==2:strata.append(dict(arm=arm,stratum=name,**score(g.label.to_numpy(),g[arm].to_numpy(),th)))
   else:strata.append(dict(arm=arm,stratum=name,n=len(g),AF=int(g.label.sum()),false_positive_fraction=float((g[arm]>=th).mean()),threshold=th))
  for label,gg in [('calibrate',allpred[allpred.role.eq('calibrate')]),('evaluate',test)]:
   q=gg[arm].to_numpy();yy2=gg.label.to_numpy();lg=logit(q).ravel();pr=np.clip(q,1e-6,1-1e-6);pbar=pr*(1-pr);sc=np.array([np.sum(yy2-pr),np.sum((yy2-pr)*lg)]);inf=np.array([[pbar.sum(),np.sum(pbar*lg)],[np.sum(pbar*lg),np.sum(pbar*lg*lg)]]);bins.append(dict(arm=arm,role=label,calibration_score_intercept=float(sc[0]),calibration_score_slope=float(sc[1]),note='Calibration score diagnostics at intercept=0 slope=1; no extra diagnostic classifier fit'))
 mm=pd.DataFrame(metrics);mm.to_csv(RUN/'metrics.csv',index=False);pd.DataFrame(strata).to_csv(RUN/'strata_metrics.csv',index=False);pd.DataFrame(bins).to_csv(RUN/'calibration_score_diagnostics.csv',index=False)
 status('paired_patient_statistics',state='running',fits_completed=28,max_fits=30)
 pairs=[('D_RR_P','D_RR'),('D_RR_P_FM','D_RR_FM'),('D_RR_FM','D_RR')];paired=patient_bootstrap(test,pairs);paired.to_csv(RUN/'paired_patient_intervals.csv',index=False);figures(test,mm,paired)
 checks=[]
 for a,b in pairs[:2]:
  ap=paired[paired.method.eq(a)&paired.baseline.eq(b)&paired.metric.eq('AP')].iloc[0];br=paired[paired.method.eq(a)&paired.baseline.eq(b)&paired.metric.eq('Brier')].iloc[0];checks.append(dict(method=a,baseline=b,AP_97_5_lower=float(ap.lower97_5),Brier_95_upper=float(br.upper95),strong_joint_increment=bool(ap.lower97_5>0 and br.upper95<=0)))
 # Recompute crucial heldout metrics from persisted probabilities.
 reload=pd.read_csv(RUN/'evaluation_predictions.csv')
 for r in mm[mm.policy.eq('calibrated')].itertuples():assert abs(average_precision_score(reload.label,reload[r.arm])-r.AP)<1e-12
 check_freeze();assert all(sha(p)==h for p,h in lock.items());ledger=read(RUN/'fit_ledger.json');assert len(ledger)==28 and all(r['status']=='complete' for r in ledger)
 clinical=dict(current_AF_diagnosis_only=True,independent_external_validation=False,expert_review='pending',hospital_branch='waiting',stroke_prediction=False,NC_submission_ready=False)
 write(RUN/'increment_results.json',dict(primary_comparisons=checks,**clinical))
 table=mm[mm.policy.eq('calibrated')].set_index('arm')[['n','AF','AUROC','AP','Brier','sensitivity','specificity','PPV','threshold']]
 text='A17R有界修复结果\n28/30次分类器/概率校准拟合，0新增P网络训练；原生500Hz，固定患者角色。\n'+json.dumps(read(RUN/'features_complete.json'),ensure_ascii=False,indent=2)+'\n\n校准后fold10（已查看开发评价）：\n'+table.to_string()+'\n\n患者配对增量及区间：\n'+paired.to_string(index=False)+'\n\n主要增量联合描述：\n'+json.dumps(checks,ensure_ascii=False,indent=2)+'\n\n标签采用官方SCP与明确冲突隔离；AF中的候选片段不是经专家确认P波。PTFV1与电轴使用滤波mV、同R序号匹配和明确物理定义，尚未成为独立临床测量真值。固定模型患者bootstrap不包含重新训练/选择不确定性。\nPTB-XL与fold10已被项目查看、属于ECG-FM预训练来源；不作新独立验证、未来AF或卒中预测解释。概率和阈值仅来自fold9校准，评价操作点未作Youden优化。\n公开左房模型先看left_atrial_source_audit，未加入本轮冻结7组，不追加拟合。医院/专家资料缺口保持pending。\n下一步：如P特征增量不成立，保留阴性结果，优先检查独立来源/可靠测量；如成立，先外来源冻结迁移，再考虑同人心房影像与脑侧表型/纵向事件，不能以AF AUC代替临床效用。\n'
 (RUN/'A17R修复结果与后续门槛.txt').write_text(text,encoding='utf-8')
 write(RUN/'numeric_completion_receipt.json',dict(state='completed_pending_export_review',records=len(df),test_records=len(test),test_patients=int(test.patient_id.nunique()),fits=28,max_fits=30,new_P_epochs=0,patient_role_overlap=0,sources_unchanged=True,primary_increment_checks=checks,**clinical))
 status('bounded_analysis_complete',state='pending_export_review',fits_completed=28,max_fits=30,**clinical)
