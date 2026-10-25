"""Gated A24 scientific vectors and native-object interchange; zero fits/inference."""
from pathlib import Path
import os, sys, json, hashlib, shutil, textwrap
_policy_path=Path(__file__).resolve().parents[1]/'FIGURE_POLICY.json'
if __name__=='__main__' and _policy_path.exists() and not any(x in sys.argv for x in ('--check-code','--backend-smoke')):
    _policy=json.loads(_policy_path.read_text(encoding='utf-8-sig'))
    if not _policy.get('allow_new_result_figures',False):
        print('RESULT_DRAWING_DISABLED_BY_USER_PROMPT_ONLY_POLICY');raise SystemExit(0)
for key in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[key]='2'
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator
from scipy.interpolate import PchipInterpolator
from sklearn.metrics import roc_curve, precision_recall_curve, roc_auc_score, average_precision_score, brier_score_loss
import fitz
import execute_E_wsl_20261023_v2 as execution

e=execution.e;P=e.P;U=e.shared.U;REC=execution.REC;A=e.shared.A;B=e.shared.B;NR=P/'recovery_N_v1'
OUT=U/'manuscript_NC_A24_20261023';S=OUT/'source_data';Q=OUT/'qa'
INK='#243B53';BLUE='#4C98C6';GOLD='#D3A13B';RED='#C96C58';PURPLE='#8F82AD';GREY='#7D8B99'
COLORS={'E1':BLUE,'E1_VN':'#287D9F','E1_VNM':'#075D70','E2':PURPLE,'E2_VN':'#9A684C','E2_VNM':RED,
        'N_VN':GREY,'N_current':BLUE,'N_displaced':RED,'N_anchor':GOLD,'N_permuted':PURPLE}
NAMES={'E1':'E1','E1_VN':'E1 + V/N','E1_VNM':'E1 + V/N/M','E2':'E2','E2_VN':'E2 + V/N','E2_VNM':'E2 + V/N/M',
       'N_VN':'V/N','N_current':'Current M','N_displaced':'Displaced M','N_anchor':'R-window M','N_permuted':'Permuted M'}
ALL=['E1','E1_VN','E1_VNM','E2','E2_VN','E2_VNM'];NN=['N_VN','N_current','N_displaced','N_anchor','N_permuted']
inventory=[];legends=[]
for name in ('arial.ttf','arialbd.ttf','ariali.ttf','arialbi.ttf'):
    font=Path('DRIVE_ROOT/Windows/Fonts')/name
    if font.exists():font_manager.fontManager.addfont(str(font))
FONT='Arial' if any(f.name=='Arial' for f in font_manager.fontManager.ttflist) else 'Liberation Sans'
plt.rcParams.update({'font.family':FONT,'font.size':8,'axes.labelsize':8,'axes.titlesize':9,
    'xtick.labelsize':7,'ytick.labelsize':7,'legend.fontsize':6.7,'axes.spines.top':False,'axes.spines.right':False,
    'axes.linewidth':.65,'lines.linewidth':1.15,'pdf.fonttype':42,'svg.fonttype':'none','text.color':INK,
    'axes.labelcolor':INK,'savefig.facecolor':'white','path.simplify':False})

def dump(path,value):e.put(path,value)
def table(name,frame):frame.to_csv(S/name,index=False)
def grid(title,cols=2):
    fig,axs=plt.subplots(3,cols,figsize=(180/25.4,238/25.4))
    fig.subplots_adjust(left=.135,right=.97,top=.92,bottom=.09,wspace=.68 if cols==2 else .86,hspace=.85)
    fig.suptitle(title,x=.065,y=.98,ha='left',fontsize=12,fontweight='bold')
    return fig,axs.ravel()
def panel(ax,letter,title):
    ax.text(-.22,1.16,letter,transform=ax.transAxes,fontsize=11,fontweight='bold')
    ax.set_title(title,loc='left',fontsize=9,pad=10)
def foot(fig,text):fig.text(.065,.024,text,fontsize=6.9,va='bottom',linespacing=1.4)
def diagram(ax):ax.axis('off');ax.set(xlim=(0,1),ylim=(0,1))
def label(ax,x,y,text,size=8,color=INK,**kwargs):ax.text(x,y,text,fontsize=size,color=color,**kwargs)
def node(ax,x,y,w,h,text,color=BLUE):
    ax.add_patch(Rectangle((x,y),w,h,ec=color,fc='white',lw=.9))
    label(ax,x+w/2,y+h/2,text,7.8,ha='center',va='center',linespacing=1.4)
def arrow(ax,x,y,xx,yy,color=GREY):ax.annotate('',xy=(xx,yy),xytext=(x,y),arrowprops={'arrowstyle':'->','color':color,'lw':.9})
def smooth(x,y):
    q=pd.DataFrame({'x':x,'y':y}).groupby('x',as_index=False).y.mean().sort_values('x')
    xx=np.linspace(q.x.min(),q.x.max(),301)
    return xx,np.clip(PchipInterpolator(q.x,q.y)(xx),0,1)
def intervals(ax,frame,metric,arms,point='estimate',lo='lower',hi='upper'):
    q=frame[frame.metric.eq(metric)].set_index('arm')
    for j,name in enumerate(arms):
        row=q.loc[name];ax.plot([row[lo],row[hi]],[j,j],c=COLORS[name],lw=1.6)
        ax.plot(row[point],j,'o',ms=4,c=COLORS[name])
    ax.set_yticks(range(len(arms)),[NAMES[n] for n in arms]);ax.set_ylim(len(arms)-.5,-.5);ax.set_xlabel(metric)
    ax.grid(axis='x',color='#E6EBEF',lw=.4)
    ax.xaxis.set_major_locator(MaxNLocator(3))
def paired(ax,frame,metric,pairs):
    for j,(method,base) in enumerate(pairs):
        row=frame[frame.metric.eq(metric)&frame.method.eq(method)&frame.baseline.eq(base)].iloc[0]
        color=COLORS[method];ax.plot([row.lower,row.upper],[j,j],c=color,lw=1.7);ax.plot(row.delta,j,'o',c=color,ms=4)
    ax.axvline(0,c=GREY,lw=.7,ls='--');ax.set_yticks(range(len(pairs)),[NAMES[a]+'\nvs '+NAMES[b] for a,b in pairs],fontsize=6.5)
    ax.set_ylim(len(pairs)-.5,-.5);ax.set_xlabel('Paired change in '+metric)
    ax.xaxis.set_major_locator(MaxNLocator(3))

def pdf_objects(file,regions):
    """Export true PDF drawing paths and selectable text for native PPTX authoring."""
    with fitz.open(file) as doc:
        page=doc[0];assert not page.get_images(full=True),'Unexpected raster layer'
        W,H=page.rect.width,page.rect.height
        def group(rect):
            cx=(rect[0]+rect[2])/2;cy=(rect[1]+rect[3])/2
            if rect[2]-rect[0]>=W*.98 and rect[3]-rect[1]>=H*.98:return 'global'
            if cy<H*.055 or cy>H*.958:return 'global'
            return min(regions,key=lambda p:(cx-(p['bbox'][0]+p['bbox'][2])/2)**2+
                       (cy-(p['bbox'][1]+p['bbox'][3])/2)**2)['letter']
        paths=[]
        for drawing in page.get_drawings():
            commands=[];last=None
            def move(p):
                nonlocal last
                xy=[float(p.x),float(p.y)]
                if last!=xy:commands.append({'moveTo':{'x':xy[0],'y':xy[1]}})
                last=xy
            def line(p):
                nonlocal last
                commands.append({'lineTo':{'x':float(p.x),'y':float(p.y)}});last=[float(p.x),float(p.y)]
            for item in drawing['items']:
                if item[0]=='l':move(item[1]);line(item[2])
                elif item[0]=='c':
                    move(item[1]);commands.append({'cubicBezTo':dict(x1=float(item[2].x),y1=float(item[2].y),
                        x2=float(item[3].x),y2=float(item[3].y),x=float(item[4].x),y=float(item[4].y))})
                    last=[float(item[4].x),float(item[4].y)]
                elif item[0]=='re':
                    r=item[1];move(r.tl);line(r.tr);line(r.br);line(r.bl);commands.append({'close':{}});last=None
                elif item[0]=='qu':
                    r=item[1];move(r.ul);line(r.ur);line(r.lr);line(r.ll);commands.append({'close':{}});last=None
                else:raise AssertionError(('Unsupported PDF vector primitive',item[0]))
            if drawing.get('closePath') and commands and 'close' not in commands[-1]:commands.append({'close':{}})
            if commands:
                rect=list(drawing['rect']);paths.append({'commands':commands,'bbox':rect,'group':group(rect),
                    'stroke':drawing.get('color'),'fill':drawing.get('fill'),'line_width':drawing.get('width',.7),
                    'stroke_opacity':drawing.get('stroke_opacity',1),'fill_opacity':drawing.get('fill_opacity',1),
                    'dashes':drawing.get('dashes','[] 0')})
        texts=[]
        for block in page.get_text('dict')['blocks']:
            for line_ in block.get('lines',[]):
                for span in line_['spans']:
                    if not span['text'].strip():continue
                    texts.append({'text':span['text'],'bbox':list(span['bbox']),'origin':list(span['origin']),
                        'size':span['size'],'font':span['font'],'color':span['color'],'direction':list(line_['dir']),
                        'group':group(span['bbox'])})
        assert len(texts)>20 and len(paths)>10
        return {'width':W,'height':H,'paths':paths,'texts':texts,'panel_regions':regions}

def save(fig,axs,name,title,legend):
    fig.canvas.draw();renderer=fig.canvas.get_renderer();W,H=fig.get_size_inches()*72
    regions=[]
    for j,ax in enumerate(axs):
        b=ax.get_tightbbox(renderer).transformed(fig.dpi_scale_trans.inverted())
        regions.append({'letter':chr(97+j),'bbox':[b.x0*72,H-b.y1*72,b.x1*72,H-b.y0*72]})
    dst=OUT/('supplementary_figures' if name.startswith('Supplementary') else 'figures')
    for ext in ('svg','pdf','png'):fig.savefig(dst/(name+'.'+ext),dpi=600)
    svg=(dst/(name+'.svg')).read_text();assert '<text' in svg and '<image' not in svg
    objects=pdf_objects(dst/(name+'.pdf'),regions);dump(S/(name+'_native_objects.json'),objects)
    with fitz.open(dst/(name+'.pdf')) as doc:doc[0].get_pixmap(matrix=fitz.Matrix(2,2)).save(Q/(name+'_from_pdf.png'))
    inventory.append({'figure':name,'title':title,'panels':len(axs),'rows':3,'width_mm':180,'height_mm':238,
        'SVG_text_objects':svg.count('<text'),'PDF_native_paths':len(objects['paths']),'PDF_text_spans':len(objects['texts']),
        'objects':str((S/(name+'_native_objects.json')).relative_to(OUT)),
        'PDF':str((dst/(name+'.pdf')).relative_to(OUT)),'PNG':str((dst/(name+'.png')).relative_to(OUT))})
    legends.append(name+' | '+title+'. '+legend);plt.close(fig)

def main():
    if _policy_path.exists() and not json.loads(_policy_path.read_text(encoding='utf-8-sig')).get('allow_new_result_figures',False):
        raise RuntimeError('Result drawing disabled by the user; written design only')
    audit=e.read(REC/'audit/completion.json');assert audit['state']=='E_independent_zero_fit_audit_PASS'
    assert audit['eligible_for_result_figure_generation'] and e.read(REC/'audit/checkpoint_replay.json')['state']=='PASS'
    assert e.read(REC/'numerical_completion.json')['eligible_for_result_figures']
    assert not (OUT/'figure_completion.json').exists(),'One figure build; preserve existing outputs'
    for d in (OUT,S,Q,OUT/'figures',OUT/'supplementary_figures',OUT/'pptx'):
        d.mkdir(parents=True,exist_ok=True)
    files=[Path(__file__),REC/'audit/completion.json',REC/'audit/checkpoint_replay.json',
        A/'analysis_dataset.csv',B/'factor_definitions.json',
        e.shared.V/'source_data/Figure1b_authentic_ECG.csv',e.shared.V/'source_data/Figure1b_trace_provenance.json',
        *[(root/'comparisons'/n) for root in (REC,NR) for n in ('evaluation_predictions.csv','model_metrics.csv','absolute_intervals.csv','paired_intervals.csv')],
        *[P/'refit'/n for n in ('paired_refit_differences.csv','replicate_metrics.csv','refit_distribution_summary.csv','fit_ledger.json')]]
    lock={str(f):e.sha(f) for f in files};dump(OUT/'figure_source_lock.json',{'time':e.now(),'files':lock,'new_fits':0,'new_encoder_forwards':0})
    e.verify(e.read(REC/'extraction/source_lock.json')['files']);e.verify(e.read(REC/'comparisons/source_lock.json')['files'])
    d=e.shared.cohort();test=d[d.role.eq('evaluate')].reset_index(drop=True)
    pred=e.shared.csv(REC/'comparisons/evaluation_predictions.csv');met=e.shared.csv(REC/'comparisons/model_metrics.csv').set_index('arm')
    ci=e.shared.csv(REC/'comparisons/absolute_intervals.csv');pa=e.shared.csv(REC/'comparisons/paired_intervals.csv')
    nm=e.shared.csv(NR/'comparisons/model_metrics.csv').set_index('arm');nc=e.shared.csv(NR/'comparisons/absolute_intervals.csv');npa=e.shared.csv(NR/'comparisons/paired_intervals.csv')
    yy=pred.label.to_numpy()
    for key in ('ecg_id','patient_id','label'):assert np.array_equal(pred[key],test[key])
    for arm in ALL:
        for metric,value in [('AUROC',roc_auc_score(yy,pred[arm])),('AP',average_precision_score(yy,pred[arm])),('Brier',brier_score_loss(yy,pred[arm]))]:
            assert np.isclose(value,met.loc[arm,metric],rtol=0,atol=1e-12)
    roles=d.groupby('role').agg(ECGs=('ecg_id','size'),patients=('patient_id','nunique'),AF=('label','sum')).reset_index();table('cohort_roles.csv',roles)
    raw=e.shared.csv(e.shared.V/'source_data/Figure1b_authentic_ECG.csv')
    assert raw.shape[0]>100 and {'time_s','I','II','V1'}<=set(raw)
    table('authentic_ECG.csv',raw)
    shutil.copy2(e.shared.V/'source_data/Figure1b_trace_provenance.json',S/'authentic_ECG_provenance.json')
    # Figure1: clinical scope, authentic signal, actual architecture and denominators.
    f,aa=grid('Atrial signal assessment for brain–heart research')
    a=aa[0];panel(a,'a','Clinical objective and observed endpoint');diagram(a)
    node(a,.04,.72,.9,.2,'ECG electrical state\nConcurrent AF labels',BLUE)
    arrow(a,.49,.71,.49,.55);node(a,.04,.3,.9,.23,'Current study\nSignal and representation validation',BLUE)
    label(a,.04,.10,'Brain injury and future stroke risk:\npaired imaging and follow-up pending',7.5,color=RED)
    a=aa[1];panel(a,'b','Authentic native 500 Hz ECG')
    for j,lead in enumerate(('I','II','V1')):
        a.plot(raw.time_s,raw[lead]+(2-j)*1.4,c=(BLUE,GOLD,RED)[j],lw=.8)
        a.text(raw.time_s.min()+.03,(2-j)*1.4+.25,lead,fontsize=7)
    a.set(xlabel='Time (s)',ylabel='mV with offsets');a.grid(color='#E6EBEF',lw=.4)
    a=aa[2];panel(a,'c','Patient isolation and exact cohort sizes');diagram(a)
    labels={'train':'Training: folds 1–7','calibrate':'Calibration: fold 9','evaluate':'Viewed evaluation: fold 10','unused':'Unused fold 8'}
    for j,row in enumerate(roles.itertuples()):
        label(a,.02,.91-j*.23,labels.get(row.role,str(row.role)),8,color=BLUE if row.role=='evaluate' else INK)
        label(a,.02,.81-j*.23,f'{row.ECGs:,} ECGs / {row.patients:,} patients; AF {row.AF}',7)
    a=aa[3];panel(a,'d','Matched support, frozen encoder pipelines');diagram(a)
    a.plot([.05,.95],[.87,.87],c=GREY,lw=1);a.plot([.275,.725],[.87,.87],c=GOLD,lw=5)
    label(a,.5,.95,'Raw 10 s: shared center 5 s',8,ha='center')
    node(a,.02,.47,.45,.22,'E1: ECG-FM\n12 × 2,500',BLUE)
    node(a,.54,.47,.45,.22,'E2: HuBERT-ECG\n6,000 flattened',PURPLE)
    arrow(a,.5,.84,.25,.70);arrow(a,.5,.84,.77,.70)
    label(a,.02,.28,'Each produces 768 frozen features',7.8)
    label(a,.02,.06,'Filters and pretraining differ.\nComparison concerns complete pipelines.',7.5,color=RED)
    a=aa[4];panel(a,'e','Conditional feature decomposition');diagram(a)
    for j,(name,n,color,body) in enumerate([('B',14,INK,'Demography, RR and quality'),('V',31,BLUE,'Candidate visibility / missingness'),('N',6,GOLD,'Candidate counts and geometry'),('M',25,RED,'Measured candidate morphology')]):
        a.add_patch(Rectangle((.02,.83-j*.23),.055,.13,fc=color,ec='none'))
        label(a,.12,.91-j*.23,f'{name}: {n} features',8,color=color)
        label(a,.12,.81-j*.23,body,6.9)
    a=aa[5];panel(a,'f','Frozen comparisons and evidence checks');diagram(a)
    for j,(n,body) in enumerate([('R: 400 fits','100 paired full-pipeline patient refits'),('N: 10 fits','Detector / displacement / permutation controls'),('E: 12 fits','Six classifiers and six calibrators')]):
        label(a,.02,.91-j*.24,n,8,color=(GREY,GOLD,BLUE)[j]);label(a,.02,.8-j*.24,body,7)
    label(a,.02,.12,'Fold 9 sets thresholds before fold 10.\nIndependent unexposed validation pending.',7.2,color=RED)
    foot(f,'B is included in every E/N model. P candidates have no independent expert identity reference.\nSource-exposed, viewed development evaluation; all endpoints concern concurrent AF.')
    save(f,aa,'Figure1_design','Study design and conditional signal assessment',
        'a, Observed concurrent endpoint and longer-term brain–heart objective. b, Stored authentic PTB-XL trace, selected by minimum eligible evaluation ID, without expert P annotation. c, Patient-isolated roles. d, Center5s supports and differing preprocessing. e, Frozen feature families. f, Allocated comparisons and thresholds. All diagrams are original; brain outcomes remain unmeasured.')
    # Figure2: exact metric values, display-only smooth ROC/PR, calibration.
    f,aa=grid('Frozen encoder performance and calibration')
    for k,kind in enumerate(('ROC','PR')):
        a=aa[k];panel(a,chr(97+k),kind+' discrimination')
        for arm in ALL:
            if kind=='ROC':xx,z,_=roc_curve(yy,pred[arm]);metric='AUROC'
            else:z,xx,_=precision_recall_curve(yy,pred[arm]);metric='AP'
            table(f'Figure2_{kind}_{arm}_exact.csv',pd.DataFrame({'x':xx,'y':z}))
            sx,sy=smooth(xx,z);table(f'Figure2_{kind}_{arm}_display.csv',pd.DataFrame({'x':sx,'y':sy}))
            a.plot(sx,sy,c=COLORS[arm],ls='--' if arm.startswith('E2') else '-',label=f'{NAMES[arm]} {met.loc[arm,metric]:.3f}')
        if kind=='ROC':a.plot([0,1],[0,1],c=GREY,lw=.6,ls=':');a.set(xlabel='False-positive rate',ylabel='Sensitivity')
        else:a.axhline(yy.mean(),c=GREY,lw=.6,ls=':');a.set(xlabel='Recall',ylabel='Precision')
        a.set(xlim=(0,1),ylim=(0,1.02));a.legend(loc='lower right' if kind=='ROC' else 'lower left',frameon=False,fontsize=5.8)
    for j,m in enumerate(('AUROC','AP','Brier')):panel(aa[j+2],chr(99+j),m+' and patient-cluster interval');intervals(aa[j+2],ci,m,ALL)
    a=aa[5];panel(a,'f','Calibration at fixed fold 9 thresholds');cal=[]
    a.plot([0,1],[0,1],c=GREY,ls='--',lw=.7)
    for arm in ('E1_VN','E1_VNM','E2_VN','E2_VNM'):
        prob=pred[arm].to_numpy();bins=np.minimum((prob*10).astype(int),9);rows=[]
        for b in range(10):
            ix=bins==b
            if ix.sum():rows.append({'arm':arm,'bin':b,'n':int(ix.sum()),'predicted':float(prob[ix].mean()),'observed':float(yy[ix].mean())})
        q=pd.DataFrame(rows);cal.extend(rows);a.plot(q.predicted,q.observed,'o-',ms=3,c=COLORS[arm],label=NAMES[arm])
    table('Figure2_calibration_bins.csv',pd.DataFrame(cal));a.set(xlabel='Mean predicted probability',ylabel='Observed AF fraction',xlim=(0,1),ylim=(0,1));a.legend(frameon=False,fontsize=5.7)
    foot(f,'Evaluation: 2,095 ECGs from 1,822 patients, including 150 AF ECGs. Intervals: 1,000 fixed-model patient draws.\nROC/PR smoothing is for display only; all metrics use original probabilities and exact discrete curves.')
    save(f,aa,'Figure2_performance','Discrimination, probability error and calibration',
        'a,b, Shape-preserving display interpolation of ROC/PR coordinates, without recomputing areas from interpolated curves. c–e, Exact AUROC/AP/Brier with exploratory pointwise95% patient-cluster intervals. f, Ten equal-width calibration bins; source data include each bin denominator. Models and thresholds were frozen before the viewed evaluation readout.')
    # Figure3: conditional changes and fixed operating-point consequences.
    f,aa=grid('Incremental information across frozen encoders')
    pairs=[('E1_VN','E1'),('E1_VNM','E1_VN'),('E2_VN','E2'),('E2_VNM','E2_VN')]
    panel(aa[0],'a','E1 conditional discrimination');paired(aa[0],pa,'AUROC',pairs[:2])
    panel(aa[1],'b','E2 conditional discrimination');paired(aa[1],pa,'AUROC',pairs[2:])
    panel(aa[2],'c','Conditional average precision');paired(aa[2],pa,'AP',pairs)
    panel(aa[3],'d','Conditional probability error');paired(aa[3],pa,'Brier',pairs)
    a=aa[4];panel(a,'e','Errors at calibration-set thresholds');xx=np.arange(6)
    a.bar(xx-.17,met.loc[ALL,'FP'],.34,label='False AF positives',color=RED)
    a.bar(xx+.17,met.loc[ALL,'FN'],.34,label='Missed AF',color=BLUE)
    a.set_xticks(xx,[NAMES[n].replace(' + ','\n+') for n in ALL],fontsize=5.8);a.set_ylabel('ECGs');a.legend(frameon=False,fontsize=5.8)
    a=aa[5];panel(a,'f','Interpreting incremental evidence');diagram(a)
    for j,(head,body) in enumerate([('Time support controlled','Both frozen encoders see center5s.'),('Covariate reference explicit','M is tested beyond B + encoder + V/N.'),('Generalization evidence limited','E2 has PTB-XL source-family exposure.'),('Clinical translation pending','No prospective or brain outcome labels.')]):
        label(a,.02,.94-j*.24,head,7.7,color=BLUE if j<2 else RED);label(a,.02,.83-j*.24,body,6.8)
    foot(f,'Paired pointwise95% intervals preserve patient clusters; zero-crossing effects remain inconclusive.\nErrors refer to current AF labels, not expert-confirmed P/no-P identity or future events.')
    save(f,aa,'Figure3_increment','Conditional effects and operating-point errors',
        'a,b, V/N and then M increments for each encoder. c,d, AP and Brier increments using the same contrasts. e, False-positive and false-negative AF counts at fixed fold9 thresholds. f, Interpretation scope. Intervals are exploratory fixed-model patient-cluster results, distinct from full-pipeline refit distributions.')
    # Figure4: all preserved N controls, with detector counts explicitly distinguished from P truth.
    f,aa=grid('Detector geometry and temporal negative controls')
    a=aa[0];panel(a,'a','Matched control operations');diagram(a)
    for j,(name,body,color) in enumerate([('Current M','Original candidate segment',BLUE),('Displaced M','Same segment shifted +400ms',RED),('R-window M','Fixed pre-R window: −200 to −80ms',GOLD),('Permuted M','Within-role finite-value permutation',PURPLE)]):
        label(a,.02,.94-j*.24,name,8,color=color);label(a,.02,.83-j*.24,body,7)
    for j,m in enumerate(('AUROC','AP','Brier')):panel(aa[j+1],chr(98+j),m+' across matched controls');intervals(aa[j+1],nc,m,NN)
    a=aa[4];panel(a,'e','Morphology contrasts with fixed V/N');paired(a,npa,'AUROC',[(n,'N_current') for n in ('N_displaced','N_anchor','N_permuted')])
    a=aa[5];panel(a,'f','AF candidate counts are not P truth')
    values=[test.loc[test.label.eq(k),'p_candidate_count'].dropna().to_numpy() for k in (0,1)]
    a.boxplot(values,positions=[0,1],widths=.45,showfliers=False,medianprops={'color':RED})
    a.set_xticks([0,1],['Non-AF','AF']);a.set_ylabel('Candidate count per ECG');a.text(.03,.95,'No expert P/no-P labels',transform=a.transAxes,fontsize=6.8,va='top',color=RED)
    foot(f,'V/N and original morphology missingness masks remain fixed. Displaced or pre-R segments are not verified no-P.\nAll N results are reused from the audited recovery; no detector inference or classifier refitting is repeated.')
    save(f,aa,'Figure4_controls','Matched geometry and morphology controls',
        'a, Frozen control definitions. b–d, Five control models with exploratory95% patient-cluster intervals. e, AUROC changes relative to current morphology. f, Candidate counts by concurrent AF label, with medians/IQR and1.5IQR whiskers; fliers hidden. Candidate counts cannot estimate clinical P false-positive rates. N classifiers/calibrators and features are reused unchanged.')
    # Figure5: R reuses the original6s ECG-FM recipe; this is not an E-center5s refit.
    delta=e.shared.csv(P/'refit/paired_refit_differences.csv');summary=e.shared.csv(P/'refit/refit_distribution_summary.csv');rm=e.shared.csv(P/'refit/replicate_metrics.csv')
    f,aa=grid('Patient-level refit uncertainty of morphology increments')
    for j,m in enumerate(('AUROC','AP','Brier')):
        a=aa[j];panel(a,chr(97+j),'Paired refit change in '+m);v=delta[delta.metric.eq(m)].delta
        a.hist(v,bins=16,color=(BLUE,GOLD,RED)[j],alpha=.8,edgecolor='white',lw=.5);a.axvline(0,c=INK,ls='--',lw=.8)
        a.set(xlabel='Change: FM + V/N/M vs FM + V/N',ylabel='Patient refits')
    a=aa[3];panel(a,'d','Central 90% empirical refit range')
    for j,m in enumerate(('AUROC','AP','Brier')):
        row=summary[summary.metric.eq(m)].iloc[0];a.plot([row.p05,row.p95],[j,j],c=(BLUE,GOLD,RED)[j],lw=1.2)
        a.plot([row.q25,row.q75],[j,j],c=(BLUE,GOLD,RED)[j],lw=5);a.plot(row['median'],j,'o',c=INK,ms=3)
    a.axvline(0,c=GREY,ls='--',lw=.7);a.set_yticks([0,1,2],['AUROC','AP','Brier']);a.set_xlabel('Paired change');a.invert_yaxis()
    a=aa[4];panel(a,'e','Calibration-threshold variability')
    for arm,color in [('FM_VN',BLUE),('FM_VNM',RED)]:
        a.hist(rm[rm.arm.eq(arm)].threshold,bins=15,histtype='step',lw=1.2,color=color,label=arm.replace('FM_','FM + '))
    a.set(xlabel='Bootstrap calibration threshold',ylabel='Patient refits');a.legend(frameon=False)
    a=aa[5];panel(a,'f','Paired directions across refits');wide=delta.pivot(index='replicate',columns='metric',values='delta')
    a.scatter(wide.AUROC,wide.AP,s=8,c=BLUE,alpha=.65);a.axvline(0,c=GREY,lw=.6);a.axhline(0,c=GREY,lw=.6)
    a.set(xlabel='Paired AUROC change',ylabel='Paired AP change')
    foot(f,'100 paired training/calibration/evaluation patient refits: 400 preserved fits, original6s ECG-FM recipe.\nEmpirical refit distributions describe recipe uncertainty; they are not confirmatory confidence intervals.')
    save(f,aa,'Figure5_refit','Full-pipeline patient refit distributions',
        'a–c,100 paired changes in AUROC,AP and Brier. d, Median,IQR and empirical5th–95th percentile ranges. e, Per-replicate calibration thresholds; original deployment thresholds remain unchanged. f, Paired directions across metrics. These R analyses reuse the original6s representation recipe and do not represent refit uncertainty for the new center5s E models.')
    f,aa=grid('Complete model performance intervals',3)
    for row,(frame,arms,group) in enumerate([(ci,ALL[:3],'E1'),(ci,ALL[3:],'E2'),(nc,NN,'N')]):
        for col,m in enumerate(('AUROC','AP','Brier')):
            j=row*3+col;panel(aa[j],chr(97+j),group+': '+m);intervals(aa[j],frame,m,arms)
    foot(f,'Absolute model performance; exploratory pointwise95% intervals from the original 1,000 patient-cluster draws.\nAll models include B. These are within-source development comparisons, not independent clinical validation.')
    save(f,aa,'SupplementaryFigure1_intervals','Complete absolute performance intervals',
        'a–c,E1. d–f,E2. g–i,N control models. Columns show AUROC,AP and Brier. Source tables preserve original estimates, confidence limits and exact labels.')
    f,aa=grid('Cohort denominators, missingness and evidence scope')
    a=aa[0];panel(a,'a','ECG counts by patient role');xx=np.arange(len(roles))
    a.bar(xx,roles.ECGs-roles.AF,color=BLUE,label='Non-AF');a.bar(xx,roles.AF,bottom=roles.ECGs-roles.AF,color=RED,label='AF')
    a.set_xticks(xx,roles.role,rotation=25,ha='right');a.set_ylabel('ECGs');a.legend(frameon=False)
    a=aa[1];panel(a,'b','Fit ledger and allocated budget');a.barh(['R','N','E'],[400,10,12],color=[GREY,GOLD,BLUE]);a.set_xlabel('Completed classifier/calibrator fits')
    a=aa[2];panel(a,'c','Encoding exposure and independence');diagram(a)
    label(a,.02,.9,'PTB-XL downstream roles are isolated.',8)
    label(a,.02,.65,'E2 pretraining includes PTB-XL family.',8,color=RED)
    label(a,.02,.40,'Encoder membership is not certified.',8)
    label(a,.02,.15,'Independent unexposed validation: pending.',7.5,color=RED)
    a=aa[3];panel(a,'d','Prespecified morphology missingness')
    factors=e.read(B/'factor_definitions.json');features=factors['factors']['M'][:6]
    for k,color in [(0,BLUE),(1,RED)]:
        rates=[float(test.loc[test.label.eq(k),c].isna().mean()) for c in features]
        a.barh(np.arange(6)+(k-.5)*.3,rates,height=.3,color=color,label='AF' if k else 'Non-AF')
    a.set_yticks(range(6),[c.replace('p_','').replace('_',' ') for c in features],fontsize=5.8);a.set_xlabel('Missing fraction');a.legend(frameon=False,fontsize=6)
    a=aa[4];panel(a,'e','Repeated ECGs per evaluation patient');repeats=test.groupby('patient_id').size();counts=repeats.value_counts().sort_index()
    a.bar(counts.index,counts.values,color=BLUE);a.set(xlabel='ECGs per patient',ylabel='Patients')
    a=aa[5];panel(a,'f','Reference evidence required for translation');diagram(a)
    for j,(head,body) in enumerate([('Available','Concurrent AF labels and native ECG'),('Waiting: signal identity','Expert P/no-P and boundary references'),('Waiting: atrial phenotype','Same-patient atrial imaging references'),('Waiting: brain–heart risk','Brain findings, events, death and censoring')]):
        label(a,.02,.94-j*.24,head,7.8,color=BLUE if j==0 else RED);label(a,.02,.83-j*.24,body,6.9)
    foot(f,'Morphology missingness uses the first six M fields in the frozen factor definition, without outcome-driven selection.\nClinical risk inference requires linked references and time-to-event follow-up beyond these concurrent AF results.')
    save(f,aa,'SupplementaryFigure2_scope','Denominators and clinical evidence boundaries',
        'a, ECG-level role composition; exact patient denominators accompany source tables. b, Completed fit attempts. c, Encoder/source exposure scope. d, Missingness in the first six frozen M fields by current AF label. e, Repeat recordings within evaluation patients. f, Available and waiting references for future clinical studies.')
    for root,prefix in [(REC,'E'),(NR,'N')]:
        for name in ('model_metrics.csv','absolute_intervals.csv','paired_intervals.csv'):
            shutil.copy2(root/'comparisons'/name,S/(prefix+'_'+name))
    for name in ('paired_refit_differences.csv','refit_distribution_summary.csv','replicate_metrics.csv'):
        shutil.copy2(P/'refit'/name,S/('R_'+name))
    (OUT/'figure_legends.txt').write_text('\n\n'.join(legends),encoding='utf-8')
    dump(OUT/'figure_inventory.json',inventory)
    e.verify(lock)
    dump(OUT/'figure_completion.json',{'time':e.now(),'state':'vector_artifacts_created_visual_review_pending',
        'main_figures':5,'supplementary_figures':2,'panels':sum(q['panels'] for q in inventory),
        'minimum_rows':3,'PNG_dpi':600,'font_family':FONT,'SVG_native_text':True,'PDF_no_raster_layers':True,
        'new_fits':0,'new_encoder_forwards':0,'PPTX_complete':False,'visual_review_complete':False,'NC_submission_ready':False})
    report=['A24 cross-encoder analyses: concurrent AF, viewed/source-exposed development evaluation.',
        'All E results passed the separate numerical/source/patient/threshold/checkpoint audit.',
        'E12 + N10 + R400 =422 fits; figure generation adds0 fits.',
        'Absolute metrics:\n'+met[['AUROC','AP','Brier','sensitivity','specificity','FP','FN']].to_string(),
        'Conditional effects (exploratory pointwise95% fixed-model patient-cluster intervals):\n'+pa.to_string(index=False),
        'Historical6s FM full-pipeline refit distribution, kept distinct from the new center5s E models:\n'+summary.to_string(index=False),
        'A confidence interval crossing zero is an inconclusive effect, not a technical failure.',
        'SPH remains quarantined. Expert P/no-P and boundary references, atrial imaging and same-patient brain/event/death/censoring data remain pending.',
        'These data do not establish future AF, brain injury or stroke prediction; NC submission readiness remains false.',
        'Figures require final visual review. PPTX native object export follows automatically; rendering alone does not establish completion.']
    (OUT/'analysis_summary.txt').write_text('\n\n'.join(report),encoding='utf-8')
    (OUT/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>A24 figures</title><style>body{font:16px Arial;max-width:1100px;margin:40px auto;color:#243B53}img{max-width:95%}section{margin:50px 0}</style><h1>A24 signal and representation analyses</h1><p>Concurrent AF development evaluation. Source exposure and clinical reference gaps persist. Automated rendering is followed by visual review.</p>'+''.join('<section><h2>'+q['title']+'</h2><a href="'+q['PDF']+'">Vector PDF</a><br><img src="'+q['PNG']+'"></section>' for q in inventory),encoding='utf-8')
    print('A24_VECTOR_FIGURES_COMPLETE_PPTX_AND_VISUAL_REVIEW_PENDING',flush=True)

def backend_smoke():
    global OUT,S,Q
    OUT=P/'software_contracts_FIGURE_backend_v1';S=OUT/'source_data';Q=OUT/'qa'
    for d in (OUT,S,Q,OUT/'figures',OUT/'supplementary_figures'):d.mkdir(parents=True,exist_ok=True)
    f,aa=grid('MOCK renderer test, no research data')
    x=np.linspace(0,1,60)
    for j,a in enumerate(aa):
        panel(a,chr(97+j),'MOCK editable evidence '+str(j))
        a.set(xlabel='Mock x',ylabel='Mock y')
        if j==0:a.plot(x,np.sin(x*5),c=BLUE,label='Mock curve');a.legend(frameon=False)
        elif j==1:a.fill_between(x,x*.5,x*.8,color=GOLD,alpha=.3);a.plot(x,x*.7,c=GOLD)
        elif j==2:a.errorbar([.2,.5,.7],[1,2,3],xerr=.08,fmt='o',c=RED);a.axvline(.5,c=GREY,ls='--')
        elif j==3:a.bar(['a','b','c'],[1,2,3],color=[BLUE,GOLD,RED])
        elif j==4:a.scatter(x[::5],np.cos(x[::5]),c=BLUE,s=15)
        else:
            diagram(a);node(a,.1,.4,.8,.3,'MOCK native text\nand editable diagram',BLUE);arrow(a,.5,.35,.5,.1)
    foot(f,'MOCK graphics backend verification. Synthetic geometry only. No ECG, model or study result is loaded.')
    save(f,aa,'MOCK_NOT_SCIENTIFIC','MOCK graphics backend','No research data; renderer-only contract.')
    dump(OUT/'figure_inventory.json',inventory)
    print('MOCK_PDF_SVG_NATIVE_INTERCHANGE_PASS_NO_RESEARCH_DATA',flush=True)

if __name__=='__main__':
    if '--check-code' in sys.argv:
        print('FIGURE_CODE_IMPORT_PASS_NO_DATA_NO_FIT_NO_FIGURE font='+FONT)
    elif '--backend-smoke' in sys.argv:
        backend_smoke()
    else:main()
