"""Plot-only readability correction; original frozen outputs remain unchanged."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
ROOT=Path('/home/shu/codex_runs/phase_A17R_20261021_v1')
d=pd.read_csv(ROOT/'paired_patient_intervals.csv')
out=ROOT/'publication_figures';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.right':False,
                     'axes.spines.top':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(12,4.8))
fig.subplots_adjust(left=.18,right=.98,bottom=.27,top=.76,wspace=.30)
fig.suptitle('Candidate P-fragment features: paired incremental value',fontsize=14,y=.94)
for ax,metric,title,scale in zip(axs,['AP','AUROC','Brier'],
    ['AP difference (higher better)','AUROC difference (higher better)',
     'Brier difference (lower better)'],[1,1,1000]):
    for j,(arm,base,color) in enumerate([('D_RR_P','D_RR','#c76f2b'),('D_RR_P_FM','D_RR_FM','#a73c60')]):
        row=d[d.method.eq(arm)&d.baseline.eq(base)&d.metric.eq(metric)].iloc[0]
        lo=float(row.lower97_5 if metric=='AP' else row.lower95)*scale
        hi=float(row.upper97_5 if metric=='AP' else row.upper95)*scale
        point=float(row.delta)*scale;y=1-j
        ax.plot([lo,hi],[y,y],color=color,lw=2)
        ax.scatter(point,y,color=color,marker=['o','s'][j],s=38,zorder=3)
        prec=2 if metric=='Brier' else 3
        label=f'{point:+.{prec}f} [{lo:+.{prec}f}, {hi:+.{prec}f}]'
        ax.text(.02,.89-j*.50,label,transform=ax.transAxes,fontsize=9)
    ax.axvline(0,color='#777777',ls='--',lw=.8)
    ax.set_ylim(-.45,1.65);ax.set_yticks([1,0])
    ax.set_yticklabels(['P beyond D + RR','P beyond D + RR + FM'] if metric=='AP' else ['', ''])
    ax.set_title(title,fontsize=10,pad=14)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    if metric=='Brier':ax.set_xlabel('Difference × 1,000')
fig.text(.05,.06,'2,000 paired patient-cluster bootstrap replicates; models held fixed.\n'
         'AP: 97.5% intervals for two primary comparisons; AUROC/Brier: descriptive 95% intervals.\n'
         'D: demographics. P: candidate fragments, not verified P waves. Viewed, source-exposed PTB-XL; concurrent AF only.',fontsize=9)
for ext in ['png','svg','pdf']:fig.savefig(out/f'Figure2_paired_increment_readable.{ext}',dpi=240,facecolor='white')
plt.close(fig)
print('Plot-only export complete; no fitting, thresholds, predictions or intervals changed')
