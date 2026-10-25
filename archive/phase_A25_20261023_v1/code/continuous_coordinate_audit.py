"""Preserve fractional reference coordinates; do not round or silently discard waves."""
from common import *
from decimal import Decimal
import zipfile,csv,io,collections,sys,traceback
import pandas as pd
OUT=RUN/'continuous_coordinate_audit';START=time.monotonic()
LEADS=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
def main():
    assert not (OUT/'started.json').exists()
    put(OUT/'started.json',dict(time=now(),pid=os.getpid(),code_SHA256=sha(__file__),max_seconds=900,mode='Coordinate-preserving schema diagnosis, no prediction or relabelling'))
    z=zipfile.ZipFile(RUN/'data/dataset.zip');names=z.namelist();records=sorted(Path(n).stem for n in names if n.startswith('dat_csv/') and n.endswith('.csv'))
    rows=[];issues=[];counts=collections.Counter();counts_rec=collections.defaultdict(set);duplicate_annotations={}
    for i,record in enumerate(records):
        deadline(START,900);allann={};pann={};fractional=0
        for lead in ['all']+LEADS:
            aa=[]
            for row in csv.reader(io.StringIO(z.read(f'ann_txt/{record}.{lead}.txt').decode('utf-8-sig'))):
                if not row or not any(v.strip() for v in row):continue
                if [v.strip().upper() for v in row]==['TYPE','START','END']:continue
                try:
                    d=tuple(Decimal(v.strip()) for v in row);assert len(d)==3 and all(v.is_finite() for v in d)
                    t,a,b=d;assert t in [0,1,2] and t==t.to_integral_value()
                    # Keep noninteger positions. Out-of-range entries remain explicitly flagged, never clipped.
                    if not (0<=a<=b<5000):issues.append(dict(record=record,lead=lead,row=row,issue='range_or_order'))
                    if a!=a.to_integral_value() or b!=b.to_integral_value():
                        counts[str(int(t))]+=1;counts_rec[str(int(t))].add(record);fractional+=1
                    aa.append((int(t),a,b))
                except Exception:issues.append(dict(record=record,lead=lead,row=row,issue='non_numeric_or_type'))
            aa=tuple(sorted(aa));allann[lead]=aa;pann[lead]=tuple(v for v in aa if v[0]==0)
        p_ii,p_v1=pann['ii'],pann['v1'];has=bool(p_ii or p_v1)
        rows.append(dict(record=record,prefix=''.join(c for c in record if not c.isdigit()),P_II=len(p_ii),P_V1=len(p_v1),has_P_II_or_V1=has,II_V1_P_identical=p_ii==p_v1,all12_P_identical=all(pann[l]==pann['i'] for l in LEADS),all12_full_equal_global=all(allann[l]==allann['all'] for l in LEADS),zero_P_all12=all(not pann[l] for l in LEADS),fractional_coordinate_rows=fractional))
        if record in ['SI0109','SI0211']:duplicate_annotations[record]={l:[[t,str(a),str(b)] for t,a,b in allann[l]] for l in ['all']+LEADS}
    df=pd.DataFrame(rows);df.to_csv(OUT/'record_reference_counts.csv',index=False);put(OUT/'coordinate_issues.json',issues);put(OUT/'duplicate_annotations_preserved.json',duplicate_annotations)
    nonempty=df.has_P_II_or_V1
    put(OUT/'completion.json',dict(time=now(),state='complete',records=len(df),coordinate_policy='Decimal values retained exactly; no rounding/clipping; no scientific admission',fractional_rows_by_type=dict(counts),records_with_fractional_by_type={k:len(v) for k,v in counts_rec.items()},out_of_range_or_nonnumeric_rows=len(issues),issues=issues[:5],all12_full_equal_global_records=int(df.all12_full_equal_global.sum()),all12_P_identical_records=int(df.all12_P_identical.sum()),nonempty_II_or_V1_records=int(nonempty.sum()),nonempty_II_V1_P_identical_records=int((nonempty&df.II_V1_P_identical).sum()),II_V1_P_different_records=int((~df.II_V1_P_identical).sum()),zero_P_all12_records=int(df.zero_P_all12.sum()),no_P_truth_inferred=False,SPH_admitted=False,new_fits=0,wall_seconds=time.monotonic()-START))
    status('continuous_coordinate_audit',state='complete',records=len(df),issues=len(issues))
if __name__=='__main__':
    try:main()
    except Exception as e:status('continuous_coordinate_audit',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
