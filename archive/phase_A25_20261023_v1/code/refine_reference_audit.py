"""Annotation-only numeric-schema refinement, separate from preserved first audit.

Decimal strings such as 480.0 are exact integer indices, not corrupt references.
No rounding, endpoint repair or new predictions. Original inputs stay untouched.
"""
from common import *
import zipfile,csv,io,collections,sys,traceback
from decimal import Decimal,InvalidOperation
import numpy as np,pandas as pd
OUT=RUN/'reference_refinement';START=time.monotonic()
LEADS=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
def parse(text):
    arr=[];issues=[];decimal_integer_rows=0;noninteger=0
    for ln in csv.reader(io.StringIO(text)):
        if not ln or not any(s.strip() for s in ln):continue
        if [s.strip().upper() for s in ln]==['TYPE','START','END']:continue
        try:
            d=[Decimal(v.strip()) for v in ln];assert len(d)==3 and all(q.is_finite() and q==q.to_integral_value() for q in d)
            t,a,b=map(int,d);assert t in [0,1,2] and 0<=a<=b<5000
            decimal_integer_rows+=int(any('.' in v for v in ln));arr.append((t,a,b))
        except Exception:issues.append(ln);noninteger+=1
    return np.asarray(sorted(arr),dtype=int).reshape(-1,3),issues,decimal_integer_rows
def main():
    assert not (OUT/'started.json').exists(),'No implicit retry'
    put(OUT/'started.json',dict(time=now(),pid=os.getpid(),code_SHA256=sha(__file__),max_seconds=900,new_fits=0,change='Accept exactly integral decimal index strings; never round or change intervals'))
    assert read(RUN/'local_receipt/completion.json')['ZIP_CRC']=='PASS'
    put(OUT/'parser_tests.json',dict(integer_strings='PASS' if np.array_equal(parse('0,1,20\n')[0],parse('0,1.0,20.0\n')[0]) else 'FAIL',fractional_rejected=len(parse('0,1.5,20\n')[1])==1,nonfinite_rejected=len(parse('0,NaN,20\n')[1])==1,last_index5000_rejected=len(parse('0,1,5000\n')[1])==1))
    z=zipfile.ZipFile(RUN/'data/dataset.zip');names=set(z.namelist());records=sorted(Path(n).stem for n in names if n.startswith('dat_csv/') and n.endswith('.csv'))
    rr=[];lr=[];issues=[];decimal_rows=0;annotation_hash={}
    for i,record in enumerate(records):
        deadline(START,900);leadann={}
        for lead in ['all']+LEADS:
            arr,bad,dec=parse(z.read(f'ann_txt/{record}.{lead}.txt').decode('utf-8-sig'));decimal_rows+=dec;leadann[lead]=arr
            if bad:issues.append(dict(record=record,lead=lead,rows=bad))
            occ=np.zeros(5000,np.int16);overlap=0
            for t,a,b in arr:overlap+=int((occ[a:b+1]>0).sum());occ[a:b+1]+=1
            counts={t:int((arr[:,0]==t).sum()) for t in [0,1,2]}
            p=arr[arr[:,0]==0];q=arr[arr[:,0]==1]
            lr.append(dict(record=record,lead=lead,P=counts[0],QRS=counts[1],T=counts[2],invalid_rows=len(bad),overlap_samples=overlap,exact_decimal_integer_rows=dec,first_annotated_sample=int(arr[:,1].min()) if len(arr) else -1,last_annotated_sample=int(arr[:,2].max()) if len(arr) else -1,central_P=int(((p[:,1]>=1000)&(p[:,2]<4000)).sum()) if len(p) else 0))
        original=leadann['all'];pp=[leadann[lead][leadann[lead][:,0]==0] for lead in LEADS]
        allsame=all(np.array_equal(leadann[lead],original) for lead in LEADS)
        psame=all(np.array_equal(a,pp[0]) for a in pp)
        ii=leadann['ii'];v1=leadann['v1'];pii=ii[ii[:,0]==0];pv1=v1[v1[:,0]==0]
        rr.append(dict(record=record,prefix=''.join(c for c in record if not c.isdigit()),all_lead_annotations_equal_global=allsame,all_lead_P_equal=psame,P_II_V1_equal=np.array_equal(pii,pv1),P_II=len(pii),P_V1=len(pv1),unique_lead_annotation_sets=len({a.tobytes() for a in leadann.values()})))
        annotation_hash[record]={lead:hashlib.sha256(arr.tobytes()).hexdigest() for lead,arr in leadann.items()}
        if i%400==0:status('reference_refinement',state='running',records=i,total=len(records))
    r=pd.DataFrame(rr);l=pd.DataFrame(lr);r.to_csv(OUT/'record_reference_details.csv',index=False);l.to_csv(OUT/'lead_reference_details.csv',index=False);put(OUT/'remaining_reference_issues.json',issues)
    mapping=pd.read_csv(RUN/'local_receipt/mapping_readonly.csv',keep_default_na=False);dup=mapping[mapping.chapman_record_id.duplicated(keep=False)]
    dupinfo=[]
    for parent,g in dup.groupby('chapman_record_id'):
        ids=g.rdb_record_id.tolist();dupinfo.append(dict(parent=parent,records=ids,exact_waveform_duplicate=True,annotation_equal_by_lead={lead:len({annotation_hash[record][lead] for record in ids})==1 for lead in ['all']+LEADS},policy='Same source group; retain both originals. No automatic representative choice or label merge.'))
    nonempty=r.P_II.gt(0)|r.P_V1.gt(0)
    result=dict(time=now(),state='complete',records=len(r),exact_decimal_integer_rows=decimal_rows,remaining_invalid_rows=int(l.invalid_rows.sum()),overlap_samples=int(l.overlap_samples.sum()),all_lead_annotations_equal_global_records=int(r.all_lead_annotations_equal_global.sum()),all_lead_P_equal_records=int(r.all_lead_P_equal.sum()),nonempty_II_or_V1_records=int(nonempty.sum()),nonempty_II_or_V1_equal_records=int((nonempty&r.P_II_V1_equal).sum()),II_V1_different_P_records=int((~r.P_II_V1_equal).sum()),zero_P_all12_records=int(l[l.lead.isin(LEADS)].groupby('record').P.sum().eq(0).sum()),duplicate_source_annotation_conflicts=dupinfo,source_conflicts=['README states no header, actual files have TYPE,START,END header','README declares mV, numerical 4.88-step scale consistent with parent paper microvolt; exact transformation not certified','README original dataset/citation includes placeholder and conflicting bibliographic links','Website lists SVT, archive/mapping use VT; rhythm correspondence not independently verified'],lead_specific_gold_admitted=False,clinical_no_P_admitted=False,patient_mapping_admitted=False,scientific_admission=False,SPH_admitted=False,new_fits=0,wall_seconds=time.monotonic()-START)
    put(OUT/'completion.json',result);status('reference_refinement',state='complete',invalid_rows=result['remaining_invalid_rows'],lead_specific_gold_admitted=False)
if __name__=='__main__':
    try:main()
    except Exception as e:status('reference_refinement',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
