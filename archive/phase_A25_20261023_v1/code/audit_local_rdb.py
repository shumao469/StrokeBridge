"""Read-only full archive/OOXML audit. Never execute supplied code or infer no-P truth."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from common import *
import zipfile,xml.etree.ElementTree as ET,io,csv,collections,re,sys,traceback
from pathlib import PurePosixPath
import numpy as np,pandas as pd
START=time.monotonic();OUT=RUN/'local_receipt';P=read(RUN/'local_receipt_protocol.json')
LEADS=['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6']
def md5(p):
    h=hashlib.md5()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def mapping(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        assert z.testzip() is None
        strings=[''.join(t.itertext()) for t in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)] if 'xl/sharedStrings.xml' in z.namelist() else []
        root=ET.fromstring(z.read('xl/worksheets/sheet1.xml'));rows=[]
        for row in root.findall('.//s:sheetData/s:row',ns):
            vals={}
            for c in row.findall('s:c',ns):
                tag=c.attrib.get('t');v=c.find('s:v',ns);value=v.text if v is not None else ''
                if tag=='inlineStr':value=''.join(c.find('s:is',ns).itertext())
                elif tag=='s':value=strings[int(value)]
                vals[re.sub(r'\d','',c.attrib['r'])]=value
            rows.append(vals)
        header=rows[0];out=[{header[k]:r.get(k,'') for k in header} for r in rows[1:]]
        assert all(not v for v in ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('.//s:f',ns)), 'Formula-based mapping needs separate audit'
        return pd.DataFrame(out)
def main():
    assert not (OUT/'started.json').exists(),'No implicit retry'
    put(OUT/'started.json',dict(time=now(),pid=os.getpid(),protocol_SHA256=sha(RUN/'local_receipt_protocol.json'),code_SHA256=sha(__file__)))
    integrity={}
    for name,want in P['files'].items():
        path=RUN/'data'/name;got=dict(size=path.stat().st_size,MD5=md5(path),SHA256=sha(path));assert got['size']==want['size'] and got['MD5']==want['MD5'];integrity[name]=got
    mp=mapping(RUN/'data/rdb_chapman_mapping.xlsx');mp.to_csv(OUT/'mapping_readonly.csv',index=False)
    mapids=mp.rdb_record_id.tolist();assert len(mapids)==len(set(mapids)),'Duplicate RDB map key'
    filled=mp.chapman_record_id.fillna('').astype(str).str.strip().ne('');parents=mp.loc[filled,'chapman_record_id']
    put(OUT/'integrity.json',dict(time=now(),files=integrity,mapping_rows=len(mp),mapped_rows=int(filled.sum()),parent_record_duplicates=int(parents.duplicated().sum()),mapping_is_patient_ID=False))
    with zipfile.ZipFile(RUN/'data/dataset.zip') as z:
        infos=z.infolist();assert len(infos)<=P['max_members'] and sum(i.file_size for i in infos)<=P['max_uncompressed_bytes']
        names=[i.filename for i in infos];assert len(names)==len(set(names))
        for name in names:
            p=PurePosixPath(name.replace('\\','/'));assert not p.is_absolute() and '..' not in p.parts and ':' not in name
        # Reading every member below verifies CRC; explicitly test the archive as a second integrity check.
        assert z.testzip() is None,'CRC failure'
        put(OUT/'archive_manifest.json',[dict(name=i.filename,size=i.file_size,CRC32=format(i.CRC,'08x')) for i in infos])
        readme=z.read('README.md').decode('utf-8-sig');(OUT/'README_as_received.txt').write_text(readme,encoding='utf-8')
        records=sorted(PurePosixPath(n).stem for n in names if n.startswith('dat_csv/') and n.endswith('.csv'))
        assert len(records)==len(set(records));rows=[];leadrows=[];anomalies=[];hashes=collections.defaultdict(list)
        for j,record in enumerate(records):
            deadline(START,P['max_wall_seconds']);raw=z.read('dat_csv/'+record+'.csv');x=np.loadtxt(io.BytesIO(raw),delimiter=',',dtype=np.float64)
            signal_ok=bool(x.shape==(5000,12) and np.isfinite(x).all())
            if not signal_ok:anomalies.append(dict(record=record,kind='signal_shape_or_finiteness',shape=list(x.shape)))
            hashes[hashlib.sha256(raw).hexdigest()].append(record)
            aa={};syntax=overlap=0;missing=[]
            for lead in ['all']+LEADS:
                name=f'ann_txt/{record}.{lead}.txt'
                if name not in names:missing.append(lead);continue
                text=z.read(name).decode('utf-8-sig');lines=list(csv.reader(io.StringIO(text)));header=bool(lines and [q.strip().upper() for q in lines[0]]==['TYPE','START','END'])
                if header:lines=lines[1:]
                arr=[];bad=[]
                for ln in lines:
                    if not ln or not any(v.strip() for v in ln):continue
                    try:
                        values=[int(v.strip()) for v in ln];assert len(values)==3
                        t,a,b=values;assert t in [0,1,2] and 0<=a<=b<5000;arr.append((t,a,b))
                    except Exception:bad.append(ln)
                a=np.asarray(arr,dtype=int).reshape(-1,3);aa[lead]=a
                occ=np.zeros(5000,np.int8);ov=0
                for t,start,end in arr:
                    ov+=int((occ[start:end+1]>0).sum());occ[start:end+1]+=1
                syntax+=len(bad);overlap+=ov
                if bad:anomalies.append(dict(record=record,lead=lead,kind='annotation_syntax',rows=bad[:5]))
                if ov:anomalies.append(dict(record=record,lead=lead,kind='annotation_overlap_samples',n=ov))
                leadrows.append(dict(record=record,lead=lead,header=header,intervals=len(a),P=int((a[:,0]==0).sum()),QRS=int((a[:,0]==1).sum()),T=int((a[:,0]==2).sum()),syntax_errors=len(bad),overlap_samples=ov))
            equal=all(lead in aa and np.array_equal(aa[lead],aa['all']) for lead in LEADS) if 'all' in aa else False
            p_equal=all(lead in aa and np.array_equal(aa[lead][aa[lead][:,0]==0],aa[LEADS[0]][aa[LEADS[0]][:,0]==0]) for lead in LEADS) if LEADS[0] in aa else False
            m=mp[mp.rdb_record_id.eq(record)];parent=m.iloc[0].chapman_record_id if len(m) else ''
            row=dict(record=record,rhythm_prefix=re.sub('[0-9]','',record),mapping_row_exists=len(m)==1,parent_record=parent,parent_present=bool(str(parent).strip()),mapping_note=m.iloc[0].get('note','') if len(m) else '',shape=str(tuple(x.shape)),finite=bool(np.isfinite(x).all()),signal_ok=signal_ok,abs_max_csv=float(np.max(abs(x))),median_lead_range_csv=float(np.median(np.ptp(x,axis=0))),median_step_csv=float(np.median(abs(np.diff(x,axis=0)))),all_12_equal_to_all=equal,all_12_P_equal=p_equal,syntax_errors=syntax,overlap_samples=overlap,missing_lead_files=';'.join(missing),csv_SHA256=hashlib.sha256(raw).hexdigest())
            rows.append(row)
            if j%200==0:status('local_receipt',state='running',processed=j,total=len(records))
        df=pd.DataFrame(rows);lf=pd.DataFrame(leadrows);df.to_csv(OUT/'record_audit.csv',index=False);lf.to_csv(OUT/'lead_annotation_audit.csv',index=False);put(OUT/'annotation_anomalies.json',anomalies)
        duplicates=[v for v in hashes.values() if len(v)>1];put(OUT/'waveform_exact_duplicates.json',duplicates)
        unmatched=df[~df.parent_present|~df.mapping_row_exists];extra_maps=sorted(set(mapids)-set(records))
        result=dict(time=now(),state='complete',files_integrity='PASS',ZIP_CRC='PASS',members=len(infos),uncompressed_bytes=sum(i.file_size for i in infos),archive_records=len(records),mapping_rows=len(mp),mapped_parent_records=int(filled.sum()),unique_parent_records=int(parents.nunique()),parent_duplicates=int(parents.duplicated().sum()),archive_rhythm_counts=df.rhythm_prefix.value_counts().to_dict(),mapping_rhythm_counts=mp.rdb_rhythm.value_counts().to_dict(),unmapped_records=unmatched.to_dict('records'),mapping_ids_not_in_archive=extra_maps,signal_shape_finite_pass=int(df.signal_ok.sum()),all_12_equal_to_all_records=int(df.all_12_equal_to_all.sum()),all_12_P_equal_records=int(df.all_12_P_equal.sum()),annotation_syntax_errors=int(lf.syntax_errors.sum()),annotation_overlap_samples=int(lf.overlap_samples.sum()),missing_lead_files=int(df.missing_lead_files.ne('').sum()),exact_waveform_duplicate_groups=duplicates,reference_P_zero_by_prefix=lf[lf.lead.eq('all')].assign(prefix=lambda t:t.record.str.replace(r'\d','',regex=True)).groupby('prefix').P.agg(['count','sum','min','max']).reset_index().to_dict('records'),units_README='mV',units_numerically_suspect=bool(df.median_lead_range_csv.median()>20),patient_identity_certified=False,clinical_no_P_certified=False,lead_specific_reference_certified=False,encoder_family_exposure={'E1':'Chapman via PhysioNet2021; exact membership uncertified','E2':'Chapman pretraining declared; exact membership uncertified','P_network_A9':'fresh LUDB-only documented; cross-source identity audit not yet passed'},scientific_admission=False,SPH_admitted=False,new_fits=0,new_figures=0,wall_seconds=time.monotonic()-START)
        put(OUT/'completion.json',result);status('local_receipt',state='complete',archive_records=len(records),mapped_parent_records=int(filled.sum()),all_12_P_equal_records=result['all_12_P_equal_records'],scientific_admission=False)
if __name__=='__main__':
    try:main()
    except Exception as e:status('local_receipt',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
