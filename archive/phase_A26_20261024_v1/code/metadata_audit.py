import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path
from datetime import datetime
import urllib.request,zipfile,io,xml.etree.ElementTree as ET,json,hashlib,time,sys,traceback
import pandas as pd
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];OUT=RUN/'metadata';P=json.loads((RUN/'metadata_protocol.json').read_text());START=time.monotonic()
def put(name,v):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def xlsx(data):
    z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None
    if 'xl/workbook.xml' not in z.namelist():
        members=[n for n in z.namelist() if n.lower().endswith('.xlsx')];assert len(members)==1;data=z.read(members[0]);z=zipfile.ZipFile(io.BytesIO(data));assert z.testzip() is None
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};strings=[]
    if 'xl/sharedStrings.xml' in z.namelist():strings=[''.join(x.itertext()) for x in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
    worksheets=[n for n in z.namelist() if n.startswith('xl/worksheets/sheet') and n.endswith('.xml')];root=ET.fromstring(z.read(worksheets[0]));rows=[]
    for row in root.findall('.//s:sheetData/s:row',ns):
        record={}
        for c in row.findall('s:c',ns):
            v=c.find('s:v',ns);value=v.text if v is not None else '';t=c.attrib.get('t')
            if t=='s':value=strings[int(value)]
            elif t=='inlineStr':value=''.join(c.find('s:is',ns).itertext())
            record[''.join(k for k in c.attrib['r'] if k.isalpha())]=value
        rows.append(record)
    header=rows[0];df=pd.DataFrame([{header[k]:r.get(k,'') for k in header} for r in rows[1:] if any(r.values())]);return data,df
def main():
    assert not (OUT/'started.json').exists();put('started.json',{'time':datetime.now().astimezone().isoformat(),'pid':os.getpid(),'protocol_SHA256':hashlib.sha256((RUN/'metadata_protocol.json').read_bytes()).hexdigest()});total=0;tables={}
    for name,url in P['urls'].items():
        assert time.monotonic()-START<P['max_wall_seconds']
        with urllib.request.urlopen(url,timeout=60) as r:data=r.read(P['max_bytes']+1)
        total+=len(data);assert total<=P['max_bytes'];(OUT/(name+'_as_downloaded.zip')).write_bytes(data)
        blob,df=xlsx(data);(OUT/(name+'.xlsx')).write_bytes(blob);df.to_csv(OUT/(name+'.csv'),index=False);tables[name]=df
        put(name+'_identity.json',dict(url=url,downloaded_bytes=len(data),downloaded_SHA256=hashlib.sha256(data).hexdigest(),xlsx_SHA256=hashlib.sha256(blob).hexdigest(),rows=len(df),columns=list(df.columns)))
    diag=tables['Diagnostics_v2'];assert 'FileName' in diag and 'Rhythm' in diag and len(diag)==10646 and diag.FileName.is_unique
    mapping=pd.read_csv(BASE/'runs/phase_A25_20261023_v1/local_receipt/mapping_readonly.csv',keep_default_na=False)
    joined=mapping.merge(diag[['FileName','Rhythm']],left_on='chapman_record_id',right_on='FileName',how='left',validate='many_to_one',indicator=True)
    joined.to_csv(OUT/'RDB_official_diagnostic_join.csv',index=False)
    vt=joined[joined.rdb_rhythm.eq('VT')];mapped=joined[joined._merge.eq('both')]
    put('completion.json',dict(time=datetime.now().astimezone().isoformat(),state='complete',parent_records=len(diag),RDB_rows=len(joined),matched=int(joined._merge.eq('both').sum()),missing=joined.loc[joined._merge.ne('both'),'rdb_record_id'].tolist(),VT_filename_official_rhythm_counts=vt.Rhythm.value_counts(dropna=False).to_dict(),all_mapped_rhythm_agreement=bool(mapped.Rhythm.eq(mapped.chapman_rhythm).all()),actual_patient_ID_column=False,SPH_admitted=False,new_fits=0,new_figures=0,wall_seconds=time.monotonic()-START));print('OFFICIAL_METADATA_COMPLETE',flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:put('stop.json',dict(state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc()));sys.exit(1)
