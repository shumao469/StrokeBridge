"""A26 official Chapman raw acquisition and exact RDB-parent comparison, no model."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
from pathlib import Path,PurePosixPath
from datetime import datetime
import json,time,hashlib,urllib.request,zipfile,io,sys,traceback
import numpy as np,pandas as pd
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A25=BASE/'runs/phase_A25_20261023_v1';P=json.loads((RUN/'source_protocol.json').read_text());START=time.monotonic()
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
def put(p,v):
    p=RUN/p;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(p)
def digest(p,kind='sha256'):
    h=hashlib.new(kind)
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def status(state,**kw):
    v=dict(time=now(),pid=os.getpid(),state=state,SPH_admitted=False,new_fits=0,new_figures=0,**kw);put('source/status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
def check():
    if time.monotonic()-START>P['max_total_audit_seconds']:raise TimeoutError('Source audit bound')
    if (RUN/'STOP').exists():raise RuntimeError('Manual STOP')
def main():
    assert not (RUN/'source/started.json').exists(),'No automatic retry'
    put('source/started.json',dict(time=now(),pid=os.getpid(),protocol_SHA256=digest(RUN/'source_protocol.json'),code_SHA256=digest(Path(__file__))))
    dest=RUN/'source/ECGData.zip';part=dest.with_suffix('.zip.partial');assert not dest.exists() and not part.exists()
    tick=time.monotonic();last=tick;status('downloading_official_raw')
    req=urllib.request.Request(P['raw_url'],headers={'User-Agent':'A26-official-source-verification/1.0'})
    with urllib.request.urlopen(req,timeout=60) as response,part.open('wb') as out:
        assert response.status==200
        cl=response.headers.get('Content-Length');assert cl and int(cl)==P['raw_bytes']
        etag=response.headers.get('ETag','').strip('"');assert etag==P['raw_MD5_from_official_S3_single_object_ETag']
        put('source/response_identity.json',dict(Content_Length=int(cl),ETag=etag,source=P['raw_url'],redirect_host=urllib.request.urlparse(response.url).hostname if hasattr(urllib.request,'urlparse') else 'official Figshare S3 redirect',signed_redirect_url_not_retained=True))
        while True:
            check()
            if time.monotonic()-tick>P['max_acquisition_seconds']:raise TimeoutError('Acquisition20min bound')
            b=response.read(1024*1024)
            if not b:break
            out.write(b)
            assert out.tell()<=P['raw_bytes'] and out.tell()<=P['max_download_bytes']
            if time.monotonic()-last>20:status('downloading_official_raw',bytes=out.tell(),total=P['raw_bytes']);last=time.monotonic()
    assert part.stat().st_size==P['raw_bytes'];assert digest(part,'md5')==P['raw_MD5_from_official_S3_single_object_ETag']
    part.replace(dest);put('source/integrity.json',dict(time=now(),bytes=dest.stat().st_size,MD5=digest(dest,'md5'),SHA256=digest(dest),provider_ETag_match=True))
    status('archive_and_all_parent_matching')
    with zipfile.ZipFile(dest) as parent,zipfile.ZipFile(A25/'data/dataset.zip') as rdb:
        infos=parent.infolist();assert len(infos)<=P['max_archive_members'] and sum(i.file_size for i in infos)<=P['max_uncompressed_archive_bytes']
        for i in infos:
            q=PurePosixPath(i.filename.replace('\\','/'));assert not q.is_absolute() and '..' not in q.parts and ':' not in i.filename
        assert parent.testzip() is None
        put('source/parent_manifest.json',[dict(name=i.filename,size=i.file_size,CRC32=format(i.CRC,'08x')) for i in infos])
        candidates={}
        for i in infos:
            if i.filename.lower().endswith('.csv'):candidates.setdefault(PurePosixPath(i.filename).stem,[]).append(i.filename)
        mappings=pd.read_csv(A25/'local_receipt/mapping_readonly.csv',keep_default_na=False);results=[];bad=[]
        for j,row in enumerate(mappings.to_dict('records')):
            check();key=row['chapman_record_id'];names=candidates.get(key,[])
            if len(names)!=1:
                results.append(dict(rdb_record_id=row['rdb_record_id'],parent_record_id=key,state='parent_missing_or_ambiguous',matches=len(names)));continue
            a=np.loadtxt(io.BytesIO(rdb.read('dat_csv/'+row['rdb_record_id']+'.csv')),delimiter=',')
            data=parent.read(names[0])
            try:b=np.loadtxt(io.BytesIO(data),delimiter=',')
            except ValueError:b=np.loadtxt(io.BytesIO(data),delimiter=',',skiprows=1)
            if b.shape==(12,5000):b=b.T
            assert b.shape==(5000,12) and np.isfinite(b).all()
            err=float(np.max(abs(a-b)));scaled=float(np.max(abs(a-b/1000.)))
            state='exact_numeric_identity' if err<=1e-12 else ('parent_div1000_match' if scaled<=1e-12 else 'unresolved_transform')
            results.append(dict(rdb_record_id=row['rdb_record_id'],parent_record_id=key,parent_member=names[0],state=state,max_identity_error=err,max_parent_div1000_error=scaled,parent_CSV_SHA256=hashlib.sha256(data).hexdigest(),rdb_prefix=row['rdb_rhythm'],mapped_parent_rhythm=row['chapman_rhythm']))
            if state=='unresolved_transform':bad.append(row['rdb_record_id'])
            if j%300==0:status('matching_parent',processed=j,total=len(mappings))
        df=pd.DataFrame(results);df.to_csv(RUN/'source/all_parent_waveform_comparison.csv',index=False)
        xlsx=[i.filename for i in infos if i.filename.lower().endswith('.xlsx')];put('source/embedded_metadata_files.json',xlsx)
        put('source/completion.json',dict(time=now(),state='complete',ZIP_CRC='PASS',parent_CSV_records=len(candidates),RDB_rows=len(mappings),match_counts=df.state.value_counts().to_dict(),unresolved_transform=bad,embedded_metadata=xlsx,units_parent_paper='microvolt',RDB_numeric_unit='same as parent, microvolt' if df.state.eq('exact_numeric_identity').all() else 'not yet uniformly certified',units_conversion_to_mV='multiply0.001 only on derived inputs; originals unchanged',global_P_validation_admitted=False,VT_SVT_official_diagnostic_metadata='pending',SPH_admitted=False,new_fits=0,new_figures=0,wall_seconds=time.monotonic()-START))
        status('source_complete_pending_reference_and_label_gate',counts=df.state.value_counts().to_dict())
if __name__=='__main__':
    try:main()
    except Exception as e:status('FAIL_STOP',error=repr(e),traceback=traceback.format_exc());sys.exit(1)
