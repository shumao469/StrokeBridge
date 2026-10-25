"""Official ZIP ranges: do not conflate sample/CRC checks with full cryptographic audit."""
import os
for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']:os.environ[k]='1'
import io,urllib.request,re,time,json,zipfile,hashlib,zlib,sys,traceback
from pathlib import Path
from datetime import datetime
import numpy as np,pandas as pd
RUN=Path(__file__).resolve().parents[1];BASE=RUN.parents[1];A25=BASE/'runs/phase_A25_20261023_v1';P=json.loads((RUN/'range_protocol.json').read_text());OUT=RUN/'range_audit';START=time.monotonic()
def put(n,v):
    p=OUT/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8')
def now():return datetime.now().astimezone().isoformat(timespec='seconds')
class Remote(io.RawIOBase):
    def __init__(self):self.pos=0;self.calls=0;self.bytes=0;self.cache=[]
    def seekable(self):return True
    def readable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        self.pos=offset if whence==0 else (self.pos+offset if whence==1 else P['expected_total_bytes']+offset);return self.pos
    def get(self,a,b):
        assert time.monotonic()-START<P['max_seconds'];self.calls+=1;assert self.calls<=P['max_requests']
        req=urllib.request.Request(P['url'],headers={'Range':f'bytes={a}-{b}','If-Range':'"'+P['expected_ETag']+'"'})
        with urllib.request.urlopen(req,timeout=P['timeout_seconds_per_request']) as r:
            assert r.status==206 and r.headers.get('Content-Range')==f'bytes {a}-{b}/{P["expected_total_bytes"]}'
            assert r.headers.get('ETag','').strip('"')==P['expected_ETag'];data=r.read(b-a+2)
        assert len(data)==b-a+1;self.bytes+=len(data);assert self.bytes<=P['max_response_bytes'];self.cache.append((a,data))
        return data
    def read(self,size=-1):
        n=P['expected_total_bytes'];size=n-self.pos if size<0 else min(size,n-self.pos)
        if size<=0:return b''
        a=self.pos;b=a+size
        for start,data in self.cache:
            if start<=a and b<=start+len(data):self.pos=b;return data[a-start:b-start]
        block=P['block_bytes'];start=a//block*block;end=min(n-1,((b+block-1)//block)*block-1);data=self.get(start,end);self.pos=b;return data[a-start:b-start]
def main():
    assert not (OUT/'started.json').exists();put('started.json',dict(time=now(),pid=os.getpid(),protocol_SHA256=hashlib.sha256((RUN/'range_protocol.json').read_bytes()).hexdigest()))
    remote=Remote();remote.get(P['expected_total_bytes']-P['tail_bytes'],P['expected_total_bytes']-1);parent=zipfile.ZipFile(remote)
    infos=parent.infolist();names={}
    for item in infos:
        if item.filename.lower().endswith('.csv'):names.setdefault(Path(item.filename).stem,[]).append(item)
    put('central_directory.json',[dict(name=i.filename,size=i.file_size,compressed=i.compress_size,CRC32=format(i.CRC,'08x')) for i in infos])
    mapping=pd.read_csv(A25/'local_receipt/mapping_readonly.csv',keep_default_na=False);allids=mapping.rdb_record_id.tolist()
    sample=[]
    for rhythm,g in mapping.groupby('rdb_rhythm'):sample.append(sorted(g.rdb_record_id)[0])
    sample+=['SI0109','SI0211','SI0198','ST0059'];sample=list(dict.fromkeys(sample))
    sample+=[k for k in sorted(set(allids)-set(sample),key=lambda v:hashlib.sha256(v.encode()).hexdigest())[:P['sample_records']-len(sample)]]
    put('samples_frozen_before_parent_reads.json',sample);rdb=zipfile.ZipFile(A25/'data/dataset.zip');rows=[];prefixes={b''}
    for record in sample:
        row=mapping[mapping.rdb_record_id.eq(record)].iloc[0];items=names.get(row.chapman_record_id,[]);assert len(items)==1
        payload=parent.read(items[0]);raw=rdb.read('dat_csv/'+record+'.csv');x=np.loadtxt(io.BytesIO(raw),delimiter=',')
        header=False
        try:y=np.loadtxt(io.BytesIO(payload),delimiter=',')
        except ValueError:y=np.loadtxt(io.BytesIO(payload),delimiter=',',skiprows=1);header=True
        assert y.shape==(5000,12) and np.isfinite(y).all()
        err=float(np.max(abs(x-y)));err_scaled=float(np.max(abs(x-y/1000)))
        rows.append(dict(record=record,parent=row.chapman_record_id,header=header,max_identity_error=err,max_parent_div1000_error=err_scaled,parent_SHA256=hashlib.sha256(payload).hexdigest(),parent_CRC32=format(items[0].CRC,'08x'),numeric_identity=err<=1e-12))
        for version in [raw,raw.replace(b'\r\n',b'\n'),raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n')]:
            if payload.endswith(version):prefixes.add(payload[:-len(version)] if version else b'')
        put('status.json',dict(time=now(),state='range_numeric_sample',completed=len(rows),total=len(sample),requests=remote.calls,bytes=remote.bytes));print(record,err,flush=True)
    pd.DataFrame(rows).to_csv(OUT/'numeric_sample_comparison.csv',index=False)
    crc=[]
    for row in mapping.to_dict('records'):
        assert time.monotonic()-START<P['max_seconds'];items=names.get(row['chapman_record_id'],[]);raw=rdb.read('dat_csv/'+row['rdb_record_id']+'.csv');hits=[]
        if len(items)==1:
            for pname,payload in [('identity',raw),('LF',raw.replace(b'\r\n',b'\n')),('CRLF',raw.replace(b'\r\n',b'\n').replace(b'\n',b'\r\n'))]:
                for prefix in prefixes:
                    content=prefix+payload
                    if len(content)==items[0].file_size and zlib.crc32(content)&0xffffffff==items[0].CRC:hits.append(dict(transform=pname,header_bytes=len(prefix)))
        crc.append(dict(record=row['rdb_record_id'],parent=row['chapman_record_id'],parent_member_count=len(items),CRC_size_consistent=bool(hits),transforms=json.dumps(hits)))
    cc=pd.DataFrame(crc);cc.to_csv(OUT/'all_source_CRC_size_comparison.csv',index=False)
    put('completion.json',dict(time=now(),state='complete',sample_count=len(rows),sample_numeric_identity_all=all(r['numeric_identity'] for r in rows),RDB_rows=len(mapping),parent_CSV_members=len(names),all_parent_names_unique_present=bool(cc.parent_member_count.eq(1).all()),all_CRC_size_consistent=bool(cc.CRC_size_consistent.all()),CRC_size_matches=int(cc.CRC_size_consistent.sum()),CRC_size_not_cryptographic_identity=True,full_parent_archive_download_verified=False,unit_evidence='All sample numeric values identical to parent raw microvolt CSV' if all(r['numeric_identity'] for r in rows) else 'Unresolved',prefix_hex=[v.hex() for v in prefixes],requests=remote.calls,response_bytes=remote.bytes,scientific_admission=False,SPH_admitted=False,new_fits=0,new_figures=0,wall_seconds=time.monotonic()-START))
if __name__=='__main__':
    try:main()
    except Exception as e:put('stop.json',dict(time=now(),state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc()));sys.exit(1)
