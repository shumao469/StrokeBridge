"""A25 official-only, bounded acquisition. Integrity != scientific admission."""
import hashlib, json, os, time, urllib.request, zipfile, traceback, sys
from pathlib import Path, PurePosixPath
from datetime import datetime
import xml.etree.ElementTree as ET

RUN = Path(__file__).resolve().parents[1]
START = time.monotonic()
P = json.loads((RUN/'audit_protocol.json').read_text(encoding='utf-8-sig'))
BYTES = 0
REQUESTS = 0

def now(): return datetime.now().astimezone().isoformat(timespec='seconds')
def put(name, obj):
    path=RUN/name; path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False),encoding='utf-8');tmp.replace(path)
def sha(p, kind='sha256'):
    h=hashlib.new(kind)
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    return h.hexdigest()
def check():
    if time.monotonic()-START>P['max_wall_seconds']:raise TimeoutError('A25 audit wall limit')
    if BYTES>P['max_response_bytes']:raise RuntimeError('A25 audit byte limit')
    if (RUN/'STOP').exists():raise RuntimeError('Manual STOP')
def status(stage,**kw):
    v=dict(time=now(),pid=os.getpid(),stage=stage,requests=REQUESTS,bytes=BYTES,SPH_admitted=False,new_fits=0,new_figures=0,**kw)
    put('audit_status.json',v);print(json.dumps(v,ensure_ascii=False),flush=True)
def fetch(url,path,expected=None):
    global REQUESTS,BYTES
    check();REQUESTS+=1
    if REQUESTS>P['max_logical_http_requests']:raise RuntimeError('HTTP request budget')
    assert url.startswith(('https://api.figshare.com/','https://ndownloader.figshare.com/'))
    path=RUN/path;path.parent.mkdir(parents=True,exist_ok=True)
    assert not path.exists(), 'No implicit overwrite / retry'
    part=path.with_suffix(path.suffix+'.partial');assert not part.exists(),'No implicit resume'
    req=urllib.request.Request(url,headers={'User-Agent':'A25-research-source-audit/1.0'})
    began=time.monotonic();last=began
    with urllib.request.urlopen(req,timeout=60) as response,part.open('wb') as f:
        put('audit/http_'+str(REQUESTS)+'.json',dict(url=url,final_url=response.url,headers=dict(response.headers),time=now()))
        while True:
            check()
            if time.monotonic()-began>P['single_file_timeout_seconds']:raise TimeoutError('single file limit')
            b=response.read(1024*1024)
            if not b:break
            f.write(b);BYTES+=len(b)
            if expected and f.tell()>expected['size']:raise RuntimeError('Provider size exceeded')
            if time.monotonic()-last>20:status('downloading',file=path.name,file_bytes=f.tell());last=time.monotonic()
    if expected:
        assert part.stat().st_size==expected['size'],'Provider size mismatch'
        assert sha(part,'md5')==expected['md5'],'Provider MD5 mismatch'
    part.replace(path)
    return path
def archive_audit(path):
    with zipfile.ZipFile(path) as z:
        infos=z.infolist();assert len(infos)<=P['max_files']
        assert sum(i.file_size for i in infos)<=P['max_uncompressed_bytes']
        names=[]
        for i in infos:
            q=PurePosixPath(i.filename.replace('\\','/'))
            assert not q.is_absolute() and '..' not in q.parts and ':' not in i.filename,'Unsafe archive member'
            names.append(i.filename)
        assert len(names)==len(set(names)),'Duplicate archive members'
        assert z.testzip() is None,'ZIP CRC mismatch'
        put('audit/archive_manifest.json',[dict(name=i.filename,size=i.file_size,compressed=i.compress_size,CRC32=format(i.CRC,'08x')) for i in infos])
        return dict(members=len(infos),uncompressed_bytes=sum(i.file_size for i in infos),CRC='PASS',sample_names=names[:20])
def mapping(path):
    ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'}
    with zipfile.ZipFile(path) as z:
        strings=[]
        if 'xl/sharedStrings.xml' in z.namelist():
            strings=[''.join(n.itertext()) for n in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
        sheets={}
        for name in z.namelist():
            if not name.startswith('xl/worksheets/sheet') or not name.endswith('.xml'):continue
            rows=[]
            for row in ET.fromstring(z.read(name)).findall('.//s:sheetData/s:row',ns):
                cells={}
                for c in row.findall('s:c',ns):
                    value=c.find('s:v',ns);v=value.text if value is not None else ''
                    if c.attrib.get('t')=='s':v=strings[int(v)]
                    elif c.attrib.get('t')=='inlineStr':v=''.join(c.find('s:is',ns).itertext())
                    cells[c.attrib['r']]=v
                rows.append(cells)
            sheets[name]=rows
        put('audit/mapping_cells.json',sheets)
        return {k:dict(rows=len(v),first_rows=v[:5]) for k,v in sheets.items()}
def main():
    assert not (RUN/'audit_started.json').exists(),'Single launch only'
    put('audit_started.json',dict(time=now(),pid=os.getpid(),protocol_SHA256=sha(RUN/'audit_protocol.json'),code_SHA256=sha(__file__)))
    status('official_metadata')
    api=fetch(P['source'],'audit/figshare_v4.json')
    meta=json.loads(api.read_text());assert meta['version']==P['required_version']
    assert meta['license']['name']==P['required_license']
    files={i['name']:i for i in meta['files']}
    for name,v in P['files'].items():
        m=files[name];assert m['id']==v['id'] and m['size']==v['size'] and m['computed_md5']==v['md5']
    # Mapping is first: its versioned identity evidence can be reviewed while archive transfers.
    mp=fetch(files['rdb_chapman_mapping.xlsx']['download_url'],'data/rdb_chapman_mapping.xlsx',P['files']['rdb_chapman_mapping.xlsx'])
    mi=mapping(mp);put('audit/mapping_summary.json',mi)
    archive=fetch(files['dataset.zip']['download_url'],'data/dataset.zip',P['files']['dataset.zip'])
    za=archive_audit(archive)
    put('audit/integrity.json',dict(time=now(),state='PASS',scientific_admission=False,source_version=meta['version'],DOI=meta['doi'],license=meta['license'],files={n:dict(size=(RUN/'data'/n).stat().st_size,MD5=sha(RUN/'data'/n,'md5'),SHA256=sha(RUN/'data'/n)) for n in P['files']},archive=za,mapping=mi,description=meta['description'],encoder_family_exposed=['E1','E2'],all_new_training=0))
    status('integrity_complete_awaiting_scientific_gate',state='PASS',wall_seconds=time.monotonic()-START)
if __name__=='__main__':
    try:main()
    except Exception as e:
        status('STOPPED',state='FAIL_STOP',error=repr(e),traceback=traceback.format_exc(),wall_seconds=time.monotonic()-START)
        sys.exit(1)
