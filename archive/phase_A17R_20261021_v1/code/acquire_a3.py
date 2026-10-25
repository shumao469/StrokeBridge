"""Public A3 assets: pinned HF weights and checksum-verified QTDB. Resumable."""
import concurrent.futures, hashlib, json, time, zipfile
from pathlib import Path
import requests

ROOT=Path('DRIVE_ROOT/脑心同查')
REV='584219ea492cdeef2e19ffbdf9c6ecc874ba427e'
SHA='4d0142bcb485eb9f0c7845e0c19ff3463f6ae9d0e458eab69136efe90ceb9b7e'

def digest(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''): h.update(b)
    return h.hexdigest()

def download(url,path,expected=None):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if expected and digest(path)!=expected: raise RuntimeError('Existing asset checksum mismatch: '+str(path))
        return
    partial=path.with_suffix(path.suffix+'.part')
    for attempt in range(20):
        try:
            n=partial.stat().st_size if partial.exists() else 0
            r=requests.get(url,headers={'Range':f'bytes={n}-'} if n else {},stream=True,timeout=(30,90))
            r.raise_for_status()
            if n and (r.status_code!=206 or not r.headers.get('Content-Range','').startswith(f'bytes {n}-')):
                raise RuntimeError('Unsafe resume response; partial preserved')
            start=time.time(); last=start; base=n
            with partial.open('ab' if n else 'wb') as f:
                for b in r.iter_content(1024*1024):
                    f.write(b); n+=len(b)
                    if time.time()-last>10:
                        print(json.dumps({'asset':path.name,'bytes':n,'MB_s':round((n-base)/(time.time()-start)/1e6,3)}),flush=True); last=time.time()
            if expected and digest(partial)!=expected: raise RuntimeError('Downloaded checksum mismatch; partial preserved')
            partial.replace(path); return
        except (requests.RequestException,OSError) as e:
            print(str(e),flush=True); time.sleep(min(5*(attempt+1),30))
    raise RuntimeError('Download retries exhausted: '+url)

def weights():
    p=ROOT/'models/ecg-fm'/REV
    name='mimic_iv_ecg_physionet_pretrained.pt'
    download(f'https://huggingface.co/wanglab/ecg-fm/resolve/{REV}/{name}',p/name,SHA)
    for name in ['mimic_iv_ecg_physionet_pretrained.yaml','README.md']:
        download(f'https://huggingface.co/wanglab/ecg-fm/resolve/{REV}/{name}',p/name)
    result={'status':'PASS','revision':REV,'sha256':SHA,'bytes':(p/'mimic_iv_ecg_physionet_pretrained.pt').stat().st_size,'source':'https://huggingface.co/wanglab/ecg-fm','license':'MIT','pretraining':['MIMIC-IV-ECG 1.0','PhysioNet2021 1.0.3'],'exposure_warning':'PTB-XL is in PhysioNet2021: not pretraining-unseen evaluation'}
    (p/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)

def qtdb():
    p=ROOT/'datasets/qtdb/1.0.0'; archive=p/'qtdb-1.0.0.zip'
    download('https://physionet.org/content/qtdb/get-zip/1.0.0/',archive)
    dest=p/'extracted'
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            if not (dest/info.filename).resolve().is_relative_to(dest.resolve()): raise RuntimeError('Unsafe ZIP path')
        if z.testzip(): raise RuntimeError('ZIP CRC mismatch')
        z.extractall(dest)
    manifest=next(dest.rglob('SHA256SUMS.txt')); failures=[]; count=0
    for line in manifest.read_text().splitlines():
        sha,name=line.split(maxsplit=1); f=manifest.parent/name.lstrip('*')
        if not f.is_file() or digest(f)!=sha: failures.append(name)
        else: count+=1
    result={'status':'PASS' if not failures else 'FAIL','root':str(manifest.parent),'archive_sha256':digest(archive),'provider_verified':count,'failures':failures,'annotation_policy':'q1c manual selected beats; pu annotations are NOT ground truth; lead identity must be read from header'}
    (p/'verification.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result),flush=True)
    if failures: raise RuntimeError('QTDB checksum failures')

if __name__=='__main__':
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as ex:
        for f in [ex.submit(weights),ex.submit(qtdb)]: f.result()
