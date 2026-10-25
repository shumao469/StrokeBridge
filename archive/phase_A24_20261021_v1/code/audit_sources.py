"""Bounded official-source retrieval; never import downloaded code or run a model."""
from pathlib import Path
from datetime import datetime
import json,hashlib,urllib.request,urllib.error,time
P=Path(__file__).resolve().parents[1];D=P/'source_audit';D.mkdir(parents=True,exist_ok=True)
ledger=[];blocked=set()
def save(n,x):
    (D/n).write_text(json.dumps(x,ensure_ascii=False,indent=2),encoding='utf-8')
def get(url,name):
    assert len(ledger)<24
    domain=urllib.parse.urlparse(url).netloc
    if domain in blocked:return None
    row={'url':url,'file':name,'time':datetime.now().astimezone().isoformat()};ledger.append(row)
    try:
        req=urllib.request.Request(url,headers={'User-Agent':'A24-Academic-Provenance-Audit/1.0','Accept':'application/json,text/plain,*/*'})
        with urllib.request.urlopen(req,timeout=90) as r:
            b=r.read(8*1024*1024);row.update(status=r.status,final_url=r.url,bytes=len(b),SHA256=hashlib.sha256(b).hexdigest())
        (D/name).write_bytes(b);return b
    except urllib.error.HTTPError as e:
        row.update(status=e.code,error=str(e),retry_after=e.headers.get('Retry-After'))
        if e.code in [401,403,429]:blocked.add(domain)
    except Exception as e:row.update(error=str(e))
    finally:save('request_ledger.json',ledger)
    return None
def main():
    assert not (D/'request_ledger.json').exists(),'No implicit repeat of source audit'
    z=get('https://zenodo.org/api/records/20388950','ECG_LFM_zenodo.json')
    for name,url in [
        ('ECG_LFM_repo.json','https://api.github.com/repos/biomed-AI/ECG-LFM'),
        ('ECG_LFM_commit.json','https://api.github.com/repos/biomed-AI/ECG-LFM/commits/master'),
        ('ECG_LFM_LICENSE.txt','https://raw.githubusercontent.com/biomed-AI/ECG-LFM/master/LICENSE'),
        ('HuBERT_repo.json','https://api.github.com/repos/Edoar-do/HuBERT-ECG'),
        ('HuBERT_commit.json','https://api.github.com/repos/Edoar-do/HuBERT-ECG/commits/master'),
        ('HuBERT_tree.json','https://api.github.com/repos/Edoar-do/HuBERT-ECG/git/trees/master?recursive=1'),
        ('HuBERT_LICENSE.txt','https://raw.githubusercontent.com/Edoar-do/HuBERT-ECG/master/LICENSE'),
        ('HuBERT_README.txt','https://raw.githubusercontent.com/Edoar-do/HuBERT-ECG/master/README.md'),
        ('HuBERT_hf_api.json','https://huggingface.co/api/models/Edoardo-Coppola/hubert-ecg-base?blobs=true'),
        ('HuBERT_config.json','https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/main/config.json'),
        ('HuBERT_hf_README.txt','https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/main/README.md')]:get(url,name)
    hf=json.loads((D/'HuBERT_hf_api.json').read_text()) if (D/'HuBERT_hf_api.json').exists() else {}
    revision=hf.get('sha');files=[s['rfilename'] for s in hf.get('siblings',[])];save('HuBERT_hf_file_inventory.json',{'revision':revision,'files':files})
    for file in [f for f in files if f.endswith('.py')][:4]:get(f'https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/{revision}/{file}','hf_'+file.replace('/','__'))
    summary={'requests':len(ledger),'blocked_domains':sorted(blocked),'hf_revision':revision,'hf_card':hf.get('cardData'),'hf_files':hf.get('siblings',[])}
    if z:
        zz=json.loads(z);summary['zenodo']={'metadata':zz.get('metadata'),'files':zz.get('files'),'links':zz.get('links')}
    save('audit_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False))
if __name__=='__main__':main()
