"""Isolated Windows inference dependencies and one verified official encoder asset."""
from pathlib import Path
from datetime import datetime
import sys,subprocess,time,json,hashlib,urllib.request,shutil
P=Path(__file__).resolve().parents[1];START=time.monotonic();R=P/'runtime';R.mkdir(exist_ok=True)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()
def put(n,v):(R/n).write_text(json.dumps(v,ensure_ascii=False,indent=2),encoding='utf-8')
def main():
    assert not (R/'started.json').exists(),'Do not implicitly repeat setup'
    put('started.json',{'time':datetime.now().astimezone().isoformat(),'protocol_SHA256':sha(P/'protocol.txt'),'code_SHA256':sha(__file__),'max_seconds':2700})
    vendor=R/'vendor';vendor.mkdir(exist_ok=True)
    args=[sys.executable,'-m','pip','install','--target',str(vendor),'--cache-dir',str(R/'pip_cache'),'--no-deps','--disable-pip-version-check','transformers==4.57.1','tokenizers==0.22.1','safetensors==0.6.2','huggingface_hub==0.36.0','omegaconf==2.3.0','antlr4-python3-runtime==4.9.3','biosppy==2.2.3','peakutils==1.3.5','bidict==0.23.1','shortuuid==1.0.13']
    done=subprocess.run(args,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=1200)
    (R/'pip_install.log').write_text(done.stdout,encoding='utf-8');assert done.returncode==0,'Isolated dependency installation failed; no retry'
    # Read-only mirrors from the already-running WSL filesystem. No WSL command or restart.
    src=Path(r'\\wsl.localhost\Ubuntu-22.04\home\shu\codex_runs\phase_A17R_20261021_v1\upstream');dst=R/'ecgfm_upstream';copied=[]
    for file in src.rglob('*'):
        if file.is_file() and file.suffix in ['.py','.md','.txt'] and '__pycache__' not in file.parts:
            target=dst/file.relative_to(src);target.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(file,target);assert sha(file)==sha(target);copied.append({'source':str(file),'copy':str(target),'SHA256':sha(target)})
    pw=Path(r'\\wsl.localhost\Ubuntu-22.04\home\shu\codex_runs\phase_A10_20261019_v1\C1_sign_orbit\seed_20261020\fold_0\best.pt')
    expected='98557608ee9c581bc0e0ee4f1f0578ad4c14f961952357b5461782f349b341d4';assert sha(pw)==expected
    checkpoint=R/'P_C1_frozen.pt';shutil.copy2(pw,checkpoint);assert sha(checkpoint)==expected
    meta=json.loads((P/'source_audit/HuBERT_hf_api.json').read_text(encoding='utf-8'));rev=meta['sha'];asset=next(f for f in meta['siblings'] if f['rfilename']=='model.safetensors')
    weightdir=Path('DRIVE_ROOT/脑心同查/models/hubert-ecg-base')/rev;weightdir.mkdir(parents=True,exist_ok=True);weight=weightdir/'model.safetensors'
    if weight.exists():assert weight.stat().st_size==asset['size'] and sha(weight)==asset['lfs']['sha256']
    else:
        url=f'https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/{rev}/model.safetensors';part=weight.with_suffix('.safetensors.partial');assert not part.exists(),'Keep partial; no implicit retry'
        req=urllib.request.Request(url,headers={'User-Agent':'A24-Academic-Frozen-Encoder/1.0'})
        with urllib.request.urlopen(req,timeout=90) as response,part.open('wb') as out:
            while True:
                if time.monotonic()-START>2700:raise TimeoutError('Setup wall budget')
                block=response.read(4*1024*1024)
                if not block:break
                out.write(block)
        assert part.stat().st_size==asset['size'] and sha(part)==asset['lfs']['sha256'],'Provider weight verification failed'
        part.replace(weight)
    for srcname,destname in [('HuBERT_config.json','config.json'),('hf_hubert_ecg.py','hubert_ecg.py'),('HuBERT_hf_README.txt','README.md'),('HuBERT_LICENSE.txt','LICENSE.txt')]:shutil.copy2(P/'source_audit'/srcname,weightdir/destname)
    put('source_mirrors.json',copied)
    put('completion.json',{'state':'acquired_not_yet_admitted_for_inference','time':datetime.now().astimezone().isoformat(),'vendor':str(vendor),'E2_weight':str(weight),'E2_SHA256':sha(weight),'P_checkpoint':str(checkpoint),'P_SHA256':sha(checkpoint),'E1_upstream':str(dst),'source_files':len(copied),'global_environment_modified':False,'WSL_restarted':False,'wall_seconds':time.monotonic()-START})
    print(json.dumps({'state':'runtime_acquired','weight_bytes':weight.stat().st_size,'source_files':len(copied)},ensure_ascii=False))
if __name__=='__main__':
    try:main()
    except Exception as e:put('stop_receipt.json',{'state':'stopped_no_retry','error':str(e),'wall_seconds':time.monotonic()-START});raise
