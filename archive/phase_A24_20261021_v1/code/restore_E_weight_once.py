"""One authorized HTTPS Range request; no retry, full official verification."""
from a24_shared import *
import urllib.request,shutil,subprocess,traceback
REC=P/'recovery_E_v1';REV='8ffa1ad3593366ca45f5351d75317d4ff8f14465'
ROOT=Path('DRIVE_ROOT/脑心同查/models/hubert-ecg-base')/REV
OLD=ROOT/'model.safetensors.partial';PART=ROOT/'model.safetensors.A24_E_recovery.partial';FINAL=ROOT/'model.safetensors'
TOTAL=372518672;PREFIX=96509823;EXPECTED='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae';OLD_SHA='efedb093538fe92106abb3d5a21b49b42731aabc241c502cfafd075110daf74a'
URL=f'https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/{REV}/model.safetensors'
START=time.monotonic()

class OfficialRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        from urllib.parse import urlparse
        parsed=urlparse(newurl);host=parsed.hostname or ''
        assert parsed.scheme=='https' and (host=='huggingface.co' or host.endswith('.huggingface.co') or host=='hf.co' or host.endswith('.hf.co')),'Non-provider redirect refused'
        f=REC/'redirects.json';rows=read(f) if f.exists() else [];rows.append({'code':code,'host':host});put(f,rows)
        return super().redirect_request(req,fp,code,msg,headers,newurl)

def worker():
    assert not PART.exists() and not FINAL.exists(),'Preserve any prior recovery; no implicit repeat'
    assert OLD.stat().st_size==PREFIX and sha(OLD)==OLD_SHA
    shutil.copy2(OLD,PART);assert PART.stat().st_size==PREFIX and sha(PART)==OLD_SHA
    request=urllib.request.Request(URL,headers={'User-Agent':'A24-Authorized-Bounded-Range/1.0','Range':f'bytes={PREFIX}-','Accept-Encoding':'identity'})
    put(REC/'request_started.json',{'time':now(),'url':URL,'Range':f'bytes={PREFIX}-','logical_request':1,'max_additional_bytes':276100000,'expected_additional_bytes':TOTAL-PREFIX})
    got=0
    with urllib.request.build_opener(OfficialRedirect()).open(request,timeout=60) as response:
        cr=response.headers.get('Content-Range');length=response.headers.get('Content-Length')
        put(REC/'response_headers.json',{'time':now(),'status':response.status,'Content-Range':cr,'Content-Length':length,'final_host':urllib.request.urlparse(response.url).hostname if hasattr(urllib.request,'urlparse') else response.url.split('/')[2]})
        assert response.status==206 and cr==f'bytes {PREFIX}-{TOTAL-1}/{TOTAL}','Range not honored exactly; stop without append'
        assert response.headers.get('Content-Encoding','identity')=='identity'
        if length is not None:assert int(length)==TOTAL-PREFIX
        with PART.open('ab') as out:
            while True:
                assert time.monotonic()-START<900,'Acquisition deadline reached'
                block=response.read(min(1024*1024,TOTAL-PREFIX-got+1))
                if not block:break
                got+=len(block);assert got<=TOTAL-PREFIX and got<=276100000,'Additional-body byte budget exceeded'
                out.write(block)
                put(REC/'download_progress.json',{'time':now(),'received_additional_bytes':got,'expected_additional_bytes':TOTAL-PREFIX,'wall_seconds':time.monotonic()-START})
    assert got==TOTAL-PREFIX and PART.stat().st_size==TOTAL,'Incomplete body: stop,no retry'
    digest=sha(PART);assert digest==EXPECTED,'Full provider SHA mismatch'
    assert sha(OLD)==OLD_SHA and OLD.stat().st_size==PREFIX
    PART.replace(FINAL)
    put(REC/'weight_completion.json',{'time':now(),'state':'official_asset_size_SHA_PASS','weight':str(FINAL),'size':TOTAL,'SHA256':digest,'additional_network_bytes':got,'logical_requests':1,'old_partial_preserved':True,'old_partial_SHA256':OLD_SHA,'wall_seconds':time.monotonic()-START})

def main():
    assert not (REC/'download_launch.json').exists(),'Exactly one recovery launch permitted'
    put(REC/'download_launch.json',{'time':now(),'pid':os.getpid(),'protocol_SHA256':sha(REC/'protocol.txt'),'code_SHA256':sha(__file__),'user_authorization':'One official Range request <=276.1MB/900s; failure stops; no R/N rerun'})
    done=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--worker'],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',timeout=900,creationflags=subprocess.CREATE_NO_WINDOW)
    (REC/'download.log').write_text(done.stdout,encoding='utf-8');assert done.returncode==0,'Authorized download failed; no retry'
    print(json.dumps(read(REC/'weight_completion.json'),ensure_ascii=False))

if __name__=='__main__':
    try:
        if '--worker' in sys.argv:worker()
        else:main()
    except Exception as e:
        put(REC/('download_worker_stop.json' if '--worker' in sys.argv else 'download_stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'wall_seconds':time.monotonic()-START,'traceback':traceback.format_exc()});raise
