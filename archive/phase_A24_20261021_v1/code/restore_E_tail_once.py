"""One newly authorized <=1MB/120s official tail request, no retry."""
from a24_shared import *
import urllib.request,urllib.parse,subprocess,shutil,traceback
REC=P/'recovery_E_tail_v1';ROOT=Path('DRIVE_ROOT/脑心同查/models/hubert-ecg-base/8ffa1ad3593366ca45f5351d75317d4ff8f14465')
OLD=ROOT/'model.safetensors.A24_E_recovery.partial';PART=ROOT/'model.safetensors.A24_E_tail.partial';FINAL=ROOT/'model.safetensors'
PREFIX=371830542;TOTAL=372518672;OLD_SHA='fbe9556c414d27af4f6027f248cc3b9913a9255709cfb359e9c664068ca24411';EXPECTED='05bc1b1317f8e3063811a03fb840f5bf8a85968191e209c0cfbaa0c52c8aa1ae'
URL='https://huggingface.co/Edoardo-Coppola/hubert-ecg-base/resolve/8ffa1ad3593366ca45f5351d75317d4ff8f14465/model.safetensors';START=time.monotonic()
class ProviderRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self,req,fp,code,msg,headers,newurl):
        parsed=urllib.parse.urlparse(newurl);host=parsed.hostname or '';assert parsed.scheme=='https' and (host=='huggingface.co' or host.endswith('.huggingface.co') or host=='hf.co' or host.endswith('.hf.co'))
        f=REC/'redirects.json';rows=read(f) if f.exists() else [];rows.append({'code':code,'host':host});put(f,rows);return super().redirect_request(req,fp,code,msg,headers,newurl)
def worker():
    assert not PART.exists() and not FINAL.exists();assert OLD.stat().st_size==PREFIX and sha(OLD)==OLD_SHA
    shutil.copy2(OLD,PART);assert PART.stat().st_size==PREFIX and sha(PART)==OLD_SHA
    put(REC/'request_started.json',{'time':now(),'url':URL,'Range':f'bytes={PREFIX}-','logical_request':1,'expected_bytes':TOTAL-PREFIX,'max_bytes':1000000})
    request=urllib.request.Request(URL,headers={'User-Agent':'A24-Authorized-Tail/1.0','Range':f'bytes={PREFIX}-','Accept-Encoding':'identity'})
    with urllib.request.build_opener(ProviderRedirect()).open(request,timeout=45) as response:
        cr=response.headers.get('Content-Range');length=response.headers.get('Content-Length')
        put(REC/'response_headers.json',{'time':now(),'status':response.status,'Content-Range':cr,'Content-Length':length,'final_host':urllib.parse.urlparse(response.url).hostname})
        assert response.status==206 and cr==f'bytes {PREFIX}-{TOTAL-1}/{TOTAL}','Range not honored:stop'
        assert response.headers.get('Content-Encoding','identity')=='identity';assert length is None or int(length)==TOTAL-PREFIX
        body=response.read(1000001);assert len(body)==TOTAL-PREFIX and len(body)<=1000000,'Short/extra tail:stop'
    with PART.open('ab') as f:f.write(body)
    assert time.monotonic()-START<120 and PART.stat().st_size==TOTAL
    digest=sha(PART);assert digest==EXPECTED,'Complete official SHA mismatch:stop'
    assert OLD.stat().st_size==PREFIX and sha(OLD)==OLD_SHA;PART.replace(FINAL)
    put(REC/'weight_completion.json',{'time':now(),'state':'official_asset_size_SHA_PASS','weight':str(FINAL),'size':TOTAL,'SHA256':digest,'additional_network_bytes':len(body),'logical_requests':1,'prior_partials_preserved':True,'wall_seconds':time.monotonic()-START})
def main():
    assert not (REC/'download_launch.json').exists(),'One launch only'
    put(REC/'download_launch.json',{'time':now(),'pid':os.getpid(),'protocol_SHA256':sha(REC/'protocol.txt'),'code_SHA256':sha(__file__),'max_seconds':120,'max_additional_bytes':1000000})
    q=subprocess.run([sys.executable,'-X','utf8','-B',str(Path(__file__).resolve()),'--worker'],capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120,creationflags=subprocess.CREATE_NO_WINDOW)
    (REC/'download.log').write_text(q.stdout+q.stderr,encoding='utf-8');assert q.returncode==0,'Tail recovery failed,no retry'
    print(json.dumps(read(REC/'weight_completion.json'),ensure_ascii=False))
if __name__=='__main__':
    try:worker() if '--worker' in sys.argv else main()
    except Exception as e:put(REC/('download_worker_stop.json' if '--worker' in sys.argv else 'download_stop.json'),{'time':now(),'state':'stopped_no_retry','error':str(e),'wall_seconds':time.monotonic()-START,'traceback':traceback.format_exc()});raise
