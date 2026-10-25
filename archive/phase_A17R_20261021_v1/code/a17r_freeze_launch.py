"""One-time source freeze, contract verification and native WSL launch."""
import sys,subprocess,shutil,json,platform,importlib.metadata
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from a17r_common import *
import numpy as np,torch
def main():
 RUN.mkdir(parents=True,exist_ok=True)
 assert not (RUN/'launch_receipt.json').exists(),'Existing launch: inspect live PID, never duplicate'
 code=RUN/'code';code.mkdir(exist_ok=True)
 names=['a17r_common.py','a17r_features.py','a17r_driver.py','a17r_freeze_launch.py','a17r_protocol_20261021.txt','ecgfm_frozen.py','acquire_a3.py']
 for name in names:shutil.copy2(HERE/name,code/name)
 shutil.copytree(HERE.parent/'ecgfm_upstream',RUN/'upstream',dirs_exist_ok=True)
 files={str(code/n):sha(code/n) for n in names}
 files.update({str(p):sha(p) for p in [META,DATA/'scp_statements.csv',DATA/'SHA256SUMS.txt',PW,FM]})
 assert files[str(FM)]==FM_SHA
 files.update({str(p):sha(p) for p in (RUN/'upstream').rglob('*.py')})
 write(RUN/'source_lock.json',dict(frozen_at=now(),files=files,planned_fits=28,max_fits=30,no_new_P_training=True,heldout_source='PTB previously viewed; adaptive development'))
 from a17r_features import morphology,Net,load_fm
 zz=np.r_[np.linspace(0,-.1,26),np.linspace(-.1,0,26)[1:]];m=morphology(zz,dict(a=0,b=50));assert -.11<m['ptf']<-.09
 theta=np.deg2rad(30);ai=np.cos(theta);aii=.5*np.cos(theta)+np.sqrt(3)/2*np.sin(theta);axis=np.arctan2((2*aii-ai)/np.sqrt(3),ai);assert abs(axis-theta)<1e-12
 pp=Net();pp.load_state_dict(torch.load(PW,map_location='cpu',weights_only=True));pp.eval()
 with torch.inference_mode():
  x=torch.randn(1,2,200);a=(pp(x)+pp(-x))/2;b=(pp(-x)+pp(x))/2;assert torch.equal(a,b)
 frozen=load_fm('cuda:1');assert all(not x.requires_grad for x in frozen.parameters())
 with torch.inference_mode():error=frozen.upstream_parity(torch.zeros(1,12,3000,device='cuda:1'))
 assert error<1e-6;del frozen;torch.cuda.empty_cache()
 write(RUN/'preflight.json',dict(P_weight_load='PASS',FM_strict_weight_load='PASS',FM_exact_upstream_max_error=error,PTF_physical_units='PASS',axis_projection='PASS',sign_invariance='PASS',classifier_fits_executed=0))
 packages={n:importlib.metadata.version(n) for n in ['torch','numpy','pandas','scipy','scikit-learn','wfdb','omegaconf','matplotlib','joblib']}
 write(RUN/'runtime.json',dict(python=platform.python_version(),packages=packages,python_executable=sys.executable,system_site_packages=True,devices=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]))
 (RUN/'requirements_observed.txt').write_text('\n'.join(f'{n}=={v}' for n,v in packages.items())+'\n',encoding='utf-8')
 with (RUN/'program.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',str(code/'a17r_driver.py')],cwd=code,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 receipt=dict(pid=child.pid,started_at=now(),python=sys.executable,max_classifier_fits=30,planned_classifier_fits=28,protocol=str(code/'a17r_protocol_20261021.txt'),native_run=str(RUN),state='launched_after_preflight')
 write(RUN/'launch_receipt.json',receipt);print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
