"""Resume only verified pre-fit A17R extraction after user restores Ubuntu."""
import sys,subprocess,json,shutil,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
sys.path.insert(0,str(HERE))
from a17r_common import *
def pin_analysis_without_overwrite(ledger):
 source=HERE/'a17r_model_analysis.py';analysis=RUN/'code/a17r_model_analysis.py';lock=RUN/'analysis_source_lock.json'
 # The D: copy was fixed while Linux was unavailable, before any analysis fit.
 manifest=read(BASE/'runs/phase_A17R_20261021_v1/D_source_manifest.json')
 source_expected=next(x['sha256'] for x in manifest['files'] if x['file']=='a17r_model_analysis.py')
 assert sha(source)==source_expected,'Working analysis changed relative to fixed D source'
 expected={str(analysis):source_expected}
 if lock.exists():
  assert read(lock)==expected,'Existing analysis source lock differs; preserve it'
  assert analysis.exists() and sha(analysis)==source_expected,'Pinned native analysis differs'
  return
 assert not ledger,'Cannot create a first analysis lock after fitting started'
 if analysis.exists():assert sha(analysis)==source_expected,'Existing native analysis differs; preserve it'
 else:shutil.copy2(source,analysis)
 write(lock,expected)
def main():
 assert RUN.exists(),'Original native run unavailable: restore files before any new launch'
 receipt=read(RUN/'launch_receipt.json');pid=receipt['pid'];proc=Path(f'/proc/{pid}/cmdline')
 ledger=read(RUN/'fit_ledger.json') if (RUN/'fit_ledger.json').exists() else []
 if proc.exists():
  cmd=proc.read_bytes().replace(b'\0',b' ').decode(errors='replace')
  if 'a17r_driver.py' in cmd:
   check_freeze();pin_analysis_without_overwrite(ledger)
   print(json.dumps(dict(state='original_coordinator_live_analysis_lock_verified_no_duplicate',pid=pid,cmd=cmd,fit_attempts=len(ledger))));return
 assert not ledger,'Existing fit attempts require ledger-specific reconciliation, no automatic restart'
 assert not (RUN/'recovery_launch_receipt.json').exists(),'Recovery already attempted; inspect its live PID first'
 check_freeze()
 # Archive the explicit infrastructure failure marker rather than dropping it.
 barrier=RUN/'STOP_NEW_WORK'
 if barrier.exists():
  old=read(barrier);reason=str(old)
  assert any(v in reason for v in ['Input/output error','No space left on device','I/O error']), 'Non-storage failure requires diagnosis before recovery'
  assert not (RUN/'storage_failure_original.json').exists();barrier.rename(RUN/'storage_failure_original.json')
 # Check old locks before any write; do not overwrite a differing prior source.
 pin_analysis_without_overwrite(ledger)
 with (RUN/'program.log').open('a') as log:
  child=subprocess.Popen([sys.executable,'-u',str(RUN/'code/a17r_driver.py')],cwd=RUN/'code',stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 out=dict(time=now(),state='storage_recovery_pre_fit_resume',pid=child.pid,original_pid=pid,prior_fit_attempts=0,max_classifier_fits=30,planned_fits=28,protocol_unchanged=True)
 write(RUN/'recovery_launch_receipt.json',out);print(json.dumps(out,indent=2))
if __name__=='__main__':main()
