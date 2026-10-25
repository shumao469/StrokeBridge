"""Bounded coordinator; no classifier retry or implicit extra training."""
from a17r_common import *
import time,traceback,importlib
def main():
 check_freeze()
 from a17r_features import extract
 extract()
 start=__import__('time').monotonic()
 while not (RUN/'analysis_source_lock.json').exists():
  if (RUN/'STOP_NEW_WORK').exists():raise RuntimeError('Stopped before fits')
  if time.monotonic()-start>6*3600:raise TimeoutError('Analysis source wait bound')
  time.sleep(5)
 pin=read(RUN/'analysis_source_lock.json');assert all(sha(p)==h for p,h in pin.items()),'Analysis sources changed'
 from a17r_model_analysis import run
 run()
if __name__=='__main__':
 try:main()
 except Exception:
  failure=traceback.format_exc();write(RUN/'STOP_NEW_WORK',dict(reason=failure,time=now()));status('failed',state='stopped_no_automatic_retry',error=failure);raise
