"""Preserve byte-verified as-generated source snapshots after adding the user's drawing stop guards."""
from pathlib import Path
import hashlib,json
P=Path(__file__).resolve().parents[1];U=P.parents[1];F=U/'manuscript_NC_A24_20261023'
OUT=U/'A24_verified_review_20261023/as_generated_figure_sources';OUT.mkdir(exist_ok=True)
sha=lambda b:hashlib.sha256(b).hexdigest()
python_header="""_policy_path=Path(__file__).resolve().parents[1]/'FIGURE_POLICY.json'
if __name__=='__main__' and _policy_path.exists() and not any(x in sys.argv for x in ('--check-code','--backend-smoke')):
    _policy=json.loads(_policy_path.read_text(encoding='utf-8-sig'))
    if not _policy.get('allow_new_result_figures',False):
        print('RESULT_DRAWING_DISABLED_BY_USER_PROMPT_ONLY_POLICY');raise SystemExit(0)
"""
python_main="""    if _policy_path.exists() and not json.loads(_policy_path.read_text(encoding='utf-8-sig')).get('allow_new_result_figures',False):
        raise RuntimeError('Result drawing disabled by the user; written design only')
"""
js_guard="""const policyFile=path.join(CODE,'../FIGURE_POLICY.json');
if(!process.argv.includes('--mock')){
  const policy=await fs.readFile(policyFile,'utf8').then(JSON.parse).catch(error=>{if(error.code==='ENOENT')return null;throw error;});
  if(policy&&!policy.allow_new_PPTX){console.log('PPTX_DISABLED_BY_USER_PROMPT_ONLY_POLICY');process.exit(0);}
}
"""
rows=[]
for name,lock_path,blocks in [('build_A24_figures.py',F/'figure_source_lock.json',[python_header,python_main]),
                             ('build_A24_editable.mjs',F/'.pptx-build/pptx_source_lock.json',[js_guard])]:
    file=P/'code'/name;source=file.read_text(encoding='utf-8');old=source
    for block in blocks:
        assert old.count(block)==1,(name,'Expected policy guard not found');old=old.replace(block,'',1)
    entries=json.loads(lock_path.read_text(encoding='utf-8-sig'))['files']
    expected=next(value for key,value in entries.items() if key.replace('\\','/').endswith('/code/'+name))
    payload=old.encode('utf-8');assert sha(payload)==expected,(name,'Historical source mismatch')
    target=OUT/(name+'.as_generated.txt');assert not target.exists();target.write_bytes(payload)
    rows.append({'source':str(file),'as_generated_snapshot':str(target),'as_generated_SHA256':expected,
                 'current_guarded_SHA256':sha(file.read_bytes()),'change':'Only explicit drawing-stop policy guards removed for snapshot reconstruction; pre-existing generation SHA verified.'})
(OUT/'source_history.json').write_text(json.dumps({'state':'PASS','historical_locks_unchanged':True,
    'research_artifacts_unchanged':True,'source_snapshots':rows},ensure_ascii=False,indent=2),encoding='utf-8')
print('AS_GENERATED_SOURCE_SNAPSHOTS_SHA_PASS_NO_ARTWORK_CHANGE')
