"""Mock-only derived-checkpoint round trip. No ECG/model loading or fitting."""
from a24_shared import *
import importlib.util,numpy as np
OUT=P/'software_contracts_E_checkpoint_v1'
def main():
    assert not (OUT/'completion.json').exists();OUT.mkdir(exist_ok=True)
    code=P/'code/execute_E_cpu8h_once.py';put(OUT/'source_lock.json',{'time':now(),'files':{str(code):sha(code),str(Path(__file__)):sha(__file__)}})
    spec=importlib.util.spec_from_file_location('checkpoint_mock_only',code);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);mod.REC=OUT/'MOCK_NOT_SCIENTIFIC';(mod.REC/'extraction').mkdir(parents=True)
    ids=np.arange(21002)+99000000;e1=np.zeros((21002,768),dtype='float32');e2=np.ones((21002,768),dtype='float32')
    for end in [*range(1024,21002,1024),21002]:mod.checkpoint(ids[:end].tolist(),e1[:end],e2[:end])
    rows=read(mod.REC/'extraction/checkpoints/manifest.json');offset=0
    for row in rows:
        assert row['start']==offset and sha(row['file'])==row['SHA256'];part=np.load(row['file']);end=row['end_exclusive']
        for key,source in [('ecg_id',ids),('E1',e1),('E2',e2)]:assert np.array_equal(part[key],source[offset:end])
        offset=end
    assert offset==21002 and len(rows)==21;verify(read(OUT/'source_lock.json')['files'])
    q={'time':now(),'state':'mock_only_checkpoint_contract_PASS','mock_records':21002,'chunks':21,'exact_array_replay':True,'ECG_reads':0,'real_encoder_forwards':0,'fits':0,'network':0,'scope':'Synthetic storage-contract QA only;not scientific records or embeddings.'};put(OUT/'completion.json',q);print(json.dumps(q,ensure_ascii=False))
if __name__=='__main__':main()
