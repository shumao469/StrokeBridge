"""Mock-only helper contract tests. No checkpoints, ECGs, models or fits loaded."""
from a24_shared import *
import ast,types,torch
from E_helper_contracts import run_timing_benchmark
OUT=P/'software_contracts_E_helper_v1'
def main():
    assert not (OUT/'completion.json').exists();OUT.mkdir(exist_ok=True)
    files=[Path(__file__),P/'code/E_helper_contracts.py',P/'code/execute_E_cpu_once.py',P/'code/execute_E_cpu8h_corrected_once.py'];put(OUT/'source_lock.json',{'time':now(),'files':{str(f):sha(f) for f in files}})
    torch.set_num_threads(2);calls={'E1_mock':0,'E2_mock':0};memory={};root=Path('MOCK_ONLY_NEVER_WRITTEN')
    class MockEncoder:
        def __init__(self,name):self.name=name
        def parameters(self):return iter([torch.zeros(1,requires_grad=False)])
        def __call__(self,x):
            calls[self.name]+=1
            return torch.zeros(x.shape[0],768) if self.name=='E1_mock' else types.SimpleNamespace(last_hidden_state=torch.zeros(x.shape[0],2,768))
    def memory_put(path,value):memory[str(path)]=value
    run_timing_benchmark(MockEncoder('E1_mock'),MockEncoder('E2_mock'),root,time.monotonic(),memory_put)
    record=memory[str(root/'extraction/CPU_benchmark.json')]
    assert len(record['synthetic_timing_batches'])==3 and record['GPU_forward_calls']==0 and record['admitted_for_bulk'] is False and record['time']
    # Test the following orchestration callback using in-memory dependencies only.
    source=P/'code/execute_E_cpu8h_corrected_once.py';text=source.read_text(encoding='utf-8');tree=ast.parse(text);node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='preprocess_unified')
    scope={'time':time,'START':time.monotonic(),'REC':root,'read':lambda path:memory[str(path)],'put':memory_put,'prev':types.SimpleNamespace(PORT=types.SimpleNamespace(preprocessing_admission=lambda *args:None))}
    exec(compile(ast.get_source_segment(text,node),str(source)+'[mock callback contract]','exec'),scope);scope['preprocess_unified'](None,None,None)
    assert memory[str(root/'extraction/CPU_benchmark.json')]['admitted_for_bulk'] is True
    assert not root.exists(),'Mock records must never be written as scientific artifacts'
    verify(read(OUT/'source_lock.json')['files'])
    result={'time':now(),'state':'mock_only_helper_contracts_PASS','now_dependency_bound':True,'timing_callback':'PASS','preprocessing_orchestration_callback':'PASS','real_encoder_loads':0,'real_encoder_forwards':0,'ECG_reads':0,'fits':0,'network':0,'mock_calls':calls,'scope':'Software namespace/record contracts only;not scientific preflight/admission/effectiveness.'};put(OUT/'completion.json',result);print(json.dumps(result,ensure_ascii=False))
if __name__=='__main__':main()
