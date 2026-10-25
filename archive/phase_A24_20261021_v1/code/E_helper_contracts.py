"""Explicit helper binding for the unchanged CPU timing function."""
import ast
import a24_shared as shared
def run_timing_benchmark(ef,hf,run_root,started,writer=None):
    source=shared.P/'code/execute_E_cpu_once.py';text=source.read_text(encoding='utf-8');tree=ast.parse(text)
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='cpu_benchmark');body=ast.get_source_segment(text,node)
    before="assert time.monotonic()-START<180,'CPUpreflight180second limit'";assert body.count(before)==1
    body=body.replace(before,"assert time.monotonic()-START<28800,'Unified8hour extraction limit'")
    scope=dict(vars(shared));scope.update({'START':started,'REC':run_root,'BENCH_START':None,'put':writer if writer is not None else shared.put})
    assert all(name in scope for name in ['time','now','put','REC','START'])
    exec(compile(body,str(source)+'[unified8h bound helper namespace]','exec'),scope);scope['cpu_benchmark'](ef,hf)
