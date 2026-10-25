"""JSON-only paired component review, no inference/fitting or operating-point changes."""
import json,itertools,time,hashlib
from pathlib import Path
from datetime import datetime
RUN=Path(__file__).resolve().parents[1];OUT=RUN/'attribution';start=time.monotonic()
assert not (OUT/'source_pair_fairness.json').exists(),'No automatic rerun'
points=json.loads((OUT/'calibration_selected_operating_points.json').read_text());paired=[]
keys=sorted({(v['index'],v['target'],v['lead']) for v in points})
for index,target,lead in keys:
    assert time.monotonic()-start<60
    group={v['mode']:v for v in points if v['index']==index and v['target']==target and v['lead']==lead}
    for a,b in itertools.combinations(['continuous','ordered','union'],2):
        x,y=group[a],group[b];feasible=x['feasible'] and y['feasible'];matched=feasible and abs(x['outer']['recall']-y['outer']['recall'])<=.02
        paired.append(dict(model_index=index,arm=x['arm'],seed=x['seed'],fold=x['fold'],lead=lead,target=target,source_A=a,source_B=b,calibration_both_feasible=feasible,outer_coverage_matched=bool(matched),A=x.get('outer'),B=y.get('outer'),threshold_A=x.get('threshold'),threshold_B=y.get('threshold'),advantage_estimable_at_matched_coverage=bool(matched)))
summary=[]
for a,b in itertools.combinations(['continuous','ordered','union'],2):
    for arm in sorted({v['arm'] for v in points}):
        for lead in ['ii','v1']:
            vv=[v for v in paired if v['source_A']==a and v['source_B']==b and v['arm']==arm and v['lead']==lead and v['target']==.8]
            summary.append(dict(arm=arm,lead=lead,target=.8,sources=[a,b],model_pairs=len(vv),calibration_feasible=sum(v['calibration_both_feasible'] for v in vv),outer_matched=sum(v['outer_coverage_matched'] for v in vv)))
report=dict(time=datetime.now().astimezone().isoformat(),protocol='Existing N0 within-model source comparison; derived from unchanged calibration-selected thresholds',source_SHA256=hashlib.sha256((OUT/'calibration_selected_operating_points.json').read_bytes()).hexdigest(),code_SHA256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),new_fits=0,new_epochs=0,new_inference=0,summary=summary,comparisons=paired)
(OUT/'source_pair_fairness.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,allow_nan=False));print(json.dumps(summary))
