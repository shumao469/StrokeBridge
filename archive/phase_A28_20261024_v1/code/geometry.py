"""Independent audit geometry. Diagnostic policies never overwrite frozen metrics."""
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

def iou(ref,pred,inclusive=False):
    a,pk,b=ref;add=1 if inclusive else 0
    return max(0,min(b,pred['offset'])-max(a,pred['onset'])+add)/max(max(b,pred['offset'])-min(a,pred['onset'])+add,1)

def assignment(scores):
    scores=np.asarray(scores,dtype=float)
    if scores.size==0 or not scores.shape[0] or not scores.shape[1]:return []
    allowed=scores>0;reward=allowed*(max(scores.shape)+1+scores)
    ii,jj=linear_sum_assignment(reward,maximize=True)
    return [(int(i),int(j),float(scores[i,j])) for i,j in zip(ii,jj) if allowed[i,j]]

def optimal(ref,pred,inclusive=False):
    if not ref or not pred:return []
    scores=np.array([[iou(r,p,inclusive) for p in pred] for r in ref]);scores[scores<.5]=0
    return assignment(scores)

def boxes_match(ref,pred):
    if not ref or not pred:return []
    scores=np.array([[float(p['box'][0]<=r[0]<=p['box'][1] and p['box'][2]<=r[2]<=p['box'][3]) for p in pred] for r in ref])
    return assignment(scores)

def nms_trace(pred):
    survivors=[];suppressed=[]
    for p in sorted(pred,key=lambda v:v['confidence'],reverse=True):
        blocker=next((v for v in survivors if iou((p['onset'],p['peak'],p['offset']),v)>=.5),None)
        if blocker is None:survivors.append(p)
        else:suppressed.append(dict(victim=p['uid'],winner=blocker['uid'],overlap=iou((p['onset'],p['peak'],p['offset']),blocker),victim_source=p['kind'],winner_source=blocker['kind'],different_source=p['kind']!=blocker['kind'],tied_score=p['confidence']==blocker['confidence']))
    return survivors,suppressed

def next_beat(offset,qrs):
    values=[(i,q) for i,q in enumerate(qrs) if q>offset]
    if not values:return None
    index,q=values[0]
    return index if q-offset<=250 else None

def anchor_beat(anchor,qrs):
    if anchor is None or not qrs:return None
    index=min(range(len(qrs)),key=lambda i:abs(qrs[i]-anchor))
    return index if abs(qrs[index]-anchor)<=50 else None

def residual(ref,pred,normalised=True,same_beat=False,ref_beat=None):
    candidates=[]
    for p in pred:
        if same_beat and (ref_beat is None or p['expert_beat']!=ref_beat):continue
        distance=max(p['box'][0]-ref[0],ref[0]-p['box'][1],p['box'][2]-ref[2],ref[2]-p['box'][3],0)
        candidates.append((distance/(p['scale'] if normalised else 1),distance,p['uid']))
    return min(candidates,default=(float('inf'),float('inf'),None))

def expanded(pred,q,normalised=True):
    if not math.isfinite(q):return []
    out=[]
    for p in pred:
        a,b,c,d=p['box'];s=p['scale'] if normalised else 1
        out.append(dict(p,box=[a-q*s,b+q*s,c-q*s,d+q*s]))
    return out

def preflight():
    scores=np.array([[.9,.8],[.8,0.]])
    assert len(assignment(scores))==2
    ref=[(10,20,30),(100,110,120)];p=[dict(onset=10,peak=20,offset=30,confidence=.9,kind='ordered',uid=0,box=[0,150,0,150],scale=5),dict(onset=11,peak=20,offset=31,confidence=.8,kind='continuous',uid=1,box=[11,11,31,31],scale=5)]
    kept,suppressed=nms_trace(p);assert len(kept)==1 and len(suppressed)==1
    assert len(boxes_match(ref,[p[0]]))==1  # Union can cover two waves, one-to-one cannot.
    assert abs((expanded([p[0]],9.6)[0]['box'][1]-expanded([p[0]],9.6)[0]['box'][0])-(150+96))<1e-12
    point=dict(p[0],box=[10,10,30,30]);assert 2*(expanded([point],9.6)[0]['box'][1]-expanded([point],9.6)[0]['box'][0])==192
    assert abs(2*(expanded([point],10.2)[0]['box'][1]-expanded([point],10.2)[0]['box'][0])-204)<1e-12
    assert next_beat(150,[100,300])==1 and next_beat(150,[100,500]) is None
    assert anchor_beat(295,[100,300])==1 and anchor_beat(220,[100,300]) is None
    assert len(optimal(ref,[]))==0
    return ['maximum_cardinality_adversarial_graph','NMS_source_trace','union_vs_one_to_one_boxes','sample_ms_width_192_204','unresolved_beat_kept_unresolved','empty_candidate_geometry']
