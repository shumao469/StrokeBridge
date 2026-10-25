"""No network or fitting: duplicated references and received partial ZIP formatting."""
import json,zipfile,hashlib,struct,zlib,csv,io
from pathlib import Path
from decimal import Decimal
from datetime import datetime
import numpy as np
RUN=Path(__file__).resolve().parents[1];A25=RUN.parents[1]/'runs/phase_A25_20261023_v1';OUT=RUN/'reference_audit';OUT.mkdir(exist_ok=True)
z=zipfile.ZipFile(A25/'data/dataset.zip');rows=[];summary=[]
for lead in ['i','ii','iii','avr','avl','avf','v1','v2','v3','v4','v5','v6','all']:
    aa=list(csv.DictReader(io.StringIO(z.read('ann_txt/SI0109.'+lead+'.txt').decode('utf-8-sig'))));bb=list(csv.DictReader(io.StringIO(z.read('ann_txt/SI0211.'+lead+'.txt').decode('utf-8-sig'))))
    for kind in ['0','1','2']:
        a=[r for r in aa if r['TYPE']==kind];b=[r for r in bb if r['TYPE']==kind];d=[]
        # Symmetric nearest endpoint discrepancy: descriptive, not adjudication or waveform matching.
        for x,other in [(a,b),(b,a)]:
            for v in x:
                if other:d.append(min(max(abs(Decimal(v['START'])-Decimal(w['START'])),abs(Decimal(v['END'])-Decimal(w['END']))) for w in other))
        summary.append(dict(lead=lead,type=kind,SI0109_rows=len(a),SI0211_rows=len(b),exact_equal=a==b,symmetric_nearest_endpoint_median_ms=float(np.median([float(v)*2 for v in d])) if d else None,symmetric_nearest_endpoint_max_ms=float(max(d)*2) if d else None))
raw_a=z.read('dat_csv/SI0109.csv');raw_b=z.read('dat_csv/SI0211.csv');partial=RUN/'source/ECGData.zip.partial';format_info={}
if partial.exists():
    with partial.open('rb') as f:
        for _ in range(5):
            h=f.read(30)
            if len(h)!=30 or h[:4]!=b'PK\x03\x04':break
            fields=struct.unpack('<IHHHHHIIIHH',h);flag,method,compressed,name_len,extra_len=fields[2],fields[3],fields[7],fields[9],fields[10];name=f.read(name_len).decode(errors='replace');f.read(extra_len)
            if compressed>2_000_000 or flag&8:format_info=dict(member=name,method=method,data_descriptor=bool(flag&8),local_header_compressed_bytes=compressed);break
            payload=f.read(compressed)
            if name.lower().endswith('.csv'):
                content=zlib.decompress(payload,-15) if method==8 else payload
                format_info=dict(first_member=name,parent_first_lines=content.decode('utf-8-sig').splitlines()[:3],rdb_example_first_lines=raw_a.decode('utf-8-sig').splitlines()[:2],assurance='format diagnostic only; this member not asserted to map SI0109');break
report=dict(time=datetime.now().astimezone().isoformat(),duplicate_waveform_byte_identical=raw_a==raw_b,duplicate_waveform_SHA256=hashlib.sha256(raw_a).hexdigest(),patient_groups=2398,RDB_annotation_versions=2399,duplicate_annotations=summary,source_partial_format=format_info,clinical_noP_gold=False,V1_specific_RDB_gold=False,AF_training_allowed=False,RDB_scoring_allowed=False,figures=0)
(OUT/'completion.json').write_text(json.dumps(report,ensure_ascii=False,indent=2));print(json.dumps(report,ensure_ascii=False))
