from pathlib import Path
import json
import audit_sources as a
a.ledger=json.loads((a.D/'request_ledger.json').read_text(encoding='utf-8'))
a.blocked=set(json.loads((a.D/'audit_summary.json').read_text(encoding='utf-8'))['blocked_domains'])
commit=json.loads((a.D/'HuBERT_commit.json').read_text(encoding='utf-8'))['sha']
for file in ['hubert_ecg/utils.py','hubert_ecg/dataset.py','pyproject.toml']:
    a.get(f'https://raw.githubusercontent.com/Edoar-do/HuBERT-ECG/{commit}/{file}','repo_'+file.replace('/','__'))
