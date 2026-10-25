"""Freeze before actual A25 fitting/inference; mock tests and read-only input admission."""
from common import *
import sys,ast,importlib.util,traceback
import numpy as np,pandas as pd,torch
from intervals import selftest as interval_test
from missingness import selftest as missing_test
def main():
    assert not (RUN/'source_lock.json').exists(),'No refreeze'
    assert read(RUN/'local_receipt/completion.json')['files_integrity']=='PASS'
    assert read(RUN/'reference_refinement/completion.json')['state']=='complete'
    assert read(RUN/'continuous_coordinate_audit/completion.json')['state']=='complete'
    admission=read(RUN/'scientific_admission.json');assert not admission['SPH_admitted']
    assert admission['existing_LUDB_development_admitted'] and admission['existing_PTB_AF_development_admitted']
    tests=dict(intervals=interval_test(),missingness=missing_test())
    # Syntax and shape checks do not load scientific weights, infer native ECGs or fit labels.
    for p in (RUN/'code').glob('*.py'):ast.parse(p.read_text(encoding='utf-8-sig'))
    module=BASE/'runs/phase_A9_20261017_v1/code/a9_method.py'
    spec=importlib.util.spec_from_file_location('a25_shape_only_a9',module);api=importlib.util.module_from_spec(spec);spec.loader.exec_module(api);torch.set_num_threads(1)
    synthetic=np.zeros((2,5000),np.float32);channels=api.channels(synthetic);assert channels.shape==(2,2,5000)
    with torch.inference_mode():logits=api.Net().eval()(torch.from_numpy(channels))
    assert logits.shape==(2,4,5000) and torch.isfinite(logits).all();tests['synthetic_shape']='PASS'
    z=np.load(A9/'input/data.npz');assert z['signals'].shape==(200,5000,12) and z['labels'].shape==(200,12,5000)
    roles=read(A9/'input/patient_roles.json');outer=[]
    for r in roles:
        sets=[set(r[k]) for k in ['train','selection','calibration','outer']]
        assert all(not a&b for i,a in enumerate(sets) for b in sets[i+1:]);outer.extend(r['outer'])
    assert sorted(outer)==list(range(200))
    d=pd.read_csv(A17/'analysis_dataset.csv');assert len(d)==21002 and d.patient_id.nunique()==18260 and d.groupby('patient_id').role.nunique().max()==1
    assert read(BASE/'A24_verified_review_20261023/verification.json')['state']=='PASS'
    files=[RUN/'protocol.txt',RUN/'local_receipt_protocol.json',RUN/'scientific_admission.json',RUN/'reference_refinement/completion.json',RUN/'continuous_coordinate_audit/completion.json',RUN/'local_receipt/completion.json',*sorted((RUN/'code').glob('*.py')),module,A9/'input/data.npz',A9/'input/patient_roles.json',A9/'input/pwave_annotations.json',A17/'analysis_dataset.csv',A18/'factor_definitions.json',A24/'extraction/common5s_embeddings.npz',BASE/'A24_verified_review_20261023/verification.json',RUN/'data/dataset.zip',RUN/'data/rdb_chapman_mapping.xlsx']
    files += [A9/f'C1_sign_orbit/seed_{s}/fold_{f}/best.pt' for s in [20261020,20261021] for f in range(5)]
    put(RUN/'preflight.json',dict(time=now(),state='PASS',tests=tests,native_P_checkpoint_forwards=0,scientific_fits=0,patient_role_overlap=0,original_A24_fits=422,planned_new_A25_fits=88))
    put(RUN/'source_lock.json',dict(time=now(),state='FROZEN_BEFORE_MODEL_RESULTS',files={str(p):sha(p) for p in files},supervised_fit_cap=88,aux_imputer_fit_cap=44,aux_scaler_fit_cap=44,neural_epochs=0,SPH_admitted=False,RDB_admitted=admission['RDB_admitted'],new_figures_allowed=False,independent_unseen_encoder_validation=False))
    print('A25_PREFLIGHT_AND_FREEZE_PASS',flush=True)
if __name__=='__main__':main()
