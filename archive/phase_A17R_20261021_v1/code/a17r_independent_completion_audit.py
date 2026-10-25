"""Post-run artifact audit and saved-model replay. Never fit or select a new model."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from a17r_common import *
import numpy as np
import pandas as pd
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve

def forbid_fit(*args, **kwargs):
    raise RuntimeError('Post-run audit forbids every classifier/calibrator fit')

LogisticRegression.fit = forbid_fit

def main():
    assert (RUN/'numeric_completion_receipt.json').exists(), 'Main numerical run not complete'
    assert not (RUN/'STOP_NEW_WORK').exists(), 'Unresolved main-run failure'
    check_freeze()
    source_lock = read(RUN/'analysis_source_lock.json')
    assert all(sha(p) == h for p, h in source_lock.items())
    ledger = read(RUN/'fit_ledger.json')
    ledger_hash = sha(RUN/'fit_ledger.json')
    assert len(ledger) == 28 <= MAX_FITS
    assert len({x['name'] for x in ledger}) == 28
    assert all(x['status'] == 'complete' and x['converged'] for x in ledger)
    assert sum('/Platt_fold9' in x['name'] for x in ledger) == 7
    metadata = pd.read_csv(RUN/'metadata_roles_labels.csv')
    data = pd.read_csv(RUN/'analysis_dataset.csv')
    raw = pd.read_csv(RUN/'waveform_audit.csv')
    features = pd.read_csv(RUN/'physical_and_candidate_features.csv')
    pred = pd.read_csv(RUN/'role_predictions.csv')
    evaluation = pd.read_csv(RUN/'evaluation_predictions.csv')
    assert len(metadata) == len(raw) == 21799
    assert raw.ecg_id.is_unique and metadata.ecg_id.is_unique
    assert set(raw.ecg_id) == set(metadata.ecg_id)
    assert raw.fs.eq(500).all() and raw.n.eq(5000).all() and raw.unit.eq('mV').all()
    assert raw.header_SHA256.str.fullmatch('[0-9a-f]{64}').all()
    assert raw.data_SHA256.str.fullmatch('[0-9a-f]{64}').all()
    assert metadata.groupby('patient_id').role.nunique().max() == 1
    eligible = metadata[metadata.label_eligible.eq(True)]
    passed = raw[raw.QC_exclusion.isna() | raw.QC_exclusion.eq('')]
    assert set(data.ecg_id) == set(eligible.ecg_id) & set(passed.ecg_id)
    assert data.ecg_id.is_unique and data.groupby('patient_id').role.nunique().max() == 1
    assert len(data) == len(pred) and list(data.ecg_id) == list(pred.ecg_id)
    assert data.label.to_list() == pred.label.to_list()
    official = pd.read_csv(META).set_index('ecg_id')
    assert (data.set_index('ecg_id').fold == official.loc[data.ecg_id, 'strat_fold']).all()
    assert set(evaluation.ecg_id) == set(data.loc[data.role.eq('evaluate'), 'ecg_id'])
    for role, folds in [('train',set(range(1,8))),('select',{8}),('calibrate',{9}),('evaluate',{10})]:
        assert set(data.loc[data.role.eq(role), 'fold']) == folds
    z = np.load(RUN/'embeddings.npz')
    index = {int(k): v for k, v in zip(z['ecg_id'], z['embedding'])}
    assert len(index) == len(features) == len(z['ecg_id'])
    assert z['embedding'].shape == (len(features),768) and np.isfinite(z['embedding']).all()
    frame = pd.concat([data.reset_index(drop=True), pd.DataFrame(
        np.stack([index[int(k)] for k in data.ecg_id]), columns=[f'FM_{i:03d}' for i in range(768)])], axis=1)
    thresholds = read(RUN/'selection_calibration_thresholds.json')
    selected = pd.read_csv(RUN/'candidate_selection.csv')
    metrics = pd.read_csv(RUN/'metrics.csv')
    frozen = read(RUN/'models_frozen_before_evaluation.json')
    assert sha(RUN/'selection_calibration_thresholds.json') == frozen['threshold_SHA256']
    checks = []
    for arm in ARMS:
        path = RUN/'models'/f'{arm}.joblib'
        assert sha(path) == frozen['model_SHA256'][path.name]
        pack = joblib.load(path)
        x = frame[pack['feature_columns']].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float)
        p = pack['classifier'].predict_proba(x)[:,1]
        logits = np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
        q = pack['calibrator'].predict_proba(logits)[:,1]
        assert np.allclose(p, pred[arm+'__raw'], atol=1e-12, rtol=1e-10)
        assert np.allclose(q, pred[arm], atol=1e-12, rtol=1e-10)
        selection = selected[selected.arm.eq(arm)]
        assert len(selection) == 3
        best = selection.iloc[int(np.argmax(selection.selection_AP.to_numpy()))]
        assert best.C == thresholds[arm]['C']
        cm = frame.role.eq('calibrate').to_numpy()
        fpr,tpr,ts = roc_curve(frame.loc[cm,'label'],q[cm])
        valid = np.flatnonzero((tpr>=.95)&np.isfinite(ts))
        idx = min(valid,key=lambda k:(fpr[k],-ts[k]))
        assert abs(float(ts[idx])-thresholds[arm]['threshold']) < 1e-12
        assert pack['threshold'] == thresholds[arm]['threshold']
        em = frame.role.eq('evaluate').to_numpy(); yy=frame.loc[em,'label'].to_numpy(); pp=q[em]
        m = metrics[metrics.arm.eq(arm)&metrics.policy.eq('calibrated')].iloc[0]
        for name,value in [('AUROC',roc_auc_score(yy,pp)),('AP',average_precision_score(yy,pp)),('Brier',brier_score_loss(yy,pp))]:
            assert abs(value-float(m[name])) < 1e-12
        checks.append(dict(arm=arm,saved_classifier_calibrator_replay='PASS',fold9_threshold_replay='PASS',evaluation_metrics='PASS'))
    intervals = pd.read_csv(RUN/'paired_patient_intervals.csv')
    assert len(intervals)==9 and intervals.replicates.eq(2000).all()
    assert intervals.patients.eq(evaluation.patient_id.nunique()).all()
    for name in ['Figure1_ROC_PR','Figure2_paired_increment','Figure3_calibration']:
        for ext in ['png','svg','pdf']:
            assert (RUN/'figures'/f'{name}.{ext}').stat().st_size > 1000
    assert sha(RUN/'fit_ledger.json') == ledger_hash
    receipt = dict(state='numerical_artifact_audit_PASS_visual_export_review_pending',time=now(),classifier_fits=28,
        fits_added_by_audit=0,source_freeze='PASS',records=len(data),patients=int(data.patient_id.nunique()),
        patient_role_overlap=0,native_fs=500,provider_hash_audit_rows=len(raw),model_checks=checks,
        bootstrap_scope='fixed-model patient-cluster uncertainty only',clinical_validation=False,
        independent_external_validation=False,stroke_prediction=False,expert_review='pending',hospital_branch='waiting')
    write(RUN/'independent_completion_audit.json',receipt)
    print(__import__('json').dumps(receipt,ensure_ascii=False))

if __name__=='__main__':main()
