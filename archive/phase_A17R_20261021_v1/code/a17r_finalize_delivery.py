"""Close the bounded phase only after numeric replay and visual review, not clinical admission."""
import sys,shutil
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from a17r_common import *

def main():
    check_freeze()
    out=BASE/'runs'/RUN.name
    numerical=read(RUN/'numeric_completion_receipt.json')
    audit=read(RUN/'independent_completion_audit.json')
    visual=read(out/'visual_export_review.json')
    assert numerical['fits']==28 and numerical['new_P_epochs']==0
    assert audit['state']=='numerical_artifact_audit_PASS_visual_export_review_pending'
    assert audit['fits_added_by_audit']==0 and len(audit['model_checks'])==7
    assert visual['recommended_increment_figure_viewed'] and visual['PNG_labels_and_captions_readable']
    assert not (RUN/'STOP_NEW_WORK').exists()
    ledger=read(RUN/'fit_ledger.json')
    assert len(ledger)==28 and all(x['status']=='complete' for x in ledger)
    la=read(out/'left_atrial_source_audit/weights_independent_verification.json')
    assert la['verified_files']==15 and la['verified_bytes']==1574934040
    assert la['state']=='independent_size_SHA256_HDF5_signature_PASS'
    for name in ['a17r_independent_completion_audit.py','a17r_readability_export.py','a17r_finalize_delivery.py']:
        shutil.copy2(Path(__file__).parent/name,out/'code'/name)
    status('bounded_phase_complete',state='completed_with_clinical_gaps_preserved',fits_completed=28,max_fits=30,
        new_P_epochs=0,patient_role_overlap=0,independent_replay='PASS',expert_review='pending',
        hospital_branch='waiting',stroke_prediction=False,NC_submission_ready=False)
    files=list(p for p in RUN.iterdir() if p.is_file())
    for folder in ['models','figures','publication_figures','code']:
        files.extend(p for p in (RUN/folder).iterdir() if p.is_file())
    manifest=[]
    for source in files:
        relative=source.relative_to(RUN);target=out/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(source,target)
        h=sha(source)
        assert sha(target)==h,f'Delivery changed: {relative}'
        manifest.append(dict(path=str(relative),bytes=target.stat().st_size,sha256=h,native_source=str(source)))
    for relative in ['A17R阶段验收与下一步.txt','visual_export_review.json',
                     'code/a17r_independent_completion_audit.py','code/a17r_readability_export.py','code/a17r_finalize_delivery.py']:
        p=out/relative
        manifest.append(dict(path=relative,bytes=p.stat().st_size,sha256=sha(p),native_source=None))
    write(out/'delivery_manifest.json',dict(state='native_to_D_SHA256_PASS',files=manifest))
    receipt=dict(state='bounded_A17R_complete_not_clinically_validated',time=now(),records=numerical['records'],
        evaluation_records=numerical['test_records'],evaluation_patients=numerical['test_patients'],
        classifier_and_calibration_fits=28,max_fits=30,reserve_fits_not_used=2,new_P_network_epochs=0,
        source_freeze='PASS',patient_role_overlap=0,saved_model_threshold_metric_replay='7/7 PASS',
        original_png_figures_reviewed=3,readability_only_increment_export_reviewed=True,
        mirrored_files=len(manifest),delivery_manifest_SHA256=sha(out/'delivery_manifest.json'),
        left_atrial_weights='15/15 integrity verified; inference pending; not in seven-arm comparison',
        primary_increment_checks=numerical['primary_increment_checks'],
        public_branch='A17R bounded development complete; independent external validation not achieved',
        hospital_branch='waiting',expert_review='pending',clinical_admission=False,stroke_prediction=False,
        NC_submission_ready=False,scope='concurrent AF recognition on previously viewed/source-exposed PTB-XL')
    write(out/'completion_receipt.json',receipt)
    write(RUN/'completion_receipt.json',receipt)
    shutil.copy2(out/'status.json',BASE/'PHASE_A17R_STATUS.json')
    print(__import__('json').dumps(receipt,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
