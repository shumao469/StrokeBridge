"""Independent zero-fit review of A34's necessary-condition admission audit.

Reconstruct identity graphs from raw cached candidates, not from production
matching results. A pass verifies the audit contract, not method effectiveness
or permission to fit. Never call a previous phase's main/check functions.
"""
import importlib.util
import math

import a34_preflight as a

STAGES = (
    'raw_identity', 'selected_identity', 'threshold_identity',
    'ideal_confidence_identity',
)
_spec = importlib.util.spec_from_file_location(
    'a34_independent_identity_capacity', a.q.old.A29 / 'code/review.py')
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def _index(rows):
    result = {(int(r['patient']), r['lead']): r for r in rows}
    assert len(result) == len(rows), 'Duplicate patient/lead rows'
    return result


def _cap(p):
    if p['kind'] == 'continuous':
        return 1.
    assert p['kind'] == 'ordered'
    if 'non_null_mass' in p:
        result = float(p['non_null_mass'])
    else:
        assert 'legacy_score' in p, 'Cached ordered PN upper bound is required'
        result = float(p['legacy_score'])
    assert math.isfinite(result) and 0 <= result <= 1
    assert float(p['confidence']) <= result + 1e-12
    return result


def _pool(row, stage, cutoff):
    if stage == 'raw_identity':
        return row['raw']
    if stage == 'selected_identity':
        return row['selected']
    if stage == 'threshold_identity':
        return [p for p in row['raw'] if p['confidence'] >= cutoff]
    assert stage == 'ideal_confidence_identity'
    return [p for p in row['raw'] if _cap(p) >= cutoff]


def _graph(ref, candidates, qrs):
    slots = sorted({p['wave_R'] for p in candidates if p['wave_R'] is not None})
    edges = [[any(p['wave_R'] == slot and v.allowed(r, p, qrs)
                  for p in candidates) for slot in slots] for r in ref]
    return slots, edges, v.cardinality(edges)


def _patient_rows(rows, per_record):
    by_patient = {}
    for r in rows:
        by_patient.setdefault(int(r['patient']), []).append(r)
    result = []
    for patient, rr in sorted(by_patient.items()):
        nref = sum(len(r['ref']) for r in rr)
        if not nref:
            continue
        matched = sum(per_record[(patient, r['lead'])] for r in rr)
        assert 0 <= matched <= nref
        names = {r['record'] for r in rr}
        assert len(names) == 1
        result.append(dict(patient=patient, reference=nref, matched=matched,
                           missing=nref - matched, record=next(iter(names))))
    return result


def _assert_patient_rows(stored, expected):
    before = {int(p['patient']): p for p in stored}
    after = {int(p['patient']): p for p in expected}
    assert len(before) == len(stored) and before.keys() == after.keys()
    for patient, wanted in after.items():
        for key in ['patient', 'reference', 'matched', 'missing', 'record']:
            assert before[patient][key] == wanted[key], (patient, key)


def _infinite_count_matches(value, ids):
    # Support either explicit patient IDs or the documented scalar count.
    if isinstance(value, list):
        got = [int(x['patient']) if isinstance(x, dict) else int(x) for x in value]
        assert len(got) == len(set(got)) and set(got) == set(ids)
    else:
        assert value == len(ids)


def main():
    a.check()
    jobs = list(a.jobs())
    assert len(jobs) == 10 and len({j['index'] for j in jobs}) == 10
    assert all(j['arm'] == 'C1_patient_report_aligned' for j in jobs)
    roles = a.read(a.q.RUN / 'patient_roles.json')
    for role in roles:
        groups = [set(role[k]) for k in ['train', 'design', 'calibration', 'outer']]
        assert list(map(len, groups)) == [124, 12, 24, 40]
        assert all(not left & right for i, left in enumerate(groups)
                   for right in groups[:i])
    assert sorted(p for role in roles for p in role['outer']) == list(range(200))
    counts = dict(cache_jobs=0,identity_records=0,identity_graphs=0,
                  calibration_patient_stage_checks=0,outer_patient_stage_checks=0)
    aggregates = {}
    for job in jobs:
        a.check()
        path = a.path(job)
        data = a.ungz(path / 'input_data.json.gz')
        graphs = a.ungz(path / 'identity_graphs.json.gz')
        summary = a.read(path / 'summary.json')
        assert data['job'] == job and summary['job'] == job
        cutoff = float(data['choice']['threshold'])
        assert 0 <= cutoff <= 1 and float(summary['cutoff']) == cutoff
        original = a.ungz(a.q.BASE / 'runs/phase_A33_20261024_v1/jobs' /
                          ('job_%02d' % job['index']) / 'candidate_cache.json.gz')
        original = original['policies']['F01_old_unclipped']
        indexed = {}
        for role in ['calibration','outer']:
            rows = data[role]
            indexed[role] = _index(rows)
            old = _index(original[role])
            pats = set(roles[job['fold']][role])
            assert set(indexed[role]) == {(int(p), lead) for p in pats
                                          for lead in ['ii','v1']}
            assert indexed[role].keys() == old.keys()
            for key, r in indexed[role].items():
                for field in ['ref','qrs','raw','selected','record','li','lead']:
                    assert r[field] == old[key][field], 'Original F01 input changed'
                assert len({p['uid'] for p in r['raw']}) == len(r['raw'])
                assert len({p['wave_R'] for p in r['selected']}) == len(r['selected'])
                assert all(p['wave_R'] is not None for p in r['selected'])
                assert {p['uid'] for p in r['selected']} <= {p['uid'] for p in r['raw']}
        expected_keys = {(role, p, lead) for role, rs in indexed.items()
                         for p, lead in rs}
        got_keys = {(g['role'], int(g['patient']), g['lead']) for g in graphs}
        assert len(graphs) == len(got_keys) == 128 and got_keys == expected_keys
        matches = {role: {stage: {} for stage in STAGES}
                   for role in ['calibration','outer']}
        for g in graphs:
            r = indexed[g['role']][(int(g['patient']), g['lead'])]
            assert g['record'] == r['record'] and g['reference'] == r['ref']
            assert g['qrs'] == r['qrs'] and g['list_candidates'] == r['raw']
            assert set(g['stages']) == set(STAGES)
            for stage in STAGES:
                stored = g['stages'][stage]
                slots, edges, matched = _graph(r['ref'], _pool(r, stage, cutoff),r['qrs'])
                assert stored['slots'] == slots and stored['edges'] == edges
                assert all(isinstance(x,bool) for row in stored['edges'] for x in row)
                assert stored['matched'] == matched
                assert 0 <= matched <= min(len(r['ref']),len(slots))
                matches[g['role']][stage][(int(r['patient']),r['lead'])] = matched
                counts['identity_graphs'] += 1
            raw = g['stages']['raw_identity']['matched']
            selected = g['stages']['selected_identity']['matched']
            threshold = g['stages']['threshold_identity']['matched']
            ideal = g['stages']['ideal_confidence_identity']['matched']
            assert selected <= threshold <= ideal <= raw
            counts['identity_records'] += 1
        independent = {'calibration':{},'outer':{}}
        for stage in STAGES:
            calrows = _patient_rows(data['calibration'],matches['calibration'][stage])
            cal = summary['calibration'][stage]
            _assert_patient_rows(cal['patients'],calrows)
            n = len(calrows); k = math.ceil((n+1)*.95)
            inf_ids = [p['patient'] for p in calrows if p['missing']>0]
            unavoidable_infinite = k>n or n-len(inf_ids)<k
            assert 22 <= n <= 24 and cal['n']==n and cal['k']==k
            _infinite_count_matches(cal['infinite_patients'],inf_ids)
            assert cal['quantile_infinite'] is unavoidable_infinite
            outerrows = _patient_rows(data['outer'],matches['outer'][stage])
            out = summary['outer'][stage]
            _assert_patient_rows(out['patients'],outerrows)
            assert out['positive_patients']==len(outerrows)
            assert out['covered_patients']==sum(p['missing']==0 for p in outerrows)
            independent['calibration'][stage] = dict(quantile_infinite=unavoidable_infinite)
            independent['outer'][stage] = outerrows
            counts['calibration_patient_stage_checks'] += n
            counts['outer_patient_stage_checks'] += len(outerrows)
        group = aggregates.setdefault(job['seed'],dict(outer_ids=[],positive_ids=[],
            blocked_ids=[],joint_possible_ids=[]))
        group['outer_ids'].extend(int(p) for p in roles[job['fold']]['outer'])
        ideal_outer = independent['outer']['ideal_confidence_identity']
        group['positive_ids'].extend(p['patient'] for p in ideal_outer)
        blocked = independent['calibration']['ideal_confidence_identity']['quantile_infinite']
        if blocked:
            group['blocked_ids'].extend(p['patient'] for p in ideal_outer)
        else:
            group['joint_possible_ids'].extend(p['patient'] for p in ideal_outer
                                                if p['missing']==0)
        counts['cache_jobs'] += 1
    assert counts['cache_jobs']==10 and counts['identity_records']==1280
    assert counts['identity_graphs']==5120
    feasibility = a.read(a.RUN / 'analysis/feasibility.json')
    before = {int(s['seed']):s for s in feasibility['seeds']}
    assert set(before)==set(aggregates) and len(before)==2
    admissible = True
    details=[]
    for seed, entry in sorted(aggregates.items()):
        assert sorted(entry['outer_ids'])==list(range(200))
        assert len(entry['positive_ids'])==len(set(entry['positive_ids']))==176
        assert len(entry['blocked_ids'])==len(set(entry['blocked_ids']))
        assert len(entry['joint_possible_ids'])==len(set(entry['joint_possible_ids']))
        blocked=len(entry['blocked_ids']);cal_ceiling=(176-blocked)/176
        joint_ceiling=len(entry['joint_possible_ids'])/176
        stored=before[seed]
        assert stored['positive_patients']==176
        assert stored['calibration_blocked_outer_patients']==blocked
        assert abs(stored['calibration_induced_patient_ceiling']-cal_ceiling)<=1e-12
        assert abs(stored['combined_ideal_patient_ceiling']-joint_ceiling)<=1e-12
        admissible = admissible and joint_ceiling>=.95
        details.append(dict(seed=seed,reference_positive_patients=176,
            calibration_blocked_outer_patients=blocked,
            calibration_induced_patient_ceiling=cal_ceiling,
            combined_ideal_patient_ceiling=joint_ceiling))
    assert feasibility['training_admissible'] is admissible
    assert feasibility['actual_fits']==0 and feasibility['automatic_training'] is False
    a.verify();a.check()
    a.put(a.RUN / 'post_review/contract_review.json',dict(
        time=a.q.now(),state='PASS',independent_matcher='augmenting_paths',
        original_F01_inputs_unchanged=True,patient_roles_disjoint=True,
        all_200_outer_patients_per_seed=True,all_176_reference_positive_per_seed=True,
        all_calibration_reference_patients_retained=True,oracle_R_capacity_one=True,
        ordered_confidence_upper_is_cached_PN=True,continuous_confidence_upper_is_one=True,
        exact_infinite_quantile_order_statistic=True,
        calibration_blocked_outer_ceiling_recomputed=True,
        necessary_condition_pass=bool(admissible),scientific_admission=False,
        effectiveness_PASS_not_implied=True,NC_submission_ready=False,
        actual_fits=0,model_forwards=0,automatic_training=False,
        source_hashes_unchanged=True,seeds=details,**counts))
