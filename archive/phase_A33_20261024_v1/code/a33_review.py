"""Independent, zero-fit A33 contract review. Never call an old phase main/check."""
import importlib.util
import math

import a33_audit as a

POLICIES = (
    'R0_A32_L0', 'R1_A32_W1', 'F00_old_clipped',
    'F01_old_unclipped', 'F10_wide_clipped', 'F11_wide_unclipped',
)
FACTUAL = {'R0_A32_L0': 'L0_legacy_decoder', 'R1_A32_W1': 'W1_window_aware'}
FACTORS = tuple(p for p in POLICIES if p not in FACTUAL)
_spec = importlib.util.spec_from_file_location(
    'a33_independent_augmenting_path_review', a.q.old.A29 / 'code/review.py')
v = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(v)


def _index(rows):
    out = {(int(r['patient']), r['lead']): r for r in rows}
    assert len(out) == len(rows), 'Duplicate patient/lead record'
    return out


def _numeric_equal(stored, expected):
    assert v.same(stored, expected), (stored, expected)


def _strict(ref, ps, qs):
    """Minimax score via independent augmenting paths, not production Hungarian."""
    edges = [[v.allowed(r, p, qs) for p in ps] for r in ref]
    cardinality = v.cardinality(edges)
    if cardinality < len(ref):
        return float('inf'), len(ref) - cardinality
    values = sorted({v.dist(r, p) for r in ref for p in ps if v.allowed(r, p, qs)})
    if not ref:
        return 0., 0
    lo, hi = 0, len(values) - 1
    while lo < hi:
        mid = (lo + hi) // 2
        count = v.cardinality([
            [edges[i][j] and v.dist(r, p) <= values[mid] for j, p in enumerate(ps)]
            for i, r in enumerate(ref)
        ])
        if count == len(ref):
            hi = mid
        else:
            lo = mid + 1
    return values[lo], 0


def _calibration(rows, cal, counts):
    expected = {int(r['patient']) for r in rows if r['ref']}
    ledger = cal['ledger']
    assert len({int(x['patient']) for x in ledger}) == len(ledger)
    assert {int(x['patient']) for x in ledger} == expected
    assert cal['n_reference_positive'] == len(expected)
    strict_values, component_values = [], []
    for entry in ledger:
        rr = [r for r in rows if int(r['patient']) == int(entry['patient']) and r['ref']]
        assert len(entry['rows']) == len(rr)
        stricts, components, missing = [], [], 0
        assert entry['reference'] == sum(len(r['ref']) for r in rr)
        for r in rr:
            ref, ps, qs = r['ref'], r['selected'], r['qrs']
            score, absent = _strict(ref, ps, qs)
            stored = next(x for x in entry['rows'] if x['lead'] == r['lead'])
            _numeric_equal(stored['strict'], score)
            assert stored['missing'] == absent
            pairs = stored['pairs']
            assert len(pairs) == len(ref) - absent
            assert len({i for i, j in pairs}) == len(pairs)
            assert len({j for i, j in pairs}) == len(pairs)
            assert all(v.allowed(ref[i], ps[j], qs) for i, j in pairs)
            residuals = [v.dist(ref[i], ps[j]) for i, j in pairs]
            component = max(residuals, default=float('inf'))
            _numeric_equal(stored['component'], component)
            assert len(stored['errors']) == len(residuals)
            assert all(abs(x - y) <= 1e-12 for x, y in zip(stored['errors'], residuals))
            stricts.append(score)
            components.append(component)
            missing += absent
            counts['calibration_records'] += 1
        strict, component = max(stricts), max(components)
        _numeric_equal(entry['strict'], strict)
        _numeric_equal(entry['component'], component)
        assert entry['missing'] == missing
        strict_values.append(strict)
        component_values.append(component)
    n = len(expected)
    k = math.ceil((n + 1) * .95)
    assert cal['k'] == k
    for key, values in [('strict', strict_values), ('component', component_values)]:
        quantile = sorted(values)[k - 1] if n and k <= n else float('inf')
        _numeric_equal(cal[key + '_q'], quantile)
    assert cal['used_q'] == cal['strict_q']
    assert cal['all_patients_retained'] and not cal['used_legacy_q']
    assert cal['missing_patient_count'] == sum(x['missing'] > 0 for x in ledger)
    assert cal['infinite_strict_patient_count'] == sum(not math.isfinite(x) for x in strict_values)
    assert cal['infinite_component_patient_count'] == sum(not math.isfinite(x) for x in component_values)


def _outer(rows, out, counts):
    assert len(rows) == len(out['records']) == 80
    expected_widths = []
    for r, stored in zip(rows, out['records']):
        ref, ps, qs = r['ref'], r['selected'], r['qrs']
        assert r['patient'] == stored['patient'] and r['lead'] == stored['lead']
        assert r['record'] == stored['record'] and len(ref) == stored['reference']
        assert ps == stored['selected']
        assert len({p['wave_R'] for p in ps}) == len(ps)
        assert len({p['uid'] for p in ps}) == len(ps)
        assert all(p['wave_R'] is not None and p['scale'] >= 5 for p in ps)
        assert all(p['confidence'] >= out['choice']['threshold'] for p in ps)
        strict, missing = _strict(ref, ps, qs)
        _numeric_equal(stored['identity_score']['strict'], strict)
        assert stored['identity_score']['missing'] == missing
        expanded = v.expand(ps, out['calibration']['used_q'])
        reported = [p for p in expanded if max(
            p['box'][1] - p['box'][0], p['box'][3] - p['box'][2]) <= 100]
        component = v.expand(ps, out['calibration']['component_q'])
        component_reported = [p for p in component if max(
            p['box'][1] - p['box'][0], p['box'][3] - p['box'][2]) <= 100]
        assert v.boundary_count(ref, expanded, qs) == stored['strict_unique_covered']
        assert v.boundary_count(ref, reported, qs) == stored['strict_unique_reported']
        assert v.boundary_count(ref, component, qs) == stored['component_unique_covered_DIAGNOSTIC']
        assert v.boundary_count(ref, component_reported, qs) == stored['component_unique_reported_DIAGNOSTIC']
        for name, values in [('raw', r['raw']), ('selected', ps), ('reported', reported),
                             ('component_reported_DIAGNOSTIC', component_reported)]:
            tab = stored['tables'][name]
            assert tab['reference'] == len(ref) and tab['predicted'] == len(values)
            assert tab['matched'] == v.point_count(ref, values, qs)
            assert tab['identity_matched'] == v.point_count(ref, values, qs, True)
        for p in reported:
            expected_widths.append(dict(patient=r['patient'], lead=r['lead'],
                onset_ms=2 * (p['box'][1] - p['box'][0]),
                offset_ms=2 * (p['box'][3] - p['box'][2])))
        counts['outer_records'] += 1
    assert expected_widths == out['widths']
    patient_rows = out['patients']
    expected = {int(r['patient']) for r in rows if r['ref']}
    assert {int(p['patient']) for p in patient_rows} == expected
    assert len(patient_rows) == len(expected)
    for p in patient_rows:
        rr = [r for r in out['records'] if r['patient'] == p['patient']]
        assert p['reference'] == sum(r['reference'] for r in rr)
        assert p['missing'] == sum(r['identity_score']['missing'] for r in rr)
        assert p['strict_all'] == all(r['strict_unique_covered'] == r['reference'] for r in rr)
        assert p['strict_reported_all'] == all(r['strict_unique_reported'] == r['reference'] for r in rr)
        assert p['component_all_DIAGNOSTIC'] == all(r['component_unique_covered_DIAGNOSTIC'] == r['reference'] for r in rr)
        assert p['component_reported_all_DIAGNOSTIC'] == all(r['component_unique_reported_DIAGNOSTIC'] == r['reference'] for r in rr)
        counts['patient_entries'] += 1


def _factual(source_cache, source_out, data, out):
    for role in ['design', 'calibration', 'outer']:
        before, after = _index(source_cache[role]), _index(data[role])
        assert before.keys() == after.keys()
        for key, r in after.items():
            for field in ['selected', 'ref', 'qrs']:
                assert r[field] == before[key][field], 'Factual candidate/reference replay mismatch'
    assert out['calibration'] == source_out['calibration']
    assert out['patients'] == source_out['patients']
    assert out['widths'] == source_out['widths']
    assert len(out['records']) == len(source_out['records'])
    for before, after in zip(source_out['records'], out['records']):
        for field in ['patient', 'lead', 'reference', 'tables', 'identity_score',
                      'strict_unique_covered', 'strict_unique_reported', 'selected']:
            assert before[field] == after[field], 'Factual metric replay mismatch'


def _oracle(diag, caches, counts):
    expected = {(policy, role, int(r['patient']), r['lead'])
                for policy in FACTORS for role in ['calibration', 'outer']
                for r in caches[policy][role]}
    keys = {(r['policy'], r['role'], int(r['patient']), r['lead']) for r in diag}
    assert len(keys) == len(diag) and keys == expected
    record_maps = {(policy, role): _index(caches[policy][role])
                   for policy in FACTORS for role in ['calibration', 'outer']}
    for row in diag:
        r = record_maps[(row['policy'], row['role'])][(int(row['patient']), row['lead'])]
        ref = r['ref']
        nr = len(ref)
        assert row['reference'] == nr or row['reference'] == ref
        for extra in row['extra_ordered_R_candidates']:
            p = extra['candidate']
            assert p['kind'] == 'ordered' and not 1000 < p['source_R'] < 4200
        assert all(1000 < p['source_R'] < 4200 for p in r['raw'] if p['kind'] == 'ordered')
        assert set(row['stages']) == {'raw', 'threshold', 'nms', 'selected'}
        for stage, kinds in row['stages'].items():
            assert set(kinds) == {'point_geometry', 'point', 'local200', 'point_local200'}
            for kind, entry in kinds.items():
                slots, edges = entry['slots'], entry['edges']
                assert len(slots) == len(set(slots)) and all(s is not None for s in slots)
                assert len(edges) == nr and all(len(e) == len(slots) for e in edges)
                assert all(isinstance(x, bool) for e in edges for x in e)
                assert entry['matched'] == v.cardinality(edges)
                assert 0 <= entry['matched'] <= min(nr, len(slots))
                if stage == 'selected':
                    assert set(slots) == {p['wave_R'] for p in r['selected']}
                    if kind in ['point', 'point_geometry']:
                        assert entry['matched'] == v.point_count(
                            ref, r['selected'], r['qrs'], kind == 'point')
                    for i, rr in enumerate(ref):
                        for j, slot in enumerate(slots):
                            candidates = [p for p in r['selected'] if p['wave_R'] == slot]
                            def allowed(p):
                                identity = v.allowed(rr, p, r['qrs'])
                                overlap = max(0, min(rr[2], p['offset']) - max(rr[0], p['onset']))
                                union = max(max(rr[2], p['offset']) - min(rr[0], p['onset']), 1)
                                point = overlap / union >= .5
                                local = max(abs(rr[0] - p['onset']), abs(rr[2] - p['offset'])) <= 50
                                if kind == 'point_geometry':
                                    return point
                                return identity and (point if kind == 'point' else local if kind == 'local200' else point and local)
                            assert edges[i][j] == any(allowed(p) for p in candidates)
                counts['oracle_graphs'] += 1
            joint = kinds['point_local200']['edges']
            # Slot order is stored independently by kind; compare by slot identity.
            for i, edge_row in enumerate(joint):
                for j, flag in enumerate(edge_row):
                    if not flag:
                        continue
                    slot = kinds['point_local200']['slots'][j]
                    assert all(slot in kinds[k]['slots'] and
                               kinds[k]['edges'][i][kinds[k]['slots'].index(slot)]
                               for k in ['point', 'local200'])
            assert kinds['point_local200']['matched'] <= min(
                kinds['point']['matched'], kinds['local200']['matched'])
            assert kinds['point']['matched'] <= kinds['point_geometry']['matched']
            for i, edges in enumerate(kinds['point']['edges']):
                for j, flag in enumerate(edges):
                    if flag:
                        slot = kinds['point']['slots'][j]
                        assert slot in kinds['point_geometry']['slots']
                        assert kinds['point_geometry']['edges'][i][kinds['point_geometry']['slots'].index(slot)]
        for kind in ['point_geometry', 'point', 'local200', 'point_local200']:
            counts_by_stage = [row['stages'][stage][kind]['matched']
                               for stage in ['raw', 'threshold', 'nms', 'selected']]
            assert all(x >= y for x, y in zip(counts_by_stage, counts_by_stage[1:])), 'Oracle stage upper bounds increased after filtering'
        counts['oracle_records'] += 1


def main():
    a.check()
    jobs = list(a.jobs())
    assert len(jobs) == 20 and len({j['index'] for j in jobs}) == 20
    roles = a.read(a.q.RUN / 'patient_roles.json')
    for role in roles:
        sets = [set(role[k]) for k in ['train', 'design', 'calibration', 'outer']]
        assert list(map(len, sets)) == [124, 12, 24, 40]
        assert all(not x & y for i, x in enumerate(sets) for y in sets[:i])
    assert sorted(p for r in roles for p in r['outer']) == list(range(200))
    counts = dict(cached_lead_records=0, factual_replays=0, factorial_evaluations=0,
                  calibration_records=0, outer_records=0, patient_entries=0,
                  oracle_records=0, oracle_graphs=0)
    outer_sets, positive_sets = {}, {}
    for job in jobs:
        a.check()
        path = a.path(job)
        done = a.read(path / 'done.json')
        assert done['cached_lead_records'] == 152 and done['factual_replays'] == 2
        assert done['factorial_evaluations'] == 4
        assert done['new_fits'] == 0 and done['model_forwards'] == 0
        cache = a.ungz(path / 'candidate_cache.json.gz')
        assert cache['job'] == job and set(cache['policies']) == set(POLICIES)
        caches = cache['policies']
        original = a.ungz(a.q.jobpath(job) / 'prediction_cache.json.gz')
        for policy in POLICIES:
            a.check()
            data = caches[policy]
            assert set(data) == {'design', 'calibration', 'outer'}
            for role in ['design', 'calibration', 'outer']:
                rows = data[role]
                indexed = _index(rows)
                pats = set(roles[job['fold']][role])
                assert set(indexed) == {(int(p), lead) for p in pats for lead in ['ii', 'v1']}
                base = _index(caches['R0_A32_L0'][role])
                for key, r in indexed.items():
                    assert r['ref'] == base[key]['ref'] and r['qrs'] == base[key]['qrs']
            out = a.ungz(path / ('result_' + policy + '.json.gz'))
            assert out['job'] == job and out['policy'] == policy
            _calibration(data['calibration'], out['calibration'], counts)
            _outer(data['outer'], out, counts)
            key = (policy, job['arm'], job['seed'])
            so = outer_sets.setdefault(key, [])
            so.extend(int(p) for p in roles[job['fold']]['outer'])
            positive_sets.setdefault(key, []).extend(int(p['patient']) for p in out['patients'])
            if policy in FACTUAL:
                oldpolicy = FACTUAL[policy]
                oldout = a.ungz(a.q.jobpath(job) / (oldpolicy + '.json.gz'))
                _factual(original['policies'][oldpolicy], oldout, data, out)
                counts['factual_replays'] += 1
            else:
                counts['factorial_evaluations'] += 1
        diag = a.ungz(path / 'oracle_diagnostics.json.gz')
        _oracle(diag, caches, counts)
        counts['cached_lead_records'] += 152
    assert len(outer_sets) == len(POLICIES) * 2 * 2
    assert all(sorted(pats) == list(range(200)) for pats in outer_sets.values())
    assert all(len(pats) == len(set(pats)) == 176 for pats in positive_sets.values())
    assert counts['cached_lead_records'] == 3040
    assert counts['factual_replays'] == 40 and counts['factorial_evaluations'] == 80
    assert counts['outer_records'] == 9600
    assert counts['oracle_records'] == 20 * 4 * 128
    assert counts['oracle_graphs'] == counts['oracle_records'] * 16
    a.verify()
    a.check()
    a.put(a.RUN / 'post_review/contract_review.json', dict(
        time=a.q.now(), state='PASS', independent_matcher='augmenting_paths',
        patient_roles_disjoint=True, all_200_outer_patients_per_seed_arm_policy=True,
        all_176_reference_positive_patients_per_seed_arm_policy=True,
        all_calibration_reference_patients_retained=True,
        strict_missing_infinity=True, exact_finite_sample_quantile=True,
        unique_same_R_selected=True, oracle_R_capacity_one=True,
        factorial_filtering_upper_bounds_monotone=True,
        source_hashes_unchanged=True, factual_selected_and_metric_replay=True,
        new_review_fits=0, model_forwards=0, new_figures=0,
        effectiveness_PASS_not_implied=True, NC_submission_ready=False,
        automatic_retraining=False, **counts))
