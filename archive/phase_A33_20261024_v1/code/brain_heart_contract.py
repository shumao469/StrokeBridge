"""Zero-fit brain-heart outcome data contract, NOT a clinical risk model.

`eligible` means this one row satisfies this schema and time-order contract. It
does not admit a source, authenticate a patient, establish sample adequacy,
validate a P measurement, or authorize fitting/prediction. There are deliberately
no fit/predict functions. All examples and tests use synthetic records only.

All times, including death_date, must be timezone-aware ISO datetimes. Date-only
source fields need an explicitly reviewed temporal-resolution policy, not an
automatic invented time. available_at is clinical information availability, not
the date a retrospective analysis script happened to run.
"""
from copy import deepcopy
from datetime import datetime
import math

CLINICALLY_VALIDATED = False
FITTING_ALLOWED = False
REAL_RISK_OUTPUT_ALLOWED = False
TASKS = ("future_first_AF", "future_first_stroke", "future_recurrent_stroke")
MODEL_GROUPS = {
    "clinical": ("clinical",),
    "clinical_ecg": ("clinical", "ecg"),
    "clinical_brain": ("clinical", "brain"),
    "clinical_ecg_brain": ("clinical", "ecg", "brain"),
}
MISSING_STATES = ("valid", "rejected", "unreadable", "not_acquired", "unknown")
OUTCOME_STATES = ("adjudicated", "censored", "unknown")
OUTCOME_TYPES = ("target_event", "event_free", "competing_event", "unknown")
DATA_CONTRACT_ONLY = (
    "Schema eligibility is not source admission, independent validation, "
    "clinical usability, or permission to fit/predict."
)


def _text(v):
    return isinstance(v, str) and bool(v.strip())


def _unknown(v):
    return v is None or (isinstance(v, str) and v == "unknown")


def _finite(v):
    if isinstance(v, float):
        return math.isfinite(v)
    if isinstance(v, (list, tuple)):
        return all(_finite(a) for a in v)
    if isinstance(v, dict):
        return all(_finite(a) for a in v.values())
    return True


def validate(row):
    """Return JSON-serializable {eligible, errors, warnings}; never infer labels.

    row.features[group] is a list of feature entries with name/status/value and
    acquired_at/available_at. Unknown/not-acquired entries may lack acquired_at,
    but their missingness must itself be known by index_time. Non-valid features
    must have value=None, not a fabricated zero or biological absence.

    An outcome needs explicit ascertainment, assessor, source and last confirmed
    follow-up. No target event plus confirmed event-free follow-up is censoring,
    not an unqualified permanent negative. A death before any target event must
    be explicitly represented as competing_event, not an ordinary censor.
    """
    errors = []
    warnings = [DATA_CONTRACT_ONLY]
    if not isinstance(row, dict):
        return dict(eligible=False, errors=["row: expected object"], warnings=warnings)

    def time(value, path, required=True):
        if value is None or value == "":
            if required:
                errors.append(path + ": required timezone-aware ISO datetime")
            return None
        try:
            if not isinstance(value, str) or "T" not in value:
                raise ValueError()
            out = datetime.fromisoformat(value.replace("Z", "+00:00"))
            if out.tzinfo is None or out.utcoffset() is None:
                raise ValueError()
            return out
        except (ValueError, TypeError, OverflowError):
            errors.append(path + ": invalid or timezone-naive datetime")
            return None

    for key in ("patient_id", "site"):
        if not _text(row.get(key)):
            errors.append(key + ": nonempty research identifier required")
    index = time(row.get("index_time"), "index_time")
    task = row.get("task")
    if task not in TASKS:
        errors.append("task: choose one future first/recurrent endpoint; concurrent AF is not this task")
    for key in ("prior_AF", "prior_stroke"):
        if key not in row:
            errors.append(key + ": explicitly supply boolean or unknown")
        elif type(row[key]) is not bool and not _unknown(row[key]):
            errors.append(key + ": expected boolean or unknown, not a numeric code")
    if task == "future_first_AF" and row.get("prior_AF") is not False:
        errors.append("future_first_AF: requires documented prior_AF=False")
    if task == "future_first_stroke" and row.get("prior_stroke") is not False:
        errors.append("future_first_stroke: requires documented prior_stroke=False")
    if task == "future_recurrent_stroke" and row.get("prior_stroke") is not True:
        errors.append("future_recurrent_stroke: requires documented prior_stroke=True")

    groups = row.get("features")
    valid_count = 0
    rejected_p = False
    claimed_no_p = row.get("biological_noP") is True
    names = {g: set() for g in ("clinical", "ecg", "brain")}
    if not isinstance(groups, dict):
        errors.append("features: expected clinical/ecg/brain object")
        groups = {}
    if set(groups) - set(names):
        errors.append("features: unrecognized group; only clinical/ecg/brain enter this contract")
    for group in names:
        entries = groups.get(group, [])
        if not isinstance(entries, list):
            errors.append("features." + group + ": expected list")
            continue
        if not entries:
            warnings.append("features." + group + ": absent; models requiring this group are not ready")
        for i, feature in enumerate(entries):
            path = "features.%s[%d]" % (group, i)
            if not isinstance(feature, dict):
                errors.append(path + ": expected feature object")
                continue
            name = feature.get("name")
            if not _text(name):
                errors.append(path + ".name: required")
                name = ""
            if name in names[group]:
                errors.append(path + ".name: duplicate feature name within group")
            names[group].add(name)
            status = feature.get("status")
            if status not in MISSING_STATES:
                errors.append(path + ".status: invalid missingness state")
            value = feature.get("value")
            if status == "valid":
                if value is None or value == "" or value == [] or value == {}:
                    errors.append(path + ".value: valid feature needs an observed value")
                elif not _finite(value):
                    errors.append(path + ".value: nonfinite value is not valid")
                else:
                    valid_count += 1
            elif value is not None:
                errors.append(path + ".value: non-valid measurement must be None, never imputed zero/absence")
            avail = time(feature.get("available_at"), path + ".available_at")
            acquired = time(feature.get("acquired_at"), path + ".acquired_at",
                            required=status in ("valid", "rejected", "unreadable"))
            if acquired and avail and acquired > avail:
                errors.append(path + ": acquired_at after available_at")
            if index and avail and avail > index:
                errors.append(path + ": feature or missingness unavailable at index (leakage)")
            if index and acquired and acquired > index:
                errors.append(path + ": acquired after index (leakage)")
            lname = name.lower()
            is_p = lname.startswith("p_") or "pwave" in lname or "p_wave" in lname or "ptfv1" in lname
            if group == "ecg" and is_p:
                rejected_p |= status in ("rejected", "unreadable", "unknown", "not_acquired")
            if group == "ecg" and lname == "p_candidate_rejected" and value is True:
                rejected_p = True
            claimed_no_p |= feature.get("biological_noP") is True
            claimed_no_p |= lname in ("biological_nop", "biological_no_p") and value is True
    if not valid_count:
        errors.append("features: no observed valid baseline predictor")
    for optional in ("age", "sex"):
        if optional not in names["clinical"]:
            warnings.append("clinical.%s: optional but missing; predeclare comparable clinical baseline" % optional)
    if rejected_p and claimed_no_p:
        errors.append("P_candidate_rejected/unknown is not biological_noP truth")
    if claimed_no_p:
        reference = row.get("noP_reference", {})
        if not isinstance(reference, dict) or reference.get("status") != "adjudicated" or not _text(reference.get("source")):
            errors.append("biological_noP: independent adjudicated no-P reference/source required")

    outcome = row.get("outcome")
    if not isinstance(outcome, dict):
        errors.append("outcome: explicit dated ascertainment required; absent event is not negative")
        return dict(eligible=not errors, errors=errors, warnings=warnings)
    if outcome.get("target") != task:
        errors.append("outcome.target: must match the declared incident/recurrent task")
    status = outcome.get("status")
    if status not in OUTCOME_STATES:
        errors.append("outcome.status: choose adjudicated/censored/unknown")
    elif status == "unknown":
        errors.append("outcome.status: unknown ascertainment cannot enter event-risk evaluation")
    for key in ("assessor", "source"):
        if not _text(outcome.get(key)):
            errors.append("outcome." + key + ": required ascertainment provenance")
    kind = outcome.get("event_type")
    if kind not in OUTCOME_TYPES or kind == "unknown":
        errors.append("outcome.event_type: explicitly distinguish target_event/event_free/competing_event")
    event = time(outcome.get("event_time"), "outcome.event_time", False)
    death = time(outcome.get("death_date"), "outcome.death_date", False)
    free = time(outcome.get("dated_event_free_followup"), "outcome.dated_event_free_followup", False)
    follow = time(outcome.get("last_confirmed_followup_time"), "outcome.last_confirmed_followup_time")
    for field, date in (("event_time", event), ("death_date", death),
                        ("dated_event_free_followup", free), ("last_confirmed_followup_time", follow)):
        if date and index and date <= index:
            errors.append("outcome." + field + ": must be after index for this future endpoint")
    if event:
        if status != "adjudicated" or kind != "target_event":
            errors.append("outcome: dated target event must be adjudicated target_event")
        if death and death < event:
            errors.append("outcome: target event after death is invalid")
        if follow and event > follow:
            errors.append("outcome: event later than last confirmed follow-up")
        if free and free >= event:
            errors.append("outcome: event-free claim at/after the target event is inconsistent")
    else:
        if not free:
            errors.append("outcome: no event_time requires explicit dated event-free ascertainment; never infer negative")
        if death:
            if kind != "competing_event" or status != "adjudicated":
                errors.append("outcome: death without preceding target must be adjudicated competing_event")
        elif status != "censored" or kind != "event_free":
            errors.append("outcome: no target/death is censored event_free, not a permanent negative")
    if free and follow and free > follow:
        errors.append("outcome: event-free ascertainment later than last confirmed follow-up")
    if death and free and free > death:
        errors.append("outcome: target event-free ascertainment after death is invalid")
    if death and follow and death > follow:
        errors.append("outcome: death is not covered by last confirmed follow-up")
    if outcome.get("label") == 0:
        warnings.append("outcome.label=0 ignored; only dated event/censor/competing evidence is used")
    return dict(eligible=not errors, errors=errors, warnings=warnings)


def sample_row():
    """Fictional structural example, not a patient, dataset row or risk output."""
    def feature(name, value):
        return dict(name=name, value=value, status="valid",
                    acquired_at="2019-12-30T09:00:00+00:00",
                    available_at="2019-12-31T09:00:00+00:00")
    return dict(
        patient_id="SYNTHETIC-ONLY-0001", site="SYNTHETIC-NOT-A-HOSPITAL",
        index_time="2020-01-01T00:00:00+00:00", task="future_first_AF",
        prior_AF=False, prior_stroke=False,
        features=dict(clinical=[feature("age", 60), feature("sex", "female")],
                      ecg=[feature("RR_ms", 800)], brain=[feature("WMH_burden", 1.0)]),
        outcome=dict(target="future_first_AF", status="adjudicated", event_type="target_event",
                     event_time="2021-01-01T00:00:00+00:00", death_date=None,
                     dated_event_free_followup=None,
                     last_confirmed_followup_time="2021-02-01T00:00:00+00:00",
                     assessor="SYNTHETIC-TEST-ONLY", source="SYNTHETIC-NOT-CLINICAL"),
    )


def mock_tests():
    """Assert synthetic leakage/missingness/outcome checks; no fitting or I/O."""
    tests = []

    def check(name, row, expected, error_contains=None):
        result = validate(row)
        assert result["eligible"] is expected, (name, result)
        if error_contains:
            assert any(error_contains in e for e in result["errors"]), (name, result)
        tests.append(dict(name=name, state="PASS", eligible=expected,
                          errors=result["errors"], warnings=result["warnings"]))

    check("synthetic_future_incident_AF", sample_row(), True)
    row = sample_row(); row["outcome"].update(event_time=None, status="censored", event_type="event_free",
        dated_event_free_followup="2021-02-01T00:00:00+00:00")
    check("explicit_event_free_followup_is_censoring", row, True)
    implicit = deepcopy(row); implicit["outcome"].update(dated_event_free_followup=None, label=0)
    check("missing_event_never_implicit_negative", implicit, False, "never infer negative")
    later = sample_row(); later["features"]["ecg"][0]["available_at"]="2020-02-01T00:00:00+00:00"
    check("future_feature_availability_is_leakage", later, False, "unavailable at index")
    reverse = sample_row(); reverse["features"]["ecg"][0]["acquired_at"]="2020-02-02T00:00:00+00:00"
    check("acquisition_after_availability_or_index", reverse, False, "acquired_at after")
    naive = sample_row(); naive["index_time"]="2020-01-01T00:00:00"
    check("timezone_naive_index_rejected", naive, False, "timezone-naive")
    dateonly = sample_row(); dateonly["outcome"]["death_date"]="2022-01-01"
    check("date_only_requires_reviewed_resolution_policy", dateonly, False, "invalid or timezone-naive")
    for state in ("rejected", "unreadable", "not_acquired", "unknown"):
        zero = sample_row(); zero["features"]["ecg"][0].update(status=state, value=0)
        check("%s_cannot_be_imputed_zero" % state, zero, False, "never imputed zero")
    rejected = sample_row(); rejected["features"]["ecg"].append(dict(
        name="P_candidate_rejected", value=True, status="valid", biological_noP=True,
        acquired_at="2019-12-30T09:00:00+00:00", available_at="2019-12-31T09:00:00+00:00"))
    check("technical_P_rejection_not_biological_absence", rejected, False, "not biological_noP")
    missing = sample_row(); missing["features"]["ecg"].append(dict(name="P_duration", value=None,
        status="rejected", acquired_at="2019-12-30T09:00:00+00:00", available_at="2019-12-31T09:00:00+00:00"))
    check("rejected_P_remains_missing_not_false_absence", missing, True)
    for prior in (True, "unknown", None):
        oldaf = sample_row(); oldaf["prior_AF"] = prior
        check("first_AF_requires_absent_history_%s" % prior, oldaf, False, "prior_AF=False")
    pre = sample_row(); pre["outcome"]["event_time"]="2019-01-01T00:00:00+00:00"
    check("baseline_AF_not_future_event", pre, False, "must be after index")
    stroke = sample_row(); stroke["task"]="future_first_stroke"; stroke["outcome"]["target"]=stroke["task"]
    check("first_stroke_separate_endpoint", stroke, True)
    stroke["prior_stroke"]=True
    check("first_stroke_cannot_include_prior_stroke", stroke, False, "prior_stroke=False")
    stroke["task"]="future_recurrent_stroke"; stroke["outcome"]["target"]=stroke["task"]
    check("recurrence_requires_prior_stroke", stroke, True)
    stroke["prior_stroke"]=False
    check("recurrence_without_prior_stroke_rejected", stroke, False, "prior_stroke=True")
    competing = deepcopy(row); competing["outcome"].update(status="adjudicated", event_type="competing_event",
        death_date="2021-02-01T00:00:00+00:00")
    check("death_without_target_is_competing_event", competing, True)
    disguised = deepcopy(competing); disguised["outcome"].update(status="censored", event_type="event_free")
    check("death_not_silently_ordinary_censoring", disguised, False, "competing_event")
    afterdeath = sample_row(); afterdeath["outcome"]["death_date"]="2020-02-01T00:00:00+00:00"
    check("target_after_death_impossible", afterdeath, False, "target event after death")
    short = sample_row(); short["outcome"]["last_confirmed_followup_time"]="2020-02-01T00:00:00+00:00"
    check("event_after_last_confirmed_followup_rejected", short, False, "event later")
    unknown = sample_row(); unknown["outcome"]["status"]="unknown"
    check("unknown_ascertainment_not_evaluable", unknown, False, "unknown ascertainment")
    nosource = sample_row(); nosource["outcome"]["source"]=""
    check("outcome_provenance_required", nosource, False, "outcome.source")
    optional = sample_row(); optional["features"]["clinical"]=[]
    check("clinical_age_sex_optional_with_explicit_warning", optional, True)
    nonfinite = sample_row(); nonfinite["features"]["brain"][0]["value"]=float("nan")
    check("nonfinite_feature_not_valid", nonfinite, False, "nonfinite")
    extra = sample_row(); extra["features"]["post_event_score"] = []
    check("unrecognized_feature_group_not_silently_used", extra, False, "unrecognized group")
    assert all(t["state"] == "PASS" for t in tests)
    return dict(state="PASS", tests=tests, count=len(tests), fitting=0,
                predictions=0, clinical_risk_outputs=0, synthetic_only=True,
                source_admission=False, clinically_validated=False)
