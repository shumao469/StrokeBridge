"""Pure diagnostic contracts; these are not learned measurement algorithms.

Coordinates in a caller's single consistent unit. An admissible identity edge
must be established before calling maximum_matching; this module never borrows
a candidate from another heartbeat and never silently reuses a candidate.
"""
from math import ceil, inf, isfinite


def maximum_matching(edges):
    """Maximum-cardinality bipartite matching, reference -> distinct candidate.

    `edges` maps each hashable reference ID to an iterable of legal candidate
    IDs. It is a diagnostic upper bound if reference information built the edges,
    not an implementable test-time selector. Deterministic input order is used.
    """
    owner = {}
    def augment(ref, seen):
        for candidate in edges[ref]:
            if candidate in seen:
                continue
            seen.add(candidate)
            if candidate not in owner or augment(owner[candidate], seen):
                owner[candidate] = ref
                return True
        return False
    for ref in edges:
        augment(ref, set())
    return {ref: candidate for candidate, ref in owner.items()}


def strict_quantile(scores, alpha=0.05):
    """Split-conformal order statistic with +infinity when k > n.

    Values are patient scores, not individual-wave scores. Missing same-heartbeat
    measurement is +infinity, not a several-second local endpoint residual.
    The statistic alone supplies no conditional/post-selection guarantee.
    """
    if not 0 < alpha < 1 or not scores:
        raise ValueError("nonempty scores and 0 < alpha < 1 required")
    values = list(scores)
    if any(not isinstance(v, (int, float)) or v != v or v < 0 for v in values):
        raise ValueError("scores must be nonnegative numbers; +infinity allowed")
    k = ceil((len(values) + 1) * (1 - alpha))
    return inf if k > len(values) else sorted(values)[k - 1]


def residual_or_detection_failure(reference, candidate, reference_R, candidate_R, scale):
    """Separate identity/detection failure from finite local boundary error."""
    if candidate is None or reference_R != candidate_R:
        return {"kind": "detection_or_identity_failure", "score": inf}
    if not isfinite(scale) or scale <= 0:
        raise ValueError("positive finite scale required")
    if len(reference) != 2 or len(candidate) != 2:
        raise ValueError("onset/offset pairs required")
    if any(not isfinite(v) for v in list(reference) + list(candidate)):
        raise ValueError("finite endpoints required")
    if reference[0] > reference[1] or candidate[0] > candidate[1]:
        raise ValueError("ordered endpoints required")
    return {"kind": "local_boundary_error",
            "score": max(abs(a - b) for a, b in zip(reference, candidate)) / scale}


def patient_joint_coverage(reference_ids, legal_edges):
    """All reference waves covered by distinct legal candidates, not union-only."""
    refs = list(reference_ids)
    if not refs:
        return {"eligible_positive_patient": False, "covered": None, "matched": 0}
    if len(refs) != len(set(refs)):
        raise ValueError("reference IDs must be unique")
    edges = {r: list(legal_edges.get(r, [])) for r in refs}
    matched = maximum_matching(edges)
    return {"eligible_positive_patient": True, "covered": len(matched) == len(refs),
            "matched": len(matched), "reference": len(refs)}


def precision_or_undefined(matched, predicted):
    """An all-reject model has undefined precision, not precision zero."""
    if not 0 <= matched <= predicted:
        raise ValueError("require 0 <= matched <= predicted")
    return None if predicted == 0 else matched / predicted

