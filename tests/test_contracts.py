import math
import unittest
from strokebridge import brain_heart_contract as clinical
from strokebridge.evidence import load_bundle
from strokebridge.reliability import (maximum_matching, strict_quantile,
    residual_or_detection_failure, patient_joint_coverage, precision_or_undefined)


class ReliabilityTests(unittest.TestCase):
    def test_augmenting_path_finds_two(self):
        self.assertEqual(len(maximum_matching({"r1": ["c1", "c2"], "r2": ["c1"]})), 2)

    def test_candidate_reuse_not_coverage(self):
        result = patient_joint_coverage(["p1", "p2"], {"p1": ["c1"], "p2": ["c1"]})
        self.assertFalse(result["covered"])

    def test_empty_positive_denominator_not_success(self):
        self.assertIsNone(patient_joint_coverage([], {})["covered"])

    def test_cross_heartbeat_is_detection_failure(self):
        r = residual_or_detection_failure((100, 150), (0, 50), 180, 60, 5)
        self.assertEqual(r["kind"], "detection_or_identity_failure")
        self.assertEqual(r["score"], math.inf)

    def test_local_endpoint_score(self):
        self.assertEqual(residual_or_detection_failure((100, 150), (95, 160), 180, 180, 5)["score"], 2)

    def test_quantile_max_at_22(self):
        self.assertEqual(strict_quantile(list(range(22))), 21)

    def test_infinite_score_not_deleted(self):
        self.assertEqual(strict_quantile(list(range(21)) + [math.inf]), math.inf)

    def test_insufficient_calibration_infinite(self):
        self.assertEqual(strict_quantile([0, 1]), math.inf)

    def test_bad_quantile_rejected(self):
        with self.assertRaises(ValueError): strict_quantile([float("nan")])

    def test_all_reject_precision_undefined(self):
        self.assertIsNone(precision_or_undefined(0, 0))

    def test_original_synthetic_contract_suite(self):
        result = clinical.mock_tests()
        self.assertEqual(result["count"], 30)
        self.assertEqual(result["fitting"], 0)

    def test_aggregate_scope_not_clinical_model(self):
        b = load_bundle()
        self.assertFalse(b["clinical_prediction_allowed"])
        self.assertEqual(b["A34_completion"]["fits"], 0)
        self.assertFalse(b["evidence_boundaries"]["future_stroke_validated"])


if __name__ == "__main__":
    unittest.main()

