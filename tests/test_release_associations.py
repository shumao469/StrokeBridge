"""Portable archive/aggregate checks; no imports of historical experiment code."""
import hashlib
import json
from pathlib import Path
import unittest

ROOT = Path(__file__).resolve().parents[1]


class ReleaseAssociations(unittest.TestCase):
    def test_original_aggregate_projection_when_workspace_present(self):
        workspace = ROOT.parents[2]
        bundle = json.loads((ROOT / "data/audited_aggregate_results.json").read_text(encoding="utf-8"))
        for phase, folder in (("A32", "phase_A32_20261024_v1"), ("A33", "phase_A33_20261024_v1")):
            source = workspace / "unified_project/runs" / folder / "analysis/summary.json"
            if not source.is_file():
                self.skipTest("Original workspace absent; archived-source and aggregate checks remain portable")
            original = json.loads(source.read_text(encoding="utf-8"))
            index = {(r["policy"], r["arm"], r["seed"], r["lead"]): r for r in original}
            for row in bundle[phase + "_results"]:
                old = index[(row["policy"], row["arm"], row["seed"], row["lead"])]
                for key in ("reference_positive_patients", "strict_patient_joint_before", "strict_patient_joint_reported", "median_width_ms"):
                    self.assertEqual(row[key], old[key])
                for stage in ("raw", "selected", "reported"):
                    now, before = row["stages"][stage], old["stages"][stage]
                    for key in ("reference", "matched", "predicted", "recall", "precision_lower", "precision_upper"):
                        self.assertEqual(now[key], before[key])
                    for polarity in ("negative", "biphasic"):
                        self.assertEqual(now[polarity + "_recall"], before["polarity"][polarity]["recall"])

    def test_99_archived_sources_match_export_hashes(self):
        manifest = json.loads((ROOT / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
        self.assertEqual(len(manifest["files"]), 99)
        for item in manifest["files"]:
            path = (ROOT / item["archive_path"]).resolve()
            self.assertTrue(path.is_relative_to(ROOT / "archive"))
            self.assertIsInstance(item["byte_identical"], bool)
            self.assertEqual(path.stat().st_size, item["bytes"])
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(digest, item["copy_sha256"])
            if item["byte_identical"]: self.assertEqual(digest, item["source_sha256"])

    def test_archive_contains_only_manifested_source_files(self):
        manifest = json.loads((ROOT / "ARCHIVE_MANIFEST.json").read_text(encoding="utf-8"))
        expected = {item["archive_path"] for item in manifest["files"]}
        actual = {p.relative_to(ROOT).as_posix() for p in (ROOT / "archive").rglob("*") if p.is_file()}
        self.assertEqual(actual, expected)
        self.assertTrue(all(Path(p).suffix in (".py", ".ps1", ".mjs", ".txt") for p in actual))

    def test_aggregate_has_no_patient_rows_and_preserves_zero_fit_stop(self):
        b = json.loads((ROOT / "data/audited_aggregate_results.json").read_text(encoding="utf-8"))
        self.assertEqual(len(b["A32_results"]), 16)
        self.assertEqual(len(b["A33_results"]), 48)
        self.assertNotIn("data/", json.dumps(b))
        self.assertEqual(b["A34_completion"]["state"], "SCIENTIFIC_STOP_ENDPOINT_ONLY_NOT_ADMISSIBLE")
        self.assertEqual(b["A34_completion"]["fits"], 0)
        self.assertFalse(b["clinical_prediction_allowed"])


if __name__ == "__main__":
    unittest.main()

