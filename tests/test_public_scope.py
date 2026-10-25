"""Publication-scope checks, not a formal disclosure or clinical safety proof."""
import json
from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]


class PublicScope(unittest.TestCase):
    def test_no_raw_data_weights_or_secrets_files(self):
        forbidden = {".dat", ".hea", ".dcm", ".edf", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".joblib", ".xlsx", ".xls"}
        for file in ROOT.rglob("*"):
            if not file.is_file() or ".git" in file.relative_to(ROOT).parts:
                continue
            self.assertNotIn(file.suffix.lower(), forbidden, file.relative_to(ROOT))
            self.assertNotEqual(file.name, "secrets.toml")
            self.assertNotIn("hospital_request", file.relative_to(ROOT).parts)

    def test_no_local_absolute_drive_paths_or_token_patterns(self):
        patterns = [re.compile(r"(?i)(?<![a-z0-9_])[a-z]:[\\/]+"),
                    re.compile(r"/mnt/[a-z]/"),
                    re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}"),
                    re.compile(r"github_pat_[A-Za-z0-9_]{20,}"),
                    re.compile(r"sk-[A-Za-z0-9]{20,}")]
        for file in ROOT.rglob("*"):
            if not file.is_file() or ".git" in file.relative_to(ROOT).parts or "__pycache__" in file.relative_to(ROOT).parts:
                continue
            if file.suffix.lower() not in {".py", ".ps1", ".mjs", ".txt", ".md", ".json", ".html", ".toml", ".svg", ".csv", ""}:
                continue
            content = file.read_text(encoding="utf-8-sig")
            for i, pattern in enumerate(patterns):
                self.assertIsNone(pattern.search(content), (file.relative_to(ROOT).as_posix(), "pattern", i))

    def test_no_patient_rows_in_aggregate_data(self):
        data = json.loads((ROOT / "data/audited_aggregate_results.json").read_text(encoding="utf-8"))
        def visit(value):
            if isinstance(value, dict):
                self.assertTrue({"patient_id", "record", "dob", "patient_name", "hospital_id"}.isdisjoint(value))
                for child in value.values():
                    visit(child)
            elif isinstance(value, list):
                for child in value:
                    visit(child)
        visit(data)
        self.assertFalse(data["clinical_prediction_allowed"])


if __name__ == "__main__":
    unittest.main()
