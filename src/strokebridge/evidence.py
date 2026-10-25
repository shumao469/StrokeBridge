"""Read-only aggregate evidence access and frozen-source verification."""
import hashlib
import json
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[2]


def load_bundle():
    result = json.loads((PACKAGE / "data/audited_aggregate_results.json").read_text(encoding="utf-8"))
    if result.get("clinical_prediction_allowed") is not False:
        raise ValueError("unsafe evidence bundle")
    return result


def verify_sources(workspace):
    """Verify original source hashes; no imports/execution of archived scripts."""
    root = Path(workspace).resolve()
    inventory = json.loads((PACKAGE / "provenance/source_inventory.json").read_text(encoding="utf-8"))
    rows = []
    for item in inventory["files"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root):
            raise ValueError("source path escapes workspace")
        if not path.is_file():
            rows.append({"path": item["path"], "state": "MISSING"})
            continue
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        rows.append({"path": item["path"], "state": "PASS" if digest == item["sha256"] else "CHANGED"})
    return {"state": "PASS" if all(r["state"] == "PASS" for r in rows) else "FAIL", "files": rows}


def summary():
    b = load_bundle()
    return {"platform": b["platform"], "version": b["schema_version"],
            "A32": b["A32_completion"], "A33": b["A33_completion"], "A34": b["A34_completion"],
            "boundaries": b["evidence_boundaries"]}

