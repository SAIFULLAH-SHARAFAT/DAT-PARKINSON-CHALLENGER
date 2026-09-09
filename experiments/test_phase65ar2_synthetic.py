import hashlib
import json
import runpy
import tempfile
import zipfile
from pathlib import Path

import numpy as np

CELL = Path(__file__).with_name("phase65_cell_165ar2_phase56_provenance_reachability_replay_audit.py")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, official: Path, with_replay: bool, reference_offset: float = 0.0):
    main = f'''import csv, os
root = r"{official.as_posix()}"
rows = list(csv.DictReader(open(os.path.join(root, "data", "submission_format.csv"))))
names = sorted(os.listdir(os.path.join(root, "data", "niftis")))
assert len(names) == len(rows)
with open(os.path.join(root, "submission.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["uid", "is_pathologic"])
    w.writeheader()
    for i, row in enumerate(rows): w.writerow({{"uid": row["uid"], "is_pathologic": 0.1 + i * 0.01}})
'''
    unused = 'import requests\nURL="https://example.com/model"\n'
    stage = root / "phase56_experimental_submission"
    stage.mkdir(parents=True)
    (stage / "main.py").write_text(main)
    (stage / "unused.py").write_text(unused)
    sub = root / "phase56_submission.zip"
    with zipfile.ZipFile(sub, "w") as z:
        z.writestr("main.py", main)
        z.writestr("unused.py", unused)
    with zipfile.ZipFile(root / "phase56_runtime_source_handoff.zip", "w") as z:
        z.writestr("main.py", main + "\n# stale\n")
    digest = sha(sub)
    (root / "phase56_archive_build_contract.json").write_text(json.dumps({"status": "accepted_candidate_not_yet_published", "submission_sha256": digest}))
    (root / "phase56_final_archive_contract.json").write_text(json.dumps({"status": "accepted", "submission_sha256": digest}))
    (root / "phase56_real_smoke_contract.json").write_text(json.dumps({"status": "accepted", "phase": "phase56_real_smoke"}))
    (root / "phase65a_corrected_deployment_fallback_contract.json").write_text(json.dumps({"status": "fallback_numeric_gate_failed_retain_phase12c_unknown_fallback"}))
    (root / "phase65ar_phase56_runtime_parity_contract.json").write_text(json.dumps({"status": "static_runtime_or_archive_parity_failed_stop"}))
    init = {"PHASE65AR2_ROOT_OVERRIDE_PRIVATE": str(root), "PHASE65AR2_OFFICIAL_ROOT_OVERRIDE_PRIVATE": str(official)}
    if with_replay:
        scans = []
        for i in range(8):
            p = root / f"dummy_{i}.nii.gz"
            p.write_bytes(b"synthetic")
            scans.append(str(p))
        init["PHASE65AR2_REPLAY_PRIVATE"] = {
            "nifti_paths": scans,
            "reference_probability": np.asarray([0.1 + i * 0.01 + reference_offset for i in range(8)]),
            "minimum_cases": 8,
        }
    result = runpy.run_path(str(CELL), init_globals=init)
    return result["PHASE65AR2_REPORT_PRIVATE"]


with tempfile.TemporaryDirectory() as d:
    base = Path(d)
    static = build(base / "static", base / "official_static", False)
    assert static["status"] == "archive_lineage_passed_private_replay_not_configured", static["status"]
    dynamic = build(base / "dynamic", base / "official_dynamic", True)
    assert dynamic["status"].startswith("accepted_phase56"), dynamic["status"]
    assert dynamic["gates"]["exact_development_to_deployment_probability_parity"] is True
    assert dynamic["artifact_provenance"]["runtime_handoff_classification"] == "non_authoritative_stale_or_pre_final_snapshot"
    mismatch = build(base / "mismatch", base / "official_mismatch", True, 0.001)
    assert mismatch["status"] == "private_runtime_or_development_parity_failed_stop"
    assert mismatch["gates"]["exact_development_to_deployment_probability_parity"] is False
print("phase65ar2 synthetic static, exact replay, and mismatch-rejection tests passed")
