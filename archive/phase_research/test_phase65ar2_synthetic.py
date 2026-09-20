import hashlib
import json
import runpy
import tempfile
import zipfile
from pathlib import Path

import numpy as np

def _require_symlink_support(test_name):
    """Exit with a visible, explained skip when the OS refuses symlinks.

    The audited cell reproduces the official /code_execution layout with a real
    symlink and guards on Path.is_symlink(). A Windows junction is not a symlink
    and would change what the audit proves, so this test is not run rather than
    run against a substituted mechanism. Enable Developer Mode (or run elevated)
    on Windows, or use the declared POSIX reference environment.
    """
    import os as _os, tempfile as _tempfile
    with _tempfile.TemporaryDirectory() as _d:
        _target = _os.path.join(_d, "t")
        _os.mkdir(_target)
        try:
            _os.symlink(_target, _os.path.join(_d, "l"))
        except OSError as exc:
            print(f"{test_name} SKIPPED: this OS refuses symlink creation ({exc.__class__.__name__}"
                  f"{' WinError ' + str(exc.winerror) if getattr(exc, 'winerror', None) else ''}).")
            print("  The audit requires real symlinks; see docs/VALIDATION.md.")
            raise SystemExit(0)


_require_symlink_support("PHASE65AR2_SYNTHETIC")


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
    # Write bytes, not text: write_text applies newline translation, so on
    # Windows the staged copies would be CRLF while zipfile.writestr keeps LF,
    # and the stage-vs-archive payload comparison would fail spuriously.
    (stage / "main.py").write_bytes(main.encode("utf-8"))
    (stage / "unused.py").write_bytes(unused.encode("utf-8"))
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
