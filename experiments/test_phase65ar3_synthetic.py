import csv
import hashlib
import json
import runpy
import tempfile
import zipfile
from pathlib import Path

import numpy as np

CELL = Path(__file__).with_name("phase65_cell_165ar3_local_torch_hub_offline_rehearsal.py")


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(root: Path, official: Path, mode: str):
    root.mkdir(parents=True)
    main = f'''import csv, os, torch
repo = os.path.join(os.path.dirname(__file__), "toy_repo")
torch.hub.load(repo, "toy", source="local", pretrained=False, force_reload=False)
base = r"{official.as_posix()}"
rows = list(csv.DictReader(open(os.path.join(base, "data", "submission_format.csv"))))
assert len(os.listdir(os.path.join(base, "data", "niftis"))) == len(rows)
with open(os.path.join(base, "submission.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["uid", "is_pathologic"]); w.writeheader()
    for i, row in enumerate(rows): w.writerow({{"uid":row["uid"], "is_pathologic":0.2+i*0.01}})
'''
    hubconf = 'def toy(pretrained=False):\n    return {"ok": True}\n'
    fake_torch = '''import importlib.util, os
class _Hub:
    def load(self, repo_or_dir, model, *args, **kwargs):
        spec = importlib.util.spec_from_file_location("toy_hubconf", os.path.join(repo_or_dir, "hubconf.py"))
        module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
        kwargs.pop("source", None); kwargs.pop("force_reload", None); kwargs.pop("trust_repo", None); kwargs.pop("skip_validation", None)
        return getattr(module, model)(*args, **kwargs)
    def load_state_dict_from_url(self, url, *args, **kwargs): raise RuntimeError("not used")
hub = _Hub()
def load(path, *args, **kwargs): return {}
'''
    archive = root / "phase56_submission.zip"
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("main.py", main)
        z.writestr("toy_repo/hubconf.py", hubconf)
        z.writestr("torch.py", fake_torch)
    sha = digest(archive)
    (root / "phase56_final_archive_contract.json").write_text(json.dumps({"status":"accepted", "submission_sha256":sha}))
    (root / "phase65ar2_phase56_provenance_reachability_replay_contract.json").write_text(json.dumps({
        "status":"archive_lineage_passed_reachable_network_call_requires_offline_replay",
        "artifact_provenance":{"authoritative_final_archive_lineage_established":True, "submission_zip":{"sha256":sha}}
    }))
    init = {"PHASE65AR3_ROOT_OVERRIDE_PRIVATE":str(root), "PHASE65AR3_OFFICIAL_ROOT_OVERRIDE_PRIVATE":str(official)}
    if mode != "static":
        scans = []
        for i in range(8):
            p = root / f"dummy_{i}.nii.gz"; p.write_bytes(b"synthetic"); scans.append(str(p))
        bundle = {"nifti_paths":scans, "minimum_cases":8}
        if mode in {"exact", "mismatch"}:
            offset = 0.0 if mode == "exact" else 0.001
            bundle["reference_probability"] = np.asarray([0.2+i*0.01+offset for i in range(8)])
        init["PHASE65AR3_REHEARSAL_PRIVATE"] = bundle
    result = runpy.run_path(str(CELL), init_globals=init)
    return result["PHASE65AR3_REPORT_PRIVATE"]


with tempfile.TemporaryDirectory() as d:
    base = Path(d)
    static = build(base/"static", base/"official_static", "static")
    assert static["status"] == "static_torch_hub_semantics_recorded_private_offline_rehearsal_required"
    offline = build(base/"offline", base/"official_offline", "offline")
    assert offline["status"] == "accepted_offline_local_torch_hub_and_runtime_path_rehearsal_numeric_parity_not_available", offline["status"]
    exact = build(base/"exact", base/"official_exact", "exact")
    assert exact["status"] == "accepted_offline_and_exact_phase56_private_replay_ready_for_corrected_phase65b", exact["status"]
    mismatch = build(base/"mismatch", base/"official_mismatch", "mismatch")
    assert mismatch["status"] == "offline_runtime_passed_exact_development_probability_parity_failed_stop", mismatch["status"]
print("phase65ar3 synthetic static, offline, exact, and mismatch tests passed")
