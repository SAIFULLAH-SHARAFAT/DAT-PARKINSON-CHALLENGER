"""
Cell 165A-R3 — Local Torch Hub semantics and offline Phase56 rehearsal

Run after Phase65A-R2 in the same Kaggle notebook. Phase56 artifacts remain
read-only. No labels are used. No private paths, UIDs, probabilities, scan
contents, source lines, or subprocess logs are written to the sanitized report.

Private rehearsal configuration (required for dynamic evidence):

PHASE65AR3_REHEARSAL_PRIVATE = {
    "nifti_paths": [...],                 # fixed private development cases
    "minimum_cases": 8,                  # 8–16 diverse cases recommended
    "uid_column": "uid",                 # optional
    "prediction_column": "is_pathologic",# optional

    # Optional: exact Phase56 development-path probabilities in the same order.
    # Omit this key when no exact reference exists. Never substitute Phase12c,
    # Phase42, or another model's OOF vector.
    "reference_probability": np.asarray([...], dtype=np.float64),

    # Optional aggregate runtime metadata, if the development runner exposes it.
    "runtime_metadata": {
        "unreadable_count": 0,
        "branch_counts": {...},
        "preprocessing_checks": {...},
    },
    "reference_branch_counts": {...},     # optional
}

The final ZIP is extracted without source modification and run twice:
  1) instrumented/offline: observes official paths, torch.hub.load semantics,
     local checkpoint loads, URL-loader calls, and blocked socket attempts;
  2) clean/offline: blocks sockets but does not instrument Torch Hub or paths.

The cell temporarily creates /code_execution only when absent and removes only
the symlink it created. If it already exists, the rehearsal stops safely.
"""

from __future__ import annotations

import ast
import csv
import hashlib
import json
import math
import os
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np


PHASE65AR3_STARTED = time.time()
PHASE65AR3_ROOT = Path(globals().get("PHASE65AR3_ROOT_OVERRIDE_PRIVATE", "/kaggle/working"))
PHASE65AR3_OFFICIAL_ROOT = Path(
    globals().get("PHASE65AR3_OFFICIAL_ROOT_OVERRIDE_PRIVATE", "/code_execution")
)
PHASE65AR3_SUBMISSION_ZIP = PHASE65AR3_ROOT / "phase56_submission.zip"
PHASE65AR3_FINAL_CONTRACT = PHASE65AR3_ROOT / "phase56_final_archive_contract.json"
PHASE65AR3_R2_CONTRACT = PHASE65AR3_ROOT / "phase65ar2_phase56_provenance_reachability_replay_contract.json"
PHASE65AR3_CONTRACT_PATH = PHASE65AR3_ROOT / "phase65ar3_local_torch_hub_offline_rehearsal_contract.json"

PHASE65AR3_CONFIG = {
    "probability_atol": 1.0e-9,
    "probability_rtol": 0.0,
    "subprocess_timeout_seconds": 1200,
    "maximum_log_capture_bytes": 2_000_000,
    "minimum_default_cases": 8,
    "require_all_hub_calls_local": True,
    "require_zero_network_attempts": True,
}


def phase65ar3_sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase65ar3_sha_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(8 * 1024 * 1024)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phase65ar3_safe_json(path: Path) -> dict:
    out = {"exists": path.is_file(), "valid_object": False, "status": None,
           "sha256": None, "payload": None, "error_type": None}
    if not out["exists"]:
        return out
    try:
        raw = path.read_bytes()
        obj = json.loads(raw.decode("utf-8"))
        out.update({"valid_object": isinstance(obj, dict), "sha256": phase65ar3_sha_bytes(raw)})
        if isinstance(obj, dict):
            out["payload"] = obj
            out["status"] = obj.get("status")
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar3_scalars(obj):
    if isinstance(obj, dict):
        for value in obj.values():
            yield from phase65ar3_scalars(value)
    elif isinstance(obj, list):
        for value in obj:
            yield from phase65ar3_scalars(value)
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        yield obj


def phase65ar3_mentions_digest(payload: dict | None, digest: str) -> bool:
    return bool(isinstance(payload, dict) and digest and any(
        isinstance(v, str) and v.lower() == digest.lower() for v in phase65ar3_scalars(payload)
    ))


def phase65ar3_safe_member(name: str) -> bool:
    p = PurePosixPath(name)
    return bool(name) and "\\" not in name and not p.is_absolute() and ".." not in p.parts


def phase65ar3_zip_audit(path: Path) -> dict:
    out = {"exists": path.is_file(), "valid": False, "sha256": None, "crc_ok": None,
           "root_main": False, "file_count": 0, "python_sources": {},
           "unsafe_count": 0, "symlink_count": 0, "duplicate_count": 0,
           "error_type": None}
    if not out["exists"]:
        return out
    try:
        out["sha256"] = phase65ar3_sha_file(path)
        with zipfile.ZipFile(path) as zf:
            infos = zf.infolist()
            names = [i.filename for i in infos]
            files = [i for i in infos if not i.is_dir()]
            out.update({
                "valid": True, "file_count": len(files), "root_main": "main.py" in names,
                "unsafe_count": sum(not phase65ar3_safe_member(n) for n in names),
                "duplicate_count": len(names) - len(set(names)),
                "symlink_count": sum(stat.S_ISLNK((i.external_attr >> 16) & 0xFFFF) for i in infos),
                "crc_ok": zf.testzip() is None,
            })
            for info in files:
                if info.filename.endswith(".py"):
                    out["python_sources"][info.filename] = zf.read(info).decode("utf-8", errors="replace")
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar3_dotted(node) -> str | None:
    parts = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
        return ".".join(reversed(parts))
    return None


def phase65ar3_literal(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (str, int, float, bool, type(None))):
        return node.value
    return None


def phase65ar3_static_hub_calls(source_map: dict[str, str]) -> dict:
    records = []
    for rel, text in sorted(source_map.items()):
        try:
            tree = ast.parse(text, filename=rel)
        except Exception:
            continue
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call) or phase65ar3_dotted(node.func) != "torch.hub.load":
                continue
            kws = {kw.arg: phase65ar3_literal(kw.value) for kw in node.keywords if kw.arg}
            repo_literal = phase65ar3_literal(node.args[0]) if node.args else None
            model_literal = phase65ar3_literal(node.args[1]) if len(node.args) > 1 else None
            records.append({
                "file": rel,
                "line": int(node.lineno),
                "source_literal": kws.get("source"),
                "source_explicitly_local": kws.get("source") == "local",
                "repo_or_dir_literal": isinstance(repo_literal, str),
                "repo_or_dir_basename": Path(repo_literal).name if isinstance(repo_literal, str) else None,
                "repo_expression_type": type(node.args[0]).__name__ if node.args else None,
                "model_literal_available": isinstance(model_literal, str),
                "pretrained_literal": kws.get("pretrained"),
                "force_reload_literal": kws.get("force_reload"),
                "trust_repo_literal": kws.get("trust_repo"),
                "skip_validation_literal": kws.get("skip_validation"),
                "source_lines_exported": False,
            })
    return {
        "call_count": len(records),
        "records": records,
        "all_calls_explicitly_local_static": bool(records and all(r["source_explicitly_local"] for r in records)),
        "dynamic_argument_observation_required": True,
    }


def phase65ar3_extract_safe(path: Path, destination: Path) -> None:
    with zipfile.ZipFile(path) as zf:
        for info in zf.infolist():
            if not phase65ar3_safe_member(info.filename):
                raise ValueError("unsafe_zip_member")
            if stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF):
                raise ValueError("zip_symlink_member")
        zf.extractall(destination)


def phase65ar3_write_sitecustomize(directory: Path, instrumented: bool) -> None:
    common = r'''
import json, os, socket
_trace = os.environ.get("PHASE65AR3_TRACE_FILE")
def _emit(record):
    try:
        if _trace:
            fd = os.open(_trace, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
            os.close(fd)
    except Exception:
        pass
def _host(value):
    try:
        value = str(value).lower()
        if value and len(value) < 200: return value
    except Exception:
        pass
    return "unknown"
def _blocked_connect(self, address):
    host = _host(address[0] if isinstance(address, tuple) and address else "unknown")
    _emit({"kind":"network_attempt", "host":host})
    raise OSError("Phase65A-R3 offline rehearsal blocked outbound network")
def _blocked_create(address, *args, **kwargs):
    host = _host(address[0] if isinstance(address, tuple) and address else "unknown")
    _emit({"kind":"network_attempt", "host":host})
    raise OSError("Phase65A-R3 offline rehearsal blocked outbound network")
socket.socket.connect = _blocked_connect
socket.create_connection = _blocked_create
'''
    extra = r'''
import builtins, io
_official = os.environ.get("PHASE65AR3_OFFICIAL_ROOT", "/code_execution").replace("\\", "/").rstrip("/")
_submission = os.path.realpath(os.environ.get("PHASE65AR3_SUBMISSION_ROOT", "."))
_open0, _io_open0, _listdir0, _scandir0 = builtins.open, io.open, os.listdir, os.scandir
def _category(path):
    try: p = os.fspath(path).replace("\\", "/")
    except Exception: return "other"
    if (_official + "/data/niftis") in p: return "niftis"
    if p.endswith(_official + "/data/submission_format.csv"): return "submission_format"
    if p.endswith(_official + "/submission.csv"): return "submission_output"
    return "other"
def _path_event(path):
    c = _category(path)
    if c != "other": _emit({"kind":"path", "category":c})
def _open(file, *a, **k): _path_event(file); return _open0(file, *a, **k)
def _io_open(file, *a, **k): _path_event(file); return _io_open0(file, *a, **k)
def _listdir(path="."): _path_event(path); return _listdir0(path)
def _scandir(path="."): _path_event(path); return _scandir0(path)
builtins.open, io.open, os.listdir, os.scandir = _open, _io_open, _listdir, _scandir
try:
    import torch
    _hub0 = torch.hub.load
    _url0 = torch.hub.load_state_dict_from_url
    _torch_load0 = torch.load
    def _inside(path):
        try:
            real = os.path.realpath(os.fspath(path))
            return os.path.commonpath([real, _submission]) == _submission
        except Exception: return False
    def _hub(repo_or_dir, model, *a, **k):
        source = k.get("source", "github")
        _emit({
            "kind":"torch_hub_load",
            "source":str(source),
            "repo_exists":bool(os.path.exists(os.fspath(repo_or_dir))),
            "repo_inside_submission":bool(_inside(repo_or_dir)),
            "pretrained":k.get("pretrained") if isinstance(k.get("pretrained"), (bool, type(None))) else "nonliteral_runtime",
            "force_reload":k.get("force_reload") if isinstance(k.get("force_reload"), (bool, type(None))) else "nonliteral_runtime",
            "trust_repo":k.get("trust_repo") if isinstance(k.get("trust_repo"), (bool, str, type(None))) else "nonliteral_runtime",
            "skip_validation":k.get("skip_validation") if isinstance(k.get("skip_validation"), (bool, type(None))) else "nonliteral_runtime",
        })
        return _hub0(repo_or_dir, model, *a, **k)
    def _url(url, *a, **k):
        _emit({"kind":"url_loader_call"})
        return _url0(url, *a, **k)
    def _torch_load(path, *a, **k):
        try:
            pathlike = isinstance(path, (str, bytes, os.PathLike))
            _emit({"kind":"torch_load", "pathlike":pathlike,
                   "exists":bool(os.path.exists(os.fspath(path))) if pathlike else None,
                   "inside_submission":bool(_inside(path)) if pathlike else None})
        except Exception:
            _emit({"kind":"torch_load", "pathlike":False, "exists":None, "inside_submission":None})
        return _torch_load0(path, *a, **k)
    torch.hub.load = _hub
    torch.hub.load_state_dict_from_url = _url
    torch.load = _torch_load
except Exception as exc:
    _emit({"kind":"instrumentation_setup_failure", "error_type":type(exc).__name__})
'''
    (directory / "sitecustomize.py").write_text(common + (extra if instrumented else ""), encoding="utf-8")


def phase65ar3_read_trace(path: Path) -> dict:
    paths = Counter()
    hub, torch_load = [], []
    url_calls = network = setup_failures = 0
    domains = Counter()
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                rec = json.loads(line)
            except Exception:
                continue
            kind = rec.get("kind")
            if kind == "path":
                paths[str(rec.get("category"))] += 1
            elif kind == "torch_hub_load":
                hub.append({k: rec.get(k) for k in (
                    "source", "repo_exists", "repo_inside_submission", "pretrained",
                    "force_reload", "trust_repo", "skip_validation")})
            elif kind == "torch_load":
                torch_load.append({k: rec.get(k) for k in ("pathlike", "exists", "inside_submission")})
            elif kind == "url_loader_call":
                url_calls += 1
            elif kind == "network_attempt":
                network += 1
                host = str(rec.get("host", "unknown")).lower()
                if re.fullmatch(r"[a-z0-9.-]+", host) and not re.fullmatch(r"[0-9.]+", host):
                    domains[host] += 1
            elif kind == "instrumentation_setup_failure":
                setup_failures += 1
    return {
        "official_path_observed": {k: paths[k] > 0 for k in ("niftis", "submission_format", "submission_output")},
        "official_path_access_counts": {k: int(paths[k]) for k in ("niftis", "submission_format", "submission_output")},
        "torch_hub_load_call_count": len(hub),
        "torch_hub_load_calls": hub,
        "all_observed_hub_calls_source_local": bool(hub and all(r["source"] == "local" for r in hub)),
        "all_observed_hub_repositories_exist": bool(hub and all(r["repo_exists"] for r in hub)),
        "all_observed_hub_repositories_inside_submission": bool(hub and all(r["repo_inside_submission"] for r in hub)),
        "torch_load_call_count": len(torch_load),
        "torch_load_pathlike_count": sum(bool(r["pathlike"]) for r in torch_load),
        "torch_load_missing_path_count": sum(r["pathlike"] and not r["exists"] for r in torch_load),
        "torch_load_outside_submission_count": sum(r["pathlike"] and not r["inside_submission"] for r in torch_load),
        "url_loader_call_count": int(url_calls),
        "socket_network_attempt_count": int(network),
        "network_attempt_domains": dict(sorted(domains.items())),
        "instrumentation_setup_failure_count": int(setup_failures),
        "private_paths_exported": False,
    }


def phase65ar3_load_output(path: Path, uid_col: str, pred_col: str, expected: list[str]) -> np.ndarray:
    with path.open("r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows or len(rows) != len(expected) or uid_col not in rows[0] or pred_col not in rows[0]:
        raise ValueError("output_schema_or_row_count_mismatch")
    mapping = {str(r[uid_col]): float(r[pred_col]) for r in rows}
    if set(mapping) != set(expected):
        raise ValueError("output_uid_set_mismatch")
    return np.asarray([mapping[u] for u in expected], dtype=np.float64)


def phase65ar3_rehearsal() -> tuple[dict, dict | None]:
    bundle = globals().get("PHASE65AR3_REHEARSAL_PRIVATE")
    report = {
        "configured": isinstance(bundle, dict), "valid_configuration": False,
        "reference_probability_available": False, "executed": False, "n": None,
        "instrumented_return_code": None, "clean_return_code": None,
        "instrumented_seconds": None, "clean_seconds": None,
        "instrumented_output_valid": False, "clean_output_valid": False,
        "finite_and_in_range": None, "instrumented_vs_clean_max_abs": None,
        "instrumentation_invariant": None, "reference_vs_clean_max_abs": None,
        "reference_vs_clean_mean_abs": None, "exact_reference_parity": None,
        "runtime_observation": None, "unreadable_count": None,
        "branch_counts_checked": False, "branch_counts_match": None,
        "preprocessing_checks_checked": False, "preprocessing_checks_all_true": None,
        "error_type": None, "error_stage": None, "case_level_values_exported": False,
        "paths_uids_or_logs_exported": False,
    }
    if not isinstance(bundle, dict):
        return report, None
    temporary = None
    created_link = False
    try:
        paths = [Path(p) for p in bundle["nifti_paths"]]
        minimum = int(bundle.get("minimum_cases", PHASE65AR3_CONFIG["minimum_default_cases"]))
        if len(paths) < minimum or not all(p.is_file() for p in paths):
            raise ValueError("invalid_rehearsal_paths_or_minimum_cases")
        ref = None
        if "reference_probability" in bundle:
            ref = np.asarray(bundle["reference_probability"], dtype=np.float64).reshape(-1)
            if ref.size != len(paths) or not np.isfinite(ref).all() or not ((ref >= 0) & (ref <= 1)).all():
                raise ValueError("invalid_reference_probability")
            report["reference_probability_available"] = True
        report["valid_configuration"] = True
        report["n"] = len(paths)
        uid_col = str(bundle.get("uid_column", "uid"))
        pred_col = str(bundle.get("prediction_column", "is_pathologic"))
        uids = [f"phase65ar3_{i:05d}" for i in range(len(paths))]
        suffixes = [".nii.gz" if p.name.lower().endswith(".nii.gz") else ".nii" for p in paths]

        temporary = Path(tempfile.mkdtemp(prefix="phase65ar3_", dir=str(PHASE65AR3_ROOT)))
        official_real = temporary / "official"
        niftis = official_real / "data" / "niftis"
        submission = temporary / "submission"
        hook_i, hook_c = temporary / "hook_i", temporary / "hook_c"
        trace = temporary / "trace.jsonl"
        niftis.mkdir(parents=True)
        submission.mkdir()
        hook_i.mkdir()
        hook_c.mkdir()
        for uid, source, suffix in zip(uids, paths, suffixes):
            os.symlink(str(source.resolve()), str(niftis / f"{uid}{suffix}"))
        with (official_real / "data" / "submission_format.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=[uid_col, pred_col])
            writer.writeheader()
            for uid in uids:
                writer.writerow({uid_col: uid, pred_col: 0.5})
        phase65ar3_extract_safe(PHASE65AR3_SUBMISSION_ZIP, submission)
        if not (submission / "main.py").is_file():
            raise ValueError("root_main_missing_after_extract")
        if PHASE65AR3_OFFICIAL_ROOT.exists() or PHASE65AR3_OFFICIAL_ROOT.is_symlink():
            raise FileExistsError("official_root_already_exists_refuse_overwrite")
        PHASE65AR3_OFFICIAL_ROOT.parent.mkdir(parents=True, exist_ok=True)
        os.symlink(str(official_real), str(PHASE65AR3_OFFICIAL_ROOT))
        created_link = True
        phase65ar3_write_sitecustomize(hook_i, True)
        phase65ar3_write_sitecustomize(hook_c, False)

        base_env = os.environ.copy()
        base_env.update({
            "PYTHONNOUSERSITE": "1", "TOKENIZERS_PARALLELISM": "false",
            "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
            "TORCH_HOME": str(submission / "torch_cache"),
            "PHASE65AR3_OFFICIAL_ROOT": str(PHASE65AR3_OFFICIAL_ROOT),
            "PHASE65AR3_SUBMISSION_ROOT": str(submission),
        })

        def execute(hook: Path, trace_enabled: bool):
            output = official_real / "submission.csv"
            if output.exists():
                output.unlink()
            env = base_env.copy()
            python_path_parts = [str(hook), str(submission)]
            if env.get("PYTHONPATH"):
                python_path_parts.append(env["PYTHONPATH"])
            env["PYTHONPATH"] = os.pathsep.join(python_path_parts)
            if trace_enabled:
                env["PHASE65AR3_TRACE_FILE"] = str(trace)
            else:
                env.pop("PHASE65AR3_TRACE_FILE", None)
            start = time.time()
            proc = subprocess.run(
                [sys.executable, "main.py"], cwd=str(submission), env=env,
                stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                timeout=PHASE65AR3_CONFIG["subprocess_timeout_seconds"], check=False,
            )
            private_log = (proc.stdout + proc.stderr)[:PHASE65AR3_CONFIG["maximum_log_capture_bytes"]]
            return int(proc.returncode), float(time.time() - start), output, private_log

        rc_i, sec_i, out_i, log_i = execute(hook_i, True)
        report["instrumented_return_code"], report["instrumented_seconds"] = rc_i, sec_i
        observation = phase65ar3_read_trace(trace)
        report["runtime_observation"] = observation
        if rc_i != 0 or not out_i.is_file():
            raise RuntimeError("instrumented_runtime_failed")
        pred_i = phase65ar3_load_output(out_i, uid_col, pred_col, uids)
        report["instrumented_output_valid"] = True

        rc_c, sec_c, out_c, log_c = execute(hook_c, False)
        report["clean_return_code"], report["clean_seconds"] = rc_c, sec_c
        if rc_c != 0 or not out_c.is_file():
            raise RuntimeError("clean_runtime_failed")
        pred_c = phase65ar3_load_output(out_c, uid_col, pred_col, uids)
        report["clean_output_valid"] = True
        report["executed"] = True
        finite_range = bool(np.isfinite(pred_i).all() and np.isfinite(pred_c).all()
                            and ((pred_i >= 0) & (pred_i <= 1)).all()
                            and ((pred_c >= 0) & (pred_c <= 1)).all())
        diff = np.abs(pred_i - pred_c)
        report["finite_and_in_range"] = finite_range
        report["instrumented_vs_clean_max_abs"] = float(diff.max())
        report["instrumentation_invariant"] = bool(finite_range and np.allclose(
            pred_i, pred_c, atol=PHASE65AR3_CONFIG["probability_atol"],
            rtol=PHASE65AR3_CONFIG["probability_rtol"]))
        if ref is not None:
            diff_ref = np.abs(ref - pred_c)
            report["reference_vs_clean_max_abs"] = float(diff_ref.max())
            report["reference_vs_clean_mean_abs"] = float(diff_ref.mean())
            report["exact_reference_parity"] = bool(finite_range and np.allclose(
                ref, pred_c, atol=PHASE65AR3_CONFIG["probability_atol"],
                rtol=PHASE65AR3_CONFIG["probability_rtol"]))

        meta = bundle.get("runtime_metadata")
        if isinstance(meta, dict):
            if "unreadable_count" in meta:
                report["unreadable_count"] = int(meta["unreadable_count"])
            checks = meta.get("preprocessing_checks")
            if isinstance(checks, dict):
                report["preprocessing_checks_checked"] = True
                report["preprocessing_checks_all_true"] = bool(checks and all(bool(v) for v in checks.values()))
            expected_branches = bundle.get("reference_branch_counts")
            if isinstance(expected_branches, dict) and isinstance(meta.get("branch_counts"), dict):
                report["branch_counts_checked"] = True
                report["branch_counts_match"] = dict(expected_branches) == dict(meta["branch_counts"])
        return report, {"instrumented_probability": pred_i, "clean_probability": pred_c,
                        "reference_probability": ref, "private_logs": (log_i, log_c)}
    except Exception as exc:
        report["error_type"] = type(exc).__name__
        known = {
            "invalid_rehearsal_paths_or_minimum_cases", "invalid_reference_probability",
            "root_main_missing_after_extract", "official_root_already_exists_refuse_overwrite",
            "instrumented_runtime_failed", "clean_runtime_failed",
            "output_schema_or_row_count_mismatch", "output_uid_set_mismatch",
            "unsafe_zip_member", "zip_symlink_member",
        }
        report["error_stage"] = str(exc) if str(exc) in known else "unexpected_private_rehearsal_failure"
        return report, bundle
    finally:
        if created_link:
            try:
                if PHASE65AR3_OFFICIAL_ROOT.is_symlink() and Path(os.readlink(PHASE65AR3_OFFICIAL_ROOT)) == official_real:
                    PHASE65AR3_OFFICIAL_ROOT.unlink()
            except Exception:
                pass
        if temporary is not None:
            shutil.rmtree(temporary, ignore_errors=True)


def phase65ar3_jsonable(value):
    if isinstance(value, dict):
        return {str(k): phase65ar3_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [phase65ar3_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# ------------------------------- audit -------------------------------

phase65ar3_zip_private = phase65ar3_zip_audit(PHASE65AR3_SUBMISSION_ZIP)
phase65ar3_final_private = phase65ar3_safe_json(PHASE65AR3_FINAL_CONTRACT)
phase65ar3_r2_private = phase65ar3_safe_json(PHASE65AR3_R2_CONTRACT)
phase65ar3_static_hub = phase65ar3_static_hub_calls(phase65ar3_zip_private["python_sources"])

archive_integrity = bool(
    phase65ar3_zip_private["valid"] and phase65ar3_zip_private["crc_ok"]
    and phase65ar3_zip_private["root_main"] and phase65ar3_zip_private["unsafe_count"] == 0
    and phase65ar3_zip_private["symlink_count"] == 0 and phase65ar3_zip_private["duplicate_count"] == 0
)
final_digest_link = bool(
    phase65ar3_final_private["valid_object"]
    and phase65ar3_final_private["status"] == "accepted"
    and phase65ar3_mentions_digest(phase65ar3_final_private["payload"], phase65ar3_zip_private["sha256"])
)
r2_lineage_passed = bool(
    phase65ar3_r2_private["valid_object"]
    and isinstance(phase65ar3_r2_private["payload"].get("artifact_provenance"), dict)
    and phase65ar3_r2_private["payload"]["artifact_provenance"].get("authoritative_final_archive_lineage_established") is True
    and phase65ar3_mentions_digest(phase65ar3_r2_private["payload"], phase65ar3_zip_private["sha256"])
)

phase65ar3_rehearsal_report, phase65ar3_rehearsal_private = phase65ar3_rehearsal()
obs = phase65ar3_rehearsal_report.get("runtime_observation")
hub_observed = bool(obs and obs["torch_hub_load_call_count"] > 0)
hub_local = bool(obs and obs["all_observed_hub_calls_source_local"]
                 and obs["all_observed_hub_repositories_exist"]
                 and obs["all_observed_hub_repositories_inside_submission"])
network_clean = bool(obs and obs["socket_network_attempt_count"] == 0
                     and obs["url_loader_call_count"] == 0)
checkpoint_loads_local = bool(obs and obs["torch_load_missing_path_count"] == 0
                              and obs["torch_load_outside_submission_count"] == 0)
paths_clean = bool(obs and all(obs["official_path_observed"].values()))
instrumentation_clean = bool(obs and obs["instrumentation_setup_failure_count"] == 0)
offline_pass = bool(
    archive_integrity and final_digest_link and r2_lineage_passed
    and phase65ar3_rehearsal_report["executed"]
    and phase65ar3_rehearsal_report["finite_and_in_range"]
    and phase65ar3_rehearsal_report["instrumentation_invariant"]
    and hub_observed and hub_local and network_clean and checkpoint_loads_local
    and paths_clean and instrumentation_clean
    and phase65ar3_rehearsal_report["unreadable_count"] in (None, 0)
    and phase65ar3_rehearsal_report["branch_counts_match"] is not False
    and phase65ar3_rehearsal_report["preprocessing_checks_all_true"] is not False
)
full_numeric_pass = bool(
    offline_pass and phase65ar3_rehearsal_report["reference_probability_available"]
    and phase65ar3_rehearsal_report["exact_reference_parity"]
)

if not (archive_integrity and final_digest_link and r2_lineage_passed):
    status = "phase56_archive_or_r2_lineage_changed_stop"
    recommendation = "Stop. Preserve Phase56 and resolve the failed integrity/digest lineage criterion."
elif not phase65ar3_rehearsal_report["configured"]:
    status = "static_torch_hub_semantics_recorded_private_offline_rehearsal_required"
    recommendation = "Configure 8–16 fixed private development scans and rerun. No exact reference vector is required for the offline gate."
elif not phase65ar3_rehearsal_report["valid_configuration"]:
    status = "private_offline_rehearsal_configuration_invalid_stop"
    recommendation = "Correct only the private rehearsal bundle; do not alter Phase56."
elif not phase65ar3_rehearsal_report["executed"]:
    status = "private_offline_rehearsal_execution_failed_stop"
    recommendation = "Inspect only PHASE65AR3_STATE_PRIVATE logs locally; do not export them or rebuild Phase56 blindly."
elif not hub_observed:
    status = "offline_rehearsal_completed_torch_hub_branch_not_observed_inconclusive"
    recommendation = "Use a rehearsal set that exercises the Phase30/DINO branch; archive modification is not justified."
elif not offline_pass:
    status = "local_torch_hub_or_offline_runtime_gate_failed_stop"
    recommendation = "Stop. Diagnose the failed local-repository, network, path, instrumentation, or output-invariance criterion before Phase65B."
elif not phase65ar3_rehearsal_report["reference_probability_available"]:
    status = "accepted_offline_local_torch_hub_and_runtime_path_rehearsal_numeric_parity_not_available"
    recommendation = "Offline deployment integrity passed. Exact development parity remains unproven; generate an exact Phase56 reference only if available without reopening model selection."
elif not full_numeric_pass:
    status = "offline_runtime_passed_exact_development_probability_parity_failed_stop"
    recommendation = "Stop. Offline execution is sound, but deployment predictions differ from the exact Phase56 development path."
else:
    status = "accepted_offline_and_exact_phase56_private_replay_ready_for_corrected_phase65b"
    recommendation = "Phase56 offline and numerical parity passed. Proceed to corrected fold-local Phase65B while retaining Phase56 as deployment anchor."

phase65ar3_report = {
    "phase": "phase65ar3_local_torch_hub_semantics_and_offline_phase56_rehearsal",
    "status": status,
    "archive_continuity": {
        "submission_sha256": phase65ar3_zip_private["sha256"],
        "file_count": phase65ar3_zip_private["file_count"],
        "archive_integrity_passed": archive_integrity,
        "accepted_final_contract_exact_digest_link": final_digest_link,
        "phase65ar2_authoritative_lineage_for_same_digest": r2_lineage_passed,
        "phase56_artifacts_modified": False,
    },
    "static_torch_hub_semantics": phase65ar3_static_hub,
    "private_offline_rehearsal": phase65ar3_rehearsal_report,
    "gates": {
        "archive_continuity": bool(archive_integrity and final_digest_link and r2_lineage_passed),
        "torch_hub_call_observed_runtime": hub_observed if phase65ar3_rehearsal_report["executed"] else None,
        "all_observed_hub_calls_local_existing_bundled_repository": hub_local if hub_observed else None,
        "zero_socket_or_url_loader_network_attempts": network_clean if phase65ar3_rehearsal_report["executed"] else None,
        "all_observed_torch_checkpoint_paths_exist_inside_submission": checkpoint_loads_local if phase65ar3_rehearsal_report["executed"] else None,
        "all_official_runtime_paths_observed": paths_clean if phase65ar3_rehearsal_report["executed"] else None,
        "instrumentation_setup_clean": instrumentation_clean if phase65ar3_rehearsal_report["executed"] else None,
        "instrumented_clean_probability_invariance": phase65ar3_rehearsal_report["instrumentation_invariant"],
        "offline_deployment_integrity_passed": offline_pass,
        "exact_development_probability_parity": phase65ar3_rehearsal_report["exact_reference_parity"],
        "full_numeric_parity_passed": full_numeric_pass,
    },
    "recommendation": recommendation,
    "interpretation_contract": {
        "torch_hub_load_is_not_a_network_call_when_source_local_and_repo_bundled": True,
        "static_network_capability_is_not_runtime_network_use": True,
        "offline_gate_can_pass_without_an_exact_development_reference": True,
        "offline_gate_does_not_prove_development_to_deployment_numeric_parity": True,
        "phase12c_or_phase42_predictions_must_not_substitute_for_phase56_reference": True,
        "public_leaderboard_used_for_gate": False,
        "phase56_remains_deployment_anchor": True,
    },
    "compliance": {
        "training_performed": False,
        "labels_read": False,
        "test_data_read": False,
        "test_time_adaptation": False,
        "private_development_scans_read_only_if_rehearsal_configured": bool(phase65ar3_rehearsal_report["configured"]),
        "case_level_predictions_exported": False,
        "private_paths_or_uids_exported": False,
        "source_lines_exported": False,
        "subprocess_logs_exported": False,
        "independent_test_case_inference_preserved": True,
    },
    "elapsed_seconds": float(time.time() - PHASE65AR3_STARTED),
}

# Private live-kernel state only. Never serialize or print its raw contents.
PHASE65AR3_STATE_PRIVATE = {
    "report": phase65ar3_report,
    "rehearsal": phase65ar3_rehearsal_private,
    "configured_bundle": globals().get("PHASE65AR3_REHEARSAL_PRIVATE"),
    "submission_sources": phase65ar3_zip_private["python_sources"],
}
PHASE65AR3_REPORT_PRIVATE = phase65ar3_report

payload = phase65ar3_jsonable(phase65ar3_report)
raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
payload["contract_sha256"] = phase65ar3_sha_bytes(raw)
payload["contract_file"] = str(PHASE65AR3_CONTRACT_PATH)
PHASE65AR3_CONTRACT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")

print("BEGIN SANITIZED_PHASE65AR3_LOCAL_TORCH_HUB_OFFLINE_REHEARSAL")
print(json.dumps(payload, indent=2))
print("END SANITIZED_PHASE65AR3_LOCAL_TORCH_HUB_OFFLINE_REHEARSAL")
