"""
Cell 165A-R — Phase56 local runtime and preprocessing parity audit

Run in the Kaggle development notebook. This cell is read-only with respect to
Phase56 artifacts. It never imports or executes the submitted main.py, never
reads challenge NIfTI files, and never reads test data.

Optional exact prediction-parity hook (private, in memory only):

PHASE65AR_DYNAMIC_PARITY_PRIVATE = {
    "reference_probability": <1-D array produced by the development path>,
    "runtime_probability": <1-D array produced by a safe local Phase56 replay>,
    "router_branch": <optional 1-D branch-id array>,
    "unreadable_count": <optional integer>,
    "preprocessing_checks": <optional dict[str, bool]>,
}

The arrays are compared but never printed or written. Without this hook the
cell can pass the static archive/source audit, but it will not claim exact
prediction parity.
"""

from __future__ import annotations

import ast
import hashlib
import json
import math
import os
import stat
import time
import zipfile
from collections import Counter
from pathlib import Path, PurePosixPath

import numpy as np


PHASE65AR_STARTED = time.time()
PHASE65AR_ROOT = Path("/kaggle/working")
PHASE65AR_CONTRACT_PATH = PHASE65AR_ROOT / "phase65ar_phase56_runtime_parity_contract.json"

PHASE65AR_PATHS = {
    "submission_zip": PHASE65AR_ROOT / "phase56_submission.zip",
    "source_handoff_zip": PHASE65AR_ROOT / "phase56_runtime_source_handoff.zip",
    "staged_submission": PHASE65AR_ROOT / "phase56_experimental_submission",
    "private_checkpoint": PHASE65AR_ROOT / "phase56_private_checkpoint",
    "archive_build_contract": PHASE65AR_ROOT / "phase56_archive_build_contract.json",
    "final_archive_contract": PHASE65AR_ROOT / "phase56_final_archive_contract.json",
    "real_smoke_contract": PHASE65AR_ROOT / "phase56_real_smoke_contract.json",
    "phase65a_contract": PHASE65AR_ROOT / "phase65a_corrected_deployment_fallback_contract.json",
}

PHASE65AR_CONFIG = {
    "exact_probability_atol": 1.0e-9,
    "exact_probability_rtol": 0.0,
    "maximum_reported_examples": 5,
    "full_zip_crc_test": True,
    "leaderboard_used_for_gate": False,
}


def phase65ar_sha256_file(path: Path, block_size: int = 8 * 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            block = f.read(block_size)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def phase65ar_sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase65ar_safe_json(path: Path) -> dict:
    result = {
        "exists": path.is_file(),
        "valid_json": False,
        "root_is_object": False,
        "status": None,
        "sha256": None,
        "payload": None,
        "error_type": None,
    }
    if not result["exists"]:
        return result
    try:
        raw = path.read_bytes()
        result["sha256"] = phase65ar_sha256_bytes(raw)
        obj = json.loads(raw.decode("utf-8"))
        result["valid_json"] = True
        result["root_is_object"] = isinstance(obj, dict)
        if isinstance(obj, dict):
            result["payload"] = obj
            result["status"] = obj.get("status")
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    return result


def phase65ar_json_scalars(obj, key_hint: str = ""):
    """Yield (key_path, scalar) without exposing whole contract structures."""
    if isinstance(obj, dict):
        for key, value in obj.items():
            new_key = f"{key_hint}.{key}" if key_hint else str(key)
            yield from phase65ar_json_scalars(value, new_key)
    elif isinstance(obj, list):
        for i, value in enumerate(obj):
            yield from phase65ar_json_scalars(value, f"{key_hint}[{i}]")
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        yield key_hint, obj


def phase65ar_contract_mentions_digest(contract_payload: dict | None, digest: str) -> bool:
    if not isinstance(contract_payload, dict):
        return False
    digest = digest.lower()
    for key, value in phase65ar_json_scalars(contract_payload):
        if "sha" in key.lower() and isinstance(value, str) and value.lower() == digest:
            return True
    return False


def phase65ar_safe_zip_name(name: str) -> bool:
    p = PurePosixPath(name)
    return (
        bool(name)
        and "\\" not in name
        and not name.startswith("/")
        and not p.is_absolute()
        and ".." not in p.parts
    )


def phase65ar_zip_inventory(path: Path, require_root_main: bool) -> dict:
    out = {
        "exists": path.is_file(),
        "valid_zip": False,
        "sha256": None,
        "compressed_bytes": None,
        "member_count": 0,
        "file_count": 0,
        "total_uncompressed_bytes": 0,
        "unsafe_member_count": 0,
        "duplicate_member_count": 0,
        "symlink_member_count": 0,
        "root_main_present": False,
        "main_count": 0,
        "selected_main": None,
        "crc_tested": False,
        "crc_ok": None,
        "error_type": None,
        "normalized_sizes": {},
        "python_hashes": {},
        "python_sources": {},
    }
    if not out["exists"]:
        return out
    try:
        out["sha256"] = phase65ar_sha256_file(path)
        out["compressed_bytes"] = int(path.stat().st_size)
        with zipfile.ZipFile(path, "r") as zf:
            infos = zf.infolist()
            out["valid_zip"] = True
            out["member_count"] = len(infos)
            names = [i.filename for i in infos]
            out["duplicate_member_count"] = len(names) - len(set(names))
            out["unsafe_member_count"] = sum(not phase65ar_safe_zip_name(n) for n in names)
            out["symlink_member_count"] = sum(
                stat.S_ISLNK((i.external_attr >> 16) & 0xFFFF) for i in infos
            )
            files = [i for i in infos if not i.is_dir()]
            out["file_count"] = len(files)
            out["total_uncompressed_bytes"] = int(sum(i.file_size for i in files))
            main_names = [i.filename for i in files if PurePosixPath(i.filename).name == "main.py"]
            out["main_count"] = len(main_names)
            out["root_main_present"] = "main.py" in names
            if require_root_main:
                selected_main = "main.py" if "main.py" in names else None
            else:
                selected_main = min(main_names, key=lambda n: (len(PurePosixPath(n).parts), n)) if main_names else None
            out["selected_main"] = selected_main

            anchor = PurePosixPath(selected_main).parent if selected_main else None
            if anchor is not None:
                for info in files:
                    p = PurePosixPath(info.filename)
                    try:
                        rel = p.relative_to(anchor)
                    except ValueError:
                        continue
                    rel_text = rel.as_posix()
                    out["normalized_sizes"][rel_text] = int(info.file_size)
                    if rel.suffix.lower() == ".py":
                        data = zf.read(info)
                        out["python_hashes"][rel_text] = phase65ar_sha256_bytes(data)
                        try:
                            out["python_sources"][rel_text] = data.decode("utf-8")
                        except UnicodeDecodeError:
                            out["python_sources"][rel_text] = ""

            if PHASE65AR_CONFIG["full_zip_crc_test"]:
                out["crc_tested"] = True
                out["crc_ok"] = zf.testzip() is None
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar_directory_inventory(path: Path) -> dict:
    out = {
        "exists": path.is_dir(),
        "file_count": 0,
        "total_bytes": 0,
        "main_count": 0,
        "selected_main": None,
        "normalized_sizes": {},
        "python_hashes": {},
        "python_sources": {},
        "error_type": None,
    }
    if not out["exists"]:
        return out
    try:
        files = [p for p in path.rglob("*") if p.is_file()]
        out["file_count"] = len(files)
        out["total_bytes"] = int(sum(p.stat().st_size for p in files))
        mains = [p for p in files if p.name == "main.py"]
        out["main_count"] = len(mains)
        selected = min(mains, key=lambda p: (len(p.relative_to(path).parts), str(p))) if mains else None
        out["selected_main"] = str(selected.relative_to(path)) if selected else None
        if selected is not None:
            anchor = selected.parent
            for file_path in files:
                try:
                    rel = file_path.relative_to(anchor)
                except ValueError:
                    continue
                rel_text = rel.as_posix()
                out["normalized_sizes"][rel_text] = int(file_path.stat().st_size)
                if rel.suffix.lower() == ".py":
                    data = file_path.read_bytes()
                    out["python_hashes"][rel_text] = phase65ar_sha256_bytes(data)
                    try:
                        out["python_sources"][rel_text] = data.decode("utf-8")
                    except UnicodeDecodeError:
                        out["python_sources"][rel_text] = ""
    except Exception as exc:
        out["error_type"] = type(exc).__name__
    return out


def phase65ar_compare_trees(left: dict, right: dict) -> dict:
    left_py = left.get("python_hashes", {})
    right_py = right.get("python_hashes", {})
    shared_py = sorted(set(left_py) & set(right_py))
    mismatched_py = [name for name in shared_py if left_py[name] != right_py[name]]
    left_sizes = left.get("normalized_sizes", {})
    right_sizes = right.get("normalized_sizes", {})
    shared_all = sorted(set(left_sizes) & set(right_sizes))
    size_mismatches = [name for name in shared_all if left_sizes[name] != right_sizes[name]]
    return {
        "left_python_count": len(left_py),
        "right_python_count": len(right_py),
        "shared_python_count": len(shared_py),
        "python_hash_mismatch_count": len(mismatched_py),
        "main_python_hash_match": (
            "main.py" in left_py and "main.py" in right_py and left_py["main.py"] == right_py["main.py"]
        ),
        "left_only_python_count": len(set(left_py) - set(right_py)),
        "right_only_python_count": len(set(right_py) - set(left_py)),
        "shared_payload_file_count": len(shared_all),
        "shared_payload_size_mismatch_count": len(size_mismatches),
        "left_payload_file_count": len(left_sizes),
        "right_payload_file_count": len(right_sizes),
    }


def phase65ar_source_audit(source_map: dict[str, str]) -> dict:
    marker_groups = {
        "nifti_reader": ("nibabel", "nib.load", ".nii.gz", "nifti"),
        "orientation_canonicalization": ("as_closest_canonical", "canonical", "orientation", "aff2axcodes"),
        "physical_spacing_or_affine": ("spacing", "voxel", "affine", "resample"),
        "percentile_or_case_normalization": ("percentile", "quantile", "p99", "normaliz"),
        "protocol_router": ("protocol", "router", "prototype", "header"),
        "phase12_fallback": ("phase12", "fallback", "unknown"),
        "sparse_adapter": ("adapter", "residual"),
        "probability_clip": ("clip", "1e-5", "0.00001"),
        "runtime_input": ("/code_execution/data/niftis", "data/niftis"),
        "submission_format": ("submission_format.csv",),
        "runtime_output": ("/code_execution/submission.csv", "submission.csv"),
    }
    combined = "\n".join(source_map.values())
    lower = combined.lower()
    markers = {
        name: any(token.lower() in lower for token in tokens)
        for name, tokens in marker_groups.items()
    }
    compile_failures = []
    syntax_tree_count = 0
    network_import_count = 0
    network_literal_count = 0
    training_call_count = 0
    forbidden_kaggle_path_count = lower.count("/kaggle/working")
    training_names = {"fit", "fit_transform", "partial_fit", "backward"}
    network_roots = {"requests", "urllib", "httpx", "socket", "wget", "curl"}
    for rel, text in sorted(source_map.items()):
        try:
            tree = ast.parse(text, filename=rel)
            compile(text, rel, "exec")
            syntax_tree_count += 1
        except Exception as exc:
            compile_failures.append(type(exc).__name__)
            continue
        for node in ast.walk(tree):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif node.module:
                    names = [node.module.split(".")[0]]
                network_import_count += sum(name in network_roots for name in names)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                value = node.value.lower()
                network_literal_count += int("http://" in value or "https://" in value)
            elif isinstance(node, ast.Call):
                func = node.func
                call_name = None
                if isinstance(func, ast.Name):
                    call_name = func.id
                elif isinstance(func, ast.Attribute):
                    call_name = func.attr
                if call_name in training_names:
                    training_call_count += 1
    return {
        "python_file_count": len(source_map),
        "compiled_python_file_count": syntax_tree_count,
        "compile_failure_count": len(compile_failures),
        "marker_presence": markers,
        "forbidden_kaggle_working_literal_count": forbidden_kaggle_path_count,
        "network_import_count": network_import_count,
        "network_url_literal_count": network_literal_count,
        "training_call_count_diagnostic": training_call_count,
        "training_call_static_scan_is_not_reachability_proof": True,
    }


def phase65ar_private_inventory(path: Path) -> dict:
    out = {
        "exists": path.is_dir(),
        "file_count": 0,
        "total_bytes": 0,
        "possible_replay_artifact_count": 0,
        "contents_read": False,
    }
    if not out["exists"]:
        return out
    files = [p for p in path.rglob("*") if p.is_file()]
    out["file_count"] = len(files)
    out["total_bytes"] = int(sum(p.stat().st_size for p in files))
    hints = ("parity", "rehears", "reference", "prediction", "probability", "oof")
    out["possible_replay_artifact_count"] = sum(any(h in p.name.lower() for h in hints) for p in files)
    return out


def phase65ar_dynamic_parity() -> tuple[dict, dict | None]:
    bundle = globals().get("PHASE65AR_DYNAMIC_PARITY_PRIVATE")
    report = {
        "available": isinstance(bundle, dict),
        "valid_structure": False,
        "n": None,
        "finite": None,
        "within_probability_range": None,
        "maximum_absolute_difference": None,
        "mean_absolute_difference": None,
        "exact_within_tolerance": None,
        "branch_count": None,
        "unreadable_count": None,
        "preprocessing_check_count": None,
        "preprocessing_all_true": None,
        "arrays_exported": False,
    }
    if not isinstance(bundle, dict):
        return report, None
    try:
        ref = np.asarray(bundle["reference_probability"], dtype=np.float64).reshape(-1)
        run = np.asarray(bundle["runtime_probability"], dtype=np.float64).reshape(-1)
        valid = ref.shape == run.shape and ref.size > 0
        report["valid_structure"] = bool(valid)
        if not valid:
            return report, bundle
        finite = bool(np.isfinite(ref).all() and np.isfinite(run).all())
        in_range = bool(((ref >= 0.0) & (ref <= 1.0)).all() and ((run >= 0.0) & (run <= 1.0)).all())
        diff = np.abs(ref - run)
        report.update({
            "n": int(ref.size),
            "finite": finite,
            "within_probability_range": in_range,
            "maximum_absolute_difference": float(np.max(diff)),
            "mean_absolute_difference": float(np.mean(diff)),
            "exact_within_tolerance": bool(
                finite and in_range and np.allclose(
                    ref,
                    run,
                    atol=PHASE65AR_CONFIG["exact_probability_atol"],
                    rtol=PHASE65AR_CONFIG["exact_probability_rtol"],
                )
            ),
        })
        if "router_branch" in bundle:
            branch = np.asarray(bundle["router_branch"]).reshape(-1)
            if branch.size == ref.size:
                report["branch_count"] = int(np.unique(branch).size)
        if "unreadable_count" in bundle:
            report["unreadable_count"] = int(bundle["unreadable_count"])
        checks = bundle.get("preprocessing_checks")
        if isinstance(checks, dict):
            values = [bool(v) for v in checks.values()]
            report["preprocessing_check_count"] = len(values)
            report["preprocessing_all_true"] = bool(values and all(values))
    except Exception:
        report["valid_structure"] = False
    return report, bundle


def phase65ar_jsonable(value):
    if isinstance(value, dict):
        return {str(k): phase65ar_jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [phase65ar_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------- read-only audit ---------------------------

phase65ar_contracts_private = {
    name: phase65ar_safe_json(PHASE65AR_PATHS[name])
    for name in (
        "archive_build_contract",
        "final_archive_contract",
        "real_smoke_contract",
        "phase65a_contract",
    )
}

phase65ar_submission_private = phase65ar_zip_inventory(
    PHASE65AR_PATHS["submission_zip"], require_root_main=True
)
phase65ar_handoff_private = phase65ar_zip_inventory(
    PHASE65AR_PATHS["source_handoff_zip"], require_root_main=False
)
phase65ar_stage_private = phase65ar_directory_inventory(PHASE65AR_PATHS["staged_submission"])
phase65ar_checkpoint_inventory = phase65ar_private_inventory(PHASE65AR_PATHS["private_checkpoint"])

phase65ar_zip_handoff_compare = phase65ar_compare_trees(
    phase65ar_submission_private, phase65ar_handoff_private
)
phase65ar_zip_stage_compare = phase65ar_compare_trees(
    phase65ar_submission_private, phase65ar_stage_private
)
phase65ar_source_audit_private = phase65ar_source_audit(
    phase65ar_submission_private.get("python_sources", {})
)
phase65ar_dynamic_report, phase65ar_dynamic_bundle_private = phase65ar_dynamic_parity()

phase65ar_submission_sha = phase65ar_submission_private.get("sha256")
phase65ar_sha_recorded = {
    name: phase65ar_contract_mentions_digest(rec.get("payload"), phase65ar_submission_sha)
    if phase65ar_submission_sha else False
    for name, rec in phase65ar_contracts_private.items()
    if name != "phase65a_contract"
}

phase65ar_contract_summary = {
    name: {
        "exists": rec["exists"],
        "valid_json_object": bool(rec["valid_json"] and rec["root_is_object"]),
        "status": rec["status"],
        "sha256": rec["sha256"],
    }
    for name, rec in phase65ar_contracts_private.items()
}

phase65ar_static_criteria = {
    "all_required_artifacts_exist": all(
        PHASE65AR_PATHS[name].exists()
        for name in (
            "submission_zip",
            "source_handoff_zip",
            "staged_submission",
            "private_checkpoint",
            "archive_build_contract",
            "final_archive_contract",
            "real_smoke_contract",
            "phase65a_contract",
        )
    ),
    "all_contracts_are_json_objects": all(
        rec["valid_json"] and rec["root_is_object"]
        for rec in phase65ar_contracts_private.values()
    ),
    "submission_zip_valid": bool(phase65ar_submission_private["valid_zip"]),
    "submission_crc_clean": bool(phase65ar_submission_private["crc_ok"]),
    "submission_paths_safe": (
        phase65ar_submission_private["unsafe_member_count"] == 0
        and phase65ar_submission_private["duplicate_member_count"] == 0
        and phase65ar_submission_private["symlink_member_count"] == 0
    ),
    "main_py_at_submission_root": bool(phase65ar_submission_private["root_main_present"]),
    "handoff_zip_valid": bool(phase65ar_handoff_private["valid_zip"]),
    "handoff_crc_clean": bool(phase65ar_handoff_private["crc_ok"]),
    "handoff_paths_safe": (
        phase65ar_handoff_private["unsafe_member_count"] == 0
        and phase65ar_handoff_private["duplicate_member_count"] == 0
        and phase65ar_handoff_private["symlink_member_count"] == 0
    ),
    "submission_handoff_main_source_identical": bool(
        phase65ar_zip_handoff_compare["main_python_hash_match"]
    ),
    "submission_handoff_shared_python_identical": (
        phase65ar_zip_handoff_compare["shared_python_count"] > 0
        and phase65ar_zip_handoff_compare["python_hash_mismatch_count"] == 0
    ),
    "staged_main_source_identical": bool(
        phase65ar_zip_stage_compare["main_python_hash_match"]
    ),
    "runtime_python_compiles": (
        phase65ar_source_audit_private["python_file_count"] > 0
        and phase65ar_source_audit_private["compile_failure_count"] == 0
    ),
    "no_kaggle_working_runtime_dependency": (
        phase65ar_source_audit_private["forbidden_kaggle_working_literal_count"] == 0
    ),
    "no_static_network_dependency": (
        phase65ar_source_audit_private["network_import_count"] == 0
    ),
    "runtime_io_contract_markers_present": all(
        phase65ar_source_audit_private["marker_presence"][key]
        for key in ("runtime_input", "submission_format", "runtime_output")
    ),
}

phase65ar_static_passed = bool(all(phase65ar_static_criteria.values()))
phase65ar_dynamic_passed = bool(
    phase65ar_dynamic_report["available"]
    and phase65ar_dynamic_report["valid_structure"]
    and phase65ar_dynamic_report["exact_within_tolerance"]
    and (
        phase65ar_dynamic_report["preprocessing_all_true"] is not False
    )
)

if not phase65ar_static_passed:
    phase65ar_status = "static_runtime_or_archive_parity_failed_stop"
    phase65ar_recommendation = "Stop. Repair the failing static criteria before any Phase65B experiment or archive rebuild."
elif not phase65ar_dynamic_report["available"]:
    phase65ar_status = "static_archive_source_parity_passed_dynamic_prediction_parity_not_available"
    phase65ar_recommendation = (
        "Static parity passed. Keep Phase56 unchanged; exact development replay parity is still required "
        "before attributing the public gap to cohort shift or advancing Phase65B."
    )
elif not phase65ar_dynamic_passed:
    phase65ar_status = "dynamic_prediction_or_preprocessing_parity_failed_stop"
    phase65ar_recommendation = "Stop. Local replay does not reproduce the reference path within 1e-9."
else:
    phase65ar_status = "accepted_full_local_runtime_and_preprocessing_parity_ready_for_phase65b"
    phase65ar_recommendation = (
        "Archive, source, preprocessing checks, and exact probabilities agree. Phase65B may run as a new "
        "validation experiment; Phase56 remains the deployment anchor until a later gate passes."
    )

phase65ar_report = {
    "phase": "phase65ar_phase56_local_runtime_and_preprocessing_parity_audit",
    "status": phase65ar_status,
    "artifact_inventory": {
        "submission_zip": {
            "exists": phase65ar_submission_private["exists"],
            "sha256": phase65ar_submission_private["sha256"],
            "compressed_bytes": phase65ar_submission_private["compressed_bytes"],
            "file_count": phase65ar_submission_private["file_count"],
            "uncompressed_bytes": phase65ar_submission_private["total_uncompressed_bytes"],
            "root_main_present": phase65ar_submission_private["root_main_present"],
            "crc_ok": phase65ar_submission_private["crc_ok"],
            "unsafe_member_count": phase65ar_submission_private["unsafe_member_count"],
            "duplicate_member_count": phase65ar_submission_private["duplicate_member_count"],
            "symlink_member_count": phase65ar_submission_private["symlink_member_count"],
        },
        "runtime_source_handoff_zip": {
            "exists": phase65ar_handoff_private["exists"],
            "sha256": phase65ar_handoff_private["sha256"],
            "compressed_bytes": phase65ar_handoff_private["compressed_bytes"],
            "file_count": phase65ar_handoff_private["file_count"],
            "crc_ok": phase65ar_handoff_private["crc_ok"],
            "unsafe_member_count": phase65ar_handoff_private["unsafe_member_count"],
        },
        "staged_submission": {
            "exists": phase65ar_stage_private["exists"],
            "file_count": phase65ar_stage_private["file_count"],
            "total_bytes": phase65ar_stage_private["total_bytes"],
            "main_count": phase65ar_stage_private["main_count"],
        },
        "private_checkpoint": phase65ar_checkpoint_inventory,
    },
    "contract_integrity": {
        "contracts": phase65ar_contract_summary,
        "submission_sha256_recorded_by_contract": phase65ar_sha_recorded,
        "sha_record_is_diagnostic_not_gate": True,
        "non_object_json_causes_clean_gate_failure_not_assertion": True,
    },
    "source_parity": {
        "submission_vs_runtime_handoff": phase65ar_zip_handoff_compare,
        "submission_vs_staged_directory": phase65ar_zip_stage_compare,
    },
    "runtime_source_audit": phase65ar_source_audit_private,
    "dynamic_prediction_parity": phase65ar_dynamic_report,
    "static_gate": {
        "criteria": phase65ar_static_criteria,
        "passed": phase65ar_static_passed,
    },
    "full_local_parity_gate": {
        "static_passed": phase65ar_static_passed,
        "dynamic_bundle_available": phase65ar_dynamic_report["available"],
        "dynamic_passed": phase65ar_dynamic_passed,
        "full_parity_passed": bool(phase65ar_static_passed and phase65ar_dynamic_passed),
        "exact_probability_atol": PHASE65AR_CONFIG["exact_probability_atol"],
        "exact_probability_rtol": PHASE65AR_CONFIG["exact_probability_rtol"],
    },
    "recommendation": phase65ar_recommendation,
    "interpretation_contract": {
        "static_source_parity_is_not_prediction_parity": True,
        "runtime_main_py_imported_or_executed": False,
        "phase56_artifacts_modified": False,
        "public_leaderboard_used_for_gate": False,
        "phase65a_fallback_decision_changed": False,
        "phase56_remains_deployment_anchor": True,
        "full_parity_requires_private_local_replay": True,
    },
    "compliance": {
        "training_performed": False,
        "training_nifti_files_read": False,
        "training_voxel_cache_read": False,
        "test_data_read": False,
        "test_labels_read": False,
        "case_level_predictions_exported": False,
        "private_arrays_exported": False,
        "models_exported": False,
        "independent_test_case_inference_preserved": True,
    },
    "elapsed_seconds": float(time.time() - PHASE65AR_STARTED),
}

# Store private state only in the live kernel. Never serialize source or arrays.
PHASE65AR_RUNTIME_PARITY_STATE_PRIVATE = {
    "report": phase65ar_report,
    "submission_inventory": phase65ar_submission_private,
    "handoff_inventory": phase65ar_handoff_private,
    "stage_inventory": phase65ar_stage_private,
    "contracts": phase65ar_contracts_private,
    "dynamic_bundle": phase65ar_dynamic_bundle_private,
}
PHASE65AR_RUNTIME_PARITY_REPORT_PRIVATE = phase65ar_report

phase65ar_contract_payload = phase65ar_jsonable(phase65ar_report)
phase65ar_contract_bytes_without_hash = json.dumps(
    phase65ar_contract_payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
).encode("utf-8")
phase65ar_contract_payload["contract_sha256"] = phase65ar_sha256_bytes(
    phase65ar_contract_bytes_without_hash
)
phase65ar_contract_payload["contract_file"] = str(PHASE65AR_CONTRACT_PATH)
PHASE65AR_CONTRACT_PATH.write_text(
    json.dumps(phase65ar_contract_payload, indent=2, sort_keys=False) + "\n",
    encoding="utf-8",
)

# Compact, aggregate-only output.
print("BEGIN SANITIZED_PHASE65AR_PHASE56_RUNTIME_PARITY_AUDIT")
print(json.dumps(phase65ar_contract_payload, indent=2, sort_keys=False))
print("END SANITIZED_PHASE65AR_PHASE56_RUNTIME_PARITY_AUDIT")
