"""
Cell 165A-R4 — Standalone corrected Phase65A-R3 offline-gate adjudication

Purpose
-------
Phase65A-R3 can report ``submission_output=false`` even when both of its runs
successfully created and validated ``/code_execution/submission.csv``.  The
reason is that its diagnostic hook observes selected Python file APIs; an
atomic rename, a C-extension writer, or another unpatched path can bypass that
hook.  Exact-path file existence plus schema/UID/probability validation is the
authoritative output-path evidence.

This cell is restart-safe and has no dependency on notebook variables or prior
cells.  It reads only persistent artifacts from ``/kaggle/working`` (or the
private environment override ``PHASE65AR4_ROOT_PRIVATE``), re-verifies the
Phase56 ZIP and accepted final contract, validates the saved R3 contract hash,
and re-adjudicates the offline gate.  It does not rerun inference, read NIfTI
files or labels, inspect test data, change Phase56, or expose private values.

Required persistent inputs
--------------------------
* phase56_submission.zip
* phase56_final_archive_contract.json
* phase65ar3_local_torch_hub_offline_rehearsal_contract.json

Output
------
* phase65ar4_corrected_offline_gate_adjudication_contract.json
* one sanitized BEGIN/END JSON block
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import time
import zipfile
from pathlib import Path, PurePosixPath
from typing import Any


PHASE65AR4_STARTED = time.time()
PHASE65AR4_ROOT = Path(
    os.environ.get("PHASE65AR4_ROOT_PRIVATE", "/kaggle/working")
).resolve()
PHASE65AR4_SUBMISSION_ZIP = PHASE65AR4_ROOT / "phase56_submission.zip"
PHASE65AR4_FINAL_CONTRACT = (
    PHASE65AR4_ROOT / "phase56_final_archive_contract.json"
)
PHASE65AR4_R3_CONTRACT = (
    PHASE65AR4_ROOT
    / "phase65ar3_local_torch_hub_offline_rehearsal_contract.json"
)
PHASE65AR4_CONTRACT_PATH = (
    PHASE65AR4_ROOT
    / "phase65ar4_corrected_offline_gate_adjudication_contract.json"
)

PHASE65AR4_EXPECTED_R3_PHASE = (
    "phase65ar3_local_torch_hub_semantics_and_offline_phase56_rehearsal"
)
PHASE65AR4_ALLOWED_R3_FALSE_GATES = {
    "all_official_runtime_paths_observed",
    "offline_deployment_integrity_passed",
    "exact_development_probability_parity",
    "full_numeric_parity_passed",
}


def phase65ar4_sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def phase65ar4_sha_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def phase65ar4_safe_json(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "valid_json_object": False,
        "sha256": None,
        "status": None,
        "payload": None,
        "error_type": None,
    }
    if not result["exists"]:
        return result
    try:
        raw = path.read_bytes()
        payload = json.loads(raw.decode("utf-8"))
        result["sha256"] = phase65ar4_sha_bytes(raw)
        result["valid_json_object"] = isinstance(payload, dict)
        if isinstance(payload, dict):
            result["payload"] = payload
            result["status"] = payload.get("status")
    except Exception as exc:  # clean sanitized stop
        result["error_type"] = type(exc).__name__
    return result


def phase65ar4_scalars(value: Any):
    if isinstance(value, dict):
        for child in value.values():
            yield from phase65ar4_scalars(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from phase65ar4_scalars(child)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        yield value


def phase65ar4_mentions_digest(payload: Any, digest: str | None) -> bool:
    if not isinstance(payload, dict) or not digest:
        return False
    target = digest.lower()
    return any(
        isinstance(value, str) and value.lower() == target
        for value in phase65ar4_scalars(payload)
    )


def phase65ar4_safe_member(name: str) -> bool:
    member = PurePosixPath(name)
    return bool(name) and "\\" not in name and not member.is_absolute() and ".." not in member.parts


def phase65ar4_zip_audit(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "exists": path.is_file(),
        "sha256": None,
        "compressed_bytes": None,
        "file_count": 0,
        "valid_zip": False,
        "crc_ok": None,
        "safe_paths": None,
        "root_main_present": None,
        "duplicate_member_count": None,
        "symlink_member_count": None,
        "error_type": None,
    }
    if not result["exists"]:
        return result
    try:
        result["compressed_bytes"] = int(path.stat().st_size)
        result["sha256"] = phase65ar4_sha_file(path)
        with zipfile.ZipFile(path) as archive:
            infos = archive.infolist()
            names = [info.filename for info in infos]
            files = [info for info in infos if not info.is_dir()]
            result.update(
                {
                    "valid_zip": True,
                    "crc_ok": archive.testzip() is None,
                    "safe_paths": all(phase65ar4_safe_member(name) for name in names),
                    "root_main_present": "main.py" in names,
                    "file_count": len(files),
                    "duplicate_member_count": len(names) - len(set(names)),
                    "symlink_member_count": sum(
                        stat.S_ISLNK((info.external_attr >> 16) & 0xFFFF)
                        for info in infos
                    ),
                }
            )
    except Exception as exc:
        result["error_type"] = type(exc).__name__
    return result


def phase65ar4_r3_self_hash(payload: Any) -> dict[str, Any]:
    result = {
        "claimed_sha256": None,
        "recomputed_sha256": None,
        "matches": False,
        "hash_scheme": "canonical_json_before_contract_sha256_and_contract_file",
    }
    if not isinstance(payload, dict):
        return result
    claimed = payload.get("contract_sha256")
    body = dict(payload)
    body.pop("contract_sha256", None)
    body.pop("contract_file", None)
    raw = json.dumps(
        body,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    recomputed = phase65ar4_sha_bytes(raw)
    result.update(
        {
            "claimed_sha256": claimed if isinstance(claimed, str) else None,
            "recomputed_sha256": recomputed,
            "matches": isinstance(claimed, str) and claimed == recomputed,
        }
    )
    return result


def phase65ar4_bool(value: Any) -> bool:
    return value is True


def phase65ar4_zero(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value == 0


def phase65ar4_nonnegative_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and value >= 0


def phase65ar4_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): phase65ar4_jsonable(child) for key, child in value.items()}
    if isinstance(value, (list, tuple)):
        return [phase65ar4_jsonable(child) for child in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


# --------------------------- load and verify ---------------------------

phase65ar4_archive_before = phase65ar4_zip_audit(PHASE65AR4_SUBMISSION_ZIP)
phase65ar4_final_private = phase65ar4_safe_json(PHASE65AR4_FINAL_CONTRACT)
phase65ar4_r3_private = phase65ar4_safe_json(PHASE65AR4_R3_CONTRACT)
phase65ar4_r3 = (
    phase65ar4_r3_private["payload"]
    if isinstance(phase65ar4_r3_private.get("payload"), dict)
    else {}
)
phase65ar4_r3_hash = phase65ar4_r3_self_hash(phase65ar4_r3)

phase65ar4_archive_integrity = bool(
    phase65ar4_archive_before["valid_zip"]
    and phase65ar4_archive_before["crc_ok"]
    and phase65ar4_archive_before["safe_paths"]
    and phase65ar4_archive_before["root_main_present"]
    and phase65ar4_archive_before["duplicate_member_count"] == 0
    and phase65ar4_archive_before["symlink_member_count"] == 0
)
phase65ar4_final_digest_link = bool(
    phase65ar4_final_private["valid_json_object"]
    and phase65ar4_final_private["status"] == "accepted"
    and phase65ar4_mentions_digest(
        phase65ar4_final_private["payload"],
        phase65ar4_archive_before["sha256"],
    )
)

phase65ar4_r3_schema_valid = bool(
    phase65ar4_r3_private["valid_json_object"]
    and phase65ar4_r3.get("phase") == PHASE65AR4_EXPECTED_R3_PHASE
    and isinstance(phase65ar4_r3.get("archive_continuity"), dict)
    and isinstance(phase65ar4_r3.get("private_offline_rehearsal"), dict)
    and isinstance(phase65ar4_r3.get("gates"), dict)
    and isinstance(phase65ar4_r3.get("compliance"), dict)
)

phase65ar4_r3_archive = phase65ar4_r3.get("archive_continuity", {})
phase65ar4_r3_rehearsal = phase65ar4_r3.get("private_offline_rehearsal", {})
phase65ar4_r3_observation = phase65ar4_r3_rehearsal.get("runtime_observation")
if not isinstance(phase65ar4_r3_observation, dict):
    phase65ar4_r3_observation = {}
phase65ar4_r3_paths = phase65ar4_r3_observation.get("official_path_observed")
if not isinstance(phase65ar4_r3_paths, dict):
    phase65ar4_r3_paths = {}
phase65ar4_r3_gates = phase65ar4_r3.get("gates", {})
phase65ar4_r3_compliance = phase65ar4_r3.get("compliance", {})

phase65ar4_r3_same_archive = bool(
    phase65ar4_r3_archive.get("submission_sha256")
    == phase65ar4_archive_before["sha256"]
    and phase65ar4_r3_archive.get("file_count")
    == phase65ar4_archive_before["file_count"]
    and phase65ar4_bool(phase65ar4_r3_archive.get("archive_integrity_passed"))
    and phase65ar4_bool(
        phase65ar4_r3_archive.get("accepted_final_contract_exact_digest_link")
    )
    and phase65ar4_r3_archive.get("phase56_artifacts_modified") is False
)

# ----------------------- corrected I/O semantics -----------------------

# R3 sets each *_output_valid flag only after the exact official output path
# exists and its CSV has the expected row count, UID set, prediction column,
# numeric parseability, and requested order.  Therefore these booleans are
# stronger output-path evidence than whether its optional open() hook fired.
phase65ar4_input_dir_observed = phase65ar4_bool(phase65ar4_r3_paths.get("niftis"))
phase65ar4_format_observed = phase65ar4_bool(
    phase65ar4_r3_paths.get("submission_format")
)
phase65ar4_output_open_hook_observed = phase65ar4_bool(
    phase65ar4_r3_paths.get("submission_output")
)
phase65ar4_instrumented_output_proven = bool(
    phase65ar4_r3_rehearsal.get("instrumented_return_code") == 0
    and phase65ar4_bool(
        phase65ar4_r3_rehearsal.get("instrumented_output_valid")
    )
)
phase65ar4_clean_output_proven = bool(
    phase65ar4_r3_rehearsal.get("clean_return_code") == 0
    and phase65ar4_bool(phase65ar4_r3_rehearsal.get("clean_output_valid"))
)
phase65ar4_output_exact_path_proven = bool(
    phase65ar4_instrumented_output_proven and phase65ar4_clean_output_proven
)
phase65ar4_corrected_io_pass = bool(
    phase65ar4_input_dir_observed
    and phase65ar4_format_observed
    and phase65ar4_output_exact_path_proven
)

# ----------------------- other offline evidence ------------------------

phase65ar4_hub_pass = bool(
    phase65ar4_r3_observation.get("torch_hub_load_call_count", 0) > 0
    and phase65ar4_bool(
        phase65ar4_r3_observation.get("all_observed_hub_calls_source_local")
    )
    and phase65ar4_bool(
        phase65ar4_r3_observation.get("all_observed_hub_repositories_exist")
    )
    and phase65ar4_bool(
        phase65ar4_r3_observation.get(
            "all_observed_hub_repositories_inside_submission"
        )
    )
)
phase65ar4_network_pass = bool(
    phase65ar4_zero(
        phase65ar4_r3_observation.get("socket_network_attempt_count")
    )
    and phase65ar4_zero(
        phase65ar4_r3_observation.get("url_loader_call_count")
    )
)
phase65ar4_checkpoint_pass = bool(
    phase65ar4_r3_observation.get("torch_load_call_count", 0) > 0
    and phase65ar4_r3_observation.get("torch_load_pathlike_count")
    == phase65ar4_r3_observation.get("torch_load_call_count")
    and phase65ar4_zero(
        phase65ar4_r3_observation.get("torch_load_missing_path_count")
    )
    and phase65ar4_zero(
        phase65ar4_r3_observation.get("torch_load_outside_submission_count")
    )
)
phase65ar4_instrumentation_pass = phase65ar4_zero(
    phase65ar4_r3_observation.get("instrumentation_setup_failure_count")
)
phase65ar4_execution_pass = bool(
    phase65ar4_bool(phase65ar4_r3_rehearsal.get("configured"))
    and phase65ar4_bool(phase65ar4_r3_rehearsal.get("valid_configuration"))
    and phase65ar4_bool(phase65ar4_r3_rehearsal.get("executed"))
    and phase65ar4_bool(phase65ar4_r3_rehearsal.get("finite_and_in_range"))
    and phase65ar4_bool(
        phase65ar4_r3_rehearsal.get("instrumentation_invariant")
    )
    and phase65ar4_r3_rehearsal.get("instrumented_vs_clean_max_abs") == 0.0
    and phase65ar4_output_exact_path_proven
)
phase65ar4_optional_metadata_pass = bool(
    phase65ar4_r3_rehearsal.get("unreadable_count") in (None, 0)
    and phase65ar4_r3_rehearsal.get("branch_counts_match") is not False
    and phase65ar4_r3_rehearsal.get("preprocessing_checks_all_true") is not False
)

phase65ar4_required_compliance = {
    "training_performed": False,
    "labels_read": False,
    "test_data_read": False,
    "test_time_adaptation": False,
    "case_level_predictions_exported": False,
    "private_paths_or_uids_exported": False,
    "source_lines_exported": False,
    "subprocess_logs_exported": False,
    "independent_test_case_inference_preserved": True,
}
phase65ar4_compliance_pass = all(
    phase65ar4_r3_compliance.get(key) is expected
    for key, expected in phase65ar4_required_compliance.items()
)

phase65ar4_false_r3_gates = sorted(
    key for key, value in phase65ar4_r3_gates.items() if value is False
)
phase65ar4_unexpected_false_r3_gates = sorted(
    set(phase65ar4_false_r3_gates) - PHASE65AR4_ALLOWED_R3_FALSE_GATES
)

phase65ar4_reference_available = phase65ar4_bool(
    phase65ar4_r3_rehearsal.get("reference_probability_available")
)
phase65ar4_exact_reference_parity = phase65ar4_r3_rehearsal.get(
    "exact_reference_parity"
)
phase65ar4_numeric_state_valid = bool(
    (
        not phase65ar4_reference_available
        and phase65ar4_exact_reference_parity is None
    )
    or (
        phase65ar4_reference_available
        and isinstance(phase65ar4_exact_reference_parity, bool)
    )
)

phase65ar4_static_provenance_pass = bool(
    phase65ar4_archive_integrity
    and phase65ar4_final_digest_link
    and phase65ar4_r3_schema_valid
    and phase65ar4_r3_hash["matches"]
    and phase65ar4_r3_same_archive
)
phase65ar4_offline_pass = bool(
    phase65ar4_static_provenance_pass
    and phase65ar4_execution_pass
    and phase65ar4_corrected_io_pass
    and phase65ar4_hub_pass
    and phase65ar4_network_pass
    and phase65ar4_checkpoint_pass
    and phase65ar4_instrumentation_pass
    and phase65ar4_optional_metadata_pass
    and phase65ar4_compliance_pass
    and not phase65ar4_unexpected_false_r3_gates
    and phase65ar4_numeric_state_valid
)
phase65ar4_full_numeric_pass = bool(
    phase65ar4_offline_pass
    and phase65ar4_reference_available
    and phase65ar4_exact_reference_parity is True
)

# Verify again after all reads: the Phase56 archive must remain byte-identical.
phase65ar4_archive_after_sha = (
    phase65ar4_sha_file(PHASE65AR4_SUBMISSION_ZIP)
    if PHASE65AR4_SUBMISSION_ZIP.is_file()
    else None
)
phase65ar4_archive_unchanged = bool(
    phase65ar4_archive_before["sha256"]
    and phase65ar4_archive_before["sha256"] == phase65ar4_archive_after_sha
)
phase65ar4_offline_pass = bool(
    phase65ar4_offline_pass and phase65ar4_archive_unchanged
)
phase65ar4_full_numeric_pass = bool(
    phase65ar4_full_numeric_pass and phase65ar4_archive_unchanged
)

if not phase65ar4_static_provenance_pass:
    phase65ar4_status = "archive_r3_contract_or_digest_provenance_failed_stop"
    phase65ar4_recommendation = (
        "Stop. Preserve Phase56 and repair only the failed archive, accepted-contract, "
        "R3-contract-hash, or same-digest provenance criterion."
    )
elif not phase65ar4_archive_unchanged:
    phase65ar4_status = "phase56_archive_changed_during_read_only_audit_stop"
    phase65ar4_recommendation = (
        "Stop. The Phase56 archive digest changed during a read-only audit."
    )
elif not phase65ar4_offline_pass:
    phase65ar4_status = "corrected_offline_deployment_integrity_failed_stop"
    phase65ar4_recommendation = (
        "Stop. The R3 output-hook false negative was corrected, but at least one "
        "independent runtime, network, checkpoint, compliance, or contract gate failed."
    )
elif phase65ar4_reference_available and not phase65ar4_full_numeric_pass:
    phase65ar4_status = "offline_passed_exact_phase56_numeric_parity_failed_stop"
    phase65ar4_recommendation = (
        "Stop. Offline deployment integrity passed, but the supplied exact Phase56 "
        "development reference does not match."
    )
elif phase65ar4_full_numeric_pass:
    phase65ar4_status = (
        "accepted_corrected_offline_and_exact_phase56_numeric_parity_"
        "ready_for_corrected_phase65b"
    )
    phase65ar4_recommendation = (
        "Proceed to corrected fold-local Phase65B while retaining the byte-identical "
        "Phase56 archive as deployment anchor."
    )
else:
    phase65ar4_status = (
        "accepted_corrected_offline_deployment_integrity_numeric_parity_"
        "not_available_ready_for_corrected_phase65b"
    )
    phase65ar4_recommendation = (
        "Offline deployment integrity passed. Proceed to corrected fold-local Phase65B; "
        "exact development-to-deployment parity remains unclaimed."
    )

phase65ar4_report = {
    "phase": "phase65ar4_standalone_corrected_phase65ar3_offline_gate_adjudication",
    "status": phase65ar4_status,
    "restart_safety": {
        "requires_previous_notebook_cells": False,
        "requires_live_kernel_variables": False,
        "uses_persistent_phase56_and_phase65ar3_artifacts_only": True,
        "inference_rerun_performed": False,
    },
    "artifact_provenance": {
        "submission_zip": {
            key: phase65ar4_archive_before[key]
            for key in (
                "exists",
                "sha256",
                "compressed_bytes",
                "file_count",
                "valid_zip",
                "crc_ok",
                "safe_paths",
                "root_main_present",
                "duplicate_member_count",
                "symlink_member_count",
            )
        },
        "archive_integrity_passed": phase65ar4_archive_integrity,
        "accepted_final_contract_exact_digest_link": phase65ar4_final_digest_link,
        "r3_contract": {
            "exists": phase65ar4_r3_private["exists"],
            "valid_json_object": phase65ar4_r3_private["valid_json_object"],
            "file_sha256": phase65ar4_r3_private["sha256"],
            "status": phase65ar4_r3_private["status"],
            "schema_valid": phase65ar4_r3_schema_valid,
            "self_hash": phase65ar4_r3_hash,
            "same_phase56_archive": phase65ar4_r3_same_archive,
        },
        "phase56_archive_unchanged": phase65ar4_archive_unchanged,
    },
    "corrected_runtime_path_adjudication": {
        "niftis_directory_observed": phase65ar4_input_dir_observed,
        "submission_format_observed": phase65ar4_format_observed,
        "submission_output_open_hook_observed_diagnostic": (
            phase65ar4_output_open_hook_observed
        ),
        "instrumented_exact_official_output_created_and_valid": (
            phase65ar4_instrumented_output_proven
        ),
        "clean_exact_official_output_created_and_valid": (
            phase65ar4_clean_output_proven
        ),
        "exact_official_output_path_proven": phase65ar4_output_exact_path_proven,
        "output_open_hook_required_for_gate": False,
        "output_proof_rule": (
            "exact_official_path_exists_and_csv_schema_uid_set_row_count_"
            "prediction_parse_validation_passes"
        ),
        "all_official_runtime_io_semantics_satisfied": phase65ar4_corrected_io_pass,
    },
    "offline_runtime_evidence": {
        "configured_valid_and_executed": phase65ar4_execution_pass,
        "instrumented_and_clean_return_codes_zero": bool(
            phase65ar4_r3_rehearsal.get("instrumented_return_code") == 0
            and phase65ar4_r3_rehearsal.get("clean_return_code") == 0
        ),
        "finite_and_in_probability_range": phase65ar4_r3_rehearsal.get(
            "finite_and_in_range"
        ),
        "instrumented_vs_clean_max_abs": phase65ar4_r3_rehearsal.get(
            "instrumented_vs_clean_max_abs"
        ),
        "instrumentation_invariant": phase65ar4_r3_rehearsal.get(
            "instrumentation_invariant"
        ),
        "torch_hub_local_existing_and_bundled": phase65ar4_hub_pass,
        "torch_hub_load_call_count": phase65ar4_r3_observation.get(
            "torch_hub_load_call_count"
        ),
        "torch_checkpoint_paths_all_existing_and_bundled": phase65ar4_checkpoint_pass,
        "torch_load_call_count": phase65ar4_r3_observation.get(
            "torch_load_call_count"
        ),
        "socket_network_attempt_count": phase65ar4_r3_observation.get(
            "socket_network_attempt_count"
        ),
        "url_loader_call_count": phase65ar4_r3_observation.get(
            "url_loader_call_count"
        ),
        "zero_runtime_network_attempts": phase65ar4_network_pass,
        "instrumentation_setup_clean": phase65ar4_instrumentation_pass,
        "optional_runtime_metadata_noncontradictory": phase65ar4_optional_metadata_pass,
        "case_level_values_exported": False,
        "private_paths_uids_or_logs_exported": False,
    },
    "prior_gate_adjudication": {
        "r3_false_gate_keys": phase65ar4_false_r3_gates,
        "allowed_derivative_false_gate_keys": sorted(
            PHASE65AR4_ALLOWED_R3_FALSE_GATES
        ),
        "unexpected_false_gate_keys": phase65ar4_unexpected_false_r3_gates,
        "root_cause": (
            "diagnostic_output_open_hook_false_negative"
            if phase65ar4_offline_pass and not phase65ar4_output_open_hook_observed
            else None
        ),
        "phase56_code_or_archive_repair_required": False
        if phase65ar4_offline_pass
        else None,
    },
    "gates": {
        "static_archive_and_contract_provenance": phase65ar4_static_provenance_pass,
        "r3_exact_contract_self_hash": phase65ar4_r3_hash["matches"],
        "corrected_official_runtime_io": phase65ar4_corrected_io_pass,
        "local_bundled_torch_hub": phase65ar4_hub_pass,
        "zero_runtime_network_attempts": phase65ar4_network_pass,
        "all_torch_checkpoints_existing_and_bundled": phase65ar4_checkpoint_pass,
        "instrumented_clean_probability_invariance": (
            phase65ar4_r3_rehearsal.get("instrumentation_invariant")
        ),
        "r3_compliance_contract_preserved": phase65ar4_compliance_pass,
        "phase56_archive_remained_unchanged": phase65ar4_archive_unchanged,
        "corrected_offline_deployment_integrity_passed": phase65ar4_offline_pass,
        "exact_development_probability_available": phase65ar4_reference_available,
        "exact_development_probability_parity": phase65ar4_exact_reference_parity,
        "full_numeric_parity_passed": phase65ar4_full_numeric_pass,
    },
    "recommendation": phase65ar4_recommendation,
    "interpretation_contract": {
        "valid_exact_official_output_file_is_stronger_than_open_hook_observation": True,
        "r3_output_valid_flags_imply_exact_path_and_schema_validation": True,
        "this_adjudication_does_not_rerun_or_change_inference": True,
        "offline_integrity_does_not_claim_development_numeric_parity": True,
        "phase12c_or_phase42_predictions_used_as_phase56_reference": False,
        "public_leaderboard_used_for_gate": False,
        "phase56_remains_deployment_anchor": True,
    },
    "compliance": {
        "training_performed": False,
        "labels_read": False,
        "nifti_files_read": False,
        "test_data_read": False,
        "test_time_adaptation": False,
        "case_level_predictions_read_or_exported": False,
        "private_paths_or_uids_read_or_exported": False,
        "source_lines_exported": False,
        "subprocess_logs_exported": False,
        "phase56_artifacts_modified": False,
        "independent_test_case_inference_preserved": True,
    },
    "elapsed_seconds": float(time.time() - PHASE65AR4_STARTED),
}

PHASE65AR4_REPORT_PRIVATE = phase65ar4_report
phase65ar4_payload = phase65ar4_jsonable(phase65ar4_report)
phase65ar4_contract_raw = json.dumps(
    phase65ar4_payload,
    sort_keys=True,
    separators=(",", ":"),
    ensure_ascii=True,
).encode("utf-8")
phase65ar4_payload["contract_sha256"] = phase65ar4_sha_bytes(
    phase65ar4_contract_raw
)
phase65ar4_payload["contract_file"] = str(PHASE65AR4_CONTRACT_PATH)
PHASE65AR4_CONTRACT_PATH.write_text(
    json.dumps(phase65ar4_payload, indent=2) + "\n",
    encoding="utf-8",
)

print("BEGIN SANITIZED_PHASE65AR4_CORRECTED_OFFLINE_GATE_ADJUDICATION")
print(json.dumps(phase65ar4_payload, indent=2))
print("END SANITIZED_PHASE65AR4_CORRECTED_OFFLINE_GATE_ADJUDICATION")
