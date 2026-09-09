import contextlib
import hashlib
import io
import json
import os
import runpy
import tempfile
import zipfile
from pathlib import Path


CELL = Path(__file__).with_name(
    "phase65_cell_165ar4_corrected_offline_gate_adjudication.py"
)


def sha_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def add_contract_hash(payload: dict, path: Path) -> None:
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    payload["contract_sha256"] = hashlib.sha256(raw).hexdigest()
    payload["contract_file"] = str(path)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def build(root: Path, mode: str) -> dict:
    root.mkdir(parents=True)
    archive = root / "phase56_submission.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("main.py", "print('synthetic')\n")
        handle.writestr("checkpoint.pt", b"synthetic checkpoint")
    digest = sha_file(archive)
    (root / "phase56_final_archive_contract.json").write_text(
        json.dumps({"status": "accepted", "submission_sha256": digest}),
        encoding="utf-8",
    )

    network_attempts = 1 if mode == "bad_network" else 0
    clean_output_valid = mode != "bad_output"
    reference_available = mode == "numeric_mismatch"
    exact_reference_parity = False if reference_available else None

    gates = {
        "archive_continuity": True,
        "torch_hub_call_observed_runtime": True,
        "all_observed_hub_calls_local_existing_bundled_repository": True,
        "zero_socket_or_url_loader_network_attempts": network_attempts == 0,
        "all_observed_torch_checkpoint_paths_exist_inside_submission": True,
        "all_official_runtime_paths_observed": False,
        "instrumentation_setup_clean": True,
        "instrumented_clean_probability_invariance": True,
        "offline_deployment_integrity_passed": False,
        "exact_development_probability_parity": exact_reference_parity,
        "full_numeric_parity_passed": False,
    }
    r3 = {
        "phase": "phase65ar3_local_torch_hub_semantics_and_offline_phase56_rehearsal",
        "status": "local_torch_hub_or_offline_runtime_gate_failed_stop",
        "archive_continuity": {
            "submission_sha256": digest,
            "file_count": 2,
            "archive_integrity_passed": True,
            "accepted_final_contract_exact_digest_link": True,
            "phase65ar2_authoritative_lineage_for_same_digest": True,
            "phase56_artifacts_modified": False,
        },
        "private_offline_rehearsal": {
            "configured": True,
            "valid_configuration": True,
            "reference_probability_available": reference_available,
            "executed": True,
            "n": 8,
            "instrumented_return_code": 0,
            "clean_return_code": 0,
            "instrumented_output_valid": True,
            "clean_output_valid": clean_output_valid,
            "finite_and_in_range": True,
            "instrumented_vs_clean_max_abs": 0.0,
            "instrumentation_invariant": True,
            "exact_reference_parity": exact_reference_parity,
            "runtime_observation": {
                "official_path_observed": {
                    "niftis": True,
                    "submission_format": True,
                    "submission_output": False,
                },
                "torch_hub_load_call_count": 1,
                "all_observed_hub_calls_source_local": True,
                "all_observed_hub_repositories_exist": True,
                "all_observed_hub_repositories_inside_submission": True,
                "torch_load_call_count": 2,
                "torch_load_pathlike_count": 2,
                "torch_load_missing_path_count": 0,
                "torch_load_outside_submission_count": 0,
                "url_loader_call_count": 0,
                "socket_network_attempt_count": network_attempts,
                "instrumentation_setup_failure_count": 0,
            },
            "unreadable_count": None,
            "branch_counts_match": None,
            "preprocessing_checks_all_true": None,
        },
        "gates": gates,
        "compliance": {
            "training_performed": False,
            "labels_read": False,
            "test_data_read": False,
            "test_time_adaptation": False,
            "case_level_predictions_exported": False,
            "private_paths_or_uids_exported": False,
            "source_lines_exported": False,
            "subprocess_logs_exported": False,
            "independent_test_case_inference_preserved": True,
        },
    }
    r3_path = root / "phase65ar3_local_torch_hub_offline_rehearsal_contract.json"
    add_contract_hash(r3, r3_path)

    if mode == "bad_digest":
        with archive.open("ab") as handle:
            handle.write(b"changed")

    old_root = os.environ.get("PHASE65AR4_ROOT_PRIVATE")
    os.environ["PHASE65AR4_ROOT_PRIVATE"] = str(root)
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            result = runpy.run_path(str(CELL))
        return result["PHASE65AR4_REPORT_PRIVATE"]
    finally:
        if old_root is None:
            os.environ.pop("PHASE65AR4_ROOT_PRIVATE", None)
        else:
            os.environ["PHASE65AR4_ROOT_PRIVATE"] = old_root


with tempfile.TemporaryDirectory() as temporary:
    base = Path(temporary)
    accepted = build(base / "accepted", "accepted")
    assert accepted["status"] == (
        "accepted_corrected_offline_deployment_integrity_numeric_parity_"
        "not_available_ready_for_corrected_phase65b"
    ), accepted["status"]
    assert accepted["gates"]["corrected_offline_deployment_integrity_passed"]
    assert not accepted["corrected_runtime_path_adjudication"][
        "submission_output_open_hook_observed_diagnostic"
    ]
    assert accepted["corrected_runtime_path_adjudication"][
        "exact_official_output_path_proven"
    ]

    bad_output = build(base / "bad_output", "bad_output")
    assert bad_output["status"] == "corrected_offline_deployment_integrity_failed_stop"

    bad_network = build(base / "bad_network", "bad_network")
    assert bad_network["status"] == "corrected_offline_deployment_integrity_failed_stop"

    bad_digest = build(base / "bad_digest", "bad_digest")
    assert bad_digest["status"] == "archive_r3_contract_or_digest_provenance_failed_stop"

    numeric_mismatch = build(base / "numeric_mismatch", "numeric_mismatch")
    assert numeric_mismatch["status"] == "offline_passed_exact_phase56_numeric_parity_failed_stop"

print("phase65ar4 synthetic acceptance and rejection tests passed")
