from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score


# Phase57R — minimal exact fresh-kernel restoration for Cell 157A.
#
# This cell restores only PHASE52_SUPPORT_GATE_STATE_PRIVATE from immutable
# training artifacts. It does not load a neural network, train a model, search
# parameters, inspect training voxels, or read smoke/challenge-test data.

phase57r_started = time.perf_counter()
PHASE57R_EXPECTED = {
    "case_count": 1362,
    "normal_count": 615,
    "pathologic_count": 747,
    "group_sizes": [
        39, 456, 145, 255, 76, 7, 49, 208,
        32, 4, 35, 10, 32, 9, 5,
    ],
    "phase39_log_loss": 0.29302734886541737,
    "phase39_auroc": 0.9455752549493367,
    "anchor_log_loss": 0.29016628416289664,
    "anchor_auroc": 0.9464742438589044,
    "anchor_mean_probability": 0.5500570231688339,
}


def phase57r_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    return np.log(probability) - np.log1p(-probability)


def phase57r_sigmoid(logit_value):
    logit_value = np.asarray(logit_value, dtype=np.float64)
    output = np.empty_like(logit_value)
    positive = logit_value >= 0.0
    output[positive] = 1.0 / (
        1.0 + np.exp(-logit_value[positive])
    )
    exponential = np.exp(logit_value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase57r_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    assert labels.shape == probability.shape
    return {
        "log_loss": float(log_loss(
            labels, probability, labels=[0, 1]
        )),
        "auroc": float(roc_auc_score(labels, probability)),
        "mean_probability": float(np.mean(probability)),
    }


def phase57r_load_vector(path):
    path = Path(path)
    assert path.is_file(), {
        "message": "Required persistent OOF artifact is missing.",
        "path": str(path),
    }
    vector = np.asarray(
        np.load(path, allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    assert vector.shape == (PHASE57R_EXPECTED["case_count"],), {
        "path": str(path),
        "shape": list(vector.shape),
    }
    assert np.all(np.isfinite(vector)), {"path": str(path)}
    return vector


def phase57r_resolve_vector(description, candidates):
    unique_candidates = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen:
            seen.add(key)
            unique_candidates.append(candidate)

    existing = [
        path for path in unique_candidates if path.is_file()
    ]
    assert existing, {
        "message": "Required persistent OOF artifact is missing.",
        "description": description,
        "checked_paths": [str(path) for path in unique_candidates],
    }
    reference = phase57r_load_vector(existing[0])
    for duplicate_path in existing[1:]:
        duplicate = phase57r_load_vector(duplicate_path)
        error = float(np.max(np.abs(duplicate - reference)))
        assert error <= 1.0e-12, {
            "message": "Duplicate persistent OOF artifacts disagree.",
            "description": description,
            "reference": str(existing[0]),
            "duplicate": str(duplicate_path),
            "maximum_error": error,
        }
    return reference, str(existing[0]), [str(path) for path in existing]


# -------------------------------------------------------------------------
# Restore training labels in the immutable cache row order.
# -------------------------------------------------------------------------

phase57r_root_candidates = []
phase57r_environment_root = os.environ.get("DAT_PRIVATE_ROOT")
if phase57r_environment_root:
    phase57r_root_candidates.append(Path(phase57r_environment_root))
phase57r_label_candidates = []
phase57r_seen_label_paths = set()
for phase57r_root in phase57r_root_candidates:
    phase57r_candidate = phase57r_root / "train_labels.csv"
    if phase57r_candidate.is_file():
        phase57r_key = str(phase57r_candidate.resolve())
        if phase57r_key not in phase57r_seen_label_paths:
            phase57r_seen_label_paths.add(phase57r_key)
            phase57r_label_candidates.append(phase57r_candidate)

if not phase57r_label_candidates:
    for phase57r_candidate in sorted(
        Path("/kaggle/input").glob("**/train_labels.csv")
    ):
        lowered = [part.lower() for part in phase57r_candidate.parts]
        if any("smoke" in part or "test" in part for part in lowered):
            continue
        phase57r_key = str(phase57r_candidate.resolve())
        if phase57r_key not in phase57r_seen_label_paths:
            phase57r_seen_label_paths.add(phase57r_key)
            phase57r_label_candidates.append(phase57r_candidate)

phase57r_valid_label_records = []
for phase57r_candidate in phase57r_label_candidates:
    try:
        phase57r_frame = pd.read_csv(
            phase57r_candidate, dtype={"uid": str}
        )
    except Exception:
        continue
    if list(phase57r_frame.columns) != ["uid", "is_pathologic"]:
        continue
    if len(phase57r_frame) != PHASE57R_EXPECTED["case_count"]:
        continue
    if not phase57r_frame["uid"].notna().all():
        continue
    if not phase57r_frame["uid"].is_unique:
        continue
    if not phase57r_frame["is_pathologic"].isin(
        [0, 1, 0.0, 1.0]
    ).all():
        continue
    phase57r_valid_label_records.append((
        phase57r_candidate,
        phase57r_frame["uid"].astype(str).to_numpy(copy=True),
        phase57r_frame["is_pathologic"].to_numpy(
            dtype=np.int64, copy=True
        ),
    ))

assert phase57r_valid_label_records, {
    "message": "Could not resolve the accepted training-label CSV.",
    "candidate_count": len(phase57r_label_candidates),
}
phase57r_label_path, phase57r_uid_order, phase57r_labels = (
    phase57r_valid_label_records[0]
)
for (
    phase57r_duplicate_path,
    phase57r_duplicate_uids,
    phase57r_duplicate_labels,
) in phase57r_valid_label_records[1:]:
    assert np.array_equal(
        phase57r_duplicate_uids, phase57r_uid_order
    ), {
        "message": "Training-label CSV copies have different row order.",
        "reference": str(phase57r_label_path),
        "duplicate": str(phase57r_duplicate_path),
    }
    assert np.array_equal(
        phase57r_duplicate_labels, phase57r_labels
    ), {
        "message": "Training-label CSV copies disagree.",
        "reference": str(phase57r_label_path),
        "duplicate": str(phase57r_duplicate_path),
    }

assert phase57r_labels.shape == (1362,)
assert int(np.sum(phase57r_labels == 0)) == (
    PHASE57R_EXPECTED["normal_count"]
)
assert int(np.sum(phase57r_labels == 1)) == (
    PHASE57R_EXPECTED["pathologic_count"]
)
del phase57r_uid_order


# -------------------------------------------------------------------------
# Restore acquisition groups and independently regenerate Phase52 partitions.
# -------------------------------------------------------------------------

phase57r_phase50_partition_path = Path(
    "/kaggle/working/phase50_private_validation/phase50_partitions.npz"
)
phase57r_phase52_partition_path = Path(
    "/kaggle/working/phase52_repeated_partition.npz"
)
phase57r_phase52_contract_path = Path(
    "/kaggle/working/phase52_support_gate_contract.json"
)
for phase57r_required_path in (
    phase57r_phase50_partition_path,
    phase57r_phase52_partition_path,
    phase57r_phase52_contract_path,
):
    assert phase57r_required_path.is_file(), {
        "message": "Required validation-state artifact is missing.",
        "path": str(phase57r_required_path),
    }

with np.load(
    phase57r_phase50_partition_path, allow_pickle=False
) as phase57r_partition50:
    phase57r_groups = np.asarray(
        phase57r_partition50["acquisition_group"],
        dtype=np.int64,
    ).reshape(-1)

with np.load(
    phase57r_phase52_partition_path, allow_pickle=False
) as phase57r_partition52:
    phase57r_fold_assignment = np.asarray(
        phase57r_partition52["fold_assignment"],
        dtype=np.int64,
    )
    phase57r_contract_sha256 = str(np.asarray(
        phase57r_partition52["contract_sha256"]
    ).item())

phase57r_contract52 = json.loads(
    phase57r_phase52_contract_path.read_text(encoding="utf-8")
)
assert phase57r_groups.shape == (1362,)
assert phase57r_fold_assignment.shape == (5, 1362)
assert np.all(
    (phase57r_fold_assignment >= 0)
    & (phase57r_fold_assignment < 5)
)
assert np.bincount(
    phase57r_groups, minlength=15
).tolist() == PHASE57R_EXPECTED["group_sizes"]
assert phase57r_contract_sha256 == str(
    phase57r_contract52["contract_sha256"]
)

phase57r_partition_digest = hashlib.sha256(
    np.ascontiguousarray(
        phase57r_fold_assignment.astype(np.int8)
    ).tobytes()
).hexdigest()
assert phase57r_partition_digest == str(
    phase57r_contract52["phase52_partition_sha256"]
)

phase57r_regenerated = np.full((5, 1362), -1, dtype=np.int64)
for phase57r_repeat, phase57r_seed in enumerate(
    [520101, 520102, 520103, 520104, 520105]
):
    phase57r_rng = np.random.default_rng(phase57r_seed)
    for phase57r_group in range(15):
        for phase57r_label in (0, 1):
            phase57r_indices = np.flatnonzero(
                (phase57r_groups == phase57r_group)
                & (phase57r_labels == phase57r_label)
            )
            if phase57r_indices.size == 0:
                continue
            phase57r_indices = phase57r_indices.copy()
            phase57r_rng.shuffle(phase57r_indices)
            phase57r_cycle = np.arange(
                phase57r_indices.size, dtype=np.int64
            )
            phase57r_cycle = (
                phase57r_cycle
                + int(phase57r_rng.integers(0, 5))
            ) % 5
            phase57r_regenerated[
                phase57r_repeat, phase57r_indices
            ] = phase57r_cycle

assert np.array_equal(
    phase57r_regenerated, phase57r_fold_assignment
), {
    "message": (
        "Training-label row order does not match the persistent Phase52 "
        "partition. Do not continue to Phase57."
    )
}
del phase57r_regenerated


# -------------------------------------------------------------------------
# Reconstruct the exact deployment-matched Phase43 OOF anchor.
# -------------------------------------------------------------------------

(
    phase57r_phase12_probability,
    phase57r_phase12_source,
    phase57r_phase12_sources,
) = phase57r_resolve_vector(
    "Phase12c OOF probability",
    [Path(
        "/kaggle/working/phase32_phase12c_oof_float64.npy"
    )],
)
(
    phase57r_phase33_probability,
    phase57r_phase33_source,
    phase57r_phase33_sources,
) = phase57r_resolve_vector(
    "Phase33 component OOF probability",
    [
        Path(
            "/kaggle/working/phase33_private_checkpoint/"
            "phase33_component_oof_float64.npy"
        ),
        Path(
            "/kaggle/working/phase33_component_oof_float64.npy"
        ),
    ],
)
(
    phase57r_combined39_residual,
    phase57r_combined39_source,
    phase57r_combined39_sources,
) = phase57r_resolve_vector(
    "Phase39 combined residual",
    [
        Path(
            "/kaggle/working/phase39_private_checkpoint/"
            "phase39_combined_residual_float64.npy"
        ),
        Path(
            "/kaggle/working/phase39_combined_residual_float64.npy"
        ),
    ],
)
(
    phase57r_phase39_probability,
    phase57r_phase39_source,
    phase57r_phase39_sources,
) = phase57r_resolve_vector(
    "Phase39 OOF probability",
    [
        Path(
            "/kaggle/working/phase39_private_checkpoint/"
            "phase39_oof_float64.npy"
        ),
        Path("/kaggle/working/phase39_oof_float64.npy"),
    ],
)

for phase57r_probability in (
    phase57r_phase12_probability,
    phase57r_phase33_probability,
    phase57r_phase39_probability,
):
    assert np.all(
        (phase57r_probability > 0.0)
        & (phase57r_probability < 1.0)
    )

phase57r_phase12_logit = phase57r_logit(
    phase57r_phase12_probability
)
phase57r_phase33_residual = (
    phase57r_logit(phase57r_phase33_probability)
    - phase57r_phase12_logit
)
phase57r_phase39_reconstructed = phase57r_sigmoid(
    phase57r_phase12_logit
    + np.clip(phase57r_combined39_residual, -2.0, 2.0)
)
phase57r_phase39_error = float(np.max(np.abs(
    phase57r_phase39_reconstructed
    - phase57r_phase39_probability
)))
assert phase57r_phase39_error <= 2.0e-12

phase57r_phase39_metrics = phase57r_metrics(
    phase57r_labels, phase57r_phase39_probability
)
assert abs(
    phase57r_phase39_metrics["log_loss"]
    - PHASE57R_EXPECTED["phase39_log_loss"]
) <= 2.0e-10
assert abs(
    phase57r_phase39_metrics["auroc"]
    - PHASE57R_EXPECTED["phase39_auroc"]
) <= 2.0e-12


def phase57r_read_phase42_gate():
    direct_candidates = [
        Path(
            "/kaggle/working/phase56_experimental_submission/"
            "phase42_assets/phase42_hierarchical_gate.json"
        ),
        Path(
            "/kaggle/working/phase43_experimental_submission/"
            "phase42_assets/phase42_hierarchical_gate.json"
        ),
        Path(
            "/kaggle/working/phase42_experimental_submission/"
            "phase42_assets/phase42_hierarchical_gate.json"
        ),
        Path(
            "/kaggle/working/phase42_hierarchical_gate.json"
        ),
    ]
    for phase57r_candidate in direct_candidates:
        if phase57r_candidate.is_file():
            return (
                json.loads(phase57r_candidate.read_text(
                    encoding="utf-8"
                )),
                str(phase57r_candidate),
            )

    archive_candidates = [
        Path("/kaggle/working/phase56_submission.zip"),
        Path("/kaggle/working/phase43_submission.zip"),
        Path("/kaggle/working/phase42_submission.zip"),
    ]
    for phase57r_archive_path in archive_candidates:
        if not phase57r_archive_path.is_file():
            continue
        with zipfile.ZipFile(
            phase57r_archive_path, mode="r"
        ) as phase57r_archive:
            phase57r_member = (
                "phase42_assets/phase42_hierarchical_gate.json"
            )
            if phase57r_member not in phase57r_archive.namelist():
                continue
            return (
                json.loads(phase57r_archive.read(
                    phase57r_member
                ).decode("utf-8")),
                f"{phase57r_archive_path}:{phase57r_member}",
            )
    raise AssertionError({
        "message": "Could not locate the accepted Phase42 gate state."
    })


phase57r_gate42, phase57r_gate42_source = phase57r_read_phase42_gate()
assert float(phase57r_gate42[
    "phase36_raw_residual_weight"
]) == 0.75
assert float(phase57r_gate42[
    "phase33_logit_residual_weight"
]) == 0.125
assert float(phase57r_gate42["residual_cap"]) == 1.0

phase57r_alpha_map = {
    int(group): float(alpha)
    for group, alpha in phase57r_gate42[
        "known_group_alpha"
    ].items()
}
assert set(phase57r_alpha_map) == set(range(15))
phase57r_group_alpha = np.asarray(    [phase57r_alpha_map[int(group)] for group in phase57r_groups],
    dtype=np.float64,
)

# Phase39 combined residual is 0.75*r36 + 0.25*r33. Phase42 uses
# 0.75*r36 + 0.125*r33, hence subtract 0.125*r33 here.
phase57r_phase42_uncapped = (
    phase57r_combined39_residual
    - 0.125 * phase57r_phase33_residual
)
phase57r_anchor_logit = (
    phase57r_phase12_logit
    + phase57r_group_alpha
    * np.clip(phase57r_phase42_uncapped, -1.0, 1.0)
)
phase57r_anchor_probability = phase57r_sigmoid(
    phase57r_anchor_logit
)
phase57r_anchor_metrics = phase57r_metrics(
    phase57r_labels, phase57r_anchor_probability
)

for phase57r_metric, phase57r_expected_key, phase57r_tolerance in (
    ("log_loss", "anchor_log_loss", 2.0e-10),
    ("auroc", "anchor_auroc", 2.0e-12),
    ("mean_probability", "anchor_mean_probability", 2.0e-10),
):
    assert abs(
        phase57r_anchor_metrics[phase57r_metric]
        - PHASE57R_EXPECTED[phase57r_expected_key]
    ) <= phase57r_tolerance, {
        "message": "Restored Phase43 anchor metric parity failed.",
        "metric": phase57r_metric,
        "observed": phase57r_anchor_metrics[phase57r_metric],
        "expected": PHASE57R_EXPECTED[phase57r_expected_key],
    }


# -------------------------------------------------------------------------
# Restore precisely the state required by Cell 157A.
# -------------------------------------------------------------------------

PHASE52_SUPPORT_GATE_STATE_PRIVATE = {
    "contract_sha256": phase57r_contract_sha256,
    "fold_assignment": phase57r_fold_assignment.copy(),
    "anchor_probability": phase57r_anchor_probability.copy(),
    "anchor_logit": phase57r_anchor_logit.copy(),
    "anchor_uncertainty": (
        4.0
        * phase57r_anchor_probability
        * (1.0 - phase57r_anchor_probability)
    ),
    "labels": phase57r_labels.copy(),
    "groups": phase57r_groups.copy(),
    "restored_for_phase57": True,
}

phase57r_report = {
    "phase": "phase57_minimal_fresh_kernel_restore",
    "status": "accepted_ready_for_cell_157a",
    "restored_state": {
        "case_count": 1362,
        "normal_count": int(np.sum(phase57r_labels == 0)),
        "pathologic_count": int(np.sum(phase57r_labels == 1)),
        "acquisition_group_count": int(
            np.unique(phase57r_groups).size
        ),
        "group_sizes": np.bincount(
            phase57r_groups, minlength=15
        ).astype(int).tolist(),
        "phase52_fold_assignment_shape": list(
            phase57r_fold_assignment.shape
        ),
        "phase52_partition_regeneration_exact": True,
        "phase52_contract_sha256": phase57r_contract_sha256,
    },
    "anchor": {
        **phase57r_anchor_metrics,
        "formula": (
            "z43_oof=z12c_oof+alpha_group*"
            "clip(combined39_raw-0.125*r33,-1,1)"
        ),
        "phase39_reconstruction_maximum_error": (
            phase57r_phase39_error
        ),
        "reference_metric_parity_exact_within_tolerance": True,
        "artifact_sources": {
            "phase12c": phase57r_phase12_source,
            "phase33": phase57r_phase33_source,
            "phase39_combined_residual": (
                phase57r_combined39_source
            ),
            "phase39_probability": phase57r_phase39_source,
            "phase42_gate": phase57r_gate42_source,
        },
        "all_discovered_duplicate_sources_verified": True,
    },
    "training_label_source_count": len(
        phase57r_valid_label_records
    ),
    "training_label_copies_identical": True,
    "neural_network_loaded": False,
    "model_training_performed": False,
    "hyperparameter_search_performed": False,
    "labels_used_only_for_state_alignment_and_metric_parity": True,
    "training_voxel_data_read": False,
    "training_nifti_files_read": False,
    "training_penultimate_cache_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "uids_displayed": False,
    "patient_rows_displayed": False,
    "case_level_predictions_displayed": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(
        time.perf_counter() - phase57r_started, 3
    ),
}

PHASE57_FRESH_RESTORE_REPORT_PRIVATE = dict(phase57r_report)

print("BEGIN SANITIZED_PHASE57_FRESH_RESTORE")
print(json.dumps(phase57r_report, indent=2))
print("END SANITIZED_PHASE57_FRESH_RESTORE")
