from __future__ import annotations

import hashlib
import io
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np


# Cell 157A — Phase57 transport reset and group-held-out experiment contract.
#
# This cell reads training-only labels, acquisition groups, and accepted OOF
# predictions. It does not read test images, smoke images, or training voxels.
# The challenge leaderboard result is recorded only as evidence that transport
# is the bottleneck; it is not used to select an architecture or parameter.

phase57a_started = time.perf_counter()

PHASE57A_CONFIG = {
    "case_count": 1362,
    "group_count": 15,
    "original_group_fold_count": 3,
    "logo_fold_count": 15,
    "phase12_fold_probability_file": (
        "/kaggle/working/"
        "phase32_phase12c_fold_probabilities_float64.npy"
    ),
    "phase12_oof_file": (
        "/kaggle/working/phase32_phase12c_oof_float64.npy"
    ),
    "highres_cache_file": (
        "/kaggle/working/phase31_highres_float16.npy"
    ),
    "partition_file": (
        "/kaggle/working/phase57_logo_partition.npz"
    ),
    "contract_file": (
        "/kaggle/working/phase57_transport_contract.json"
    ),
    "public_reference": {
        "phase42": {"log_loss": 0.3254, "auroc": 0.9251},
        "phase56": {"log_loss": 0.3220, "auroc": 0.9297},
    },
    "selection_thresholds": {
        "minimum_log_loss_gain_over_anchor": 0.005,
        "minimum_auroc_gain_over_anchor": 0.002,
        "minimum_original_group_fold_wins": 2,
        "maximum_original_group_fold_regret": 0.005,
        "maximum_major_group_harm": 0.015,
    },
    "logo_confirmation_thresholds": {
        "minimum_log_loss_gain_over_retrained_baseline": 0.003,
        "minimum_auroc_gain_over_retrained_baseline": 0.0015,
        "minimum_logo_group_wins": 9,
        "maximum_logo_group_regret": 0.015,
        "minimum_major_group_wins": 7,
        "maximum_major_group_harm": 0.015,
    },
}


# -------------------------------------------------------------------------
# Resolve the accepted training-only Phase52 state.
# -------------------------------------------------------------------------

assert isinstance(
    globals().get("PHASE52_SUPPORT_GATE_STATE_PRIVATE"), dict
), {
    "message": (
        "Phase57 requires PHASE52_SUPPORT_GATE_STATE_PRIVATE. "
        "If the kernel was restarted, run the Phase57 minimal fresh-kernel "
        "restore cell before Cell 157A."
    )
}

phase57a_state52 = PHASE52_SUPPORT_GATE_STATE_PRIVATE
phase57a_labels = np.asarray(
    phase57a_state52["labels"], dtype=np.int64
).reshape(-1)
phase57a_groups = np.asarray(
    phase57a_state52["groups"], dtype=np.int64
).reshape(-1)
phase57a_phase43_anchor = np.asarray(
    phase57a_state52["anchor_probability"], dtype=np.float64
).reshape(-1)

assert phase57a_labels.shape == (PHASE57A_CONFIG["case_count"],)
assert phase57a_groups.shape == (PHASE57A_CONFIG["case_count"],)
assert phase57a_phase43_anchor.shape == (
    PHASE57A_CONFIG["case_count"],
)
assert set(np.unique(phase57a_labels).tolist()) == {0, 1}
assert np.array_equal(
    np.unique(phase57a_groups), np.arange(15, dtype=np.int64)
)
assert np.all(np.isfinite(phase57a_phase43_anchor))
assert np.all(
    (phase57a_phase43_anchor > 0.0)
    & (phase57a_phase43_anchor < 1.0)
)


def phase57a_log_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    assert labels.shape == probability.shape
    return float(np.mean(-(
        labels * np.log(probability)
        + (1.0 - labels) * np.log1p(-probability)
    )))


def phase57a_auroc(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    assert labels.shape == score.shape
    assert set(np.unique(labels).tolist()) == {0, 1}
    assert np.all(np.isfinite(score))

    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    rank = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while (
            stop < score.size
            and sorted_score[stop] == sorted_score[start]
        ):
            stop += 1
        # One-based average rank for the complete tie block.
        average_rank = 0.5 * ((start + 1) + stop)
        rank[order[start:stop]] = average_rank
        start = stop

    positive = labels == 1
    positive_n = int(np.sum(positive))
    negative_n = int(labels.size - positive_n)
    assert positive_n > 0 and negative_n > 0
    mann_whitney = (
        float(np.sum(rank[positive]))
        - positive_n * (positive_n + 1) / 2.0
    )
    return float(mann_whitney / (positive_n * negative_n))


def phase57a_metrics(probability):
    return {
        "log_loss": phase57a_log_loss(
            phase57a_labels, probability
        ),
        "auroc": phase57a_auroc(
            phase57a_labels, probability
        ),
        "mean_probability": float(np.mean(probability)),
    }


# -------------------------------------------------------------------------
# Resolve the accepted acquisition router without reading any test case.
# -------------------------------------------------------------------------

phase57a_router_stage_candidates = [
    Path(
        "/kaggle/working/phase56_experimental_submission/"
        "models/phase30_acquisition_router.npz"
    ),
    Path(
        "/kaggle/working/phase43_experimental_submission/"
        "models/phase30_acquisition_router.npz"
    ),
]
phase57a_router_path = next(
    (
        path for path in phase57a_router_stage_candidates
        if path.is_file()
    ),
    None,
)

phase57a_router_source = None
if phase57a_router_path is not None:
    phase57a_router_asset = np.load(
        phase57a_router_path, allow_pickle=False
    )
    phase57a_router_source = str(phase57a_router_path)
else:
    phase57a_archive_candidates = [
        Path("/kaggle/working/phase56_submission.zip"),
        Path("/kaggle/working/phase43_submission.zip"),
    ]
    phase57a_archive_path = next(
        (path for path in phase57a_archive_candidates if path.is_file()),
        None,
    )
    assert phase57a_archive_path is not None, {
        "message": "Could not resolve the accepted acquisition router."
    }
    with zipfile.ZipFile(
        phase57a_archive_path, mode="r"
    ) as phase57a_archive:
        phase57a_router_bytes = phase57a_archive.read(
            "models/phase30_acquisition_router.npz"
        )
    phase57a_router_asset = np.load(
        io.BytesIO(phase57a_router_bytes), allow_pickle=False
    )
    phase57a_router_source = (
        f"{phase57a_archive_path}:"
        "models/phase30_acquisition_router.npz"
    )

with phase57a_router_asset as phase57a_router:
    phase57a_header_columns = [
        str(value)
        for value in phase57a_router["header_columns"].tolist()
    ]
    phase57a_prototypes = np.asarray(
        phase57a_router["prototypes"], dtype=np.float64
    )
    phase57a_prototype_groups = np.asarray(
        phase57a_router["prototype_groups"], dtype=np.int64
    )
    phase57a_feature_mean = np.asarray(
        phase57a_router["feature_mean"], dtype=np.float64
    )
    phase57a_feature_scale = np.asarray(
        phase57a_router["feature_scale"], dtype=np.float64
    )
    phase57a_fold_for_group = np.asarray(
        phase57a_router["fold_for_group"], dtype=np.int64
    )
    phase57a_rounding_decimals = int(
        phase57a_router["rounding_decimals"][0]
    )

assert len(phase57a_header_columns) == 10
assert phase57a_prototypes.shape == (137, 10)
assert phase57a_prototype_groups.shape == (137,)
assert phase57a_feature_mean.shape == (10,)
assert phase57a_feature_scale.shape == (10,)
assert phase57a_fold_for_group.shape == (15,)
assert np.all(phase57a_feature_scale > 0.0)
assert np.all(np.isin(phase57a_fold_for_group, [0, 1, 2]))
assert phase57a_rounding_decimals == 6


# -------------------------------------------------------------------------
# Reconstruct the only eligible Phase12c OOF prediction. Demonstrate why the
# apparent complete-ensemble training score cannot be used for selection.
# -------------------------------------------------------------------------

phase57a_fold_probability_path = Path(
    PHASE57A_CONFIG["phase12_fold_probability_file"]
)
phase57a_phase12_oof_path = Path(
    PHASE57A_CONFIG["phase12_oof_file"]
)
phase57a_highres_cache_path = Path(
    PHASE57A_CONFIG["highres_cache_file"]
)
assert phase57a_fold_probability_path.is_file()
assert phase57a_phase12_oof_path.is_file()
assert phase57a_highres_cache_path.is_file()

phase57a_fold_probability = np.load(
    phase57a_fold_probability_path, mmap_mode="r"
)
phase57a_phase12_oof = np.asarray(
    np.load(phase57a_phase12_oof_path, mmap_mode="r"),
    dtype=np.float64,
).reshape(-1)
phase57a_highres_cache = np.load(
    phase57a_highres_cache_path, mmap_mode="r"
)

assert phase57a_fold_probability.shape == (3, 1362)
assert phase57a_fold_probability.dtype == np.float64
assert phase57a_phase12_oof.shape == (1362,)
assert phase57a_highres_cache.shape == (1362, 80, 80, 80)
assert phase57a_highres_cache.dtype == np.float16

phase57a_original_fold = phase57a_fold_for_group[phase57a_groups]
assert np.bincount(
    phase57a_original_fold, minlength=3
).tolist() == [467, 443, 452]

phase57a_routed_reconstruction = np.asarray(
    phase57a_fold_probability[
        phase57a_original_fold,
        np.arange(1362, dtype=np.int64),
    ],
    dtype=np.float64,
)
phase57a_routed_reconstruction_error = np.abs(
    phase57a_routed_reconstruction - phase57a_phase12_oof
)
assert float(np.max(phase57a_routed_reconstruction_error)) <= 1.0e-12

phase57a_phase12_metrics = phase57a_metrics(phase57a_phase12_oof)
assert abs(
    phase57a_phase12_metrics["log_loss"] - 0.30761581113729136
) <= 1.0e-9
assert abs(
    phase57a_phase12_metrics["auroc"] - 0.9389144654498752
) <= 1.0e-9

phase57a_phase43_metrics = phase57a_metrics(phase57a_phase43_anchor)
assert abs(
    phase57a_phase43_metrics["log_loss"] - 0.29016628416289664
) <= 5.0e-7
assert abs(
    phase57a_phase43_metrics["auroc"] - 0.9464742438589044
) <= 5.0e-7

phase57a_contaminated_complete_ensemble = np.mean(
    np.asarray(phase57a_fold_probability, dtype=np.float64),
    axis=0,
)
assert phase57a_contaminated_complete_ensemble.shape == (1362,)
assert np.all(np.isfinite(
    phase57a_contaminated_complete_ensemble
))
assert np.all(
    (phase57a_contaminated_complete_ensemble > 0.0)
    & (phase57a_contaminated_complete_ensemble < 1.0)
)
phase57a_contaminated_metrics = phase57a_metrics(
    phase57a_contaminated_complete_ensemble
)


# -------------------------------------------------------------------------
# Freeze the 15-fold leave-one-acquisition-group-out confirmation partition.
# No partition search is performed. A future confirmation must retrain both
# the baseline and the frozen candidate without the held-out group.
# -------------------------------------------------------------------------

phase57a_group_records = []
for phase57a_group in range(15):
    phase57a_valid_mask = phase57a_groups == phase57a_group
    phase57a_train_mask = ~phase57a_valid_mask
    assert np.any(phase57a_valid_mask)
    assert not np.any(
        np.isin(
            phase57a_groups[phase57a_train_mask],
            [phase57a_group],
        )
    )
    phase57a_group_records.append({
        "held_out_group": int(phase57a_group),
        "train_n": int(np.sum(phase57a_train_mask)),
        "valid_n": int(np.sum(phase57a_valid_mask)),
        "valid_prevalence": float(np.mean(
            phase57a_labels[phase57a_valid_mask]
        )),
        "original_route_fold": int(
            phase57a_fold_for_group[phase57a_group]
        ),
        "prototype_count": int(np.sum(
            phase57a_prototype_groups == phase57a_group
        )),
    })

assert len(phase57a_group_records) == 15
assert sum(record["valid_n"] for record in phase57a_group_records) == 1362
assert all(record["prototype_count"] > 0 for record in (
    phase57a_group_records
))

phase57a_partition_payload = {
    "schema_version": "phase57_transport_logo_v1",
    "case_count": 1362,
    "group_count": 15,
    "held_out_groups": list(range(15)),
    "original_fold_for_group": phase57a_fold_for_group.tolist(),
    "selection_partition": (
        "accepted_original_three_group_disjoint_folds"
    ),
    "confirmation_partition": (
        "fifteen_leave_one_acquisition_group_out_folds"
    ),
    "partition_search_performed": False,
}
phase57a_partition_sha256 = hashlib.sha256(
    json.dumps(
        phase57a_partition_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


# -------------------------------------------------------------------------
# Pre-register the controlled model experiment and rule-compliance boundary.
# -------------------------------------------------------------------------

phase57a_experiment_plan = {
    "phase": "phase57_scanner_robust_bilateral_3d_model",
    "input": "one_independently_preprocessed_80x80x80_volume",
    "input_channels": [
        "original",
        "left_right_symmetric",
        "absolute_left_right_antisymmetric",
    ],
    "normalization": (
        "per_case_frozen_preprocessing_plus_group_norm; "
        "no_test_batch_statistics"
    ),
    "architecture_family": (
        "compact_multiscale_3d_residual_encoder_with_bilateral_pooling"
    ),
    "training_objectives": [
        "group_balanced_binary_cross_entropy",
        "group_dro_over_training_acquisition_groups",
        "bounded_within_training_group_pairwise_rank_loss",
    ],
    "scanner_augmentations": [
        "per_case_intensity_scale_and_gamma",
        "poisson_and_gaussian_noise",
        "gaussian_and_anisotropic_blur",
        "resolution_downsample_then_restore",
        "small_rigid_translation_and_rotation",
        "random_left_right_reflection",
    ],
    "selection": (
        "original_three_acquisition_group_disjoint_folds_only"
    ),
    "confirmation": (
        "retrain_frozen_baseline_and_frozen_candidate_in_each_of_15_LOGO_folds"
    ),
    "deployment_if_confirmed": (
        "fixed_member_ensemble_trained_only_on_training_data"
    ),
    "soft_routing_status": (
        "deferred_until_transport_safe_oof_residuals_exist"
    ),
}

phase57a_contract_payload = {
    "schema_version": "phase57_transport_reset_contract_v1",
    "partition_sha256": phase57a_partition_sha256,
    "experiment_plan": phase57a_experiment_plan,
    "selection_thresholds": PHASE57A_CONFIG["selection_thresholds"],
    "logo_confirmation_thresholds": PHASE57A_CONFIG[
        "logo_confirmation_thresholds"
    ],
    "stretch_target": {
        "maximum_log_loss": 0.24,
        "minimum_auroc": 0.959,
        "used_as_hyperparameter": False,
        "guaranteed": False,
    },
    "test_data_rule": {
        "each_test_case_processed_independently": True,
        "test_batch_statistics": False,
        "test_time_training": False,
        "test_time_unsupervised_adaptation": False,
        "pseudo_labeling": False,
        "model_weights_independent_of_test_set": True,
        "fitted_feature_parameters_independent_of_test_set": True,
    },
}
phase57a_contract_sha256 = hashlib.sha256(
    json.dumps(
        phase57a_contract_payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


def phase57a_atomic_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def phase57a_atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


phase57a_atomic_npz(
    PHASE57A_CONFIG["partition_file"],
    contract_sha256=np.asarray(phase57a_contract_sha256),
    partition_sha256=np.asarray(phase57a_partition_sha256),
    held_out_groups=np.arange(15, dtype=np.int8),
    original_fold_for_group=phase57a_fold_for_group.astype(np.int8),
)
phase57a_atomic_json(
    PHASE57A_CONFIG["contract_file"],
    {
        **phase57a_contract_payload,
        "contract_sha256": phase57a_contract_sha256,
        "contains_labels": False,
        "contains_probabilities": False,
        "contains_case_indices": False,
        "contains_voxel_data": False,
    },
)

PHASE57_TRANSPORT_CONFIG_PRIVATE = json.loads(json.dumps(
    PHASE57A_CONFIG
))
PHASE57_TRANSPORT_STATE_PRIVATE = {
    "contract_sha256": phase57a_contract_sha256,
    "partition_sha256": phase57a_partition_sha256,
    "labels": phase57a_labels.copy(),
    "groups": phase57a_groups.copy(),
    "original_fold": phase57a_original_fold.copy(),
    "original_fold_for_group": phase57a_fold_for_group.copy(),
    "phase12_oof_probability": phase57a_phase12_oof.copy(),
    "phase43_anchor_probability": phase57a_phase43_anchor.copy(),
    "highres_cache_file": str(phase57a_highres_cache_path),
    "logo_held_out_groups": np.arange(15, dtype=np.int64),
}


phase57a_report = {
    "phase": "phase57_transport_reset_and_logo_contract",
    "status": "accepted_ready_for_scanner_robust_architecture_contract",
    "motivation": {
        "phase42_public": PHASE57A_CONFIG["public_reference"]["phase42"],
        "phase56_public": PHASE57A_CONFIG["public_reference"]["phase56"],
        "public_log_loss_gain": 0.0034,
        "public_auroc_gain": 0.0046,
        "public_score_used_for_problem_diagnosis": True,
        "public_score_used_for_candidate_or_hyperparameter_selection": False,
        "identified_bottleneck": (
            "acquisition_domain_transport_and_exact_route_abstention"
        ),
    },
    "training_data": {
        "case_count": 1362,
        "normal_count": int(np.sum(phase57a_labels == 0)),
        "pathologic_count": int(np.sum(phase57a_labels == 1)),
        "acquisition_group_count": 15,
        "highres_cache_shape": list(phase57a_highres_cache.shape),
        "highres_cache_dtype": str(phase57a_highres_cache.dtype),
        "highres_cache_memory_mapped": True,
    },
    "exact_router_audit": {
        "header_feature_count": len(phase57a_header_columns),
        "header_features": phase57a_header_columns,
        "prototype_count": int(phase57a_prototypes.shape[0]),
        "rounding_decimals": phase57a_rounding_decimals,
        "exact_match_required": True,
        "recognized_group_count": 15,
        "unknown_protocol_current_policy": "anchor_only",
        "router_source": phase57a_router_source,
    },
    "eligible_reference_metrics": {
        "phase12c_group_held_out_oof": phase57a_phase12_metrics,
        "phase43_deployment_matched_oof": phase57a_phase43_metrics,
        "routed_phase12c_reconstruction_maximum_error": float(
            np.max(phase57a_routed_reconstruction_error)
        ),
    },
    "ineligible_raw_phase12c_complete_ensemble_training_diagnostic": {
        **phase57a_contaminated_metrics,
        "label_leakage_contaminated": True,
        "eligible_for_selection": False,
        "reason": (
            "two_of_three_fold_models_saw_each_training_group; "
            "deployment_ensemble_requires_retraining_for_honest_evaluation"
        ),
        "not_the_phase42_full_ensemble_diagnostic": True,
    },
    "selection_validation": {
        "fold_count": 3,
        "held_out_unit": "sets_of_complete_acquisition_groups",
        "fold_case_counts": np.bincount(
            phase57a_original_fold, minlength=3
        ).astype(int).tolist(),
        "shared_acquisition_groups_between_train_and_valid": 0,
        "existing_partition_reused_for_initial_screen": True,
    },
    "logo_confirmation": {
        "fold_count": 15,
        "held_out_unit": "one_complete_acquisition_group",
        "partition_search_performed": False,
        "baseline_retraining_required_in_every_fold": True,
        "candidate_retraining_required_in_every_fold": True,
        "records": phase57a_group_records,
    },
    "experiment_plan": phase57a_experiment_plan,
    "selection_thresholds": PHASE57A_CONFIG["selection_thresholds"],
    "logo_confirmation_thresholds": PHASE57A_CONFIG[
        "logo_confirmation_thresholds"
    ],
    "soft_routing_old_components_allowed_now": False,
    "soft_routing_reason": (
        "existing non-routed fold models can contain held-out-group cases"
    ),
    "test_data_rule": phase57a_contract_payload["test_data_rule"],
    "contract_sha256": phase57a_contract_sha256,
    "partition_sha256": phase57a_partition_sha256,
    "labels_used_for_training_metric_audit": True,
    "labels_used_for_partition_search": False,
    "training_voxel_cache_metadata_read": True,
    "training_voxel_arrays_read": False,
    "training_nifti_files_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(
        time.perf_counter() - phase57a_started, 3
    ),
}

PHASE57_TRANSPORT_REPORT_PRIVATE = dict(phase57a_report)

print("BEGIN SANITIZED_PHASE57_TRANSPORT_CONTRACT")
print(json.dumps(phase57a_report, indent=2))
print("END SANITIZED_PHASE57_TRANSPORT_CONTRACT")
