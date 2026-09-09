from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 161A — historical OOF complementarity and provenance audit.
#
# Run after Cell 160A in the same live kernel. This cell performs no training,
# reads no NIfTI/voxel/test/smoke data, and writes no case-level artifact. It
# examines only an explicit allowlist of existing training OOF vectors. Oracle
# blends are diagnostic ceilings and are never eligible for deployment.

phase61a_started = time.perf_counter()

assert isinstance(globals().get("PHASE60_INDEPENDENT_3D_REPORT_PRIVATE"), dict)
assert PHASE60_INDEPENDENT_3D_REPORT_PRIVATE["status"] == (
    "outer_fold_transport_gate_failed_stop_candidate"
)
assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)

phase61a_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase61a_labels = np.asarray(phase61a_state["labels"], dtype=np.int64).reshape(-1)
phase61a_groups = np.asarray(phase61a_state["groups"], dtype=np.int64).reshape(-1)
phase61a_anchor = np.asarray(
    phase61a_state["anchor_probability"], dtype=np.float64
).reshape(-1)

if "phase57c_original_fold" in globals():
    phase61a_folds = np.asarray(phase57c_original_fold, dtype=np.int64).reshape(-1)
elif isinstance(globals().get("PHASE57_TRANSPORT_STATE_PRIVATE"), dict):
    phase61a_folds = np.asarray(
        PHASE57_TRANSPORT_STATE_PRIVATE["original_fold"], dtype=np.int64
    ).reshape(-1)
else:
    raise AssertionError("Accepted three-fold acquisition assignment is unavailable.")

assert phase61a_labels.shape == phase61a_groups.shape == phase61a_anchor.shape == (1362,)
assert phase61a_folds.shape == (1362,)
assert set(np.unique(phase61a_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase61a_groups), np.arange(15))
assert np.array_equal(np.unique(phase61a_folds), np.arange(3))
assert np.all(np.isfinite(phase61a_anchor))
assert np.all((phase61a_anchor > 0.0) & (phase61a_anchor < 1.0))


PHASE61A_CONFIG = {
    "schema_version": "phase61_historical_oof_complementarity_audit_v1",
    "case_count": 1362,
    "major_groups": [1, 3],
    "confidence_threshold": 0.80,
    "hard_case_fraction": 0.05,
    "probability_clip": 1.0e-7,
    "oracle_logit_blend_grid": [0.10, 0.20, 0.30, 0.40, 0.50, 0.75, 1.00],
    "family_screen": {
        "minimum_major_group_auroc_gain": 0.003,
        "maximum_individual_major_group_auroc_regret": 0.002,
        "minimum_oracle_blend_log_loss_gain": 0.002,
        "maximum_oracle_fold_log_loss_regret": 0.003,
    },
    "contract_file": "/kaggle/working/phase61_historical_oof_complementarity_contract.json",
}


# Eligibility here means only that the vector has a previously reconstructed
# deployment-matched OOF provenance. It does not make an oracle blend eligible.
# Unknown entries can nominate a family for provenance review/retraining only.
PHASE61A_CANDIDATES = [
    {
        "name": "phase12c_oof",
        "paths": ["/kaggle/working/phase32_phase12c_oof_float64.npy"],
        "provenance": "verified_deployment_matched_oof",
        "deployment_eligible_vector": True,
    },
    {
        "name": "phase31_classical_oof",
        "paths": ["/kaggle/working/phase31_classical_oof_private.npy"],
        "provenance": "unverified_requires_phase31_selection_contract_review",
        "deployment_eligible_vector": False,
    },
    {
        "name": "phase33_component_oof",
        "paths": [
            "/kaggle/working/phase33_private_checkpoint/phase33_component_oof_float64.npy",
            "/kaggle/working/phase33_component_oof_float64.npy",
        ],
        "provenance": "verified_component_oof_reconstructed_in_phase57_restore",
        "deployment_eligible_vector": True,
    },
    {
        "name": "phase39_oof",
        "paths": [
            "/kaggle/working/phase39_private_checkpoint/phase39_oof_float64.npy",
            "/kaggle/working/phase39_oof_float64.npy",
        ],
        "provenance": "verified_deployment_matched_oof_reconstructed_in_phase57_restore",
        "deployment_eligible_vector": True,
    },
    {
        "name": "phase41_selected_raw_oof",
        "paths": [
            "/kaggle/working/phase41_selected_development_raw_oof_float32.npy"
        ],
        "provenance": "unverified_requires_phase41_selection_contract_review",
        "deployment_eligible_vector": False,
    },
    {
        "name": "phase42_nested_oof",
        "paths": ["/kaggle/working/phase42_nested_oof_float64.npy"],
        "provenance": "unverified_name_indicates_nested_but_contract_review_required",
        "deployment_eligible_vector": False,
    },
    {
        "name": "phase42_shared_oof",
        "paths": ["/kaggle/working/phase42_shared_oof_float64.npy"],
        "provenance": "diagnostic_shared_selection_not_assumed_deployment_eligible",
        "deployment_eligible_vector": False,
    },
]


def phase61a_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE61A_CONFIG["probability_clip"],
        1.0 - PHASE61A_CONFIG["probability_clip"],
    )
    return np.log(probability) - np.log1p(-probability)


def phase61a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    result = np.empty_like(logit)
    positive = logit >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def phase61a_auc(labels, score):
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive_n = int(np.sum(labels == 1))
    negative_n = int(np.sum(labels == 0))
    if positive_n == 0 or negative_n == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    u_statistic = float(np.sum(ranks[labels == 1])) - positive_n * (positive_n + 1) / 2.0
    return float(u_statistic / (positive_n * negative_n))


def phase61a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE61A_CONFIG["probability_clip"],
        1.0 - PHASE61A_CONFIG["probability_clip"],
    )
    case_loss = -(
        labels * np.log(probability) + (1 - labels) * np.log1p(-probability)
    )
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(case_loss)),
        "auroc": phase61a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase61a_resolve_vector(specification):
    existing = [Path(path) for path in specification["paths"] if Path(path).is_file()]
    if not existing:
        return None, None, "missing"
    reference = np.asarray(
        np.load(existing[0], allow_pickle=False, mmap_mode="r"), dtype=np.float64
    ).reshape(-1)
    if reference.shape != (PHASE61A_CONFIG["case_count"],):
        return None, str(existing[0]), "wrong_shape"
    if not np.all(np.isfinite(reference)):
        return None, str(existing[0]), "nonfinite"
    for duplicate_path in existing[1:]:
        duplicate = np.asarray(
            np.load(duplicate_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64
        ).reshape(-1)
        if duplicate.shape != reference.shape:
            return None, str(existing[0]), "duplicate_shape_disagreement"
        if float(np.max(np.abs(duplicate - reference))) > 1.0e-10:
            return None, str(existing[0]), "duplicate_value_disagreement"
    return reference, str(existing[0]), "accepted"


anchor_metrics = phase61a_metrics(phase61a_labels, phase61a_anchor)
anchor_group_metrics = {
    str(group): phase61a_metrics(
        phase61a_labels[phase61a_groups == group],
        phase61a_anchor[phase61a_groups == group],
    )
    for group in PHASE61A_CONFIG["major_groups"]
}
major_mask = np.isin(phase61a_groups, PHASE61A_CONFIG["major_groups"])
anchor_major_metrics = phase61a_metrics(
    phase61a_labels[major_mask], phase61a_anchor[major_mask]
)

anchor_predicted = (phase61a_anchor >= 0.5).astype(np.int64)
anchor_confident = (
    (phase61a_anchor >= PHASE61A_CONFIG["confidence_threshold"])
    | (phase61a_anchor <= 1.0 - PHASE61A_CONFIG["confidence_threshold"])
)
anchor_confident_error = anchor_confident & (anchor_predicted != phase61a_labels)
anchor_case_loss = -(
    phase61a_labels * np.log(np.clip(phase61a_anchor, 1.0e-7, 1.0 - 1.0e-7))
    + (1 - phase61a_labels)
    * np.log1p(-np.clip(phase61a_anchor, 1.0e-7, 1.0 - 1.0e-7))
)
hard_count = max(1, int(np.ceil(PHASE61A_CONFIG["hard_case_fraction"] * 1362)))
hard_indices = np.argsort(anchor_case_loss, kind="mergesort")[-hard_count:]
hard_mask = np.zeros(1362, dtype=bool)
hard_mask[hard_indices] = True

candidate_records = []
promising_families = []

for specification in PHASE61A_CANDIDATES:
    vector, source, load_status = phase61a_resolve_vector(specification)
    record = {
        "name": specification["name"],
        "available": vector is not None,
        "load_status": load_status,
        "source": source,
        "provenance": specification["provenance"],
        "deployment_eligible_vector": specification["deployment_eligible_vector"],
    }
    if vector is None:
        candidate_records.append(record)
        continue

    is_probability = bool(np.all((vector > 0.0) & (vector < 1.0)))
    record["value_type"] = "probability" if is_probability else "unbounded_score"
    record["pooled_auroc"] = phase61a_auc(phase61a_labels, vector)
    record["anchor_score_pearson"] = float(np.corrcoef(
        phase61a_logit(phase61a_anchor),
        phase61a_logit(vector) if is_probability else vector,
    )[0, 1])

    group_metrics = {}
    for group in PHASE61A_CONFIG["major_groups"]:
        mask = phase61a_groups == group
        if is_probability:
            group_metrics[str(group)] = phase61a_metrics(
                phase61a_labels[mask], vector[mask]
            )
        else:
            group_metrics[str(group)] = {
                "n": int(np.sum(mask)),
                "auroc": phase61a_auc(phase61a_labels[mask], vector[mask]),
            }
    record["major_group_metrics"] = group_metrics

    if not is_probability:
        record["screen_decision"] = "score_only_requires_probability_provenance_and_calibration"
        candidate_records.append(record)
        continue

    candidate_metrics = phase61a_metrics(phase61a_labels, vector)
    candidate_major_metrics = phase61a_metrics(
        phase61a_labels[major_mask], vector[major_mask]
    )
    candidate_predicted = (vector >= 0.5).astype(np.int64)
    rescued_confident_errors = anchor_confident_error & (
        candidate_predicted == phase61a_labels
    )
    candidate_case_loss = -(
        phase61a_labels * np.log(np.clip(vector, 1.0e-7, 1.0 - 1.0e-7))
        + (1 - phase61a_labels)
        * np.log1p(-np.clip(vector, 1.0e-7, 1.0 - 1.0e-7))
    )
    record.update({
        "pooled": candidate_metrics,
        "major_groups_combined": candidate_major_metrics,
        "major_group_auroc_gain": (
            candidate_major_metrics["auroc"] - anchor_major_metrics["auroc"]
        ),
        "confident_anchor_errors": int(np.sum(anchor_confident_error)),
        "confident_anchor_errors_rescued": int(np.sum(rescued_confident_errors)),
        "hardest_anchor_5_percent_mean_log_loss": float(np.mean(candidate_case_loss[hard_mask])),
        "remaining_95_percent_mean_log_loss": float(np.mean(candidate_case_loss[~hard_mask])),
    })

    oracle_records = []
    anchor_logit = phase61a_logit(phase61a_anchor)
    candidate_logit = phase61a_logit(vector)
    for alpha in PHASE61A_CONFIG["oracle_logit_blend_grid"]:
        blend = phase61a_sigmoid(
            (1.0 - float(alpha)) * anchor_logit + float(alpha) * candidate_logit
        )
        pooled = phase61a_metrics(phase61a_labels, blend)
        folds = [
            phase61a_metrics(
                phase61a_labels[phase61a_folds == fold],
                blend[phase61a_folds == fold],
            )
            for fold in range(3)
        ]
        anchor_folds = [
            phase61a_metrics(
                phase61a_labels[phase61a_folds == fold],
                phase61a_anchor[phase61a_folds == fold],
            )
            for fold in range(3)
        ]
        oracle_records.append({
            "alpha": float(alpha),
            "log_loss": pooled["log_loss"],
            "auroc": pooled["auroc"],
            "log_loss_gain": anchor_metrics["log_loss"] - pooled["log_loss"],
            "maximum_fold_log_loss_regret": float(max(
                folds[fold]["log_loss"] - anchor_folds[fold]["log_loss"]
                for fold in range(3)
            )),
        })
    best_oracle = min(
        oracle_records, key=lambda item: (item["log_loss"], -item["auroc"])
    )
    record["diagnostic_oracle_blend"] = best_oracle
    record["oracle_blend_deployment_eligible"] = False

    threshold = PHASE61A_CONFIG["family_screen"]
    individual_group_regrets = [
        anchor_group_metrics[str(group)]["auroc"]
        - group_metrics[str(group)]["auroc"]
        for group in PHASE61A_CONFIG["major_groups"]
    ]
    family_promising = bool(
        record["major_group_auroc_gain"]
        >= threshold["minimum_major_group_auroc_gain"]
        and max(individual_group_regrets)
        <= threshold["maximum_individual_major_group_auroc_regret"]
        and best_oracle["log_loss_gain"]
        >= threshold["minimum_oracle_blend_log_loss_gain"]
        and best_oracle["maximum_fold_log_loss_regret"]
        <= threshold["maximum_oracle_fold_log_loss_regret"]
    )
    record["family_screen_promising"] = family_promising
    record["screen_decision"] = (
        "provenance_review_then_strict_nested_retraining"
        if family_promising
        else "no_historical_complementarity_signal"
    )
    if family_promising:
        promising_families.append(specification["name"])
    candidate_records.append(record)


phase61a_status = (
    "historical_family_signal_found_requires_provenance_and_nested_retraining"
    if promising_families
    else "no_historical_family_signal_ready_for_anatomy_invariant_retraining_design"
)

contract_core = {
    "schema_version": PHASE61A_CONFIG["schema_version"],
    "status": phase61a_status,
    "config": PHASE61A_CONFIG,
    "candidate_names": [item["name"] for item in PHASE61A_CANDIDATES],
    "oracle_blends_deployment_eligible": False,
    "test_data_used": False,
}
contract_sha256 = hashlib.sha256(json.dumps(
    contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE61A_CONFIG["contract_file"])
contract_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = contract_path.with_suffix(contract_path.suffix + ".tmp")
temporary_path.write_text(json.dumps({
    **contract_core,
    "contract_sha256": contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(temporary_path, contract_path)

phase61a_report = {
    "phase": "phase61_historical_oof_complementarity_and_provenance_audit",
    "status": phase61a_status,
    "anchor": {
        "pooled": anchor_metrics,
        "major_groups_combined": anchor_major_metrics,
        "major_group_metrics": anchor_group_metrics,
        "confident_error_count": int(np.sum(anchor_confident_error)),
    },
    "candidates": candidate_records,
    "promising_family_names": promising_families,
    "interpretation_contract": {
        "historical_vectors_used_for_diagnostic_only": True,
        "oracle_blends_used_for_candidate_selection": False,
        "oracle_blends_eligible_for_deployment": False,
        "unknown_provenance_vectors_eligible_for_deployment": False,
        "promotion_requires_contract_review": True,
        "promotion_requires_strict_group_held_out_retraining": True,
    },
    "training_performed": False,
    "training_voxel_data_read": False,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "uids_displayed": False,
    "case_level_predictions_displayed": False,
    "case_level_predictions_exported": False,
    "contract_sha256": contract_sha256,
    "contract_file": str(contract_path),
    "elapsed_seconds": round(time.perf_counter() - phase61a_started, 3),
}

PHASE61_HISTORICAL_COMPLEMENTARITY_REPORT_PRIVATE = phase61a_report

print("BEGIN SANITIZED_PHASE61_HISTORICAL_OOF_COMPLEMENTARITY_AUDIT")
print(json.dumps(phase61a_report, indent=2))
print("END SANITIZED_PHASE61_HISTORICAL_OOF_COMPLEMENTARITY_AUDIT")

