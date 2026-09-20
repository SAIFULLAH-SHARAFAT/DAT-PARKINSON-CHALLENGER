from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 163A — sparse confidence-risk localization and ceiling audit.
#
# Run after Cell 162B in the same live kernel. This is diagnostic only. It does
# not fit a model, choose a deployable threshold, change the anchor, read NIfTI
# files, or touch test data. It asks a narrower question after Phase 62B failed:
# can label-blind disagreement quantities concentrate the anchor's catastrophic
# mistakes into a small enough set that a separately nested sparse intervention
# would be worth building?

phase63a_started = time.perf_counter()

assert isinstance(globals().get("PHASE62_VALIDATION_RESET_REPORT_PRIVATE"), dict)
assert PHASE62_VALIDATION_RESET_REPORT_PRIVATE["status"] == (
    "accepted_validation_reset_ready_for_phase62b"
)
assert isinstance(globals().get("PHASE62_VALIDATION_RESET_STATE_PRIVATE"), dict)
assert isinstance(globals().get("PHASE62B_MULTITEMPLATE_REPORT_PRIVATE"), dict)
assert PHASE62B_MULTITEMPLATE_REPORT_PRIVATE["status"] == (
    "validation_gate_failed_stop_multitemplate_candidate"
)
assert isinstance(globals().get("PHASE62B_MULTITEMPLATE_STATE_PRIVATE"), dict)

phase63a_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE
phase63a_multitemplate = PHASE62B_MULTITEMPLATE_STATE_PRIVATE

phase63a_labels = np.asarray(
    phase63a_validation["labels"], dtype=np.int64
).reshape(-1)
phase63a_groups = np.asarray(
    phase63a_validation["groups"], dtype=np.int64
).reshape(-1)
phase63a_folds = np.asarray(
    phase63a_validation["original_fold"], dtype=np.int64
).reshape(-1)
phase63a_anchor = np.asarray(
    phase63a_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase63a_negative_control = np.asarray(
    phase63a_validation["negative_control_probability"], dtype=np.float64
).reshape(-1)
phase63a_domain_id = np.asarray(
    phase63a_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase63a_domain_definitions = phase63a_validation["stress_domain_definitions"]

phase63a_anatomy = np.asarray(
    phase63a_multitemplate["anatomy_oof_probability"], dtype=np.float64
).reshape(-1)
phase63a_phase62_candidate = np.asarray(
    phase63a_multitemplate["candidate_oof_probability"], dtype=np.float64
).reshape(-1)
phase63a_sensitivity = np.asarray(
    phase63a_multitemplate["sensitivity_oof_probability"], dtype=np.float64
).reshape(-1)
phase63a_specificity = np.asarray(
    phase63a_multitemplate["specificity_oof_probability"], dtype=np.float64
).reshape(-1)

phase63a_expected_n = 1362
phase63a_vectors = [
    phase63a_labels,
    phase63a_groups,
    phase63a_folds,
    phase63a_anchor,
    phase63a_negative_control,
    phase63a_domain_id,
    phase63a_anatomy,
    phase63a_phase62_candidate,
    phase63a_sensitivity,
    phase63a_specificity,
]
assert all(vector.shape == (phase63a_expected_n,) for vector in phase63a_vectors)
assert set(np.unique(phase63a_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase63a_groups), np.arange(15))
assert np.array_equal(np.unique(phase63a_folds), np.arange(3))
for probability in [
    phase63a_anchor,
    phase63a_negative_control,
    phase63a_anatomy,
    phase63a_phase62_candidate,
    phase63a_sensitivity,
    phase63a_specificity,
]:
    assert np.all(np.isfinite(probability))
    assert np.all((probability > 0.0) & (probability < 1.0))


PHASE63A_CONFIG = {
    "schema_version": "phase63_sparse_confidence_risk_ceiling_audit_v1",
    "probability_clip": 1.0e-7,
    "confidence_probability": 0.80,
    "risk_confidence_reference": 0.80,
    "risk_confidence_saturation": 0.99,
    "risk_boundaryward_width": 2.0,
    "top_fractions": [0.025, 0.05, 0.075, 0.10, 0.15, 0.20],
    "diagnostic_shrink_strengths": [0.10, 0.20, 0.30, 0.40, 0.50],
    "diagnostic_anatomy_strengths": [0.10, 0.20, 0.30],
    "diagnostic_anatomy_residual_cap": 1.50,
    "major_groups": [1, 3],
    "minimum_primary_confident_error_auroc": 0.70,
    "minimum_primary_top10_confident_error_recall": 0.40,
    "minimum_diagnostic_ceiling_log_loss_gain": 0.003,
    "maximum_diagnostic_fold_regret": 0.002,
    "maximum_diagnostic_major_group_regret": 0.002,
    "contract_file": "/kaggle/working/phase63_sparse_confidence_risk_ceiling_contract.json",
}


def phase63a_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE63A_CONFIG["probability_clip"],
        1.0 - PHASE63A_CONFIG["probability_clip"],
    )


def phase63a_logit(probability):
    probability = phase63a_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase63a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase63a_auc(labels, score):
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
    u_statistic = float(np.sum(ranks[labels == 1]))
    u_statistic -= positive_n * (positive_n + 1) / 2.0
    return float(u_statistic / (positive_n * negative_n))


def phase63a_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase63a_clip(probability)
    return -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )


def phase63a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase63a_clip(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase63a_case_loss(labels, probability))),
        "auroc": phase63a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase63a_compare(labels, anchor, candidate):
    anchor_metrics = phase63a_metrics(labels, anchor)
    candidate_metrics = phase63a_metrics(labels, candidate)
    return {
        "n": anchor_metrics["n"],
        "anchor_log_loss": anchor_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            anchor_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "anchor_auroc": anchor_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": float(
            candidate_metrics["auroc"] - anchor_metrics["auroc"]
        ),
        "brier_gain": float(anchor_metrics["brier"] - candidate_metrics["brier"]),
    }


def phase63a_top_mask(score, fraction):
    score = np.asarray(score, dtype=np.float64)
    count = max(1, int(np.ceil(float(fraction) * score.size)))
    order = np.argsort(-score, kind="mergesort")
    mask = np.zeros(score.size, dtype=bool)
    mask[order[:count]] = True
    return mask


def phase63a_safe_divide(numerator, denominator):
    return None if float(denominator) == 0.0 else float(numerator / denominator)


phase63a_anchor_logit = phase63a_logit(phase63a_anchor)
phase63a_anatomy_logit = phase63a_logit(phase63a_anatomy)
phase63a_sensitivity_logit = phase63a_logit(phase63a_sensitivity)
phase63a_specificity_logit = phase63a_logit(phase63a_specificity)

# Recover the Phase31 logit only for an explicitly excluded diagnostic score.
# Cell 162A defined z_control = 0.8*z_anchor + 0.2*z_phase31.
phase63a_phase31_logit = (
    phase63a_logit(phase63a_negative_control)
    - 0.80 * phase63a_anchor_logit
) / 0.20

phase63a_direction = np.where(phase63a_anchor_logit >= 0.0, 1.0, -1.0)
phase63a_anatomy_boundaryward = np.maximum(
    0.0,
    -phase63a_direction * (phase63a_anatomy_logit - phase63a_anchor_logit),
)
phase63a_phase31_boundaryward = np.maximum(
    0.0,
    -phase63a_direction * (phase63a_phase31_logit - phase63a_anchor_logit),
)

phase63a_confidence_floor = abs(
    phase63a_logit(PHASE63A_CONFIG["risk_confidence_reference"])
)
phase63a_confidence_ceiling = abs(
    phase63a_logit(PHASE63A_CONFIG["risk_confidence_saturation"])
)
phase63a_confidence_gate = np.clip(
    (np.abs(phase63a_anchor_logit) - phase63a_confidence_floor)
    / (phase63a_confidence_ceiling - phase63a_confidence_floor),
    0.0,
    1.0,
)
phase63a_anatomy_gate = np.clip(
    phase63a_anatomy_boundaryward
    / PHASE63A_CONFIG["risk_boundaryward_width"],
    0.0,
    1.0,
)
phase63a_phase31_gate = np.clip(
    phase63a_phase31_boundaryward
    / PHASE63A_CONFIG["risk_boundaryward_width"],
    0.0,
    1.0,
)
phase63a_sensitivity_boundaryward = np.maximum(
    0.0,
    -phase63a_direction * (phase63a_sensitivity_logit - phase63a_anchor_logit),
)
phase63a_specificity_boundaryward = np.maximum(
    0.0,
    -phase63a_direction * (phase63a_specificity_logit - phase63a_anchor_logit),
)
phase63a_sensitivity_gate = np.clip(
    phase63a_sensitivity_boundaryward
    / PHASE63A_CONFIG["risk_boundaryward_width"],
    0.0,
    1.0,
)
phase63a_specificity_gate = np.clip(
    phase63a_specificity_boundaryward
    / PHASE63A_CONFIG["risk_boundaryward_width"],
    0.0,
    1.0,
)

# This is the only score allowed to support a future Phase63B branch. It is a
# deterministic function of OOF probabilities and contains no labels, groups,
# folds, prevalence, or case identity.
phase63a_three_head_gates = np.stack([
    phase63a_anatomy_gate,
    phase63a_sensitivity_gate,
    phase63a_specificity_gate,
], axis=1)
phase63a_sorted_head_gates = np.sort(phase63a_three_head_gates, axis=1)
phase63a_consensus_gate = (
    0.50 * phase63a_sorted_head_gates[:, 0]
    + 0.50 * phase63a_sorted_head_gates[:, 1]
)
phase63a_primary_risk = (
    phase63a_confidence_gate
    * phase63a_consensus_gate
)

# Negative control: how strongly the already-failed Phase62 router moved a case.
phase63a_phase62_movement_risk = np.abs(
    phase63a_logit(phase63a_phase62_candidate) - phase63a_anchor_logit
)

# Historical diagnostic only. Phase31 provenance is unverified, so this score
# cannot advance or appear in submission code regardless of its audit result.
phase63a_unverified_three_expert_risk = (
    phase63a_confidence_gate
    * np.minimum(phase63a_anatomy_gate, phase63a_phase31_gate)
)

phase63a_scores = {
    "primary_three_anatomy_heads_boundaryward_consensus": phase63a_primary_risk,
    "failed_phase62_movement_negative_control": phase63a_phase62_movement_risk,
    "unverified_phase31_three_expert_diagnostic_only": (
        phase63a_unverified_three_expert_risk
    ),
}

phase63a_predicted = (phase63a_anchor >= 0.50).astype(np.int64)
phase63a_error = phase63a_predicted != phase63a_labels
phase63a_confident_error = (
    ((phase63a_labels == 1) & (phase63a_anchor <= 0.20))
    | ((phase63a_labels == 0) & (phase63a_anchor >= 0.80))
)
phase63a_anchor_loss = phase63a_case_loss(phase63a_labels, phase63a_anchor)
phase63a_hard_loss_threshold = float(
    np.quantile(phase63a_anchor_loss, 0.95, method="higher")
)
phase63a_hard_loss = phase63a_anchor_loss >= phase63a_hard_loss_threshold


def phase63a_score_audit(name, score):
    score = np.asarray(score, dtype=np.float64)
    assert score.shape == (phase63a_expected_n,)
    assert np.all(np.isfinite(score))
    rows = []
    total_loss = float(np.sum(phase63a_anchor_loss))
    confident_error_n = int(np.sum(phase63a_confident_error))
    all_error_n = int(np.sum(phase63a_error))
    for fraction in PHASE63A_CONFIG["top_fractions"]:
        mask = phase63a_top_mask(score, fraction)
        count = int(np.sum(mask))
        confident_captured = int(np.sum(mask & phase63a_confident_error))
        all_error_captured = int(np.sum(mask & phase63a_error))
        rows.append({
            "top_fraction": float(fraction),
            "n": count,
            "confident_errors_captured": confident_captured,
            "confident_error_recall": phase63a_safe_divide(
                confident_captured, confident_error_n
            ),
            "all_errors_captured": all_error_captured,
            "all_error_recall": phase63a_safe_divide(all_error_captured, all_error_n),
            "error_precision": float(np.mean(phase63a_error[mask])),
            "anchor_loss_share": phase63a_safe_divide(
                np.sum(phase63a_anchor_loss[mask]), total_loss
            ),
            "mean_anchor_log_loss": float(np.mean(phase63a_anchor_loss[mask])),
        })
    eligible = bool(name == "primary_three_anatomy_heads_boundaryward_consensus")
    return {
        "name": name,
        "eligible_to_support_phase63b": eligible,
        "score_nonzero_fraction": float(np.mean(score > 0.0)),
        "anchor_error_detection_auroc": phase63a_auc(
            phase63a_error.astype(np.int64), score
        ),
        "confident_error_detection_auroc": phase63a_auc(
            phase63a_confident_error.astype(np.int64), score
        ),
        "hardest_loss_5_percent_detection_auroc": phase63a_auc(
            phase63a_hard_loss.astype(np.int64), score
        ),
        "top_risk": rows if eligible else [
            row for row in rows if abs(row["top_fraction"] - 0.10) < 1.0e-12
        ],
    }


phase63a_score_reports = [
    phase63a_score_audit(name, score)
    for name, score in phase63a_scores.items()
]
phase63a_primary_report = phase63a_score_reports[0]
phase63a_primary_top10 = next(
    row for row in phase63a_primary_report["top_risk"]
    if abs(row["top_fraction"] - 0.10) < 1.0e-12
)


def phase63a_candidate_stability(candidate):
    pooled = phase63a_compare(
        phase63a_labels, phase63a_anchor, candidate
    )
    fold_rows = []
    fold_regrets = []
    for fold in range(3):
        mask = phase63a_folds == fold
        comparison = phase63a_compare(
            phase63a_labels[mask], phase63a_anchor[mask], candidate[mask]
        )
        fold_rows.append({
            "fold": int(fold),
            "n": int(np.sum(mask)),
            "log_loss_gain": comparison["log_loss_gain"],
            "auroc_gain": comparison["auroc_gain"],
            "brier_gain": comparison["brier_gain"],
        })
        fold_regrets.append(max(0.0, -comparison["log_loss_gain"]))
    major_rows = []
    major_regrets = []
    for group in PHASE63A_CONFIG["major_groups"]:
        mask = phase63a_groups == group
        comparison = phase63a_compare(
            phase63a_labels[mask], phase63a_anchor[mask], candidate[mask]
        )
        major_rows.append({
            "group": int(group),
            "n": int(np.sum(mask)),
            "log_loss_gain": comparison["log_loss_gain"],
            "auroc_gain": comparison["auroc_gain"],
            "brier_gain": comparison["brier_gain"],
        })
        major_regrets.append(max(0.0, -comparison["log_loss_gain"]))
    return {
        "pooled": pooled,
        "folds": fold_rows,
        "major_groups": major_rows,
        "maximum_fold_log_loss_regret": float(max(fold_regrets)),
        "maximum_major_group_log_loss_regret": float(max(major_regrets)),
    }


# Every result below is an in-sample diagnostic oracle over threshold and
# strength. No row is eligible for selection, deployment, or leaderboard use.
phase63a_ceiling_rows = []
phase63a_capped_anatomy_residual = np.clip(
    phase63a_anatomy_logit - phase63a_anchor_logit,
    -PHASE63A_CONFIG["diagnostic_anatomy_residual_cap"],
    PHASE63A_CONFIG["diagnostic_anatomy_residual_cap"],
)
for fraction in PHASE63A_CONFIG["top_fractions"]:
    risk_mask = phase63a_top_mask(phase63a_primary_risk, fraction)
    for strength in PHASE63A_CONFIG["diagnostic_shrink_strengths"]:
        candidate_logit = phase63a_anchor_logit.copy()
        candidate_logit[risk_mask] *= 1.0 - float(strength)
        candidate = phase63a_sigmoid(candidate_logit)
        stability = phase63a_candidate_stability(candidate)
        phase63a_ceiling_rows.append({
            "family": "risk_subset_anchor_logit_shrink",
            "top_fraction": float(fraction),
            "strength": float(strength),
            "stability": stability,
        })
    for strength in PHASE63A_CONFIG["diagnostic_anatomy_strengths"]:
        candidate_logit = phase63a_anchor_logit.copy()
        candidate_logit[risk_mask] += (
            float(strength) * phase63a_capped_anatomy_residual[risk_mask]
        )
        candidate = phase63a_sigmoid(candidate_logit)
        stability = phase63a_candidate_stability(candidate)
        phase63a_ceiling_rows.append({
            "family": "risk_subset_bounded_anatomy_residual",
            "top_fraction": float(fraction),
            "strength": float(strength),
            "stability": stability,
        })


def phase63a_oracle_key(row):
    pooled = row["stability"]["pooled"]
    return (
        pooled["log_loss_gain"],
        pooled["auroc_gain"],
        -row["stability"]["maximum_fold_log_loss_regret"],
    )


phase63a_best_oracle = max(phase63a_ceiling_rows, key=phase63a_oracle_key)
phase63a_best_stability = phase63a_best_oracle["stability"]

phase63a_primary_error_auc = phase63a_primary_report[
    "confident_error_detection_auroc"
]
phase63a_primary_top10_recall = phase63a_primary_top10[
    "confident_error_recall"
]
phase63a_ceiling_gain = phase63a_best_stability["pooled"]["log_loss_gain"]
phase63a_ceiling_fold_regret = phase63a_best_stability[
    "maximum_fold_log_loss_regret"
]
phase63a_ceiling_major_group_regret = phase63a_best_stability[
    "maximum_major_group_log_loss_regret"
]
phase63a_ceiling_brier_gain = phase63a_best_stability["pooled"]["brier_gain"]

phase63a_supported = bool(
    phase63a_primary_error_auc is not None
    and phase63a_primary_error_auc
    >= PHASE63A_CONFIG["minimum_primary_confident_error_auroc"]
    and phase63a_primary_top10_recall is not None
    and phase63a_primary_top10_recall
    >= PHASE63A_CONFIG["minimum_primary_top10_confident_error_recall"]
    and phase63a_ceiling_gain
    >= PHASE63A_CONFIG["minimum_diagnostic_ceiling_log_loss_gain"]
    and phase63a_ceiling_fold_regret
    <= PHASE63A_CONFIG["maximum_diagnostic_fold_regret"]
    and phase63a_ceiling_major_group_regret
    <= PHASE63A_CONFIG["maximum_diagnostic_major_group_regret"]
    and phase63a_ceiling_brier_gain >= 0.0
)

phase63a_status = (
    "accepted_sparse_risk_localization_hypothesis_requires_fresh_nested_phase63b"
    if phase63a_supported
    else "sparse_risk_localization_rejected_stop_router_family"
)

phase63a_contract_core = {
    "schema_version": PHASE63A_CONFIG["schema_version"],
    "status": phase63a_status,
    "input_contracts": {
        "phase62_validation_reset": phase63a_validation["contract_sha256"],
        "phase62_multitemplate": phase63a_multitemplate["contract_sha256"],
    },
    "primary_risk_formula": (
        "anchor_confidence_gate_times_consensus_of_boundaryward_disagreement_"
        "from_anatomy_sensitivity_and_specificity_heads"
    ),
    "primary_risk_uses_labels_groups_folds_or_uids": False,
    "diagnostic_oracle_used_for_selection_or_deployment": False,
    "phase31_diagnostic_eligible_for_deployment": False,
    "new_candidate_evaluated": False,
    "test_data_read": False,
}
phase63a_contract_payload = json.dumps(
    phase63a_contract_core, sort_keys=True, separators=(",", ":")
).encode("utf-8")
phase63a_contract_sha256 = hashlib.sha256(phase63a_contract_payload).hexdigest()

phase63a_contract_path = Path(PHASE63A_CONFIG["contract_file"])
phase63a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase63a_temporary_path = phase63a_contract_path.with_suffix(
    phase63a_contract_path.suffix + ".tmp"
)
phase63a_temporary_path.write_text(json.dumps({
    **phase63a_contract_core,
    "contract_sha256": phase63a_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
    "contains_embeddings": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase63a_temporary_path, phase63a_contract_path)

PHASE63A_SPARSE_RISK_REPORT_PRIVATE = None
PHASE63A_SPARSE_RISK_STATE_PRIVATE = {
    "contract_sha256": phase63a_contract_sha256,
    "status": phase63a_status,
    "primary_risk_score": phase63a_primary_risk.copy(),
}

phase63a_report = {
    "phase": "phase63_sparse_confidence_risk_ceiling_audit",
    "status": phase63a_status,
    "anchor": phase63a_metrics(phase63a_labels, phase63a_anchor),
    "failure_targets": {
        "classification_error_count": int(np.sum(phase63a_error)),
        "confident_error_count": int(np.sum(phase63a_confident_error)),
        "hardest_loss_5_percent_count": int(np.sum(phase63a_hard_loss)),
        "confident_error_total_loss_share": float(
            np.sum(phase63a_anchor_loss[phase63a_confident_error])
            / np.sum(phase63a_anchor_loss)
        ),
    },
    "risk_score_audits": phase63a_score_reports,
    "diagnostic_oracle_ceiling": {
        "evaluated_rows": int(len(phase63a_ceiling_rows)),
        "best_family": phase63a_best_oracle["family"],
        "best_top_fraction": phase63a_best_oracle["top_fraction"],
        "best_strength": phase63a_best_oracle["strength"],
        "best_stability": phase63a_best_stability,
        "target_log_loss": 0.2279,
        "target_gap_before": float(
            phase63a_metrics(phase63a_labels, phase63a_anchor)["log_loss"]
            - 0.2279
        ),
        "fraction_of_target_gap_explained": float(
            phase63a_ceiling_gain
            / (
                phase63a_metrics(phase63a_labels, phase63a_anchor)["log_loss"]
                - 0.2279
            )
        ),
        "oracle_fit_to_current_oof_labels": True,
        "eligible_for_candidate_selection_or_deployment": False,
    },
    "support_gate": {
        "primary_confident_error_detection_auroc": phase63a_primary_error_auc,
        "primary_top10_confident_error_recall": phase63a_primary_top10_recall,
        "best_diagnostic_ceiling_log_loss_gain": phase63a_ceiling_gain,
        "best_diagnostic_maximum_fold_regret": phase63a_ceiling_fold_regret,
        "best_diagnostic_maximum_major_group_regret": (
            phase63a_ceiling_major_group_regret
        ),
        "best_diagnostic_brier_gain": phase63a_ceiling_brier_gain,
        "thresholds": {
            "minimum_primary_confident_error_auroc": PHASE63A_CONFIG[
                "minimum_primary_confident_error_auroc"
            ],
            "minimum_primary_top10_confident_error_recall": PHASE63A_CONFIG[
                "minimum_primary_top10_confident_error_recall"
            ],
            "minimum_diagnostic_ceiling_log_loss_gain": PHASE63A_CONFIG[
                "minimum_diagnostic_ceiling_log_loss_gain"
            ],
            "maximum_diagnostic_fold_regret": PHASE63A_CONFIG[
                "maximum_diagnostic_fold_regret"
            ],
            "maximum_diagnostic_major_group_regret": PHASE63A_CONFIG[
                "maximum_diagnostic_major_group_regret"
            ],
            "minimum_diagnostic_brier_gain": 0.0,
        },
        "supported": phase63a_supported,
    },
    "interpretation_contract": {
        "primary_risk_is_label_blind": True,
        "phase62_candidate_remains_rejected": True,
        "phase31_provenance_remains_unverified": True,
        "phase31_score_cannot_support_advancement": True,
        "oracle_thresholds_and_strengths_are_not_candidates": True,
        "no_deployable_threshold_or_strength_selected": True,
        "fresh_nested_oof_required_before_any_probability_change": True,
    },
    "training_performed": False,
    "training_voxel_cache_read": False,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "models_exported": False,
    "contract_sha256": phase63a_contract_sha256,
    "contract_file": str(phase63a_contract_path),
    "elapsed_seconds": float(time.perf_counter() - phase63a_started),
}

PHASE63A_SPARSE_RISK_REPORT_PRIVATE = phase63a_report

print("BEGIN SANITIZED_PHASE63_SPARSE_CONFIDENCE_RISK_CEILING_AUDIT")
print(json.dumps(phase63a_report, indent=2, sort_keys=False))
print("END SANITIZED_PHASE63_SPARSE_CONFIDENCE_RISK_CEILING_AUDIT")
