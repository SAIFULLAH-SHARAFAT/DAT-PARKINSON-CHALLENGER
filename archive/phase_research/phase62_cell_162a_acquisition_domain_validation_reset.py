from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 162A — acquisition-domain validation reset and next-candidate contract.
#
# Run immediately after Cell 161B in the same live kernel. This cell performs
# no model training and reads no NIfTI, voxel, test, smoke-test, embedding, or
# patient-row data. It freezes the validation constitution that Cell 162B must
# obey before the anatomy/uncertainty candidate is trained.

phase62a_started = time.perf_counter()

assert isinstance(
    globals().get("PHASE61_PHASE31_FIXED_BLEND_REPORT_PRIVATE"), dict
)
assert PHASE61_PHASE31_FIXED_BLEND_REPORT_PRIVATE["status"] == (
    "fixed_blend_stability_failed_stop_phase31_blend"
)
assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)

phase62a_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase62a_labels = np.asarray(phase62a_state["labels"], dtype=np.int64).reshape(-1)
phase62a_groups = np.asarray(phase62a_state["groups"], dtype=np.int64).reshape(-1)
phase62a_anchor = np.asarray(
    phase62a_state["anchor_probability"], dtype=np.float64
).reshape(-1)

if "phase57c_original_fold" in globals():
    phase62a_original_fold = np.asarray(
        phase57c_original_fold, dtype=np.int64
    ).reshape(-1)
elif isinstance(globals().get("PHASE57_TRANSPORT_STATE_PRIVATE"), dict):
    phase62a_original_fold = np.asarray(
        PHASE57_TRANSPORT_STATE_PRIVATE["original_fold"], dtype=np.int64
    ).reshape(-1)
else:
    raise AssertionError("Accepted acquisition-group folds are unavailable.")

assert phase62a_labels.shape == phase62a_groups.shape == phase62a_anchor.shape == (1362,)
assert phase62a_original_fold.shape == (1362,)
assert set(np.unique(phase62a_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase62a_groups), np.arange(15))
assert np.array_equal(np.unique(phase62a_original_fold), np.arange(3))
assert np.all(np.isfinite(phase62a_anchor))
assert np.all((phase62a_anchor > 0.0) & (phase62a_anchor < 1.0))


PHASE62A_CONFIG = {
    "schema_version": "phase62_acquisition_domain_validation_reset_v1",
    "case_count": 1362,
    "group_count": 15,
    "individual_stress_domain_minimum_n": 30,
    "major_groups": [1, 3],
    "confidence_threshold": 0.80,
    "group_blocked_repeat_count": 5,
    "group_blocked_seed": 620162,
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 620163,
    "probability_clip": 1.0e-7,
    "phase31_oof_file": "/kaggle/working/phase31_classical_oof_private.npy",
    "user_reported_leaderboard_reference": {
        "log_loss": 0.2279,
        "auroc": 0.9676,
        "diagnostic_only": True,
        "used_for_fitting_or_selection": False,
    },
    "next_candidate": {
        "name": "fold_local_multitemplate_anatomy_uncertainty_expert",
        "single_predeclared_candidate": True,
        "phase31_global_blend_reused": False,
        "outer_labels_available_to_candidate": False,
        "test_information_available_to_candidate": False,
    },
    "advancement_gate": {
        "minimum_pooled_log_loss_gain": 0.005,
        "minimum_pooled_auroc_gain": 0.0015,
        "minimum_major_groups_combined_log_loss_gain": 0.005,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "maximum_original_fold_log_loss_regret": 0.001,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
        "maximum_non_confident_case_log_loss_regret": 0.00025,
        "brier_regret_allowed": 0.0,
    },
    "contract_file": "/kaggle/working/phase62_acquisition_domain_validation_contract.json",
}


def phase62a_auc(labels, score):
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


def phase62a_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE62A_CONFIG["probability_clip"],
        1.0 - PHASE62A_CONFIG["probability_clip"],
    )
    return -(labels * np.log(probability) + (1 - labels) * np.log1p(-probability))


def phase62a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase62a_case_loss(labels, probability))),
        "auroc": phase62a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
    }


def phase62a_compare(labels, anchor, candidate):
    anchor_metrics = phase62a_metrics(labels, anchor)
    candidate_metrics = phase62a_metrics(labels, candidate)
    return {
        "n": int(np.asarray(labels).size),
        "anchor_log_loss": anchor_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": anchor_metrics["log_loss"] - candidate_metrics["log_loss"],
        "anchor_auroc": anchor_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": (
            None
            if anchor_metrics["auroc"] is None or candidate_metrics["auroc"] is None
            else candidate_metrics["auroc"] - anchor_metrics["auroc"]
        ),
        "brier_gain": anchor_metrics["brier"] - candidate_metrics["brier"],
    }


def phase62a_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE62A_CONFIG["probability_clip"],
        1.0 - PHASE62A_CONFIG["probability_clip"],
    )
    return np.log(probability) - np.log1p(-probability)


def phase62a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


# Phase31 is retained only as a frozen negative-control diagnostic. Its failed
# Phase61B global blend is not promoted and cannot become the Phase62 candidate.
phase62a_phase31_path = Path(PHASE62A_CONFIG["phase31_oof_file"])
assert phase62a_phase31_path.is_file(), {"missing": str(phase62a_phase31_path)}
phase62a_phase31 = np.asarray(
    np.load(phase62a_phase31_path, allow_pickle=False, mmap_mode="r"),
    dtype=np.float64,
).reshape(-1)
assert phase62a_phase31.shape == (1362,)
assert np.all(np.isfinite(phase62a_phase31))
assert np.all((phase62a_phase31 > 0.0) & (phase62a_phase31 < 1.0))
phase62a_negative_control = phase62a_sigmoid(
    0.80 * phase62a_logit(phase62a_anchor)
    + 0.20 * phase62a_logit(phase62a_phase31)
)


# Stress domains are defined only by acquisition-group sample counts. Groups
# with n>=30 remain individual domains; all tiny groups form one pooled domain.
phase62a_group_sizes = np.bincount(phase62a_groups, minlength=15)
phase62a_individual_groups = [
    int(group)
    for group, count in enumerate(phase62a_group_sizes)
    if int(count) >= PHASE62A_CONFIG["individual_stress_domain_minimum_n"]
]
phase62a_tiny_groups = [
    int(group) for group in range(15) if group not in phase62a_individual_groups
]
assert set(phase62a_individual_groups + phase62a_tiny_groups) == set(range(15))
assert set(PHASE62A_CONFIG["major_groups"]).issubset(phase62a_individual_groups)

phase62a_domain_definitions = [
    {"name": f"group_{group}", "groups": [group]}
    for group in phase62a_individual_groups
]
phase62a_domain_definitions.append({
    "name": "tiny_groups_pooled",
    "groups": phase62a_tiny_groups,
})

phase62a_domain_id = np.full(1362, -1, dtype=np.int64)
phase62a_domain_records = []
for domain_index, definition in enumerate(phase62a_domain_definitions):
    mask = np.isin(phase62a_groups, definition["groups"])
    assert int(np.sum(mask)) > 0
    phase62a_domain_id[mask] = domain_index
    comparison = phase62a_compare(
        phase62a_labels[mask],
        phase62a_anchor[mask],
        phase62a_negative_control[mask],
    )
    phase62a_domain_records.append({
        "domain": definition["name"],
        "groups": definition["groups"],
        "n": comparison["n"],
        "anchor_log_loss": comparison["anchor_log_loss"],
        "anchor_auroc": comparison["anchor_auroc"],
        "negative_control_log_loss_gain": comparison["log_loss_gain"],
        "negative_control_auroc_gain": comparison["auroc_gain"],
    })
assert np.all(phase62a_domain_id >= 0)


# Five group-blocked reassignment maps are generated without labels. They are
# frozen now and may be used only as secondary stability evaluations in 162B.
phase62a_repeat_rng = np.random.default_rng(PHASE62A_CONFIG["group_blocked_seed"])
phase62a_group_repeat_assignment = []
phase62a_repeat_summaries = []
phase62a_seen_assignments = set()
phase62a_attempt = 0
while len(phase62a_group_repeat_assignment) < PHASE62A_CONFIG["group_blocked_repeat_count"]:
    phase62a_attempt += 1
    assert phase62a_attempt <= 100000
    order = phase62a_repeat_rng.permutation(15)
    fold_load = np.zeros(3, dtype=np.int64)
    group_to_fold = np.full(15, -1, dtype=np.int64)
    for group in order:
        minimum_load = np.min(fold_load)
        choices = np.flatnonzero(fold_load == minimum_load)
        selected_fold = int(phase62a_repeat_rng.choice(choices))
        group_to_fold[int(group)] = selected_fold
        fold_load[selected_fold] += int(phase62a_group_sizes[int(group)])
    if int(np.max(fold_load) - np.min(fold_load)) > 180:
        continue
    group_counts = np.bincount(group_to_fold, minlength=3)
    if int(np.min(group_counts)) < 2:
        continue
    canonical = tuple(int(value) for value in group_to_fold)
    if canonical in phase62a_seen_assignments:
        continue
    phase62a_seen_assignments.add(canonical)
    phase62a_group_repeat_assignment.append(group_to_fold)
    phase62a_repeat_summaries.append({
        "repeat": len(phase62a_repeat_summaries),
        "fold_case_counts": [int(value) for value in fold_load],
        "fold_group_counts": [int(value) for value in group_counts],
    })

phase62a_group_repeat_assignment = np.stack(
    phase62a_group_repeat_assignment, axis=0
)
phase62a_case_repeat_assignment = phase62a_group_repeat_assignment[:, phase62a_groups]
assert phase62a_case_repeat_assignment.shape == (
    PHASE62A_CONFIG["group_blocked_repeat_count"], 1362
)
for repeat in range(PHASE62A_CONFIG["group_blocked_repeat_count"]):
    for group in range(15):
        assert np.unique(
            phase62a_case_repeat_assignment[repeat, phase62a_groups == group]
        ).size == 1


# Domain-level bootstrap weights each acquisition domain equally, preventing
# group 1 from hiding instability in smaller protocols.
phase62a_domain_anchor_loss = []
phase62a_domain_control_loss = []
for definition in phase62a_domain_definitions:
    mask = np.isin(phase62a_groups, definition["groups"])
    phase62a_domain_anchor_loss.append(float(np.mean(
        phase62a_case_loss(phase62a_labels[mask], phase62a_anchor[mask])
    )))
    phase62a_domain_control_loss.append(float(np.mean(
        phase62a_case_loss(phase62a_labels[mask], phase62a_negative_control[mask])
    )))
phase62a_domain_gain = (
    np.asarray(phase62a_domain_anchor_loss)
    - np.asarray(phase62a_domain_control_loss)
)

phase62a_bootstrap_rng = np.random.default_rng(
    PHASE62A_CONFIG["domain_bootstrap_seed"]
)
phase62a_bootstrap_gain = np.empty(
    PHASE62A_CONFIG["domain_bootstrap_replicates"], dtype=np.float64
)
domain_count = len(phase62a_domain_definitions)
for replicate in range(PHASE62A_CONFIG["domain_bootstrap_replicates"]):
    sampled = phase62a_bootstrap_rng.integers(0, domain_count, size=domain_count)
    phase62a_bootstrap_gain[replicate] = float(np.mean(phase62a_domain_gain[sampled]))

phase62a_anchor_metrics = phase62a_metrics(phase62a_labels, phase62a_anchor)
phase62a_control_comparison = phase62a_compare(
    phase62a_labels, phase62a_anchor, phase62a_negative_control
)
phase62a_major_mask = np.isin(phase62a_groups, PHASE62A_CONFIG["major_groups"])
phase62a_major_comparison = phase62a_compare(
    phase62a_labels[phase62a_major_mask],
    phase62a_anchor[phase62a_major_mask],
    phase62a_negative_control[phase62a_major_mask],
)

phase62a_anchor_predicted = (phase62a_anchor >= 0.5).astype(np.int64)
phase62a_confident = (
    (phase62a_anchor >= PHASE62A_CONFIG["confidence_threshold"])
    | (phase62a_anchor <= 1.0 - PHASE62A_CONFIG["confidence_threshold"])
)
phase62a_confident_error = phase62a_confident & (
    phase62a_anchor_predicted != phase62a_labels
)
phase62a_anchor_loss = phase62a_case_loss(phase62a_labels, phase62a_anchor)
phase62a_control_loss = phase62a_case_loss(
    phase62a_labels, phase62a_negative_control
)

leaderboard_reference = PHASE62A_CONFIG["user_reported_leaderboard_reference"]
phase62a_target_gap = {
    "anchor_log_loss_gap": float(
        phase62a_anchor_metrics["log_loss"] - leaderboard_reference["log_loss"]
    ),
    "anchor_auroc_gap": float(
        leaderboard_reference["auroc"] - phase62a_anchor_metrics["auroc"]
    ),
    "negative_control_log_loss_gap": float(
        phase62a_control_comparison["candidate_log_loss"]
        - leaderboard_reference["log_loss"]
    ),
    "negative_control_auroc_gap": float(
        leaderboard_reference["auroc"]
        - phase62a_control_comparison["candidate_auroc"]
    ),
}

phase62a_contract_core = {
    "schema_version": PHASE62A_CONFIG["schema_version"],
    "status": "accepted_validation_reset_ready_for_phase62b",
    "stress_domain_definitions": phase62a_domain_definitions,
    "group_blocked_repeat_group_to_fold": phase62a_group_repeat_assignment.tolist(),
    "next_candidate": PHASE62A_CONFIG["next_candidate"],
    "advancement_gate": PHASE62A_CONFIG["advancement_gate"],
    "test_independence": {
        "test_batch_statistics": False,
        "test_time_training": False,
        "test_time_unsupervised_adaptation": False,
        "pseudo_labeling": False,
        "weights_independent_of_test_set": True,
        "fitted_parameters_independent_of_test_set": True,
    },
}
phase62a_contract_sha256 = hashlib.sha256(json.dumps(
    phase62a_contract_core, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()

phase62a_contract_path = Path(PHASE62A_CONFIG["contract_file"])
phase62a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase62a_temporary_path = phase62a_contract_path.with_suffix(
    phase62a_contract_path.suffix + ".tmp"
)
phase62a_temporary_path.write_text(json.dumps({
    **phase62a_contract_core,
    "contract_sha256": phase62a_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase62a_temporary_path, phase62a_contract_path)

PHASE62_VALIDATION_RESET_CONFIG_PRIVATE = json.loads(json.dumps(PHASE62A_CONFIG))
PHASE62_VALIDATION_RESET_STATE_PRIVATE = {
    "contract_sha256": phase62a_contract_sha256,
    "labels": phase62a_labels.copy(),
    "groups": phase62a_groups.copy(),
    "original_fold": phase62a_original_fold.copy(),
    "anchor_probability": phase62a_anchor.copy(),
    "negative_control_probability": phase62a_negative_control.copy(),
    "stress_domain_id": phase62a_domain_id.copy(),
    "stress_domain_definitions": json.loads(json.dumps(phase62a_domain_definitions)),
    "group_blocked_repeat_group_to_fold": phase62a_group_repeat_assignment.copy(),
    "group_blocked_repeat_case_fold": phase62a_case_repeat_assignment.copy(),
}

phase62a_report = {
    "phase": "phase62_acquisition_domain_validation_reset",
    "status": "accepted_validation_reset_ready_for_phase62b",
    "leaderboard_reference": leaderboard_reference,
    "current_gap": phase62a_target_gap,
    "anchor": phase62a_anchor_metrics,
    "rejected_negative_control": {
        "name": "phase31_global_logit_blend_alpha_020",
        "deployment_eligible": False,
        "pooled_log_loss_gain": phase62a_control_comparison["log_loss_gain"],
        "pooled_auroc_gain": phase62a_control_comparison["auroc_gain"],
        "major_groups_combined_log_loss_gain": phase62a_major_comparison["log_loss_gain"],
        "domain_macro_log_loss_gain": float(np.mean(phase62a_domain_gain)),
        "domain_bootstrap_lower_95_log_loss_gain": float(
            np.quantile(phase62a_bootstrap_gain, 0.025)
        ),
        "confident_error_mean_log_loss_gain": float(np.mean(
            phase62a_anchor_loss[phase62a_confident_error]
            - phase62a_control_loss[phase62a_confident_error]
        )),
        "remaining_case_mean_log_loss_gain": float(np.mean(
            phase62a_anchor_loss[~phase62a_confident_error]
            - phase62a_control_loss[~phase62a_confident_error]
        )),
    },
    "stress_domains": phase62a_domain_records,
    "group_blocked_repeats": phase62a_repeat_summaries,
    "frozen_next_candidate": PHASE62A_CONFIG["next_candidate"],
    "frozen_advancement_gate": PHASE62A_CONFIG["advancement_gate"],
    "interpretation_contract": {
        "original_three_fold_oof_is_development_evidence": True,
        "case_resampling_is_not_domain_holdout": True,
        "phase31_global_blend_remains_rejected": True,
        "only_one_phase62b_candidate_may_be_evaluated": True,
        "phase62b_outer_predictions_read_once_after_freezing": True,
        "leaderboard_reference_used_for_fitting_or_selection": False,
    },
    "training_performed": False,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "contract_sha256": phase62a_contract_sha256,
    "contract_file": str(phase62a_contract_path),
    "elapsed_seconds": round(time.perf_counter() - phase62a_started, 3),
}

PHASE62_VALIDATION_RESET_REPORT_PRIVATE = phase62a_report

phase62a_serialized_report = json.dumps(phase62a_report, indent=2)
phase62a_output_lines = phase62a_serialized_report.splitlines()
assert len(phase62a_output_lines) + 2 <= 300
assert max(len(line) for line in phase62a_output_lines) <= 300

print("BEGIN SANITIZED_PHASE62_ACQUISITION_DOMAIN_VALIDATION_RESET")
print(phase62a_serialized_report)
print("END SANITIZED_PHASE62_ACQUISITION_DOMAIN_VALIDATION_RESET")
