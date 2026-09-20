from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np


# Cell 158A — Phase58 anchor-gradient transport audit.
#
# Run immediately after accepted Phase57 Cell 157D in the same kernel.
# This cell performs no model training and reads no voxel, NIfTI, smoke, or
# test data. It uses only the already-loaded fit-side labels, acquisition
# groups, anchor OOF probabilities, and the private Phase57 pilot partitions.
# Original outer-fold-0 validation remains completely untouched.
#
# Purpose:
#   1. explain the positive intercept-like drift analytically;
#   2. distinguish calibration pressure from within-group residual signal;
#   3. construct private, group-centered logistic pseudo-residual targets for
#      the next bounded 3D residual learner;
#   4. freeze a no-data contract before any additional model is trained.

phase58a_started = time.perf_counter()


for phase58a_required_name in (
    "PHASE57_TRAINING_ENGINE_STATE_PRIVATE",
    "PHASE57_TRAINING_ENGINE_REPORT_PRIVATE",
    "PHASE57_ANCHOR_OPTIMIZATION_STATE_PRIVATE",
    "PHASE57_ANCHOR_OPTIMIZATION_REPORT_PRIVATE",
    "phase57c_internal_fit_indices",
    "phase57c_pilot_train_indices",
    "phase57c_internal_monitor_indices",
    "phase57c_outer_valid_indices",
):
    assert phase58a_required_name in globals(), {
        "message": "Cell 158A requires the accepted Phase57A-157D state.",
        "missing_name": phase58a_required_name,
    }


phase58a_engine_state = PHASE57_TRAINING_ENGINE_STATE_PRIVATE
phase58a_engine_report = PHASE57_TRAINING_ENGINE_REPORT_PRIVATE
phase58a_optimization_state = PHASE57_ANCHOR_OPTIMIZATION_STATE_PRIVATE
phase58a_optimization_report = PHASE57_ANCHOR_OPTIMIZATION_REPORT_PRIVATE

assert phase58a_engine_report["status"] == (
    "accepted_ready_for_frozen_architecture_screen"
)
assert phase58a_optimization_report["status"] == (
    "no_update_selected_stop_before_architecture_screen"
)
assert phase58a_optimization_state["gate_advanced"] is False
assert phase58a_optimization_state["selected_record"] is None
assert phase58a_optimization_state["selected_checkpoint"] is None


PHASE58A_CONFIG = {
    "schema_version": "phase58_anchor_gradient_transport_audit_v1",
    "log_loss_clip": 1.0e-7,
    "intercept_search_bound": 8.0,
    "intercept_search_iterations": 100,
    "newton_denominator_floor": 0.05,
    "newton_target_cap": 2.0,
    "weight_schemes": (
        "ordinary",
        "sqrt_inverse_group",
        "inverse_group",
        "prior_corrected_sqrt_inverse_group",
    ),
    "phase57d_arm_mix_values": tuple(
        float(arm["group_balance_mix"])
        for arm in phase58a_optimization_state["config"]["arms"]
    ),
    "contract_file": (
        "/kaggle/working/phase58_gradient_transport_contract.json"
    ),
}


phase58a_labels = np.asarray(
    phase58a_engine_state["labels"], dtype=np.int64
).reshape(-1)
phase58a_groups = np.asarray(
    phase58a_engine_state["groups"], dtype=np.int64
).reshape(-1)
phase58a_anchor_probability = np.asarray(
    phase58a_engine_state["anchor_probability"], dtype=np.float64
).reshape(-1)
phase58a_anchor_probability = np.clip(
    phase58a_anchor_probability,
    PHASE58A_CONFIG["log_loss_clip"],
    1.0 - PHASE58A_CONFIG["log_loss_clip"],
)
phase58a_anchor_logit = (
    np.log(phase58a_anchor_probability)
    - np.log1p(-phase58a_anchor_probability)
)

phase58a_case_count = int(phase58a_labels.size)
assert phase58a_case_count == 1362
assert phase58a_groups.shape == (phase58a_case_count,)
assert phase58a_anchor_probability.shape == (phase58a_case_count,)
assert set(np.unique(phase58a_labels).tolist()) == {0, 1}
assert np.all(np.isfinite(phase58a_anchor_probability))
assert np.all(np.isfinite(phase58a_anchor_logit))

phase58a_internal_fit_indices = np.asarray(
    phase57c_internal_fit_indices, dtype=np.int64
).reshape(-1)
phase58a_pilot_indices = np.asarray(
    phase57c_pilot_train_indices, dtype=np.int64
).reshape(-1)
phase58a_monitor_indices = np.asarray(
    phase57c_internal_monitor_indices, dtype=np.int64
).reshape(-1)
phase58a_outer_valid_indices = np.asarray(
    phase57c_outer_valid_indices, dtype=np.int64
).reshape(-1)

assert phase58a_pilot_indices.size == 256
assert phase58a_monitor_indices.size == 109
assert np.all(np.isin(
    phase58a_pilot_indices, phase58a_internal_fit_indices
))
assert np.intersect1d(
    phase58a_internal_fit_indices, phase58a_monitor_indices
).size == 0
assert np.intersect1d(
    phase58a_internal_fit_indices, phase58a_outer_valid_indices
).size == 0
assert np.intersect1d(
    phase58a_monitor_indices, phase58a_outer_valid_indices
).size == 0


def phase58a_sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase58a_log_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        PHASE58A_CONFIG["log_loss_clip"],
        1.0 - PHASE58A_CONFIG["log_loss_clip"],
    )
    assert labels.shape == probability.shape
    return float(np.mean(-(
        labels * np.log(probability)
        + (1.0 - labels) * np.log1p(-probability)
    )))


def phase58a_auroc(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    assert labels.shape == score.shape
    assert set(np.unique(labels).tolist()) == {0, 1}
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    rank = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        rank[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive = labels == 1
    positive_n = int(np.sum(positive))
    negative_n = int(labels.size - positive_n)
    assert positive_n > 0 and negative_n > 0
    statistic = (
        float(np.sum(rank[positive]))
        - positive_n * (positive_n + 1) / 2.0
    )
    return float(statistic / (positive_n * negative_n))


def phase58a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    return {
        "n": int(labels.size),
        "prevalence": float(np.mean(labels)),
        "mean_probability": float(np.mean(probability)),
        "log_loss": phase58a_log_loss(labels, probability),
        "auroc": phase58a_auroc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
    }


def phase58a_normalize_weights(weight):
    weight = np.asarray(weight, dtype=np.float64).reshape(-1)
    assert np.all(np.isfinite(weight))
    assert np.all(weight > 0.0)
    return weight / float(np.mean(weight))


def phase58a_group_weights(groups, power):
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    counts = np.bincount(groups, minlength=15).astype(np.float64)
    active = counts > 0.0
    lookup = np.ones(15, dtype=np.float64)
    lookup[active] = np.power(counts[active], -float(power))
    return phase58a_normalize_weights(lookup[groups])


def phase58a_prior_correction(labels, target_prevalence):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    source_prevalence = float(np.mean(labels))
    target_prevalence = float(target_prevalence)
    assert 0.0 < source_prevalence < 1.0
    assert 0.0 < target_prevalence < 1.0
    positive_weight = target_prevalence / source_prevalence
    negative_weight = (
        (1.0 - target_prevalence) / (1.0 - source_prevalence)
    )
    return np.where(labels == 1, positive_weight, negative_weight)


def phase58a_weight_map(indices, scheme, target_prevalence):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    labels = phase58a_labels[indices]
    groups = phase58a_groups[indices]
    if scheme == "ordinary":
        weight = np.ones(indices.size, dtype=np.float64)
    elif scheme == "sqrt_inverse_group":
        weight = phase58a_group_weights(groups, power=0.5)
    elif scheme == "inverse_group":
        weight = phase58a_group_weights(groups, power=1.0)
    elif scheme == "prior_corrected_sqrt_inverse_group":
        weight = (
            phase58a_group_weights(groups, power=0.5)
            * phase58a_prior_correction(labels, target_prevalence)
        )
    else:
        raise KeyError(scheme)
    return phase58a_normalize_weights(weight)


def phase58a_uniform_gradient(indices, weight):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    weight = phase58a_normalize_weights(weight)
    gradient = phase58a_anchor_probability[indices] - phase58a_labels[indices]
    hessian = (
        phase58a_anchor_probability[indices]
        * (1.0 - phase58a_anchor_probability[indices])
    )
    return {
        "gradient": float(np.mean(weight * gradient)),
        "hessian": float(np.mean(weight * hessian)),
        "newton_step": float(
            -np.mean(weight * gradient) / np.mean(weight * hessian)
        ),
    }


def phase58a_group_gradient_summary(indices):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    labels = phase58a_labels[indices]
    groups = phase58a_groups[indices]
    probability = phase58a_anchor_probability[indices]
    values = []
    for group_value in np.unique(groups):
        mask = groups == int(group_value)
        values.append(float(np.mean(probability[mask] - labels[mask])))
    values = np.asarray(values, dtype=np.float64)
    return {
        "active_group_count": int(values.size),
        "negative_pressure_group_count": int(np.sum(values < 0.0)),
        "positive_pressure_group_count": int(np.sum(values > 0.0)),
        "minimum": float(np.min(values)),
        "median": float(np.median(values)),
        "maximum": float(np.max(values)),
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
    }


def phase58a_optimal_intercept(indices, weight):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    weight = phase58a_normalize_weights(weight)
    labels = phase58a_labels[indices].astype(np.float64)
    logit = phase58a_anchor_logit[indices]
    bound = float(PHASE58A_CONFIG["intercept_search_bound"])
    lower = -bound
    upper = bound

    def derivative(offset):
        return float(np.mean(
            weight * (phase58a_sigmoid(logit + offset) - labels)
        ))

    lower_derivative = derivative(lower)
    upper_derivative = derivative(upper)
    assert lower_derivative < 0.0 < upper_derivative
    for _ in range(int(PHASE58A_CONFIG["intercept_search_iterations"])):
        middle = 0.5 * (lower + upper)
        if derivative(middle) > 0.0:
            upper = middle
        else:
            lower = middle
    return float(0.5 * (lower + upper))


def phase58a_apply_intercept(indices, offset):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    return phase58a_sigmoid(phase58a_anchor_logit[indices] + float(offset))


def phase58a_center_within_group(target, groups):
    target = np.asarray(target, dtype=np.float64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    assert target.shape == groups.shape
    centered = target.copy()
    group_centers = []
    for group_value in np.unique(groups):
        mask = groups == int(group_value)
        center = float(np.mean(target[mask]))
        centered[mask] -= center
        group_centers.append(center)
        assert abs(float(np.mean(centered[mask]))) <= 1.0e-12
    return centered, np.asarray(group_centers, dtype=np.float64)


def phase58a_distribution(values):
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = np.quantile(values, [0.0, 0.05, 0.25, 0.5, 0.75, 0.95, 1.0])
    return {
        "mean": float(np.mean(values)),
        "standard_deviation": float(np.std(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "minimum": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "maximum": float(quantiles[6]),
    }


# -------------------------------------------------------------------------
# Partition-level anchor audit. Outer validation is intentionally omitted.
# -------------------------------------------------------------------------

phase58a_partition_records = {}
for phase58a_partition_name, phase58a_indices in (
    ("pilot_train", phase58a_pilot_indices),
    ("internal_fit_all", phase58a_internal_fit_indices),
    ("internal_monitor", phase58a_monitor_indices),
):
    phase58a_partition_records[phase58a_partition_name] = {
        "anchor": phase58a_metrics(
            phase58a_labels[phase58a_indices],
            phase58a_anchor_probability[phase58a_indices],
        ),
        "ordinary_uniform_pressure": phase58a_uniform_gradient(
            phase58a_indices,
            np.ones(phase58a_indices.size, dtype=np.float64),
        ),
        "group_pressure": phase58a_group_gradient_summary(
            phase58a_indices
        ),
    }


phase58a_target_prevalence = float(np.mean(
    phase58a_labels[phase58a_internal_fit_indices]
))
phase58a_weight_records = {}
phase58a_weight_vectors_private = {}
for phase58a_scheme in PHASE58A_CONFIG["weight_schemes"]:
    phase58a_weight = phase58a_weight_map(
        phase58a_pilot_indices,
        phase58a_scheme,
        phase58a_target_prevalence,
    )
    phase58a_weight_vectors_private[phase58a_scheme] = phase58a_weight
    phase58a_offset = phase58a_optimal_intercept(
        phase58a_pilot_indices, phase58a_weight
    )
    phase58a_monitor_identity = phase58a_metrics(
        phase58a_labels[phase58a_monitor_indices],
        phase58a_anchor_probability[phase58a_monitor_indices],
    )
    phase58a_monitor_shifted = phase58a_metrics(
        phase58a_labels[phase58a_monitor_indices],
        phase58a_apply_intercept(
            phase58a_monitor_indices, phase58a_offset
        ),
    )
    phase58a_pressure = phase58a_uniform_gradient(
        phase58a_pilot_indices, phase58a_weight
    )
    phase58a_weight_records[phase58a_scheme] = {
        "weight_distribution": phase58a_distribution(phase58a_weight),
        "uniform_pressure_at_identity": phase58a_pressure,
        "fit_optimal_intercept": phase58a_offset,
        "monitor_log_loss_change_from_fit_intercept": float(
            phase58a_monitor_shifted["log_loss"]
            - phase58a_monitor_identity["log_loss"]
        ),
        "monitor_auroc_change_from_fit_intercept": float(
            phase58a_monitor_shifted["auroc"]
            - phase58a_monitor_identity["auroc"]
        ),
        "monitor_mean_probability_change_from_fit_intercept": float(
            phase58a_monitor_shifted["mean_probability"]
            - phase58a_monitor_identity["mean_probability"]
        ),
    }


# -------------------------------------------------------------------------
# Reconstruct the uniform-logit gradient of Phase57C and every Phase57D arm.
# Ranking, residual-L2, and residual-center terms are shift-neutral at the
# exact identity checkpoint. The smooth regret term is zero-valued at identity
# but has a nonzero derivative because subtracting a constant does not alter
# the softplus gradient.
# -------------------------------------------------------------------------

phase58a_pilot_probability = phase58a_anchor_probability[phase58a_pilot_indices]
phase58a_pilot_labels = phase58a_labels[phase58a_pilot_indices]
phase58a_pilot_groups = phase58a_groups[phase58a_pilot_indices]
phase58a_case_gradient = phase58a_pilot_probability - phase58a_pilot_labels
phase58a_inverse_group_weight = phase58a_weight_vectors_private[
    "inverse_group"
]

phase58a_group_mean_gradients = []
for phase58a_group_value in np.unique(phase58a_pilot_groups):
    phase58a_mask = phase58a_pilot_groups == int(phase58a_group_value)
    phase58a_group_mean_gradients.append(float(np.mean(
        phase58a_case_gradient[phase58a_mask]
    )))
phase58a_group_mean_gradients = np.asarray(
    phase58a_group_mean_gradients, dtype=np.float64
)

phase58a_phase57c_uniform_gradient = float(
    np.mean(phase58a_inverse_group_weight * phase58a_case_gradient)
    + float(PHASE57_TRAINING_ENGINE_CONFIG_PRIVATE["group_dro_weight"])
    * np.mean(phase58a_group_mean_gradients)
)

phase58a_arm_gradient_records = []
for phase58a_arm_index, phase58a_arm in enumerate(
    phase58a_optimization_state["config"]["arms"]
):
    phase58a_mix = float(phase58a_arm["group_balance_mix"])
    phase58a_classification_weight = (
        (1.0 - phase58a_mix)
        + phase58a_mix * phase58a_inverse_group_weight
    )
    phase58a_classification_gradient = float(np.mean(
        phase58a_classification_weight * phase58a_case_gradient
    ))
    phase58a_regret_gradient = float(
        0.5 * np.mean(phase58a_group_mean_gradients)
    )
    phase58a_total_uniform_gradient = float(
        phase58a_classification_gradient
        + float(phase58a_arm["anchor_regret_weight"])
        * phase58a_regret_gradient
    )
    phase58a_arm_gradient_records.append({
        "arm_index": int(phase58a_arm_index),
        "label": str(phase58a_arm["label"]),
        "classification_uniform_gradient": (
            phase58a_classification_gradient
        ),
        "smooth_regret_uniform_gradient_at_identity": (
            phase58a_regret_gradient
        ),
        "total_uniform_gradient_at_identity": (
            phase58a_total_uniform_gradient
        ),
        "gradient_descent_initial_shift_direction": (
            "positive_logit"
            if phase58a_total_uniform_gradient < 0.0
            else "negative_logit"
            if phase58a_total_uniform_gradient > 0.0
            else "stationary"
        ),
    })


# -------------------------------------------------------------------------
# Construct private stagewise logistic targets. For Bernoulli log loss, y-p is
# the exact negative gradient with respect to the anchor logit. Group centering
# removes acquisition-specific intercept pressure and leaves only within-group
# case-discriminative signal. A damped Newton target is retained as a secondary
# candidate; it is never selected in this diagnostic cell.
# -------------------------------------------------------------------------


def phase58a_make_targets(indices):
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    labels = phase58a_labels[indices].astype(np.float64)
    groups = phase58a_groups[indices]
    probability = phase58a_anchor_probability[indices]
    first_order = labels - probability
    hessian = probability * (1.0 - probability)
    newton = first_order / (
        hessian + float(PHASE58A_CONFIG["newton_denominator_floor"])
    )
    newton = np.clip(
        newton,
        -float(PHASE58A_CONFIG["newton_target_cap"]),
        float(PHASE58A_CONFIG["newton_target_cap"]),
    )
    centered_first_order, first_order_centers = (
        phase58a_center_within_group(first_order, groups)
    )
    centered_newton, newton_centers = phase58a_center_within_group(
        newton, groups
    )
    return {
        "first_order": first_order,
        "centered_first_order": centered_first_order,
        "newton": newton,
        "centered_newton": centered_newton,
        "first_order_group_centers": first_order_centers,
        "newton_group_centers": newton_centers,
    }


phase58a_pilot_targets = phase58a_make_targets(
    phase58a_pilot_indices
)
phase58a_internal_fit_targets = phase58a_make_targets(
    phase58a_internal_fit_indices
)

phase58a_target_report = {}
for phase58a_target_name in (
    "first_order",
    "centered_first_order",
    "newton",
    "centered_newton",
):
    phase58a_target_report[phase58a_target_name] = {
        "pilot": phase58a_distribution(
            phase58a_pilot_targets[phase58a_target_name]
        ),
        "internal_fit_all": phase58a_distribution(
            phase58a_internal_fit_targets[phase58a_target_name]
        ),
    }

phase58a_centered_signal_fraction = float(
    np.var(phase58a_internal_fit_targets["centered_first_order"])
    / max(
        np.var(phase58a_internal_fit_targets["first_order"]),
        1.0e-12,
    )
)

phase58a_all_arm_positive_shift = all(
    record["gradient_descent_initial_shift_direction"] == "positive_logit"
    for record in phase58a_arm_gradient_records
)
phase58a_centered_signal_present = bool(
    phase58a_centered_signal_fraction >= 0.50
    and phase58a_target_report["centered_first_order"][
        "internal_fit_all"
    ]["standard_deviation"] >= 0.10
)

phase58a_status = (
    "accepted_positive_intercept_pressure_confirmed_"
    "ready_for_group_centered_gradient_residual_pilot"
    if phase58a_all_arm_positive_shift and phase58a_centered_signal_present
    else "accepted_diagnostic_complete_manual_review_before_training"
)


phase58a_contract_core = {
    "schema_version": PHASE58A_CONFIG["schema_version"],
    "phase57_training_engine_contract_sha256": (
        phase58a_engine_state["contract_sha256"]
    ),
    "phase57_anchor_optimization_contract_sha256": (
        phase58a_optimization_state["contract_sha256"]
    ),
    "method": {
        "primary_target": "within_acquisition_group_centered_y_minus_p_anchor",
        "secondary_target": (
            "within_acquisition_group_centered_damped_newton_residual"
        ),
        "model_training_performed": False,
        "candidate_selection_performed": False,
        "outer_validation_used": False,
    },
    "next_gate": {
        "pilot_partition": "same_phase57_internal_fit_and_monitor_only",
        "identity_anchor_must_remain_selectable": True,
        "minimum_monitor_log_loss_gain": 0.001,
        "minimum_monitor_auroc_gain": 0.001,
        "maximum_monitor_log_loss_excess_for_auroc_gain": 0.0005,
        "maximum_absolute_mean_residual": 0.02,
        "outer_fold_zero_remains_untouched_until_candidate_frozen": True,
    },
}
phase58a_contract_sha256 = hashlib.sha256(json.dumps(
    phase58a_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

phase58a_contract_path = Path(PHASE58A_CONFIG["contract_file"])
if not phase58a_contract_path.parent.exists():
    phase58a_contract_path = (
        Path(phase58a_engine_state["highres_cache_file"]).parent
        / phase58a_contract_path.name
    )
phase58a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase58a_temporary_path = phase58a_contract_path.with_suffix(
    phase58a_contract_path.suffix + ".tmp"
)
phase58a_temporary_path.write_text(json.dumps({
    **phase58a_contract_core,
    "contract_sha256": phase58a_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase58a_temporary_path, phase58a_contract_path)


PHASE58_GRADIENT_TRANSPORT_CONFIG_PRIVATE = json.loads(json.dumps(
    PHASE58A_CONFIG
))
PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE = {
    "contract_sha256": phase58a_contract_sha256,
    "status": phase58a_status,
    "labels": phase58a_labels.copy(),
    "groups": phase58a_groups.copy(),
    "anchor_probability": phase58a_anchor_probability.copy(),
    "anchor_logit": phase58a_anchor_logit.copy(),
    "internal_fit_indices": phase58a_internal_fit_indices.copy(),
    "pilot_indices": phase58a_pilot_indices.copy(),
    "monitor_indices": phase58a_monitor_indices.copy(),
    "pilot_centered_first_order_target": (
        phase58a_pilot_targets["centered_first_order"].astype(np.float32)
    ),
    "internal_fit_centered_first_order_target": (
        phase58a_internal_fit_targets["centered_first_order"].astype(
            np.float32
        )
    ),
    "pilot_centered_newton_target": (
        phase58a_pilot_targets["centered_newton"].astype(np.float32)
    ),
    "internal_fit_centered_newton_target": (
        phase58a_internal_fit_targets["centered_newton"].astype(np.float32)
    ),
    "highres_cache_file": str(
        phase58a_engine_state["highres_cache_file"]
    ),
}


phase58a_report = {
    "phase": "phase58_anchor_gradient_transport_audit",
    "status": phase58a_status,
    "source": {
        "phase57d_status_verified": True,
        "phase57d_no_update_verified": True,
        "phase57d_contract_sha256": (
            phase58a_optimization_state["contract_sha256"]
        ),
    },
    "partitions": phase58a_partition_records,
    "pilot_weight_schemes": phase58a_weight_records,
    "objective_uniform_gradient": {
        "phase57c_initial_objective": (
            phase58a_phase57c_uniform_gradient
        ),
        "phase57d_arms": phase58a_arm_gradient_records,
        "all_phase57d_arms_predict_positive_logit_shift": (
            phase58a_all_arm_positive_shift
        ),
        "interpretation": (
            "negative_uniform_gradient_means_gradient_descent_increases_"
            "the_logit_intercept"
        ),
    },
    "stagewise_targets": {
        "identity": "negative_logistic_gradient_y_minus_p_anchor",
        "target_distributions": phase58a_target_report,
        "centered_first_order_variance_fraction": (
            phase58a_centered_signal_fraction
        ),
        "centered_discriminative_signal_present": (
            phase58a_centered_signal_present
        ),
        "case_level_targets_exported": False,
    },
    "next_experiment": phase58a_contract_core["next_gate"],
    "contract_sha256": phase58a_contract_sha256,
    "contract_file": str(phase58a_contract_path),
    "model_training_performed": False,
    "hyperparameter_selection_performed": False,
    "fit_labels_used_for_diagnostic": True,
    "internal_monitor_labels_used_for_diagnostic": True,
    "outer_validation_images_used": False,
    "outer_validation_labels_used": False,
    "logo_labels_used": False,
    "public_leaderboard_used": False,
    "training_voxel_cache_read": False,
    "training_nifti_files_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "uids_displayed": False,
    "case_level_predictions_displayed": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase58a_started, 3),
}

PHASE58_GRADIENT_TRANSPORT_REPORT_PRIVATE = dict(phase58a_report)

print("BEGIN SANITIZED_PHASE58_GRADIENT_TRANSPORT_AUDIT")
print(json.dumps(phase58a_report, indent=2))
print("END SANITIZED_PHASE58_GRADIENT_TRANSPORT_AUDIT")
