from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 159A — deployment-matched OOF failure-anatomy audit.
# Run after accepted Phase58A/158B in the same live kernel. This is aggregate
# diagnostics only: no images, test cases, UIDs, rows, predictions, or labels
# are written or printed.

phase59a_started = time.perf_counter()

assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)
assert isinstance(globals().get("PHASE58_GROUP_CENTERED_PILOT_REPORT_PRIVATE"), dict)
assert PHASE58_GROUP_CENTERED_PILOT_REPORT_PRIVATE["status"] == (
    "no_update_selected_stop_group_centered_residual_path"
)

phase59a_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase59a_labels = np.asarray(phase59a_state["labels"], dtype=np.int64).reshape(-1)
phase59a_groups = np.asarray(phase59a_state["groups"], dtype=np.int64).reshape(-1)
phase59a_probability = np.asarray(
    phase59a_state["anchor_probability"], dtype=np.float64
).reshape(-1)

if "phase57c_original_fold" in globals():
    phase59a_folds = np.asarray(phase57c_original_fold, dtype=np.int64).reshape(-1)
elif isinstance(globals().get("PHASE57_TRANSPORT_STATE_PRIVATE"), dict):
    phase59a_folds = np.asarray(
        PHASE57_TRANSPORT_STATE_PRIVATE["original_fold"], dtype=np.int64
    ).reshape(-1)
else:
    raise AssertionError("Accepted three-fold assignment is unavailable.")

assert phase59a_labels.shape == phase59a_groups.shape == phase59a_probability.shape == (1362,)
assert phase59a_folds.shape == (1362,)
assert set(np.unique(phase59a_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase59a_groups), np.arange(15))
assert np.array_equal(np.unique(phase59a_folds), np.arange(3))
assert np.all(np.isfinite(phase59a_probability))
assert np.all((phase59a_probability > 0) & (phase59a_probability < 1))

PHASE59A_CONFIG = {
    "schema_version": "phase59_oof_failure_anatomy_audit_v1",
    "probability_clip": 1.0e-7,
    "confidence_thresholds": [0.80, 0.90, 0.95],
    "ece_bin_count": 10,
    "minimum_group_class_count_for_auroc": 1,
    "contract_file": "/kaggle/working/phase59_oof_failure_anatomy_contract.json",
}


def phase59a_logit(p):
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(p) - np.log1p(-p)


def phase59a_sigmoid(z):
    z = np.asarray(z, dtype=np.float64)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    ez = np.exp(z[~pos])
    out[~pos] = ez / (1.0 + ez)
    return out


def phase59a_auc(y, s):
    y = np.asarray(y, dtype=np.int64)
    s = np.asarray(s, dtype=np.float64)
    positives = int(np.sum(y == 1))
    negatives = int(np.sum(y == 0))
    if positives == 0 or negatives == 0:
        return None
    order = np.argsort(s, kind="mergesort")
    sorted_s = s[order]
    ranks = np.empty(s.size, dtype=np.float64)
    start = 0
    while start < s.size:
        stop = start + 1
        while stop < s.size and sorted_s[stop] == sorted_s[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    u = float(np.sum(ranks[y == 1])) - positives * (positives + 1) / 2
    return float(u / (positives * negatives))


def phase59a_calibration_fit(y, p, fit_slope=True):
    # Descriptive in-sample logistic recalibration only; never a deployable fit.
    y = np.asarray(y, dtype=np.float64)
    x = phase59a_logit(p)
    design = np.column_stack([np.ones_like(x), x]) if fit_slope else np.ones((x.size, 1))
    beta = np.asarray([0.0, 1.0] if fit_slope else [0.0], dtype=np.float64)
    ridge = np.diag([1e-8, 1e-8] if fit_slope else [1e-8])
    for _ in range(100):
        q = phase59a_sigmoid(design @ beta)
        gradient = design.T @ (q - y) + ridge @ beta
        weight = np.maximum(q * (1 - q), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + ridge
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return {
        "intercept": float(beta[0]),
        "slope": float(beta[1]) if fit_slope else 1.0,
    }


def phase59a_ece(y, p, bins):
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.minimum(np.searchsorted(edges, p, side="right") - 1, bins - 1)
    ece = 0.0
    maximum_gap = 0.0
    occupied = 0
    for b in range(int(bins)):
        mask = index == b
        if not np.any(mask):
            continue
        gap = abs(float(np.mean(p[mask])) - float(np.mean(y[mask])))
        ece += float(np.mean(mask)) * gap
        maximum_gap = max(maximum_gap, gap)
        occupied += 1
    return {"ece": float(ece), "maximum_bin_gap": float(maximum_gap), "occupied_bins": occupied}


def phase59a_metrics(y, p):
    y = np.asarray(y, dtype=np.int64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    loss = -(y * np.log(p) + (1 - y) * np.log1p(-p))
    calibration = phase59a_calibration_fit(y, p, fit_slope=True)
    return {
        "n": int(y.size),
        "normal_n": int(np.sum(y == 0)),
        "pathologic_n": int(np.sum(y == 1)),
        "prevalence": float(np.mean(y)),
        "mean_probability": float(np.mean(p)),
        "log_loss": float(np.mean(loss)),
        "auroc": phase59a_auc(y, p),
        "brier": float(np.mean(np.square(p - y))),
        "calibration_intercept": calibration["intercept"],
        "calibration_slope": calibration["slope"],
        **phase59a_ece(y, p, PHASE59A_CONFIG["ece_bin_count"]),
    }


phase59a_case_loss = -(
    phase59a_labels * np.log(np.clip(phase59a_probability, 1e-7, 1 - 1e-7))
    + (1 - phase59a_labels) * np.log1p(-np.clip(phase59a_probability, 1e-7, 1 - 1e-7))
)
phase59a_total_loss = float(np.sum(phase59a_case_loss))
phase59a_descending_loss = np.sort(phase59a_case_loss)[::-1]

phase59a_concentration = {}
for fraction in (0.01, 0.05, 0.10, 0.20):
    count = max(1, int(np.ceil(fraction * phase59a_labels.size)))
    phase59a_concentration[f"top_{int(100*fraction)}_percent"] = {
        "case_count": count,
        "fraction_of_total_log_loss": float(np.sum(phase59a_descending_loss[:count]) / phase59a_total_loss),
        "minimum_case_loss_in_subset": float(phase59a_descending_loss[count - 1]),
    }

phase59a_confidence = []
predicted = (phase59a_probability >= 0.5).astype(np.int64)
for threshold in PHASE59A_CONFIG["confidence_thresholds"]:
    mask = (phase59a_probability >= threshold) | (phase59a_probability <= 1 - threshold)
    wrong = mask & (predicted != phase59a_labels)
    phase59a_confidence.append({
        "threshold": float(threshold),
        "confident_case_count": int(np.sum(mask)),
        "confident_error_count": int(np.sum(wrong)),
        "confident_error_rate": None if not np.any(mask) else float(np.mean(predicted[mask] != phase59a_labels[mask])),
        "fraction_of_total_log_loss_from_confident_errors": float(np.sum(phase59a_case_loss[wrong]) / phase59a_total_loss),
    })

phase59a_fold_records = []
for fold in range(3):
    mask = phase59a_folds == fold
    phase59a_fold_records.append({"fold": fold, **phase59a_metrics(phase59a_labels[mask], phase59a_probability[mask])})

phase59a_group_records = []
for group in range(15):
    mask = phase59a_groups == group
    record = {"group": group, "original_fold": int(np.unique(phase59a_folds[mask])[0]), **phase59a_metrics(phase59a_labels[mask], phase59a_probability[mask])}
    record["mean_case_log_loss_minus_pooled"] = float(np.mean(phase59a_case_loss[mask]) - np.mean(phase59a_case_loss))
    record["fraction_of_total_log_loss"] = float(np.sum(phase59a_case_loss[mask]) / phase59a_total_loss)
    phase59a_group_records.append(record)

worst_groups = sorted(
    phase59a_group_records, key=lambda r: r["log_loss"], reverse=True
)
best_groups = sorted(
    phase59a_group_records, key=lambda r: r["log_loss"]
)

phase59a_pooled = phase59a_metrics(phase59a_labels, phase59a_probability)
phase59a_report = {
    "phase": "phase59_deployment_matched_oof_failure_anatomy_audit",
    "status": "accepted_ready_for_failure_mode_decision",
    "source": {
        "phase58b_status_verified": True,
        "anchor_is_phase43_deployment_matched_oof": True,
    },
    "pooled": phase59a_pooled,
    "folds": phase59a_fold_records,
    "groups": phase59a_group_records,
    "worst_group_order": [r["group"] for r in worst_groups],
    "best_group_order": [r["group"] for r in best_groups],
    "loss_concentration": phase59a_concentration,
    "confidence_failures": phase59a_confidence,
    "interpretation_contract": {
        "calibration_statistics_are_descriptive_in_sample_only": True,
        "calibration_statistics_eligible_for_deployment_selection": False,
        "case_level_outputs_exported": False,
        "next_branch_if_fold_or_group_concentrated": "transport_and_router_retraining",
        "next_branch_if_diffuse_and_slope_miscalibrated": "nested_calibration",
        "next_branch_if_diffuse_and_well_calibrated": "independent_discrimination_model",
    },
    "fit_labels_used_for_aggregate_diagnostic": True,
    "training_voxel_cache_read": False,
    "training_nifti_files_read": False,
    "outer_validation_predictions_reused_only_as_existing_oof": True,
    "test_data_read": False,
    "uids_displayed": False,
    "case_level_predictions_displayed": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase59a_started, 3),
}

contract_core = {
    "schema_version": PHASE59A_CONFIG["schema_version"],
    "status": phase59a_report["status"],
    "config": PHASE59A_CONFIG,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}
contract_sha256 = hashlib.sha256(json.dumps(
    contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE59A_CONFIG["contract_file"])
contract_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = contract_path.with_suffix(contract_path.suffix + ".tmp")
temporary_path.write_text(json.dumps({
    **contract_core, "contract_sha256": contract_sha256
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, contract_path)
phase59a_report["contract_sha256"] = contract_sha256
phase59a_report["contract_file"] = str(contract_path)

PHASE59_OOF_FAILURE_AUDIT_REPORT_PRIVATE = phase59a_report

print("BEGIN SANITIZED_PHASE59_OOF_FAILURE_ANATOMY_AUDIT")
print(json.dumps(phase59a_report, indent=2))
print("END SANITIZED_PHASE59_OOF_FAILURE_ANATOMY_AUDIT")
