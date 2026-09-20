from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 159B — frozen-anchor calibration ceiling and confidence-risk audit.
# Run immediately after accepted Cell 159A. This cell performs descriptive
# diagnostics only. Oracle fits are explicitly ineligible for deployment.

phase59b_started = time.perf_counter()

for phase59b_name in (
    "PHASE59_OOF_FAILURE_AUDIT_REPORT_PRIVATE",
    "phase59a_labels",
    "phase59a_groups",
    "phase59a_folds",
    "phase59a_probability",
    "phase59a_metrics",
    "phase59a_logit",
    "phase59a_sigmoid",
):
    assert phase59b_name in globals(), {"missing": phase59b_name}

assert PHASE59_OOF_FAILURE_AUDIT_REPORT_PRIVATE["status"] == (
    "accepted_ready_for_failure_mode_decision"
)

PHASE59B_CONFIG = {
    "schema_version": "phase59_calibration_ceiling_audit_v1",
    "logit_scale_grid": [0.60, 0.70, 0.80, 0.90, 1.00, 1.10],
    "probability_cap_grid": [0.90, 0.925, 0.95, 0.975, 0.99],
    "major_group_minimum_n": 30,
    "newton_iterations": 100,
    "ridge": 1.0e-6,
    "material_calibration_gain": 0.005,
    "target_log_loss": 0.24,
    "contract_file": "/kaggle/working/phase59_calibration_ceiling_contract.json",
}

phase59b_y = np.asarray(phase59a_labels, dtype=np.int64)
phase59b_p = np.asarray(phase59a_probability, dtype=np.float64)
phase59b_g = np.asarray(phase59a_groups, dtype=np.int64)
phase59b_f = np.asarray(phase59a_folds, dtype=np.int64)
phase59b_z = phase59a_logit(phase59b_p)
phase59b_anchor = phase59a_metrics(phase59b_y, phase59b_p)


def phase59b_log_loss(y, p):
    y = np.asarray(y, dtype=np.float64)
    p = np.clip(np.asarray(p, dtype=np.float64), 1e-7, 1 - 1e-7)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log1p(-p))))


def phase59b_fit(y, z, mode):
    # Unregularized-near-oracle descriptive recalibration. Not deployable.
    y = np.asarray(y, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    if mode == "intercept":
        design = np.ones((y.size, 1), dtype=np.float64)
        beta = np.zeros(1, dtype=np.float64)
        offset = z
    elif mode == "temperature":
        design = z[:, None]
        beta = np.ones(1, dtype=np.float64)
        offset = np.zeros_like(z)
    elif mode == "platt":
        design = np.column_stack([np.ones_like(z), z])
        beta = np.asarray([0.0, 1.0], dtype=np.float64)
        offset = np.zeros_like(z)
    else:
        raise ValueError(mode)
    ridge = PHASE59B_CONFIG["ridge"] * np.eye(beta.size)
    for _ in range(PHASE59B_CONFIG["newton_iterations"]):
        q = phase59a_sigmoid(offset + design @ beta)
        gradient = design.T @ (q - y) + ridge @ beta
        weight = np.maximum(q * (1 - q), 1e-8)
        hessian = design.T @ (weight[:, None] * design) + ridge
        step = np.linalg.solve(hessian, gradient)
        beta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return beta


def phase59b_apply(z, beta, mode):
    z = np.asarray(z, dtype=np.float64)
    if mode == "intercept":
        transformed = z + float(beta[0])
    elif mode == "temperature":
        transformed = float(beta[0]) * z
    elif mode == "platt":
        transformed = float(beta[0]) + float(beta[1]) * z
    else:
        raise ValueError(mode)
    return phase59a_sigmoid(transformed)


phase59b_pooled_oracles = []
for mode in ("intercept", "temperature", "platt"):
    beta = phase59b_fit(phase59b_y, phase59b_z, mode)
    q = phase59b_apply(phase59b_z, beta, mode)
    metrics = phase59a_metrics(phase59b_y, q)
    phase59b_pooled_oracles.append({
        "mode": mode,
        "parameters": beta.astype(float).tolist(),
        "metrics": metrics,
        "log_loss_gain": float(phase59b_anchor["log_loss"] - metrics["log_loss"]),
        "descriptive_oracle_not_deployable": True,
    })

phase59b_fixed_scale = []
for scale in PHASE59B_CONFIG["logit_scale_grid"]:
    q = phase59a_sigmoid(float(scale) * phase59b_z)
    metrics = phase59a_metrics(phase59b_y, q)
    phase59b_fixed_scale.append({
        "logit_scale": float(scale),
        "metrics": metrics,
        "log_loss_gain": float(phase59b_anchor["log_loss"] - metrics["log_loss"]),
    })

phase59b_fixed_caps = []
for cap in PHASE59B_CONFIG["probability_cap_grid"]:
    q = np.clip(phase59b_p, 1.0 - float(cap), float(cap))
    metrics = phase59a_metrics(phase59b_y, q)
    phase59b_fixed_caps.append({
        "symmetric_probability_cap": float(cap),
        "metrics": metrics,
        "log_loss_gain": float(phase59b_anchor["log_loss"] - metrics["log_loss"]),
    })


def phase59b_partition_oracle(partition, minimum_n, label):
    q = phase59b_p.copy()
    records = []
    for value in np.unique(partition):
        mask = partition == value
        if int(np.sum(mask)) < int(minimum_n):
            records.append({
                "partition_value": int(value),
                "n": int(np.sum(mask)),
                "fitted": False,
            })
            continue
        beta = phase59b_fit(phase59b_y[mask], phase59b_z[mask], "platt")
        q[mask] = phase59b_apply(phase59b_z[mask], beta, "platt")
        records.append({
            "partition_value": int(value),
            "n": int(np.sum(mask)),
            "fitted": True,
            "parameters": beta.astype(float).tolist(),
            "identity_log_loss": phase59b_log_loss(phase59b_y[mask], phase59b_p[mask]),
            "oracle_log_loss": phase59b_log_loss(phase59b_y[mask], q[mask]),
        })
    metrics = phase59a_metrics(phase59b_y, q)
    return {
        "partition": label,
        "minimum_n": int(minimum_n),
        "records": records,
        "metrics": metrics,
        "log_loss_gain": float(phase59b_anchor["log_loss"] - metrics["log_loss"]),
        "descriptive_oracle_not_deployable": True,
    }


phase59b_fold_oracle = phase59b_partition_oracle(phase59b_f, 1, "original_fold")
phase59b_major_group_oracle = phase59b_partition_oracle(
    phase59b_g, PHASE59B_CONFIG["major_group_minimum_n"], "major_acquisition_group"
)

phase59b_best_pooled = max(phase59b_pooled_oracles, key=lambda r: r["log_loss_gain"])
phase59b_best_fixed_scale = max(phase59b_fixed_scale, key=lambda r: r["log_loss_gain"])
phase59b_best_fixed_cap = max(phase59b_fixed_caps, key=lambda r: r["log_loss_gain"])
phase59b_gap_to_target = float(phase59b_anchor["log_loss"] - PHASE59B_CONFIG["target_log_loss"])
phase59b_best_pooled_fraction = float(
    phase59b_best_pooled["log_loss_gain"] / phase59b_gap_to_target
)

if phase59b_best_pooled["log_loss_gain"] >= PHASE59B_CONFIG["material_calibration_gain"]:
    phase59b_status = "calibration_material_but_requires_strict_nested_confirmation"
else:
    phase59b_status = "calibration_secondary_transport_discrimination_primary"

phase59b_report = {
    "phase": "phase59_frozen_anchor_calibration_ceiling_audit",
    "status": phase59b_status,
    "anchor": phase59b_anchor,
    "target_gap": phase59b_gap_to_target,
    "pooled_oracle_calibrators": phase59b_pooled_oracles,
    "fixed_logit_scale_grid": phase59b_fixed_scale,
    "fixed_probability_cap_grid": phase59b_fixed_caps,
    "fold_specific_oracle": phase59b_fold_oracle,
    "major_group_specific_oracle": phase59b_major_group_oracle,
    "decision_summary": {
        "best_pooled_oracle_mode": phase59b_best_pooled["mode"],
        "best_pooled_oracle_log_loss_gain": phase59b_best_pooled["log_loss_gain"],
        "fraction_of_target_gap_explained_by_best_pooled_oracle": phase59b_best_pooled_fraction,
        "best_fixed_logit_scale": phase59b_best_fixed_scale["logit_scale"],
        "best_fixed_logit_scale_gain": phase59b_best_fixed_scale["log_loss_gain"],
        "best_fixed_probability_cap": phase59b_best_fixed_cap["symmetric_probability_cap"],
        "best_fixed_probability_cap_gain": phase59b_best_fixed_cap["log_loss_gain"],
        "fold_oracle_gain": phase59b_fold_oracle["log_loss_gain"],
        "major_group_oracle_gain": phase59b_major_group_oracle["log_loss_gain"],
        "oracle_results_eligible_for_selection": False,
    },
    "compliance": {
        "model_training_performed": False,
        "test_data_read": False,
        "outer_predictions_reused_only_as_existing_oof": True,
        "case_level_predictions_displayed": False,
        "case_level_predictions_exported": False,
        "oracle_calibrators_banned_from_deployment": True,
    },
    "elapsed_seconds": round(time.perf_counter() - phase59b_started, 3),
}

contract_core = {
    "schema_version": PHASE59B_CONFIG["schema_version"],
    "status": phase59b_status,
    "config": PHASE59B_CONFIG,
    "oracle_calibrators_deployable": False,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}
contract_sha256 = hashlib.sha256(json.dumps(
    contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE59B_CONFIG["contract_file"])
contract_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = contract_path.with_suffix(contract_path.suffix + ".tmp")
temporary_path.write_text(json.dumps({
    **contract_core, "contract_sha256": contract_sha256
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, contract_path)
phase59b_report["contract_sha256"] = contract_sha256
phase59b_report["contract_file"] = str(contract_path)

PHASE59_CALIBRATION_CEILING_REPORT_PRIVATE = phase59b_report

print("BEGIN SANITIZED_PHASE59_CALIBRATION_CEILING_AUDIT")
print(json.dumps(phase59b_report, indent=2))
print("END SANITIZED_PHASE59_CALIBRATION_CEILING_AUDIT")
