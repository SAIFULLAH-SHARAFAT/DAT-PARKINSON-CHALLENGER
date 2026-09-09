from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


# Cell 161B — Phase31 fixed-blend stability and provenance gate.
#
# Run after Cell 161A in the same live kernel. Alpha=0.20 is frozen from the
# Phase61A diagnostic and is the only advancement candidate. Neighbouring
# alphas are sensitivity diagnostics only. This cell reads no images, test
# cases, embeddings, or patient rows and writes no case-level output.

phase61b_started = time.perf_counter()

assert isinstance(
    globals().get("PHASE61_HISTORICAL_COMPLEMENTARITY_REPORT_PRIVATE"), dict
)
assert PHASE61_HISTORICAL_COMPLEMENTARITY_REPORT_PRIVATE["status"] == (
    "no_historical_family_signal_ready_for_anatomy_invariant_retraining_design"
)
assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)

phase61b_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase61b_labels = np.asarray(phase61b_state["labels"], dtype=np.int64).reshape(-1)
phase61b_groups = np.asarray(phase61b_state["groups"], dtype=np.int64).reshape(-1)
phase61b_anchor = np.asarray(
    phase61b_state["anchor_probability"], dtype=np.float64
).reshape(-1)

if "phase57c_original_fold" in globals():
    phase61b_folds = np.asarray(phase57c_original_fold, dtype=np.int64).reshape(-1)
elif isinstance(globals().get("PHASE57_TRANSPORT_STATE_PRIVATE"), dict):
    phase61b_folds = np.asarray(
        PHASE57_TRANSPORT_STATE_PRIVATE["original_fold"], dtype=np.int64
    ).reshape(-1)
else:
    raise AssertionError("Accepted acquisition-group fold assignment is unavailable.")

assert phase61b_labels.shape == phase61b_groups.shape == phase61b_anchor.shape == (1362,)
assert phase61b_folds.shape == (1362,)
assert set(np.unique(phase61b_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase61b_groups), np.arange(15))
assert np.array_equal(np.unique(phase61b_folds), np.arange(3))
assert np.all(np.isfinite(phase61b_anchor))
assert np.all((phase61b_anchor > 0.0) & (phase61b_anchor < 1.0))


PHASE61B_CONFIG = {
    "schema_version": "phase61_phase31_fixed_blend_stability_gate_v1",
    "fixed_alpha": 0.20,
    "sensitivity_alphas": [0.10, 0.15, 0.20, 0.25, 0.30],
    "probability_clip": 1.0e-7,
    "major_groups": [1, 3],
    "confidence_threshold": 0.80,
    "group_bootstrap_replicates": 5000,
    "group_bootstrap_seed": 611161,
    "phase31_oof_file": "/kaggle/working/phase31_classical_oof_private.npy",
    "phase31_selection_file": "/kaggle/working/phase31_classical_selection.json",
    "phase52_partition_file": "/kaggle/working/phase52_repeated_partition.npz",
    "numeric_gate": {
        "minimum_pooled_log_loss_gain": 0.003,
        "minimum_pooled_auroc_gain": 0.0,
        "minimum_original_fold_log_loss_gain": 0.001,
        "minimum_original_fold_wins": 3,
        "minimum_group_wins": 9,
        "maximum_major_group_log_loss_regret": 0.005,
        "minimum_leave_one_group_out_log_loss_gain": 0.001,
        "minimum_group_bootstrap_lower_95_log_loss_gain": 0.0,
    },
    "contract_file": "/kaggle/working/phase61_phase31_fixed_blend_stability_contract.json",
}


def phase61b_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE61B_CONFIG["probability_clip"],
        1.0 - PHASE61B_CONFIG["probability_clip"],
    )
    return np.log(probability) - np.log1p(-probability)


def phase61b_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    result = np.empty_like(logit)
    positive = logit >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def phase61b_auc(labels, score):
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


def phase61b_case_loss(labels, probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE61B_CONFIG["probability_clip"],
        1.0 - PHASE61B_CONFIG["probability_clip"],
    )
    labels = np.asarray(labels, dtype=np.int64)
    return -(
        labels * np.log(probability) + (1 - labels) * np.log1p(-probability)
    )


def phase61b_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = np.asarray(probability, dtype=np.float64)
    loss = phase61b_case_loss(labels, probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(loss)),
        "auroc": phase61b_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase61b_metric_comparison(labels, anchor, candidate):
    anchor_metrics = phase61b_metrics(labels, anchor)
    candidate_metrics = phase61b_metrics(labels, candidate)
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
        "anchor_brier": anchor_metrics["brier"],
        "candidate_brier": candidate_metrics["brier"],
        "brier_gain": anchor_metrics["brier"] - candidate_metrics["brier"],
    }


phase61b_oof_path = Path(PHASE61B_CONFIG["phase31_oof_file"])
phase61b_selection_path = Path(PHASE61B_CONFIG["phase31_selection_file"])
assert phase61b_oof_path.is_file(), {"missing": str(phase61b_oof_path)}
assert phase61b_selection_path.is_file(), {"missing": str(phase61b_selection_path)}

phase61b_classical = np.asarray(
    np.load(phase61b_oof_path, allow_pickle=False, mmap_mode="r"), dtype=np.float64
).reshape(-1)
assert phase61b_classical.shape == (1362,)
assert np.all(np.isfinite(phase61b_classical))
assert np.all((phase61b_classical > 0.0) & (phase61b_classical < 1.0))

phase61b_selection_bytes = phase61b_selection_path.read_bytes()
phase61b_selection = json.loads(phase61b_selection_bytes.decode("utf-8"))
assert isinstance(phase61b_selection, (dict, list))


def phase61b_safe_metadata_summary(value, path="", depth=0):
    """Retain only compact aggregate/provenance scalars; never row arrays."""
    if depth > 5:
        return {}
    blocked = ("uid", "patient", "case_index", "indices", "labels", "prediction")
    allowed = (
        "model", "feature", "fold", "group", "oof", "split", "seed", "metric",
        "loss", "auc", "calib", "status", "selected", "license", "count", "n_",
    )
    output = {}
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            lowered = key_text.lower()
            child_path = f"{path}.{key_text}" if path else key_text
            if any(token in lowered for token in blocked):
                continue
            if isinstance(child, dict):
                nested = phase61b_safe_metadata_summary(child, child_path, depth + 1)
                output.update(nested)
            elif isinstance(child, list):
                if len(child) <= 12 and all(
                    isinstance(item, (str, int, float, bool, type(None))) for item in child
                ) and any(token in lowered for token in allowed):
                    output[child_path] = child
                elif any(token in lowered for token in allowed):
                    output[child_path] = {"list_length": len(child), "values_omitted": True}
            elif isinstance(child, (str, int, float, bool, type(None))):
                if any(token in lowered for token in allowed):
                    output[child_path] = child
    elif isinstance(value, list):
        # Selection histories are commonly stored as a list of aggregate model
        # records. Inspect only the first eight compact records and never emit
        # long vectors or row-like arrays.
        for item_index, child in enumerate(value[:8]):
            if isinstance(child, dict):
                child_path = f"{path}[{item_index}]" if path else f"record[{item_index}]"
                nested = phase61b_safe_metadata_summary(child, child_path, depth + 1)
                output.update(nested)
    return output


phase61b_metadata_summary = phase61b_safe_metadata_summary(phase61b_selection)
phase61b_metadata_summary = dict(list(sorted(phase61b_metadata_summary.items()))[:80])

if isinstance(phase61b_selection, dict):
    phase61b_selection_structure = {
        "top_level_type": "object",
        "top_level_keys": sorted(str(key) for key in phase61b_selection.keys()),
    }
else:
    phase61b_record_key_union = sorted({
        str(key)
        for record in phase61b_selection[:20]
        if isinstance(record, dict)
        for key in record.keys()
    })
    phase61b_selection_structure = {
        "top_level_type": "list",
        "record_count": len(phase61b_selection),
        "first_twenty_record_key_union": phase61b_record_key_union,
        "all_first_twenty_records_are_objects": all(
            isinstance(record, dict) for record in phase61b_selection[:20]
        ),
    }

anchor_logit = phase61b_logit(phase61b_anchor)
classical_logit = phase61b_logit(phase61b_classical)


def phase61b_blend(alpha):
    return phase61b_sigmoid(
        (1.0 - float(alpha)) * anchor_logit + float(alpha) * classical_logit
    )


fixed_probability = phase61b_blend(PHASE61B_CONFIG["fixed_alpha"])
pooled = phase61b_metric_comparison(
    phase61b_labels, phase61b_anchor, fixed_probability
)

fold_records = []
for fold in range(3):
    mask = phase61b_folds == fold
    fold_records.append({
        "fold": fold,
        **phase61b_metric_comparison(
            phase61b_labels[mask], phase61b_anchor[mask], fixed_probability[mask]
        ),
    })

group_records = []
for group in range(15):
    mask = phase61b_groups == group
    group_records.append({
        "group": group,
        "original_fold": int(np.unique(phase61b_folds[mask])[0]),
        **phase61b_metric_comparison(
            phase61b_labels[mask], phase61b_anchor[mask], fixed_probability[mask]
        ),
    })

leave_one_group_out = []
for group in range(15):
    mask = phase61b_groups != group
    comparison = phase61b_metric_comparison(
        phase61b_labels[mask], phase61b_anchor[mask], fixed_probability[mask]
    )
    leave_one_group_out.append({
        "excluded_group": group,
        "n": comparison["n"],
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
    })

# Cluster bootstrap over acquisition groups. Group multiplicity is respected by
# concatenation; this estimates sensitivity to the observed group composition.
bootstrap_rng = np.random.default_rng(PHASE61B_CONFIG["group_bootstrap_seed"])
group_indices = [np.flatnonzero(phase61b_groups == group) for group in range(15)]
bootstrap_gain = np.empty(PHASE61B_CONFIG["group_bootstrap_replicates"], dtype=np.float64)
for replicate in range(PHASE61B_CONFIG["group_bootstrap_replicates"]):
    sampled_groups = bootstrap_rng.integers(0, 15, size=15)
    sampled_indices = np.concatenate([group_indices[int(group)] for group in sampled_groups])
    anchor_loss = phase61b_case_loss(
        phase61b_labels[sampled_indices], phase61b_anchor[sampled_indices]
    )
    candidate_loss = phase61b_case_loss(
        phase61b_labels[sampled_indices], fixed_probability[sampled_indices]
    )
    bootstrap_gain[replicate] = float(np.mean(anchor_loss - candidate_loss))

bootstrap_report = {
    "replicates": int(bootstrap_gain.size),
    "mean_log_loss_gain": float(np.mean(bootstrap_gain)),
    "lower_95_log_loss_gain": float(np.quantile(bootstrap_gain, 0.025)),
    "median_log_loss_gain": float(np.quantile(bootstrap_gain, 0.50)),
    "upper_95_log_loss_gain": float(np.quantile(bootstrap_gain, 0.975)),
    "fraction_positive": float(np.mean(bootstrap_gain > 0.0)),
}

sensitivity_records = []
for alpha in PHASE61B_CONFIG["sensitivity_alphas"]:
    probability = phase61b_blend(alpha)
    comparison = phase61b_metric_comparison(
        phase61b_labels, phase61b_anchor, probability
    )
    sensitivity_records.append({
        "alpha": float(alpha),
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
        "deployment_selection_eligible": bool(
            abs(float(alpha) - PHASE61B_CONFIG["fixed_alpha"]) < 1.0e-12
        ),
    })

anchor_predicted = (phase61b_anchor >= 0.5).astype(np.int64)
confident_mask = (
    (phase61b_anchor >= PHASE61B_CONFIG["confidence_threshold"])
    | (phase61b_anchor <= 1.0 - PHASE61B_CONFIG["confidence_threshold"])
)
confident_error_mask = confident_mask & (anchor_predicted != phase61b_labels)
anchor_loss = phase61b_case_loss(phase61b_labels, phase61b_anchor)
fixed_loss = phase61b_case_loss(phase61b_labels, fixed_probability)
confidence_report = {
    "anchor_confident_error_count": int(np.sum(confident_error_mask)),
    "anchor_confident_error_mean_loss": float(np.mean(anchor_loss[confident_error_mask])),
    "fixed_blend_confident_error_mean_loss": float(np.mean(fixed_loss[confident_error_mask])),
    "confident_error_mean_log_loss_gain": float(np.mean(
        anchor_loss[confident_error_mask] - fixed_loss[confident_error_mask]
    )),
    "remaining_cases_mean_log_loss_gain": float(np.mean(
        anchor_loss[~confident_error_mask] - fixed_loss[~confident_error_mask]
    )),
}

repeated_partition_report = {
    "available": False,
    "interpretation": "secondary_case_resampling_stability_not_domain_holdout",
}
partition_path = Path(PHASE61B_CONFIG["phase52_partition_file"])
if partition_path.is_file():
    with np.load(partition_path, allow_pickle=False) as partition_asset:
        repeated_assignment = np.asarray(partition_asset["fold_assignment"], dtype=np.int64)
    assert repeated_assignment.shape == (5, 1362)
    repeated_records = []
    for repeat in range(5):
        for fold in range(5):
            mask = repeated_assignment[repeat] == fold
            comparison = phase61b_metric_comparison(
                phase61b_labels[mask], phase61b_anchor[mask], fixed_probability[mask]
            )
            repeated_records.append({
                "repeat": repeat,
                "fold": fold,
                "n": comparison["n"],
                "log_loss_gain": comparison["log_loss_gain"],
                "auroc_gain": comparison["auroc_gain"],
            })
    repeated_partition_report = {
        "available": True,
        "partition_count": len(repeated_records),
        "log_loss_win_count": int(sum(item["log_loss_gain"] > 0 for item in repeated_records)),
        "minimum_log_loss_gain": float(min(item["log_loss_gain"] for item in repeated_records)),
        "median_log_loss_gain": float(np.median([
            item["log_loss_gain"] for item in repeated_records
        ])),
        "maximum_log_loss_gain": float(max(item["log_loss_gain"] for item in repeated_records)),
        "minimum_auroc_gain": float(min(item["auroc_gain"] for item in repeated_records)),
        "interpretation": "secondary_case_resampling_stability_not_domain_holdout",
    }

gate = PHASE61B_CONFIG["numeric_gate"]
fold_wins = int(sum(
    record["log_loss_gain"] >= gate["minimum_original_fold_log_loss_gain"]
    for record in fold_records
))
group_wins = int(sum(record["log_loss_gain"] > 0.0 for record in group_records))
major_group_maximum_regret = float(max(
    -group_records[group]["log_loss_gain"] for group in PHASE61B_CONFIG["major_groups"]
))
minimum_logo_gain = float(min(
    record["log_loss_gain"] for record in leave_one_group_out
))

numeric_advanced = bool(
    pooled["log_loss_gain"] >= gate["minimum_pooled_log_loss_gain"]
    and pooled["auroc_gain"] >= gate["minimum_pooled_auroc_gain"]
    and fold_wins >= gate["minimum_original_fold_wins"]
    and group_wins >= gate["minimum_group_wins"]
    and major_group_maximum_regret <= gate["maximum_major_group_log_loss_regret"]
    and minimum_logo_gain >= gate["minimum_leave_one_group_out_log_loss_gain"]
    and bootstrap_report["lower_95_log_loss_gain"]
    >= gate["minimum_group_bootstrap_lower_95_log_loss_gain"]
)

phase61b_status = (
    "numeric_stability_passed_requires_phase31_provenance_review"
    if numeric_advanced
    else "fixed_blend_stability_failed_stop_phase31_blend"
)

contract_core = {
    "schema_version": PHASE61B_CONFIG["schema_version"],
    "status": phase61b_status,
    "config": PHASE61B_CONFIG,
    "fixed_alpha_selected_before_cell": True,
    "numeric_advanced": numeric_advanced,
    "provenance_advanced": False,
    "test_data_used": False,
}
contract_sha256 = hashlib.sha256(json.dumps(
    contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE61B_CONFIG["contract_file"])
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

phase61b_report = {
    "phase": "phase61_phase31_fixed_blend_stability_and_provenance_gate",
    "status": phase61b_status,
    "candidate": {
        "formula": "sigmoid(0.80*anchor_logit+0.20*phase31_classical_logit)",
        "fixed_alpha": PHASE61B_CONFIG["fixed_alpha"],
        "fixed_before_cell_execution": True,
        "numeric_advanced": numeric_advanced,
        "deployment_advanced": False,
    },
    "pooled": pooled,
    "original_folds": fold_records,
    "groups": group_records,
    "stability_summary": {
        "original_fold_win_count": fold_wins,
        "group_win_count": group_wins,
        "major_group_maximum_log_loss_regret": major_group_maximum_regret,
        "minimum_leave_one_group_out_log_loss_gain": minimum_logo_gain,
        "cluster_bootstrap": bootstrap_report,
        "repeated_partition": repeated_partition_report,
    },
    "confidence_effect": confidence_report,
    "alpha_sensitivity_diagnostic": sensitivity_records,
    "phase31_provenance": {
        "selection_file_present": True,
        "selection_file_sha256": hashlib.sha256(phase61b_selection_bytes).hexdigest(),
        "selection_file_structure": phase61b_selection_structure,
        "safe_aggregate_metadata": phase61b_metadata_summary,
        "automatic_provenance_approval": False,
        "manual_code_or_contract_review_required": True,
    },
    "interpretation_contract": {
        "alpha_020_only_advancement_candidate": True,
        "sensitivity_alphas_eligible_for_selection": False,
        "numeric_pass_sufficient_for_deployment": False,
        "phase31_oof_provenance_must_be_verified": True,
        "runtime_asset_reproducibility_must_be_verified": True,
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
    "elapsed_seconds": round(time.perf_counter() - phase61b_started, 3),
}

PHASE61_PHASE31_FIXED_BLEND_REPORT_PRIVATE = phase61b_report

print("BEGIN SANITIZED_PHASE61_PHASE31_FIXED_BLEND_STABILITY_GATE")
print(json.dumps(phase61b_report, indent=2))
print("END SANITIZED_PHASE61_PHASE31_FIXED_BLEND_STABILITY_GATE")
