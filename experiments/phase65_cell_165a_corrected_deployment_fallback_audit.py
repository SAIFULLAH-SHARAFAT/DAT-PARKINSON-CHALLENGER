# Cell 165A — corrected Phase65A deployment fallback and parity audit
#
# Pre-registration
# ----------------
# H65A-1. The Phase56 runtime uses Phase12c alone for unknown acquisition
# protocols. The exact Phase43 deployment-matched three-member anchor is
# stronger overall. This cell asks one development-only question: is replacing
# the unknown-protocol fallback with the full three-member anchor (sparse
# adapter disabled) supported across the frozen development domains?
#
# H65A-2. The public Phase56 displacement is an operational warning, not an
# estimator of unknown-protocol exposure. Log loss is linear across cases, but
# an exposure fraction is not identifiable when routing is domain-dependent;
# AUROC is not linear under case-wise mixture at all. No exposure fraction is
# estimated and no public score enters a fitted quantity or advancement gate.
#
# H65A-3. Pure class-prior shift changes a calibrated logit by an additive
# intercept, not by multiplying the logit. A static minimax intercept is
# therefore audited over a declared prevalence family. It is diagnostic only:
# the available anchor vector is Phase43 deployment-matched OOF, not an exact
# OOF reconstruction of every Phase56 routed/adapted runtime branch. The
# selected deployable offset remains exactly zero.
#
# Data boundary. Development labels, acquisition groups, frozen validation
# partitions, and deployment-matched OOF vectors only. No NIfTI, voxel cache,
# test data, smoke-test data, external assets, or leaderboard predictions.
#
# Test independence. Any fallback recommendation is a static branch change.
# It uses only the current case header plus packaged development constants.
# Model weights and fitted transforms remain identical for every possible test
# set, including an empty test set.
#
# Stop rule. The fallback candidate is evaluated once. Failure of any frozen
# criterion retains Phase12c for unknown protocols. A numerical prior-offset
# result never changes deployment in this cell.

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np


phase65a_started = time.perf_counter()

assert isinstance(globals().get("PHASE62_VALIDATION_RESET_STATE_PRIVATE"), dict), {
    "message": "Cell 165A requires the accepted Phase62A validation reset state."
}
assert isinstance(globals().get("PHASE64A_DIFFUSION_REPORT_PRIVATE"), dict), {
    "message": "Cell 165A must run after the completed Phase64A gate."
}
assert PHASE64A_DIFFUSION_REPORT_PRIVATE["status"] == (
    "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
), {"message": "Phase64 must remain rejected before Phase65 begins."}

phase65a_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE


PHASE65A_CONFIG = {
    "schema_version": "phase65a_corrected_deployment_fallback_audit_v2",
    "seed": 650165,
    "probability_clip": 1.0e-7,
    "major_groups": [1, 3],
    # Historical public outcomes are descriptive deployment observations only.
    # They are excluded from every fitted quantity and gate below.
    "observed_public": [
        {
            "name": "phase42_submission",
            "log_loss": 0.3254,
            "auroc": 0.9251,
            "deployment_path": "phase42_nested_alpha",
        },
        {
            "name": "phase56_submission",
            "log_loss": 0.3220,
            "auroc": 0.9297,
            "deployment_path": (
                "three_member_average_with_sparse_adapter_and_"
                "phase12c_only_unknown_protocol_fallback"
            ),
        },
    ],
    "public_reference_leader": {"log_loss": 0.2279, "auroc": 0.9676},
    "vector_catalogue": [
        {
            "name": "phase12c_oof",
            "paths": ["/kaggle/working/phase32_phase12c_oof_float64.npy"],
            "deployment_eligible": True,
            "role": "current_unknown_protocol_fallback",
            "required": True,
        },
        # Optional inventory only. These vectors never enter a reconstructed
        # candidate because the exact deployment-matched anchor is supplied by
        # the accepted Phase62 state.
        {
            "name": "phase33_component_oof",
            "paths": [
                "/kaggle/working/phase33_private_checkpoint/"
                "phase33_component_oof_float64.npy",
                "/kaggle/working/phase33_component_oof_float64.npy",
            ],
            "deployment_eligible": True,
            "role": "inventory_only_ensemble_member",
            "required": False,
        },
        {
            "name": "phase39_oof",
            "paths": [
                "/kaggle/working/phase39_private_checkpoint/"
                "phase39_oof_float64.npy",
                "/kaggle/working/phase39_oof_float64.npy",
            ],
            "deployment_eligible": True,
            "role": "inventory_only_ensemble_member",
            "required": False,
        },
    ],
    # This is a fallback-policy comparison between two already accepted OOF
    # vectors, not promotion of a newly trained expert. Bounded small-group
    # regret is permitted, but the two major failing groups and every original
    # fold receive zero-regret protection.
    "fallback_gate": {
        "minimum_pooled_log_loss_gain": 0.005,
        "minimum_pooled_auroc_gain": 0.002,
        "brier_regret_allowed": 0.0,
        "maximum_individual_group_log_loss_regret": 0.005,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "maximum_individual_domain_log_loss_regret": 0.005,
        "maximum_original_fold_log_loss_regret": 0.0,
        "minimum_improved_group_fraction": 0.75,
        "minimum_repeat_partition_log_loss_win_fraction": 0.85,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
    },
    # Diagnostic only. A pure label-prior change is an additive logit offset.
    "prior_uncertainty_diagnostic": {
        "prevalence_grid": [
            0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70
        ],
        "logit_offset_grid": [
            -0.80, -0.70, -0.60, -0.50, -0.40, -0.30, -0.20,
            -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15, 0.20,
            0.30, 0.40, 0.50, 0.60, 0.70, 0.80,
        ],
        "numeric_support_gate": {
            "maximum_in_distribution_log_loss_regret": 0.0005,
            "minimum_worst_case_log_loss_gain": 0.005,
            "maximum_individual_group_in_distribution_regret": 0.002,
        },
        "deployment_eligible_in_this_cell": False,
        "reason": (
            "exact_phase56_routed_and_adapted_oof_vector_not_available;_"
            "phase43_anchor_is_not_sufficient_to_modify_every_runtime_branch"
        ),
    },
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 650166,
    "contract_file": (
        "/kaggle/working/phase65a_corrected_deployment_fallback_contract.json"
    ),
}


phase65a_override = globals().get("PHASE65A_SYNTHETIC_TEST_OVERRIDE_PRIVATE")
phase65a_is_synthetic = isinstance(phase65a_override, dict)
if phase65a_is_synthetic:
    PHASE65A_CONFIG = {
        **PHASE65A_CONFIG,
        "vector_catalogue": list(phase65a_override["vector_catalogue"]),
        "domain_bootstrap_replicates": int(
            phase65a_override.get("domain_bootstrap_replicates", 500)
        ),
        "contract_file": str(
            phase65a_override.get(
                "contract_file", "/tmp/phase65a_corrected_synthetic_contract.json"
            )
        ),
    }

phase65a_expected_n = (
    int(phase65a_override["case_count"]) if phase65a_is_synthetic else 1362
)
phase65a_expected_groups = (
    int(phase65a_override.get("group_count", 15))
    if phase65a_is_synthetic
    else 15
)


phase65a_labels = np.asarray(
    phase65a_validation["labels"], dtype=np.int64
).reshape(-1)
phase65a_groups = np.asarray(
    phase65a_validation["groups"], dtype=np.int64
).reshape(-1)
phase65a_anchor = np.asarray(
    phase65a_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase65a_original_fold = np.asarray(
    phase65a_validation["original_fold"], dtype=np.int64
).reshape(-1)
phase65a_domain_id = np.asarray(
    phase65a_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase65a_domain_definitions = phase65a_validation["stress_domain_definitions"]
phase65a_repeat_fold = np.asarray(
    phase65a_validation["group_blocked_repeat_case_fold"], dtype=np.int64
)

assert phase65a_labels.shape == (phase65a_expected_n,)
assert phase65a_groups.shape == (phase65a_expected_n,)
assert phase65a_anchor.shape == (phase65a_expected_n,)
assert phase65a_original_fold.shape == (phase65a_expected_n,)
assert phase65a_domain_id.shape == (phase65a_expected_n,)
assert set(np.unique(phase65a_labels).tolist()) == {0, 1}
assert np.array_equal(
    np.unique(phase65a_groups), np.arange(phase65a_expected_groups)
)
assert np.all(np.isfinite(phase65a_anchor))
assert np.all((phase65a_anchor > 0.0) & (phase65a_anchor < 1.0))
assert phase65a_repeat_fold.ndim == 2
assert phase65a_repeat_fold.shape[1] == phase65a_expected_n
assert np.all(np.isin(phase65a_repeat_fold, [0, 1, 2]))
for phase65a_repeat in range(phase65a_repeat_fold.shape[0]):
    for phase65a_group in range(phase65a_expected_groups):
        assert np.unique(
            phase65a_repeat_fold[
                phase65a_repeat, phase65a_groups == phase65a_group
            ]
        ).size == 1


def phase65a_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE65A_CONFIG["probability_clip"],
        1.0 - PHASE65A_CONFIG["probability_clip"],
    )


def phase65a_logit(probability):
    probability = phase65a_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase65a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase65a_auc(labels, score):
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
    statistic = float(np.sum(ranks[labels == 1]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def phase65a_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase65a_clip(probability)
    return -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )


def phase65a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase65a_clip(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase65a_case_loss(labels, probability))),
        "auroc": phase65a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase65a_compare(labels, baseline, candidate):
    baseline_metrics = phase65a_metrics(labels, baseline)
    candidate_metrics = phase65a_metrics(labels, candidate)
    auroc_gain = None
    if (
        baseline_metrics["auroc"] is not None
        and candidate_metrics["auroc"] is not None
    ):
        auroc_gain = float(
            candidate_metrics["auroc"] - baseline_metrics["auroc"]
        )
    return {
        "n": int(labels.size),
        "baseline_log_loss": baseline_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            baseline_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "baseline_auroc": baseline_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": auroc_gain,
        "brier_gain": float(
            baseline_metrics["brier"] - candidate_metrics["brier"]
        ),
    }


def phase65a_domain_gains(baseline, candidate):
    gains = []
    for domain_index in range(len(phase65a_domain_definitions)):
        mask = phase65a_domain_id == domain_index
        assert int(np.sum(mask)) > 0
        gains.append(
            float(
                np.mean(
                    phase65a_case_loss(
                        phase65a_labels[mask], baseline[mask]
                    )
                )
                - np.mean(
                    phase65a_case_loss(
                        phase65a_labels[mask], candidate[mask]
                    )
                )
            )
        )
    return np.asarray(gains, dtype=np.float64)


def phase65a_domain_bootstrap(domain_gains):
    domain_gains = np.asarray(domain_gains, dtype=np.float64)
    assert domain_gains.ndim == 1 and domain_gains.size > 0
    rng = np.random.default_rng(PHASE65A_CONFIG["domain_bootstrap_seed"])
    replicate_index = rng.integers(
        0,
        domain_gains.size,
        size=(
            PHASE65A_CONFIG["domain_bootstrap_replicates"],
            domain_gains.size,
        ),
    )
    replicate = np.mean(domain_gains[replicate_index], axis=1)
    lower, median, upper = np.quantile(
        replicate, [0.025, 0.50, 0.975]
    )
    return {
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "replicates": int(PHASE65A_CONFIG["domain_bootstrap_replicates"]),
    }


# ---------------------------------------------------------------------------
# 1. Load the exact Phase12c OOF vector and optional inventory vectors.
# ---------------------------------------------------------------------------

phase65a_vectors = {}
phase65a_vector_records = []
for phase65a_entry in PHASE65A_CONFIG["vector_catalogue"]:
    resolved = None
    for candidate_path in phase65a_entry["paths"]:
        if Path(candidate_path).is_file():
            resolved = Path(candidate_path)
            break
    if resolved is None:
        assert not phase65a_entry.get("required", False), {
            "message": "A required deployment-matched OOF vector is missing.",
            "vector": phase65a_entry["name"],
            "searched": list(phase65a_entry["paths"]),
        }
        phase65a_vector_records.append(
            {
                "name": phase65a_entry["name"],
                "available": False,
                "role": phase65a_entry["role"],
            }
        )
        continue
    values = np.asarray(
        np.load(resolved, allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    assert values.shape == (phase65a_expected_n,), {
        "vector": phase65a_entry["name"],
        "shape": list(values.shape),
    }
    assert np.all(np.isfinite(values))
    assert np.all((values > 0.0) & (values < 1.0))
    phase65a_vectors[phase65a_entry["name"]] = values
    phase65a_vector_records.append(
        {
            "name": phase65a_entry["name"],
            "available": True,
            "role": phase65a_entry["role"],
            "deployment_eligible": bool(
                phase65a_entry["deployment_eligible"]
            ),
            "pooled": phase65a_metrics(phase65a_labels, values),
        }
    )

assert "phase12c_oof" in phase65a_vectors
phase65a_phase12c = phase65a_vectors["phase12c_oof"]

phase65a_anchor_metrics = phase65a_metrics(
    phase65a_labels, phase65a_anchor
)
phase65a_phase12c_metrics = phase65a_metrics(
    phase65a_labels, phase65a_phase12c
)

if not phase65a_is_synthetic:
    assert abs(
        phase65a_anchor_metrics["log_loss"] - 0.29016628416289664
    ) <= 5.0e-7
    assert abs(
        phase65a_anchor_metrics["auroc"] - 0.9464742438589044
    ) <= 5.0e-7
    assert abs(
        phase65a_phase12c_metrics["log_loss"] - 0.30761581113729136
    ) <= 1.0e-9
    assert abs(
        phase65a_phase12c_metrics["auroc"] - 0.9389144654498752
    ) <= 1.0e-9


# ---------------------------------------------------------------------------
# 2. Descriptive deployment discrepancy. No exposure fraction is estimated.
# ---------------------------------------------------------------------------

phase65a_public_discrepancy = []
for observation in PHASE65A_CONFIG["observed_public"]:
    phase65a_public_discrepancy.append(
        {
            "submission": observation["name"],
            "deployment_path": observation["deployment_path"],
            "observed_log_loss": float(observation["log_loss"]),
            "observed_auroc": float(observation["auroc"]),
            "log_loss_gap_vs_development_anchor": float(
                observation["log_loss"]
                - phase65a_anchor_metrics["log_loss"]
            ),
            "log_loss_gap_vs_development_phase12c": float(
                observation["log_loss"]
                - phase65a_phase12c_metrics["log_loss"]
            ),
            "auroc_gap_vs_development_anchor": float(
                observation["auroc"]
                - phase65a_anchor_metrics["auroc"]
            ),
            "auroc_gap_vs_development_phase12c": float(
                observation["auroc"]
                - phase65a_phase12c_metrics["auroc"]
            ),
            "causal_exposure_fraction_estimated": False,
        }
    )

phase65a_discrepancy_interpretation = {
    "log_loss_mixture_fraction_not_identifiable": True,
    "reason_log_loss": (
        "unknown_protocol_routing_is_domain_dependent_and_test_difficulty_"
        "is_unobserved"
    ),
    "auroc_linear_interpolation_forbidden": True,
    "reason_auroc": (
        "auroc_contains_cross_exposure_positive_negative_pairs_and_is_not_"
        "a_linear_average_of_component_aurocs"
    ),
    "used_for_fitting_or_advancement": False,
}


# ---------------------------------------------------------------------------
# 3. Optional prototype-frequency diagnostic.
# ---------------------------------------------------------------------------

phase65a_prototype = globals().get("PHASE65A_HEADER_PROTOTYPE_ID_PRIVATE")
phase65a_prototype_report = {
    "available": False,
    "used_for_gate": False,
    "reason": "PHASE65A_HEADER_PROTOTYPE_ID_PRIVATE_not_present",
}
if phase65a_prototype is not None:
    prototype = np.asarray(phase65a_prototype).reshape(-1)
    assert prototype.shape == (phase65a_expected_n,)
    _, counts = np.unique(prototype, return_counts=True)
    singletons = int(np.sum(counts == 1))
    doubletons = int(np.sum(counts == 2))
    good_turing = float(singletons / max(float(prototype.size), 1.0))
    chao1 = float(
        counts.size
        + (singletons * (singletons - 1)) / (2.0 * (doubletons + 1))
    )
    phase65a_prototype_report = {
        "available": True,
        "observed_prototype_count": int(counts.size),
        "singleton_prototype_count": singletons,
        "doubleton_prototype_count": doubletons,
        "iid_exchangeable_good_turing_unseen_mass_estimate": good_turing,
        "chao1_lower_estimate_distinct_prototypes": chao1,
        "used_for_gate": False,
        "not_a_bound_under_cohort_or_domain_shift": True,
    }


# ---------------------------------------------------------------------------
# 4. Eligible development-only fallback comparison.
#    Baseline: Phase12c alone.
#    Candidate: exact Phase43 deployment-matched anchor, with no new adapter.
# ---------------------------------------------------------------------------

phase65a_fallback_pooled = phase65a_compare(
    phase65a_labels, phase65a_phase12c, phase65a_anchor
)

phase65a_fallback_group_rows = []
for group in range(phase65a_expected_groups):
    mask = phase65a_groups == group
    if int(np.sum(mask)) == 0:
        continue
    row = phase65a_compare(
        phase65a_labels[mask],
        phase65a_phase12c[mask],
        phase65a_anchor[mask],
    )
    row["group"] = int(group)
    row["is_major_group"] = bool(group in PHASE65A_CONFIG["major_groups"])
    phase65a_fallback_group_rows.append(row)

phase65a_fallback_domain_rows = []
for domain_index, definition in enumerate(phase65a_domain_definitions):
    mask = phase65a_domain_id == domain_index
    assert int(np.sum(mask)) > 0
    row = phase65a_compare(
        phase65a_labels[mask],
        phase65a_phase12c[mask],
        phase65a_anchor[mask],
    )
    row["domain"] = str(definition["name"])
    row["groups"] = [int(value) for value in definition["groups"]]
    phase65a_fallback_domain_rows.append(row)

phase65a_fallback_original_fold_rows = []
for fold in sorted(np.unique(phase65a_original_fold).tolist()):
    mask = phase65a_original_fold == int(fold)
    row = phase65a_compare(
        phase65a_labels[mask],
        phase65a_phase12c[mask],
        phase65a_anchor[mask],
    )
    row["fold"] = int(fold)
    phase65a_fallback_original_fold_rows.append(row)

phase65a_fallback_repeat_rows = []
for repeat in range(phase65a_repeat_fold.shape[0]):
    for fold in range(3):
        mask = phase65a_repeat_fold[repeat] == fold
        if int(np.sum(mask)) == 0:
            continue
        row = phase65a_compare(
            phase65a_labels[mask],
            phase65a_phase12c[mask],
            phase65a_anchor[mask],
        )
        row["repeat"] = int(repeat)
        row["fold"] = int(fold)
        phase65a_fallback_repeat_rows.append(row)

phase65a_group_regret = float(
    max(
        [0.0]
        + [
            max(0.0, -row["log_loss_gain"])
            for row in phase65a_fallback_group_rows
        ]
    )
)
phase65a_major_group_regret = float(
    max(
        [0.0]
        + [
            max(0.0, -row["log_loss_gain"])
            for row in phase65a_fallback_group_rows
            if row["is_major_group"]
        ]
    )
)
phase65a_domain_regret = float(
    max(
        [0.0]
        + [
            max(0.0, -row["log_loss_gain"])
            for row in phase65a_fallback_domain_rows
        ]
    )
)
phase65a_original_fold_regret = float(
    max(
        [0.0]
        + [
            max(0.0, -row["log_loss_gain"])
            for row in phase65a_fallback_original_fold_rows
        ]
    )
)
phase65a_group_win_fraction = float(
    np.mean(
        [row["log_loss_gain"] > 0.0 for row in phase65a_fallback_group_rows]
    )
)
phase65a_repeat_win_fraction = float(
    np.mean(
        [row["log_loss_gain"] > 0.0 for row in phase65a_fallback_repeat_rows]
    )
)
phase65a_fallback_domain_gain = phase65a_domain_gains(
    phase65a_phase12c, phase65a_anchor
)
phase65a_fallback_bootstrap = phase65a_domain_bootstrap(
    phase65a_fallback_domain_gain
)

phase65a_fallback_gate = PHASE65A_CONFIG["fallback_gate"]
phase65a_fallback_criteria = {
    "pooled_log_loss_gain": bool(
        phase65a_fallback_pooled["log_loss_gain"]
        >= phase65a_fallback_gate["minimum_pooled_log_loss_gain"]
    ),
    "pooled_auroc_gain": bool(
        phase65a_fallback_pooled["auroc_gain"] is not None
        and phase65a_fallback_pooled["auroc_gain"]
        >= phase65a_fallback_gate["minimum_pooled_auroc_gain"]
    ),
    "no_brier_regret": bool(
        phase65a_fallback_pooled["brier_gain"]
        >= -phase65a_fallback_gate["brier_regret_allowed"]
    ),
    "bounded_any_group_regret": bool(
        phase65a_group_regret
        <= phase65a_fallback_gate[
            "maximum_individual_group_log_loss_regret"
        ]
    ),
    "no_major_group_regret": bool(
        phase65a_major_group_regret
        <= phase65a_fallback_gate[
            "maximum_individual_major_group_log_loss_regret"
        ]
    ),
    "bounded_domain_regret": bool(
        phase65a_domain_regret
        <= phase65a_fallback_gate[
            "maximum_individual_domain_log_loss_regret"
        ]
    ),
    "no_original_fold_regret": bool(
        phase65a_original_fold_regret
        <= phase65a_fallback_gate["maximum_original_fold_log_loss_regret"]
    ),
    "group_win_fraction": bool(
        phase65a_group_win_fraction
        >= phase65a_fallback_gate["minimum_improved_group_fraction"]
    ),
    "repeat_partition_win_fraction": bool(
        phase65a_repeat_win_fraction
        >= phase65a_fallback_gate[
            "minimum_repeat_partition_log_loss_win_fraction"
        ]
    ),
    "domain_bootstrap_lower_95": bool(
        phase65a_fallback_bootstrap["lower_95"]
        >= phase65a_fallback_gate[
            "minimum_domain_bootstrap_lower_95_log_loss_gain"
        ]
    ),
}
phase65a_fallback_numeric_advanced = bool(
    all(phase65a_fallback_criteria.values())
)


# ---------------------------------------------------------------------------
# 5. Correct prior-uncertainty diagnostic: additive logit intercept.
#    This section cannot alter deployment without exact Phase56 routed OOF.
# ---------------------------------------------------------------------------

phase65a_positive = phase65a_labels == 1
phase65a_negative = ~phase65a_positive
assert int(np.sum(phase65a_positive)) > 0
assert int(np.sum(phase65a_negative)) > 0
phase65a_anchor_logit = phase65a_logit(phase65a_anchor)


def phase65a_prevalence_weighted_log_loss(probability, prevalence):
    loss = phase65a_case_loss(phase65a_labels, probability)
    return float(
        float(prevalence) * np.mean(loss[phase65a_positive])
        + (1.0 - float(prevalence)) * np.mean(loss[phase65a_negative])
    )


phase65a_prior_config = PHASE65A_CONFIG["prior_uncertainty_diagnostic"]
phase65a_offset_rows = []
for offset in phase65a_prior_config["logit_offset_grid"]:
    probability = phase65a_sigmoid(phase65a_anchor_logit + float(offset))
    metrics = phase65a_metrics(phase65a_labels, probability)
    prevalence_loss = {
        f"{prevalence:.2f}": phase65a_prevalence_weighted_log_loss(
            probability, prevalence
        )
        for prevalence in phase65a_prior_config["prevalence_grid"]
    }
    group_regret = 0.0
    for group in range(phase65a_expected_groups):
        mask = phase65a_groups == group
        if int(np.sum(mask)) == 0:
            continue
        group_regret = max(
            group_regret,
            float(
                np.mean(
                    phase65a_case_loss(
                        phase65a_labels[mask], probability[mask]
                    )
                )
                - np.mean(
                    phase65a_case_loss(
                        phase65a_labels[mask], phase65a_anchor[mask]
                    )
                )
            ),
        )
    auroc_change = float(metrics["auroc"] - phase65a_anchor_metrics["auroc"])
    assert abs(auroc_change) <= 1.0e-12, {
        "message": "A constant logit offset must preserve AUROC exactly.",
        "offset": float(offset),
        "auroc_change": auroc_change,
    }
    phase65a_offset_rows.append(
        {
            "logit_offset": float(offset),
            "in_distribution": metrics,
            "in_distribution_log_loss_regret": float(
                metrics["log_loss"] - phase65a_anchor_metrics["log_loss"]
            ),
            "maximum_group_log_loss_regret": float(group_regret),
            "prevalence_family_log_loss": prevalence_loss,
            "worst_case_log_loss": float(max(prevalence_loss.values())),
        }
    )

phase65a_zero_offset_row = next(
    row
    for row in phase65a_offset_rows
    if abs(row["logit_offset"]) <= 1.0e-12
)
phase65a_minimax_offset_row = min(
    phase65a_offset_rows,
    key=lambda row: (
        row["worst_case_log_loss"],
        abs(row["logit_offset"]),
    ),
)
phase65a_offset_worst_case_gain = float(
    phase65a_zero_offset_row["worst_case_log_loss"]
    - phase65a_minimax_offset_row["worst_case_log_loss"]
)
phase65a_offset_gate = phase65a_prior_config["numeric_support_gate"]
phase65a_offset_numeric_supported = bool(
    abs(phase65a_minimax_offset_row["logit_offset"]) > 1.0e-12
    and phase65a_minimax_offset_row["in_distribution_log_loss_regret"]
    <= phase65a_offset_gate["maximum_in_distribution_log_loss_regret"]
    and phase65a_offset_worst_case_gain
    >= phase65a_offset_gate["minimum_worst_case_log_loss_gain"]
    and phase65a_minimax_offset_row["maximum_group_log_loss_regret"]
    <= phase65a_offset_gate[
        "maximum_individual_group_in_distribution_regret"
    ]
)

# Deliberately frozen to identity. The numerical diagnostic cannot modify all
# Phase56 branches without an exact routed/adapted OOF reconstruction.
phase65a_offset_deployment_advanced = False
phase65a_selected_deployment_logit_offset = 0.0


# ---------------------------------------------------------------------------
# 6. Runtime-parity prerequisite and non-causal optional public diagnostic.
# ---------------------------------------------------------------------------

phase65a_runtime_parity_plan = {
    "required_before_any_archive_change_or_submission_ablation": True,
    "checks": [
        "recompute_development_predictions_inside_the_submission_archive",
        "assert_probability_agreement_with_the_development_path_within_1e-9",
        "verify_resampling_orientation_spacing_and_crop_for_each_known_group",
        "verify_phase12c_full_average_residual_and_adapter_branch_counts",
        "assert_unreadable_case_fallback_count_is_zero_on_local_rehearsal",
        "verify_dtype_logit_blending_clipping_and_csv_column_order",
        "verify_main_py_and_packaged_checkpoint_checksums",
    ],
    "parity_failure_action": (
        "stop_and_repair_packaging_before_any_new_model_or_public_ablation"
    ),
}

phase65a_optional_public_ablation = {
    "recommended_now": False,
    "prerequisite": "complete_local_runtime_parity_first",
    "arms": [
        "phase12c_only_for_every_case",
        "full_three_member_no_sparse_adapter_for_every_case",
    ],
    "causal_identification_claimed": False,
    "may_not_estimate_unknown_protocol_fraction": True,
    "may_not_distinguish_packaging_defect_from_domain_shift_by_itself": True,
    "may_not_select_the_final_private_leaderboard_path": True,
    "leaderboard_feedback_used_for_fitting_or_gate_in_this_cell": False,
}


# ---------------------------------------------------------------------------
# 7. Frozen recommendation and contract.
# ---------------------------------------------------------------------------

phase65a_runtime_parity_required = bool(phase65a_fallback_numeric_advanced)
if phase65a_fallback_numeric_advanced:
    phase65a_status = (
        "fallback_numeric_gate_passed_runtime_parity_required_before_change"
    )
    phase65a_fallback_recommendation = (
        "If and only if local container parity confirms the exact candidate, "
        "unknown or unsupported protocols should use the full Phase43 "
        "three-member deployment-matched anchor with the sparse adapter "
        "disabled, instead of Phase12c alone."
    )
else:
    phase65a_status = (
        "fallback_numeric_gate_failed_retain_phase12c_unknown_fallback"
    )
    phase65a_fallback_recommendation = (
        "Retain Phase12c alone for unknown or unsupported protocols. Do not "
        "search another fallback on these same development labels."
    )

phase65a_contract_core = {
    "schema_version": PHASE65A_CONFIG["schema_version"],
    "status": phase65a_status,
    "source_contract_sha256": str(phase65a_validation["contract_sha256"]),
    "fallback_numeric_gate_passed": phase65a_fallback_numeric_advanced,
    "runtime_parity_required_before_change": phase65a_runtime_parity_required,
    "prior_offset_deployment_advanced": False,
    "selected_deployment_logit_offset": 0.0,
    "public_score_used_for_fitting_or_advancement": False,
    "exposure_fraction_estimated": False,
    "outer_labels_used_for_fitting": False,
    "test_data_read": False,
    "new_model_trained": False,
}
phase65a_contract_sha256 = hashlib.sha256(
    json.dumps(
        phase65a_contract_core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
).hexdigest()

phase65a_contract_path = Path(PHASE65A_CONFIG["contract_file"])
phase65a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase65a_temporary_path = phase65a_contract_path.with_suffix(
    phase65a_contract_path.suffix + ".tmp"
)
phase65a_temporary_path.write_text(
    json.dumps(
        {
            **phase65a_contract_core,
            "contract_sha256": phase65a_contract_sha256,
            "contains_labels": False,
            "contains_probabilities": False,
            "contains_case_indices": False,
            "contains_voxel_data": False,
            "contains_embeddings": False,
        },
        indent=2,
        sort_keys=True,
    )
    + "\n",
    encoding="utf-8",
)
os.replace(phase65a_temporary_path, phase65a_contract_path)


PHASE65A_DEPLOYMENT_EXPOSURE_STATE_PRIVATE = {
    "contract_sha256": phase65a_contract_sha256,
    "status": phase65a_status,
    "fallback_numeric_gate_passed": phase65a_fallback_numeric_advanced,
    "runtime_parity_required_before_change": phase65a_runtime_parity_required,
    "prior_offset_deployment_advanced": False,
    "selected_deployment_logit_offset": 0.0,
    "phase12c_probability": phase65a_phase12c.copy(),
    "anchor_probability": phase65a_anchor.copy(),
}


phase65a_report = {
    "phase": "phase65a_corrected_deployment_fallback_audit",
    "status": phase65a_status,
    "development_reference": {
        "phase43_anchor": phase65a_anchor_metrics,
        "phase12c_unknown_fallback": phase65a_phase12c_metrics,
    },
    "public_deployment_discrepancy": {
        "records": phase65a_public_discrepancy,
        "interpretation": phase65a_discrepancy_interpretation,
        "leader_reference_diagnostic_only": PHASE65A_CONFIG[
            "public_reference_leader"
        ],
    },
    "prototype_frequency_diagnostic": phase65a_prototype_report,
    "fallback_gate": {
        "baseline": "phase12c_alone",
        "candidate": "phase43_full_three_member_no_sparse_adapter",
        "pooled": phase65a_fallback_pooled,
        "groups": phase65a_fallback_group_rows,
        "domains": phase65a_fallback_domain_rows,
        "original_folds": phase65a_fallback_original_fold_rows,
        "repeat_partitions": phase65a_fallback_repeat_rows,
        "repeat_partitions_are_subsets_of_fixed_oof_vectors_not_refits": True,
        "maximum_group_log_loss_regret": phase65a_group_regret,
        "maximum_major_group_log_loss_regret": phase65a_major_group_regret,
        "maximum_domain_log_loss_regret": phase65a_domain_regret,
        "maximum_original_fold_log_loss_regret": (
            phase65a_original_fold_regret
        ),
        "group_log_loss_win_fraction": phase65a_group_win_fraction,
        "repeat_partition_log_loss_win_fraction": (
            phase65a_repeat_win_fraction
        ),
        "domain_bootstrap": phase65a_fallback_bootstrap,
        "thresholds": dict(phase65a_fallback_gate),
        "criteria": phase65a_fallback_criteria,
        "numeric_advanced": phase65a_fallback_numeric_advanced,
        "runtime_parity_required_before_change": (
            phase65a_runtime_parity_required
        ),
    },
    "prior_uncertainty_diagnostic": {
        "correct_operation": "additive_logit_offset_not_temperature_scaling",
        "development_prevalence": float(np.mean(phase65a_labels)),
        "declared_prevalence_family": phase65a_prior_config[
            "prevalence_grid"
        ],
        "baseline_worst_case_log_loss": phase65a_zero_offset_row[
            "worst_case_log_loss"
        ],
        "numeric_minimax_logit_offset": phase65a_minimax_offset_row[
            "logit_offset"
        ],
        "numeric_worst_case_log_loss_gain": phase65a_offset_worst_case_gain,
        "numeric_support_gate": dict(phase65a_offset_gate),
        "numeric_supported": phase65a_offset_numeric_supported,
        "deployment_eligible": False,
        "deployment_advanced": False,
        "selected_deployment_logit_offset": 0.0,
        "reason": phase65a_prior_config["reason"],
    },
    "runtime_parity_plan": phase65a_runtime_parity_plan,
    "optional_public_ablation": phase65a_optional_public_ablation,
    "recommendation": phase65a_fallback_recommendation,
    "interpretation_contract": {
        "public_scores_are_descriptive_only": True,
        "public_scores_do_not_enter_any_gate": True,
        "auroc_exposure_interpolation_removed": True,
        "unknown_protocol_fraction_not_estimated": True,
        "good_turing_is_not_a_domain_shift_bound": True,
        "fallback_change_requires_runtime_parity": True,
        "prior_offset_is_diagnostic_only": True,
        "phase43_phase56_anchor_remains_current_until_all_requirements_pass": True,
    },
    "compliance": {
        "test_data_read": False,
        "training_performed": False,
        "training_nifti_files_read": False,
        "training_voxel_cache_read": False,
        "outer_labels_used_for_fitting": False,
        "case_level_predictions_exported": False,
        "leaderboard_used_for_hyperparameter_or_candidate_selection": False,
        "independent_test_case_inference_preserved": True,
    },
    "synthetic_test_mode": phase65a_is_synthetic,
    "contract_sha256": phase65a_contract_sha256,
    "contract_file": str(phase65a_contract_path),
    "elapsed_seconds": float(time.perf_counter() - phase65a_started),
}

PHASE65A_DEPLOYMENT_EXPOSURE_REPORT_PRIVATE = phase65a_report


# Keep the shared block compact. Full group/domain rows remain in the private
# in-memory report and are not printed or exported case-by-case.
phase65a_sanitized_report = {
    "phase": phase65a_report["phase"],
    "status": phase65a_report["status"],
    "development_reference": phase65a_report["development_reference"],
    "public_deployment_discrepancy": (
        phase65a_report["public_deployment_discrepancy"]
    ),
    "prototype_frequency_diagnostic": phase65a_prototype_report,
    "fallback_gate": {
        "pooled": phase65a_fallback_pooled,
        "maximum_group_log_loss_regret": phase65a_group_regret,
        "maximum_major_group_log_loss_regret": phase65a_major_group_regret,
        "maximum_domain_log_loss_regret": phase65a_domain_regret,
        "maximum_original_fold_log_loss_regret": (
            phase65a_original_fold_regret
        ),
        "group_log_loss_win_fraction": phase65a_group_win_fraction,
        "repeat_partition_log_loss_win_fraction": (
            phase65a_repeat_win_fraction
        ),
        "domain_bootstrap": phase65a_fallback_bootstrap,
        "thresholds": dict(phase65a_fallback_gate),
        "criteria": phase65a_fallback_criteria,
        "numeric_advanced": phase65a_fallback_numeric_advanced,
        "runtime_parity_required_before_change": (
            phase65a_runtime_parity_required
        ),
    },
    "prior_uncertainty_diagnostic": (
        phase65a_report["prior_uncertainty_diagnostic"]
    ),
    "runtime_parity_required": phase65a_runtime_parity_plan[
        "required_before_any_archive_change_or_submission_ablation"
    ],
    "recommendation": phase65a_fallback_recommendation,
    "interpretation_contract": phase65a_report["interpretation_contract"],
    "compliance": phase65a_report["compliance"],
    "contract_sha256": phase65a_contract_sha256,
    "contract_file": str(phase65a_contract_path),
    "elapsed_seconds": phase65a_report["elapsed_seconds"],
}

print("BEGIN SANITIZED_PHASE65A_CORRECTED_DEPLOYMENT_FALLBACK_AUDIT")
print(json.dumps(phase65a_sanitized_report, indent=2, sort_keys=False))
print("END SANITIZED_PHASE65A_CORRECTED_DEPLOYMENT_FALLBACK_AUDIT")
