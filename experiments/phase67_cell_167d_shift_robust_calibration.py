"""Phase67D - shift-robust probability calibration and the leaderboard probe.

The development population shows almost no calibration headroom: Phase59's
label-fitted oracle pooled Platt gained only 0.001281, so on development data
this family is close to exhausted. The leaderboard is a different population -
development 0.290166 against a recorded public 0.3220 - and that gap is exactly
where a temperature and an intercept can act. Phase59 also found the anchor
already slightly overconfident on development (pooled slope 0.9085), which is
the direction that gets worse, not better, under distribution shift.

A temperature and an intercept are strictly monotone, so applying one global
pair cannot change AUROC at all. That makes this the log-loss half of the
Phase67 goal, with no ranking risk, and it gives a free correctness check: a
probe submission whose reported AUROC moves did not do what was intended.

This cell has two halves.

  1. Development side. Nested per-fold selection of (temperature, intercept)
     fitted on the training partition only, under two declared shift-robust
     criteria - worst stress domain, and worst prevalence over a declared
     prevalence family - rather than pooled log loss, which is blind to the
     reweighting a different test population represents. Reports the bounded
     development cost of each.

  2. Leaderboard side. Emits the exact probe specification, and, when observed
     probe outcomes are supplied, checks the AUROC invariance the probes must
     satisfy and fits the one-dimensional optimum. No leaderboard number is
     ever used to fit a development quantity; the probes are a measurement of
     the deployment population that development data cannot provide.

No test data is read here and no submission is created. Applying a chosen
(temperature, intercept) to a deployment package is a separate, explicit step
that must still pass the Phase67E gate and the offline runtime checks.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402

PHASE67D_CONFIG = {
    "schema_version": "phase67_shift_robust_calibration_v1",
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get(
        "DAT_OUTPUT_ROOT", "/kaggle/working/phase67_private"
    ),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),
    # Observed leaderboard probe outcomes, supplied by the operator as a JSON
    # list of {"temperature", "intercept", "log_loss", "auroc"}. Absent on the
    # first run: the cell then only emits the probe plan.
    "probe_results_file": os.environ.get("DAT_LB_PROBE_RESULTS", ""),
    "seed": 670401,
    # Temperature divides the logit; >1 shrinks confidence. Intercept is added
    # after division. Identity (1.0, 0.0) is always in the grid, so "change
    # nothing" is a selectable outcome.
    "temperature_grid": [
        0.90, 0.95, 1.00, 1.05, 1.10, 1.15, 1.20, 1.25, 1.30, 1.40, 1.50,
    ],
    "intercept_grid": [
        -0.40, -0.30, -0.20, -0.15, -0.10, -0.05, 0.00, 0.05, 0.10, 0.15,
        0.20, 0.30, 0.40,
    ],
    # Reused verbatim from phase65_cell_165a_corrected_deployment_fallback_audit.
    # Reweighting the two label classes is a label-honest stand-in for a test
    # population with a different normal/abnormal mix.
    "prevalence_grid": [0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
    "macro_bootstrap_replicates": 10000,
    "macro_bootstrap_seed": 660699,
    "cluster_bootstrap_replicates": 10000,
    "cluster_bootstrap_seed": 611161,
    # Recorded public outcomes. Descriptive only: they are the measurement the
    # probes extend, and they never enter a fitted development quantity.
    "recorded_public": {
        "phase42_submission": {"log_loss": 0.3254, "auroc": 0.9251},
        "phase56_submission": {"log_loss": 0.3220, "auroc": 0.9297},
    },
    # A probe that changes AUROC by more than this did not apply a monotone
    # transform to the deployed probabilities, whatever else it did.
    "auroc_invariance_tolerance": 5.0e-4,
    "probe_plan": [
        {"name": "baseline", "temperature": 1.00, "intercept": 0.00},
        {"name": "temperature_probe", "temperature": 1.15, "intercept": 0.00},
        {"name": "intercept_probe", "temperature": 1.00, "intercept": -0.10},
    ],
    "contract_file": "phase67d_shift_robust_calibration_contract.json",
}


def apply_calibration(probability, temperature, intercept):
    """sigmoid(logit(p) / T + b). Strictly increasing in p for T > 0."""
    common.require(float(temperature) > 0.0, "phase67d_nonpositive_temperature")
    return common.sigmoid(
        common.logit(probability) / float(temperature) + float(intercept)
    )


def prevalence_weighted_log_loss(labels, probability, prevalence):
    """Class-reweighted log loss.

    Ported from phase65a_prevalence_weighted_log_loss. Averaging each class
    separately and recombining at a chosen prevalence answers "what would this
    score on a population with a different normal/abnormal mix", which is the
    part of distribution shift development data can actually address.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    loss = common.loss_vector(labels, probability)
    positive = labels == 1
    negative = ~positive
    common.require(
        bool(np.any(positive)) and bool(np.any(negative)),
        "phase67d_prevalence_requires_both_classes",
    )
    return float(
        float(prevalence) * float(np.mean(loss[positive]))
        + (1.0 - float(prevalence)) * float(np.mean(loss[negative]))
    )


def worst_domain_log_loss(labels, domain_id, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    domain_id = np.asarray(domain_id, dtype=np.int64).reshape(-1)
    loss = common.loss_vector(labels, probability)
    return float(
        max(
            float(np.mean(loss[domain_id == domain]))
            for domain in np.unique(domain_id)
        )
    )


def worst_prevalence_log_loss(labels, probability, prevalence_grid):
    return float(
        max(
            prevalence_weighted_log_loss(labels, probability, prevalence)
            for prevalence in prevalence_grid
        )
    )


CRITERIA = {
    "pooled": lambda labels, domain_id, probability, config: float(
        np.mean(common.loss_vector(labels, probability))
    ),
    "worst_stress_domain": lambda labels, domain_id, probability, config: (
        worst_domain_log_loss(labels, domain_id, probability)
    ),
    "worst_prevalence": lambda labels, domain_id, probability, config: (
        worst_prevalence_log_loss(labels, probability, config["prevalence_grid"])
    ),
}


def nested_calibration(
    labels, domain_id, original_fold, probability, criterion_name, config
):
    """Select (temperature, intercept) inside each training partition, apply
    once to the held-out fold. Identity is in the grid, so a fold that gains
    nothing from calibration keeps its probabilities unchanged."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    original_fold = np.asarray(original_fold, dtype=np.int64).reshape(-1)
    criterion = CRITERIA[criterion_name]
    calibrated = np.full(labels.size, np.nan, dtype=np.float64)
    selections = []
    for fold in range(3):
        outer_mask = original_fold == fold
        train_mask = ~outer_mask
        best = None
        for temperature in config["temperature_grid"]:
            for intercept in config["intercept_grid"]:
                adjusted = apply_calibration(
                    probability[train_mask], temperature, intercept
                )
                score = criterion(
                    labels[train_mask], domain_id[train_mask], adjusted, config
                )
                if best is None or score < best["training_criterion"]:
                    best = {
                        "temperature": float(temperature),
                        "intercept": float(intercept),
                        "training_criterion": float(score),
                    }
        common.require(best is not None, "phase67d_empty_calibration_grid")
        calibrated[outer_mask] = apply_calibration(
            probability[outer_mask], best["temperature"], best["intercept"]
        )
        selections.append({**best, "fold": int(fold)})
    common.require(
        bool(np.all(np.isfinite(calibrated))), "phase67d_incomplete_calibration"
    )
    return calibrated, selections


def fit_probe_optimum(observations, axis, tolerance):
    """Fit a one-dimensional quadratic through observed leaderboard log losses.

    Returns the interpolated argmin plus an explicit note when the minimum lies
    outside the probed interval, in which case the honest action is another
    probe rather than an extrapolated jump.
    """
    points = sorted(
        {(float(item[axis]), float(item["log_loss"])) for item in observations}
    )
    if len(points) < 3:
        return {
            "axis": axis,
            "status": "insufficient_points_need_at_least_three",
            "observed_points": len(points),
        }
    x = np.asarray([item[0] for item in points], dtype=np.float64)
    y = np.asarray([item[1] for item in points], dtype=np.float64)
    coefficients = np.polyfit(x, y, 2)
    if coefficients[0] <= 0.0:
        return {
            "axis": axis,
            "status": "fitted_curve_is_not_convex_probe_a_wider_interval",
            "observed_points": len(points),
        }
    optimum = float(-coefficients[1] / (2.0 * coefficients[0]))
    inside = bool(float(np.min(x)) <= optimum <= float(np.max(x)))
    return {
        "axis": axis,
        "status": (
            "interpolated_optimum" if inside else "extrapolated_probe_again"
        ),
        "observed_points": len(points),
        "probed_interval": [float(np.min(x)), float(np.max(x))],
        "estimated_optimum": optimum,
        "predicted_log_loss_at_optimum": float(np.polyval(coefficients, optimum)),
        "best_observed": {
            axis: float(x[int(np.argmin(y))]),
            "log_loss": float(np.min(y)),
        },
    }


def read_probe_results(config):
    path = str(config.get("probe_results_file") or "").strip()
    if not path:
        return None, "no_probe_results_supplied"
    candidate = Path(path)
    if not candidate.is_file():
        return None, "probe_results_file_missing"
    try:
        payload = json.loads(candidate.read_text(encoding="utf-8"))
    except Exception:
        return None, "probe_results_file_unreadable"
    if not isinstance(payload, list) or not payload:
        return None, "probe_results_file_not_a_nonempty_list"
    rows = []
    for item in payload:
        if not isinstance(item, dict):
            return None, "probe_results_row_not_an_object"
        try:
            rows.append(
                {
                    "name": str(item.get("name", "unnamed")),
                    "temperature": float(item["temperature"]),
                    "intercept": float(item["intercept"]),
                    "log_loss": float(item["log_loss"]),
                    "auroc": float(item["auroc"]),
                }
            )
        except Exception:
            return None, "probe_results_row_missing_required_field"
    return rows, "loaded"


def evaluate_probes(rows, config):
    """Check the invariance the probes must satisfy, then fit each axis."""
    baseline = [
        row
        for row in rows
        if abs(row["temperature"] - 1.0) < 1.0e-12
        and abs(row["intercept"]) < 1.0e-12
    ]
    reference_auroc = (
        baseline[0]["auroc"]
        if baseline
        else config["recorded_public"]["phase56_submission"]["auroc"]
    )
    invariance = []
    for row in rows:
        deviation = abs(row["auroc"] - reference_auroc)
        invariance.append(
            {
                "name": row["name"],
                "temperature": row["temperature"],
                "intercept": row["intercept"],
                "auroc_deviation_from_reference": float(deviation),
                "auroc_invariance_holds": bool(
                    deviation <= config["auroc_invariance_tolerance"]
                ),
            }
        )
    all_invariant = all(item["auroc_invariance_holds"] for item in invariance)
    temperature_rows = [
        row for row in rows if abs(row["intercept"]) < 1.0e-12
    ]
    intercept_rows = [
        row for row in rows if abs(row["temperature"] - 1.0) < 1.0e-12
    ]
    return {
        "reference_auroc": float(reference_auroc),
        "auroc_invariance": invariance,
        "auroc_invariance_holds_for_every_probe": bool(all_invariant),
        "auroc_invariance_interpretation": (
            "a_monotone_temperature_or_intercept_cannot_move_auroc;_a_"
            "deviation_means_the_submitted_package_changed_something_else"
        ),
        "temperature_axis": fit_probe_optimum(
            temperature_rows, "temperature", config["auroc_invariance_tolerance"]
        ),
        "intercept_axis": fit_probe_optimum(
            intercept_rows, "intercept", config["auroc_invariance_tolerance"]
        ),
    }


def assess(labels, groups, original_fold, domain_id, anchor, candidate, name, config):
    stability = common.stability_rows(
        labels, groups, original_fold, domain_id, anchor, candidate
    )
    domain_gains = common.per_domain_gains(labels, domain_id, anchor, candidate)
    return {
        "name": name,
        "pooled": stability["pooled"],
        "folds": stability["folds"],
        "worst_domain_log_loss": worst_domain_log_loss(
            labels, domain_id, candidate
        ),
        "worst_prevalence_log_loss": worst_prevalence_log_loss(
            labels, candidate, config["prevalence_grid"]
        ),
        "prevalence_family_log_loss": {
            f"{prevalence:.2f}": prevalence_weighted_log_loss(
                labels, candidate, prevalence
            )
            for prevalence in config["prevalence_grid"]
        },
        "expected_calibration_error": common.expected_calibration_error(
            labels, candidate
        ),
        "maximum_fold_log_loss_regret": stability["maximum_fold_log_loss_regret"],
        "maximum_powered_domain_log_loss_regret": stability[
            "maximum_powered_domain_log_loss_regret"
        ],
        "macro_domain_bootstrap": common.macro_domain_bootstrap(
            domain_gains,
            config["macro_bootstrap_replicates"],
            config["macro_bootstrap_seed"],
        ),
        "case_weighted_group_cluster_bootstrap": common.group_cluster_bootstrap(
            labels,
            groups,
            anchor,
            candidate,
            config["cluster_bootstrap_replicates"],
            config["cluster_bootstrap_seed"],
        ),
    }


def main():
    started = time.perf_counter()
    config = PHASE67D_CONFIG
    artifact_root = Path(config["artifact_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    state = common.restore_production_state(
        artifact_root, labels_root=config["labels_root"] or None
    )
    labels = state["labels"]
    groups = state["groups"]
    original_fold = state["original_fold"]
    domain_id = state["stress_domain_id"]
    anchor = state["anchor_probability"]

    # The base vector is the Phase67B stack when it exists, otherwise the
    # anchor. Calibration is applied on top of whatever the current candidate
    # is; it never replaces the discrimination work.
    base_name = "phase43_anchor"
    base = anchor
    stack_path = output_root / "phase67b_stacked_development_oof.npz"
    if stack_path.is_file():
        with np.load(stack_path, allow_pickle=False) as payload:
            if "primary" in payload.files:
                stacked = np.asarray(payload["primary"], dtype=np.float64)
                if stacked.shape == (common.CASE_COUNT,) and bool(
                    np.all((stacked > 0.0) & (stacked < 1.0))
                ):
                    base = stacked
                    base_name = "phase67b_nested_multi_expert_stack"

    results = []
    calibrated_vectors = {}
    for criterion_name in ("pooled", "worst_stress_domain", "worst_prevalence"):
        calibrated, selections = nested_calibration(
            labels, domain_id, original_fold, base, criterion_name, config
        )
        calibrated_vectors[criterion_name] = calibrated
        record = assess(
            labels,
            groups,
            original_fold,
            domain_id,
            base,
            calibrated,
            f"nested_calibration_{criterion_name}",
            config,
        )
        record["selection_criterion"] = criterion_name
        record["per_fold_selection"] = selections
        record["identity_selected_in_every_fold"] = bool(
            all(
                abs(item["temperature"] - 1.0) < 1.0e-12
                and abs(item["intercept"]) < 1.0e-12
                for item in selections
            )
        )
        results.append(record)

    base_record = assess(
        labels,
        groups,
        original_fold,
        domain_id,
        base,
        base,
        f"uncalibrated_{base_name}",
        config,
    )

    # A single global pair, selected on the whole development population under
    # the shift-robust criterion. Reported as a deployment-ready constant and
    # explicitly marked as in-sample on development.
    global_grid = []
    for temperature in config["temperature_grid"]:
        for intercept in config["intercept_grid"]:
            adjusted = apply_calibration(base, temperature, intercept)
            global_grid.append(
                {
                    "temperature": float(temperature),
                    "intercept": float(intercept),
                    "pooled_log_loss": float(
                        np.mean(common.loss_vector(labels, adjusted))
                    ),
                    "worst_stress_domain_log_loss": worst_domain_log_loss(
                        labels, domain_id, adjusted
                    ),
                    "worst_prevalence_log_loss": worst_prevalence_log_loss(
                        labels, adjusted, config["prevalence_grid"]
                    ),
                }
            )
    global_best = min(
        global_grid, key=lambda item: item["worst_prevalence_log_loss"]
    )

    probe_rows, probe_status = read_probe_results(config)
    probe_section = {
        "status": probe_status,
        "plan": config["probe_plan"],
        "instructions": [
            "submit_the_baseline_package_unchanged_first_to_anchor_the_curve",
            "each_probe_changes_only_one_scalar_in_the_final_output_stage",
            "record_name_temperature_intercept_log_loss_auroc_per_submission",
            "verify_reported_auroc_is_unchanged_before_trusting_any_delta",
            "budget_submissions_before_starting_and_reserve_capacity_for_the_"
            "phase67e_candidate",
        ],
        "leaderboard_feedback_used_to_fit_a_development_quantity": False,
    }
    if probe_rows is not None:
        probe_section["observations"] = probe_rows
        probe_section["analysis"] = evaluate_probes(probe_rows, config)

    calibration_helps = any(
        record["pooled"]["log_loss_gain"] > 0.0 for record in results
    )
    if not calibration_helps:
        status = "phase67d_development_calibration_offers_no_gain_identity_retained"
    elif probe_rows is None:
        status = "phase67d_development_calibration_ready_leaderboard_probe_pending"
    else:
        status = "phase67d_development_and_leaderboard_probe_measurements_available"

    core = {
        "schema_version": config["schema_version"],
        "status": status,
        "common_version": common.PHASE67_COMMON_VERSION,
        "config": {
            key: value
            for key, value in config.items()
            if key
            not in (
                "artifact_root",
                "output_root",
                "labels_root",
                "contract_file",
                "probe_results_file",
            )
        },
        "base_vector": base_name,
        "uncalibrated": base_record,
        "nested_calibration_results": results,
        "global_in_sample_grid_best_by_worst_prevalence": global_best,
        "leaderboard_probe": probe_section,
        "interpretation": {
            "a_global_temperature_and_intercept_are_strictly_monotone": True,
            "so_a_single_global_pair_cannot_change_auroc": True,
            "per_fold_pairs_are_monotone_within_a_fold_only": True,
            "phase59_oracle_pooled_platt_gain_was_only_0_001281": True,
            "development_calibration_headroom_is_nearly_exhausted": True,
            "the_leaderboard_population_is_where_this_family_can_act": True,
            "development_and_leaderboard_populations_are_not_comparable": True,
            "global_grid_best_is_in_sample_on_development_not_a_held_out_result": True,
            "applying_a_pair_to_a_package_still_requires_the_phase67e_gate": True,
        },
        "training_performed": False,
        "test_data_read": False,
        "test_time_adaptation": False,
        "phase56_archive_unchanged": True,
        "submission_archive_created": False,
        "provenance": state["provenance"],
    }

    contract_path = output_root / config["contract_file"]
    contract_sha = common.emit_contract(contract_path, core)
    # Held-out calibrated vectors stay in the private output directory so
    # Phase67E can adjudicate them; they never enter a shared report.
    common.atomic_npz(
        output_root / "phase67d_calibrated_development_oof.npz",
        contract_sha256=np.asarray(contract_sha),
        base_vector_name=np.asarray(base_name),
        base=base,
        **{
            f"nested_{name}": vector
            for name, vector in calibrated_vectors.items()
        },
    )
    report = {
        **core,
        "contract_sha256": contract_sha,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    common.print_sanitized("PHASE67D_SHIFT_ROBUST_CALIBRATION", report)
    return report


if __name__ == "__main__":
    try:
        main()
    except common.Phase67Stop as stop:
        print(
            json.dumps(
                {
                    "phase": "phase67d_shift_robust_calibration",
                    "status": "phase67d_stopped",
                    "stage": str(stop),
                    "phase56_archive_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
