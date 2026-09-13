"""Phase67B - nested group-held multi-expert stack.

Trains nothing. Every earlier phase compared exactly one new expert against the
anchor at one predeclared alpha; three separate families then produced real
repeated gains near +0.004 log loss and were each rejected by a group
resampling bootstrap that 15 acquisition groups cannot power. This cell tests
the combination those phases never tried, and it measures how much of any
resulting gain its own selection procedure manufactures from noise.

What it does:

  1. Assembles every available 1,362-length development OOF vector and screens
     their complementarity (pairwise logit correlation, per-domain gain
     profiles, single-expert blend screen). That screen is the declared go/no-go
     for the GPU work in Phase67C.
  2. Fits a deliberately small constrained stack in logit space, expressed as a
     bounded departure from the anchor so that anchor-only is the exact null.
  3. Evaluates it under two group-held protocols and reports both.
  4. Runs a within-group label-permutation null through the identical procedure
     to bound the selection advantage.

Honest limits, restated in the emitted contract rather than only here:

  - The historical OOF vectors are out-of-fold with respect to the ORIGINAL
    three whole-group folds. Fitting meta-weights on folds {1,2} therefore uses
    features whose base models saw fold-0 cases. This is the same limitation
    Phase65D recorded and it cannot be removed without full nested base refits.
    The anchor comparator carries the identical original-three-fold status, so
    the comparison is apples-to-apples; the extra freedom is only the stack
    parameters, and the permutation null is what bounds their contribution.
  - Repeated partitions reuse the same patients. They are a sensitivity
    diagnostic, never independent confirmation.
  - A development gain is not a leaderboard result and not a deployment
    decision. Deployment eligibility is tracked per expert and any stack that
    uses an unverified vector is reported as diagnostic only.

No test data is read. No case-level value, UID or path is exported.
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

PHASE67B_CONFIG = {
    "schema_version": "phase67_nested_multi_expert_stack_v1",
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get(
        "DAT_OUTPUT_ROOT", "/kaggle/working/phase67_private"
    ),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),
    "seed": 670201,
    # Stack geometry. anchor_weight_floor caps how far the stack may depart
    # from the anchor: sum of expert weights <= 1 - floor.
    "anchor_weight_floor": 0.55,
    # Declared hyperparameter grid. Every combination is selected on inner
    # group-held data only. The size of this grid is exactly what the
    # permutation null measures, so it is fixed before any outer number is read.
    #
    # The anchor scale must be able to reach 0. If its floor is bounded away
    # from zero, the recalibration baseline cannot fully shrink an
    # uninformative anchor, and the stack then uses expert weights to supply
    # the missing shrinkage -- which the permutation null would misread as
    # manufactured expert signal. Synthetic checking showed exactly that: a
    # grid floored at 0.85 reported ~0.045 of spurious expert gain on shuffled
    # labels. Reaching 0 makes the baseline able to collapse to the prevalence
    # constant, which is the correct optimum under shuffled labels.
    "anchor_scale_grid": [0.00, 0.25, 0.50, 0.70, 0.85, 0.95, 1.00, 1.05],
    "support_kappa_grid": [0.0, 20.0, 40.0, 80.0],
    "ridge_grid": [0.03, 0.30, 3.00],
    "robust_lambda": 1.0,
    # Accelerated projected-gradient solver.
    "maximum_iterations": 400,
    "convergence_tolerance": 1.0e-8,
    # Single-expert screen, reported for comparability with Phase61A.
    "screen_alpha_grid": [0.10, 0.15, 0.20, 0.30, 0.40],
    "complementarity_gate": {
        "maximum_pairwise_anchor_correlation": 0.985,
        "minimum_promising_expert_count": 2,
    },
    "permutation_replicates": 200,
    "permutation_seed": 670203,
    "macro_bootstrap_replicates": 10000,
    "macro_bootstrap_seed": 660699,
    "cluster_bootstrap_replicates": 10000,
    "cluster_bootstrap_seed": 611161,
    # Declared Phase67 targets, from the quantified gate analysis: Phase65D had
    # a macro-domain mean gain of ~0.0034 against an across-domain standard
    # deviation of ~0.0079, so it needed >0.0047 to clear its own bound.
    "target": {
        "minimum_macro_domain_mean_gain": 0.006,
        "maximum_across_domain_standard_deviation": 0.0055,
        "minimum_pooled_auroc_gain": 0.0025,
        "maximum_permutation_null_fraction": 0.25,
    },
    "contract_file": "phase67b_nested_multi_expert_stack_contract.json",
}

# Expert catalogue. "deployment_eligible" means only that the vector has a
# previously reconstructed deployment-matched provenance; it does not mean a
# runnable offline inference path has been demonstrated for Phase67.
#
# For experts whose OOF lives in a phase-specific private output directory,
# consolidate the vector once to
# <artifact_root>/phase67_expert_<name>_float64.npy and it will be picked up.
EXPERT_CATALOGUE = [
    {
        "name": "phase12c",
        "relative_paths": ["phase32_phase12c_oof_float64.npy"],
        "deployment_eligible": True,
        "provenance": "verified_deployment_matched_oof",
    },
    {
        "name": "phase33_component",
        "relative_paths": [
            "phase33_private_checkpoint/phase33_component_oof_float64.npy",
            "phase33_component_oof_float64.npy",
        ],
        "deployment_eligible": True,
        "provenance": "verified_component_oof_reconstructed_in_phase57_restore",
    },
    {
        "name": "phase39",
        "relative_paths": [
            "phase39_private_checkpoint/phase39_oof_float64.npy",
            "phase39_oof_float64.npy",
        ],
        "deployment_eligible": True,
        "provenance": "verified_deployment_matched_oof",
    },
    {
        "name": "phase31_classical",
        "relative_paths": [
            "phase31_classical_oof_private.npy",
            "phase67_expert_phase31_classical_float64.npy",
        ],
        "deployment_eligible": False,
        "provenance": "unverified_requires_phase31_selection_contract_review",
    },
    {
        "name": "phase62_anatomy",
        "relative_paths": [
            "phase67_expert_phase62_anatomy_float64.npy",
            "phase62b_multitemplate_anatomy_expert_oof_float64.npy",
        ],
        "deployment_eligible": False,
        "provenance": "requires_fold_local_refit_before_deployment",
    },
    {
        "name": "phase65c_domain_invariant",
        "relative_paths": [
            "phase67_expert_phase65c_domain_invariant_float64.npy",
            "phase65c_domain_invariant_expert_oof_float64.npy",
        ],
        "deployment_eligible": False,
        "provenance": "refit_as_multiseed_ensemble_in_phase67c",
    },
    {
        "name": "phase65d_averaged",
        "relative_paths": [
            "phase67_expert_phase65d_averaged_float64.npy",
            "phase65d_averaged_expert_oof_float64.npy",
        ],
        "deployment_eligible": False,
        "provenance": "repeated_partition_average_reuses_original_anchor_oof",
    },
    {
        "name": "phase66f_dinov2",
        "relative_paths": [
            "phase67_expert_phase66f_dinov2_float64.npy",
            "phase66f_fresh_dualgpu_private/phase66f_two_seed_oof_float64.npy",
        ],
        "deployment_eligible": False,
        "provenance": "rejected_candidate_retained_for_complementarity_test",
    },
]


# ---------------------------------------------------------------------------
# Constrained stack: z = a*z_anchor + sum_k u_k * d_k + b
#
# with d_k(i) = s(group_i) * (z_k(i) - z_anchor(i)), u_k >= 0 and
# sum_k u_k <= (1 - anchor_weight_floor) * a.
#
# Writing the stack as a bounded departure FROM the anchor makes anchor-only
# the exact null (u = 0, a = 1, b = 0) and turns the anchor floor into a single
# linear inequality, so the whole program stays convex.
# ---------------------------------------------------------------------------

def project_onto_simplex_slack(value, ceiling):
    """Euclidean projection onto {u >= 0, sum(u) <= ceiling}."""
    value = np.asarray(value, dtype=np.float64)
    clipped = np.maximum(value, 0.0)
    if ceiling <= 0.0:
        return np.zeros_like(clipped)
    if float(np.sum(clipped)) <= ceiling:
        return clipped
    # Projection onto the scaled simplex sum(u) == ceiling.
    ordered = np.sort(value)[::-1]
    cumulative = np.cumsum(ordered) - ceiling
    index = np.arange(1, value.size + 1, dtype=np.float64)
    condition = ordered - cumulative / index > 0
    rho = int(np.flatnonzero(condition)[-1]) + 1
    theta = cumulative[rho - 1] / float(rho)
    return np.maximum(value - theta, 0.0)


def support_multiplier(groups, group_sizes, kappa):
    """Label-free shrinkage of the expert contribution.

    Under whole-group holdout a case's own group is absent from the training
    partition by construction, so an own-group training count is always zero and
    cannot serve as support. The declared proxy is instead how common the
    protocol is in the development population: s(g) = n(g) / (n(g) + kappa).
    kappa = 0 disables shrinkage.

    This directly targets the failure that rejected Phase65C and Phase65D: at
    kappa = 40 the two harmed domains are damped to 0.44 (group12, n=32) and
    0.78 (group2, n=145) while group1 (n=456) keeps 0.92. It is a fixed
    per-group constant, so it is computable for a test case from the router
    assignment alone and adds no inference-time fitting.
    """
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    if float(kappa) <= 0.0:
        return np.ones(groups.size, dtype=np.float64)
    sizes = np.asarray(group_sizes, dtype=np.float64)
    per_group = sizes / (sizes + float(kappa))
    return per_group[groups]


def fit_stack(
    design_anchor,
    design_expert,
    labels,
    anchor_scale,
    ridge,
    config,
    warm_start=None,
    allow_experts=True,
):
    """Ridge-penalized logistic fit of (u, b) at a fixed anchor scale.

    Convex with one linear inequality, solved by accelerated projected gradient.
    The step is 1/L with L the exact Lipschitz constant of the smooth part,
    0.25 * lambda_max(X'X / n) + ridge, computed from the small augmented Gram
    matrix. Using the maximum row norm instead would understate the admissible
    step by about two orders of magnitude at logit scale and make the nested
    search unusably slow. Deterministic; warm_start changes speed only.

    allow_experts=False forces every expert weight to zero, which reduces the
    model to a nested anchor recalibration (scale and intercept only). That is
    the baseline the expert contribution is measured against.
    """
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    offset = float(anchor_scale) * np.asarray(design_anchor, dtype=np.float64)
    features = np.asarray(design_expert, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    n, k = features.shape
    ceiling = (
        (1.0 - float(config["anchor_weight_floor"])) * float(anchor_scale)
        if allow_experts
        else 0.0
    )
    # Last column is the unconstrained intercept.
    design = np.column_stack([features, np.ones(n, dtype=np.float64)])
    gram = design.T @ design / float(n)
    lipschitz = 0.25 * float(np.linalg.eigvalsh(gram)[-1]) + float(ridge)
    step = 1.0 / max(lipschitz, 1.0e-12)

    theta = np.zeros(k + 1, dtype=np.float64)
    if warm_start is not None:
        candidate = np.asarray(warm_start, dtype=np.float64).reshape(-1)
        if candidate.size == k + 1 and bool(np.all(np.isfinite(candidate))):
            theta = candidate.copy()
    theta[:k] = project_onto_simplex_slack(theta[:k], ceiling)
    momentum = theta.copy()
    weight = 1.0
    tolerance = float(config["convergence_tolerance"])
    for _ in range(int(config["maximum_iterations"])):
        probability = common.sigmoid(offset + design @ momentum)
        gradient = design.T @ (probability - labels) / float(n)
        gradient[:k] += float(ridge) * momentum[:k]
        proposal = momentum - step * gradient
        updated = np.empty_like(proposal)
        updated[:k] = project_onto_simplex_slack(proposal[:k], ceiling)
        updated[k] = proposal[k]
        next_weight = 0.5 * (1.0 + np.sqrt(1.0 + 4.0 * weight * weight))
        momentum = updated + ((weight - 1.0) / next_weight) * (updated - theta)
        momentum[:k] = project_onto_simplex_slack(momentum[:k], ceiling)
        delta = float(np.max(np.abs(updated - theta)))
        theta = updated
        weight = next_weight
        if delta <= tolerance:
            break
    return theta[:k].copy(), float(theta[k])


def apply_stack(design_anchor, design_expert, anchor_scale, weights, intercept):
    features = np.asarray(design_expert, dtype=np.float64)
    if features.ndim == 1:
        features = features[:, None]
    return common.sigmoid(
        float(anchor_scale) * np.asarray(design_anchor, dtype=np.float64)
        + features @ np.asarray(weights, dtype=np.float64)
        + float(intercept)
    )


def robust_criterion(labels, domain_id, anchor, candidate, robust_lambda):
    """Selection objective: mean per-domain gain penalized by its own spread.

    This is the criterion the promotion bound actually measures, so it is what
    inner selection optimizes -- rather than pooled log loss, which is blind to
    the across-domain variance that rejected Phase65C and Phase65D.
    """
    gains = common.per_domain_gains(labels, domain_id, anchor, candidate)
    if gains.size < 2:
        return float(np.mean(gains))
    spread = float(np.std(gains, ddof=1)) / float(np.sqrt(gains.size))
    return float(np.mean(gains) - float(robust_lambda) * spread)


def hyperparameter_grid(config):
    grid = []
    for kappa in config["support_kappa_grid"]:
        for anchor_scale in config["anchor_scale_grid"]:
            for ridge in config["ridge_grid"]:
                grid.append(
                    {
                        "support_kappa": float(kappa),
                        "anchor_scale": float(anchor_scale),
                        "ridge": float(ridge),
                    }
                )
    return grid


def stacked_predictions(
    labels,
    groups,
    group_sizes,
    domain_id,
    anchor_logit,
    expert_logit,
    outer_fold,
    outer_values,
    config,
    grid,
    expected_domains=None,
    allow_experts=True,
):
    """One complete nested pass.

    For every outer unit: select hyperparameters by leave-one-training-group-out
    inside the training partition, refit on the whole training partition, then
    predict the held-out unit exactly once. Returns the pooled prediction vector
    and the per-outer-unit selection record.

    The kappa loop is outermost so each support-shrunk departure matrix is built
    once per kappa rather than once per grid combination, and the solver is
    warm-started across the (anchor_scale, ridge) sub-grid.

    allow_experts=False runs the identical nesting with every expert weight
    pinned to zero, producing the nested anchor-recalibration baseline. The
    support kappa and the ridge are inert in that mode, so the grid collapses to
    the anchor-scale axis alone.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    outer_fold = np.asarray(outer_fold).reshape(-1)
    domain_id = np.asarray(domain_id, dtype=np.int64).reshape(-1)
    anchor_probability = common.sigmoid(anchor_logit)

    if allow_experts:
        kappa_values = sorted({float(item["support_kappa"]) for item in grid})
        sub_grid = sorted(
            {(float(item["anchor_scale"]), float(item["ridge"])) for item in grid}
        )
    else:
        kappa_values = [0.0]
        first_ridge = min(float(item["ridge"]) for item in grid)
        sub_grid = sorted(
            {(float(item["anchor_scale"]), first_ridge) for item in grid}
        )
    departures = {}
    for kappa in kappa_values:
        support = support_multiplier(groups, group_sizes, kappa)
        departures[kappa] = support[:, None] * (
            expert_logit - anchor_logit[:, None]
        )

    prediction = np.full(labels.size, np.nan, dtype=np.float64)
    selections = []
    for outer in outer_values:
        outer_mask = outer_fold == outer
        train_mask = ~outer_mask
        train_groups = np.unique(groups[train_mask])
        best = None
        for kappa in kappa_values:
            departure = departures[kappa]
            inner = {
                key: np.full(labels.size, np.nan, dtype=np.float64)
                for key in sub_grid
            }
            for held_group in train_groups:
                inner_valid = train_mask & (groups == held_group)
                inner_train = train_mask & (groups != held_group)
                if int(np.sum(inner_train)) < 50:
                    continue
                if np.unique(labels[inner_train]).size < 2:
                    continue
                train_anchor = anchor_logit[inner_train]
                train_departure = departure[inner_train]
                train_labels = labels[inner_train]
                valid_anchor = anchor_logit[inner_valid]
                valid_departure = departure[inner_valid]
                warm = None
                for anchor_scale, ridge in sub_grid:
                    weights, intercept = fit_stack(
                        train_anchor,
                        train_departure,
                        train_labels,
                        anchor_scale,
                        ridge,
                        config,
                        warm_start=warm,
                        allow_experts=allow_experts,
                    )
                    warm = np.append(weights, intercept)
                    inner[(anchor_scale, ridge)][inner_valid] = apply_stack(
                        valid_anchor,
                        valid_departure,
                        anchor_scale,
                        weights,
                        intercept,
                    )
            for anchor_scale, ridge in sub_grid:
                values = inner[(anchor_scale, ridge)]
                covered = np.isfinite(values)
                if int(np.sum(covered)) == 0:
                    continue
                score = robust_criterion(
                    labels[covered],
                    domain_id[covered],
                    anchor_probability[covered],
                    values[covered],
                    config["robust_lambda"],
                )
                if best is None or score > best["inner_robust_score"]:
                    best = {
                        "support_kappa": float(kappa),
                        "anchor_scale": float(anchor_scale),
                        "ridge": float(ridge),
                        "inner_robust_score": float(score),
                        "inner_covered": int(np.sum(covered)),
                    }
        common.require(best is not None, "phase67b_inner_selection_empty")
        departure = departures[best["support_kappa"]]
        weights, intercept = fit_stack(
            anchor_logit[train_mask],
            departure[train_mask],
            labels[train_mask],
            best["anchor_scale"],
            best["ridge"],
            config,
            allow_experts=allow_experts,
        )
        prediction[outer_mask] = apply_stack(
            anchor_logit[outer_mask],
            departure[outer_mask],
            best["anchor_scale"],
            weights,
            intercept,
        )
        selections.append(
            {
                "outer_unit": int(outer),
                "support_kappa": best["support_kappa"],
                "anchor_scale": best["anchor_scale"],
                "ridge": best["ridge"],
                "inner_robust_score": best["inner_robust_score"],
                "inner_covered": best["inner_covered"],
                "expert_weights": [float(value) for value in weights],
                "expert_weight_sum": float(np.sum(weights)),
                "intercept": float(intercept),
            }
        )
    common.require(
        bool(np.all(np.isfinite(prediction))),
        "phase67b_incomplete_outer_coverage",
    )
    common.require(
        bool(np.all((prediction > 0.0) & (prediction < 1.0))),
        "phase67b_prediction_out_of_range",
    )
    if expected_domains is not None:
        common.require(
            np.array_equal(np.unique(domain_id), np.arange(int(expected_domains))),
            "phase67b_domain_coverage_incomplete",
        )
    return prediction, selections


def screen_experts(labels, groups, domain_id, anchor, anchor_logit, catalogue, config):
    """Assemble the expert vectors and screen their complementarity.

    Returns (records, accepted_names, expert_logit_matrix). Phase61A screened
    historical vectors one at a time with an oracle alpha grid; the same fixed
    alpha screen is reproduced here for comparability, but its role is only to
    decide whether combining them is worth GPU time.
    """
    anchor_metrics = common.metrics(labels, anchor)
    records = []
    accepted_names = []
    accepted_logits = []
    for specification in catalogue:
        paths = specification["resolved_paths"]
        vector = common.optional_vector(specification["name"], paths)
        record = {
            "name": specification["name"],
            "available": vector is not None,
            "deployment_eligible": bool(specification["deployment_eligible"]),
            "provenance": specification["provenance"],
        }
        if vector is None:
            records.append(record)
            continue
        is_probability = bool(np.all((vector > 0.0) & (vector < 1.0)))
        record["value_type"] = (
            "probability" if is_probability else "unbounded_score"
        )
        if not is_probability:
            # Matches Phase61A: a raw score needs probability provenance and a
            # calibrator before it can enter a logit-space blend.
            record["excluded_reason"] = (
                "score_only_requires_probability_provenance_and_calibration"
            )
            records.append(record)
            continue
        expert_logit_vector = common.logit(vector)
        record["pooled"] = common.metrics(labels, vector)
        record["anchor_logit_pearson"] = float(
            np.corrcoef(anchor_logit, expert_logit_vector)[0, 1]
        )
        screen = []
        for alpha in config["screen_alpha_grid"]:
            blend = common.sigmoid(
                (1.0 - float(alpha)) * anchor_logit
                + float(alpha) * expert_logit_vector
            )
            gains = common.per_domain_gains(labels, domain_id, anchor, blend)
            screen.append(
                {
                    "alpha": float(alpha),
                    "log_loss_gain": float(
                        anchor_metrics["log_loss"]
                        - common.metrics(labels, blend)["log_loss"]
                    ),
                    "macro_domain_mean_gain": float(np.mean(gains)),
                    "across_domain_standard_deviation": float(
                        np.std(gains, ddof=1)
                    ),
                    "worst_domain_gain": float(np.min(gains)),
                }
            )
        record["fixed_alpha_screen"] = screen
        record["best_fixed_alpha_log_loss_gain"] = max(
            row["log_loss_gain"] for row in screen
        )
        records.append(record)
        accepted_names.append(specification["name"])
        accepted_logits.append(expert_logit_vector)

    common.require(bool(accepted_names), "phase67b_no_usable_expert_vector")
    return records, accepted_names, np.column_stack(accepted_logits)


def main():
    started = time.perf_counter()
    config = PHASE67B_CONFIG
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
    anchor_logit = common.logit(anchor)
    group_sizes = np.bincount(groups, minlength=common.GROUP_COUNT)
    anchor_metrics = state["anchor_metrics"]
    domain_count = len(state["stress_domain_definitions"])

    catalogue = [
        {
            **specification,
            "resolved_paths": [
                artifact_root / item for item in specification["relative_paths"]
            ],
        }
        for specification in EXPERT_CATALOGUE
    ]
    expert_records, accepted_names, expert_logit = screen_experts(
        labels, groups, domain_id, anchor, anchor_logit, catalogue, config
    )

    # Pairwise correlation among the accepted experts. Highly correlated
    # experts cannot cancel each other's per-domain harm, which is the entire
    # mechanism this cell is testing.
    columns = [anchor_logit] + [expert_logit[:, i] for i in range(expert_logit.shape[1])]
    correlation = np.corrcoef(np.column_stack(columns), rowvar=False)
    names = ["anchor"] + accepted_names
    pairwise = []
    for first in range(len(names)):
        for second in range(first + 1, len(names)):
            pairwise.append(
                {
                    "pair": [names[first], names[second]],
                    "logit_pearson": float(correlation[first, second]),
                }
            )
    promising = [
        record["name"]
        for record in expert_records
        if record.get("best_fixed_alpha_log_loss_gain", -1.0) > 0.0
        and abs(record.get("anchor_logit_pearson", 1.0))
        <= config["complementarity_gate"]["maximum_pairwise_anchor_correlation"]
    ]
    complementarity_go = bool(
        len(promising)
        >= config["complementarity_gate"]["minimum_promising_expert_count"]
    )

    grid = hyperparameter_grid(config)

    # ------------------------------------------------- protocol A (primary)
    # Outer units are the original three whole-acquisition-group folds, the
    # exact partition the anchor comparator itself was built on.
    primary, primary_selections = stacked_predictions(
        labels,
        groups,
        group_sizes,
        domain_id,
        anchor_logit,
        expert_logit,
        original_fold,
        list(range(3)),
        config,
        grid,
        expected_domains=domain_count,
    )

    # Nested anchor-recalibration baseline: identical nesting, expert weights
    # pinned to zero. It separates the log-loss gain that comes from fixing the
    # anchor's confidence from the gain that comes from genuinely new
    # information.
    #
    # The recalibration is monotone WITHIN each outer fold but each fold
    # receives its own scale and intercept, so the pooled ranking can shift
    # slightly and a small AUROC change is expected rather than exactly zero.
    # That residual is reported, not assumed away.
    calibrated, calibration_selections = stacked_predictions(
        labels,
        groups,
        group_sizes,
        domain_id,
        anchor_logit,
        expert_logit,
        original_fold,
        list(range(3)),
        config,
        grid,
        allow_experts=False,
    )

    # ---------------------------------------------- protocol B (corroborating)
    # Outer units are the 15 individual acquisition groups.
    secondary, secondary_selections = stacked_predictions(
        labels,
        groups,
        group_sizes,
        domain_id,
        anchor_logit,
        expert_logit,
        groups,
        list(range(common.GROUP_COUNT)),
        config,
        grid,
    )

    primary_stability = common.stability_rows(
        labels, groups, original_fold, domain_id, anchor, primary
    )
    primary_domain_gains = common.per_domain_gains(
        labels, domain_id, anchor, primary, expected_domains=domain_count
    )
    primary_macro = common.macro_domain_bootstrap(
        primary_domain_gains,
        config["macro_bootstrap_replicates"],
        config["macro_bootstrap_seed"],
    )
    primary_cluster = common.group_cluster_bootstrap(
        labels,
        groups,
        anchor,
        primary,
        config["cluster_bootstrap_replicates"],
        config["cluster_bootstrap_seed"],
    )
    secondary_comparison = common.compare(labels, anchor, secondary)

    # Gain decomposition. calibration_gain is AUROC-neutral by construction;
    # expert_gain is the part that carries new ranking information.
    calibration_comparison = common.compare(labels, anchor, calibrated)
    expert_comparison = common.compare(labels, calibrated, primary)
    expert_domain_gains = common.per_domain_gains(
        labels, domain_id, calibrated, primary, expected_domains=domain_count
    )
    expert_macro = common.macro_domain_bootstrap(
        expert_domain_gains,
        config["macro_bootstrap_replicates"],
        config["macro_bootstrap_seed"],
    )

    # -------------------------------------- repeated-partition sensitivity
    repeat_rows = []
    try:
        _, repeat_case_fold = common.restore_repeat_partition(
            artifact_root, groups, phase62_contract=state["phase62_contract"]
        )
        for repeat in range(repeat_case_fold.shape[0]):
            repeat_prediction, _ = stacked_predictions(
                labels,
                groups,
                group_sizes,
                domain_id,
                anchor_logit,
                expert_logit,
                repeat_case_fold[repeat],
                list(range(3)),
                config,
                grid,
            )
            row = common.compare(labels, anchor, repeat_prediction)
            row["repeat"] = int(repeat)
            repeat_rows.append(row)
        repeat_status = "computed"
    except common.Phase67Stop as stop:
        repeat_status = f"unavailable_{stop}"

    # ------------------------------------------------------- permutation null
    # Shuffling labels WITHIN each acquisition group destroys the experts'
    # signal while preserving group sizes and per-group prevalence.
    #
    # The null is measured against the nested recalibration baseline, not the
    # raw anchor. Under shuffled labels the anchor is confidently wrong, so a
    # temperature and intercept legitimately recover a large amount of log loss
    # that has nothing to do with overfitting; comparing against the raw anchor
    # would report that repair as manufactured gain and make the null useless.
    # What is at risk of being manufactured is the EXPERT weight contribution,
    # so that is what this measures.
    permutation_rng = np.random.default_rng(config["permutation_seed"])
    permutation_gains = []
    group_index = [
        np.flatnonzero(groups == group) for group in range(common.GROUP_COUNT)
    ]
    for _ in range(int(config["permutation_replicates"])):
        shuffled = labels.copy()
        for indices in group_index:
            shuffled[indices] = permutation_rng.permutation(labels[indices])
        if np.unique(shuffled).size < 2:
            continue
        null_full, _ = stacked_predictions(
            shuffled,
            groups,
            group_sizes,
            domain_id,
            anchor_logit,
            expert_logit,
            original_fold,
            list(range(3)),
            config,
            grid,
        )
        null_calibrated, _ = stacked_predictions(
            shuffled,
            groups,
            group_sizes,
            domain_id,
            anchor_logit,
            expert_logit,
            original_fold,
            list(range(3)),
            config,
            grid,
            allow_experts=False,
        )
        permutation_gains.append(
            float(np.mean(common.loss_vector(shuffled, null_calibrated)))
            - float(np.mean(common.loss_vector(shuffled, null_full)))
        )
    permutation_gains = np.asarray(permutation_gains, dtype=np.float64)
    common.require(permutation_gains.size > 0, "phase67b_permutation_null_empty")
    permutation_null = {
        "replicates": int(permutation_gains.size),
        "seed": int(config["permutation_seed"]),
        "shuffle_unit": "within_acquisition_group",
        "measured_quantity": (
            "expert_weight_log_loss_gain_over_nested_recalibration_baseline"
        ),
        "mean_manufactured_log_loss_gain": float(np.mean(permutation_gains)),
        "percentile_97_5_manufactured_log_loss_gain": float(
            np.quantile(permutation_gains, 0.975)
        ),
        "maximum_manufactured_log_loss_gain": float(np.max(permutation_gains)),
    }

    # The null bounds the expert contribution, so it is compared against the
    # expert contribution rather than against the total gain.
    observed_gain = expert_comparison["log_loss_gain"]
    permutation_fraction = (
        float(
            permutation_null["percentile_97_5_manufactured_log_loss_gain"]
            / observed_gain
        )
        if observed_gain > 0.0
        else None
    )

    target = config["target"]
    targets_met = {
        "macro_domain_mean_gain": bool(
            primary_macro["point"] >= target["minimum_macro_domain_mean_gain"]
        ),
        "across_domain_standard_deviation": bool(
            primary_macro["standard_deviation_across_domains"]
            <= target["maximum_across_domain_standard_deviation"]
        ),
        "pooled_auroc_gain": bool(
            primary_stability["pooled"]["auroc_gain"] is not None
            and primary_stability["pooled"]["auroc_gain"]
            >= target["minimum_pooled_auroc_gain"]
        ),
        "permutation_null_bounded": bool(
            permutation_fraction is not None
            and permutation_fraction <= target["maximum_permutation_null_fraction"]
        ),
        "cluster_bootstrap_lower_positive": bool(primary_cluster["lower_95"] > 0.0),
        "macro_bootstrap_lower_positive": bool(primary_macro["lower_95"] > 0.0),
    }
    accepted_records = [
        record for record in expert_records if record["name"] in accepted_names
    ]
    stack_is_deployment_eligible = all(
        record["deployment_eligible"] for record in accepted_records
    )

    if all(targets_met.values()):
        status = "phase67b_stack_met_declared_targets_proceed_to_adjudication"
    elif complementarity_go:
        status = "phase67b_stack_below_targets_complementarity_supports_phase67c"
    else:
        status = "phase67b_no_complementarity_signal_do_not_spend_gpu_budget"

    core = {
        "schema_version": config["schema_version"],
        "status": status,
        "common_version": common.PHASE67_COMMON_VERSION,
        "config": {
            key: value
            for key, value in config.items()
            if key
            not in ("artifact_root", "output_root", "labels_root", "contract_file")
        },
        "anchor": anchor_metrics,
        "experts": expert_records,
        "accepted_expert_names": accepted_names,
        "pairwise_logit_correlation": pairwise,
        "promising_expert_names": promising,
        "complementarity_supports_gpu_work": complementarity_go,
        "primary_protocol": "nested_original_three_whole_group_folds",
        "primary_pooled": primary_stability["pooled"],
        "primary_folds": primary_stability["folds"],
        "primary_domains": primary_stability["domains"],
        "primary_major_groups_combined": primary_stability["major_groups_combined"],
        "primary_maximum_fold_log_loss_regret": primary_stability[
            "maximum_fold_log_loss_regret"
        ],
        "primary_maximum_powered_domain_log_loss_regret": primary_stability[
            "maximum_powered_domain_log_loss_regret"
        ],
        "primary_improved_group_fraction": primary_stability[
            "improved_group_fraction"
        ],
        "primary_improved_domain_fraction": primary_stability[
            "improved_domain_fraction"
        ],
        "primary_selections": primary_selections,
        "primary_case_weighted_group_cluster_bootstrap": primary_cluster,
        "primary_macro_domain_bootstrap": primary_macro,
        "gain_decomposition": {
            "note": (
                "per_fold_recalibration_is_monotone_within_a_fold_but_not_"
                "across_the_pooled_vector_so_a_small_auroc_residual_is_"
                "expected_and_is_reported_here"
            ),
            "nested_recalibration_over_anchor": calibration_comparison,
            "experts_over_nested_recalibration": expert_comparison,
            "expert_macro_domain_bootstrap": expert_macro,
            "calibration_selections": calibration_selections,
        },
        "secondary_protocol": "meta_level_leave_one_acquisition_group_out",
        "secondary_pooled": secondary_comparison,
        "secondary_selection_count": len(secondary_selections),
        "repeated_partition_status": repeat_status,
        "repeated_partition_rows": repeat_rows,
        "permutation_null": permutation_null,
        "permutation_null_fraction_of_observed_gain": permutation_fraction,
        "declared_targets": target,
        "targets_met": targets_met,
        "stack_uses_only_deployment_eligible_vectors": stack_is_deployment_eligible,
        "interpretation": {
            "historical_oof_vectors_are_original_three_fold_oof": True,
            "meta_features_may_reflect_base_models_that_saw_other_outer_folds": True,
            "anchor_comparator_carries_the_identical_oof_status": True,
            "repeated_partitions_reuse_the_same_patients": True,
            "repeated_partitions_are_sensitivity_only": True,
            "macro_domain_bootstrap_weights_a_32_case_domain_like_a_456_case_one": True,
            "case_weighted_cluster_bootstrap_matches_the_scored_metric": True,
            "bootstrap_is_descriptive_after_many_prior_experiments": True,
            "development_gain_is_not_a_leaderboard_result": True,
            "passing_targets_does_not_establish_deployment_parity": True,
        },
        "training_performed": False,
        "training_voxel_data_read": False,
        "test_data_read": False,
        "test_time_adaptation": False,
        "phase56_archive_unchanged": True,
        "submission_archive_created": False,
        "provenance": state["provenance"],
    }

    contract_path = output_root / config["contract_file"]
    contract_sha = common.emit_contract(contract_path, core)
    # The stacked development vector stays private; downstream cells read it
    # from the private output directory, never from a shared report.
    common.atomic_npz(
        output_root / "phase67b_stacked_development_oof.npz",
        contract_sha256=np.asarray(contract_sha),
        primary=primary,
        secondary=secondary,
        nested_recalibration_baseline=calibrated,
    )
    report = {
        **core,
        "contract_sha256": contract_sha,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    common.print_sanitized("PHASE67B_NESTED_MULTI_EXPERT_STACK", report)
    return report


if __name__ == "__main__":
    try:
        main()
    except common.Phase67Stop as stop:
        print(
            json.dumps(
                {
                    "phase": "phase67b_nested_multi_expert_stack",
                    "status": "phase67b_stopped",
                    "stage": str(stop),
                    "phase56_archive_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
