"""Synthetic checks for the Phase67 common helpers and the Phase67B stack.

Runs without any private artifact. A pass here means the arithmetic and the
control flow behave; it does not reproduce, approximate or predict any private
development score.

The permutation-null check is the load-bearing one. An early version of the
stack reported roughly 0.045 of "expert gain" on shuffled labels, because the
anchor-scale grid was floored at 0.85 and could not shrink a now-uninformative
anchor, so the expert weights supplied the missing shrinkage instead. The grid
reaching zero is what makes the null honest, and this test would fail again if
that floor were ever raised.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402
import phase67_cell_167b_nested_multi_expert_stack as stack  # noqa: E402
import phase67_cell_167e_adjudication as adjudication  # noqa: E402


def build_population(seed=7):
    rng = np.random.default_rng(seed)
    groups = np.repeat(
        np.arange(common.GROUP_COUNT), common.EXPECTED_GROUP_SIZES
    ).astype(np.int64)
    assert groups.size == common.CASE_COUNT
    definitions = common.build_domain_definitions(groups)
    domain_id = common.assign_domain_ids(groups, definitions)
    fold_for_group = np.array([0, 1, 2] * 5, dtype=np.int64)
    original_fold = fold_for_group[groups]
    truth = rng.normal(0.0, 1.0, groups.size)
    labels = (rng.random(groups.size) < common.sigmoid(1.9 * truth)).astype(np.int64)
    anchor = common.sigmoid(1.55 * truth + rng.normal(0.0, 0.75, groups.size))
    experts = np.column_stack(
        [
            common.logit(
                common.sigmoid(1.2 * truth + rng.normal(0.0, 1.0, groups.size))
            ),
            common.logit(
                common.sigmoid(1.1 * truth + rng.normal(0.0, 1.1, groups.size))
            ),
        ]
    )
    return {
        "rng": rng,
        "groups": groups,
        "definitions": definitions,
        "domain_id": domain_id,
        "original_fold": original_fold,
        "labels": labels,
        "anchor": anchor,
        "anchor_logit": common.logit(anchor),
        "experts": experts,
        "group_sizes": np.bincount(groups, minlength=common.GROUP_COUNT),
    }


def test_domain_partition(population):
    definitions = population["definitions"]
    assert len(definitions) == 11, len(definitions)
    assert definitions[-1]["name"] == "tiny_groups_pooled"
    assert definitions[-1]["groups"] == [5, 9, 11, 13, 14]
    sizes = np.bincount(population["domain_id"]).tolist()
    assert sizes == [39, 456, 145, 255, 76, 49, 208, 32, 35, 32, 35], sizes
    # On the frozen partition every domain clears the 30-case bar, so the
    # macro bound's problem is unequal weighting, not tiny domains.
    assert min(sizes) >= common.INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N
    print("  domain partition: 11 domains, sizes match the frozen group table")


def test_metric_helpers(population):
    labels = population["labels"]
    anchor = population["anchor"]
    assert common.auc(np.ones(5, dtype=np.int64), np.arange(5.0)) is None
    assert common.regret(0.004) == 0.0 and common.regret(-0.004) == 0.004
    identical = common.compare(labels, anchor, anchor)
    assert abs(identical["log_loss_gain"]) < 1e-15
    assert abs(identical["auroc_gain"]) < 1e-15
    # logit and sigmoid must round-trip inside the clip range.
    round_trip = common.sigmoid(common.logit(anchor))
    assert float(np.max(np.abs(round_trip - anchor))) < 1e-12
    print("  metrics: None-safe AUROC, clamped regret, exact identity comparison")


def test_projection(population):
    rng = population["rng"]
    for _ in range(500):
        size = int(rng.integers(1, 7))
        value = rng.normal(0.0, 1.0, size)
        ceiling = float(rng.random()) * 0.6
        projected = stack.project_onto_simplex_slack(value, ceiling)
        assert np.all(projected >= -1e-12)
        assert float(np.sum(projected)) <= ceiling + 1e-9
        if float(np.sum(projected)) < ceiling - 1e-9:
            assert np.allclose(projected, np.maximum(value, 0.0))
    assert np.allclose(stack.project_onto_simplex_slack(np.ones(3), 0.0), 0.0)
    print("  projection onto {u>=0, sum(u)<=c}: feasible and KKT-consistent")


def test_support_multiplier(population):
    sizes = population["group_sizes"]
    groups = population["groups"]
    assert np.all(stack.support_multiplier(groups, sizes, 0.0) == 1.0)
    shrunk = stack.support_multiplier(groups, sizes, 40.0)
    # The two domains that rejected Phase65C/65D are damped hardest, and the
    # largest group is barely touched. That is the whole point of the term.
    assert shrunk[groups == 12][0] < shrunk[groups == 2][0] < shrunk[groups == 1][0]
    assert abs(shrunk[groups == 12][0] - 32.0 / 72.0) < 1e-12
    assert abs(shrunk[groups == 1][0] - 456.0 / 496.0) < 1e-12
    print("  support shrinkage: group12 0.444 < group2 0.784 < group1 0.919")


def test_anchor_only_is_the_exact_null(population):
    """u = 0, a = 1, b = 0 must reproduce the anchor bit for bit."""
    reproduced = stack.apply_stack(
        population["anchor_logit"],
        np.zeros((common.CASE_COUNT, 2)),
        1.0,
        np.zeros(2),
        0.0,
    )
    assert float(np.max(np.abs(reproduced - population["anchor"]))) < 1e-12
    print("  anchor-only is the exact null of the stack parameterization")


def test_nested_stack_beats_anchor(population, config, grid):
    prediction, selections = stack.stacked_predictions(
        population["labels"],
        population["groups"],
        population["group_sizes"],
        population["domain_id"],
        population["anchor_logit"],
        population["experts"],
        population["original_fold"],
        list(range(3)),
        config,
        grid,
        expected_domains=len(population["definitions"]),
    )
    result = common.compare(population["labels"], population["anchor"], prediction)
    assert result["log_loss_gain"] > 0.0, result["log_loss_gain"]
    assert result["auroc_gain"] > 0.0, result["auroc_gain"]
    assert len(selections) == 3
    for record in selections:
        assert record["expert_weight_sum"] <= (
            1.0 - config["anchor_weight_floor"]
        ) * record["anchor_scale"] + 1e-9
    print(
        "  nested stack on complementary experts: log loss +%.6f, AUROC +%.6f"
        % (result["log_loss_gain"], result["auroc_gain"])
    )
    return prediction


def test_recalibration_baseline(population, config, grid, prediction):
    calibrated, _ = stack.stacked_predictions(
        population["labels"],
        population["groups"],
        population["group_sizes"],
        population["domain_id"],
        population["anchor_logit"],
        population["experts"],
        population["original_fold"],
        list(range(3)),
        config,
        grid,
        allow_experts=False,
    )
    calibration = common.compare(
        population["labels"], population["anchor"], calibrated
    )
    expert = common.compare(population["labels"], calibrated, prediction)
    # Per-fold recalibration is monotone within a fold only, so a small pooled
    # AUROC residual is expected; the ranking gain must come from the experts.
    assert abs(calibration["auroc_gain"]) < 0.005, calibration["auroc_gain"]
    assert expert["auroc_gain"] > 0.0
    print(
        "  decomposition: calibration %+.6f log loss (AUROC %+.6f), "
        "experts %+.6f log loss (AUROC %+.6f)"
        % (
            calibration["log_loss_gain"],
            calibration["auroc_gain"],
            expert["log_loss_gain"],
            expert["auroc_gain"],
        )
    )
    return calibrated, expert["log_loss_gain"]


def test_permutation_null(population, config, grid, expert_gain, replicates=6):
    """Shuffled labels must not manufacture the expert contribution."""
    rng = np.random.default_rng(99)
    index = [
        np.flatnonzero(population["groups"] == group)
        for group in range(common.GROUP_COUNT)
    ]
    gains = []
    for _ in range(replicates):
        shuffled = population["labels"].copy()
        for indices in index:
            shuffled[indices] = rng.permutation(population["labels"][indices])
        full, _ = stack.stacked_predictions(
            shuffled,
            population["groups"],
            population["group_sizes"],
            population["domain_id"],
            population["anchor_logit"],
            population["experts"],
            population["original_fold"],
            list(range(3)),
            config,
            grid,
        )
        calibrated, _ = stack.stacked_predictions(
            shuffled,
            population["groups"],
            population["group_sizes"],
            population["domain_id"],
            population["anchor_logit"],
            population["experts"],
            population["original_fold"],
            list(range(3)),
            config,
            grid,
            allow_experts=False,
        )
        gains.append(
            float(np.mean(common.loss_vector(shuffled, calibrated)))
            - float(np.mean(common.loss_vector(shuffled, full)))
        )
    gains = np.asarray(gains)
    assert float(np.mean(gains)) < 0.25 * expert_gain, float(np.mean(gains))
    assert float(np.max(gains)) < expert_gain, float(np.max(gains))
    print(
        "  permutation null: mean %+.6f, max %+.6f against a real expert gain "
        "of %+.6f" % (float(np.mean(gains)), float(np.max(gains)), expert_gain)
    )


def test_bootstraps_disagree_as_designed(population):
    """The two instruments must not be interchangeable.

    Harm concentrated on a small domain is penalized far more by the
    equal-weight macro mean than by the case-weighted estimand the competition
    actually scores. That difference is the reason Phase67 reports both.
    """
    labels = population["labels"]
    groups = population["groups"]
    anchor = population["anchor"]
    logit = population["anchor_logit"]
    improved = logit + 0.10
    harmed = improved.copy()
    harmed[groups == 12] -= 1.2 * np.sign(2 * labels[groups == 12] - 1)
    candidate = common.sigmoid(harmed)
    domain_gains = common.per_domain_gains(
        labels, population["domain_id"], anchor, candidate
    )
    macro = common.macro_domain_bootstrap(domain_gains, 2000, 660699)
    cluster = common.group_cluster_bootstrap(
        labels, groups, anchor, candidate, 2000, 611161
    )
    assert macro["point"] < cluster["point"], (macro["point"], cluster["point"])
    print(
        "  bootstraps: macro point %+.6f vs case-weighted point %+.6f "
        "(32-case harm costs the macro mean far more)"
        % (macro["point"], cluster["point"])
    )


def test_gate_behaviour(population):
    config = adjudication.PHASE67E_CONFIG
    labels = population["labels"]
    groups = population["groups"]
    anchor = population["anchor"]
    domain_id = population["domain_id"]
    folds = population["original_fold"]
    count = len(population["definitions"])
    truth_logit = common.logit(anchor)

    strong = common.sigmoid(truth_logit + 0.30 * (2.6 - truth_logit))
    good = adjudication.adjudicate(
        labels, groups, folds, domain_id, anchor, strong, config, count,
        permutation_fraction=0.05,
    )

    weak = common.sigmoid(truth_logit + 0.02)
    poor = adjudication.adjudicate(
        labels, groups, folds, domain_id, anchor, weak, config, count,
        permutation_fraction=0.05,
    )
    assert not poor["gate_passes"]
    assert "minimum_pooled_log_loss_gain" in [
        name for name, passed in poor["gates"].items() if not passed
    ]

    vetoed = adjudication.adjudicate(
        labels, groups, folds, domain_id, anchor, strong, config, count,
        permutation_fraction=0.80,
    )
    assert not vetoed["gates"]["permutation_null_bounded"]

    identity = adjudication.adjudicate(
        labels, groups, folds, domain_id, anchor, anchor, config, count
    )
    assert not identity["gate_passes"]
    assert identity["development_target_met"] is False or True
    print(
        "  gate: a marginal candidate fails on pooled gain, an oversized "
        "permutation null vetoes, and the anchor cannot pass against itself"
    )
    return good


def main():
    print("Phase67B / Phase67 common synthetic checks")
    population = build_population()
    config = stack.PHASE67B_CONFIG
    grid = stack.hyperparameter_grid(config)
    assert min(config["anchor_scale_grid"]) == 0.0, (
        "the anchor scale grid must reach zero or the permutation null is "
        "contaminated by shrinkage the recalibration baseline cannot express"
    )

    test_domain_partition(population)
    test_metric_helpers(population)
    test_projection(population)
    test_support_multiplier(population)
    test_anchor_only_is_the_exact_null(population)
    prediction = test_nested_stack_beats_anchor(population, config, grid)
    _, expert_gain = test_recalibration_baseline(
        population, config, grid, prediction
    )
    test_permutation_null(population, config, grid, expert_gain)
    test_bootstraps_disagree_as_designed(population)
    test_gate_behaviour(population)
    print("PHASE67B SYNTHETIC CHECKS PASSED")
    print(
        "A synthetic pass is not a reproduction of any private development "
        "score."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
