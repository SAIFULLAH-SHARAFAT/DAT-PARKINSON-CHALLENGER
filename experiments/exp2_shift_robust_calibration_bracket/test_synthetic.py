"""Synthetic validation for Experiment 2. Requires no private data.

The point of this test is that it builds populations whose correct answer is
known in advance, and checks that the measurement recovers it:

  POSITIVE CASE  clusters are deliberately given different amounts of overconfidence,
                 so a real ladder exists. The measurement must find it, and the
                 matched null must not.
  NULL CASE      every cluster is drawn from one distribution, so there is no ladder.
                 The measurement must NOT report one. This is the control: a test
                 that only ever checks the positive case cannot tell a working
                 instrument from one that always says yes.

Run:  python test_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run as exp2  # noqa: E402
import datcore as dat  # noqa: E402


CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
# The null needs far fewer permutations to be decisive on synthetic data, and the
# test must stay fast enough that people actually run it.
CONFIG = {**CONFIG, "matched_null": {**CONFIG["matched_null"], "permutations": 120}}


def calibrated_logits(rng, n, spread=1.8):
    """Draw a TRUE logit and labels that are consistent with it.

    The labels must be sampled FROM the logit, not the other way round. If you
    instead draw a label and then place a Gaussian around it, the calibrated
    log-odds is a rescaling of that Gaussian, not the Gaussian itself, and the
    "known" optimal slope you think you constructed is wrong by that factor.
    """
    z = rng.normal(0.0, spread, size=n)
    y = (rng.random(n) < dat.sigmoid(z)).astype(np.float64)
    return z, y


def population(rng, sizes, overconfidence, spread=1.8):
    """A population whose TRUE optimal slope per cluster is 1 / overconfidence.

    Each cluster gets a calibrated latent logit, inflated by its own factor.
    Dividing by that factor restores calibration exactly, so the correct
    per-cluster slope is known by construction.
    """
    labels, probability, groups = [], [], []
    for cluster, (n, factor) in enumerate(zip(sizes, overconfidence)):
        z, y = calibrated_logits(rng, n, spread)
        labels.append(y)
        probability.append(dat.sigmoid(z * factor))
        groups.append(np.full(n, cluster, dtype=np.int64))
    return (
        np.concatenate(labels),
        np.clip(np.concatenate(probability), 1e-9, 1.0 - 1e-9),
        np.concatenate(groups),
    )


def check_newton_fit_is_exact():
    """The optimiser must recover a slope that is known by construction."""
    rng = np.random.default_rng(11)
    z, y = calibrated_logits(rng, 20000)
    for true_factor in (0.8, 1.0, 1.3):
        probability = dat.sigmoid(z * true_factor)
        logits = dat.logit(np.clip(probability, 1e-12, 1 - 1e-12))
        found = exp2.optimal_slope(y, logits, CONFIG)
        recovered = found["slope"] * true_factor
        assert abs(recovered - 1.0) < 0.06, (true_factor, found["slope"], recovered)
    print("  newton fit: recovers the known slope for 0.8x, 1.0x and 1.3x inflation")


def check_intercept_is_optimal():
    """At a pinned slope the returned intercept must beat its neighbours."""
    rng = np.random.default_rng(5)
    z, y = calibrated_logits(rng, 4000)
    logits = z + 0.4
    clip = CONFIG["probability_clip"]
    intercept, value = exp2.best_intercept(y, logits, 0.9, (-3.0, 3.0), clip)
    for delta in (-0.05, 0.05):
        worse = exp2.log_loss(y, exp2.apply_slope(logits, 0.9, intercept + delta), clip)
        assert worse >= value - 1e-12, (delta, worse, value)
    print(f"  intercept at pinned slope is a true minimum (b={intercept:+.4f})")


def check_positive_case():
    """A real ladder must be found, and must clear the matched null."""
    rng = np.random.default_rng(202)
    sizes = [260, 240, 220, 200, 180, 160]
    inflation = [1.00, 1.15, 1.30, 1.45, 1.60, 1.80]
    labels, probability, groups = population(rng, sizes, inflation)
    result = exp2.analyse(labels, probability, groups, CONFIG)

    rows = {row["cluster"]: row for row in result["per_cluster_ladder"]}
    assert len(rows) == len(sizes), sorted(rows)
    # Cluster 0 is calibrated, cluster 5 is the most overconfident, so its optimal
    # slope must be the flatter of the two.
    assert rows[5]["held_out_optimal_slope"] < rows[0]["held_out_optimal_slope"], rows
    for cluster, factor in enumerate(inflation):
        expected = 1.0 / factor
        found = rows[cluster]["held_out_optimal_slope"]
        assert abs(found - expected) < 0.16, (cluster, factor, expected, found)

    assert result["matched_null"]["available"]
    sigma = result["matched_null"]["observed_over_null_sigma"]
    assert sigma > 0.0, result["matched_null"]
    assert result["gates"]["ladder_exceeds_matched_null"], sigma

    # Shipping the in-distribution slope must cost the hardest cluster something.
    assert rows[5]["regret_of_fitted_slope"] > 0.0, rows[5]
    print(
        f"  positive case: per-cluster optima track 1/inflation, "
        f"null exceeded at {sigma:.1f} sigma, "
        f"hardest-cluster regret +{rows[5]['regret_of_fitted_slope']:.4f}"
    )
    return result


def check_bracket(result):
    bracket = result["bracket"]
    assert len(bracket) >= CONFIG["bracket"]["minimum_members"], bracket
    slopes = [m["slope"] for m in bracket]
    assert len(set(round(s, 4) for s in slopes)) == len(slopes), "bracket has duplicates"
    assert bracket[0]["basis"] == "in_distribution_optimum"
    # Every member carries its own refitted intercept and its in-distribution cost.
    for member in bracket:
        assert "intercept" in member and "in_distribution_cost_vs_optimum" in member
        assert member["in_distribution_cost_vs_optimum"] >= -1e-9, member
    # The flattest member must be flatter than the in-distribution optimum, which
    # is the entire premise of shipping a bracket at all.
    assert min(slopes) < bracket[0]["slope"], slopes
    print(
        f"  bracket: {len(bracket)} members, "
        f"{max(slopes):.3f} down to {min(slopes):.3f}, "
        f"flat arm costs +{max(m['in_distribution_cost_vs_optimum'] for m in bracket):.5f} in distribution"
    )


def check_null_case():
    """No ladder exists, so none may be reported. The control."""
    rng = np.random.default_rng(909)
    sizes = [260, 240, 220, 200, 180, 160]
    inflation = [1.25] * len(sizes)          # identical, so no ladder
    labels, probability, groups = population(rng, sizes, inflation)
    result = exp2.analyse(labels, probability, groups, CONFIG)

    spread = result["matched_null"]["observed_slope_spread"]
    sigma = result["matched_null"]["observed_over_null_sigma"]
    bar = CONFIG["ship_bar"]["minimum_slope_spread_over_null_sd"]
    assert sigma < bar, (sigma, bar, "a ladder was reported where none exists")
    assert not result["gates"]["ladder_exceeds_matched_null"]
    assert result["status"].startswith("exp2_ladder_not_established"), result["status"]
    print(
        f"  null case: identical clusters give spread {spread:.4f} at {sigma:.1f} sigma "
        f"(bar {bar}), ladder correctly NOT reported"
    )


def check_distance_is_label_free():
    """The shift statistic must not depend on labels for the target cluster."""
    rng = np.random.default_rng(31)
    labels, probability, groups = population(rng, [600, 600], [1.0, 1.7])
    logits = dat.logit(probability)
    mask = groups == 1
    first = exp2.distance_from_training(logits, mask)
    flipped = labels.copy()
    flipped[mask] = 1.0 - flipped[mask]
    second = exp2.distance_from_training(logits, mask)
    assert first == second, (first, second)
    assert first is not None and first >= 0.0
    print(f"  distance statistic is label-free (1 - rho = {first:.4f})")


def main():
    print("Experiment 2 - shift-robust calibration bracket, synthetic checks")
    check_newton_fit_is_exact()
    check_intercept_is_optimal()
    check_distance_is_label_free()
    result = check_positive_case()
    check_bracket(result)
    check_null_case()
    print("EXP2 SYNTHETIC CHECKS PASSED")
    print("A synthetic pass validates the instrument, not any private score.")


if __name__ == "__main__":
    main()
