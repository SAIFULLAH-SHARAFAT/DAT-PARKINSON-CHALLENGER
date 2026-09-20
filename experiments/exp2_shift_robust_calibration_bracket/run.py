"""Experiment 2 - the shift-robust calibration bracket.

HYPOTHESIS
    The optimal calibration slope falls monotonically as the evaluation set moves
    away from the training distribution. If that is true and measurable offline,
    the in-distribution optimum is the wrong slope to ship, and the size of the
    error is estimable from held-out acquisition clusters alone.

WHY THIS EXPERIMENT EXISTS
    Phase68R measured 1.00 -> 1.15 -> 1.30 on the public board, improving
    monotonically, and stopped at 1.30. That is this lever, pulled once and
    abandoned. Two independent top-ten solutions measured that the optimum keeps
    flattening as the evaluation population hardens, and one of them finished
    sixth on a submission that had looked 0.0004 WORSE in public and was written
    off for it. This experiment makes that ladder measurable before the fact.

WHAT IT DOES NOT DO
    It does not retrain anything. It reads one out-of-fold probability vector and
    the partition it was produced under. A ladder measured here is evidence about
    where the optimum moves, never proof of a private score.

INPUTS (one local directory, from --data-root or $DAT_DATA_ROOT)
    train_labels.csv        uid,is_pathologic
    oof_probability.npy     out-of-fold probability per case
    acquisition_group.npy   acquisition cluster per case
    original_fold.npy       outer fold per case
    Each can be overridden individually on the command line.

OUTPUT
    A sanitized JSON contract in --output-root (or $DAT_OUTPUT_ROOT). No
    case-level values, no identifiers, no paths.

    python run.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import datcore as dat  # noqa: E402


CONFIG_FILE = Path(__file__).with_name("config.json")


# --------------------------------------------------------------------------
# Calibration primitives
# --------------------------------------------------------------------------

def apply_slope(logits, slope, intercept):
    return dat.sigmoid(slope * logits + intercept)


def log_loss(labels, probability, clip):
    p = np.clip(probability, clip, 1.0 - clip)
    return float(-np.mean(labels * np.log(p) + (1.0 - labels) * np.log(1.0 - p)))


def _newton_platt(labels, logits, clip, pinned_slope=None):
    """Exact two-parameter (or intercept-only) calibration fit by Newton-Raphson.

    Fitting p = sigmoid(a*z + b) to minimise log loss IS logistic regression on a
    single feature, so the Hessian is available in closed form and the fit
    converges in a handful of iterations. A grid search over slopes with a nested
    line search would be thousands of times slower for a worse answer, which
    matters because the matched null refits this many times.
    """
    labels = np.asarray(labels, dtype=np.float64)
    logits = np.asarray(logits, dtype=np.float64)
    if pinned_slope is None:
        design = np.stack([logits, np.ones_like(logits)], axis=1)
        theta = np.array([1.0, 0.0], dtype=np.float64)
    else:
        design = np.ones((logits.size, 1), dtype=np.float64)
        theta = np.zeros(1, dtype=np.float64)
    ridge = 1e-9 * np.eye(design.shape[1])
    for _ in range(100):
        linear = design @ theta + (0.0 if pinned_slope is None else pinned_slope * logits)
        probability = np.clip(dat.sigmoid(linear), clip, 1.0 - clip)
        gradient = design.T @ (probability - labels)
        weight = probability * (1.0 - probability)
        hessian = design.T @ (design * weight[:, None]) + ridge
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            break
        theta = theta - step
        if float(np.max(np.abs(step))) < 1e-11:
            break
    if pinned_slope is None:
        return float(theta[0]), float(theta[1])
    return float(pinned_slope), float(theta[0])


def best_intercept(labels, logits, slope, bounds, clip):
    """The intercept minimising log loss at a pinned slope, clamped to bounds."""
    _, intercept = _newton_platt(labels, logits, clip, pinned_slope=float(slope))
    intercept = float(np.clip(intercept, bounds[0], bounds[1]))
    return intercept, log_loss(labels, apply_slope(logits, slope, intercept), clip)


def optimal_slope(labels, logits, config):
    """The (slope, intercept) minimising log loss on this evaluation set.

    The fit is unconstrained; the result is then clamped into the declared grid
    range so a degenerate cluster cannot emit a slope the bracket would never
    ship. `clamped` records whether that happened, because a clamped optimum is
    evidence the evaluation set is too small to calibrate on.
    """
    clip = float(config["probability_clip"])
    lower = float(config["slope_grid"]["lower"])
    upper = float(config["slope_grid"]["upper"])
    bounds = tuple(config["intercept_bounds"])
    slope, _ = _newton_platt(labels, logits, clip)
    clamped = not (lower <= slope <= upper) or not np.isfinite(slope)
    if clamped:
        slope = float(np.clip(slope if np.isfinite(slope) else 1.0, lower, upper))
    intercept, value = best_intercept(labels, logits, slope, bounds, clip)
    return {
        "slope": float(slope),
        "intercept": float(intercept),
        "log_loss": float(value),
        "clamped": bool(clamped),
    }


# --------------------------------------------------------------------------
# The ladder
# --------------------------------------------------------------------------

def distance_from_training(logits, mask):
    """How far a held-out cluster sits from the rest, in class-centred logit space.

    Returns 1 - rho, where rho is the correlation between the held-out cluster's
    logit distribution and the remainder's, compared through matched quantiles.
    This is a population-shift statistic, not a performance statistic, so it can
    be computed without labels for the target set.
    """
    inside, outside = logits[mask], logits[~mask]
    if inside.size < 2 or outside.size < 2:
        return None
    quantiles = np.linspace(0.02, 0.98, 49)
    a = np.quantile(inside, quantiles)
    b = np.quantile(outside, quantiles)
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return None
    rho = float(np.corrcoef(a, b)[0, 1])
    return float(1.0 - rho)


def ladder(labels, logits, groups, config):
    """Per-cluster optimal slopes, each measured on a cluster held out entirely."""
    rows = []
    minimum_n = int(config["minimum_cluster_n"])
    for cluster in sorted(set(int(g) for g in groups)):
        mask = groups == cluster
        n = int(mask.sum())
        if n < minimum_n:
            continue
        held = optimal_slope(labels[mask], logits[mask], config)
        # The slope a developer WOULD have shipped, fitted without this cluster.
        fitted = optimal_slope(labels[~mask], logits[~mask], config)
        clip = float(config["probability_clip"])
        cost = log_loss(
            labels[mask], apply_slope(logits[mask], fitted["slope"], fitted["intercept"]), clip
        )
        rows.append(
            {
                "cluster": int(cluster),
                "n": n,
                "held_out_optimal_slope": held["slope"],
                "held_out_optimal_log_loss": held["log_loss"],
                "in_distribution_fitted_slope": fitted["slope"],
                "log_loss_at_fitted_slope": cost,
                "regret_of_fitted_slope": float(cost - held["log_loss"]),
                "distance_from_training": distance_from_training(logits, mask),
            }
        )
    return rows


def matched_null(labels, logits, groups, config):
    """Permute cluster identity, keeping sizes fixed.

    If the observed spread of per-cluster optimal slopes is no wider than this
    null, the ladder is an artefact of small evaluation sets and must not be
    shipped. This is the control that separates a real ladder from sampling noise.
    """
    rng = np.random.default_rng(int(config["matched_null"]["seed"]))
    observed = [row["held_out_optimal_slope"] for row in ladder(labels, logits, groups, config)]
    if len(observed) < 2:
        return {"available": False}
    observed_sd = float(np.std(observed, ddof=1))
    spreads = []
    permutations = int(config["matched_null"]["permutations"])
    for _ in range(permutations):
        shuffled = rng.permutation(groups)
        slopes = [
            optimal_slope(labels[shuffled == c], logits[shuffled == c], config)["slope"]
            for c in sorted(set(int(g) for g in shuffled))
            if int((shuffled == c).sum()) >= int(config["minimum_cluster_n"])
        ]
        if len(slopes) >= 2:
            spreads.append(float(np.std(slopes, ddof=1)))
    if not spreads:
        return {"available": False}
    null_mean = float(np.mean(spreads))
    null_sd = float(np.std(spreads, ddof=1)) if len(spreads) > 1 else 0.0
    sigma = float((observed_sd - null_mean) / null_sd) if null_sd > 0 else 0.0
    return {
        "available": True,
        "observed_slope_spread": observed_sd,
        "null_slope_spread_mean": null_mean,
        "null_slope_spread_sd": null_sd,
        "observed_over_null_sigma": sigma,
        "permutations": len(spreads),
    }


def extrapolate(rows, in_distribution_slope):
    """Regress the held-out optimal slope on distance, then read it out further.

    A straight line is deliberate. With at most a handful of clusters there is no
    power to fit anything richer, and the claim being tested is only that the
    slope falls with distance -- the sign and rough magnitude, not the shape.
    """
    usable = [r for r in rows if r["distance_from_training"] is not None]
    if len(usable) < 3:
        return {"available": False, "reason": "fewer_than_three_usable_clusters"}
    x = np.asarray([r["distance_from_training"] for r in usable], dtype=np.float64)
    y = np.asarray([r["held_out_optimal_slope"] for r in usable], dtype=np.float64)
    if float(np.std(x)) < 1e-12:
        return {"available": False, "reason": "no_spread_in_distance"}
    slope_coef, intercept_coef = np.polyfit(x, y, 1)
    predicted = intercept_coef + slope_coef * x
    residual = y - predicted
    denominator = float(np.sum((y - y.mean()) ** 2))
    r_squared = float(1.0 - np.sum(residual**2) / denominator) if denominator > 0 else 0.0
    return {
        "available": True,
        "clusters_used": len(usable),
        "slope_per_unit_distance": float(slope_coef),
        "intercept": float(intercept_coef),
        "r_squared": r_squared,
        "falls_with_distance": bool(slope_coef < 0.0),
        "in_distribution_slope": float(in_distribution_slope),
        "observed_distance_range": [float(x.min()), float(x.max())],
        "_reading": (
            "A negative slope_per_unit_distance is the ladder: the further the "
            "evaluation population sits from training, the flatter the optimum."
        ),
    }


def build_bracket(rows, in_distribution, fit, config):
    """Emit the slopes to actually submit.

    Never a point estimate. The anchor is the in-distribution optimum; the
    flatter members come from the measured per-cluster optima, which are the only
    direct evidence available offline about a harder population.
    """
    quantiles = config["bracket"]["quantiles_of_extrapolated_slope"]
    observed = sorted(r["held_out_optimal_slope"] for r in rows)
    members = [
        {
            "slope": float(in_distribution["slope"]),
            "basis": "in_distribution_optimum",
            "_warning": "Ship this only as the sharp end of the bracket, never alone.",
        }
    ]
    if observed:
        for q in quantiles:
            members.append(
                {
                    "slope": float(np.quantile(observed, q)),
                    "basis": f"quantile_{q}_of_held_out_cluster_optima",
                }
            )
    if fit.get("available") and fit.get("falls_with_distance"):
        beyond = float(fit["intercept"] + fit["slope_per_unit_distance"]
                       * (fit["observed_distance_range"][1] * 1.5))
        members.append(
            {
                "slope": float(np.clip(beyond, config["slope_grid"]["lower"],
                                       config["slope_grid"]["upper"])),
                "basis": "linear_extrapolation_to_1p5x_the_widest_observed_distance",
                "_warning": "Extrapolated, not measured. The flattest member of the bracket.",
            }
        )
    # Deduplicate while preserving order, sharpest first.
    seen, unique = set(), []
    for member in members:
        key = round(member["slope"], 4)
        if key not in seen:
            seen.add(key)
            unique.append(member)
    return unique


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

def load_inputs(args):
    data_root = dat.resolve_root(args.data_root, "DAT_DATA_ROOT")
    labels = dat.load_labels(args.labels or data_root / "train_labels.csv")
    probability = np.load(
        Path(args.oof or data_root / "oof_probability.npy").expanduser(), allow_pickle=False
    ).astype(np.float64)
    dat.require(probability.shape == (dat.CASE_COUNT,), "oof_probability_has_the_wrong_length")
    dat.require(
        bool(np.all((probability > 0.0) & (probability < 1.0))),
        "oof_probability_outside_open_unit_interval",
    )
    groups = dat.load_vector(args.groups or data_root / "acquisition_group.npy",
                             "acquisition_group.npy")
    folds = dat.load_vector(args.folds or data_root / "original_fold.npy", "original_fold.npy")
    return labels, probability, groups, folds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default="",
                        help="directory holding the inputs (default: $DAT_DATA_ROOT)")
    parser.add_argument("--output-root", default=os.environ.get("DAT_OUTPUT_ROOT", ""),
                        help="writable directory for the contract (default: $DAT_OUTPUT_ROOT)")
    parser.add_argument("--labels", default="", help="override: train_labels.csv")
    parser.add_argument("--oof", default="", help="out-of-fold probability .npy")
    parser.add_argument("--groups", default="", help="acquisition cluster id .npy")
    parser.add_argument("--folds", default="", help="outer fold id .npy")
    parser.add_argument("--config", default=str(CONFIG_FILE))
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dat.require(bool(args.output_root), "DAT_OUTPUT_ROOT_not_set")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    started = time.time()
    labels, probability, groups, folds = load_inputs(args)
    core = analyse(labels, probability, groups, config)
    core["elapsed_seconds"] = round(time.time() - started, 3)

    contract_path = Path(args.output_root) / config["contract_file"]
    contract_sha = dat.emit_contract(contract_path, core)
    report = {**core, "contract_sha256": contract_sha}
    dat.print_sanitized("EXP2_SHIFT_ROBUST_CALIBRATION_BRACKET", report)
    return 0 if all(core["gates"].values()) else 2


def analyse(labels, probability, groups, config):
    """The whole measurement, on arrays only.

    Separated from `main` so the synthetic test exercises exactly the code that
    runs on real data, rather than a re-implementation of it.
    """
    labels = np.asarray(labels, dtype=np.float64)
    probability = np.asarray(probability, dtype=np.float64)
    groups = np.asarray(groups, dtype=np.int64)
    clip = float(config["probability_clip"])
    logits = dat.logit(np.clip(probability, clip, 1.0 - clip))

    baseline = dat.metrics(labels, probability)
    in_distribution = optimal_slope(labels, logits, config)
    rows = ladder(labels, logits, groups, config)
    null = matched_null(labels, logits, groups, config)
    fit = extrapolate(rows, in_distribution["slope"])
    bracket = build_bracket(rows, in_distribution, fit, config)

    for member in bracket:
        intercept, value = best_intercept(
            labels, logits, member["slope"], tuple(config["intercept_bounds"]), clip
        )
        member["intercept"] = intercept
        member["in_distribution_log_loss"] = value
        member["in_distribution_cost_vs_optimum"] = float(
            value - in_distribution["log_loss"]
        )

    sharp = min(m["in_distribution_cost_vs_optimum"] for m in bracket)
    flattest = max(bracket, key=lambda m: m["in_distribution_cost_vs_optimum"])
    bar = config["ship_bar"]
    gates = {
        "ladder_exceeds_matched_null": bool(
            null.get("available")
            and null["observed_over_null_sigma"] >= bar["minimum_slope_spread_over_null_sd"]
        ),
        "slope_falls_with_distance": bool(fit.get("available") and fit.get("falls_with_distance")),
        "flat_arm_in_distribution_cost_within_bound": bool(
            flattest["in_distribution_cost_vs_optimum"]
            <= bar["maximum_in_distribution_log_loss_cost"]
        ),
        "bracket_has_minimum_members": bool(len(bracket) >= config["bracket"]["minimum_members"]),
    }
    if bar.get("require_monotone_slope_vs_distance"):
        gates["monotone_required_and_observed"] = gates["slope_falls_with_distance"]

    passed = all(gates.values())
    status = (
        "exp2_ladder_measured_ship_the_bracket"
        if passed
        else "exp2_ladder_not_established_ship_the_in_distribution_optimum_only"
    )

    core = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "status": status,
        "baseline_uncalibrated": baseline,
        "in_distribution_optimum": in_distribution,
        "per_cluster_ladder": rows,
        "matched_null": null,
        "distance_fit": fit,
        "bracket": bracket,
        "sharpest_member_cost": float(sharp),
        "gates": gates,
        "declared_ship_bar": bar,
        "distance_instrument": config["distance_instrument"],
        "interpretation": {
            "retraining_performed": False,
            "test_data_read": False,
            "case_level_values_exported": False,
            "a_measured_ladder_is_not_a_private_score": True,
            "extrapolated_bracket_members_are_modelled_not_observed": True,
        },
    }
    return core


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except dat.DatStop as stop:
        print(f"EXP2 STOP: {stop}", file=sys.stderr)
        raise SystemExit(2)
