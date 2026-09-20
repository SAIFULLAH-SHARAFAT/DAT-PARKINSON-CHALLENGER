# Experiment 2 — the shift-robust calibration bracket

**Run this one first.** It needs no GPU, no retraining, and no new model. It reads one
out-of-fold probability vector and tells you which calibration slopes to submit.

## Hypothesis

The optimal calibration slope falls monotonically as the evaluation set moves away from the
training distribution. If that holds, the in-distribution optimum is the wrong slope to ship, and
the size of the error is estimable offline from held-out acquisition clusters.

## Why this experiment exists

Phase68R measured 1.00 → 1.15 → 1.30 on the public board, improving at every step, and stopped at
1.30. That was the only change in the whole 67-phase archive that improved a public score
(0.3220 → 0.3131). The lever was found and then put down.

Two independent top-ten solutions measured the same thing and drew opposite conclusions:

| | in-distribution verdict | what the private board said |
|---|---|---|
| 6th place | ladder measured: optimum 1.036 out-of-fold, ~0.85 public, ~0.74 private | shipped a bracket; the arm that looked **0.0004 worse** in public won the private split by 0.0022 and is the score they finished on |
| 10th place | "all calibration ±0.0001" measured in-distribution; axis closed | their calibrated submission lost in public by 0.0003 and **won in private by 0.0012** |

The mechanism: a harder evaluation set makes confident predictions wrong more often, and
flattening the logits buys that back. Nothing measured in-distribution can see it, because
in-distribution the model is not wrong as often.

## What it measures

1. **The in-distribution optimum** — the slope you would naively ship.
2. **A per-cluster ladder** — for each acquisition cluster with enough cases, the optimal slope
   measured *on that cluster held out entirely*, plus the regret of having shipped the
   in-distribution slope instead.
3. **A matched null** — cluster identity permuted with sizes fixed. If the observed spread of
   per-cluster optima is no wider than this null, the ladder is sampling noise and is not
   reported. This is the control that stops the experiment from always saying yes.
4. **A distance fit** — the held-out optimum regressed on a label-free population-shift statistic
   (`1 − rho` between the cluster's logit quantiles and the remainder's), so the extrapolation to
   an unseen population is estimated rather than guessed.
5. **A bracket** — the slopes to actually submit, each with its own refitted intercept and its
   measured in-distribution cost.

## Declared gates

All in `config.json`, written before the first run. Do not edit them after seeing a result.

| gate | bar |
|---|---|
| `ladder_exceeds_matched_null` | observed spread ≥ 3σ above the permutation null |
| `slope_falls_with_distance` | regression coefficient negative |
| `flat_arm_in_distribution_cost_within_bound` | ≤ 0.0015 log loss |
| `bracket_has_minimum_members` | ≥ 3 |

Exit code 0 when every gate passes, 2 otherwise. A failed run is a real result: it means ship the
in-distribution optimum alone, because no ladder was established.

## Running it

```bash
# one-off, or put these in config/local.env
export DAT_DATA_ROOT="/path/to/data"      # labels + prepared arrays
export DAT_OUTPUT_ROOT="/path/to/runs"    # writable, for the contract

python run.py
```

Inputs under `$DAT_DATA_ROOT`, one value per case in the canonical order. Override any of them
individually with `--labels`, `--oof`, `--groups`, `--folds`:

| file | contents |
|---|---|
| `oof_probability.npy` | out-of-fold probability, strictly inside (0, 1) |
| `acquisition_group.npy` | acquisition cluster id |
| `original_fold.npy` | outer fold id |

Output is a sanitized JSON contract in `DAT_OUTPUT_ROOT`. No case-level values, no identifiers,
no paths.

## Validate it without any data

```bash
python test_synthetic.py
```

Builds populations whose correct answer is known by construction — labels sampled *from* a latent
logit, then inflated by a per-cluster factor, so the true optimal slope is exactly `1 / factor` —
and checks the measurement recovers it. Then it runs the same measurement on a population with
**no** ladder and asserts none is reported.

## Reading the result

- `per_cluster_ladder[].regret_of_fitted_slope` is the cost, in log loss, of having shipped the
  in-distribution slope to that cluster. If those are all near zero, there is nothing here.
- `bracket[].in_distribution_cost_vs_optimum` is the premium you pay on the easy split for
  insurance against the hard one. The published precedent is that this premium is worth a few
  ten-thousandths and the payout a few thousandths.
- `distance_fit.slope_per_unit_distance` negative **is** the ladder.

A measured ladder is evidence about where the optimum moves. It is not a private score, and
`interpretation` in the contract says so explicitly.
