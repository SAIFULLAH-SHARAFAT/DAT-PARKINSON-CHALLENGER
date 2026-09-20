# Methodology

How a result earns the right to be believed in this repository. These rules were written after the
competition, from what actually separated the top ten from the rest of the field — and from the
places my own 67 phases spent effort without buying anything.

## 1. Measure the noise floor before the first comparison

Train one configuration three times, changing only the seed, and report the standard deviation.
Published measurements on this dataset put per-fold seed noise at 0.005–0.009 log loss for a
DenseNet and 0.02–0.07 for an EfficientNet-B0. A single-seed screen on the latter can show a 0.05
"improvement" that is entirely noise.

Nothing is promoted below **3× the measured floor**. Each experiment reports
`measured_seed_noise_floor` in its contract before any gain is read.

## 2. Declare the ship bar before the run

Every `config.json` carries its thresholds and a `_declared_before_results` note. The bar is a
conjunction — `d(log loss) ≤ −X AND d(AUROC) ≥ 0` — never a single metric that can be cherry-picked
after the fact. Editing a bar after seeing an outcome is how a search becomes a story; add a new
config file instead.

## 3. Every screen gets a matched null

Same shape, same dimension, same parameter count, same pipeline — signal removed.

| experiment | its null |
|---|---|
| 1 | identical model and auxiliary head, stratum labels permuted *within* the pathological cases |
| 2 | cluster identity permuted with cluster sizes held fixed |
| 3 | the same head on the same input with no pretraining at all |

Without a null you are measuring your harness. The control in experiment 3 is the one that decides
whether a foundation model contributed anything, or whether the representation did all the work —
the same control that turned a published "equivariance works" result into "equivariance contributes
nothing."

## 4. Hold whole acquisition groups out, then read transfer separately

Complete acquisition groups are held out together, and templates, calibration and any fitted
threshold are derived **inside the training partition only**. Experiment 1's synthetic test proves
this by tampering with held-out descriptors and asserting the fitted threshold does not move, then
tampering with training descriptors and asserting it does.

Group hold-out alone is not enough. A roughly 0.007–0.010 in-distribution gain from changing the
*training distribution* is the known signature of fitting the acquisition mix harder rather than
building a better model — three separate published arms passed a clean in-distribution screen and
then failed out-of-cluster. Experiments 1 and 3 therefore also carry a per-cluster read.

## 5. No absolute thresholds

The augmentation chain applies a 0.46×–2.3× global gain. An absolute intensity cut is therefore
wrong by the gain factor during training and exactly right at validation, which manufactures a leak
that looks like a result. Every descriptor and channel in this repository is a ratio against the
scan's own reference, and the synthetic tests assert invariance at both ends of the gain range.

## 6. Ship a calibration bracket, never a point estimate

The optimal calibration slope falls as the evaluation population moves away from training. The
in-distribution optimum is therefore the wrong slope to ship. Experiment 2 measures the ladder and
emits a bracket; `docs/RESULTS.md` records why this mattered more than any architectural change.

## 7. Score the instruments, not only the arms

An instrument that has never been checked against an outcome is a belief. Published work on this
task retired a recon-fragility measure, an external-dataset check and a leave-one-cluster-out read
after each mis-ranked a package that was later scored. When a gate passes something that
subsequently fails, record that the gate was wrong.

## What the release checks do and do not cover

`scripts/verify_release.py` performs **no syntax compilation** — that is the separate
`python -m compileall` step. It checks forbidden filenames and suffixes, file size, symlinks, UTF-8
decodability, secret and path patterns, and manifest agreement compared as a mapping keyed by path
so the result cannot depend on the platform's file sort order.

The gates validate the forward path and the release surface. They do not validate a training loop,
and a synthetic pass is not a reproduction of any private score.
