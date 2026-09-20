# Experiment 1 — is the residual loss a *syndrome* problem, not a severity problem?

**The highest-value idea available, and nobody has tested it.** An entrant who reached the same
conclusion said outright that they ran out of time before they could try it.

## The observation this is built on

An independent entrant embedded a ResNet's features and found that high-loss **pathological**
scans sit inside the healthy cluster — and that the overlap is **asymmetric**: healthy cases never
bleed the other way. Reproduced across 20 models on 20 different splits, which makes it a property
of the data rather than an artefact of one model.

A nuclear-medicine physician in the same discussion supplied the mechanism. The scans they got
most confidently wrong were *bright, symmetric, comma-shaped and stored as abnormal*. Symmetric
bilateral reduction fits PSP or MSA better than typical Parkinson's, and the reporting guidance is
explicit that **abnormal does not mean asymmetric**.

Put together: the binary head is being asked to learn one decision boundary for two
positive populations that do not look alike. The hypothesis is that the confusion is about
**syndrome, not severity**.

This archive contains the same failure from the other side — Phase59's OOF failure anatomy,
and abnormal-labelled scans sitting at the 39th–54th percentile of clear normals on putamen
uptake.

## The design

Split the pathological class into **asymmetric-typical** and **symmetric-atypical** using the
scans themselves, supervise that split as an **auxiliary head**, and keep the shipped output
binary. The auxiliary head is discarded at inference, so deployment is unchanged.

Three things make this more than a three-class reskin:

1. **The stratum is fold-local.** The asymmetry threshold is a quantile fitted over pathological
   *training* cases in each outer fold and applied unchanged to the held-out fold. No held-out case
   informs its own stratum. `test_synthetic.py` proves this by tampering with held-out descriptors
   and asserting the threshold does not move — and then tampering with training descriptors and
   asserting it does, so the check has teeth.
2. **There is a matched null.** The identical model with the identical auxiliary head, trained on
   stratum labels **permuted within the pathological cases**. Class balance, stratum sizes, head
   width and parameter count are all preserved; only the scan-to-stratum correspondence is
   destroyed. If the real split does not beat the permuted one, the auxiliary head was buying
   capacity, not structure. This is the control that turned a competitor's "equivariance works"
   result into "equivariance contributes nothing."
3. **Loss is reported decomposed by stratum**, so a gain in the contested tail cannot be hidden by
   the easy majority, and harm to the majority cannot be hidden by the tail.

## No absolute thresholds, anywhere

Every descriptor is a ratio against the scan's own reference. This is not fastidiousness: the
training augmentation applies a randomised global gain, so an absolute intensity cut is wrong by
the gain factor during training and exactly right at validation. That manufactures a leak which
looks like a result. The synthetic test asserts gain invariance at 0.5× and 2.3×.

## Declared gates

In `config.json`, written before the first run.

| gate | bar |
|---|---|
| `minimum_pooled_log_loss_gain` | ≥ 0.005 |
| `minimum_auroc_gain` | ≥ 0 |
| `beats_matched_null` | ≥ 0.003 over the permuted-stratum control |
| `stratum_regret_within_bound` | no stratum worse by > 0.002 |
| `fold_regret_within_bound` | no fold worse by > 0.002 |
| `gain_exceeds_measured_seed_noise` | ≥ 3 × the measured seed standard deviation |
| `all_seeds_same_sign` | required |

The seed noise floor is **measured, not assumed** — three seeds, reported in the contract before
any comparison is read.

## Running it

```bash
# one-off, or put these in config/local.env
export DAT_DATA_ROOT="/path/to/data"      # labels + prepared arrays
export DAT_OUTPUT_ROOT="/path/to/runs"    # writable, for the contract

python run.py --device cuda
```

| input under `$DAT_DATA_ROOT` | contents |
|---|---|
| `volume_cache.npy` | `(cases, 80, 80, 80)`, memory-mapped |
| `acquisition_group.npy` | acquisition cluster per case |
| `original_fold.npy` | outer fold per case |

Six training runs (three seeds × candidate and null) over three outer folds. Exit code 0 if every
gate passes, 2 otherwise.

## Validate it without any data

```bash
python test_synthetic.py
```

Builds volumes with a *planted* syndrome split — symmetric-but-reduced versus one-sided — and
checks the descriptors separate them, the stratum recovers the planted truth on >85% of
pathological cases, the threshold is fold-local, the permutation preserves what it must, a
candidate identical to its control scores exactly zero, and the two-headed model trains end to end.

## If it fails

That is a real result and worth recording. It would mean the pathological class is not usefully
separable on lateralisation alone, and the next thing to vary is the descriptor — an
antero-posterior tail profile, or a caudate-to-putamen ratio — not the model.
