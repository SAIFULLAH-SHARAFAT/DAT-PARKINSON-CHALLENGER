<div align="center">

# DaT SPECT — Parkinson's abnormality classification

**Solution archive and follow-up research for the [DrivenData DaT Parkinson's Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/)**
French Society of Nuclear Medicine · 2026 · 1,009 entrants

[![CI](https://github.com/SAIFULLAH-SHARAFAT/DaT-Parkinson-Challenger/actions/workflows/ci.yml/badge.svg)](https://github.com/SAIFULLAH-SHARAFAT/DaT-Parkinson-Challenger/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
![Log loss 0.3131](https://img.shields.io/badge/public%20log%20loss-0.3131-brightgreen)
![AUROC 0.9297](https://img.shields.io/badge/public%20AUROC-0.9297-brightgreen)

</div>

---

Predict the probability that a dopamine-transporter SPECT scan is abnormal, scored by log loss,
across 1,362 scans from ten French hospitals with wide acquisition and reconstruction variation.

This repository holds three things: **what I submitted**, **the complete record of what failed**,
and **three experiments designed from everything the competition taught**, ready to run the moment
the organisers open the data.

## Result

| | |
|:--|:--|
| Public leaderboard | **log loss 0.3131 · AUROC 0.9297 · rank #155 of 1,009** |
| Snapshot | 13 September 2026 |
| Development OOF (Phase43 comparator) | log loss 0.290166 · AUROC 0.946474 |
| Deployed model | Phase56 |

> The competition closed on **16 September 2026**. The figure above is a pre-close *public* number;
> the private board reordered the field substantially and my closing rank is not yet recorded here.

The one change that moved the board was **output calibration** — no retraining, no new model:

| probe | temperature | public log loss |
|:--|--:|--:|
| Phase56 unchanged | 1.00 | 0.3220 |
| Phase56 | 1.15 | 0.3143 |
| **Phase56** | **1.30** | **0.3131** |

Full numbers in [`docs/RESULTS.md`](docs/RESULTS.md).

## The three experiments

Each is a complete, runnable study with a declared ship bar, a measured seed-noise floor, a matched
null, whole-group hold-out, a per-cluster transfer read, and a synthetic test that runs with **no
data at all**.

| | experiment | hypothesis | cost |
|:--|:--|:--|:--|
| 🥇 | [**exp2 — calibration bracket**](experiments/exp2_shift_robust_calibration_bracket/) | the optimal slope falls as the evaluation set moves away from training, and that is measurable offline | minutes, CPU |
| 🥈 | [**exp1 — atypical syndrome structure**](experiments/exp1_atypical_syndrome_structure/) | the residual loss is a *syndrome* problem, not a severity problem | 6 fits, GPU |
| 🥉 | [**exp3 — DINOv3 on a physical projection**](experiments/exp3_dinov3_projection_expert/) | a foundation model can contribute *if* fed physics rather than a raw volume | 9 fits, GPU |

**Start with exp2.** It needs no GPU and no retraining, and calibration under distribution shift was
worth more than every architectural change made by anyone in the top ten.

Exp3 is the SOTA arm, built so it can lose conclusively: its mandatory `scratch_control` is a
matched null on the identical input, so the bar is *DINOv3 minus that control* — not DINOv3 minus an
old baseline.

## What failed, and what it cost

67 phases. Nothing was promoted after Phase56.

| phase | idea | outcome |
|:--|:--|:--|
| 57–58 | transport reset, scanner-robust architecture, gradient-pressure audit | trained arms selected epoch-zero identity |
| 59 | out-of-fold failure anatomy, calibration ceiling | loss concentrated in a thin contested tail |
| 60–61 | bilateral 3D classifier, historical complementarity, fixed blend | gains reversed on the outer fold, or failed stability |
| 62–63 | acquisition-domain reset, multitemplate expert, confidence-risk ceiling | domain harm; risk score did not locate confident errors |
| 64 | group-blocked diffusion / Nyström expert | worse overall and by major domain |
| 65A-R | archive provenance, offline runtime, Torch Hub locality | offline deployment integrity **accepted** |
| 65B–D | physics-synthetic pretraining, domain adversary, repeated-group confirmation | small average gains, all failed the domain-stability gate |
| 66 / 66F | DINOv2 ViT-S/14 volume adaptation, six fits on two GPUs | **−0.091 log loss, −0.027 AUROC — rejected** |
| 67 | nested multi-expert stack, multiseed domain-invariant expert | unreviewed work in progress |

Every number, gap and interpretation limit: [`docs/EXPERIMENT_HISTORY.md`](docs/EXPERIMENT_HISTORY.md).

## How a result earns belief here

Written up in [`docs/METHODOLOGY.md`](docs/METHODOLOGY.md). In short: measure the seed-noise floor
before the first comparison and promote nothing below 3× it; declare the ship bar before the run;
give every screen a matched null; hold whole acquisition groups out *and* read transfer separately;
never compute an absolute intensity threshold under a randomised gain; ship a calibration bracket
rather than a point estimate; and score the instruments, not only the arms.

## Layout

```
datcore/        shared library: metrics, contracts, local loaders, 3D encoder blocks
experiments/    exp1, exp2, exp3 — each with run.py, config.json, README.md, test_synthetic.py
archive/        the original Phase57–67 notebook exports, kept as history (see archive/README.md)
docs/           results, methodology, running instructions, policy, full experiment ledger
reports/        sanitized aggregate results
scripts/        release manifest + privacy verification
config/         local.env.example
```

## Quick start

```bash
pip install -r requirements.txt

# validate everything without any data
python experiments/exp2_shift_robust_calibration_bracket/test_synthetic.py

# check the repository itself
python scripts/verify_release.py
```

Full setup, inputs and flags: [`docs/RUNNING.md`](docs/RUNNING.md).

## Data and rules

Code, configuration and whole-dataset aggregate metrics **only**. No scans, labels, UIDs,
per-case predictions, derived features, embeddings, checkpoints or model weights — the competition
rules prohibit redistributing the data, and labels and identifiers are part of it. Two overlapping
layers enforce this: `.gitignore` by pattern, and `scripts/verify_release.py`, which fails closed on
forbidden files, secret patterns and any disagreement with the signed manifest.

Details in [`docs/POLICY_AND_PROVENANCE.md`](docs/POLICY_AND_PROVENANCE.md).

> **Not for clinical use.** Research software and aggregate documentation only. No clinical
> decision-support claim is made.

## Citation

See [`CITATION.cff`](CITATION.cff). Cite the challenge organisers, data providers and third-party
methods separately per their own pages and licences. Code is MIT ([`LICENSE`](LICENSE)); the licence
does not extend to the competition data.
