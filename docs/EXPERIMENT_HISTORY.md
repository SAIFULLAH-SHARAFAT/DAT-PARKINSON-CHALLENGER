# DaT Parkinson’s Challenge — Project handoff, Phases 1–67

Updated 13 September 2026. Covers all experiments and engineering steps recoverable from the supplied history, with explicit gaps. This is a reconstruction from historical summaries, source code and sanitized reports, not a fresh reanalysis of private scans or predictions.

## Read this first

You built a binary classifier for **normal versus abnormal dopamine-transporter SPECT scans**. The objective is accurate probabilities that generalize to different acquisition/reconstruction protocols. The project progressed from compact 3D CNNs to anatomy-aware ensembles, pretrained transformers, self-supervised learning, residual corrections, and experiments aimed at acquisition robustness.

**Phase56 remains the accepted deployment archive. No later experimental model was promoted.** Phase65C and Phase65D showed small average development improvements but failed their declared domain-stability gates. Phase66F completed a fresh, source-consistent two-seed/three-fold DINOv2 experiment; it underperformed the comparator on log loss, AUROC, Brier score, all three folds, and the domain-stability gate.

The existing archive’s offline execution concern was cleared by Phase65A-R4. This is separate from exact development-to-deployment probability parity, which remains unproven for the complete Phase56 path because its exact private development reference vector was unavailable.

| Result | Log loss ↓ | AUROC ↑ | Population and interpretation |
|---|---:|---:|---|
| Phase5 multiscale development reference | ~0.330355 | Not recovered | Early development result |
| Phase12c anatomy model | 0.307616 | 0.938914 | Historical development OOF |
| Phase43 reconstructed comparator | 0.290166 | 0.946474 | Original 1,362-case development OOF; comparator in recent audits |
| Phase42 submission | 0.3254 | 0.9251 | Historical public leaderboard |
| Phase56 submission | 0.3220 | 0.9297 | Historical public leaderboard; retained model/archive |
| Phase68R Phase56 temperature 1.30 | 0.3131 | 0.9297 | Best owner-reported public submission; output-only calibration |
| Phase61 historical-feature blend | 0.285263 | 0.948638 | Attractive development diagnostic; rejected for stability and provenance |
| Phase65B physics-pretrained blend | 0.295285 | 0.944711 | Development; rejected |
| Phase65C domain-adversarial blend | 0.285425 | 0.948274 | Development; failed domain bootstrap gate |
| Phase65D averaged confirmation candidate | 0.286342 | 0.947987 | Repeated-partition development ensemble; failed gate |
| Phase66F DINOv2 volume adaptation | 0.381349 | 0.919615 | Fresh six-fit development OOF; rejected |

Your original development aspiration was **log loss ≤0.24 and AUROC ≥0.95**. From the reconstructed comparator, that requires approximately 0.0502 lower log loss. Recent promising gains were around 0.004–0.005. The participant-provided historical leader reference of 0.2279/0.9676 belongs to the leaderboard population; it must not be treated as a same-population comparator or used to fit model parameters.

## How to read the history

- **Log loss:** measures probability quality; confident mistakes are expensive. Lower is better and this is the competition’s primary metric.
- **AUROC:** measures how well abnormal cases rank above normal cases. Higher is better; good ranking alone does not guarantee good probabilities.
- **Brier score:** another probability-error measure; lower is better.
- **OOF:** out-of-fold predictions from models that did not train on the held-out cases. Here, entire acquisition groups should be excluded from each corresponding model fit.
- **Inner monitor:** development subset used for permitted selection. Its best result is not a held-out outer result.
- **Acquisition group/domain:** a collection inferred from scan-header/geometry characteristics, not a disease class and not necessarily a verified institution or scanner ID.
- **Anchor:** the established prediction path against which new candidates are compared.
- **Gate:** a declared set of requirements for promotion, including overall gain, fold/domain harm and stability.
- **Oracle diagnostic:** a calculation using outcome labels to estimate a ceiling or explain errors. It is not a deployable model or an unbiased performance estimate.
- **Offline integrity:** execution works with bundled assets and the required input/output paths. It does not establish that a new model is more accurate.
- **Numeric parity:** two specified inference paths produce the same probabilities for the same ordered inputs. Running the same archive twice cannot by itself establish parity with a different development model.

## Data, validation and deployment anchor

The historical training audit reported 1,362 examinations: 615 normal and 747 abnormal (prevalence 0.548458). Inputs are reconstructed 3D NIfTI volumes with varying geometry, spacing and intensity distributions. Header/geometry analysis produced approximately 137 prototypes consolidated into 15 acquisition groups.

Recorded group sizes, in group-ID order 0–14:

```text
[39, 456, 145, 255, 76, 7, 49, 208, 32, 4, 35, 10, 32, 9, 5]
```

| Original outer fold | Cases | Comparator log loss | Comparator AUROC |
|---:|---:|---:|---:|
| 0 | 467 | 0.329297 | 0.934867 |
| 1 | 443 | 0.244154 | 0.958548 |
| 2 | 452 | 0.294834 | 0.948633 |

The exact pooled comparator is log loss **0.29016628416289664**, AUROC **0.9464742438589044**, Brier **0.08906692318612011**. Groups 1 and 3 contain 711 cases, or 52.2% of the population; their combined comparator log loss is 0.332306694. They account for roughly 59.8% of total comparator loss. However, smaller groups also matter: Phase65C/D improved the large groups while harming others.

Fit templates, PCA, calibration and feature selection only on the permitted training partition. Lock candidate choices before outer evaluation. Repeated folds help assess sensitivity to partitioning, but reuse the same patients; they do not create independent external validation. After this much sequential experimentation, the original folds are development evidence, not pristine confirmation data. Domain bootstrap bounds are descriptive and do not adjust for the entire historical model search.

The accepted deployment artifact is identified by:

```text
phase56_submission.zip
SHA-256: d012af37307f27a8f92bf95eecc271b7a2153ede6231480670f2e769f4d6c78c
104 archived files; 189,162,616 compressed bytes; 206,351,672 uncompressed bytes
```

It retains the Phase43-derived anchor and a restricted sparse adapter supported for groups 0, 1, 2 and 7. Eleven other known groups abstain from that adapter; unknown protocols retain the Phase12c fallback. The package uses three final adapter checkpoints, their average, and residual cap 0.5. **The Phase43 comparator vector is not an exact development reference for every prediction of this complete Phase56 package.**

## Evidence coverage and missing details

The early history was recorded partly as families of experiments. Phases 8–11, 14–17 and 22–23 cannot be separated into complete individual trial logs. Independent details are absent for Phases 34–35, 37, 44, 46 and 48; Phase54 is represented by a checkpoint inventory entry without its full selection record. These are gaps in the available record, not evidence that nothing happened. No score, hyperparameter or acceptance decision is invented for them.

The following ledger preserves every recoverable family and named subexperiment. A missing numerical result means it was not recovered, not that its metric was zero or identical to another phase.

## Experiment ledger: Phases 1–17

The earliest phases established compliance, robust folds, and the first multiscale 3D anchor. Not every exploratory cell became a numbered promoted artifact, so this section records the experimental families and decisions rather than claiming a metric for every cell.

### Phase 1 — Compliance, integrity, and local EDA

- Established the no-test-learning and no-data-export boundaries.
- Audited file integrity, image shapes, spacing, orientation, intensity ranges, and label balance locally.
- Avoided patient-level output in reports.

### Phase 2 — Acquisition grouping and physical-space preprocessing

- Built header/geometry acquisition groups.
- Used seeded stratified group folds rather than random case folds.
- Developed physical-space center/striatal ROI candidates.
- Made preprocessing robust to variable shape and spacing.

### Phase 3 — Repaired validation and interpretable baselines

- Formed three balanced outer folds across 15 acquisition families.
- Enforced fold-local preprocessing.
- Evaluated compact, interpretable baselines before larger neural models.

### Phase 4 — Compact 3D residual CNN

- Used approximately 160 mm physical crop resampled to `64³`.
- Per-case high-percentile intensity normalization, approximately p99.5.
- Instance normalization rather than batch normalization.
- `BCEWithLogitsLoss`, AdamW, fixed physical augmentations.
- Nested Platt calibration.

### Phase 5 — Multiscale ensemble

- Combined two independently trained 160 mm models with one 192 mm context model.
- Reference OOF log loss was approximately **0.330355**.

### Phase 6 — High-resolution striatal branch

- Tested approximately 128 mm/`80³` high-resolution striatal input.
- Did not improve robust validation and was excluded.

### Phase 7 — First accepted deployment anchor

- Three outer folds.
- Per fold: two independent 160 mm models plus one 192 mm context model.
- Nine total models.
- Context weight and Platt calibration chosen only within inner training/monitor data.
- Averaged calibrated fold probabilities.
- Clinical and high-resolution branches were excluded.
- Included a fixed training-prevalence fallback for unreadable cases.

### Phases 8–11 — Robustness and alternative model families

The record describes Phases 8, 9, 10 and 11 collectively; it does not support assigning each family or score to an individual number. Explored and rejected or deferred:

- Stronger blur/noise and acquisition-style augmentation.
- GroupDRO and KL-based robustness losses.
- Axial slice attention.
- Registered clinical/striatal features with logistic regression and LightGBM.
- Equal-weight and Phase 7 anchored blends.
- Spatial morphology CNNs.
- Multiplanar projection CNNs.

A late Phase 8 candidate was rejected and Phase 7 remained the safer anchor. The broad lesson was that increasing model complexity did not reliably improve acquisition-held performance.

### Phase 12 — Anatomy-token CNN/transformer

- Built 16 anatomy tokens:
  - Eight RAS octants.
  - Four bilateral mean tokens.
  - Four absolute left-right difference tokens.
- Used no external pretraining.
- Phase 12c became a core long-lived anchor component.
- Exact Phase 12c OOF:
  - Log loss **0.3076158111**.
  - AUROC **0.9389144654**.

### Phase 13 — Fold-local masked latent self-supervision

- Investigated label-free masked latent learning on training-fold volumes.
- Kept fitting fold-local to avoid leakage.
- It contributed research evidence but did not independently replace the anchor.

### Phases 14–17 — Consolidation and transport contracts

Only a grouped historical description survives; individual attribution and numerical results for Phases 14, 15, 16 and 17 are unavailable.

- Consolidated multiscale, anatomy-aware, and residual ideas.
- Tightened fold-local state handling, selection, calibration, and deployment parity.
- Established the pattern later formalized in Phase 24: fit-only shadow refits, monitor-locked choices, and one-time outer evaluation.

---

## Experiment ledger: Phases 18–32

### Phase 18 — Registration, templates, and BiMorphNet family

The notebook included:

- Fresh environment and acquisition-held folds.
- Nested fit/monitor split and multiscale cache.
- Four-phenotype fold-local templates.
- Multitemplate rigid registration.
- Registration reliability and diversity diagnostics.
- Registration-free reflection-invariant features.
- Proxy complementarity tests.
- MT-BiMorphNet with auxiliary targets and EMA.
- Paired-view, orientation-preserving BiMorphNet variants.

The family expanded anatomical modeling but did not produce a sufficiently stable promoted replacement.

### Phase 19 — Exact Phase 12c restoration and expert tokens

- Restored the exact Phase 12c model and legacy caches.
- Built a frozen reflection-invariant expert cache.
- Generated cross-fitted expert tokens.
- Performed leakage-safe training and promotion checks.

### Phase 20 — Acquisition-robust expert evidence

- Developed low-dimensional expert evidence.
- Used group-balanced leave-one-group-out residual training.
- Continued to prioritize transport over pooled fit.

### Phase 21 — MedicalNet and clinical-token path

- Tested a MedicalNet ResNet-18 external-weight path.
- Built clinical-token architectures, extraction, heads, and OOF evaluation.
- External-asset licensing and deployment compatibility remained part of the gate.

### Phases 22–23 — Bilateral striatal canonicalization and patches

These methods are recorded jointly. The evidence does not reliably separate which substep belongs to Phase22 versus Phase23.

- Bilateral striatal landmark canonicalization.
- Bilateral patch proxy.
- Siamese initialization from Phase 12.
- Outer-training refits, family OOF, and anchored-delta promotion tests.

### Phase 24 — Formal nested candidate contract

- Defined live-object contracts.
- Fit-only shadow refits.
- Monitor-locked alpha selection.
- One-time outer evaluation after freezing.

This structure became foundational for later staged gates.

### Phase 25 — Exact component reconstruction

- Frozen experts and calibrated-family component caches.
- Numerical-equivalence checks.
- Exact-baseline anchored selection and OOF evaluation.

### Phase 26 — DINOv3 dense multiplanar representation

- ViT-S/16 architecture.
- Dense multiplanar cache.
- Fixed representation and nested linear proxies.
- OOF evaluation.

This was technically promising but DINOv3's custom license remains an unresolved prize-path concern.

### Phase 27 — Structured attentive multi-view probe

- Honest fit/monitor selection.
- Outer evaluation after freezing.
- Did not establish a robust deployable replacement.

### Phase 28 — Confidence-gated DINOv3 correction

- Built a confidence-gated correction.
- Used a monitor gate followed by outer evaluation.
- Confidence routing did not establish stable acquisition transport.

### Phase 29 — Label-free DINOv3 adaptation trajectories

- Explored training-only, label-free adaptation trajectories.
- Conducted outer evaluation.
- No test-time adaptation was permitted.

### Phase 30 — Clinical multiplanar views and deployment packaging

- Twenty-one clinical views:
  - Fifteen thin slabs.
  - Six bilateral views.
- Two-GPU feature extraction.
- Nested linear probes and outer evaluation.
- Exact deployment handoff, runtime rehearsal, smoke tests, and full-ensemble deployment patching.
- Produced source and runtime handoff archives and submission rehearsals.

### Phase 31 — Classical, radiomics, high-resolution, and sequence features

Private artifacts included:

- Classical OOF and selection state.
- High-resolution features.
- Localization diagnostics.
- Radiomics features.
- Sequence features and metadata.

These features later supplied diagnostic complementarity, but their exact provenance/runtime reproducibility was not sufficient for deployment promotion in Phase 61.

### Phase 32 — Bilateral dense and P3CA projections

- Bilateral dense features.
- P3CA projected sequence and summary features.
- Fold states and exact Phase 12c fold probabilities.
- Projected summary shape: `[2, 3, 1362, 768]`.
- Projected sequence shape: `[2, 3, 1362, 84, 48]`.
- Explained variance range: approximately 0.5699–0.6224.
- Labels were not used in this projection audit.

---

## Experiment ledger: Phases 33–56

### Phases 33–39 — Component, subregion, and residual ensemble development

- Phase 33 produced a component model, deployment states, and component/gated OOF predictions.
- Phases 34–35 have no separately recovered experiment descriptions or results.
- Phase 36 consolidated a deployment checkpoint.
- Phase 37 has no separately recovered experiment description or result.
- Phase 38 developed striatal subregion features and diagnostics.
- Phase 39 built a bounded residual correction and deployment package.
- The later exact Phase 43 anchor combined Phase 12c, the Phase 33 component, and Phase 39 residual behavior.

A broader historical system at this stage included a 21-model volumetric ensemble, frozen transformer features, multithreshold radiomics, PCA representations, domain-invariant fusion, and an XGBoost residual correction. Historical OOF improved roughly from 0.3076/0.9389 to 0.2930/0.9456 before exact deployment matching. These figures describe development history, not a newer accepted metric than the Phase 43 anchor.

### Phase 40 — Archive and runtime validation

- Archive-build contract.
- Real smoke contract.
- Final archive contract.
- Submission packaging and parity checks.

### Phase 41 — Ranker and validation partitions

- Private ranker selection.
- Selected development raw OOF.
- Validation-partition metadata and private partition state.

### Phase 42 — Nested/shared alpha selection and submission

- Known protocol folds.
- Nested and shared alpha OOF.
- Private nested selection.
- Runtime source handoff and submission archive.
- Historical public result later recorded:
  - Log loss **0.3254**.
  - AUROC **0.9251**.

### Phase 43 — Reconstructed development comparator

- Exact reconstruction of the deployable probability path.
- Canonical development comparator metric (not an exact vector for the complete later Phase56 sparse adapter):
  - Log loss **0.2901662842**.
  - AUROC **0.9464742439**.

### Phase 44 — Detail not recovered

No separately attributable experiment specification, metric, or decision was recovered. Do not fill this gap by inferring an experiment from its phase number.

### Phase 45 — Fold-local 3D self-supervised reconstruction

- Label-free, fold-local 3D reconstruction pretraining.
- Selected epochs: `[16, 15, 15]`.
- Reconstruction loss improved strongly; examples:
  - Fold 0 total approximately 0.3553 to 0.08167.
  - Fold 2 total approximately 0.3311 to 0.08606.
- Frozen embeddings, linear probes, and integration were evaluated.
- Good reconstruction did not translate into reliable deployable classification gain.

### Phase 46 — Detail not recovered

No separately attributable experiment specification, metric, or decision was recovered.

### Phases 47–50 — Prefix cache, uptake score, and validation

- Phase 47 private prefix cache.
- Phase 48 has no separately recovered experiment description or result.
- Phase 49 patch-uptake score and metadata.
- Patch uptake was historically nonzero mainly in fold 2 and was stopped before broad promotion.
- Phase 50 private validation framework.

### Phase 51 — DINOv3 penultimate-token LoRA

- Repeated adapter training and selection.
- Penultimate-token cache and metadata.
- Confirmation and future-audit checkpoints.
- Did not demonstrate sufficiently reliable promotion.
- DINOv3 licensing remained unresolved for prize eligibility.

### Phase 52 — Repeated partition and support-aware residual gate

- Repeated-partition stability evaluation.
- Support-aware residual gating.
- Final decision remained anchor-only when support was insufficient.

### Phases 53–55 — Fresh runtime and adapter confirmation

- Phase 53 fresh runtime and checkpoint.
- Phase 54 private checkpoint exists in the recorded inventory; its individual selection experiment and result were not recovered.
- Phase 55 confirmed candidate index 1, named `freqfit_only_rank`:
  - Selected epoch 3.
  - Scheduler horizon 8.
  - Confirmed for deployment refit under its restricted support contract.

### Phase 56 — Final refit and package

- Three final checkpoints.
- Complete Phase 43 deployment-matched anchor probability.
- Sparse adapter enabled only for groups `[0, 1, 2, 7]`.
- Abstained for 11 unsupported groups.
- Unknown protocols use Phase 12c anchor only.
- Three-member average.
- Residual cap 0.5.
- Archive and smoke contracts, with later provenance/offline audits described under Phase65A-R through R4. The existence of a contract is not independent proof that all licensing questions or exact Phase56 development parity are resolved.
- Historical public result:
  - Log loss **0.3220**.
  - AUROC **0.9297**.

These are historical public results on a different population from the development folds. Neither result proves performance on the private leaderboard.

---

## Experiment ledger: Phases 57–64

### Phase 57 — Transport reset, architecture, engine and residual optimization

**Substeps:** 57A reset the transport/partition contract; 57B defined the scanner-robust bilateral architecture; 57C tested the training engine; 57D compared anchor-preserving optimization objectives. The two-epoch 57C smoke used 256 internal-fit cases and a 109-case monitor, with monitor log losses approximately 0.242650 and 0.244711. This was an engine check, not an accepted final classifier. The 57D results below concern the optimization pilot.

**Purpose:** Improve the exact anchor using a bounded 3D residual learner without endangering deployment behavior.

**Validation:** The available internal partition contained 786 fit cases and 109 monitor cases, with outer fold isolated; the earlier engine smoke used a smaller fit subset. Multiple loss/objective arms retained epoch-zero identity as a selectable candidate.

**Monitor anchor:**

- Log loss 0.241108.
- AUROC 0.965517.

**Outcome:** All four trained arms selected epoch 0/no update. Trained residuals became nearly constant positive logit shifts. AUROC stayed essentially unchanged while log loss worsened.

**Interpretation:** The learner synthesized an intercept rather than case-specific discrimination. A planned wider program of approximately 72 fits was cancelled. Approximate runtime was 481 seconds with only 1.29 GB peak VRAM, showing that this pilot used little memory. Low memory use alone does not establish whether model capacity was sufficient.

**Decision:** Rejected; retain anchor.

### Phase 58 — Gradient-transport audit and centered pseudo-residual pilot

**Phase 58A audit:**

- Confirmed analytically that Phase 57 objectives exerted positive intercept pressure.
- A fit-optimal intercept near `+0.46` worsened the monitor by approximately 0.0128 log loss.
- Centering `y - p_anchor` within training acquisition groups retained about **97.3%** of target variance.
- Extreme monitor group-pressure values made raw Newton targets and learned group offsets unsafe.

**Phase 58B pilot:**

- Regressed centered first-order pseudo-residuals.
- Two fixed target-shrinkage arms: 0.5 and 1.0.
- Smooth-L1 loss, zero output bias, and global/group-centering penalties.

**Outcome:** Best eligible log-loss gains were only approximately 0.000117 and 0.000179, with no meaningful AUROC improvement. Outputs again collapsed to nearly constant negative corrections.

**Decision:** Rejected. Wider/longer versions and relaxed residual gates were explicitly dropped.

### Phase 59 — OOF failure anatomy and calibration ceiling

**Failure concentration:**

- Top 1% of cases generated **13.2%** of total loss.
- Top 5% generated **43.0%**.
- Top 10% generated **62.9%**.
- Top 20% generated **82.3%**.
- Sixty-two errors at confidence beyond 0.80 generated **40.2%** of total loss.
- Thirty-two errors beyond 0.90 generated **25.6%**.
- Eighteen errors beyond 0.95 generated **16.4%**.

**Calibration:**

- Pooled intercept approximately -0.0494.
- Pooled slope approximately 0.9085.
- Pooled ECE approximately 0.0164.
- Fold-specific shifts oppose each other:
  - Fold 0 overpredicts pathology; intercept approximately -0.374.
  - Fold 2 underpredicts pathology; intercept approximately +0.470.

The pooled calibration therefore hides domain-specific errors.

**Oracle ceilings, diagnostic only:**

| Calibration diagnostic | Log-loss gain | Resulting log loss |
|---|---:|---:|
| Pooled intercept | ~0.000016 | ~0.290150 |
| Pooled temperature | ~0.001177 | ~0.288989 |
| Pooled Platt | **0.001281** | **0.288885** |
| Fold-specific oracle | 0.006834 | 0.283333 |
| Major-group oracle | 0.014757 | ~0.275409 |

Probability clipping worsened performance. Even invalid label-fitted group calibration remained far above 0.24.

**Decision:** Calibration is secondary and cannot close the target gap.

### Phase 60 — Independent bilateral 3D classifier

**Purpose:** Test a genuinely independent classifier rather than another bounded residual.

**Design:**

- All 786 internal-fit cases.
- Bilateral symmetric/asymmetric input channels.
- Per-case normalization and GroupNorm, no BatchNorm.
- Physical scanner augmentations.
- EMA/raw variants.
- Blend with anchor using predeclared alpha values.
- Twenty epochs.
- Approximately 2.20 million parameters.

**Internal monitor result:**

- Selected epoch 11, EMA, alpha 0.4.
- Anchor: 0.241108 log loss / 0.965517 AUROC.
- Candidate: **0.232812 / 0.971602**.
- Gains: **0.008297 log loss** and **0.006085 AUROC**.

**Frozen outer fold 0 result:**

- Anchor: 0.329297 / 0.934867.
- Candidate: **0.344622 / 0.932140**.
- Gains: **-0.015325 log loss** and **-0.002727 AUROC**.

**Decision:** Rejected decisively. This is the clearest example of an attractive internal result failing acquisition transport.

### Phase 61 — Historical complementarity and Phase 31 negative control

**Candidate:** Fixed alpha 0.20 blend using the historical Phase 31 classical family.

**Pooled diagnostic:**

- Candidate log loss 0.285263.
- Candidate AUROC 0.948638.
- Gains: 0.004903 log loss and 0.002164 AUROC.
- Brier gain approximately 0.001283.
- All three original folds improved numerically.

**Why it was rejected:**

- Cluster-bootstrap lower 95% log-loss gain: approximately -0.003699.
- Repeated-partition minimum gain: approximately -0.002266.
- Remaining non-confident cases worsened by about 0.001223.
- Exact provenance and runtime reproducibility were not sufficiently verified.

**Decision:** Numeric gate and deployment-eligibility gate failed. Phase 31 remains a rejected negative control, not an ensemble member.

### Phase 62 — Acquisition-domain validation reset and multitemplate anatomy uncertainty expert

**Validation reset:**

- Five repeated three-fold whole-acquisition-group partitions.
- Leaderboard reference recorded but excluded from fitting and selection.
- Only one predeclared Phase 62B candidate allowed.

**Expert design:**

- Fold-local multitemplate anatomy expert.
- 127 static features.
- Template base dimension 1,000.
- Three templates per class, constructed only from training folds.
- Bounded anchor-plus-anatomy residual with uncertainty shrinkage.

**Expert alone:**

- Log loss 0.373656.
- AUROC 0.912818.

**Routed candidate:**

- Log loss 0.288438.
- AUROC 0.947114.
- Gains: 0.001728 log loss and 0.000640 AUROC.
- Brier gain 0.000289.

**Stability failures:**

- Fold 2 log-loss regret 0.004095.
- Groups 1+3 combined log-loss gain -0.001055.
- Group 3 regret 0.005630.
- Domain-bootstrap lower 95% gain -0.002999.
- The router helped confident errors but worsened the much larger remaining population.

**Decision:** Rejected.

### Phase 63 — Sparse confidence-risk ceiling audit

**Purpose:** Determine whether a label-blind risk score could identify the few catastrophic confident errors and support sparse intervention.

**Failure targets:**

- Classification errors: 167.
- Confident errors: 62.
- Hardest 5%: 69 cases.
- Confident errors contributed 40.19% of total loss.

**Primary label-blind risk score:**

- Error-detection AUROC 0.2854.
- Confident-error-detection AUROC 0.4838.
- Hardest-5%-detection AUROC 0.4543.
- Top 10% risk captured **zero** confident errors and zero total errors.
- Top 20% captured only 4 of 62 confident errors.

**Diagnostic label-fitted oracle:**

- Best log-loss gain only 0.000432.
- AUROC gain 0.000148.
- Harmed fold 0 and group 1.

**Decision:** Sparse router family rejected. No deployable threshold or intervention strength was selected.

### Phase 64 — Group-blocked diffusion/Nyström classifier

**Purpose:** Test whether training-manifold geometry could provide an acquisition-robust independent expert.

**Representation:**

- Case-level positive quantile normalization and log compression.
- Bilateral symmetric and absolute-asymmetric channels.
- Pooled shape `20×20×20`.
- DCT keep shape `8×8×8`.
- 1,024 features.
- Fold-local group-weighted PCA to 64 dimensions.
- Self-tuning neighborhood rank 15.
- Diffusion dimension 24.
- Nyström out-of-sample extension.

**Validation:** Five repeated three-fold whole-group-held fits, 15 total. Test inference would use the training manifold only and process cases independently.

**Fixed candidate:** `sigmoid(0.80 × anchor_logit + 0.20 × diffusion_logit)`.

**Results:**

- Diffusion expert alone: 0.532622 log loss / 0.843219 AUROC.
- Candidate: 0.294492 / 0.944848.
- Gains: **-0.004325 log loss**, **-0.001626 AUROC**.
- Brier gain: -0.001686.
- Repeat log-loss wins: 0 of 5.
- Repeat AUROC wins: 0 of 5.
- Groups 1+3 regret: 0.008574.
- Group 1 regret: 0.004519.
- Group 3 regret: 0.015826.
- Domain-bootstrap lower 95% gain: -0.009655.

**Decision:** Rejected decisively; no full-training refit.

---

## Experiment ledger: Phase65 — deployment repair and the final robustness sequence

### Phase65A — Conservative fallback audit

Audited whether the unknown-protocol fallback could safely change. The numerical gate failed. The recorded decision remained `fallback_numeric_gate_failed_retain_phase12c_unknown_fallback`. No replacement fallback was accepted, and the Phase56 archive was preserved.

### Phase65A-R — Static archive/source/runtime parity audit

The final ZIP passed CRC and path-safety checks, contained root-level `main.py`, and all 32 Python files compiled. All 32 archived Python files matched the staged submission exactly. The runtime handoff contained only seven shared Python files, with a different `main.py`.

The static scanner also found one network-related import, five URL literals, and failed to find its expected input-path marker. It stopped, but had not established an actual download, invalid input path, or broken inference. No dynamic probability replay was available.

**Decision:** Diagnose provenance and runtime behavior; do not rebuild Phase56 to match the handoff.

### Phase65A-R2 — Authoritative provenance and conservative reachability

All 104 archived payload files matched the staged submission. The accepted final contract and build contract recorded the exact submission digest; the accepted smoke contract was linked to the Phase56 lineage.

The handoff mismatch was resolved: its entry point referred to `predict_phase42_safe` and `phase42_runtime`, whereas the final package referred to `predict_phase56_safe` and `phase56_runtime`. It was classified as a non-authoritative pre-final/stale snapshot.

A conservative import graph marked eight of 32 Python files reachable from `main.py`. The only reachable network-capable call was `torch.hub.load`; a DINOv3 URL and `urllib.parse` import were outside that graph. The graph is an approximation, not a proof of dynamic reachability. The initial R2 run had no configured private replay.

**Decision:** Archive lineage passed; inspect Torch Hub arguments and run an offline rehearsal.

### Phase65A-R3 — Local Torch Hub semantics and private offline rehearsal

The single static call in `phase30_accepted_main.py` used `source="local"`, `pretrained=False` and `trust_repo=True`. Its repository path was dynamically constructed, so runtime confirmation was required. The first static-only run stopped because no private rehearsal had been configured.

The subsequent configured rehearsal used eight private development scans:

- Instrumented and clean processes both returned zero and produced valid, finite probabilities in range.
- Instrumented-versus-clean maximum absolute probability difference was **0.0**.
- One observed Torch Hub call used an existing bundled local repository.
- All 34 observed Torch checkpoint paths existed inside the extracted submission.
- Zero socket-network attempts and zero URL-loader calls were observed; instrumentation setup had no recorded failures.
- Instrumented runtime was about 15.92 seconds; clean runtime about 11.95 seconds on that rehearsal environment.
- NIfTI access was observed 40 times and submission-format access once. The output-open hook recorded zero accesses, causing the overall runtime-path gate to fail despite both validated output files.
- Exact Phase56 development reference probabilities were unavailable. Unreadable-case count, branch counts and preprocessing checks were not established by this report; null/unchecked fields must not be restated as zero or passed.

**Decision:** The remaining stop concerned output-path evidence, not an observed internet dependency or probability change.

### Phase65A-R4 — Corrected output-path adjudication

This restart-safe cell read persistent contracts and archive metadata; it did not rerun inference. It verified the R3 contract self-hash and same-archive linkage. It interpreted the earlier output-valid flags according to R3’s exact-output-path, schema, UID-set, row-count and prediction-parsing validation.

On that interpretation, a missing diagnostic open-hook observation did not invalidate the output-file evidence. The corrected offline deployment-integrity gate passed, and Phase65B could proceed. The adjudication depends on what the R3 validator actually checked; it supplies no new inference observation.

**Accepted state:** offline deployment integrity cleared, original archive unchanged, exact Phase56 development numeric parity still unavailable. An offline pass alone does not prove model accuracy or resolve every unmeasured preprocessing/branch diagnostic.

R4 contract digest:

```text
6a9f96fea5a8b739f2af29b00131a37dcb5813d544808f492ea580c2aac3c480
```

### Phase65B — Procedural physics/anatomy pretraining

**Hypothesis:** synthetic uptake and acquisition variation could teach a stronger discriminative prior before real-data training.

Generated 6,400 procedural volumes and used 800 pretraining steps on seven synthetic targets covering abnormality, uptake, bilateral/putamen/caudate relationships, background and blur. Then trained the real-data expert for 12 epochs within each original whole-group fold. The fixed candidate was:

```text
sigmoid(0.75 × anchor_logit + 0.25 × physics_expert_logit)
```

It restored training state from persistent artifacts without prior notebook variables. This was actual challenge-development training performed locally, not merely a synthetic software test.

The physics expert alone scored **0.382698551 / 0.911193827**. Candidate log loss **0.295285500**, AUROC **0.944711094**. All three folds lost on both metrics; the lower domain-bootstrap log-loss gain was **−0.013636118**. Maximum fold log-loss regret was 0.011559578; maximum major-group regret 0.015025979.

**Decision:** Rejected. The simulator/pretraining implementation did not provide the needed transfer; no full-training refit or deployment promotion.

### Phase65C — Class-conditional acquisition adversary and scanner-view consistency

**Hypothesis:** suppress acquisition information in real training data while retaining disease discrimination.

Used a bilateral 3D network, class-conditional domain adversary, two scanner-randomized views, consistency losses and EMA, without BatchNorm or external pretraining. The implementation used 14 epochs, gradient-reversal strength up to 0.15, EMA 0.995, and fixed logit/feature consistency terms. The fixed candidate was:

```text
sigmoid(0.85 × anchor_logit + 0.15 × domain_invariant_expert_logit)
```

The three-fold job resumed successfully from persistent state. The expert alone scored approximately **0.317099 / 0.940631**. The fixed blend scored **0.285424694 / 0.948274398**, a log-loss improvement of approximately 0.004742 over the comparator. All original folds and both major groups improved. Eight of 11 stress domains improved.

However, group12 worsened by approximately 0.011556 log loss, group2 by 0.001627 and group6 by 0.000956. The lower domain-bootstrap gain was **−0.000528327**.

**Decision:** The original gate failed. It was evidence of modest complementarity, not acceptance. A subsequent fixed-candidate confirmation was permitted to study its stability without changing the candidate.

Training logs showing domain-head accuracy around 0.49–0.50 are not sufficient to claim successful invariance. Acquisition classes are imbalanced; an appropriate class-conditional majority/control baseline is required. Chance is not automatically `1 / number_of_groups` in this setting.

### Phase65D — Unchanged-candidate repeated whole-group confirmation

Repeated the same Phase65C candidate over five three-fold acquisition-group partitions: 15 fits, 14 epochs per fit. The implementation restored completed fits and the last completed epoch after interruptions. This was a confirmation attempt, not a newly tuned architecture or blend.

| Measurement | Result |
|---|---:|
| Mean expert alone, log loss / AUROC | 0.315747 / 0.939076 |
| Averaged candidate, log loss / AUROC | **0.286341587 / 0.947987070** |
| Averaged candidate log-loss gain | **0.003824697** |
| Brier gain | ~0.000968 |
| Repeat wins, log loss and AUROC | **5/5 for each** |
| Individual-fold log-loss wins | **14/15** |
| Largest individual-fold log-loss regret | 0.000361571 |
| Groups1+3 combined log-loss gain | **0.004501152** |
| Stress-domain wins | 7/11 |
| Domain-bootstrap lower 95% gain | **−0.001364244** |
| Domain-bootstrap median / upper gain | 0.003382249 / 0.007992539 |
| Group12 log-loss harm | **0.010897878** |
| Group2 log-loss harm | **0.008837112** |
| Group6 log-loss harm | 0.000379618 |

The final invocation reported about 4,313 seconds. Because earlier work was resumed, this is not a reliable total compute cost for all 15 fits.

**Decision:** Failed the declared confirmation gate; retain Phase56. The repeated positive means do not make a negative lower domain bound positive or remove observed group harms. They also do not prove there is no real average benefit—the conclusion is that evidence was insufficient for the specified promotion rule.

Two interpretation limits matter. First, repeated partitions reuse the same patients. Second, the refolded experts reused the original anchor OOF vector: each anchor prediction excluded its original acquisition group, but that does not prove exclusion of all other groups assigned to its new fold. Averaging repeated OOF vectors also evaluates an ensemble, not the performance of one future full-data refit.

### Phase65E — Proposed only

The confirmation code described a future refit/offline replay if Phase65D passed. It did not pass. No executed or accepted Phase65E refit is present in the supplied record.

## Phase66 — One final pretrained volume-adaptation attempt

### Why it was chosen

A targeted literature review through 6 September 2026 compared newer volumetric pretraining, slice-to-volume adaptation, generative diffusion pretraining, intensity adversaries and striatal graph methods with the project’s prior work. It did not exhaust every possible paper or establish a guaranteed route to the target.

Selected a small, fixed **AnyMC3D-inspired DINOv2 volume-adaptation experiment**. The cited paper reports CT/MRI experiments; it is not evidence of a demonstrated DaT-SPECT gain. This implementation is task-specific and is not a reproduction of that paper. Earlier phases already tested DINO features, attention probes and penultimate-token LoRA; Phase66’s distinction is adapting patch and attention projections throughout a frozen backbone and learning volume pooling, using DINOv2. [AnyMC3D manuscript](https://arxiv.org/html/2512.12887v2)

Other reviewed ideas were not executed: 3DINO volumetric pretraining, generative diffusion-autoencoder dopamine-imaging pretraining, AdverIN intensity transforms, max-tree striatal graphs, and MedDINOv3. External-asset permissions, task/domain mismatch, overlap with tested families and implementation cost limited their suitability for one last bounded attempt. Phase64’s graph diffusion must not be confused with generative diffusion. The prior research/setup note contains the comparison and source links.

### Fixed design

| Setting | Declared value |
|---|---|
| Base encoder | Standard non-register DINOv2 ViT-S/14; frozen base weights |
| Training input | Existing 80³ development cache |
| Views | Three axes; 12 three-voxel slabs per axis; resized to 224² |
| Adaptation | Separate axis-specific rank8/scale16 adapters at patch projection and all 12 attention blocks’ QKV/output projections |
| Pooling | Learned slice and axis queries |
| Trainable size | Approximately 690,000 parameters including pooling/head |
| Folds and seeds | Original three whole-group folds × two seeds = six fits |
| Training | 14 epochs per fit; effective batch16; AdamW; final EMA0.99 |
| Learning rates | Adapters 1e-4; pooling/head 3e-4; weight decay0.05 |
| Objective | Ordinary BCE; no class/group reweighting; mild training augmentation |
| Primary candidate | Mean probabilities of the two seeds; no rescue anchor blend/calibrator |
| Budget | Stop launching epochs after 12 hours of checkpointed training time; possible one-epoch overrun |

The gate requires at least **0.01 lower log loss**, **0.002 higher AUROC**, improved Brier score, gains for each seed, bounded fold/domain harm, and a positive lower descriptive domain-bootstrap bound. Attaining the original 0.24/0.95 target is a separate, stronger flag. The 12-hour counter does not include setup, inference or interrupted unsaved work and is not a wall-time guarantee.

The source is standalone in the sense that it reconstructs state from persistent files and does not require earlier cells. It still needs installed NumPy/PyTorch, the existing development artifacts and the correct public pretrained weights. The final Phase66F coordinator assigns one independent fold/seed fit to each available GPU; it does not use DDP.

### What was verified before delivery

The recorded implementation tests used public DINOv2 weights and synthetic data on CPU: strict state-dictionary loading, equality of extracted features to the official backbone, zero-adapter identity, gradient flow into early adapters with the base frozen, case-independent evaluation, interrupted-versus-uninterrupted synthetic resume, completed-fit restoration, stale-checkpoint rejection and synthetic gate pass/fail behavior.

Those software tests do not establish CUDA/Kaggle compatibility, challenge-data performance, raw-NIfTI preprocessing parity or deployment runtime. No successful Phase66 training outcome should be inferred from them.

### Setup failures, recovery, and resolved cause

The first actual Phase66 invocation restored the original folds and comparator, then stopped before training in `pretrained_weights_and_cache` with `expected_unmodified_nonregister_dinov2_vits14_state_dict`. That broad error did not initially reveal which load step failed.

The standalone read-only diagnostic then reported:

```text
restricted load: passed
candidate files: 1
file SHA-256: fae16ba4df8f9b3183b95c248441d15cdb42522ac43067c50d5a64056c67bca8
file bytes: 88,281,415
tensor mapping: one dictionary level inside a wrapper
tensors: 175 expected, 0 missing, 0 unexpected, 0 shape mismatches
all finite: true
```

**Established cause:** the original loader required the tensor dictionary at the file root. The supplied checkpoint has a complete standard non-register ViT-S/14 schema one wrapper level below the root. This explains the original strict-load failure.

**Remaining provenance check:** matching names, shapes and finiteness do not prove that the tensor values are the intended unchanged official pretrained values. The file-byte digest differs from the raw public reference because the container differs; that fact alone neither proves nor disproves tensor equality.

Phase66 v2 therefore accepts a raw dictionary or a dictionary reached through only `state_dict`, `model`, `teacher`, `student`, or `backbone` wrapper keys, up to three levels. It requires exactly one complete 175-tensor mapping, float32 finite tensors, an exact content fingerprint matching the previously tested official public checkpoint, and a strict load into the exact Phase66 model. It does not drop unexpected keys, remove arbitrary prefixes, use `strict=False`, or enable unrestricted pickle loading. The file remains unchanged.

The corrected v2 run subsequently completed and restored four of the six declared fits: all three folds for seed 1 and fold 1 for seed 2. Before seed 2/fold 2 began, the cumulative completed-epoch timer exceeded the declared 12-hour budget and produced `fixed_training_budget_exhausted_no_candidate_promotion` (contract SHA-256 `ddc69e99c903db2acaceeaa0e40418d2a21e2aa6ff367384c9a760ad4b02a87d`). No aggregate Phase66 score exists because two folds remain missing.

Rerunning v2 unchanged could not progress because its budget was cumulative. Raising `training_budget_hours` changed the frozen manifest and prevented reuse of the completed fits. Three attempted continuation coordinators then stopped safely because the exact historical implementation hash could not be recovered. They did not mix checkpoint implementations or alter the accepted Phase56 archive.

### Phase66F — final source-consistent dual-GPU run

Phase66F discarded all historical Phase66 predictions and checkpoints and executed a fresh, implementation-consistent run across all six fits. It launched at most two independent fits concurrently, one per T4 GPU. The checkpoint schema and tensor fingerprint were verified, strict model loading passed, and all six fits completed.

| Measurement | Phase43 comparator | Phase66F candidate | Candidate gain |
|---|---:|---:|---:|
| Log loss | 0.290166 | 0.381349 | -0.091183 |
| AUROC | 0.946474 | 0.919615 | -0.026860 |
| Brier score | 0.089067 | 0.118668 | -0.029601 |

Individual seeds scored 0.421280/0.909760 and 0.400534/0.911385 for log loss/AUROC. All three original folds lost on log loss. Nine of the eleven stress domains lost; only domains 7 and 8 showed small positive gains. The macro-domain bootstrap 95% interval for log-loss gain was `[-0.118300, -0.043209]`. Every declared promotion gate failed.

**Final Phase66 decision:** reject the DINOv2 adaptation, retain Phase56, create no candidate submission archive, and stop this model-search branch. The negative result is useful evidence that this particular slice-to-volume DINOv2 adaptation did not solve the acquisition-generalization problem under the fixed design; it is not a claim that all pretrained transformers are unsuitable for DaT-SPECT.

The official DINOv2 model table distinguishes standard ViT-S/14 from its register variant. [Official pretrained-model table](https://github.com/facebookresearch/dinov2#pretrained-models), [PyTorch serialization documentation](https://docs.pytorch.org/docs/stable/notes/serialization.html)

## Post-Phase66 public output-calibration probes

After the model-search branch closed, four Phase56 output variants were compared on the public leaderboard. The unchanged output scored 0.3220 log loss, an intercept shift of −0.10 worsened it to 0.3256, temperature 1.15 improved it to 0.3143, and temperature 1.30 improved it further to 0.3131. All retained the reported 0.9297 AUROC because positive-temperature logit scaling is monotonic.

The selected public variant applies `sigmoid(logit(p) / 1.30)` independently to each prediction. It does not retrain the model, fit on test cases, or use cross-case statistics, and the Phase56 archive remains unchanged. Because the temperature was selected after observing public leaderboard scores, this result may overfit the public subset and does not establish private-leaderboard improvement. The predeclared decision is to retain temperature 1.30 and stop further leaderboard calibration probes.

## What all these experiments have taught us

1. **Acquisition generalization is the strongest recurring limitation in the observed evidence.** Phase60’s internal-monitor gain reversed on the held-out acquisition fold. Several later families show domain-specific harm. This supports a domain-transport diagnosis; it does not prove that every representation learned only scanner identity or that any particular architecture is universally incapable.
2. **You already improved substantially over the early development system.** Phase12c and the later reconstructed ensemble reduced historical development loss from about 0.3304 to about 0.2902. Exact changes are subject to each phase’s comparator and validation definition.
3. **Pooled calibration hides opposing errors.** The fold-specific shifts found in Phase59 point in different directions. The tested calibration families cannot close the target gap, even under optimistic label-fitted diagnostics.
4. **A few confident errors dominate loss, but the tested risk scores do not locate them reliably.** This explains why a tempting sparse correction did not become a valid deployable solution. It is not proof that those cases are mislabeled.
5. **The adversarial family offers small complementarity, not demonstrated domain invariance.** Phase65C/D improved overall results and the large groups, but group2/group12 harms and the stability bounds prevented promotion.
6. **More validation folds are not more independent patients.** The long search has used the same development population repeatedly. Bootstrap intervals and repeat win counts should not be presented as pristine confirmation or adjusted evidence over the whole research history.
7. **Packaging correctness and predictive quality are separate.** The Torch Hub finding was consistent with local loading; R4 cleared the output-hook issue. Neither explains or fixes the remaining accuracy gap.
8. **A completed negative result is more valuable than an ambiguous interrupted run.** Phase66F resolved the implementation/restart uncertainty and showed that the fixed DINOv2 candidate was materially worse. It should not be extended or promoted.

## Final state and defensible next work

### 1. Preserve the Phase56 deployment anchor

Do not replace Phase56 with Phase65C, Phase65D, or Phase66F. The offline-integrity audit passed and the archive digest is recorded in this history, but the archive itself and its model weights are intentionally excluded from this public repository.

### 2. Stop tuning on the same development population

The same 1,362 cases have supported a long sequential search. Repeated partitions and domain bootstraps are useful diagnostics, but they do not create new independent patients. Further architectural search on the same population is likely to overfit the validation process.

### 3. Add new evidence before another model family

The highest-value next step is an independently defined, legally usable development cohort or blinded expert review of acquisition/preprocessing failure patterns inside an authorized private environment. Predeclare the evaluation and promotion gate before inspecting outcomes.

### 4. Keep research and deployment claims separate

Any future candidate must first pass group-held development gates, then prove raw-NIfTI preprocessing equivalence, case-independent inference, local-weight loading, required output schema, and offline runtime behavior. Check the [official code-execution page](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/) before packaging because runtime specifications may change.

Earlier reviewed external-data/model directions remain proposals, not completed experiments. Their permissions and actual task transfer require verification before use. In particular, the organizer stated that models trained on PPMI data would not be prize eligible; it is not a straightforward fallback data source. [Organizer’s PPMI clarification](https://community.drivendata.org/t/is-ppmi-dua-gated-acceptable-as-external-data-for-prize-eligibility/11472)

## Private reproducibility inputs (not distributed)

Standalone means **no earlier notebook execution or live variables**. It does not mean no installed libraries, authorized data, persistent development artifacts or pretrained weights. The following items are listed so the code contract is understandable; none is included in this repository.

| Artifact family | Role |
|---|---|
| `phase56_submission.zip` and final/build/smoke contracts | Retained deployment anchor and provenance |
| `phase65ar3_local_torch_hub_offline_rehearsal_contract.json` | Recorded private rehearsal evidence |
| `phase65ar4_corrected_offline_gate_adjudication_contract.json` | Accepted correction of the output-hook gate |
| Phase50/52/57 partitions and transport/support contracts | Restore original labels/order, groups and validation constitution |
| Phase32 Phase12c and Phase33/39 private OOF components | Reconstruct the fixed development comparator |
| `phase31_highres_float16.npy` | Existing Phase66 training-volume cache |
| Original training-label CSV | Local supervised training labels; never include in shared reports |
| Standard public DINOv2 ViT-S/14 checkpoint | Phase66 frozen pretrained base |
| Phase66/Phase66F private output directories | Private restart state, model/OOF state and sanitized final contract |

The prior Phase66 setup note specifies these required paths relative to the artifact root:

```text
phase56_submission.zip
phase65ar4_corrected_offline_gate_adjudication_contract.json
phase62_acquisition_domain_validation_contract.json
phase64_group_blocked_diffusion_nystrom_contract.json
phase65a_corrected_deployment_fallback_contract.json
phase50_private_validation/phase50_partitions.npz
phase52_repeated_partition.npz
phase52_support_gate_contract.json
phase32_phase12c_oof_float64.npy
phase33_private_checkpoint/phase33_component_oof_float64.npy
phase39_private_checkpoint/phase39_combined_residual_float64.npy
phase39_private_checkpoint/phase39_oof_float64.npy
phase57_logo_partition.npz
phase57_transport_contract.json
phase31_highres_float16.npy
dinov2_vits14_lvd142m.pth
```

The official raw weight filename `dinov2_vits14_pretrain.pth` is also recognized. The Phase33/39 files can use previously supported root-level alternatives. If artifacts are under read-only `/kaggle/input/...`, set the artifact root accordingly and keep the output root writable. Set the training-label root if discovery needs help. Phase65B/C/D models and prior live notebook objects are not required for Phase66.


## Phase67 — unreviewed work in progress

Phase67 is present in `experiments/` and in the manifest, but it has **not** been through the
review the rest of this archive has. It is published so the record is complete, not because it is
finished. Anyone reading a Phase67 contract should know the following before believing it.

**What it contains.** A shared module, `phase67_common.py`, and five cells: 167A coverage and
cache parity, 167B a nested multi-expert stack, 167C a multiseed domain-invariant expert, 167D
shift-robust calibration, 167E adjudication. It is the first phase whose cells are not standalone.

**Known defects, all still open.**

1. `167A`'s cache-parity recomputation cannot run without an operator-supplied preprocessing
   callable and NIfTI root. Until it does, every cache-trained candidate stays deployment-blocked,
   and 167C and 167E both gate on that verdict.
2. `167E`'s per-seed criterion compares the raw per-seed expert probabilities against the anchor,
   while the candidate is the blend. The comparison is against the wrong quantity and cannot be
   satisfied. The contract now reports the mismatch explicitly rather than emitting an unreachable
   `False`; choosing the correct statistic is a research decision that has not been made.
3. `167A` and `167D` have no tests at all. `167E` is only partially covered.

**Repairs applied during the 20 September 2026 maintenance pass.** `167E`'s cache-parity block now
fails closed rather than open when 167A has not run, and covers `phase67b`, which becomes
cache-trained as soon as 167C's vector is folded into the stack. `167B` now surfaces which experts
in an accepted stack are not deployment-eligible. `167A`'s documented callable contract now matches
its call site.

**Numbering.** There is no Phase67 result in the results table above; the Phase68R entry refers to
output-only temperature scaling of the Phase56 anchor, which was carried out independently of the
Phase67 code and does not depend on it.

## Boundaries and corrections for anyone taking over

- Keep the Phase56 archive digest unchanged unless a separately validated candidate is explicitly being packaged. A stale Phase42 handoff is not its authoritative source.
- Keep private images, labels, UIDs, case-level predictions, embeddings, caches, model checkpoints and raw subprocess logs out of shared reports. Read and train on authorized development data only inside the private execution environment.
- No test-set learning, pseudo-labeling, fitted batch statistics, transductive test graphs or cross-test adaptation. Each test case must be processed independently using fixed training-derived state. [Competition execution rules](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/)
- No full Phase56 development parity claim from Phase43 OOF, Phase12c OOF, Phase42 OOF, or two copies of the same archive.
- Phase60’s 0.232812/0.971602 was the inner monitor; its frozen outer result was worse than the anchor.
- Unmeasured unreadable/branch/preprocessing fields remain unknown. An accepted offline gate does not turn them into zeros or passes.
- Phase65C/D are not accepted ensemble components. Their small gains and domain failures should both be preserved in the record.
- A filename, a URL in source, low GPU memory use, or domain-head accuracy in isolation does not establish the corresponding architectural/network/capacity/invariance claim.
- DINOv3 prize-license compatibility and the previously flagged private-storage policy question remain historically unresolved in this record. Archive/runtime acceptance does not settle those questions. Standard DINOv2’s license is documented separately in the [official repository](https://github.com/facebookresearch/dinov2#license).
- No target score can be promised from the current evidence. A future positive development result still requires deployment verification and does not guarantee a leaderboard score.

## Source trail and verification limits

This handoff uses the earlier Phase1–64 reconstruction; the Phase67 sources as shipped; sanitized Phase65A-R/R2/R3/R4 outputs; the Phase65B, Phase65C and Phase65D reports; saved Phase57–67 source; `../archive/phase_research/PHASE66_RESEARCH.md`; recorded synthetic implementation tests; and the final Phase66F report. The early phase-by-phase primary notebooks are not all available, hence the marked gaps.

No private challenge data were reanalyzed to create this document. The public release was checked using syntax compilation, synthetic tests and a denylist-based privacy scanner. Those checks do not reproduce private development scores; the metrics here are historical aggregate reports.
