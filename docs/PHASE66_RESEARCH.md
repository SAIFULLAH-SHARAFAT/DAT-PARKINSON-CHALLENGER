# Phase66 — one final fixed experiment

Research checked through 6 September 2026. This is a broad, targeted review of relevant methods, not a claim to have exhausted every paper or possible approach.

> **Outcome update (9 September 2026):** Phase66F completed the fixed two-seed/three-fold DINOv2 experiment and failed every promotion gate (0.381349 log loss, 0.919615 AUROC). This document is preserved as the historical pre-experiment research plan. Its recommendation has been executed and is no longer an open next step.

**Original recommendation:** run one DINOv2 volume-adaptation experiment, then end the model search if its fixed gate fails. Do not rerun Phase65D, increase its blend weight, or reinterpret its failed confidence bound as a pass. No method found provides defensible grounds to promise log loss ≤0.24 and AUROC ≥0.95 on this challenge.

## What the existing evidence supports

| Development result | Log loss | AUROC | Interpretation |
|---|---:|---:|---|
| Reconstructed Phase43 comparator used throughout the recent audits | 0.290166 | 0.946474 | Stable numerical comparator; not an exact replay of every Phase56 sparse-adapter prediction |
| Phase65B synthetic-pretraining candidate | 0.295285 | 0.944711 | Rejected |
| Phase65C domain-adversarial candidate | 0.285425 | 0.948274 | Small improvement, failed its robustness gate |
| Phase65D averaged candidate | 0.286342 | 0.947987 | Failed; domain bootstrap lower bound −0.001364 |

Phase65D’s five partitions reuse the same patients. They are not five independent validation datasets. Furthermore, its reassigned expert folds reuse the original anchor OOF vector: each anchor prediction excluded its original group, but that does not establish exclusion of every other group in each newly assigned fold. Averaging across repeated partitions also evaluates an ensemble, not a single refit. These limitations do not erase the observed small gains; they limit the claims those gains support.

The original target requires a development log-loss reduction of about 0.0502 from the comparator. The recent improvements were about 0.004. Development and leaderboard populations are different, so their numerical gap is not a transport estimate.

## Research comparison

| Approach examined | New evidence and relationship to earlier work | Final-attempt decision |
|---|---|---|
| **AnyMC3D-style volume adaptation** | The March 2026 manuscript adapts frozen image backbones and pools slice representations. Its evaluation covers CT/MRI, not DaT-SPECT. Your history includes frozen DINO features, attention probes and penultimate-token LoRA, so this is a related family with a more extensive adaptation boundary. | **Selected**, using existing DINOv2 weights and a small fixed experiment. [Paper](https://arxiv.org/html/2512.12887v2) |
| **3DINO / pretrained volumetric transformers** | 2025 multimodal 3D pretraining offers a different prior from your random 3D networks and earlier MedicalNet trial. The authors’ repository lists CC BY-NC-ND 4.0 for code and weights. | Excluded from this prize-oriented attempt because of the stated license restrictions. [Official repository](https://github.com/AICONSlab/3DINO) |
| **Diffusion-autoencoder self-supervision for dopamine imaging** | Won et al. provide a diffusion-based upstream/downstream framework and FP-CIT-PET materials. This is different from Phase64’s graph diffusion and Nyström features; Phase64 did not test generative diffusion. | Scientifically relevant, but the available evidence does not settle transfer to this SPECT task or asset permissions. Developing a new pretraining pipeline is a poor fit for one bounded final attempt. [Authors’ code](https://github.com/Krying/PD_SSL_ZOO) |
| **AdverIN / adversarial monotonic intensity transforms** | The 2023 preprint, later published in 2025, studies intensity robustness for medical segmentation. Your scanner augmentation and adversarial experiments already address related nuisance variation. | Not selected. My concern is that aggressive intensity transformations can also alter informative uptake relationships in this task; segmentation results do not establish probability-calibration gains here. [Paper](https://arxiv.org/abs/2304.02720) |
| **Max-tree striatal graphs** | A 2025 DaTscan paper combines threshold-tree segmentation, local features and graph learning. Your phases already cover multithreshold radiomics, bilateral anatomy and multitemplate models. | A distinct graph implementation remains possible, but the overlap is substantial and there is no direct evidence that it resolves your acquisition-group failures. [Original publication](https://ieeexplore.ieee.org/document/11226573/) |
| **MedDINOv3 / medical-domain adaptation** | The official implementation adds medical pretraining and multiscale features, reporting CT segmentation benchmarks. | Not selected: domain and task transfer remain uncertain, and your DINOv3 history already includes adaptation. [Official implementation](https://github.com/ricklisz/MedDINOv3) |
| **Test-time adaptation, pseudo-labeling and test-set harmonization** | These could exploit the evaluation population, but they conflict with the independent-sample inference boundary when fitted across test cases. | Excluded. [Competition inference rules](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/page/989/) |
| **External PPMI pretraining** | It is an obvious source of related scans, but the organizer explicitly stated that models trained on PPMI data would not be prize eligible. | Excluded. [Organizer’s clarification, August 2026](https://community.drivendata.org/t/is-ppmi-dua-gated-acceptable-as-external-data-for-prize-eligibility/11472) |

Calibration, confidence routing, larger ensembles, more random 3D capacity, reconstruction SSL, and stronger nuisance augmentation have already received substantial testing in this project. I do not recommend another open-ended search over those families. Their prior failures are evidence about these experiments, not universal impossibility results.

## What this cell actually tests

This is an independent, task-specific implementation inspired by AnyMC3D, **not a reproduction of the paper**. It uses the older DINOv2 ViT-S/14 checkpoint because that file is already in your artifact inventory and the standard DINOv2 code and weights are Apache-2.0 licensed. It does not load the separately licensed Cell-DINO or XRay-DINO assets. [DINOv2 repository](https://github.com/facebookresearch/dinov2#license)

The frozen base receives separate low-rank adapters for each of three slicing axes, at the patch projection and every attention block’s QKV and output projections. Learned queries pool slices and then axes. Training uses ordinary binary cross-entropy, without class or acquisition-group reweighting; no claim is made that this alone guarantees deployment calibration. Its primary prediction is the average probability of two seeds. No anchor blend, fitted calibrator, per-domain correction, epoch selection or architecture search follows evaluation.

| Setting | Fixed value |
|---|---|
| Input | Existing 80×80×80 training cache; three axes, 12 three-voxel slabs per axis |
| Image encoder | DINOv2 ViT-S/14, input resized to 224×224 |
| Adaptation | Rank 8, scale 16; approximately 690,000 trainable parameters including queries/head |
| Folds | Original three complete acquisition-group folds |
| Seeds | Two per fold, six fits total |
| Training | 14 epochs per fit, effective batch size 16 through accumulation |
| Optimizer | AdamW; adapter LR 1e-4, head LR 3e-4; weight decay 0.05 |
| Regularization | Mild intensity/noise/blur and left-right exchange; fixed EMA 0.99 |
| Selection | Final EMA only; outer results reported after all fits finish |
| Budget | Stop launching another epoch after 12 hours of checkpointed training time |

The budget can overrun by one epoch. Interrupted, unsaved work and setup/inference time are not included in that counter. This is a budget limit, not a runtime prediction. The original runner described here used one GPU; the later Phase66F coordinator superseded that operational detail and launched one independent fit per GPU on T4×2.

The main development gate requires at least 0.01 log-loss improvement and 0.002 AUROC improvement over the reconstructed comparator, improved Brier score, improvement for each seed, bounded fold/domain harm and a positive lower descriptive domain-bootstrap bound. **Target attainment is reported separately.** The bootstrap does not correct for the many historical experiments, and this dataset is no longer an untouched confirmation set.

## Setup and run

1. Import `phase66_final_dinov2_volume_adaptation.ipynb` into Kaggle. It contains one executable cell. Alternatively paste the entire `.py` file into one cell.
2. Enable a GPU. A100 is preferable when available. Internet can remain off; the cell does not install packages, download weights or clone repositories. NumPy and a CUDA-capable PyTorch 2.x environment must already be installed.
3. Restore your persistent development artifacts with their existing directory structure. The default artifact root is `/kaggle/working`. If they are attached read-only under `/kaggle/input/...`, change `CONFIG['artifact_root']`; leave `CONFIG['output_root']` in writable working storage.
4. Keep the original training-label CSV attached. If discovery cannot resolve it, set `CONFIG['labels_root']` to the directory containing `train_labels.csv`.
5. Use your existing `dinov2_vits14_lvd142m.pth`. The alternative official filename `dinov2_vits14_pretrain.pth` is also recognized. If the file is elsewhere, set `CONFIG['weights_file']` to its exact local path. The cell expects the original non-register ViT-S/14 state dictionary; it stops on incompatible keys instead of silently accepting a partial load.
6. Run the complete cell. Do not run earlier cells. Do not alter the mathematical settings after inspecting results.

If the public weight file is missing, download the non-register ViT-S/14 backbone from the [official model table](https://github.com/facebookresearch/dinov2#pretrained-models) outside the offline run, then attach it privately as an input artifact. The public checkpoint used in implementation testing had SHA-256 `b938bf1bc15cd2ec0feacfe3a1bb553fe8ea9ca46a7e1d8d00217f29aef60cd9`.

Required persistent artifacts, relative to the artifact root:

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

The Phase33/39 vectors are also accepted at their previously supported root-level locations. The training-label CSV can live in the attached input dataset. Phase65B, C and D scripts, their models, and any live notebook variables are not required.

## Stop and restart

Rerun the same complete cell. Completed fits are restored; unfinished fits resume from the last complete epoch. An interrupted epoch may repeat. Checkpoints include model adapters, EMA, optimizer, AMP scaler and RNG state. Atomic writes and a process lock protect ordinary interruption and accidental concurrent runs. Changed data, code, settings or software/hardware identity cause a sanitized stop instead of silently reusing incompatible checkpoints.

Python cannot recover files that Kaggle deletes at the end of a session. To resume in a **new** session, retain the persistent input artifacts and the entire `phase66_final_private` output directory in your private storage, restore that directory to a writable location, and use the same environment. A kernel restart within an intact filesystem needs no earlier cells.

The cell binds the existing cache by content hash and checks its shape, dtype and finiteness. It relies on the project's established cache ordering; it does not independently reconstruct that cache from raw NIfTI files or prove its preprocessing equivalence.

## Interpret the result

- `phase66_final_attempt_failed_retain_phase56_stop_model_search`: stop the model search and retain Phase56. Do not tune a rescue blend using these same outcomes.
- `phase66_fixed_candidate_development_gate_passed_deployment_parity_pending`: the new expert passed the declared development gate. Verify raw-NIfTI preprocessing, exported-model probability parity and the actual offline runtime before packaging or submitting.
- `development_target_met=true`: the new development predictions reached ≤0.24 log loss and ≥0.95 AUROC. This is **not** a leaderboard claim.
- `phase66_interrupted_or_setup_runtime_stop_retain_phase56`: inspect the sanitized `stage` and `reason`. A fixed-budget exhaustion is a stop rule, not an invitation to change epochs or continue searching.

The report is written to `/kaggle/working/phase66_final_private/phase66_final_attempt_contract.json` by default. Share only the block between `BEGIN SANITIZED_PHASE66_FINAL_DINOV2_VOLUME_ADAPTATION` and `END SANITIZED_PHASE66_FINAL_DINOV2_VOLUME_ADAPTATION`. Keep checkpoints, labels, cache, case-level predictions and paths private.

## Verification and practical limit

The implementation was checked with public DINOv2 weights and synthetic data on CPU: strict pretrained checkpoint loading; exact feature equality against the official backbone; zero-adapter identity; gradients reaching patch and early-attention adapters while base weights stay frozen; case-independent evaluation; exact interrupted-versus-uninterrupted synthetic training; completed-fit restoration; stale-checkpoint rejection; and synthetic pass/fail gate behavior.

CUDA training, actual Kaggle artifact restoration, challenge-data performance, raw-NIfTI preprocessing parity and deployment runtime were not tested here. This deliverable is the final training/validation experiment, not a submission ZIP or a claimed winning model. Phase56 remains unchanged throughout.
