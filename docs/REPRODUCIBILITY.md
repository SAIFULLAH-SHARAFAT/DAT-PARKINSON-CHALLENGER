# Reproducibility and private-environment setup

## Scope

The checked-in Python files are research/audit cells preserved as standalone scripts where possible. “Standalone” means they do not require earlier notebook cells or live variables. It does **not** mean they contain data, fitted artifacts, model weights, or every dependency needed to reproduce private metrics.

The complete public history is in `EXPERIMENT_HISTORY.md`. The recommended deployment state is the retained Phase56 archive; that archive is not distributed here. Phase66F is included to reproduce the implementation and negative-result pathway, not as a recommendation to spend more compute.

## Reference environment

- Python 3.12
- NumPy, pandas, SciPy, scikit-learn and PyTorch from `requirements.txt`
- Kaggle T4 ×2 for Phase66F
- Offline inference for code-execution packaging
- Writable experiment output root
- Authorized training data and labels mounted privately

Exact framework versions should be frozen by the person reproducing a run because Kaggle images and competition limits may change. Record `python --version`, `pip freeze`, GPU model, CUDA version, seed, source commit, and artifact digests in private run metadata.

## Environment variables

Use a private setup cell or shell environment. Do not put the real values in Git:

```bash
export DAT_ARTIFACT_ROOT="/kaggle/working"
export DAT_OUTPUT_ROOT="/kaggle/working/phase66f_fresh_dualgpu_private"
export DAT_LABELS_ROOT="/kaggle/input/YOUR_PRIVATE_DATASET_MOUNT"
export DAT_PRIVATE_ROOT="$DAT_LABELS_ROOT"
export DAT_DINOV2_WEIGHTS="/kaggle/working/dinov2_vits14_lvd142m.pth"
```

The public Phase66/66F sources use these variables. Older research cells may use the generic `/kaggle/working` artifact root directly or their phase-specific override variable; inspect each script’s configuration block before execution.

## Required private artifact families

The following names describe the preserved project state. They are not present in this repository:

| Artifact | Purpose |
|---|---|
| `phase56_submission.zip` plus build/final/smoke contracts | Retained deployment anchor and provenance |
| Phase50/52/57 partition and transport contracts | Original group-held validation constitution |
| Phase32 Phase12c OOF vector | Base development component |
| Phase33 component and Phase39 residual OOF vectors | Reconstruct Phase43 comparator |
| `phase31_highres_float16.npy` | Ordered 80³ training-volume cache used by Phase66F |
| Authorized training-label CSV | Supervised development labels and order |
| DINOv2 ViT-S/14 public checkpoint | Frozen Phase66F foundation weights |
| Phase-specific private output directory | Checkpoints, OOF state, and restart metadata |

The historical scripts also check exact lengths, label balance, fold assignments, hashes, and manifest identities. Do not bypass those checks with `strict=False`, arbitrary prefix deletion, replacement arrays, or altered folds.

## Phase66F on two T4 GPUs

Phase66F launches at most two independent fold/seed workers per wave, assigning one worker to each GPU. This is task-level parallelism, not distributed data parallel training. It preserves the intended per-fit optimizer and effective batch semantics.

From a fresh kernel:

```bash
python experiments/phase66_weights_diagnostic.py
python experiments/phase66f_fresh_dual_gpu_final.py
```

Expected behavior when all private prerequisites match:

1. Restore original folds and comparator.
2. Verify the local DINOv2 tensor mapping and strict model load.
3. Reuse only source-consistent Phase66F checkpoints.
4. Run up to two remaining independent fits concurrently.
5. Aggregate two seeds across three original group folds.
6. Apply the predeclared log-loss, AUROC, Brier, fold, domain and bootstrap gates.

The recorded final result rejected the candidate. Running it again is primarily a reproducibility exercise.

## Offline deployment checks

The Phase65A-R2/R3/R4 scripts audit archive lineage, local Torch Hub semantics, checkpoint locality, absence of observed network attempts, official runtime I/O, and instrumented/clean invariance. The official code-execution layout used in the audit is public competition interface information:

```text
/code_execution/data/niftis/
/code_execution/data/submission_format.csv
/code_execution/submission.csv
```

Do not mistake an offline-runtime pass for a predictive-quality pass or exact development numeric parity. The Phase56 exact development reference vector was unavailable.

## Validation rules retained by this project

- Hold complete acquisition groups out together.
- Fit templates, PCA, feature selection and calibration inside the permitted training partition.
- Freeze choices before outer evaluation.
- Preserve independent case inference.
- Do not use test data for pseudo-labeling, harmonization, batch statistics, transductive graphs, threshold selection, or adaptation.
- Report failures and uncertainty; do not select a rescue blend after seeing outer results.

## Verification before publishing

Run:

```bash
python scripts/verify_release.py
python -m compileall -q experiments scripts
git status --short
```

The verification script rejects medical-image/data/checkpoint files, user-specific paths, secret-like text, unexpected large files, and manifest differences.
