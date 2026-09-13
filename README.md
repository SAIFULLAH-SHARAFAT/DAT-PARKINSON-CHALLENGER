# DaT Parkinson’s Challenge — public solution archive

This repository documents a 66-phase effort for the [DaT Parkinson’s Challenge](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/). The task is binary probability prediction from dopamine-transporter SPECT volumes under substantial acquisition and reconstruction variation.

The release preserves the experiment history, reusable Phase57–66 research code, deployment-integrity audits, synthetic validations, and final negative results. It intentionally excludes challenge data, labels, UIDs, predictions, embeddings, caches, checkpoints, pretrained weights, submission archives, private logs, notebook history, and user-specific storage paths.

## Public leaderboard snapshot

Snapshot supplied by the project owner on **13 September 2026**. Use the linked leaderboard for the current position.

| Rank | Participant | Log loss ↓ | AUROC ↑ |
|---:|---|---:|---:|
| #147 | [**MD_SHAIFULLAH_SHARAFAT**](https://www.drivendata.org/users/MD_SHAIFULLAH_SHARAFAT/ "View MD_SHAIFULLAH_SHARAFAT's profile") | **0.3131** | **0.9297** |

[View the DaT Parkinson’s Challenge leaderboard](https://www.drivendata.org/competitions/311/dat-parkinsons-challenge/leaderboard/)

## Final outcome

The accepted deployment anchor is Phase56. Its archived SHA-256 was:

```text
d012af37307f27a8f92bf95eecc271b7a2153ede6231480670f2e769f4d6c78c
```

The archive itself is not included because it contains weights and other deployment assets. Phase65A-R4 established offline deployment integrity without claiming exact development-to-deployment probability parity.

The final Phase66F DINOv2 experiment completed all six source-consistent fits on two GPUs and was rejected:

| Development metric | Phase43 comparator | Phase66F | Difference in desired direction |
|---|---:|---:|---:|
| Log loss ↓ | 0.290166 | 0.381349 | -0.091183 |
| AUROC ↑ | 0.946474 | 0.919615 | -0.026860 |
| Brier ↓ | 0.089067 | 0.118668 | -0.029601 |

## Repository map

- `docs/EXPERIMENT_HISTORY.md` — complete recoverable Phase1–66 ledger, decisions, metrics, gaps, and interpretation limits.
- `docs/REPRODUCIBILITY.md` — private-environment setup and safe execution instructions.
- `docs/DATA_PRIVACY.md` — what is excluded and why.
- `docs/ASSETS_AND_LICENSES.md` — third-party asset and competition-sharing notes.
- `docs/VALIDATION.md` — checks executed for this release and the PyTorch-only boundary.
- `docs/PUBLIC_RELEASE_CHANGES.md` — path sanitization, transcription repairs, and hash-compatibility warning.
- `docs/PHASE66_RESEARCH.md` — historical literature review and fixed Phase66 design; its recommendation is superseded by the final Phase66F result.
- `experiments/` — all shareable Phase57–66 Python sources available in the recovered workspace, plus synthetic validations.
- `reports/` — sanitized aggregate results only.
- `scripts/verify_release.py` — syntax, manifest, privacy, and forbidden-binary checks.
- `config/kaggle.env.example` — placeholder-only environment configuration.

## Quick verification

Python 3.12 is the reference interpreter. From the repository root:

```bash
python scripts/verify_release.py
python -m compileall -q experiments scripts
```

Synthetic validations that do not require private artifacts can then be run individually, for example:

```bash
python experiments/test_phase65ar4_synthetic.py
python experiments/test_phase65b_static_contract.py
```

Several historical tests are contract-specific scripts rather than a unified package test suite. Do not interpret a synthetic pass as reproduction of a private development score.

## Running a development experiment on Kaggle

1. Accept the competition rules and attach the authorized training data privately.
2. Add the required historical artifacts to writable or read-only Kaggle storage. They are listed in `docs/REPRODUCIBILITY.md` and are not included here.
3. Copy `config/kaggle.env.example` to a private setup cell and replace placeholders locally. Never commit the resulting values.
4. Enable two T4 GPUs only for `phase66f_fresh_dual_gpu_final.py`; the coordinator runs at most one independent fit per GPU and does not use DDP.
5. Run a script from a fresh kernel. The standalone Phase65B/C/D and Phase66F files do not require earlier notebook cells, but they do require their declared persistent artifacts and installed libraries.

Example private setup:

```bash
export DAT_ARTIFACT_ROOT="/kaggle/working"
export DAT_OUTPUT_ROOT="/kaggle/working/phase66f_fresh_dualgpu_private"
export DAT_LABELS_ROOT="/kaggle/input/YOUR_PRIVATE_DATASET_MOUNT"
export DAT_PRIVATE_ROOT="$DAT_LABELS_ROOT"
export DAT_DINOV2_WEIGHTS="/kaggle/working/dinov2_vits14_lvd142m.pth"
python experiments/phase66f_fresh_dual_gpu_final.py
```

## Citation

Use `CITATION.cff` for repository citation. Challenge organizers, data providers, and third-party methods should be cited separately according to their official pages and licenses.
