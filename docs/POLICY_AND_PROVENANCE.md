# Policy and provenance

## What is never committed

- Challenge NIfTI/DICOM files, or renders of individual scans.
- Training or test labels, sample-level metadata, UIDs, filenames, patient rows, or path listings.
- Submission CSVs or case-level predictions.
- Out-of-fold vectors, fold-probability arrays, learned embeddings, feature caches, templates, PCA
  states, or derived voxel arrays.
- Model checkpoints, pretrained weights, Torch caches, optimizer states, or restart directories.
- Raw logs that may carry paths, UIDs, predictions, environment details, or signed URLs.
- Notebook history, hidden checkpoints, credentials, tokens, or local `.env` files.
- User-specific dataset mount names or machine paths.

The competition rules state that participants agree not to transmit, duplicate, publish or
redistribute the data. Labels and identifiers are part of that data; code is not, and publicly
shared code is explicitly permitted.

## What is safe to publish here

- Research and audit source code.
- Synthetic tests that construct no real case identifier or value.
- Aggregate metrics and aggregate group/fold diagnostics from sanitized reports.
- Public competition interface paths and public source links.
- Hashes of public release files, used only for provenance.

## Enforcement

Two layers, deliberately overlapping:

- `.gitignore` blocks medical volume formats, arrays, checkpoints, archives, tabular stores,
  notebooks and credential files by pattern.
- `scripts/verify_release.py` fails closed on forbidden filenames and suffixes, oversized files,
  symlinks, non-UTF-8 text, secret and path patterns, and any disagreement with
  `PUBLIC_RELEASE_MANIFEST.json`.

`scripts/update_manifest.py` runs those per-file checks first and refuses to write a manifest for a
tree that fails them, so the manifest can never be used to bless a leak.

The scanner is a guardrail, not a substitute for review. Deleting a file from the working tree does
not remove it from git history; rotate any exposed credential and rewrite history before pushing.

## Third-party assets

**Competition data.** Not redistributed. Obtain authorized access from DrivenData and follow the
current competition and data terms. The MIT licence in this repository does not apply to the data.

**DINOv2 / DINOv3.** No checkpoint or upstream source is included. Obtain them from the official
repositories, verify the asset and its licence, preserve required notices, and record a local
digest. Experiment 3 loads weights from a local path and never downloads.

**Other reviewed methods.** Papers and repositories referenced in the archived Phase66 research
note are cited for comparison. Their presence is not an endorsement, a bundled dependency, or a
licence grant.

## Provenance of this repository

Derived from a recovered workspace allowlist, not byte-identical to every historical notebook
export. Specifically:

1. Concrete user-owned dataset mount literals were removed; private locations are supplied through
   environment variables.
2. Two merged-line transcription defects in the recovered Phase57C and Phase57D exports were
   repaired.
3. The failed Phase66R/R2/R3 continuations were excluded from the runnable source set; Phase66F
   supersedes them.
4. Notebooks were excluded to avoid cell-output, execution-history, widget and path metadata.
5. All private and third-party binary artefacts were excluded.
6. In September 2026 the release was repaired and restructured: a broken Markdown fence that was
   hiding the Phase66F results table was fixed, a leaderboard rank conflict was reconciled, a
   placeholder citation URL was filled, previously undocumented environment variables were
   documented, temporary-directory literals were made portable, the verifier and `.gitignore` were
   hardened, the manifest walk was made platform-independent, the notebook-host configuration was
   replaced with a local one, and the Phase57–67 research was moved to `archive/phase_research/` so
   the repository leads with work that runs.

Historical implementation hashes embedded in private restart manifests should therefore not be
expected to match these public copies. Do not use a public copy to resume a private checkpoint that
requires the original source hash.

## Medical-data caution

This repository provides research software and aggregate documentation. It makes no clinical
decision-support claim and must not be used for patient care without independent validation,
regulatory review and appropriate governance.
