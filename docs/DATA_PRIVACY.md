# Data and privacy policy for this release

## Never commit

- Challenge NIfTI/DICOM files or screenshots of individual scans.
- Training/test labels, sample-level metadata, UIDs, filenames, patient rows, or path listings.
- Submission CSVs or case-level predictions.
- OOF vectors, fold-probability arrays, learned embeddings, feature caches, templates, PCA states, or derived voxel arrays.
- Model checkpoints, pretrained weights, Torch caches, optimizer states, or restart directories.
- Raw logs that may contain paths, UIDs, predictions, environment details, or signed URLs.
- Kaggle notebook history, virtual documents, hidden checkpoints, credentials, tokens, or local `.env` files.
- User-specific dataset mount names or machine paths.

## Safe to publish here

- Generic research and audit source code.
- Synthetic tests that construct no real case identifiers or values.
- Aggregate metrics and aggregate group/fold diagnostics from sanitized reports.
- Public competition interface paths and public source links.
- Hashes of public release files and the retained submission archive, when used only for provenance.

## Before every push

Run `python scripts/verify_release.py`, inspect `git diff --cached`, and confirm that Git LFS is not staging excluded weights or arrays. The scanner is a guardrail, not a substitute for human review.

Useful manual commands:

```bash
git status --short
git diff --cached --stat
git diff --cached
git ls-files
```

If a private file was ever committed, deleting it from the working tree is not enough; it remains in Git history. Rotate any exposed credential, rebuild public history with an appropriate history-rewrite tool, and verify the resulting objects before pushing.

## Medical-data caution

This repository provides research software and aggregate documentation. It contains no clinical decision support claim and must not be used for patient care without independent validation, regulatory review, and appropriate governance.

