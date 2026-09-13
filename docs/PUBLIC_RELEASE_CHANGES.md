# Changes made for the public release

The public code is derived from the recovered workspace allowlist. It is not byte-identical to every historical notebook export.

1. Removed concrete user-owned Kaggle dataset mount literals. Private training locations must be supplied through `DAT_LABELS_ROOT` or `DAT_PRIVATE_ROOT`.
2. Made Phase66, Phase66F, and the weight diagnostic accept artifact, output, label, and public-weight locations through environment variables.
3. Repaired two obvious merged-line transcription defects in the recovered Phase57C and Phase57D Python exports:
   - separated `assert len(result) == 4` from `return result`;
   - separated the best-metrics assignment from the best-probability assignment.
4. Excluded the failed Phase66R/R2/R3 continuation implementations from the runnable source set. Their safe-stop history remains documented; Phase66F supersedes them.
5. Excluded notebooks to avoid cell-output, execution-history, widget, and path metadata. Equivalent recovered Python sources are included.
6. Excluded all private and third-party binary artifacts.
7. Updated the owner-reported public leaderboard snapshot on 13 September 2026 and documented the Phase56 temperature-scaling probes using aggregate scores only.

These changes mean historical implementation hashes embedded in private restart manifests should not be expected to match the public copies. Do not use public copies to resume a private checkpoint that requires the original source hash. Start a new run or use the exact private source that created the checkpoint.
