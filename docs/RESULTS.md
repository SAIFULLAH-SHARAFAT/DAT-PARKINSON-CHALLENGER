# Results

## Public leaderboard

| | |
|---|---|
| Log loss | **0.3131** |
| AUROC | **0.9297** |
| Rank | **#155 of 1,009** |
| Snapshot | 13 September 2026 |

The competition closed on 16 September 2026. The final private leaderboard reordered the field
substantially, and my closing private rank is not yet recorded here.

## Development

| quantity | value |
|---|---|
| Phase43 comparator, out-of-fold log loss | 0.290166 |
| Phase43 comparator, AUROC | 0.946474 |
| Phase43 comparator, Brier | 0.089067 |
| Cases | 1,362 (615 normal, 747 abnormal) |
| Acquisition groups | 15, consolidated from ~137 header/geometry prototypes |
| Outer folds | 3, whole acquisition groups held out together |

Development and leaderboard populations are not directly comparable, and no result here is an
unbiased estimate of private performance.

## The one change that moved the board

Output calibration of the Phase56 anchor. No retraining, no new model.

| probe | temperature | public log loss |
|---|---|---|
| Phase56 unchanged | 1.00 | 0.3220 |
| Phase56 | 1.15 | 0.3143 |
| **Phase56** | **1.30** | **0.3131** |

Improving monotonically, and stopped at 1.30. Two independent top-ten solutions measured that the
optimal slope keeps flattening as the evaluation set hardens — which is why
`experiments/exp2_shift_robust_calibration_bracket/` exists.

## The final modelling experiment

Phase66F: DINOv2 ViT-S/14 volume adaptation, six source-consistent fits across two GPUs, two seeds
over three original group folds.

| metric | Phase43 comparator | Phase66F | difference in the desired direction |
|---|---|---|---|
| Log loss ↓ | 0.290166 | 0.381349 | **−0.091183** |
| AUROC ↑ | 0.946474 | 0.919615 | **−0.026860** |
| Brier ↓ | 0.089067 | 0.118668 | **−0.029601** |

Every promotion gate failed; the macro-domain bootstrap 95% interval was
[−0.118300, −0.043209]. Decision: reject, retain Phase56, stop the model search.

## Deployment anchor

Phase56 remains the accepted deployment archive. It retains the Phase43-derived anchor plus a
restricted sparse adapter supported for four acquisition groups; eleven other known groups abstain
from that adapter and unknown protocols fall back to the Phase12c path. Three final adapter
checkpoints, their average, and a residual cap of 0.5.

Phase65A-R4 established offline deployment integrity. That is separate from exact
development-to-deployment probability parity, which remains unproven because the exact private
development reference vector was unavailable.

Machine-readable versions: [`../reports/results_summary.json`](../reports/results_summary.json) and
[`../reports/phase66f_final_sanitized.json`](../reports/phase66f_final_sanitized.json).
The complete phase-by-phase ledger is in [`EXPERIMENT_HISTORY.md`](EXPERIMENT_HISTORY.md).
