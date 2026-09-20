"""Frozen facts about the challenge dataset.

These are counts and reference scores, not data: no case identifier, label or
prediction appears here. They exist so that a script can refuse to run against a
cache of the wrong length instead of silently producing a different experiment.
"""

from __future__ import annotations

CASE_COUNT = 1362
GROUP_COUNT = 15
NORMAL_COUNT = 615
PATHOLOGIC_COUNT = 747
EXPECTED_GROUP_SIZES = [39, 456, 145, 255, 76, 7, 49, 208, 32, 4, 35, 10, 32, 9, 5]
EXPECTED_ORIGINAL_FOLD_SIZES = [467, 443, 452]
MAJOR_GROUPS = [1, 3]
ANCHOR_LOG_LOSS = 0.29016628416289664
ANCHOR_AUROC = 0.9464742438589044
ANCHOR_MEAN_PROBABILITY = 0.5500570231688339
PROBABILITY_CLIP = 1.0e-7
INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N = 30
