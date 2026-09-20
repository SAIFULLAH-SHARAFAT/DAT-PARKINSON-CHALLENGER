"""Shared library for the DaT SPECT experiments.

Small on purpose. Everything here is used by more than one experiment; anything
used by exactly one lives in that experiment's own `run.py`.
"""
from __future__ import annotations

from .constants import (
    ANCHOR_AUROC,
    ANCHOR_LOG_LOSS,
    ANCHOR_MEAN_PROBABILITY,
    CASE_COUNT,
    EXPECTED_GROUP_SIZES,
    EXPECTED_ORIGINAL_FOLD_SIZES,
    GROUP_COUNT,
    MAJOR_GROUPS,
    NORMAL_COUNT,
    PATHOLOGIC_COUNT,
    PROBABILITY_CLIP,
)
from .data import load_cache, load_labels, load_vector, resolve_root
from .metrics import (
    DatStop,
    atomic_json,
    auc,
    canonical_hash,
    clip_probability,
    compare,
    emit_contract,
    logit,
    loss_vector,
    metrics,
    print_sanitized,
    require,
    sha_file,
    sigmoid,
)

__all__ = [
    "ANCHOR_AUROC", "ANCHOR_LOG_LOSS", "ANCHOR_MEAN_PROBABILITY", "CASE_COUNT",
    "EXPECTED_GROUP_SIZES", "EXPECTED_ORIGINAL_FOLD_SIZES", "GROUP_COUNT",
    "MAJOR_GROUPS", "NORMAL_COUNT", "PATHOLOGIC_COUNT", "PROBABILITY_CLIP",
    "DatStop", "atomic_json", "auc", "canonical_hash", "clip_probability",
    "compare", "emit_contract", "logit", "loss_vector", "metrics",
    "print_sanitized", "require", "sha_file", "sigmoid",
    "load_cache", "load_labels", "load_vector", "resolve_root",
]
