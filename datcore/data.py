"""Loading the training labels and the derived arrays, from local paths only.

The competition ran on hosted notebooks, so the original code searched a mounted
dataset tree for a labels file. This repository runs on a workstation: every
location is an explicit path or an environment variable, nothing is discovered by
globbing, and no cloud layout is assumed.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np

from .constants import CASE_COUNT, NORMAL_COUNT, PATHOLOGIC_COUNT
from .metrics import DatStop, require

LABEL_COLUMNS = ["uid", "is_pathologic"]


def resolve_root(explicit: str | None, variable: str) -> Path:
    """An explicit path wins; otherwise the environment variable; otherwise stop."""
    value = explicit or os.environ.get(variable, "")
    require(bool(value), f"{variable}_not_set")
    root = Path(value).expanduser()
    require(root.exists(), f"{variable}_does_not_exist")
    return root


def load_labels(labels_path: str | Path) -> np.ndarray:
    """Read the training labels and refuse anything that is not the real file.

    The checks are not defensive padding. A silently truncated or reordered label
    file produces a different experiment that still runs to completion and still
    prints a plausible score, which is the most expensive class of mistake
    available here.
    """
    path = Path(labels_path).expanduser()
    require(path.is_file(), "labels_file_not_found")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(bool(rows), "labels_file_is_empty")
    require(list(rows[0].keys()) == LABEL_COLUMNS, "labels_file_has_unexpected_columns")
    require(len(rows) == CASE_COUNT, "labels_file_has_the_wrong_number_of_rows")

    uids = [str(row["uid"]) for row in rows]
    require(len(set(uids)) == CASE_COUNT, "labels_file_contains_duplicate_uids")
    require(all(uids), "labels_file_contains_an_empty_uid")

    try:
        raw = np.asarray([float(row["is_pathologic"]) for row in rows])
    except ValueError as exc:
        raise DatStop("labels_file_contains_a_non_numeric_label") from exc
    require(bool(np.all(np.isin(raw, [0.0, 1.0]))), "labels_must_be_zero_or_one")

    labels = raw.astype(np.float64)
    require(int((labels == 0).sum()) == NORMAL_COUNT, "normal_count_does_not_match")
    require(int((labels == 1).sum()) == PATHOLOGIC_COUNT, "pathologic_count_does_not_match")
    return labels


def load_vector(path: str | Path, name: str, dtype=np.int64) -> np.ndarray:
    """One value per case, in the canonical order, length-checked."""
    resolved = Path(path).expanduser()
    require(resolved.is_file(), f"missing_input_{name}")
    value = np.load(resolved, allow_pickle=False)
    require(value.shape == (CASE_COUNT,), f"{name}_has_the_wrong_length")
    return value.astype(dtype)


def load_cache(path: str | Path):
    """Memory-map the preprocessed volume cache.

    Memory-mapped so a cache larger than RAM still works, and read-only so an
    experiment cannot modify the input it is measuring.
    """
    resolved = Path(path).expanduser()
    require(resolved.is_file(), "missing_input_volume_cache")
    cache = np.load(resolved, mmap_mode="r", allow_pickle=False)
    require(cache.ndim == 4, "volume_cache_must_be_four_dimensional")
    require(cache.shape[0] == CASE_COUNT, "volume_cache_has_the_wrong_number_of_cases")
    return cache
