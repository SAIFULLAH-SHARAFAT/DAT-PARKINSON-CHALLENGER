"""Metrics, hashing and sanitized contract output.

Lifted unchanged from the Phase67 research code so that a number computed by a
new experiment is bit-identical to one computed by the archived phases. The
archive keeps its own copy; this is the supported one.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from .constants import PROBABILITY_CLIP


class DatStop(RuntimeError):
    """Raised when a precondition fails. Carries a stage name, never a path."""


def require(condition, stage):
    if not bool(condition):
        raise DatStop(str(stage))


def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit_contract(path, core):
    """Write a self-hashed contract. The contains_* flags are appended after
    hashing, which is exactly the set contract_core_hash_valid strips."""
    payload = {
        **core,
        "contract_sha256": canonical_hash(core),
        "contains_labels": False,
        "contains_probabilities": False,
        "contains_case_indices": False,
        "contains_voxel_data": False,
        "contains_embeddings": False,
        "contains_uids": False,
    }
    atomic_json(path, payload)
    return payload["contract_sha256"]


def print_sanitized(tag, report):
    print(f"BEGIN SANITIZED_{tag}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"END SANITIZED_{tag}")


def clip_probability(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )


def logit(probability):
    probability = clip_probability(probability)
    return np.log(probability) - np.log1p(-probability)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def auc(labels, score):
    """Mann-Whitney U with mid-rank tie handling. Returns None on a
    single-class slice, so small stress domains cannot raise."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    require(labels.shape == score.shape, "metric_shape_mismatch")
    positive_n = int(np.sum(labels == 1))
    negative_n = int(labels.size - positive_n)
    if positive_n == 0 or negative_n == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    rank = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        rank[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    statistic = float(np.sum(rank[labels == 1]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def loss_vector(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = clip_probability(probability)
    require(labels.shape == probability.shape, "metric_shape_mismatch")
    return -(
        labels * np.log(probability) + (1 - labels) * np.log1p(-probability)
    )


def metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = clip_probability(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(loss_vector(labels, probability))),
        "auroc": auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def compare(labels, baseline, candidate):
    """Positive gain always means the candidate is better, matching every
    earlier phase's *_compare contract."""
    baseline_metrics = metrics(labels, baseline)
    candidate_metrics = metrics(labels, candidate)
    auroc_gain = None
    if (
        baseline_metrics["auroc"] is not None
        and candidate_metrics["auroc"] is not None
    ):
        auroc_gain = float(candidate_metrics["auroc"] - baseline_metrics["auroc"])
    return {
        "n": int(np.asarray(labels).size),
        "baseline_log_loss": baseline_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            baseline_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "baseline_auroc": baseline_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": auroc_gain,
        "baseline_brier": baseline_metrics["brier"],
        "candidate_brier": candidate_metrics["brier"],
        "brier_gain": float(baseline_metrics["brier"] - candidate_metrics["brier"]),
    }
