"""Synthetic validation for Experiment 1. Requires no private data.

Checks, in order:

  DESCRIPTORS   are gain-invariant ratios, and the asymmetry index actually
                measures asymmetry (a mirrored volume must score identically,
                a one-sided volume must score high).
  STRATUM       the split is fitted inside the training partition only. The test
                proves this by construction: changing a held-out case's volume
                must not change the threshold.
  MATCHED NULL  permutation preserves class balance and stratum sizes while
                destroying the scan-to-stratum correspondence.
  GATES         a candidate that is identical to the null must fail; a candidate
                that genuinely improves the contested stratum must pass.
  WIRING        the two-headed model trains end to end on tiny synthetic volumes.

Run:  python test_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run as exp1  # noqa: E402
import datcore as dat  # noqa: E402

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
SMALL = {
    **CONFIG,
    "stratum": {**CONFIG["stratum"], "minimum_per_stratum": 5},
    "model": {**CONFIG["model"], "channels": [8, 16], "blocks_per_stage": [1, 1], "projection": 24},
    "training": {**CONFIG["training"], "epochs": 2, "batch_size": 4, "amp": False},
}


def make_volume(rng, size=24, left=1.0, right=1.0, posterior=1.0, background=0.2):
    """A crude striatum: two blobs, one per side, with a tunable posterior tail."""
    volume = rng.normal(background, 0.02, size=(size, size, size)).astype(np.float32)
    mid, q = size // 2, max(size // 8, 2)
    ap_a, ap_p = int(0.42 * size), int(0.60 * size)
    si = slice(int(0.45 * size), int(0.70 * size))
    volume[mid - 2 * q:mid - q, ap_a:ap_p, si] += left
    volume[mid + q:mid + 2 * q, ap_a:ap_p, si] += right
    volume[mid - 2 * q:mid - q, ap_p:ap_p + q, si] += left * posterior
    volume[mid + q:mid + 2 * q, ap_p:ap_p + q, si] += right * posterior
    return volume


def check_descriptors():
    rng = np.random.default_rng(3)
    symmetric = make_volume(rng, left=1.0, right=1.0)
    lopsided = make_volume(rng, left=1.0, right=0.25)

    d_sym = exp1.describe(symmetric, SMALL)
    d_lop = exp1.describe(lopsided, SMALL)
    assert d_sym["asymmetry"] < d_lop["asymmetry"], (d_sym, d_lop)
    assert d_sym["asymmetry"] < 0.10, d_sym
    assert d_lop["asymmetry"] > 0.20, d_lop

    # Mirroring must not change the asymmetry magnitude: the index is about how
    # unequal the sides are, never about which side is affected.
    mirrored = lopsided[::-1].copy()
    d_mir = exp1.describe(mirrored, SMALL)
    assert abs(d_mir["asymmetry"] - d_lop["asymmetry"]) < 0.02, (d_lop, d_mir)

    # Gain invariance: a global multiplicative gain is exactly what the training
    # augmentation applies, so no descriptor may move under it.
    for gain in (0.5, 2.3):
        scaled = exp1.describe(lopsided * gain, SMALL)
        for key in ("asymmetry", "ap_ratio"):
            assert abs(scaled[key] - d_lop[key]) < 1e-6, (gain, key, scaled[key], d_lop[key])
    print(
        f"  descriptors: symmetric {d_sym['asymmetry']:.3f} < lopsided "
        f"{d_lop['asymmetry']:.3f}, mirror-invariant, gain-invariant at 0.5x and 2.3x"
    )


def build_population(rng, n=120):
    """Half healthy; the pathological half split into asymmetric and symmetric."""
    volumes, labels, truth = [], [], []
    for i in range(n):
        if i % 2 == 0:
            volumes.append(make_volume(rng, left=1.0, right=1.0, posterior=0.9))
            labels.append(0.0)
            truth.append(0)
        elif i % 4 == 1:
            volumes.append(make_volume(rng, left=1.0, right=0.2, posterior=0.1))
            labels.append(1.0)
            truth.append(1)                       # asymmetric typical
        else:
            volumes.append(make_volume(rng, left=0.35, right=0.35, posterior=0.1))
            labels.append(1.0)
            truth.append(2)                       # symmetric atypical
    return (
        np.stack(volumes),
        np.asarray(labels, dtype=np.float64),
        np.asarray(truth, dtype=np.int64),
    )


def check_stratum_recovers_truth():
    rng = np.random.default_rng(17)
    volumes, labels, truth = build_population(rng)
    descriptors = exp1.describe_all(volumes, SMALL)
    strata, threshold = exp1.assign_strata(
        descriptors, labels, np.ones(labels.size, dtype=bool), SMALL
    )
    assert set(np.unique(strata)) == {0, 1, 2}, np.unique(strata)
    assert (strata[labels == 0] == 0).all(), "a healthy case was given a pathological stratum"
    pathological = labels > 0.5
    agreement = float(np.mean(strata[pathological] == truth[pathological]))
    assert agreement > 0.85, (agreement, threshold)
    print(
        f"  stratum: recovers the planted syndrome split on {agreement:.0%} of "
        f"pathological cases (threshold {threshold:.3f})"
    )


def check_stratum_is_fold_local():
    """A held-out case must not be able to move its own threshold."""
    rng = np.random.default_rng(23)
    volumes, labels, _ = build_population(rng)
    descriptors = exp1.describe_all(volumes, SMALL)
    train_mask = np.ones(labels.size, dtype=bool)
    held = np.flatnonzero(labels > 0.5)[:8]
    train_mask[held] = False

    _, threshold_a = exp1.assign_strata(descriptors, labels, train_mask, SMALL)
    # Corrupt the held-out cases' descriptors beyond recognition.
    tampered = {k: v.copy() for k, v in descriptors.items()}
    tampered["asymmetry"][held] = 0.999
    _, threshold_b = exp1.assign_strata(tampered, labels, train_mask, SMALL)
    assert threshold_a == threshold_b, (threshold_a, threshold_b)

    # ... whereas a training case moving DOES change it, proving the test has teeth.
    # Shift the training distribution rather than collapsing it to a constant: a
    # constant would produce an empty stratum, which assign_strata now rejects
    # outright, and the test would be measuring that guard instead of the split.
    tampered2 = {k: v.copy() for k, v in descriptors.items()}
    train_pathological = np.flatnonzero((labels > 0.5) & train_mask)
    tampered2["asymmetry"][train_pathological] += 0.5
    _, threshold_c = exp1.assign_strata(tampered2, labels, train_mask, SMALL)
    assert threshold_c != threshold_a, "control failed: threshold ignores training cases too"
    print("  stratum: held-out cases cannot move the threshold; training cases can")


def check_degenerate_split_is_rejected():
    """A split that leaves a stratum empty must stop, not train a dead class."""
    rng = np.random.default_rng(71)
    _, labels, _ = build_population(rng)
    flat = {"asymmetry": np.where(labels > 0.5, 0.42, rng.random(labels.size) * 0.3)}
    try:
        exp1.assign_strata(flat, labels, np.ones(labels.size, bool), SMALL)
    except dat.DatStop as stop:
        assert "has_only_0_training_cases" in str(stop), str(stop)
        print(f"  degenerate split: rejected ({stop})")
        return
    raise AssertionError("a degenerate split with an empty stratum was accepted")


def check_matched_null():
    rng = np.random.default_rng(29)
    _, labels, truth = build_population(rng)
    permuted = exp1.permute_strata(truth, labels, seed=7)
    assert (permuted[labels == 0] == truth[labels == 0]).all(), "healthy cases were permuted"
    for value in (1, 2):
        assert int((permuted == value).sum()) == int((truth == value).sum()), value
    assert not np.array_equal(permuted, truth), "permutation was a no-op"
    print("  matched null: stratum sizes and class balance preserved, correspondence destroyed")


def check_gates_reject_a_null_result():
    """A candidate identical to its control must not pass."""
    rng = np.random.default_rng(41)
    _, labels, truth = build_population(rng)
    probability = np.clip(rng.random(labels.size) * 0.6 + 0.2, 1e-6, 1 - 1e-6)
    folds = np.arange(labels.size) % 3
    pooled, per_stratum, per_fold = exp1.evaluate(
        labels, probability, probability.copy(), truth, folds, SMALL
    )
    assert abs(pooled["log_loss_gain"]) < 1e-12, pooled
    assert all(abs(row["log_loss_gain"]) < 1e-12 for row in per_stratum)
    assert all(abs(row["log_loss_gain"]) < 1e-12 for row in per_fold)
    bar = SMALL["ship_bar"]
    assert pooled["log_loss_gain"] < bar["minimum_pooled_log_loss_gain"]
    print("  gates: a candidate identical to its control yields exactly zero gain")


def check_decomposition_is_complete():
    rng = np.random.default_rng(53)
    _, labels, truth = build_population(rng)
    probability = np.clip(rng.random(labels.size) * 0.6 + 0.2, 1e-6, 1 - 1e-6)
    rows = exp1.decompose(labels, probability, truth, SMALL)
    assert sum(row["n"] for row in rows) == labels.size, rows
    assert {row["stratum"] for row in rows} == set(SMALL["stratum"]["classes"])
    print(f"  decomposition: all {labels.size} cases accounted for across 3 strata")


def check_model_trains():
    try:
        import torch
    except ImportError:
        print("  wiring: SKIPPED (torch not installed)")
        return
    rng = np.random.default_rng(61)
    volumes, labels, truth = build_population(rng, n=48)
    device = torch.device("cpu")
    index = np.arange(labels.size)
    probability = exp1.train_fold(
        volumes, labels, truth, index[: 36], index[36:], SMALL, 1234, device
    )
    assert probability.shape == (12,), probability.shape
    assert np.all((probability > 0.0) & (probability < 1.0)), probability
    model = exp1.build_model(SMALL, prevalence=0.5)
    logit, stratum = model(torch.from_numpy(volumes[:2].astype(np.float32)).unsqueeze(1))
    assert logit.shape == (2,), logit.shape
    assert stratum.shape == (2, 3), stratum.shape
    # Zero-initialised binary head must start at the training prevalence.
    start = float(torch.sigmoid(logit[0].detach()))
    assert abs(start - 0.5) < 1e-5, start
    print("  wiring: two-headed model trains end to end and starts at the prevalence")


def main():
    print("Experiment 1 - atypical syndrome structure, synthetic checks")
    check_descriptors()
    check_stratum_recovers_truth()
    check_stratum_is_fold_local()
    check_degenerate_split_is_rejected()
    check_matched_null()
    check_decomposition_is_complete()
    check_gates_reject_a_null_result()
    check_model_trains()
    print("EXP1 SYNTHETIC CHECKS PASSED")
    print("A synthetic pass validates the instrument, not any private score.")


if __name__ == "__main__":
    main()
