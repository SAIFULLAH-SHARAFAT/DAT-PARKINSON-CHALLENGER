"""Synthetic checks for the Phase67C multi-seed domain-invariant expert.

Runs on CPU with a small synthetic cache and no private artifact. It checks
that the ported architecture is numerically the same object Phase65C trained,
that the adversary and the augmentation behave, and that a fit completes,
checkpoints and resumes.

It does not establish CUDA or Kaggle compatibility, challenge-data
performance, raw-NIfTI preprocessing parity, or deployment runtime. No
successful Phase67C training outcome should be inferred from a pass here.
"""
from __future__ import annotations

import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402
import phase67_cell_167c_multiseed_domain_invariant_expert as expert  # noqa: E402

# Parameter counts recorded for the Phase65C model this port must reproduce.
PHASE65C_ENCODER_PARAMETERS = 2158832
PHASE65C_HEAD_PARAMETERS = 37825
PHASE65C_BACKBONE_PARAMETERS = 2196657
PHASE65C_FULL_MODEL_PARAMETERS = 2274635


def test_architecture_parity():
    architecture = expert.PHASE67C_CONFIG["architecture"]
    backbone = expert.ScannerRobustBilateral3D(architecture, 8.0)
    encoder = sum(p.numel() for p in backbone.encoder.parameters())
    head = sum(p.numel() for p in backbone.head.parameters())
    assert encoder == PHASE65C_ENCODER_PARAMETERS, encoder
    assert head == PHASE65C_HEAD_PARAMETERS, head
    assert encoder + head == PHASE65C_BACKBONE_PARAMETERS
    assert backbone.representation_dimension == 384
    assert backbone.encoder.output_dimension == 192
    model = expert.DomainInvariantBilateral3D(
        architecture, 0.548458, 13, expert.PHASE67C_CONFIG
    )
    total = sum(p.numel() for p in model.parameters())
    assert total == PHASE65C_FULL_MODEL_PARAMETERS, total
    assert total < 6_000_000
    assert not any(
        isinstance(module, torch.nn.BatchNorm3d) for module in model.modules()
    )
    print(
        "  architecture parity: encoder %d, head %d, full model %d - exactly "
        "the Phase65C counts, no BatchNorm" % (encoder, head, total)
    )


def test_bilateral_channels():
    architecture = expert.PHASE67C_CONFIG["architecture"]
    backbone = expert.ScannerRobustBilateral3D(architecture, 8.0)
    volume = torch.rand(2, 1, 24, 24, 24) * 1.5
    original, reflected = backbone.make_bilateral_inputs(volume)
    assert original.shape == (2, 3, 24, 24, 24)
    # Mirroring the input must exchange the two branches exactly, which is the
    # invariance the whole bilateral construction is built on.
    mirrored = torch.flip(volume, dims=[backbone.LEFT_RIGHT_DIMENSION])
    mirrored_original, mirrored_reflected = backbone.make_bilateral_inputs(mirrored)
    assert torch.allclose(original, mirrored_reflected, atol=1e-5)
    assert torch.allclose(reflected, mirrored_original, atol=1e-5)
    # Channels 1 and 2 are the symmetric and antisymmetric parts, so both are
    # unchanged by the mirror.
    assert torch.allclose(original[:, 1:], mirrored_original[:, 1:], atol=1e-5)
    representation = backbone.representation(volume)
    assert representation.shape == (2, 384)
    print("  bilateral channels: mirroring exchanges the two branches exactly")


def test_zero_initialised_head_is_constant_prevalence():
    architecture = expert.PHASE67C_CONFIG["architecture"]
    prevalence = 615.0 / 1362.0
    prevalence = 1.0 - prevalence
    model = expert.DomainInvariantBilateral3D(
        architecture, prevalence, 13, expert.PHASE67C_CONFIG
    )
    volume = torch.rand(3, 1, 24, 24, 24) * 1.5
    with torch.no_grad():
        output = model(volume, torch.tensor([0, 1, 0]), 0.0)
    expected = float(np.log(prevalence) - np.log1p(-prevalence))
    assert float(np.max(np.abs(output["logit"].numpy() - expected))) < 1e-4
    assert output["domain"].shape == (3, 13)
    print(
        "  zero-initialised head starts at logit(prevalence)=%.6f for every "
        "case" % expected
    )


def test_gradient_reversal():
    value = torch.ones(3, 4, requires_grad=True)
    expert.GradientReverse.apply(value, 0.15).sum().backward()
    assert abs(float(value.grad[0, 0]) + 0.15) < 1e-6
    print("  gradient reversal returns -strength * gradient")


def test_augmentation():
    config = expert.PHASE67C_CONFIG["scanner_augmentation"]
    volume = torch.rand(1, 1, 24, 24, 24) * 1.5
    first = expert.augment_one(volume, 12345, config)
    second = expert.augment_one(volume, 12345, config)
    third = expert.augment_one(volume, 999, config)
    assert torch.equal(first, second), "augmentation must be seed-determined"
    assert not torch.equal(first, third)
    assert float(first.min()) >= 0.0 and float(first.max()) <= 1.5
    assert first.shape == volume.shape
    print("  augmentation: seed-determined, range-clamped, shape preserving")


def test_schedule_and_weights():
    config = expert.PHASE67C_CONFIG
    total = 100
    multipliers = [
        expert.learning_rate_multiplier(step, total, config)
        for step in (0, 4, 8, 50, 99)
    ]
    assert multipliers[0] < multipliers[2]
    assert multipliers[2] > multipliers[-1]
    assert multipliers[-1] >= config["minimum_learning_rate_multiplier"] - 1e-9
    groups = np.repeat(
        np.arange(common.GROUP_COUNT), common.EXPECTED_GROUP_SIZES
    )
    weights = expert.group_weight(groups)
    assert abs(float(np.mean(weights)) - 1.0) < 1e-9
    assert weights[groups == 9][0] > weights[groups == 1][0]
    print(
        "  schedule warms up then decays to the floor; group weights are "
        "inverse-root and mean-1"
    )


def build_tiny_state(root, case_count=90, size=24, seed=11):
    rng = np.random.default_rng(seed)
    cache = (rng.random((case_count, size, size, size)) * 1.5).astype(np.float16)
    cache_path = Path(root) / "tiny_cache.npy"
    np.save(cache_path, cache, allow_pickle=False)
    groups = np.repeat(
        np.arange(common.GROUP_COUNT), case_count // common.GROUP_COUNT
    ).astype(np.int64)
    fold_for_group = np.array([0, 1, 2] * 5, dtype=np.int64)
    labels = (rng.random(groups.size) < 0.5).astype(np.int64)
    for fold in range(3):
        mask = fold_for_group[groups] == fold
        if np.unique(labels[~mask]).size < 2:
            labels[np.flatnonzero(~mask)[:2]] = [0, 1]
    return {
        "labels": labels,
        "groups": groups,
        "original_fold": fold_for_group[groups],
        "anchor_probability": np.full(groups.size, 0.5),
        "highres_cache_file": str(cache_path),
        "provenance": {"phase56_submission_sha256": "synthetic"},
    }


def test_fit_and_resume():
    root = Path(tempfile.mkdtemp(prefix="phase67c_synthetic_"))
    try:
        state = build_tiny_state(root)
        config = {
            **expert.PHASE67C_CONFIG,
            "epoch_count": 1,
            "batch_size": 4,
            "worker_count": 0,
            "seeds": [670301],
            "per_fit_budget_hours": 1.0,
        }
        run_hash = "synthetic_run_hash"
        started = time.perf_counter()
        indices, logits, spent = expert.fit_one(
            0, 0, state, config, run_hash, torch.device("cpu"), root
        )
        first_elapsed = time.perf_counter() - started
        assert indices.size == int(np.sum(state["original_fold"] == 0))
        assert logits.shape == indices.shape
        assert np.all(np.isfinite(logits))
        assert float(np.max(np.abs(logits))) <= config["independent_logit_cap"] + 1e-6
        checkpoint = root / "fit_0_0.pt"
        assert checkpoint.is_file()

        # A completed fit must be restored, not retrained, and must return the
        # identical held-out logits.
        started = time.perf_counter()
        indices_two, logits_two, _ = expert.fit_one(
            0, 0, state, config, run_hash, torch.device("cpu"), root
        )
        resume_elapsed = time.perf_counter() - started
        assert np.array_equal(indices, indices_two)
        assert np.array_equal(logits, logits_two)
        assert resume_elapsed < first_elapsed

        # A checkpoint written under a different run hash must be ignored.
        indices_three, logits_three, _ = expert.fit_one(
            0, 0, state, config, "a_different_run_hash", torch.device("cpu"), root
        )
        assert np.array_equal(indices, indices_three)
        print(
            "  fit: %d held-out logits in %.1fs, completed fit restored in "
            "%.2fs, foreign run hash refused and refitted"
            % (logits.size, first_elapsed, resume_elapsed)
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_frozen_manifest_detects_change():
    root = Path(tempfile.mkdtemp(prefix="phase67c_freeze_"))
    try:
        state = build_tiny_state(root)
        config = {**expert.PHASE67C_CONFIG, "artifact_root": str(root)}
        first = expert.build_freeze(state, config)
        assert common.canonical_hash(first) == common.canonical_hash(
            expert.build_freeze(state, config)
        )
        changed = expert.build_freeze(
            state, {**config, "epoch_count": config["epoch_count"] + 1}
        )
        assert common.canonical_hash(first) != common.canonical_hash(changed)
        moved = dict(state)
        moved["labels"] = state["labels"].copy()
        moved["labels"][0] = 1 - moved["labels"][0]
        assert common.canonical_hash(
            expert.build_freeze(moved, config)
        ) != common.canonical_hash(first)
        assert len(expert.implementation_hash()) == 64
        print(
            "  frozen manifest changes with settings, with data and with the "
            "implementation bytecode"
        )
    finally:
        shutil.rmtree(root, ignore_errors=True)


def main():
    print("Phase67C synthetic checks (CPU)")
    test_architecture_parity()
    test_bilateral_channels()
    test_zero_initialised_head_is_constant_prevalence()
    test_gradient_reversal()
    test_augmentation()
    test_schedule_and_weights()
    test_fit_and_resume()
    test_frozen_manifest_detects_change()
    print("PHASE67C SYNTHETIC CHECKS PASSED")
    print(
        "CUDA compatibility, challenge-data performance, raw-NIfTI "
        "preprocessing parity and deployment runtime are NOT tested here."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
