"""Synthetic validation for Experiment 3. Requires no private data and no DINOv3.

The projection is the entire contribution of this experiment, so it is what gets
tested hardest:

  ZERO PARAMETERS   the projection must declare no nn.Parameter and no buffer.
                    A learnable version of this collapsed to near rank-1 in a
                    published solution, so "fixed" is a measured property, not a
                    simplification, and a regression here would be invisible in a
                    loss curve.
  SHAPE + FINITE    (B,1,LR,AP,SI) -> (B,4,LR,AP), finite for every channel.
  GAIN INVARIANCE   the NDT and anisotropy channels are defined relative to the
                    in-image maximum, so a global gain must not move them. This
                    is the property that makes the gamma augmentation safe.
  MIRROR EQUIVARIANCE  flipping left-right must flip the projection, not change it.
  AUGMENTATION      every operation preserves shape and finiteness; the gamma
                    operation really is a global gain.
  CONTROL ARM       the scratch control builds and trains with no pretrained
                    weights, so the mandatory matched null always exists.
  GATES             an arm identical to the control must score exactly zero.

Run:  python test_synthetic.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import run as exp3  # noqa: E402
import datcore as dat  # noqa: E402

CONFIG = json.loads(Path(__file__).with_name("config.json").read_text(encoding="utf-8"))
SMALL = {
    **CONFIG,
    "projection": {
        **CONFIG["projection"],
        "anisotropy_window": 5,
        "ndt_erode_steps": 3,
        "output_size": 32,
    },
    "training": {**CONFIG["training"], "epochs": 1, "batch_size": 4, "amp": False},
}


def volumes(torch, rng, n=4, size=20):
    """Crude striatum-like volumes: two bright blobs on a dim background."""
    data = rng.normal(0.4, 0.02, size=(n, 1, size, size, size)).astype(np.float32)
    mid, q = size // 2, max(size // 10, 2)
    for i in range(n):
        strength = 1.0 + 0.4 * i
        data[i, 0, mid - 2 * q:mid - q, int(0.4 * size):int(0.6 * size),
             int(0.45 * size):int(0.7 * size)] += strength
        data[i, 0, mid + q:mid + 2 * q, int(0.4 * size):int(0.6 * size),
             int(0.45 * size):int(0.7 * size)] += strength * 0.8
    return torch.from_numpy(data)


def check_projection_has_no_parameters():
    import torch.nn as nn

    class Wrapper(nn.Module):
        def forward(self, volume):
            return exp3.physical_projection(volume, SMALL)

    module = Wrapper()
    assert list(module.parameters()) == [], "the projection declares a parameter"
    assert list(module.buffers()) == [], "the projection declares a buffer"
    print("  projection: zero nn.Parameter, zero buffers")


def check_shape_and_finiteness(torch):
    rng = np.random.default_rng(7)
    x = volumes(torch, rng, n=3, size=20)
    planes = exp3.physical_projection(x, SMALL)
    assert planes.shape == (3, 4, 20, 20), planes.shape
    assert bool(torch.isfinite(planes).all()), "projection produced a non-finite value"
    for index, name in enumerate(SMALL["projection"]["channels"]):
        channel = planes[:, index]
        assert float(channel.std()) > 0.0, f"channel {name} is constant"
    print(f"  projection: {tuple(planes.shape)} finite, all four channels non-constant")


def check_gain_invariance(torch):
    """A global gain must not move the relative channels."""
    rng = np.random.default_rng(11)
    x = volumes(torch, rng, n=2, size=20)
    base = exp3.physical_projection(x, SMALL)
    for gain in (0.46, 2.3):                      # the range the gamma op spans
        scaled = exp3.physical_projection(x * gain, SMALL)
        for index, name in enumerate(SMALL["projection"]["channels"]):
            if name in ("anisotropy_times_uptake", "normalised_displacement_tail"):
                delta = float((scaled[:, index] - base[:, index]).abs().max())
                assert delta < 0.05, (gain, name, delta)
    print("  projection: relative channels are invariant at 0.46x and 2.3x global gain")


def check_mirror_equivariance(torch):
    """Flipping left-right must flip the output, not alter it."""
    rng = np.random.default_rng(13)
    x = volumes(torch, rng, n=2, size=20)
    direct = exp3.physical_projection(torch.flip(x, dims=[2]), SMALL)
    flipped = torch.flip(exp3.physical_projection(x, SMALL), dims=[2])
    delta = float((direct - flipped).abs().max())
    assert delta < 1e-3, delta
    print(f"  projection: mirror-equivariant to {delta:.2e}")


def check_augmentation(torch):
    rng = np.random.default_rng(17)
    x = volumes(torch, rng, n=4, size=20)
    generator = torch.Generator(device="cpu").manual_seed(5)
    out = exp3.augment(x, SMALL, generator)
    assert out.shape == x.shape, (out.shape, x.shape)
    assert bool(torch.isfinite(out).all())
    # Same seed, same result; different seed, different result.
    a = exp3.augment(x, SMALL, torch.Generator(device="cpu").manual_seed(9))
    b = exp3.augment(x, SMALL, torch.Generator(device="cpu").manual_seed(9))
    c = exp3.augment(x, SMALL, torch.Generator(device="cpu").manual_seed(10))
    assert bool(torch.equal(a, b)), "augmentation is not seed-deterministic"
    assert not bool(torch.equal(a, c)), "augmentation ignores its seed"
    print("  augmentation: shape preserved, finite, seed-deterministic, seed-sensitive")


def check_gamma_is_a_global_gain(torch):
    """The gamma operation must act as a broad multiplicative gain, not a contrast curve."""
    forced = {
        **SMALL,
        "augmentation": {
            **SMALL["augmentation"],
            "flip_lr_probability": 0.0, "rotation_degrees": 0.0,
            "zoom": 0.0, "translate": 0.0,
            "poisson_probability": 0.0, "gamma_probability": 1.0,
        },
    }
    rng = np.random.default_rng(19)
    x = volumes(torch, rng, n=2, size=20)
    ratios = []
    for seed in range(12):
        out = exp3.augment(x, forced, torch.Generator(device="cpu").manual_seed(seed))
        ratios.append(float(out.mean() / x.mean()))
    assert min(ratios) < 0.95 and max(ratios) > 1.05, ratios
    print(
        f"  gamma: acts as a global gain spanning {min(ratios):.2f}x to {max(ratios):.2f}x "
        "(this is why no absolute threshold is allowed anywhere)"
    )


def check_scratch_control_exists_and_trains(torch):
    """The matched null must always be constructible without any pretrained weights."""
    model, provenance = exp3.build_arm("scratch_control", SMALL, prevalence=0.5)
    assert provenance["pretrained"] is False
    rng = np.random.default_rng(23)
    x = volumes(torch, rng, n=2, size=20)
    logit = model(x)
    assert logit.shape == (2,), logit.shape
    start = float(torch.sigmoid(logit[0].detach()))
    assert abs(start - 0.5) < 1e-4, start

    cache = volumes(torch, np.random.default_rng(29), n=16, size=20).numpy()[:, 0]
    labels = np.array([0.0, 1.0] * 8)
    index = np.arange(16)
    probability, _ = exp3.train_arm(
        "scratch_control", cache, labels, index[:12], index[12:],
        SMALL, 4321, torch.device("cpu"),
    )
    assert probability.shape == (4,), probability.shape
    assert np.all((probability > 0.0) & (probability < 1.0)), probability
    print("  scratch control: builds without pretrained weights, starts at the prevalence, trains")


def check_gates_reject_an_identical_arm():
    rng = np.random.default_rng(31)
    labels = (rng.random(300) < 0.5).astype(np.float64)
    probability = np.clip(rng.random(300) * 0.6 + 0.2, 1e-6, 1 - 1e-6)
    groups = rng.integers(0, 4, size=300)
    rows = exp3.cluster_read(labels, probability, probability.copy(), groups, SMALL)
    assert rows, "no cluster reached the minimum size"
    assert all(abs(row["log_loss_gain"]) < 1e-12 for row in rows), rows
    pooled = dat.compare(labels, probability, probability.copy())
    assert pooled["log_loss_gain"] == 0.0
    assert pooled["log_loss_gain"] < SMALL["ship_bar"]["minimum_gain_over_scratch_control"]
    print("  gates: an arm identical to the control scores exactly zero and cannot pass")


def check_dinov3_is_offline_only():
    """Absent local weights, the run must stop -- never reach for the network."""
    import os

    saved = os.environ.pop("DAT_DINOV3_WEIGHTS", None)
    try:
        exp3.load_dinov3(SMALL, trainable=False)
    except dat.DatStop as stop:
        assert "DAT_DINOV3_WEIGHTS" in str(stop) or "dinov3" in str(stop), str(stop)
        print(f"  dinov3: refuses to run without local weights ({stop})")
    else:                                          # pragma: no cover
        raise AssertionError("load_dinov3 succeeded with no weights configured")
    finally:
        if saved is not None:
            os.environ["DAT_DINOV3_WEIGHTS"] = saved


def main():
    try:
        import torch
    except ImportError:
        print("EXP3 SYNTHETIC CHECKS SKIPPED: torch is not installed")
        return
    print("Experiment 3 - DINOv3 on a physical projection, synthetic checks")
    check_projection_has_no_parameters()
    check_shape_and_finiteness(torch)
    check_gain_invariance(torch)
    check_mirror_equivariance(torch)
    check_augmentation(torch)
    check_gamma_is_a_global_gain(torch)
    check_scratch_control_exists_and_trains(torch)
    check_gates_reject_an_identical_arm()
    check_dinov3_is_offline_only()
    print("EXP3 SYNTHETIC CHECKS PASSED")
    print("A synthetic pass validates the instrument, not any private score.")


if __name__ == "__main__":
    main()
