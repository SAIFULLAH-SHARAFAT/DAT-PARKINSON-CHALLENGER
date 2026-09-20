"""Experiment 3 - can a foundation model contribute on a PHYSICAL representation?

HYPOTHESIS
    DINOv3 has never actually been tested on this task. What has been tested, and
    falsified three times, is a foundation model applied to a RAW VOLUME:

        Phase66F, DINOv2 on the volume        -0.091 log loss, -0.027 AUROC
        6th place, DINOv2                     +0.05 log loss
        6th place, eva02_base                 0.2328 on one cluster, 0.2874 on another
        10th place, frozen DINOv2 features    AUROC 0.93 alone, -0.0002 to the ensemble

    Their shared conclusion was that pretrained extractors on this modality are
    ACQUISITION-CONDITIONAL. The physical reason is that a ~9 mm point spread
    function at 2 mm sampling leaves the volume roughly twice oversampled, so
    added capacity has little to consume.

    This experiment changes the input, not the model size. The volume is reduced
    to four physically meaningful 2D channels first, and the foundation model is
    asked to read THAT.

THE ARM THAT MATTERS IS THE CONTROL
    Three arms share one harness and one input: DINOv3 frozen, DINOv3 fine-tuned,
    and a from-scratch network. The scratch control is the matched null. If all
    three land together, the representation did the work and DINOv3 contributed
    nothing -- which is exactly the control that turned a published
    "equivariance works" result into "equivariance contributes nothing".

    Beating your old raw-volume baseline would prove only that the projection
    works. The declared bar is therefore DINOv3 minus the scratch control.

INPUTS (one local directory, from --data-root or $DAT_DATA_ROOT)
    train_labels.csv        uid,is_pathologic
    volume_cache.npy        (cases, 80, 80, 80)
    acquisition_group.npy   acquisition cluster per case
    original_fold.npy       outer fold per case
    Each can be overridden individually on the command line.

OUTPUT
    A sanitized JSON contract in --output-root (or $DAT_OUTPUT_ROOT). No
    case-level values, no identifiers, no paths.

    DAT_DINOV3_WEIGHTS  local DINOv3 checkpoint. Never downloaded.

    python run.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import datcore as dat  # noqa: E402

CONFIG_FILE = Path(__file__).with_name("config.json")


# --------------------------------------------------------------------------
# The projection - fixed, zero learnable parameters
# --------------------------------------------------------------------------

def _ramp(length, device, dtype):
    """Coordinate ramp built by cumsum, never torch.arange(device=...).

    arange with an explicit device bakes that device into a TorchScript trace.
    Using one habit everywhere is cheaper than remembering when it matters.
    """
    import torch

    return torch.ones(length, device=device, dtype=dtype).cumsum(0) - 1.0


def physical_projection(volume, config):
    """(B, 1, LR, AP, SI) -> (B, 4, LR, AP), reducing the superior-inferior axis.

    Channels
        peak    tau-softened log-mean-exp over SI. tau -> 0 gives the mean,
                tau -> inf gives the max, and tau is set per scan from that
                scan's own mean so nothing here is an absolute level.
        mean    arithmetic mean over SI.
        aniso   local structure anisotropy multiplied by relative uptake.
        ndt     normalised displacement tail: an eroded soft indicator of the
                region that stays bright relative to the in-image maximum.

    No learnable parameter appears anywhere in this function.
    """
    import torch
    import torch.nn.functional as F

    spec = config["projection"]
    x = volume.to(torch.float32)
    b, _, lr, ap, si = x.shape

    # Normalise each scan to its OWN reference before anything else. Without this
    # step tau responds to the absolute level, so the global gain that the gamma
    # augmentation applies would move every downstream channel -- the projection
    # would be measuring the camera rather than the patient. The reference is the
    # mean over voxels above a FRACTION of that scan's own high quantile, so it is
    # relative end to end and a global gain cancels exactly.
    flat = x.reshape(b, -1)
    high = torch.quantile(flat, float(spec.get("reference_quantile", 0.999)), dim=1)
    threshold = float(spec.get("reference_fraction", 0.15)) * high
    mask = flat > threshold.unsqueeze(1)
    counts = mask.sum(dim=1).clamp(min=1)
    reference = (flat * mask).sum(dim=1) / counts
    reference = torch.where(counts > 32, reference, flat.mean(dim=1)).clamp(min=1e-6)
    x = x / reference.reshape(b, 1, 1, 1, 1)

    flat = x.reshape(b, -1)
    mu = flat.mean(dim=1)
    tau = (spec["tau_slope"] * mu + spec["tau_intercept"]).clamp(min=spec["tau_minimum"])
    tau = tau.reshape(b, 1, 1)

    xs = x.squeeze(1)
    maximum = xs.amax(dim=-1)
    shifted = (xs - maximum.unsqueeze(-1)) * tau.unsqueeze(-1)
    peak = maximum + (torch.logsumexp(shifted, dim=-1) - float(np.log(si))) / tau
    mean = xs.mean(dim=-1)

    # Local second-moment anisotropy of the peak map.
    window = int(spec["anisotropy_window"])
    pad = window // 2
    pk = peak.unsqueeze(1)
    ones = torch.ones(1, 1, window, window, device=x.device, dtype=x.dtype)
    weight = F.conv2d(pk, ones, padding=pad) + 1e-6
    yy = _ramp(pk.shape[-2], x.device, x.dtype).reshape(1, 1, -1, 1)
    xx = _ramp(pk.shape[-1], x.device, x.dtype).reshape(1, 1, 1, -1)
    ey = F.conv2d(pk * yy, ones, padding=pad) / weight
    ex = F.conv2d(pk * xx, ones, padding=pad) / weight
    vy = F.conv2d(pk * yy * yy, ones, padding=pad) / weight - ey * ey
    vx = F.conv2d(pk * xx * xx, ones, padding=pad) / weight - ex * ex
    vxy = F.conv2d(pk * xx * yy, ones, padding=pad) / weight - ex * ey
    trace = (vy + vx).clamp(min=0.0)
    gap = torch.sqrt(((vy - vx) ** 2 + 4.0 * vxy**2).clamp(min=0.0))
    anisotropy = (gap / (trace + 1e-6)).clamp(0.0, 1.0)
    relative = pk / (pk.amax(dim=(-2, -1), keepdim=True) + 1e-6)
    aniso_channel = (anisotropy * relative).squeeze(1)

    # Normalised displacement tail: relative to the in-image maximum, so a global
    # gain cannot move it. Erosion is a min-pool, implemented as -maxpool(-x).
    level = spec["ndt_fraction"] * pk.amax(dim=(-2, -1), keepdim=True)
    soft = torch.sigmoid(12.0 * (pk - level) / (level + 1e-6))
    eroded = soft
    for _ in range(int(spec["ndt_erode_steps"])):
        eroded = -F.max_pool2d(-eroded, 3, stride=1, padding=1)
        soft = soft + eroded
    ndt = (soft / (soft.amax(dim=(-2, -1), keepdim=True) + 1e-6)).squeeze(1)

    return torch.stack([peak, mean, aniso_channel, ndt], dim=1)


def to_backbone_input(planes, config):
    """Resize to the backbone's grid and expand 4 physical channels to 3 RGB-like.

    The mapping is deliberate and fixed: peak, mean and the anisotropy channel
    form the three inputs, and the NDT channel is added to all three at low
    weight rather than discarded. A learned 4->3 adapter was not used because it
    would put a trainable parameter inside a projection whose whole claim is that
    it has none.
    """
    import torch
    import torch.nn.functional as F

    size = int(config["projection"]["output_size"])
    resized = F.interpolate(planes, size=(size, size), mode="bilinear", align_corners=False)
    rgb = resized[:, :3]
    return rgb + 0.25 * resized[:, 3:4]


# --------------------------------------------------------------------------
# Physics-consistent augmentation
# --------------------------------------------------------------------------

def augment(volume, config, generator):
    import torch
    import torch.nn.functional as F

    spec = config["augmentation"]
    x = volume
    if float(torch.rand(1, generator=generator).item()) < spec["flip_lr_probability"]:
        x = torch.flip(x, dims=[2])          # axis 0 of the cache is left-right

    angle = float(
        (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * spec["rotation_degrees"]
    ) * np.pi / 180.0
    zoom = 1.0 + float(
        (torch.rand(1, generator=generator).item() * 2.0 - 1.0) * spec["zoom"]
    )
    shift = [
        float((torch.rand(1, generator=generator).item() * 2.0 - 1.0) * spec["translate"])
        for _ in range(3)
    ]
    cos, sin = float(np.cos(angle)) / zoom, float(np.sin(angle)) / zoom
    theta = torch.tensor(
        [[cos, -sin, 0.0, shift[0]], [sin, cos, 0.0, shift[1]], [0.0, 0.0, 1.0 / zoom, shift[2]]],
        dtype=x.dtype, device=x.device,
    ).unsqueeze(0).expand(x.shape[0], 3, 4)
    grid = F.affine_grid(theta, list(x.shape), align_corners=False)
    x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")

    if float(torch.rand(1, generator=generator).item()) < spec["poisson_probability"]:
        low, high = spec["poisson_counts"]
        counts = float(torch.rand(1, generator=generator).item()) * (high - low) + low
        scale = counts / (x.mean().clamp(min=1e-6))
        x = torch.poisson((x * scale).clamp(min=0.0), generator=generator) / scale

    if float(torch.rand(1, generator=generator).item()) < spec["gamma_probability"]:
        low, high = spec["gamma_range"]
        gamma = float(torch.rand(1, generator=generator).item()) * (high - low) + low
        # On mean-normalised input this is a 0.46x-2.3x GLOBAL GAIN, which is the
        # whole point: it forces gain invariance and forbids absolute thresholds.
        x = ((x / 12.0).clamp(min=0.0) ** gamma) * 12.0

    return x


# --------------------------------------------------------------------------
# Backbones
# --------------------------------------------------------------------------

def load_dinov3(config, trainable):
    """Load DINOv3 from a LOCAL checkpoint. No download is ever attempted."""
    import torch

    spec = config["backbone"]
    path = os.environ.get(spec["dinov3_weights_env"], "")
    dat.require(bool(path), f"{spec['dinov3_weights_env']}_not_set")
    weights_path = Path(path)
    dat.require(weights_path.is_file(), "dinov3_weights_not_found")

    try:
        import timm
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise dat.DatStop("timm_required_for_dinov3_backbone") from exc

    model = timm.create_model(
        spec["timm_model_name"], pretrained=False, num_classes=0, in_chans=3
    )
    state = torch.load(weights_path, map_location="cpu", weights_only=True)
    for key in ("model", "teacher", "student", "state_dict", "backbone"):
        if isinstance(state, dict) and key in state and isinstance(state[key], dict):
            state = state[key]
    state = {k.replace("backbone.", ""): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    dat.require(
        len(missing) < 0.2 * len(list(model.state_dict())),
        "dinov3_state_dict_did_not_match_the_model",
    )
    if not trainable:
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        model.eval()
    return model, int(spec["embedding_dimension"]), {
        "checkpoint_sha256": dat.sha_file(weights_path),
        "missing_keys": len(missing),
        "unexpected_keys": len(unexpected),
        "downloaded": False,
    }


def build_scratch(config):
    """The matched null: same input, same head, no pretraining."""
    import torch.nn as nn

    width = int(config["backbone"]["embedding_dimension"])
    layers, channels = [], 3
    for out in (32, 64, 128, 256):
        layers += [
            nn.Conv2d(channels, out, 3, stride=2, padding=1, bias=False),
            nn.GroupNorm(min(8, out), out),
            nn.SiLU(inplace=False),
        ]
        channels = out
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(channels, width)]
    return nn.Sequential(*layers), width, {"pretrained": False}


def build_arm(arm_name, config, prevalence):
    import torch
    import torch.nn as nn

    spec = config["arms"][arm_name]
    if spec["backbone"] == "dinov3":
        trunk, width, provenance = load_dinov3(config, bool(spec["trainable"]))
    else:
        trunk, width, provenance = build_scratch(config)

    class ProjectionExpert(nn.Module):
        def __init__(self):
            super().__init__()
            self.trunk = trunk
            self.dropout = nn.Dropout(float(config["training"]["dropout"]))
            self.head = nn.Linear(width, 1)
            nn.init.zeros_(self.head.weight)
            nn.init.constant_(
                self.head.bias, float(np.log(prevalence / (1.0 - prevalence)))
            )

        def forward(self, volume):
            planes = physical_projection(volume, config)
            features = self.trunk(to_backbone_input(planes, config))
            if features.ndim > 2:
                features = features.mean(dim=tuple(range(1, features.ndim - 1)))
            return self.head(self.dropout(features)).squeeze(-1)

    return ProjectionExpert(), provenance


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train_arm(arm_name, cache, labels, train_index, valid_index, config, seed, device):
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    import random as _random
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    spec = config["training"]
    prevalence = float(np.mean(labels[train_index]))
    model, provenance = build_arm(arm_name, config, prevalence)
    model = model.to(device)

    multiplier = float(config["arms"][arm_name].get("learning_rate_multiplier", 1.0))
    trunk_parameters = [p for p in model.trunk.parameters() if p.requires_grad]
    head_parameters = list(model.head.parameters())
    groups = [{"params": head_parameters, "lr": float(spec["learning_rate"])}]
    if trunk_parameters:
        groups.append(
            {"params": trunk_parameters, "lr": float(spec["learning_rate"]) * multiplier}
        )
    optimiser = torch.optim.AdamW(groups, weight_decay=float(spec["weight_decay"]))
    batch = int(spec["batch_size"])
    steps = max(len(train_index) // batch, 1) * int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=[g["lr"] for g in groups], total_steps=steps,
        pct_start=float(spec["warmup_fraction"]),
    )
    use_amp = bool(spec["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(int(spec["epochs"])):
        order = rng.permutation(train_index)
        for start in range(0, len(order) - batch + 1, batch):
            picked = order[start:start + batch]
            volume = torch.from_numpy(
                np.asarray(cache[picked], dtype=np.float32)
            ).unsqueeze(1)
            volume = augment(volume, config, generator).to(device)
            target = torch.from_numpy(labels[picked].astype(np.float32)).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                loss = F.binary_cross_entropy_with_logits(model(volume), target)
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad],
                float(spec["gradient_clip"]),
            )
            previous = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            if scaler.get_scale() >= previous:
                scheduler.step()

    model.eval()
    clamp = float(spec["logit_clamp"])
    out = np.zeros(valid_index.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, valid_index.size, batch):
            picked = valid_index[start:start + batch]
            volume = torch.from_numpy(
                np.asarray(cache[picked], dtype=np.float32)
            ).unsqueeze(1).to(device)
            logit = model(volume).float()
            if spec["flip_tta"]:
                logit = 0.5 * (logit + model(torch.flip(volume, dims=[2])).float())
            logit = logit.clamp(-clamp, clamp)
            out[start:start + picked.size] = torch.sigmoid(logit).cpu().numpy().reshape(-1)
    return np.clip(out, config["probability_clip"], 1.0 - config["probability_clip"]), provenance


def arm_out_of_fold(arm_name, cache, labels, folds, config, seed, device):
    probability = np.zeros(labels.shape[0], dtype=np.float64)
    provenance = None
    for fold in sorted(set(int(f) for f in folds)):
        valid = folds == fold
        probability[valid], provenance = train_arm(
            arm_name, cache, labels,
            np.flatnonzero(~valid), np.flatnonzero(valid),
            config, seed + 100 * fold, device,
        )
    return probability, provenance


def cluster_read(labels, candidate, control, groups, config):
    """Per-acquisition-cluster comparison. The read neither top-ten repo had."""
    rows = []
    minimum = int(config["leave_one_cluster_out"]["minimum_cluster_n"])
    for cluster in sorted(set(int(g) for g in groups)):
        mask = groups == cluster
        if int(mask.sum()) < minimum:
            continue
        rows.append(
            {"cluster": int(cluster), "n": int(mask.sum()),
             **dat.compare(labels[mask], control[mask], candidate[mask])}
        )
    return rows


# --------------------------------------------------------------------------

def load_inputs(args):
    data_root = dat.resolve_root(args.data_root, "DAT_DATA_ROOT")
    labels = dat.load_labels(args.labels or data_root / "train_labels.csv")
    cache = dat.load_cache(args.cache or data_root / "volume_cache.npy")
    groups = dat.load_vector(args.groups or data_root / "acquisition_group.npy",
                             "acquisition_group.npy")
    folds = dat.load_vector(args.folds or data_root / "original_fold.npy", "original_fold.npy")
    return cache, labels, groups, folds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default="",
                        help="directory holding the inputs (default: $DAT_DATA_ROOT)")
    parser.add_argument("--output-root", default=os.environ.get("DAT_OUTPUT_ROOT", ""),
                        help="writable directory for the contract (default: $DAT_OUTPUT_ROOT)")
    parser.add_argument("--labels", default="", help="override: train_labels.csv")
    parser.add_argument("--cache", default="")
    parser.add_argument("--groups", default="")
    parser.add_argument("--folds", default="")
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--arms", default="", help="comma-separated subset of arms, default all"
    )
    args = parser.parse_args(argv)

    import torch

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dat.require(bool(args.output_root), "DAT_OUTPUT_ROOT_not_set")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    started = time.time()
    cache, labels, groups, folds = load_inputs(args)
    names = [a.strip() for a in args.arms.split(",") if a.strip()] or list(config["arms"])
    dat.require("scratch_control" in names, "scratch_control_arm_is_mandatory")

    results, provenance = {}, {}
    for arm in names:
        runs = []
        for seed in config["seeds"]["values"]:
            probability, arm_provenance = arm_out_of_fold(
                arm, cache, labels, folds, config, seed, device
            )
            runs.append(probability)
        provenance[arm] = arm_provenance
        results[arm] = {
            "runs": runs,
            "mean": np.mean(runs, axis=0),
            "seed_log_losses": [dat.metrics(labels, p)["log_loss"] for p in runs],
        }

    control = results["scratch_control"]["mean"]
    bar = config["ship_bar"]
    arm_rows, gates = [], {}
    for arm in names:
        entry = results[arm]
        noise = (
            float(np.std(entry["seed_log_losses"], ddof=1))
            if len(entry["seed_log_losses"]) > 1 else None
        )
        pooled = dat.compare(labels, control, entry["mean"])
        clusters = cluster_read(labels, entry["mean"], control, groups, config)
        worst = max((-row["log_loss_gain"] for row in clusters), default=0.0)
        arm_rows.append({
            "arm": arm,
            "metrics": dat.metrics(labels, entry["mean"]),
            "measured_seed_noise_floor": noise,
            "seed_log_losses": entry["seed_log_losses"],
            "vs_scratch_control": pooled,
            "per_cluster_vs_control": clusters,
            "worst_cluster_regret": float(worst),
            "backbone_provenance": provenance[arm],
        })
        if arm == "scratch_control":
            continue
        seed_signs = [
            dat.compare(labels, control, p)["log_loss_gain"] > 0.0 for p in entry["runs"]
        ]
        gates[arm] = {
            "beats_scratch_control": bool(
                pooled["log_loss_gain"] >= bar["minimum_gain_over_scratch_control"]
            ),
            "auroc_not_worse": bool(
                (pooled["auroc_gain"] or 0.0) >= bar["minimum_auroc_gain_over_scratch_control"]
            ),
            "all_seeds_same_sign": bool(all(seed_signs) or not any(seed_signs))
            if bar["require_all_seeds_same_sign"] else True,
            "gain_exceeds_measured_seed_noise": bool(
                noise is not None
                and pooled["log_loss_gain"] >= 3.0 * noise
            ),
            "no_cluster_collapse": bool(worst <= bar["cluster_collapse_threshold"]),
            "cluster_regret_within_bound": bool(
                worst <= bar["maximum_cluster_log_loss_regret"]
            ),
        }

    winners = [arm for arm, g in gates.items() if all(g.values())]
    status = (
        f"exp3_{winners[0]}_beat_the_scratch_control"
        if winners
        else "exp3_no_pretrained_arm_beat_the_scratch_control_on_this_representation"
    )

    core = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "status": status,
        "arms": arm_rows,
        "gates": gates,
        "declared_ship_bar": bar,
        "projection_has_zero_learnable_parameters": True,
        "interpretation": {
            "the_comparison_is_against_the_scratch_control_not_a_raw_volume_baseline": True,
            "beating_phase66f_would_only_prove_the_projection_works": True,
            "weights_were_loaded_locally_and_never_downloaded": True,
            "test_data_read": False,
            "case_level_values_exported": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    contract_path = output_root / config["contract_file"]
    contract_sha = dat.emit_contract(contract_path, core)
    dat.print_sanitized(
        "EXP3_DINOV3_PROJECTION_EXPERT", {**core, "contract_sha256": contract_sha}
    )
    return 0 if winners else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except dat.DatStop as stop:
        print(f"EXP3 STOP: {stop}", file=sys.stderr)
        raise SystemExit(2)
