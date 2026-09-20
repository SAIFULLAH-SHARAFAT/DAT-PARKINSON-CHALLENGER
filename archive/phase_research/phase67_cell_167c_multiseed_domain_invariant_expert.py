"""Phase67C - five-seed domain-invariant bilateral 3D expert.

Phase65C is the only family in the whole record that improved the pooled
result, both original major groups and eight of eleven stress domains, failing
only on group12, group2 and group6. Phase65D then spent 15 fits confirming it -
but spent them on five REASSIGNED whole-group partitions, which forced it to
reuse the original anchor OOF vector under folds it was never built for. That
is the leak Phase65D had to disclose: each anchor prediction excluded its own
original acquisition group, but not the other groups that a reassigned fold
placed alongside it.

This cell spends the same 15 fits differently: five seeds across the ORIGINAL
three whole-group folds, averaged per fold. Seed averaging buys the same
variance reduction that made Phase65D's largest individual-fold regret only
0.000362, and it carries no reassignment caveat at all, because every fit is
evaluated on exactly the fold structure the anchor was built for. Same cost,
strictly cleaner evidence, and it was never run.

The architecture, the adversary, the two scanner-randomized views, the
consistency losses, the EMA and the augmentation are Phase65C unchanged. The
one deliberate difference is what happens afterwards: this cell does NOT pick a
blend weight. Phase65C fixed alpha at 0.15 and never revisited it even though
the expert alone scores about 0.317 log loss and 0.9406 AUROC - close to the
long-serving Phase12c component - so at 0.15 the expert is barely used. Rather
than replace one arbitrary constant with another, this cell emits the
seed-averaged expert as an ordinary vector for the Phase67B stack to weight
through machinery that is already nested, support-shrunk and checked against a
permutation null. A fixed alpha 0.15 blend is still reported, purely so the
result is directly comparable to the recorded Phase65C and Phase65D numbers.

Outputs, written to the private output directory only:
  phase67_expert_phase67c_multiseed_float64.npy  - seed-averaged expert OOF
  phase67c_multiseed_expert_oof.npz              - per-seed vectors and blend

Copy the .npy to the artifact root and rerun Phase67B to fold it into the
stack, then Phase67D, then Phase67E.

Every fit is checkpointed each epoch and resumes after interruption. The frozen
manifest binds data, code, settings and runtime identity, so edited code
refuses to reuse stale checkpoints instead of silently mixing implementations -
the failure that stranded the original Phase66 at four of six fits.
"""
from __future__ import annotations

import copy
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402

ALGORITHM_VERSION = "phase67c_multiseed_domain_invariant_bilateral3d_v1"

PHASE67C_CONFIG = {
    "schema_version": "phase67_multiseed_domain_invariant_expert_v1",
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get(
        "DAT_OUTPUT_ROOT", "/kaggle/working/phase67_private"
    ),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),
    # Five seeds x three original whole-group folds = 15 fits, the same budget
    # Phase65D spent on five reassigned partitions.
    "seeds": [670301, 670302, 670303, 670304, 670305],
    "fold_count": 3,
    "input_shape": [80, 80, 80],
    "batch_size": 8,
    "worker_count": 2,
    # Phase65C training constants, unchanged.
    "epoch_count": 14,
    "encoder_lr": 6.0e-5,
    "disease_head_lr": 3.0e-4,
    "domain_head_lr": 3.0e-4,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "warmup_fraction": 0.08,
    "minimum_learning_rate_multiplier": 0.05,
    "group_balance_mix": 0.35,
    "logit_view_consistency_weight": 0.10,
    "representation_view_consistency_weight": 0.05,
    "maximum_gradient_reversal": 0.15,
    "gradient_reversal_ramp": 10.0,
    "ema_decay": 0.995,
    "independent_logit_cap": 8.0,
    # Reported only, for comparability with the recorded Phase65C/65D result.
    # Phase67B selects the operative weight through its nested stack.
    "reference_anchor_logit_weight": 0.85,
    "reference_expert_logit_weight": 0.15,
    "per_fit_budget_hours": 3.0,
    "architecture": {
        "channels": [16, 32, 64, 112],
        "blocks_per_stage": [1, 1, 2, 2],
        "projection_dimension": 48,
        "head_hidden_dimension": 96,
    },
    "scanner_augmentation": {
        "intensity_scale": [0.85, 1.15],
        "gamma": [0.80, 1.25],
        "poisson_count": [80.0, 240.0],
        "gaussian_noise_std": [0.0, 0.025],
        "blur_sigma": [0.0, 1.2],
        "resolution_scale": [0.70, 1.0],
        "rotation_degrees": 6.0,
        "translation_voxels": 4.0,
        "reflection_probability": 0.5,
    },
    "contract_file": "phase67c_multiseed_domain_invariant_expert_contract.json",
}


# ---------------------------------------------------------------------------
# Architecture. Ported unchanged from Phase65C so a comparison against the
# recorded Phase65C and Phase65D numbers stays meaningful.
# ---------------------------------------------------------------------------

def group_count_for(channels):
    for candidate in (8, 4, 2):
        if int(channels) % candidate == 0:
            return candidate
    return 1


class ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = group_count_for(channels)
        self.norm_one = nn.GroupNorm(groups, channels)
        self.conv_one = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm_two = nn.GroupNorm(groups, channels)
        self.conv_two = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.activation = nn.SiLU(inplace=False)

    def forward(self, value):
        residual = self.conv_one(self.activation(self.norm_one(value)))
        residual = self.conv_two(self.activation(self.norm_two(residual)))
        return value + residual


class Downsample3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.norm = nn.GroupNorm(group_count_for(in_channels), in_channels)
        self.activation = nn.SiLU(inplace=False)
        self.conv = nn.Conv3d(
            in_channels, out_channels, 3, stride=2, padding=1, bias=False
        )

    def forward(self, value):
        return self.conv(self.activation(self.norm(value)))


class MultiscaleEncoder3D(nn.Module):
    """Four GroupNorm stages; each stage summarized by concatenated average and
    max pooling, projected to a fixed width. No BatchNorm anywhere, so a fitted
    batch statistic can never leak across cases at inference time."""

    def __init__(self, input_channels, channels, blocks_per_stage, projection):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(group_count_for(channels[0]), channels[0]),
            nn.SiLU(inplace=False),
        )
        stages = []
        projections = []
        current = channels[0]
        for index, width in enumerate(channels):
            layers = []
            if index > 0:
                layers.append(Downsample3D(current, width))
                current = width
            for _ in range(blocks_per_stage[index]):
                layers.append(ResidualBlock3D(current))
            stages.append(nn.Sequential(*layers))
            projections.append(
                nn.Sequential(
                    nn.Linear(2 * current, projection),
                    nn.LayerNorm(projection),
                    nn.SiLU(inplace=False),
                )
            )
        self.stages = nn.ModuleList(stages)
        self.projections = nn.ModuleList(projections)
        self.output_dimension = projection * len(channels)

    def forward(self, value):
        value = self.stem(value)
        summaries = []
        for stage, projection in zip(self.stages, self.projections):
            value = stage(value)
            average = F.adaptive_avg_pool3d(value, 1).flatten(1)
            maximum = F.adaptive_max_pool3d(value, 1).flatten(1)
            summaries.append(projection(torch.cat([average, maximum], dim=1)))
        return torch.cat(summaries, dim=1)


class ScannerRobustBilateral3D(nn.Module):
    """Bilateral encoder. The left-right axis is spatial axis 0 of the cache.

    Each case is presented twice, as itself and as its mirror image, through
    one shared encoder; the representation is the symmetric half-sum and the
    absolute asymmetry of the two embeddings. The head is zero-initialized so
    the model starts as an exact constant.
    """

    LEFT_RIGHT_DIMENSION = 2

    def __init__(self, architecture, residual_cap):
        super().__init__()
        self.encoder = MultiscaleEncoder3D(
            3,
            architecture["channels"],
            architecture["blocks_per_stage"],
            architecture["projection_dimension"],
        )
        representation = 2 * self.encoder.output_dimension
        self.representation_dimension = representation
        self.head = nn.Sequential(
            nn.LayerNorm(representation),
            nn.Linear(representation, architecture["head_hidden_dimension"]),
            nn.SiLU(inplace=False),
            nn.Linear(architecture["head_hidden_dimension"], 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)
        self.residual_cap = float(residual_cap)

    @staticmethod
    def normalize_per_case(volume):
        value = torch.clamp(volume.float(), 0.0, 1.5) / 1.5
        mean = value.mean(dim=(2, 3, 4), keepdim=True)
        variance = torch.mean(torch.square(value - mean), dim=(2, 3, 4), keepdim=True)
        return torch.clamp((value - mean) / torch.sqrt(variance + 1.0e-6), -6.0, 6.0)

    def make_bilateral_inputs(self, volume):
        value = self.normalize_per_case(volume)
        reflected = torch.flip(value, dims=[self.LEFT_RIGHT_DIMENSION])
        symmetric = 0.5 * (value + reflected)
        antisymmetric = torch.abs(value - reflected)
        return (
            torch.cat([value, symmetric, antisymmetric], dim=1),
            torch.cat([reflected, symmetric, antisymmetric], dim=1),
        )

    def representation(self, volume):
        common.require(
            volume.ndim == 5 and volume.shape[1] == 1, "phase67c_input_shape"
        )
        original, reflected = self.make_bilateral_inputs(volume)
        encoded = self.encoder(torch.cat([original, reflected], dim=0))
        first, second = encoded.chunk(2, dim=0)
        return torch.cat([0.5 * (first + second), torch.abs(first - second)], dim=1)


class GradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class DomainInvariantBilateral3D(nn.Module):
    """Disease head plus one acquisition adversary per class label.

    Class-conditional adversaries are the point: suppressing acquisition
    information separately within normals and within abnormals removes scanner
    signal without also removing the disease signal that correlates with it.
    """

    def __init__(self, architecture, prevalence, domain_class_count, config):
        super().__init__()
        self.backbone = ScannerRobustBilateral3D(
            architecture, config["independent_logit_cap"]
        )
        representation = self.backbone.representation_dimension
        hidden = max(32, min(96, representation // 2))
        self.domain_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(representation),
                    nn.Linear(representation, hidden),
                    nn.SiLU(inplace=False),
                    nn.Linear(hidden, int(domain_class_count)),
                )
                for _ in range(2)
            ]
        )
        cap = float(config["independent_logit_cap"])
        prevalence_logit = float(
            np.log(float(prevalence)) - np.log1p(-float(prevalence))
        )
        raw = float(np.arctanh(np.clip(prevalence_logit / cap, -0.999, 0.999)))
        with torch.no_grad():
            self.backbone.head[-1].bias.fill_(raw)
        self.logit_cap = cap

    def disease(self, representation):
        raw = self.backbone.head(representation).reshape(-1).float()
        return self.logit_cap * torch.tanh(raw)

    def domain_logits(self, representation, labels, reversal_strength):
        reversed_representation = GradientReverse.apply(
            representation, float(reversal_strength)
        )
        stacked = torch.stack(
            [head(reversed_representation) for head in self.domain_heads], dim=1
        )
        row = torch.arange(
            labels.numel(), device=labels.device, dtype=torch.long
        )
        return stacked[row, labels.long().reshape(-1)]

    def forward(self, volume, labels, reversal_strength):
        representation = self.backbone.representation(volume)
        return {
            "representation": representation,
            "logit": self.disease(representation),
            "domain": self.domain_logits(representation, labels, reversal_strength),
        }


# ---------------------------------------------------------------------------
# Physical scanner augmentation. Ported unchanged from Phase57B / Phase65C.
# ---------------------------------------------------------------------------

def uniform_cpu(generator, lower, upper):
    return float(
        torch.rand(1, generator=generator).item() * (float(upper) - float(lower))
        + float(lower)
    )


def gaussian_kernel_1d(sigma, device, dtype):
    radius = 2
    positions = torch.arange(
        -radius, radius + 1, device=device, dtype=dtype
    )
    kernel = torch.exp(-0.5 * torch.square(positions / max(float(sigma), 1.0e-6)))
    return kernel / kernel.sum()


def anisotropic_blur(volume, sigmas):
    value = volume
    for axis, sigma in enumerate(sigmas):
        if float(sigma) <= 0.02:
            continue
        kernel = gaussian_kernel_1d(sigma, value.device, value.dtype)
        shape = [1, 1, 1, 1, 1]
        shape[2 + axis] = kernel.numel()
        padding = [0, 0, 0]
        padding[axis] = kernel.numel() // 2
        value = F.conv3d(
            value,
            kernel.reshape(shape),
            padding=(padding[0], padding[1], padding[2]),
        )
    return value


def rotation_matrix(angles):
    cos = [math.cos(math.radians(angle)) for angle in angles]
    sin = [math.sin(math.radians(angle)) for angle in angles]
    rx = torch.tensor(
        [[1, 0, 0], [0, cos[0], -sin[0]], [0, sin[0], cos[0]]], dtype=torch.float32
    )
    ry = torch.tensor(
        [[cos[1], 0, sin[1]], [0, 1, 0], [-sin[1], 0, cos[1]]], dtype=torch.float32
    )
    rz = torch.tensor(
        [[cos[2], -sin[2], 0], [sin[2], cos[2], 0], [0, 0, 1]], dtype=torch.float32
    )
    return rz @ ry @ rx


def augment_one(volume, seed, config):
    """Seed-determined physical scanner augmentation of a single case.

    Order matters and follows the recorded pipeline: intensity and gamma, then
    Poisson counts, then additive noise, then anisotropic blur, then resolution
    loss, then rigid motion, then a left-right reflection.
    """
    common.require(
        volume.ndim == 5 and volume.shape[0] == 1 and volume.shape[1] == 1,
        "phase67c_augment_shape",
    )
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    value = torch.clamp(volume.float(), 0.0, 1.5)

    gamma = uniform_cpu(generator, *config["gamma"])
    scale = uniform_cpu(generator, *config["intensity_scale"])
    value = 1.5 * torch.pow(torch.clamp(value / 1.5, 0.0, 1.0), gamma) * scale

    counts = uniform_cpu(generator, *config["poisson_count"])
    if counts > 0.0:
        rate = torch.clamp(value / 1.5, 0.0, 1.0) * counts
        value = 1.5 * torch.poisson(rate, generator=generator) / counts

    noise_std = uniform_cpu(generator, *config["gaussian_noise_std"])
    if noise_std > 0.0:
        value = value + torch.randn(
            value.shape, generator=generator, dtype=value.dtype
        ) * noise_std

    base_sigma = uniform_cpu(generator, *config["blur_sigma"])
    if base_sigma > 0.02:
        sigmas = [
            base_sigma * uniform_cpu(generator, 0.65, 1.35) for _ in range(3)
        ]
        value = anisotropic_blur(value, sigmas)

    resolution = uniform_cpu(generator, *config["resolution_scale"])
    if resolution < 0.995:
        original = value.shape[-3:]
        reduced = [max(8, int(round(size * resolution))) for size in original]
        value = F.interpolate(value, size=reduced, mode="trilinear", align_corners=False)
        value = F.interpolate(
            value, size=original, mode="trilinear", align_corners=False
        )

    degrees = float(config["rotation_degrees"])
    angles = [uniform_cpu(generator, -degrees, degrees) for _ in range(3)]
    voxels = float(config["translation_voxels"])
    size = value.shape[-1]
    translation = [
        uniform_cpu(generator, -voxels, voxels) * 2.0 / max(size - 1, 1)
        for _ in range(3)
    ]
    affine = torch.zeros(1, 3, 4, dtype=torch.float32)
    affine[0, :, :3] = rotation_matrix(angles)
    affine[0, :, 3] = torch.tensor(translation, dtype=torch.float32)
    grid = F.affine_grid(affine, list(value.shape), align_corners=False)
    value = F.grid_sample(
        value, grid, mode="bilinear", padding_mode="zeros", align_corners=False
    )

    if uniform_cpu(generator, 0.0, 1.0) < float(config["reflection_probability"]):
        value = torch.flip(value, dims=[2])
    return torch.clamp(value, 0.0, 1.5)


class RealDataset(Dataset):
    """Two independently seeded scanner views of the same case.

    The two views drive the consistency terms: the model is asked to give the
    same answer to the same patient imaged two ways, which is the property the
    acquisition-transport failures in this project keep violating.
    """

    def __init__(self, cache_file, indices, labels, groups, two_views, seed, config):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices, dtype=np.int64)
        self.labels = np.asarray(labels, dtype=np.int64)
        self.groups = np.asarray(groups, dtype=np.int64)
        self.two_views = bool(two_views)
        self.seed = int(seed)
        self.config = config
        self.epoch = 0
        self.cache = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def resolve_cache(self):
        if self.cache is None:
            self.cache = np.load(
                self.cache_file, mmap_mode="r", allow_pickle=False
            )
        return self.cache

    def __len__(self):
        return int(self.indices.size)

    def __getitem__(self, position):
        case_index = int(self.indices[position])
        cache = self.resolve_cache()
        volume = torch.from_numpy(
            np.asarray(cache[case_index], dtype=np.float32).copy()
        )[None]
        if self.two_views:
            base = self.seed + self.epoch * 1_000_003 + case_index * 193
            view_a = augment_one(
                volume[None], base + 17, self.config["scanner_augmentation"]
            )[0]
            view_b = augment_one(
                volume[None], base + 71, self.config["scanner_augmentation"]
            )[0]
        else:
            view_a = volume
            view_b = volume
        return (
            view_a,
            view_b,
            int(self.labels[case_index]),
            int(self.groups[case_index]),
            case_index,
        )


def group_weight(groups):
    """Inverse-square-root group weighting, renormalized to mean case weight 1."""
    groups = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(groups, minlength=common.GROUP_COUNT).astype(np.float64)
    weight = np.zeros_like(counts)
    positive = counts > 0
    weight[positive] = counts[positive] ** -0.5
    per_case = weight[groups]
    return per_case / max(float(np.mean(per_case)), 1.0e-12)


def learning_rate_multiplier(step, total_steps, config):
    warmup = max(1, int(round(float(config["warmup_fraction"]) * total_steps)))
    if step < warmup:
        return float(step + 1) / float(warmup)
    progress = (step - warmup) / max(1, total_steps - warmup)
    floor = float(config["minimum_learning_rate_multiplier"])
    return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))


def update_ema(ema_model, model, decay):
    with torch.no_grad():
        parameters = dict(model.named_parameters())
        for name, value in ema_model.named_parameters():
            value.mul_(decay).add_(parameters[name], alpha=1.0 - decay)
        buffers = dict(model.named_buffers())
        for name, value in ema_model.named_buffers():
            value.copy_(buffers[name])


def implementation_hash():
    """Bytecode digest of every function this cell owns.

    Editing any of them changes the run hash, so checkpoints from a different
    implementation are refused rather than silently mixed - the failure mode
    that stranded the original Phase66 run.
    """
    owned = [
        group_count_for, augment_one, anisotropic_blur, rotation_matrix,
        gaussian_kernel_1d, uniform_cpu, group_weight,
        learning_rate_multiplier, update_ema, fit_one, predict,
    ]
    digest = []
    for function in owned:
        code = function.__code__
        digest.append(
            {
                "name": function.__name__,
                "code": code.co_code.hex(),
                "names": list(code.co_names),
                "varnames": list(code.co_varnames),
                "argcount": int(code.co_argcount),
            }
        )
    return common.canonical_hash(digest)


@torch.inference_mode()
def predict(model, dataset, config, device):
    """Deterministic, case-independent inference with the EMA weights.

    Row order is asserted against the dataset's own index vector, so a
    misaligned prediction stops the run instead of silently scoring the wrong
    patients.
    """
    model.eval()
    loader = DataLoader(
        dataset,
        batch_size=int(config["batch_size"]),
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )
    logits = []
    seen = []
    for view_a, _, labels, _, case_index in loader:
        volume = view_a.to(device, non_blocking=True)
        target = labels.to(device)
        output = model(volume, target, 0.0)
        logits.append(output["logit"].detach().float().cpu().numpy())
        seen.append(np.asarray(case_index, dtype=np.int64))
    logits = np.concatenate(logits)
    seen = np.concatenate(seen)
    common.require(
        np.array_equal(seen, dataset.indices), "phase67c_prediction_row_order"
    )
    return logits


def fit_one(seed_index, fold, state, config, run_hash, device, output_root):
    """Train one (seed, fold) fit and return its held-out logits.

    Resumes from a per-epoch checkpoint when one exists for the same run hash
    and the same partition; a checkpoint from any other data, code or setting
    is refused.
    """
    labels = state["labels"]
    groups = state["groups"]
    original_fold = state["original_fold"]
    seed = int(config["seeds"][seed_index]) + 1009 * int(fold)

    valid_indices = np.flatnonzero(original_fold == fold)
    train_indices = np.flatnonzero(original_fold != fold)
    common.require(
        not set(groups[valid_indices].tolist())
        & set(groups[train_indices].tolist()),
        "phase67c_outer_group_leakage",
    )
    common.require(
        np.unique(labels[train_indices]).size == 2, "phase67c_single_class_fit"
    )

    checkpoint_path = Path(output_root) / f"fit_{seed_index}_{fold}.pt"
    partition_hash = common.canonical_hash(
        {
            "train": common.array_hash(train_indices),
            "valid": common.array_hash(valid_indices),
        }
    )
    if checkpoint_path.is_file():
        try:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
        except Exception:
            payload = None
        if (
            isinstance(payload, dict)
            and payload.get("run_hash") == run_hash
            and payload.get("partition_hash") == partition_hash
            and bool(payload.get("complete"))
        ):
            stored = np.asarray(payload["valid_logits"], dtype=np.float64)
            if stored.shape == valid_indices.shape:
                return valid_indices, stored, float(payload.get("spent_seconds", 0.0))

    torch.manual_seed(seed)
    np.random.seed(seed % (2**31 - 1))

    train_groups = np.unique(groups[train_indices])
    local_group = {int(value): index for index, value in enumerate(train_groups)}
    prevalence = float(np.mean(labels[train_indices]))
    model = DomainInvariantBilateral3D(
        config["architecture"], prevalence, len(train_groups), config
    ).to(device)
    common.require(
        not any(isinstance(module, nn.BatchNorm3d) for module in model.modules()),
        "phase67c_batchnorm_present",
    )
    ema_model = copy.deepcopy(model).eval().requires_grad_(False)

    optimizer = torch.optim.AdamW(
        [
            {"params": model.backbone.encoder.parameters(), "lr": config["encoder_lr"]},
            {"params": model.backbone.head.parameters(), "lr": config["disease_head_lr"]},
            {"params": model.domain_heads.parameters(), "lr": config["domain_head_lr"]},
        ],
        weight_decay=float(config["weight_decay"]),
    )
    base_rates = [
        config["encoder_lr"], config["disease_head_lr"], config["domain_head_lr"]
    ]

    train_dataset = RealDataset(
        state["highres_cache_file"], train_indices, labels, groups, True, seed, config
    )
    valid_dataset = RealDataset(
        state["highres_cache_file"], valid_indices, labels, groups, False, seed, config
    )
    case_weight = group_weight(groups[train_indices])
    weight_lookup = np.zeros(common.CASE_COUNT, dtype=np.float64)
    weight_lookup[train_indices] = case_weight

    steps_per_epoch = max(
        1, int(math.ceil(train_indices.size / float(config["batch_size"])))
    )
    total_steps = steps_per_epoch * int(config["epoch_count"])
    start_epoch = 0
    spent = 0.0

    if checkpoint_path.is_file():
        try:
            payload = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            if (
                isinstance(payload, dict)
                and payload.get("run_hash") == run_hash
                and payload.get("partition_hash") == partition_hash
            ):
                model.load_state_dict(payload["model"])
                ema_model.load_state_dict(payload["ema"])
                optimizer.load_state_dict(payload["optimizer"])
                start_epoch = int(payload["epoch"]) + 1
                spent = float(payload.get("spent_seconds", 0.0))
        except Exception:
            start_epoch = 0
            spent = 0.0

    budget = float(config["per_fit_budget_hours"]) * 3600.0
    for epoch in range(start_epoch, int(config["epoch_count"])):
        if spent >= budget:
            raise common.Phase67Stop("phase67c_per_fit_budget_exhausted")
        epoch_started = time.perf_counter()
        train_dataset.set_epoch(epoch)
        loader = DataLoader(
            train_dataset,
            batch_size=int(config["batch_size"]),
            shuffle=True,
            num_workers=int(config["worker_count"]),
            drop_last=False,
            generator=torch.Generator().manual_seed(seed + epoch),
        )
        model.train()
        for position, batch in enumerate(loader):
            view_a, view_b, batch_labels, batch_groups, batch_index = batch
            step = epoch * steps_per_epoch + position
            multiplier = learning_rate_multiplier(step, total_steps, config)
            for parameter_group, base in zip(optimizer.param_groups, base_rates):
                parameter_group["lr"] = float(base) * multiplier
            progress = step / max(1, total_steps - 1)
            reversal = float(config["maximum_gradient_reversal"]) * (
                2.0
                / (
                    1.0
                    + math.exp(
                        -float(config["gradient_reversal_ramp"]) * progress
                    )
                )
                - 1.0
            )

            volume_a = view_a.to(device, non_blocking=True)
            volume_b = view_b.to(device, non_blocking=True)
            target = batch_labels.to(device).float()
            domain_target = torch.tensor(
                [local_group[int(value)] for value in batch_groups.tolist()],
                device=device,
                dtype=torch.long,
            )
            weights = torch.tensor(
                weight_lookup[np.asarray(batch_index, dtype=np.int64)],
                device=device,
                dtype=torch.float32,
            )

            output_a = model(volume_a, target.long(), reversal)
            output_b = model(volume_b, target.long(), reversal)

            per_case = 0.5 * (
                F.binary_cross_entropy_with_logits(
                    output_a["logit"], target, reduction="none"
                )
                + F.binary_cross_entropy_with_logits(
                    output_b["logit"], target, reduction="none"
                )
            )
            mix = float(config["group_balance_mix"])
            classification = (1.0 - mix) * per_case.mean() + mix * torch.mean(
                per_case * weights
            )
            logit_consistency = torch.mean(
                torch.square(output_a["logit"] - output_b["logit"])
            )
            representation_consistency = torch.mean(
                1.0
                - F.cosine_similarity(
                    output_a["representation"].float(),
                    output_b["representation"].float(),
                    dim=1,
                    eps=1.0e-6,
                )
            )
            consistency = (
                float(config["logit_view_consistency_weight"]) * logit_consistency
                + float(config["representation_view_consistency_weight"])
                * representation_consistency
            )
            domain_loss = 0.5 * (
                F.cross_entropy(output_a["domain"], domain_target)
                + F.cross_entropy(output_b["domain"], domain_target)
            )
            total = classification + consistency + domain_loss
            common.require(bool(torch.isfinite(total)), "phase67c_nonfinite_loss")

            optimizer.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(config["gradient_clip"])
            )
            optimizer.step()
            update_ema(ema_model, model, float(config["ema_decay"]))

        spent += time.perf_counter() - epoch_started
        torch.save(
            {
                "run_hash": run_hash,
                "partition_hash": partition_hash,
                "model": model.state_dict(),
                "ema": ema_model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": int(epoch),
                "spent_seconds": float(spent),
                "complete": False,
            },
            checkpoint_path.with_suffix(".pt.tmp"),
        )
        os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)

    valid_logits = predict(ema_model, valid_dataset, config, device).astype(np.float64)
    torch.save(
        {
            "run_hash": run_hash,
            "partition_hash": partition_hash,
            "model": model.state_dict(),
            "ema": ema_model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": int(config["epoch_count"]) - 1,
            "spent_seconds": float(spent),
            "valid_logits": valid_logits,
            "complete": True,
        },
        checkpoint_path.with_suffix(".pt.tmp"),
    )
    os.replace(checkpoint_path.with_suffix(".pt.tmp"), checkpoint_path)
    return valid_indices, valid_logits, spent


def build_freeze(state, config):
    return {
        "algorithm": ALGORITHM_VERSION,
        "implementation_sha": implementation_hash(),
        "config": {
            key: value
            for key, value in config.items()
            if key
            not in ("artifact_root", "output_root", "labels_root", "contract_file")
        },
        "labels_sha": common.array_hash(state["labels"]),
        "groups_sha": common.array_hash(state["groups"]),
        "folds_sha": common.array_hash(state["original_fold"]),
        "anchor_sha": common.array_hash(state["anchor_probability"]),
        "cache_sha": common.sha_file(state["highres_cache_file"]),
        "phase56_sha": state["provenance"]["phase56_submission_sha256"],
        "torch": str(torch.__version__),
        "numpy": str(np.__version__),
    }


def resolve_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    return torch.device("cpu")


def worker(seed_index, fold):
    """Single-fit entry point used by the dual-GPU coordinator.

    Each worker sees exactly one GPU through CUDA_VISIBLE_DEVICES, so this is
    task-level parallelism and not distributed data parallel: the per-fit
    optimizer and batch semantics are untouched.
    """
    config = PHASE67C_CONFIG
    try:
        state = common.restore_production_state(
            Path(config["artifact_root"]),
            labels_root=config["labels_root"] or None,
        )
        state["highres_cache_file"] = str(
            Path(config["artifact_root"]) / "phase31_highres_float16.npy"
        )
        output_root = Path(config["output_root"]) / "fits"
        output_root.mkdir(parents=True, exist_ok=True)
        freeze = build_freeze(state, config)
        manifest = Path(config["output_root"]) / "phase67c_frozen_run.json"
        common.require(manifest.is_file(), "phase67c_worker_missing_manifest")
        stored = common.read_json(manifest, "phase67c_frozen_run")
        common.require(stored == freeze, "frozen_inputs_code_settings_changed")
        run_hash = common.canonical_hash(freeze)
        fit_one(
            int(seed_index),
            int(fold),
            state,
            config,
            run_hash,
            resolve_device(),
            output_root,
        )
        return 0
    except common.Phase67Stop as stop:
        print(f"PHASE67C_WORKER_STOP reason={stop}")
        return 2
    except Exception as error:  # noqa: BLE001
        print(f"PHASE67C_WORKER_STOP reason=unhandled_{type(error).__name__}")
        return 2


def run():
    started = time.perf_counter()
    config = PHASE67C_CONFIG
    artifact_root = Path(config["artifact_root"])
    output_root = Path(config["output_root"])
    fits_root = output_root / "fits"
    fits_root.mkdir(parents=True, exist_ok=True)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cuda.matmul.allow_tf32 = False

    state = common.restore_production_state(
        artifact_root, labels_root=config["labels_root"] or None
    )
    state["highres_cache_file"] = str(artifact_root / "phase31_highres_float16.npy")
    labels = state["labels"]
    groups = state["groups"]
    original_fold = state["original_fold"]
    anchor = state["anchor_probability"]
    domain_count = len(state["stress_domain_definitions"])

    freeze = build_freeze(state, config)
    manifest = output_root / "phase67c_frozen_run.json"
    if manifest.is_file():
        stored = common.read_json(manifest, "phase67c_frozen_run")
        common.require(stored == freeze, "frozen_inputs_code_settings_changed")
    else:
        common.require(
            not list(fits_root.glob("fit_*_*.pt")),
            "orphan_checkpoints_without_manifest",
        )
        common.atomic_json(manifest, freeze)
    run_hash = common.canonical_hash(freeze)

    seed_count = len(config["seeds"])
    fold_count = int(config["fold_count"])
    work = [
        (seed_index, fold)
        for seed_index in range(seed_count)
        for fold in range(fold_count)
    ]

    device_count = torch.cuda.device_count()
    if device_count >= 2:
        # One independent fit per GPU, in waves. Not DDP.
        for start in range(0, len(work), 2):
            wave = work[start : start + 2]
            processes = []
            for gpu_index, (seed_index, fold) in enumerate(wave):
                environment = dict(os.environ)
                environment["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
                processes.append(
                    subprocess.Popen(
                        [
                            sys.executable,
                            __file__,
                            "--phase67c-worker",
                            str(seed_index),
                            str(fold),
                        ],
                        env=environment,
                    )
                )
            for process in processes:
                common.require(process.wait() == 0, "phase67c_worker_failed")
        device = resolve_device()
        results = {}
        for seed_index, fold in work:
            valid_indices, logits, spent = fit_one(
                seed_index, fold, state, config, run_hash, device, fits_root
            )
            results[(seed_index, fold)] = (valid_indices, logits, spent)
    else:
        device = resolve_device()
        results = {}
        for seed_index, fold in work:
            results[(seed_index, fold)] = fit_one(
                seed_index, fold, state, config, run_hash, device, fits_root
            )

    seed_probability = np.zeros((seed_count, common.CASE_COUNT), dtype=np.float64)
    total_spent = 0.0
    for (seed_index, fold), (valid_indices, logits, spent) in results.items():
        seed_probability[seed_index, valid_indices] = common.sigmoid(logits)
        total_spent += float(spent)
    common.require(
        bool(np.all((seed_probability > 0.0) & (seed_probability < 1.0))),
        "phase67c_incomplete_seed_coverage",
    )

    # Seed averaging in probability space, matching the Phase66F aggregation.
    expert = seed_probability.mean(axis=0)
    reference_blend = common.sigmoid(
        float(config["reference_anchor_logit_weight"]) * common.logit(anchor)
        + float(config["reference_expert_logit_weight"]) * common.logit(expert)
    )

    stability = common.stability_rows(
        labels, groups, original_fold, state["stress_domain_id"], anchor, reference_blend
    )
    domain_gains = common.per_domain_gains(
        labels,
        state["stress_domain_id"],
        anchor,
        reference_blend,
        expected_domains=domain_count,
    )

    expert_path = artifact_root / "phase67_expert_phase67c_multiseed_float64.npy"
    np.save(output_root / expert_path.name, expert, allow_pickle=False)
    common.atomic_npz(
        output_root / "phase67c_multiseed_expert_oof.npz",
        expert=expert,
        seed_probability=seed_probability,
        candidate=reference_blend,
    )

    core = {
        "schema_version": config["schema_version"],
        "status": "phase67c_multiseed_expert_complete_feed_into_phase67b_stack",
        "common_version": common.PHASE67_COMMON_VERSION,
        "algorithm": ALGORITHM_VERSION,
        "config": freeze["config"],
        "implementation_sha": freeze["implementation_sha"],
        "run_sha256": run_hash,
        "fit_count": len(work),
        "seed_count": seed_count,
        "fold_count": fold_count,
        "partition": "original_three_whole_acquisition_group_folds",
        "parallelism": (
            "one_independent_seed_fold_fit_per_gpu"
            if device_count >= 2
            else "single_device_sequential"
        ),
        "ddp_used": False,
        "anchor": state["anchor_metrics"],
        "expert_alone": common.metrics(labels, expert),
        "seed_metrics": [
            common.metrics(labels, seed_probability[index])
            for index in range(seed_count)
        ],
        "reference_fixed_alpha_blend": {
            "formula": "sigmoid(0.85*anchor_logit+0.15*expert_logit)",
            "reported_for_comparability_with_phase65c_and_phase65d": True,
            "used_to_select_anything": False,
            "pooled": stability["pooled"],
            "folds": stability["folds"],
            "domains": stability["domains"],
            "macro_domain_bootstrap": common.macro_domain_bootstrap(
                domain_gains, 10000, 660699
            ),
        },
        "blend_weight_selected_here": False,
        "blend_weight_selection_deferred_to": "phase67b_nested_stack",
        "interpretation": {
            "five_seeds_over_the_original_folds_not_reassigned_partitions": True,
            "no_reassigned_fold_anchor_reuse_caveat_applies_to_this_expert": True,
            "seed_averaging_reduces_variance_it_does_not_add_information": True,
            "expert_is_trained_on_the_80_cubed_cache_whose_raw_nifti_parity_is_unproven": True,
            "deployment_blocked_until_phase67a_cache_parity_passes": True,
            "development_result_is_not_a_leaderboard_claim": True,
        },
        "total_recorded_fit_seconds": round(total_spent, 3),
        "test_data_read": False,
        "test_time_adaptation": False,
        "phase56_archive_unchanged": True,
        "submission_archive_created": False,
        "provenance": state["provenance"],
    }

    contract_sha = common.emit_contract(
        output_root / config["contract_file"], core
    )
    report = {
        **core,
        "contract_sha256": contract_sha,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    common.print_sanitized("PHASE67C_MULTISEED_DOMAIN_INVARIANT_EXPERT", report)
    return report


if __name__ == "__main__":
    if len(sys.argv) == 4 and sys.argv[1] == "--phase67c-worker":
        raise SystemExit(worker(sys.argv[2], sys.argv[3]))
    try:
        run()
    except common.Phase67Stop as stop:
        print(
            json.dumps(
                {
                    "phase": "phase67c_multiseed_domain_invariant_expert",
                    "status": "phase67c_stopped",
                    "stage": str(stop),
                    "phase56_archive_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
