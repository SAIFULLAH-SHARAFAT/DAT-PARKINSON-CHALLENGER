from __future__ import annotations

import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Cell 157B — Phase57 scanner-robust bilateral 3D architecture contract.
#
# Run the accepted Cell 157A first. This cell reads a small, deterministic
# contract batch from the training-only 80^3 cache. It does not train or select
# a candidate, inspect any test/smoke case, or use public leaderboard values.
# The model is initialized as an exact identity update to the eligible Phase43
# anchor. All normalization is per case or GroupNorm; BatchNorm is forbidden.

phase57b_started = time.perf_counter()

assert isinstance(
    globals().get("PHASE57_TRANSPORT_STATE_PRIVATE"), dict
), {
    "message": (
        "Cell 157B requires PHASE57_TRANSPORT_STATE_PRIVATE. "
        "Run the accepted Phase57 Cell 157A in this kernel first."
    )
}

phase57b_state = PHASE57_TRANSPORT_STATE_PRIVATE
phase57b_labels = np.asarray(
    phase57b_state["labels"], dtype=np.int64
).reshape(-1)
phase57b_groups = np.asarray(
    phase57b_state["groups"], dtype=np.int64
).reshape(-1)
phase57b_original_fold = np.asarray(
    phase57b_state["original_fold"], dtype=np.int64
).reshape(-1)
phase57b_anchor_probability = np.asarray(
    phase57b_state["phase43_anchor_probability"], dtype=np.float64
).reshape(-1)
phase57b_highres_path = Path(phase57b_state["highres_cache_file"])

assert phase57b_labels.shape == (1362,)
assert phase57b_groups.shape == (1362,)
assert phase57b_original_fold.shape == (1362,)
assert phase57b_anchor_probability.shape == (1362,)
assert set(np.unique(phase57b_labels).tolist()) == {0, 1}
assert np.array_equal(
    np.unique(phase57b_groups), np.arange(15, dtype=np.int64)
)
assert np.all(np.isin(phase57b_original_fold, [0, 1, 2]))
assert np.all(np.isfinite(phase57b_anchor_probability))
assert np.all(
    (phase57b_anchor_probability > 0.0)
    & (phase57b_anchor_probability < 1.0)
)
assert phase57b_highres_path.is_file(), {
    "message": "The accepted 80^3 training cache is missing.",
    "path": str(phase57b_highres_path),
}


PHASE57B_CONFIG = {
    "seed": 570157,
    "input_shape": [1, 80, 80, 80],
    "left_right_tensor_dimension": 2,
    "left_right_volume_axis": 0,
    "channels": [16, 32, 64, 112],
    "blocks_per_stage": [1, 1, 2, 2],
    "multiscale_projection_dimension": 48,
    "branch_embedding_dimension": 192,
    "bilateral_embedding_dimension": 384,
    "head_hidden_dimension": 96,
    "residual_cap": 1.0,
    "contract_outer_fold": 0,
    "contract_case_count": 4,
    "architecture_contract_file": str(
        phase57b_highres_path.parent
        / "phase57_architecture_contract.json"
    ),
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
    "planned_screen": {
        "width_multipliers": [0.75, 1.0, 1.25],
        "residual_caps": [0.5, 1.0],
        "rank_loss_weights": [0.0, 0.05],
        "group_dro_values": [0.0, 0.05],
        "candidate_count": 24,
        "shared_specification_across_three_group_folds": True,
        "epoch_zero_anchor_identity_allowed": True,
    },
}

torch.manual_seed(PHASE57B_CONFIG["seed"])
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(PHASE57B_CONFIG["seed"])

phase57b_device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
phase57b_cuda = phase57b_device.type == "cuda"
phase57b_amp_dtype = torch.bfloat16


def phase57b_probability_to_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    return np.log(probability) - np.log1p(-probability)


def phase57b_group_count(channels):
    """Largest divisor no greater than eight; never one when avoidable."""
    channels = int(channels)
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class Phase57ResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = phase57b_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(
            channels, channels, kernel_size=3, padding=1, bias=False
        )

    def forward(self, value):
        residual = value
        value = self.conv1(F.silu(self.norm1(value), inplace=False))
        value = self.conv2(F.silu(self.norm2(value), inplace=False))
        return residual + value


class Phase57Downsample3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.norm = nn.GroupNorm(
            phase57b_group_count(input_channels), input_channels
        )
        self.conv = nn.Conv3d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=2,
            padding=1,
            bias=False,
        )

    def forward(self, value):
        return self.conv(F.silu(self.norm(value), inplace=False))


class Phase57MultiscaleEncoder3D(nn.Module):
    def __init__(
        self,
        input_channels=3,
        channels=(16, 32, 64, 112),
        blocks_per_stage=(1, 1, 2, 2),
        projection_dimension=48,
    ):
        super().__init__()
        channels = tuple(int(value) for value in channels)
        blocks_per_stage = tuple(int(value) for value in blocks_per_stage)
        assert len(channels) == len(blocks_per_stage) == 4

        self.stem = nn.Sequential(
            nn.Conv3d(
                input_channels,
                channels[0],
                kernel_size=5,
                stride=2,
                padding=2,
                bias=False,
            ),
            nn.GroupNorm(
                phase57b_group_count(channels[0]), channels[0]
            ),
            nn.SiLU(inplace=False),
        )

        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.projections = nn.ModuleList()

        for stage_index, (width, block_count) in enumerate(zip(
            channels, blocks_per_stage
        )):
            self.stages.append(nn.Sequential(*[
                Phase57ResidualBlock3D(width)
                for _ in range(block_count)
            ]))
            self.projections.append(nn.Sequential(
                nn.Linear(2 * width, projection_dimension),
                nn.LayerNorm(projection_dimension),
                nn.SiLU(inplace=False),
            ))
            if stage_index + 1 < len(channels):
                self.downsamples.append(Phase57Downsample3D(
                    width, channels[stage_index + 1]
                ))

        self.output_dimension = (
            len(channels) * int(projection_dimension)
        )

    def forward(self, value):
        value = self.stem(value)
        summaries = []
        for stage_index, stage in enumerate(self.stages):
            value = stage(value)
            average = F.adaptive_avg_pool3d(value, 1).flatten(1)
            maximum = F.adaptive_max_pool3d(value, 1).flatten(1)
            summaries.append(self.projections[stage_index](
                torch.cat([average, maximum], dim=1)
            ))
            if stage_index < len(self.downsamples):
                value = self.downsamples[stage_index](value)
        return torch.cat(summaries, dim=1)


class Phase57ScannerRobustBilateral3D(nn.Module):
    """Reflection-invariant, Phase43-anchored bounded residual model."""

    def __init__(
        self,
        channels=(16, 32, 64, 112),
        blocks_per_stage=(1, 1, 2, 2),
        projection_dimension=48,
        head_hidden_dimension=96,
        residual_cap=1.0,
    ):
        super().__init__()
        self.left_right_dimension = 2
        self.residual_cap = float(residual_cap)
        assert self.residual_cap > 0.0

        self.encoder = Phase57MultiscaleEncoder3D(
            input_channels=3,
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            projection_dimension=projection_dimension,
        )
        branch_dimension = self.encoder.output_dimension
        bilateral_dimension = 2 * branch_dimension
        self.head = nn.Sequential(
            nn.LayerNorm(bilateral_dimension),
            nn.Linear(bilateral_dimension, head_hidden_dimension),
            nn.SiLU(inplace=False),
            nn.Linear(head_hidden_dimension, 1),
        )
        # Exact anchor identity at initialization. The first step trains the
        # final head only; gradients reach the encoder from the second step.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def normalize_per_case(volume):
        assert volume.ndim == 5 and volume.shape[1] == 1
        value = torch.clamp(volume.float(), 0.0, 1.5) / 1.5
        mean = value.mean(dim=(2, 3, 4), keepdim=True)
        variance = torch.mean(
            torch.square(value - mean), dim=(2, 3, 4), keepdim=True
        )
        standard_deviation = torch.sqrt(variance + 1.0e-6)
        return torch.clamp(
            (value - mean) / standard_deviation, -6.0, 6.0
        )

    def make_bilateral_inputs(self, volume):
        value = self.normalize_per_case(volume)
        reflected = torch.flip(value, dims=[self.left_right_dimension])
        symmetric = 0.5 * (value + reflected)
        antisymmetric = torch.abs(value - reflected)
        original_branch = torch.cat(
            [value, symmetric, antisymmetric], dim=1
        )
        reflected_branch = torch.cat(
            [reflected, symmetric, antisymmetric], dim=1
        )
        return original_branch, reflected_branch

    def forward(self, volume, anchor_logit):
        assert volume.ndim == 5 and volume.shape[1] == 1
        assert tuple(volume.shape[2:]) == (80, 80, 80)
        anchor_logit = anchor_logit.reshape(-1).to(
            device=volume.device, dtype=torch.float32
        )
        assert anchor_logit.shape[0] == volume.shape[0]

        original, reflected = self.make_bilateral_inputs(volume)
        # A single shared-encoder call prevents branch-specific state.
        encoded = self.encoder(torch.cat([original, reflected], dim=0))
        original_embedding, reflected_embedding = encoded.chunk(2, dim=0)
        symmetric_embedding = 0.5 * (
            original_embedding + reflected_embedding
        )
        antisymmetric_embedding = torch.abs(
            original_embedding - reflected_embedding
        )
        representation = torch.cat(
            [symmetric_embedding, antisymmetric_embedding], dim=1
        )
        raw_residual = self.head(representation).reshape(-1).float()
        residual = self.residual_cap * torch.tanh(raw_residual)
        logit = anchor_logit + residual
        probability = torch.sigmoid(logit)
        return {
            "logit": logit,
            "probability": probability,
            "raw_residual": raw_residual,
            "residual": residual,
            "representation": representation,
            "original_embedding": original_embedding,
            "reflected_embedding": reflected_embedding,
        }


def phase57b_uniform(generator, lower, upper):
    return float(lower + (upper - lower) * torch.rand(
        (), generator=generator
    ).item())


def phase57b_gaussian_kernel_1d(sigma, device, dtype):
    if sigma <= 1.0e-6:
        return torch.ones(1, device=device, dtype=dtype)
    radius = 2
    coordinate = torch.arange(
        -radius, radius + 1, device=device, dtype=dtype
    )
    kernel = torch.exp(-0.5 * torch.square(coordinate / sigma))
    return kernel / torch.sum(kernel)


def phase57b_anisotropic_blur(volume, sigmas):
    value = volume
    for spatial_index, sigma in enumerate(sigmas):
        if sigma <= 1.0e-6:
            continue
        kernel = phase57b_gaussian_kernel_1d(
            sigma, value.device, value.dtype
        )
        shape = [1, 1, 1, 1, 1]
        shape[2 + spatial_index] = kernel.numel()
        weight = kernel.reshape(shape)
        padding = [0, 0, 0]
        padding[spatial_index] = kernel.numel() // 2
        value = F.conv3d(value, weight, padding=tuple(padding))
    return value


def phase57b_rotation_matrix(angles):
    ax, ay, az = angles
    one = torch.ones((), dtype=torch.float32)
    zero = torch.zeros((), dtype=torch.float32)
    cx, sx = torch.cos(ax), torch.sin(ax)
    cy, sy = torch.cos(ay), torch.sin(ay)
    cz, sz = torch.cos(az), torch.sin(az)
    rx = torch.stack([
        torch.stack([one, zero, zero]),
        torch.stack([zero, cx, -sx]),
        torch.stack([zero, sx, cx]),
    ])
    ry = torch.stack([
        torch.stack([cy, zero, sy]),
        torch.stack([zero, one, zero]),
        torch.stack([-sy, zero, cy]),
    ])
    rz = torch.stack([
        torch.stack([cz, -sz, zero]),
        torch.stack([sz, cz, zero]),
        torch.stack([zero, zero, one]),
    ])
    return rz @ ry @ rx


def phase57b_augment_one(volume, seed, config):
    """Training-only augmentation with a per-case CPU random generator."""
    assert volume.shape == (1, 1, 80, 80, 80)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.clamp(volume.float(), 0.0, 1.5)

    scale = phase57b_uniform(
        generator, *config["intensity_scale"]
    )
    gamma = phase57b_uniform(generator, *config["gamma"])
    value = 1.5 * torch.pow(
        torch.clamp(value / 1.5, 0.0, 1.0), gamma
    ) * scale

    poisson_count = phase57b_uniform(
        generator, *config["poisson_count"]
    )
    cpu_value = value.detach().cpu()
    cpu_value = torch.poisson(
        torch.clamp(cpu_value / 1.5, 0.0, 1.0) * poisson_count,
        generator=generator,
    ) / poisson_count
    value = (1.5 * cpu_value).to(volume.device)

    noise_std = phase57b_uniform(
        generator, *config["gaussian_noise_std"]
    )
    noise = torch.randn(
        value.shape, generator=generator, dtype=torch.float32
    ).to(value.device)
    value = value + noise_std * noise

    sigma = phase57b_uniform(generator, *config["blur_sigma"])
    anisotropy = [
        sigma * phase57b_uniform(generator, 0.65, 1.35)
        for _ in range(3)
    ]
    value = phase57b_anisotropic_blur(value, anisotropy)

    resolution_scale = phase57b_uniform(
        generator, *config["resolution_scale"]
    )
    if resolution_scale < 0.995:
        reduced_size = max(16, int(round(80 * resolution_scale)))
        value = F.interpolate(
            value,
            size=(reduced_size, reduced_size, reduced_size),
            mode="trilinear",
            align_corners=False,
        )
        value = F.interpolate(
            value,
            size=(80, 80, 80),
            mode="trilinear",
            align_corners=False,
        )

    maximum_angle = math.radians(float(config["rotation_degrees"]))
    angles = torch.tensor([
        phase57b_uniform(generator, -maximum_angle, maximum_angle)
        for _ in range(3)
    ], dtype=torch.float32)
    rotation = phase57b_rotation_matrix(angles)
    maximum_translation = float(config["translation_voxels"])
    # affine_grid translations are normalized to approximately [-1, 1].
    translation = torch.tensor([
        phase57b_uniform(
            generator, -maximum_translation, maximum_translation
        ) * (2.0 / 79.0)
        for _ in range(3)
    ], dtype=torch.float32)
    theta = torch.cat([rotation, translation[:, None]], dim=1)[None]
    theta = theta.to(value.device)
    grid = F.affine_grid(theta, value.shape, align_corners=False)
    value = F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )

    if phase57b_uniform(generator, 0.0, 1.0) < float(
        config["reflection_probability"]
    ):
        value = torch.flip(value, dims=[2])
    return torch.clamp(value, 0.0, 1.5)


def phase57b_gradient_record(parameters):
    gradients = [
        parameter.grad.detach().float()
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return {"tensor_count": 0, "norm": 0.0, "all_finite": True}
    squared_norm = sum(
        float(torch.sum(torch.square(gradient)).item())
        for gradient in gradients
    )
    return {
        "tensor_count": len(gradients),
        "norm": float(math.sqrt(squared_norm)),
        "all_finite": bool(all(
            torch.all(torch.isfinite(gradient)).item()
            for gradient in gradients
        )),
    }


# Deterministic contract cases come only from the fit side of original fold 0.
phase57b_fit_mask = phase57b_original_fold != int(
    PHASE57B_CONFIG["contract_outer_fold"]
)
phase57b_contract_indices = []
for phase57b_label in (0, 1):
    phase57b_label_groups = np.unique(phase57b_groups[
        phase57b_fit_mask & (phase57b_labels == phase57b_label)
    ])
    for phase57b_group in phase57b_label_groups[:2]:
        phase57b_matches = np.flatnonzero(
            phase57b_fit_mask
            & (phase57b_labels == phase57b_label)
            & (phase57b_groups == phase57b_group)
        )
        assert phase57b_matches.size > 0
        phase57b_contract_indices.append(int(phase57b_matches[0]))

phase57b_contract_indices = np.asarray(
    phase57b_contract_indices, dtype=np.int64
)
assert phase57b_contract_indices.shape == (
    PHASE57B_CONFIG["contract_case_count"],
)
assert np.all(phase57b_fit_mask[phase57b_contract_indices])
assert set(phase57b_labels[phase57b_contract_indices].tolist()) == {0, 1}
assert len(np.unique(phase57b_groups[phase57b_contract_indices])) >= 2

phase57b_highres = np.load(phase57b_highres_path, mmap_mode="r")
assert phase57b_highres.shape == (1362, 80, 80, 80)
assert phase57b_highres.dtype == np.float16
phase57b_contract_numpy = np.asarray(
    phase57b_highres[phase57b_contract_indices], dtype=np.float32
)
assert phase57b_contract_numpy.shape == (4, 80, 80, 80)
assert np.all(np.isfinite(phase57b_contract_numpy))
phase57b_contract_volume = torch.from_numpy(
    phase57b_contract_numpy[:, None]
).to(phase57b_device)
phase57b_contract_label = torch.from_numpy(
    phase57b_labels[phase57b_contract_indices].astype(np.float32)
).to(phase57b_device)
phase57b_contract_anchor_logit = torch.from_numpy(
    phase57b_probability_to_logit(
        phase57b_anchor_probability[phase57b_contract_indices]
    ).astype(np.float32)
).to(phase57b_device)

phase57b_model = Phase57ScannerRobustBilateral3D(
    channels=PHASE57B_CONFIG["channels"],
    blocks_per_stage=PHASE57B_CONFIG["blocks_per_stage"],
    projection_dimension=PHASE57B_CONFIG[
        "multiscale_projection_dimension"
    ],
    head_hidden_dimension=PHASE57B_CONFIG["head_hidden_dimension"],
    residual_cap=PHASE57B_CONFIG["residual_cap"],
).to(phase57b_device)

if phase57b_cuda:
    torch.cuda.reset_peak_memory_stats(phase57b_device)

assert not any(
    isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d))
    for module in phase57b_model.modules()
)

phase57b_total_parameter_count = sum(
    parameter.numel() for parameter in phase57b_model.parameters()
)
phase57b_trainable_parameter_count = sum(
    parameter.numel()
    for parameter in phase57b_model.parameters()
    if parameter.requires_grad
)
assert phase57b_total_parameter_count == phase57b_trainable_parameter_count
assert phase57b_total_parameter_count < 5_000_000

# Identity, shape, reflection, and inference batch-independence contracts.
phase57b_model.eval()
with torch.no_grad():
    (
        phase57b_original_input,
        phase57b_reflected_input,
    ) = phase57b_model.make_bilateral_inputs(phase57b_contract_volume)
    (
        phase57b_flipped_original_input,
        phase57b_flipped_reflected_input,
    ) = phase57b_model.make_bilateral_inputs(
        torch.flip(phase57b_contract_volume, dims=[2])
    )
    with torch.amp.autocast(
        device_type=phase57b_device.type,
        dtype=phase57b_amp_dtype,
        enabled=phase57b_cuda,
    ):
        phase57b_initial = phase57b_model(
            phase57b_contract_volume, phase57b_contract_anchor_logit
        )
        phase57b_reflected = phase57b_model(
            torch.flip(phase57b_contract_volume, dims=[2]),
            phase57b_contract_anchor_logit,
        )
    # Evaluate batch independence in float32 so the contract tests model
    # semantics rather than a possible batch-size-dependent AMP kernel choice.
    phase57b_batch_float32 = phase57b_model(
        phase57b_contract_volume, phase57b_contract_anchor_logit
    )
    phase57b_reflected_float32 = phase57b_model(
        torch.flip(phase57b_contract_volume, dims=[2]),
        phase57b_contract_anchor_logit,
    )
    phase57b_single_float32 = phase57b_model(
        phase57b_contract_volume[:1],
        phase57b_contract_anchor_logit[:1],
    )

phase57b_anchor_probability_tensor = torch.sigmoid(
    phase57b_contract_anchor_logit.float()
)
phase57b_bilateral_input_swap_error = float(max(
    torch.max(torch.abs(
        phase57b_original_input
        - phase57b_flipped_reflected_input
    )).item(),
    torch.max(torch.abs(
        phase57b_reflected_input
        - phase57b_flipped_original_input
    )).item(),
))
phase57b_identity_logit_error = float(torch.max(torch.abs(
    phase57b_initial["logit"].float()
    - phase57b_contract_anchor_logit.float()
)).item())
phase57b_identity_probability_error = float(torch.max(torch.abs(
    phase57b_initial["probability"].float()
    - phase57b_anchor_probability_tensor
)).item())
phase57b_amp_reflection_logit_error = float(torch.max(torch.abs(
    phase57b_initial["logit"].float()
    - phase57b_reflected["logit"].float()
)).item())
phase57b_amp_reflection_representation_error = float(torch.max(torch.abs(
    phase57b_initial["representation"].float()
    - phase57b_reflected["representation"].float()
)).item())
phase57b_reflection_logit_error = float(torch.max(torch.abs(
    phase57b_batch_float32["logit"].float()
    - phase57b_reflected_float32["logit"].float()
)).item())
phase57b_reflection_representation_error = float(torch.max(torch.abs(
    phase57b_batch_float32["representation"].float()
    - phase57b_reflected_float32["representation"].float()
)).item())
phase57b_batch_independence_error = float(torch.max(torch.abs(
    phase57b_batch_float32["probability"][:1].float()
    - phase57b_single_float32["probability"].float()
)).item())
phase57b_batch_independence_representation_error = float(torch.max(
    torch.abs(
        phase57b_batch_float32["representation"][:1].float()
        - phase57b_single_float32["representation"].float()
    )
).item())

assert phase57b_initial["representation"].shape == (4, 384)
assert phase57b_initial["probability"].shape == (4,)
assert phase57b_identity_logit_error == 0.0, {
    "identity_logit_error": phase57b_identity_logit_error,
}
assert phase57b_identity_probability_error == 0.0, {
    "identity_probability_error": phase57b_identity_probability_error,
}
assert phase57b_bilateral_input_swap_error <= 1.0e-6, {
    "bilateral_input_swap_error": phase57b_bilateral_input_swap_error,
}
assert phase57b_reflection_logit_error <= 1.0e-6, {
    "float32_reflection_logit_error": phase57b_reflection_logit_error,
}
assert phase57b_reflection_representation_error <= 2.0e-3, {
    "float32_reflection_representation_error": (
        phase57b_reflection_representation_error
    ),
    "amp_reflection_representation_error_diagnostic": (
        phase57b_amp_reflection_representation_error
    ),
}
assert phase57b_batch_independence_error <= 2.0e-6, {
    "float32_batch_independence_probability_error": (
        phase57b_batch_independence_error
    ),
}
assert phase57b_batch_independence_representation_error <= 1.0e-4, {
    "float32_batch_independence_representation_error": (
        phase57b_batch_independence_representation_error
    ),
}
assert torch.all(torch.isfinite(
    phase57b_initial["representation"]
)).item()

# Augmentation is deterministic per (case, seed) and independent of companions.
phase57b_augmentation_seed = PHASE57B_CONFIG["seed"] + int(
    phase57b_contract_indices[0]
)
with torch.no_grad():
    phase57b_augmented_a = phase57b_augment_one(
        phase57b_contract_volume[:1],
        phase57b_augmentation_seed,
        PHASE57B_CONFIG["scanner_augmentation"],
    )
    phase57b_augmented_b = phase57b_augment_one(
        phase57b_contract_volume[:1],
        phase57b_augmentation_seed,
        PHASE57B_CONFIG["scanner_augmentation"],
    )
phase57b_augmentation_replay_error = float(torch.max(torch.abs(
    phase57b_augmented_a - phase57b_augmented_b
)).item())
phase57b_augmentation_change = float(torch.mean(torch.abs(
    phase57b_augmented_a - phase57b_contract_volume[:1]
)).item())
assert phase57b_augmentation_replay_error == 0.0
assert phase57b_augmentation_change > 0.0
assert torch.all(torch.isfinite(phase57b_augmented_a)).item()

# Two-step backward contract: zero initialization gives exact anchor identity,
# while the encoder must receive a nonzero gradient after the head has moved.
phase57b_model.train()
phase57b_optimizer = torch.optim.AdamW([
    {
        "params": list(phase57b_model.encoder.parameters()),
        "lr": 2.0e-4,
    },
    {
        "params": list(phase57b_model.head.parameters()),
        "lr": 8.0e-4,
    },
], weight_decay=0.02)

phase57b_backward_records = []
for phase57b_step in (1, 2):
    phase57b_optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=phase57b_device.type,
        dtype=phase57b_amp_dtype,
        enabled=phase57b_cuda,
    ):
        phase57b_output = phase57b_model(
            phase57b_contract_volume,
            phase57b_contract_anchor_logit,
        )
        phase57b_loss = F.binary_cross_entropy_with_logits(
            phase57b_output["logit"].float(),
            phase57b_contract_label.float(),
        )
    phase57b_loss.backward()
    phase57b_encoder_gradient = phase57b_gradient_record(
        phase57b_model.encoder.parameters()
    )
    phase57b_head_gradient = phase57b_gradient_record(
        phase57b_model.head.parameters()
    )
    assert phase57b_encoder_gradient["all_finite"]
    assert phase57b_head_gradient["all_finite"]
    torch.nn.utils.clip_grad_norm_(phase57b_model.parameters(), 1.0)
    phase57b_optimizer.step()
    phase57b_backward_records.append({
        "step": phase57b_step,
        "loss": float(phase57b_loss.detach().item()),
        "encoder_gradient": phase57b_encoder_gradient,
        "head_gradient": phase57b_head_gradient,
        "maximum_absolute_residual": float(torch.max(torch.abs(
            phase57b_output["residual"].detach().float()
        )).item()),
    })

assert phase57b_backward_records[0]["encoder_gradient"]["norm"] == 0.0
assert phase57b_backward_records[0]["head_gradient"]["norm"] > 0.0
assert phase57b_backward_records[1]["encoder_gradient"]["norm"] > 0.0
assert phase57b_backward_records[1]["head_gradient"]["norm"] > 0.0

phase57b_peak_vram_mb = 0.0
if phase57b_cuda:
    phase57b_peak_vram_mb = float(
        torch.cuda.max_memory_allocated(phase57b_device) / (1024 ** 2)
    )

phase57b_contract_core = {
    "schema_version": "phase57_scanner_robust_architecture_v1",
    "transport_contract_sha256": phase57b_state["contract_sha256"],
    "architecture": {
        "model": "Phase57ScannerRobustBilateral3D",
        "input_shape": PHASE57B_CONFIG["input_shape"],
        "left_right_volume_axis": 0,
        "bilateral_channels": [
            "original", "left_right_symmetric", "absolute_antisymmetric"
        ],
        "channels": PHASE57B_CONFIG["channels"],
        "blocks_per_stage": PHASE57B_CONFIG["blocks_per_stage"],
        "branch_embedding_dimension": 192,
        "bilateral_embedding_dimension": 384,
        "residual_cap": PHASE57B_CONFIG["residual_cap"],
        "normalization": "per_case_standardization_plus_group_norm",
        "batch_normalization_used": False,
        "explicit_group_feature_used": False,
        "header_router_feature_used": False,
    },
    "anchor": {
        "source": "eligible_phase43_deployment_matched_probability",
        "zero_initialized_bounded_logit_residual": True,
        "exact_identity_at_initialization": True,
    },
    "scanner_augmentation": PHASE57B_CONFIG["scanner_augmentation"],
    "planned_screen": PHASE57B_CONFIG["planned_screen"],
    "validation": {
        "selection": "original_three_acquisition_group_disjoint_folds",
        "confirmation": "fifteen_retrained_logo_folds_after_freeze",
        "contract_cases_from_outer_fold_0_fit_only": True,
        "public_leaderboard_used": False,
    },
    "test_rule": {
        "each_case_processed_independently": True,
        "test_batch_statistics": False,
        "test_time_training": False,
        "test_time_adaptation": False,
        "pseudo_labeling": False,
    },
}
phase57b_contract_sha256 = hashlib.sha256(json.dumps(
    phase57b_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

phase57b_contract_path = Path(
    PHASE57B_CONFIG["architecture_contract_file"]
)
phase57b_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase57b_temporary_path = phase57b_contract_path.with_suffix(
    phase57b_contract_path.suffix + ".tmp"
)
phase57b_temporary_path.write_text(json.dumps({
    **phase57b_contract_core,
    "contract_sha256": phase57b_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase57b_temporary_path, phase57b_contract_path)

PHASE57_SCANNER_ROBUST_CONFIG_PRIVATE = json.loads(json.dumps(
    PHASE57B_CONFIG
))
PHASE57_SCANNER_ROBUST_STATE_PRIVATE = {
    "contract_sha256": phase57b_contract_sha256,
    "transport_contract_sha256": phase57b_state["contract_sha256"],
    "model_class": Phase57ScannerRobustBilateral3D,
    "residual_block_class": Phase57ResidualBlock3D,
    "augment_one": phase57b_augment_one,
    "labels": phase57b_labels.copy(),
    "groups": phase57b_groups.copy(),
    "original_fold": phase57b_original_fold.copy(),
    "anchor_probability": phase57b_anchor_probability.copy(),
    "highres_cache_file": str(phase57b_highres_path),
}

phase57b_report = {
    "phase": "phase57_scanner_robust_bilateral_3d_architecture_contract",
    "status": "accepted_ready_for_three_fold_training_engine_pilot",
    "architecture": phase57b_contract_core["architecture"],
    "parameters": {
        "total": phase57b_total_parameter_count,
        "trainable": phase57b_trainable_parameter_count,
    },
    "initial_contract": {
        "logit_identity_error": phase57b_identity_logit_error,
        "probability_identity_error": phase57b_identity_probability_error,
        "bilateral_input_swap_error": phase57b_bilateral_input_swap_error,
        "amp_reflection_logit_error_diagnostic": (
            phase57b_amp_reflection_logit_error
        ),
        "amp_reflection_representation_error_diagnostic": (
            phase57b_amp_reflection_representation_error
        ),
        "reflection_logit_error": phase57b_reflection_logit_error,
        "reflection_representation_error": (
            phase57b_reflection_representation_error
        ),
        "inference_batch_independence_probability_error": (
            phase57b_batch_independence_error
        ),
        "inference_batch_independence_representation_error": (
            phase57b_batch_independence_representation_error
        ),
        "representation_shape": list(
            phase57b_initial["representation"].shape
        ),
        "all_values_finite": True,
    },
    "augmentation_contract": {
        "per_case_seeded": True,
        "replay_maximum_error": phase57b_augmentation_replay_error,
        "mean_absolute_change": phase57b_augmentation_change,
        "all_values_finite": True,
    },
    "two_step_backward_contract": {
        "records": phase57b_backward_records,
        "head_receives_first_step_gradient": True,
        "encoder_receives_second_step_gradient": True,
        "all_gradients_finite": True,
    },
    "training_objective_plan": {
        "group_balanced_binary_cross_entropy": True,
        "group_dro_over_training_groups": True,
        "bounded_within_group_pairwise_rank_loss": True,
        "anchor_identity_epoch_zero_allowed": True,
    },
    "scanner_augmentation": PHASE57B_CONFIG["scanner_augmentation"],
    "planned_screen": PHASE57B_CONFIG["planned_screen"],
    "validation": phase57b_contract_core["validation"],
    "test_rule": phase57b_contract_core["test_rule"],
    "contract_sha256": phase57b_contract_sha256,
    "device": str(phase57b_device),
    "mixed_precision": (
        "bfloat16" if phase57b_cuda else "disabled_on_cpu"
    ),
    "peak_vram_mb": phase57b_peak_vram_mb,
    "fit_labels_used_for_backward_contract": True,
    "selection_labels_used": False,
    "logo_labels_used": False,
    "public_leaderboard_used": False,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase57b_started, 3),
}

PHASE57_SCANNER_ROBUST_REPORT_PRIVATE = dict(phase57b_report)

print("BEGIN SANITIZED_PHASE57_ARCHITECTURE")
print(json.dumps(phase57b_report, indent=2))
print("END SANITIZED_PHASE57_ARCHITECTURE")
