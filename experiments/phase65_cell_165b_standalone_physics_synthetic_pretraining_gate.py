from __future__ import annotations

import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# ---------------------------------------------------------------------------
# Standalone Phase65B restoration boundary
# ---------------------------------------------------------------------------
#
# Production execution requires no live notebook variables.  The exact state
# formerly supplied by Phases57-64 is reconstructed from immutable training
# artifacts, then parity-checked before any model or voxel cache is used for
# fitting.  A private synthetic override remains available only for local unit
# tests; it is never active in the competition notebook unless explicitly set.

PHASE65B_WORKING_ROOT = Path(
    os.environ.get("PHASE65B_WORKING_ROOT_PRIVATE", "/kaggle/working")
).resolve()
PHASE65B_PRODUCTION_RESTORE_USED = False
PHASE65B_RESTORE_SUMMARY_PRIVATE = None


class Phase65BRestoreStop(RuntimeError):
    pass


def phase65b_require(condition, stage):
    if not bool(condition):
        raise Phase65BRestoreStop(str(stage))


def phase65b_sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def phase65b_json(path, stage):
    path = Path(path)
    phase65b_require(path.is_file(), f"missing_{stage}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase65BRestoreStop(f"invalid_json_{stage}") from exc
    phase65b_require(isinstance(value, dict), f"non_object_{stage}")
    return value


def phase65b_contract_core_hash_valid(payload):
    phase65b_require(isinstance(payload, dict), "invalid_contract_payload")
    claimed = payload.get("contract_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256" and not key.startswith("contains_")
    }
    recomputed = hashlib.sha256(json.dumps(
        core, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")).hexdigest()
    return isinstance(claimed, str) and claimed == recomputed


def phase65b_report_hash_valid(payload):
    phase65b_require(isinstance(payload, dict), "invalid_report_payload")
    claimed = payload.get("contract_sha256")
    core = dict(payload)
    core.pop("contract_sha256", None)
    core.pop("contract_file", None)
    recomputed = hashlib.sha256(json.dumps(
        core,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")).hexdigest()
    return isinstance(claimed, str) and claimed == recomputed


def phase65b_logit_restore(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    return np.log(probability) - np.log1p(-probability)


def phase65b_sigmoid_restore(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase65b_auc_restore(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    phase65b_require(labels.shape == score.shape, "restore_metric_shape")
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    rank = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        rank[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    positive = labels == 1
    positive_n = int(np.sum(positive))
    negative_n = int(labels.size - positive_n)
    phase65b_require(positive_n > 0 and negative_n > 0, "restore_metric_classes")
    statistic = float(np.sum(rank[positive]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def phase65b_metrics_restore(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    phase65b_require(labels.shape == probability.shape, "restore_metric_shape")
    loss = -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )
    return {
        "log_loss": float(np.mean(loss)),
        "auroc": phase65b_auc_restore(labels, probability),
        "mean_probability": float(np.mean(probability)),
    }


def phase65b_load_vector(description, candidates, expected_n=1362):
    existing = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen and candidate.is_file():
            seen.add(key)
            existing.append(candidate)
    phase65b_require(bool(existing), f"missing_vector_{description}")
    reference = np.asarray(
        np.load(existing[0], allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    phase65b_require(
        reference.shape == (expected_n,), f"shape_vector_{description}"
    )
    phase65b_require(np.all(np.isfinite(reference)), f"finite_vector_{description}")
    for duplicate_path in existing[1:]:
        duplicate = np.asarray(
            np.load(duplicate_path, allow_pickle=False), dtype=np.float64
        ).reshape(-1)
        phase65b_require(
            duplicate.shape == reference.shape
            and float(np.max(np.abs(duplicate - reference))) <= 1.0e-12,
            f"duplicate_vector_mismatch_{description}",
        )
    return reference


def phase65b_restore_labels(root):
    root_candidates = []
    private_root = os.environ.get("DAT_PRIVATE_ROOT")
    if private_root:
        root_candidates.append(Path(private_root))
    candidates = []
    seen = set()
    for candidate_root in root_candidates:
        candidate = candidate_root / "train_labels.csv"
        if candidate.is_file() and str(candidate.resolve()) not in seen:
            seen.add(str(candidate.resolve()))
            candidates.append(candidate)
    if not candidates and Path("/kaggle/input").is_dir():
        for candidate in sorted(Path("/kaggle/input").glob("**/train_labels.csv")):
            lowered = [part.lower() for part in candidate.parts]
            if any("test" in part or "smoke" in part for part in lowered):
                continue
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(candidate)

    records = []
    for candidate in candidates:
        try:
            with candidate.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        if not rows or list(rows[0].keys()) != ["uid", "is_pathologic"]:
            continue
        if len(rows) != 1362:
            continue
        uids = [str(row["uid"]) for row in rows]
        if len(set(uids)) != 1362 or any(not uid for uid in uids):
            continue
        try:
            labels = np.asarray(
                [int(float(row["is_pathologic"])) for row in rows],
                dtype=np.int64,
            )
        except Exception:
            continue
        if not np.all(np.isin(labels, [0, 1])):
            continue
        records.append((uids, labels))
    phase65b_require(bool(records), "training_labels_not_resolved")
    reference_uids, reference_labels = records[0]
    for duplicate_uids, duplicate_labels in records[1:]:
        phase65b_require(
            duplicate_uids == reference_uids,
            "training_label_duplicate_uid_order_mismatch",
        )
        phase65b_require(
            np.array_equal(duplicate_labels, reference_labels),
            "training_label_duplicate_value_mismatch",
        )
    phase65b_require(
        int(np.sum(reference_labels == 0)) == 615
        and int(np.sum(reference_labels == 1)) == 747,
        "training_label_count_mismatch",
    )
    return reference_labels


def phase65b_read_phase42_gate(root):
    direct = [
        root / "phase56_experimental_submission/phase42_assets/phase42_hierarchical_gate.json",
        root / "phase43_experimental_submission/phase42_assets/phase42_hierarchical_gate.json",
        root / "phase42_experimental_submission/phase42_assets/phase42_hierarchical_gate.json",
        root / "phase42_hierarchical_gate.json",
    ]
    for candidate in direct:
        if candidate.is_file():
            return phase65b_json(candidate, "phase42_gate")
    for archive_path in (
        root / "phase56_submission.zip",
        root / "phase43_submission.zip",
        root / "phase42_submission.zip",
    ):
        if not archive_path.is_file():
            continue
        with zipfile.ZipFile(archive_path) as archive:
            member = "phase42_assets/phase42_hierarchical_gate.json"
            if member in archive.namelist():
                payload = json.loads(archive.read(member).decode("utf-8"))
                phase65b_require(isinstance(payload, dict), "invalid_phase42_gate")
                return payload
    raise Phase65BRestoreStop("phase42_gate_not_resolved")


def phase65b_read_router_fold(root):
    stage_candidates = [
        root / "phase56_experimental_submission/models/phase30_acquisition_router.npz",
        root / "phase43_experimental_submission/models/phase30_acquisition_router.npz",
    ]
    for candidate in stage_candidates:
        if candidate.is_file():
            with np.load(candidate, allow_pickle=False) as router:
                return np.asarray(router["fold_for_group"], dtype=np.int64)
    archive_path = root / "phase56_submission.zip"
    phase65b_require(archive_path.is_file(), "missing_phase56_submission")
    with zipfile.ZipFile(archive_path) as archive:
        member = "models/phase30_acquisition_router.npz"
        phase65b_require(member in archive.namelist(), "missing_router_in_phase56")
        with np.load(io.BytesIO(archive.read(member)), allow_pickle=False) as router:
            return np.asarray(router["fold_for_group"], dtype=np.int64)


def phase65b_group_count(channels):
    channels = int(channels)
    for candidate in (8, 4, 2):
        if channels % candidate == 0:
            return candidate
    return 1


class Phase65BResidualBlock3D(nn.Module):
    def __init__(self, channels):
        super().__init__()
        groups = phase65b_group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1, bias=False)

    def forward(self, value):
        residual = value
        value = self.conv1(F.silu(self.norm1(value), inplace=False))
        value = self.conv2(F.silu(self.norm2(value), inplace=False))
        return residual + value


class Phase65BDownsample3D(nn.Module):
    def __init__(self, input_channels, output_channels):
        super().__init__()
        self.norm = nn.GroupNorm(
            phase65b_group_count(input_channels), input_channels
        )
        self.conv = nn.Conv3d(
            input_channels, output_channels, 3, stride=2, padding=1, bias=False
        )

    def forward(self, value):
        return self.conv(F.silu(self.norm(value), inplace=False))


class Phase65BMultiscaleEncoder3D(nn.Module):
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
        phase65b_require(
            len(channels) == len(blocks_per_stage) == 4,
            "invalid_architecture_stage_count",
        )
        self.stem = nn.Sequential(
            nn.Conv3d(input_channels, channels[0], 5, stride=2, padding=2, bias=False),
            nn.GroupNorm(phase65b_group_count(channels[0]), channels[0]),
            nn.SiLU(inplace=False),
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        self.projections = nn.ModuleList()
        for stage_index, (width, block_count) in enumerate(
            zip(channels, blocks_per_stage)
        ):
            self.stages.append(nn.Sequential(*[
                Phase65BResidualBlock3D(width) for _ in range(block_count)
            ]))
            self.projections.append(nn.Sequential(
                nn.Linear(2 * width, projection_dimension),
                nn.LayerNorm(projection_dimension),
                nn.SiLU(inplace=False),
            ))
            if stage_index + 1 < len(channels):
                self.downsamples.append(
                    Phase65BDownsample3D(width, channels[stage_index + 1])
                )
        self.output_dimension = len(channels) * int(projection_dimension)

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


class Phase65BScannerRobustBilateral3D(nn.Module):
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
        self.encoder = Phase65BMultiscaleEncoder3D(
            input_channels=3,
            channels=channels,
            blocks_per_stage=blocks_per_stage,
            projection_dimension=projection_dimension,
        )
        bilateral_dimension = 2 * self.encoder.output_dimension
        self.head = nn.Sequential(
            nn.LayerNorm(bilateral_dimension),
            nn.Linear(bilateral_dimension, head_hidden_dimension),
            nn.SiLU(inplace=False),
            nn.Linear(head_hidden_dimension, 1),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    @staticmethod
    def normalize_per_case(volume):
        value = torch.clamp(volume.float(), 0.0, 1.5) / 1.5
        mean = value.mean(dim=(2, 3, 4), keepdim=True)
        variance = torch.mean(
            torch.square(value - mean), dim=(2, 3, 4), keepdim=True
        )
        return torch.clamp(
            (value - mean) / torch.sqrt(variance + 1.0e-6), -6.0, 6.0
        )

    def make_bilateral_inputs(self, volume):
        value = self.normalize_per_case(volume)
        reflected = torch.flip(value, dims=[self.left_right_dimension])
        symmetric = 0.5 * (value + reflected)
        antisymmetric = torch.abs(value - reflected)
        return (
            torch.cat([value, symmetric, antisymmetric], dim=1),
            torch.cat([reflected, symmetric, antisymmetric], dim=1),
        )

    def forward(self, volume, anchor_logit):
        phase65b_require(
            volume.ndim == 5 and volume.shape[1] == 1,
            "invalid_model_input_shape",
        )
        anchor_logit = anchor_logit.reshape(-1).to(
            device=volume.device, dtype=torch.float32
        )
        original, reflected = self.make_bilateral_inputs(volume)
        encoded = self.encoder(torch.cat([original, reflected], dim=0))
        original_embedding, reflected_embedding = encoded.chunk(2, dim=0)
        representation = torch.cat([
            0.5 * (original_embedding + reflected_embedding),
            torch.abs(original_embedding - reflected_embedding),
        ], dim=1)
        raw_residual = self.head(representation).reshape(-1).float()
        logit = anchor_logit + self.residual_cap * torch.tanh(raw_residual)
        return {
            "logit": logit,
            "probability": torch.sigmoid(logit),
            "representation": representation,
        }


def phase65b_uniform_cpu(generator, lower, upper):
    return float(lower + (upper - lower) * torch.rand(
        (), generator=generator
    ).item())


def phase65b_gaussian_kernel_1d(sigma, device, dtype):
    if sigma <= 1.0e-6:
        return torch.ones(1, device=device, dtype=dtype)
    coordinate = torch.arange(-2, 3, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * torch.square(coordinate / sigma))
    return kernel / torch.sum(kernel)


def phase65b_anisotropic_blur(volume, sigmas):
    value = volume
    for spatial_index, sigma in enumerate(sigmas):
        if sigma <= 1.0e-6:
            continue
        kernel = phase65b_gaussian_kernel_1d(
            sigma, value.device, value.dtype
        )
        shape = [1, 1, 1, 1, 1]
        shape[2 + spatial_index] = kernel.numel()
        padding = [0, 0, 0]
        padding[spatial_index] = kernel.numel() // 2
        value = F.conv3d(value, kernel.reshape(shape), padding=tuple(padding))
    return value


def phase65b_rotation_matrix(angles):
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


def phase65b_augment_one(volume, seed, config):
    phase65b_require(
        volume.ndim == 5 and volume.shape[0] == volume.shape[1] == 1,
        "invalid_augmentation_input",
    )
    spatial_size = tuple(int(value) for value in volume.shape[2:])
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    value = torch.clamp(volume.float(), 0.0, 1.5)
    scale = phase65b_uniform_cpu(generator, *config["intensity_scale"])
    gamma = phase65b_uniform_cpu(generator, *config["gamma"])
    value = 1.5 * torch.pow(
        torch.clamp(value / 1.5, 0.0, 1.0), gamma
    ) * scale
    poisson_count = phase65b_uniform_cpu(generator, *config["poisson_count"])
    cpu_value = value.detach().cpu()
    cpu_value = torch.poisson(
        torch.clamp(cpu_value / 1.5, 0.0, 1.0) * poisson_count,
        generator=generator,
    ) / poisson_count
    value = (1.5 * cpu_value).to(volume.device)
    noise_std = phase65b_uniform_cpu(
        generator, *config["gaussian_noise_std"]
    )
    noise = torch.randn(
        value.shape, generator=generator, dtype=torch.float32
    ).to(value.device)
    value = value + noise_std * noise
    sigma = phase65b_uniform_cpu(generator, *config["blur_sigma"])
    anisotropy = [
        sigma * phase65b_uniform_cpu(generator, 0.65, 1.35)
        for _ in range(3)
    ]
    value = phase65b_anisotropic_blur(value, anisotropy)
    resolution_scale = phase65b_uniform_cpu(
        generator, *config["resolution_scale"]
    )
    if resolution_scale < 0.995:
        reduced = tuple(max(4, int(round(size * resolution_scale))) for size in spatial_size)
        value = F.interpolate(
            value, size=reduced, mode="trilinear", align_corners=False
        )
        value = F.interpolate(
            value, size=spatial_size, mode="trilinear", align_corners=False
        )
    maximum_angle = math.radians(float(config["rotation_degrees"]))
    angles = torch.tensor([
        phase65b_uniform_cpu(generator, -maximum_angle, maximum_angle)
        for _ in range(3)
    ], dtype=torch.float32)
    rotation = phase65b_rotation_matrix(angles)
    maximum_translation = float(config["translation_voxels"])
    translation = torch.tensor([
        phase65b_uniform_cpu(generator, -maximum_translation, maximum_translation)
        * (2.0 / max(1.0, float(spatial_size[index] - 1)))
        for index in range(3)
    ], dtype=torch.float32)
    theta = torch.cat([rotation, translation[:, None]], dim=1)[None].to(value.device)
    grid = F.affine_grid(theta, value.shape, align_corners=False)
    value = F.grid_sample(
        value,
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    if phase65b_uniform_cpu(generator, 0.0, 1.0) < float(
        config["reflection_probability"]
    ):
        value = torch.flip(value, dims=[2])
    return torch.clamp(value, 0.0, 1.5)


PHASE65B_ARCHITECTURE_CONFIG = {
    "channels": [16, 32, 64, 112],
    "blocks_per_stage": [1, 1, 2, 2],
    "multiscale_projection_dimension": 48,
    "head_hidden_dimension": 96,
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
}


def phase65b_restore_production_state():
    root = PHASE65B_WORKING_ROOT

    # Accepted deployment boundary must be present and internally hashed.
    ar4 = phase65b_json(
        root / "phase65ar4_corrected_offline_gate_adjudication_contract.json",
        "phase65ar4_contract",
    )
    phase65b_require(
        ar4.get("status") == (
            "accepted_corrected_offline_deployment_integrity_numeric_parity_"
            "not_available_ready_for_corrected_phase65b"
        )
        or ar4.get("status") == (
            "accepted_corrected_offline_and_exact_phase56_numeric_parity_"
            "ready_for_corrected_phase65b"
        ),
        "phase65ar4_not_accepted",
    )
    phase65b_require(phase65b_report_hash_valid(ar4), "phase65ar4_hash_mismatch")
    phase65b_require(
        ar4.get("gates", {}).get(
            "corrected_offline_deployment_integrity_passed"
        ) is True,
        "phase65ar4_offline_gate_not_passed",
    )
    submission_zip = root / "phase56_submission.zip"
    phase65b_require(submission_zip.is_file(), "missing_phase56_submission")
    submission_sha = phase65b_sha_file(submission_zip)
    phase65b_require(
        ar4.get("artifact_provenance", {}).get("submission_zip", {}).get("sha256")
        == submission_sha,
        "phase56_digest_changed_after_phase65ar4",
    )

    phase62 = phase65b_json(
        root / "phase62_acquisition_domain_validation_contract.json",
        "phase62_contract",
    )
    phase65b_require(
        phase62.get("status") == "accepted_validation_reset_ready_for_phase62b",
        "phase62_validation_not_accepted",
    )
    phase65b_require(
        phase65b_contract_core_hash_valid(phase62), "phase62_contract_hash_mismatch"
    )
    phase64 = phase65b_json(
        root / "phase64_group_blocked_diffusion_nystrom_contract.json",
        "phase64_contract",
    )
    phase65b_require(
        phase64.get("status")
        == "repeated_group_blocked_gate_failed_stop_diffusion_candidate",
        "phase64_rejection_not_preserved",
    )
    phase65b_require(
        phase65b_contract_core_hash_valid(phase64), "phase64_contract_hash_mismatch"
    )
    phase65a = phase65b_json(
        root / "phase65a_corrected_deployment_fallback_contract.json",
        "phase65a_contract",
    )
    phase65b_require(
        phase65a.get("status")
        == "fallback_numeric_gate_failed_retain_phase12c_unknown_fallback",
        "phase65a_fallback_decision_not_preserved",
    )
    phase65b_require(
        phase65b_contract_core_hash_valid(phase65a), "phase65a_contract_hash_mismatch"
    )

    labels = phase65b_restore_labels(root)

    phase50_path = root / "phase50_private_validation/phase50_partitions.npz"
    phase52_path = root / "phase52_repeated_partition.npz"
    phase52_contract = phase65b_json(
        root / "phase52_support_gate_contract.json", "phase52_contract"
    )
    phase65b_require(phase50_path.is_file(), "missing_phase50_partition")
    phase65b_require(phase52_path.is_file(), "missing_phase52_partition")
    with np.load(phase50_path, allow_pickle=False) as partition50:
        groups = np.asarray(
            partition50["acquisition_group"], dtype=np.int64
        ).reshape(-1)
    with np.load(phase52_path, allow_pickle=False) as partition52:
        fold_assignment = np.asarray(
            partition52["fold_assignment"], dtype=np.int64
        )
        phase52_sha = str(np.asarray(partition52["contract_sha256"]).item())
    phase65b_require(groups.shape == (1362,), "phase50_group_shape_mismatch")
    expected_group_sizes = [
        39, 456, 145, 255, 76, 7, 49, 208, 32, 4, 35, 10, 32, 9, 5
    ]
    phase65b_require(
        np.bincount(groups, minlength=15).tolist() == expected_group_sizes,
        "phase50_group_counts_mismatch",
    )
    phase65b_require(
        fold_assignment.shape == (5, 1362), "phase52_fold_shape_mismatch"
    )
    phase65b_require(
        phase52_sha == str(phase52_contract.get("contract_sha256")),
        "phase52_contract_link_mismatch",
    )
    partition_digest = hashlib.sha256(np.ascontiguousarray(
        fold_assignment.astype(np.int8)
    ).tobytes()).hexdigest()
    phase65b_require(
        partition_digest == str(phase52_contract.get("phase52_partition_sha256")),
        "phase52_partition_digest_mismatch",
    )
    regenerated = np.full((5, 1362), -1, dtype=np.int64)
    for repeat, seed in enumerate([520101, 520102, 520103, 520104, 520105]):
        rng = np.random.default_rng(seed)
        for group in range(15):
            for label in (0, 1):
                indices = np.flatnonzero((groups == group) & (labels == label)).copy()
                if not indices.size:
                    continue
                rng.shuffle(indices)
                cycle = (
                    np.arange(indices.size, dtype=np.int64)
                    + int(rng.integers(0, 5))
                ) % 5
                regenerated[repeat, indices] = cycle
    phase65b_require(
        np.array_equal(regenerated, fold_assignment),
        "training_label_cache_row_order_mismatch",
    )

    phase12 = phase65b_load_vector(
        "phase12c_oof", [root / "phase32_phase12c_oof_float64.npy"]
    )
    phase33 = phase65b_load_vector(
        "phase33_component_oof",
        [
            root / "phase33_private_checkpoint/phase33_component_oof_float64.npy",
            root / "phase33_component_oof_float64.npy",
        ],
    )
    combined39 = phase65b_load_vector(
        "phase39_combined_residual",
        [
            root / "phase39_private_checkpoint/phase39_combined_residual_float64.npy",
            root / "phase39_combined_residual_float64.npy",
        ],
    )
    phase39 = phase65b_load_vector(
        "phase39_oof",
        [
            root / "phase39_private_checkpoint/phase39_oof_float64.npy",
            root / "phase39_oof_float64.npy",
        ],
    )
    phase12_logit = phase65b_logit_restore(phase12)
    phase33_residual = phase65b_logit_restore(phase33) - phase12_logit
    phase39_reconstructed = phase65b_sigmoid_restore(
        phase12_logit + np.clip(combined39, -2.0, 2.0)
    )
    phase65b_require(
        float(np.max(np.abs(phase39_reconstructed - phase39))) <= 2.0e-12,
        "phase39_reconstruction_mismatch",
    )
    gate42 = phase65b_read_phase42_gate(root)
    phase65b_require(
        float(gate42.get("phase36_raw_residual_weight")) == 0.75
        and float(gate42.get("phase33_logit_residual_weight")) == 0.125
        and float(gate42.get("residual_cap")) == 1.0,
        "phase42_gate_constants_mismatch",
    )
    alpha_map = {
        int(group): float(alpha)
        for group, alpha in gate42["known_group_alpha"].items()
    }
    phase65b_require(set(alpha_map) == set(range(15)), "phase42_alpha_map_mismatch")
    group_alpha = np.asarray([alpha_map[int(group)] for group in groups])
    phase42_uncapped = combined39 - 0.125 * phase33_residual
    anchor = phase65b_sigmoid_restore(
        phase12_logit + group_alpha * np.clip(phase42_uncapped, -1.0, 1.0)
    )
    anchor_metrics = phase65b_metrics_restore(labels, anchor)
    phase65b_require(
        abs(anchor_metrics["log_loss"] - 0.29016628416289664) <= 2.0e-10
        and abs(anchor_metrics["auroc"] - 0.9464742438589044) <= 2.0e-12
        and abs(anchor_metrics["mean_probability"] - 0.5500570231688339)
        <= 2.0e-10,
        "phase43_anchor_metric_parity_mismatch",
    )

    logo_path = root / "phase57_logo_partition.npz"
    phase57_contract = phase65b_json(
        root / "phase57_transport_contract.json", "phase57_transport_contract"
    )
    phase65b_require(logo_path.is_file(), "missing_phase57_logo_partition")
    with np.load(logo_path, allow_pickle=False) as logo:
        fold_for_group = np.asarray(
            logo["original_fold_for_group"], dtype=np.int64
        ).reshape(-1)
        logo_contract_sha = str(np.asarray(logo["contract_sha256"]).item())
    phase65b_require(
        phase65b_contract_core_hash_valid(phase57_contract),
        "phase57_transport_contract_hash_mismatch",
    )
    phase65b_require(
        logo_contract_sha == phase57_contract.get("contract_sha256"),
        "phase57_logo_contract_link_mismatch",
    )
    router_fold = phase65b_read_router_fold(root).reshape(-1)
    phase65b_require(
        fold_for_group.shape == (15,)
        and np.array_equal(fold_for_group, router_fold),
        "phase57_router_fold_mismatch",
    )
    original_fold = fold_for_group[groups]
    phase65b_require(
        np.bincount(original_fold, minlength=3).tolist() == [467, 443, 452],
        "phase57_original_fold_counts_mismatch",
    )
    for group in range(15):
        phase65b_require(
            np.unique(original_fold[groups == group]).size == 1,
            "phase57_group_split_leakage",
        )

    group_sizes = np.bincount(groups, minlength=15)
    individual_groups = [
        int(group) for group, count in enumerate(group_sizes) if int(count) >= 30
    ]
    tiny_groups = [group for group in range(15) if group not in individual_groups]
    domain_definitions = [
        {"name": f"group_{group}", "groups": [group]}
        for group in individual_groups
    ] + [{"name": "tiny_groups_pooled", "groups": tiny_groups}]
    phase65b_require(
        domain_definitions == phase62.get("stress_domain_definitions"),
        "phase62_stress_domain_definition_mismatch",
    )
    domain_id = np.full(1362, -1, dtype=np.int64)
    for domain, definition in enumerate(domain_definitions):
        domain_id[np.isin(groups, definition["groups"])] = domain
    phase65b_require(np.all(domain_id >= 0), "phase62_domain_assignment_incomplete")

    cache_path = root / "phase31_highres_float16.npy"
    phase65b_require(cache_path.is_file(), "missing_phase31_highres_cache")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    phase65b_require(
        cache.shape == (1362, 80, 80, 80) and cache.dtype == np.float16,
        "phase31_highres_cache_contract_mismatch",
    )
    del cache

    validation_state = {
        "contract_sha256": str(phase62["contract_sha256"]),
        "labels": labels.copy(),
        "groups": groups.copy(),
        "original_fold": original_fold.copy(),
        "anchor_probability": anchor.copy(),
        "stress_domain_id": domain_id.copy(),
        "stress_domain_definitions": json.loads(json.dumps(domain_definitions)),
    }
    return {
        "validation_state": validation_state,
        "engine_state": {"highres_cache_file": str(cache_path)},
        "phase64_report": {
            "status": "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
        },
        "architecture_config": json.loads(json.dumps(PHASE65B_ARCHITECTURE_CONFIG)),
        "model_class": Phase65BScannerRobustBilateral3D,
        "augmentation_function": phase65b_augment_one,
        "submission_sha256": submission_sha,
        "anchor_metrics": anchor_metrics,
        "phase62_contract_sha256": str(phase62["contract_sha256"]),
        "phase64_contract_sha256": str(phase64["contract_sha256"]),
        "phase65a_contract_sha256": str(phase65a["contract_sha256"]),
        "phase65ar4_contract_sha256": str(ar4["contract_sha256"]),
        "row_order_verified": True,
        "router_fold_verified": True,
        "cache_contract_verified": True,
    }


phase65b_required_live_names = (
    "PHASE64A_DIFFUSION_REPORT_PRIVATE",
    "PHASE62_VALIDATION_RESET_STATE_PRIVATE",
    "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE",
    "phase57c_model_class",
    "phase57c_architecture_config",
    "phase57c_augment_one",
    "phase57c_device",
    "phase57c_cuda",
    "phase57c_amp_dtype",
)
phase65b_test_override = globals().get("PHASE65A_SYNTHETIC_TEST_OVERRIDE_PRIVATE")
phase65b_test_state_supplied = bool(
    isinstance(phase65b_test_override, dict)
    and all(name in globals() for name in phase65b_required_live_names)
)

if not phase65b_test_state_supplied:
    phase65b_restored = phase65b_restore_production_state()
    PHASE64A_DIFFUSION_REPORT_PRIVATE = phase65b_restored["phase64_report"]
    PHASE62_VALIDATION_RESET_STATE_PRIVATE = phase65b_restored["validation_state"]
    PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE = phase65b_restored["engine_state"]
    phase57c_model_class = phase65b_restored["model_class"]
    phase57c_architecture_config = phase65b_restored["architecture_config"]
    phase57c_augment_one = phase65b_restored["augmentation_function"]
    phase57c_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    phase57c_cuda = phase57c_device.type == "cuda"
    phase57c_amp_dtype = torch.bfloat16
    phase65b_require(phase57c_cuda, "cuda_required_for_phase65b_production")
    PHASE65B_PRODUCTION_RESTORE_USED = True
    PHASE65B_RESTORE_SUMMARY_PRIVATE = phase65b_restored
else:
    PHASE65B_RESTORE_SUMMARY_PRIVATE = {
        "row_order_verified": True,
        "router_fold_verified": True,
        "cache_contract_verified": True,
        "synthetic_test_state": True,
    }


# Cell 165B — restart-safe physics-informed synthetic pretraining followed by
# one frozen three-fold acquisition-group-held-out candidate. No previous cell
# or live variable is required in production. Synthetic pretraining is purely
# procedural and never observes a real image or label. Every real OOF model is
# fitted only on the other two accepted acquisition-group folds. Epoch count,
# EMA, architecture, optimizer, and blend are frozen before OOF evaluation.

phase65a_started = time.perf_counter()

for phase65a_required_name in (
    "PHASE64A_DIFFUSION_REPORT_PRIVATE",
    "PHASE62_VALIDATION_RESET_STATE_PRIVATE",
    "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE",
    "phase57c_model_class",
    "phase57c_architecture_config",
    "phase57c_augment_one",
    "phase57c_device",
    "phase57c_cuda",
    "phase57c_amp_dtype",
):
    assert phase65a_required_name in globals(), {
        "missing": phase65a_required_name
    }

assert PHASE64A_DIFFUSION_REPORT_PRIVATE["status"] == (
    "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
)

phase65a_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE
phase65a_engine = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase65a_labels = np.asarray(
    phase65a_validation["labels"], dtype=np.int64
).reshape(-1)
phase65a_groups = np.asarray(
    phase65a_validation["groups"], dtype=np.int64
).reshape(-1)
phase65a_original_fold = np.asarray(
    phase65a_validation["original_fold"], dtype=np.int64
).reshape(-1)
phase65a_anchor = np.asarray(
    phase65a_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase65a_domain_id = np.asarray(
    phase65a_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase65a_domain_definitions = phase65a_validation[
    "stress_domain_definitions"
]

phase65a_test_override = globals().get(
    "PHASE65A_SYNTHETIC_TEST_OVERRIDE_PRIVATE"
)
phase65a_is_test = isinstance(phase65a_test_override, dict)
phase65a_expected_n = (
    int(phase65a_test_override["case_count"])
    if phase65a_is_test else 1362
)
phase65a_expected_group_count = (
    int(phase65a_test_override.get("group_count", 15))
    if phase65a_is_test else 15
)

assert phase65a_labels.shape == phase65a_groups.shape
assert phase65a_labels.shape == phase65a_original_fold.shape
assert phase65a_labels.shape == phase65a_anchor.shape
assert phase65a_labels.shape == phase65a_domain_id.shape
assert phase65a_labels.shape == (phase65a_expected_n,)
assert set(np.unique(phase65a_labels).tolist()) == {0, 1}
assert np.array_equal(
    np.unique(phase65a_groups),
    np.arange(phase65a_expected_group_count, dtype=np.int64),
)
assert np.array_equal(np.unique(phase65a_original_fold), np.arange(3))
assert np.all(np.isfinite(phase65a_anchor))
assert np.all((phase65a_anchor > 0.0) & (phase65a_anchor < 1.0))
for phase65a_group in range(phase65a_expected_group_count):
    assert np.unique(
        phase65a_original_fold[phase65a_groups == phase65a_group]
    ).size == 1


PHASE65A_CONFIG = {
    "schema_version": "phase65b_standalone_physics_synthetic_pretraining_gate_v2",
    "seed": 650165,
    "input_shape": [80, 80, 80],
    "synthetic_batch_size": 8,
    "synthetic_pretraining_steps": 800,
    "synthetic_auxiliary_weight": 0.35,
    "synthetic_learning_rate": 3.0e-4,
    "synthetic_weight_decay": 0.01,
    "real_batch_size": 8,
    "real_worker_count": 2,
    "real_epoch_count": 12,
    "encoder_learning_rate": 5.0e-5,
    "head_learning_rate": 2.0e-4,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "warmup_fraction": 0.08,
    "minimum_learning_rate_multiplier": 0.05,
    "group_balance_mix": 0.25,
    "ema_decay": 0.995,
    "independent_logit_cap": 8.0,
    "anchor_logit_weight": 0.75,
    "synthetic_expert_logit_weight": 0.25,
    "major_groups": [1, 3],
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 650166,
    "probability_clip": 1.0e-7,
    "advancement_gate": {
        "minimum_pooled_log_loss_gain": 0.005,
        "minimum_pooled_auroc_gain": 0.0015,
        "minimum_fold_log_loss_wins": 2,
        "minimum_fold_auroc_wins": 2,
        "maximum_fold_log_loss_regret": 0.001,
        "minimum_major_groups_combined_log_loss_gain": 0.005,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
        "brier_regret_allowed": 0.0,
    },
    "contract_file": str(
        PHASE65B_WORKING_ROOT
        / "phase65b_physics_synthetic_pretraining_gate_contract.json"
    ),
    "checkpoint_directory": str(
        PHASE65B_WORKING_ROOT / "phase65b_private_checkpoint"
    ),
}

if phase65a_is_test:
    PHASE65A_CONFIG = {
        **PHASE65A_CONFIG,
        "input_shape": list(phase65a_test_override.get(
            "input_shape", [16, 16, 16]
        )),
        "synthetic_batch_size": int(
            phase65a_test_override.get("synthetic_batch_size", 4)
        ),
        "synthetic_pretraining_steps": int(
            phase65a_test_override.get("synthetic_pretraining_steps", 3)
        ),
        "real_batch_size": int(
            phase65a_test_override.get("real_batch_size", 6)
        ),
        "real_worker_count": 0,
        "real_epoch_count": int(
            phase65a_test_override.get("real_epoch_count", 1)
        ),
        "domain_bootstrap_replicates": int(
            phase65a_test_override.get("domain_bootstrap_replicates", 200)
        ),
        "contract_file": str(phase65a_test_override.get(
            "contract_file", "/tmp/phase65b_synthetic_contract.json"
        )),
        "checkpoint_directory": str(phase65a_test_override.get(
            "checkpoint_directory", "/tmp/phase65b_private_checkpoint"
        )),
    }

assert abs(
    PHASE65A_CONFIG["anchor_logit_weight"]
    + PHASE65A_CONFIG["synthetic_expert_logit_weight"]
    - 1.0
) <= 1.0e-15

phase65a_device = phase57c_device
phase65a_cuda = bool(phase57c_cuda)
phase65a_amp_dtype = phase57c_amp_dtype
phase65a_cache_file = str(phase65a_engine["highres_cache_file"])
phase65a_cache_path = Path(phase65a_cache_file)
assert phase65a_cache_path.is_file(), {"missing": phase65a_cache_file}
phase65a_cache_contract = np.load(
    phase65a_cache_path, mmap_mode="r", allow_pickle=False
)
assert phase65a_cache_contract.shape == (
    phase65a_expected_n, *PHASE65A_CONFIG["input_shape"]
)
assert phase65a_cache_contract.dtype == np.float16
del phase65a_cache_contract


def phase65b_array_sha256(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


phase65b_run_contract_core = {
    "schema_version": PHASE65A_CONFIG["schema_version"],
    "config": {
        key: value
        for key, value in PHASE65A_CONFIG.items()
        if key not in {"contract_file", "checkpoint_directory"}
    },
    "architecture": phase57c_architecture_config,
    "source_contract_sha256": str(phase65a_validation["contract_sha256"]),
    "label_sha256": phase65b_array_sha256(phase65a_labels.astype(np.int8)),
    "group_sha256": phase65b_array_sha256(phase65a_groups.astype(np.int8)),
    "fold_sha256": phase65b_array_sha256(phase65a_original_fold.astype(np.int8)),
    "anchor_sha256": phase65b_array_sha256(phase65a_anchor.astype(np.float64)),
    "cache_shape": [phase65a_expected_n, *PHASE65A_CONFIG["input_shape"]],
    "cache_dtype": "float16",
}
phase65b_run_sha256 = hashlib.sha256(json.dumps(
    phase65b_run_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()
phase65b_checkpoint_directory = (
    Path(PHASE65A_CONFIG["checkpoint_directory"])
    / phase65b_run_sha256[:16]
)
phase65b_checkpoint_directory.mkdir(parents=True, exist_ok=True)
phase65b_pretrain_checkpoint_path = (
    phase65b_checkpoint_directory / "procedural_pretraining.pt"
)
phase65b_pretraining_resumed_from_step = 0
phase65b_completed_folds_resumed = []
phase65b_invalid_checkpoint_count = 0


def phase65b_atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def phase65b_atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)

random.seed(PHASE65A_CONFIG["seed"])
np.random.seed(PHASE65A_CONFIG["seed"])
torch.manual_seed(PHASE65A_CONFIG["seed"])
if phase65a_cuda:
    torch.cuda.manual_seed_all(PHASE65A_CONFIG["seed"])


def phase65a_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE65A_CONFIG["probability_clip"],
        1.0 - PHASE65A_CONFIG["probability_clip"],
    )


def phase65a_logit(probability):
    probability = phase65a_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase65a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase65a_auc(labels, score):
    labels = np.asarray(labels, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    positive_n = int(np.sum(labels == 1))
    negative_n = int(np.sum(labels == 0))
    if positive_n == 0 or negative_n == 0:
        return None
    order = np.argsort(score, kind="mergesort")
    sorted_score = score[order]
    ranks = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    statistic = float(np.sum(ranks[labels == 1]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def phase65a_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase65a_clip(probability)
    return -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )


def phase65a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase65a_clip(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase65a_case_loss(labels, probability))),
        "auroc": phase65a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase65a_compare(labels, anchor, candidate):
    anchor_metrics = phase65a_metrics(labels, anchor)
    candidate_metrics = phase65a_metrics(labels, candidate)
    return {
        "n": int(labels.size),
        "anchor_log_loss": anchor_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            anchor_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "anchor_auroc": anchor_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": float(
            candidate_metrics["auroc"] - anchor_metrics["auroc"]
        ),
        "brier_gain": float(
            anchor_metrics["brier"] - candidate_metrics["brier"]
        ),
    }


def phase65a_group_weight(groups):
    groups = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(
        groups, minlength=phase65a_expected_group_count
    ).astype(np.float64)
    lookup = np.ones(phase65a_expected_group_count, dtype=np.float64)
    active = counts > 0.0
    lookup[active] = np.power(counts[active], -0.5)
    case_weight = lookup[groups]
    lookup /= float(np.mean(case_weight))
    return lookup.astype(np.float32)


def phase65a_build_model(seed, prevalence):
    torch.manual_seed(int(seed))
    if phase65a_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model = phase57c_model_class(
        channels=phase57c_architecture_config["channels"],
        blocks_per_stage=phase57c_architecture_config["blocks_per_stage"],
        projection_dimension=phase57c_architecture_config[
            "multiscale_projection_dimension"
        ],
        head_hidden_dimension=phase57c_architecture_config[
            "head_hidden_dimension"
        ],
        residual_cap=PHASE65A_CONFIG["independent_logit_cap"],
    ).to(phase65a_device)
    prevalence = float(np.clip(prevalence, 1.0e-4, 1.0 - 1.0e-4))
    prevalence_logit = math.log(prevalence / (1.0 - prevalence))
    raw_bias = np.arctanh(
        np.clip(
            prevalence_logit / PHASE65A_CONFIG["independent_logit_cap"],
            -0.999,
            0.999,
        )
    )
    with torch.no_grad():
        model.head[-1].bias.fill_(float(raw_bias))
    phase65b_require(
        not any(isinstance(module, (
            nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
        )) for module in model.modules()),
        "batch_normalization_forbidden",
    )
    phase65b_require(
        sum(parameter.numel() for parameter in model.parameters()) < 5_000_000,
        "model_parameter_ceiling_exceeded",
    )
    return model


def phase65a_representation(model, volume):
    original, reflected = model.make_bilateral_inputs(volume)
    encoded = model.encoder(torch.cat([original, reflected], dim=0))
    original_embedding, reflected_embedding = encoded.chunk(2, dim=0)
    return torch.cat([
        0.5 * (original_embedding + reflected_embedding),
        torch.abs(original_embedding - reflected_embedding),
    ], dim=1)


class Phase65RealDataset(Dataset):
    def __init__(self, indices, augment, seed):
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.augment = bool(augment)
        self.seed = int(seed)
        self.epoch = 0
        self.cache = None

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.indices.size)

    def resolve_cache(self):
        if self.cache is None:
            self.cache = np.load(
                phase65a_cache_file, mmap_mode="r", allow_pickle=False
            )
        return self.cache

    def __getitem__(self, item):
        case_index = int(self.indices[int(item)])
        volume = np.asarray(
            self.resolve_cache()[case_index], dtype=np.float32
        ).copy()
        tensor = torch.from_numpy(volume)[None]
        if self.augment:
            augmentation_seed = (
                self.seed + self.epoch * 1_000_003 + case_index * 97
            )
            tensor = phase57c_augment_one(
                tensor[None],
                augmentation_seed,
                phase57c_architecture_config["scanner_augmentation"],
            )[0]
        return {
            "volume": tensor,
            "label": torch.tensor(
                phase65a_labels[case_index], dtype=torch.float32
            ),
            "group": torch.tensor(
                phase65a_groups[case_index], dtype=torch.long
            ),
            "case_index": torch.tensor(case_index, dtype=torch.long),
        }


def phase65a_uniform(generator, shape, lower, upper):
    value = torch.rand(
        shape, generator=generator, device=phase65a_device,
        dtype=torch.float32,
    )
    return float(lower) + (float(upper) - float(lower)) * value


phase65a_grid_axes = [
    torch.linspace(
        -1.0, 1.0, int(size), device=phase65a_device,
        dtype=torch.float32,
    )
    for size in PHASE65A_CONFIG["input_shape"]
]
phase65a_grid_x, phase65a_grid_y, phase65a_grid_z = torch.meshgrid(
    *phase65a_grid_axes, indexing="ij"
)
phase65a_grid_x = phase65a_grid_x[None]
phase65a_grid_y = phase65a_grid_y[None]
phase65a_grid_z = phase65a_grid_z[None]


def phase65a_gaussian(cx, cy, cz, sx, sy, sz):
    return torch.exp(-0.5 * (
        torch.square((phase65a_grid_x - cx) / sx)
        + torch.square((phase65a_grid_y - cy) / sy)
        + torch.square((phase65a_grid_z - cz) / sz)
    ))


def phase65a_synthetic_batch(batch_size, generator):
    batch_size = int(batch_size)
    scalar_shape = (batch_size, 1, 1, 1)
    label = (
        phase65a_uniform(generator, (batch_size,), 0.0, 1.0) < 0.5
    ).float()
    abnormal = label[:, None, None, None]
    scenario = phase65a_uniform(generator, scalar_shape, 0.0, 1.0)
    affected_left = (
        phase65a_uniform(generator, scalar_shape, 0.0, 1.0) < 0.5
    ).float()

    normal_putamen = phase65a_uniform(
        generator, scalar_shape, 0.78, 1.12
    )
    normal_delta = phase65a_uniform(
        generator, scalar_shape, -0.08, 0.08
    )
    putamen_left = normal_putamen * (1.0 + normal_delta)
    putamen_right = normal_putamen * (1.0 - normal_delta)
    caudate_left = phase65a_uniform(
        generator, scalar_shape, 0.82, 1.18
    )
    caudate_right = caudate_left * (
        1.0 + phase65a_uniform(generator, scalar_shape, -0.07, 0.07)
    )

    unilateral = (scenario < 0.48).float() * abnormal
    bilateral = ((scenario >= 0.48) & (scenario < 0.82)).float() * abnormal
    weak_comma = (scenario >= 0.82).float() * abnormal
    affected_putamen = phase65a_uniform(
        generator, scalar_shape, 0.04, 0.42
    )
    spared_putamen = phase65a_uniform(
        generator, scalar_shape, 0.58, 0.94
    )
    bilateral_left = phase65a_uniform(
        generator, scalar_shape, 0.10, 0.55
    )
    bilateral_right = phase65a_uniform(
        generator, scalar_shape, 0.10, 0.55
    )
    weak_putamen_left = phase65a_uniform(
        generator, scalar_shape, 0.18, 0.62
    )
    weak_putamen_right = phase65a_uniform(
        generator, scalar_shape, 0.18, 0.62
    )

    abnormal_left = (
        unilateral * (
            affected_left * affected_putamen
            + (1.0 - affected_left) * spared_putamen
        )
        + bilateral * bilateral_left
        + weak_comma * weak_putamen_left
    )
    abnormal_right = (
        unilateral * (
            affected_left * spared_putamen
            + (1.0 - affected_left) * affected_putamen
        )
        + bilateral * bilateral_right
        + weak_comma * weak_putamen_right
    )
    putamen_left = (1.0 - abnormal) * putamen_left + abnormal_left
    putamen_right = (1.0 - abnormal) * putamen_right + abnormal_right

    abnormal_caudate_left = (
        (unilateral + bilateral) * phase65a_uniform(
            generator, scalar_shape, 0.62, 1.02
        )
        + weak_comma * phase65a_uniform(
            generator, scalar_shape, 0.32, 0.72
        )
    )
    abnormal_caudate_right = (
        (unilateral + bilateral) * phase65a_uniform(
            generator, scalar_shape, 0.62, 1.02
        )
        + weak_comma * phase65a_uniform(
            generator, scalar_shape, 0.32, 0.72
        )
    )
    caudate_left = (
        (1.0 - abnormal) * caudate_left + abnormal_caudate_left
    )
    caudate_right = (
        (1.0 - abnormal) * caudate_right + abnormal_caudate_right
    )

    blur_factor = phase65a_uniform(generator, scalar_shape, 0.80, 1.85)
    shift_x = phase65a_uniform(generator, scalar_shape, -0.035, 0.035)
    shift_y = phase65a_uniform(generator, scalar_shape, -0.055, 0.055)
    shift_z = phase65a_uniform(generator, scalar_shape, -0.045, 0.045)
    lateral = phase65a_uniform(generator, scalar_shape, 0.18, 0.26)
    caudate_y = phase65a_uniform(generator, scalar_shape, -0.12, -0.02)
    putamen_y = phase65a_uniform(generator, scalar_shape, 0.10, 0.22)
    side_jitter = phase65a_uniform(generator, scalar_shape, -0.025, 0.025)

    caudate_sigma = blur_factor
    putamen_sigma = blur_factor
    caudate_l = phase65a_gaussian(
        -lateral + shift_x,
        caudate_y + shift_y,
        shift_z + side_jitter,
        0.095 * caudate_sigma,
        0.105 * caudate_sigma,
        0.090 * caudate_sigma,
    )
    caudate_r = phase65a_gaussian(
        lateral + shift_x,
        caudate_y + shift_y,
        shift_z - side_jitter,
        0.095 * caudate_sigma,
        0.105 * caudate_sigma,
        0.090 * caudate_sigma,
    )
    putamen_l_1 = phase65a_gaussian(
        -lateral + shift_x,
        putamen_y + shift_y,
        shift_z + side_jitter,
        0.078 * putamen_sigma,
        0.155 * putamen_sigma,
        0.082 * putamen_sigma,
    )
    putamen_l_2 = phase65a_gaussian(
        -lateral + shift_x,
        putamen_y + 0.14 + shift_y,
        shift_z + side_jitter,
        0.066 * putamen_sigma,
        0.125 * putamen_sigma,
        0.076 * putamen_sigma,
    )
    putamen_r_1 = phase65a_gaussian(
        lateral + shift_x,
        putamen_y + shift_y,
        shift_z - side_jitter,
        0.078 * putamen_sigma,
        0.155 * putamen_sigma,
        0.082 * putamen_sigma,
    )
    putamen_r_2 = phase65a_gaussian(
        lateral + shift_x,
        putamen_y + 0.14 + shift_y,
        shift_z - side_jitter,
        0.066 * putamen_sigma,
        0.125 * putamen_sigma,
        0.076 * putamen_sigma,
    )

    background_amplitude = phase65a_uniform(
        generator, scalar_shape, 0.035, 0.22
    ) + weak_comma * phase65a_uniform(
        generator, scalar_shape, 0.02, 0.12
    )
    brain = phase65a_gaussian(
        shift_x, shift_y, shift_z,
        0.62 * torch.ones_like(blur_factor),
        0.72 * torch.ones_like(blur_factor),
        0.58 * torch.ones_like(blur_factor),
    )
    volume = background_amplitude * brain
    volume = volume + caudate_left * caudate_l + caudate_right * caudate_r
    volume = volume + putamen_left * torch.maximum(putamen_l_1, putamen_l_2)
    volume = volume + putamen_right * torch.maximum(putamen_r_1, putamen_r_2)

    intensity_scale = phase65a_uniform(generator, scalar_shape, 0.75, 1.25)
    gamma = phase65a_uniform(generator, scalar_shape, 0.72, 1.35)
    volume = 1.5 * torch.pow(
        torch.clamp(volume * intensity_scale / 1.5, 0.0, 1.0), gamma
    )
    poisson_count = phase65a_uniform(generator, scalar_shape, 35.0, 260.0)
    volume = 1.5 * torch.poisson(
        torch.clamp(volume / 1.5, 0.0, 1.0) * poisson_count,
        generator=generator,
    ) / poisson_count
    noise = torch.randn(
        volume.shape, generator=generator, device=phase65a_device,
        dtype=torch.float32,
    )
    noise_scale = phase65a_uniform(generator, scalar_shape, 0.0, 0.035)
    volume = torch.clamp(volume + noise_scale * noise, 0.0, 1.5)

    target = torch.cat([
        torch.clamp(0.5 * (putamen_left + putamen_right) / 1.2, 0.0, 1.0),
        torch.clamp(torch.minimum(putamen_left, putamen_right) / 1.2, 0.0, 1.0),
        torch.clamp(torch.abs(putamen_left - putamen_right) / 1.2, 0.0, 1.0),
        torch.clamp(0.5 * (caudate_left + caudate_right) / 1.2, 0.0, 1.0),
        torch.clamp(background_amplitude / 0.34, 0.0, 1.0),
        torch.clamp((blur_factor - 0.80) / 1.05, 0.0, 1.0),
    ], dim=1).reshape(batch_size, 6)
    return volume[:, None], label, target


phase65a_pretrain_model = phase65a_build_model(
    PHASE65A_CONFIG["seed"] + 10, 0.5
)
phase65a_representation_dimension = int(
    phase65a_pretrain_model.head[1].in_features
)
phase65a_auxiliary_head = nn.Sequential(
    nn.LayerNorm(phase65a_representation_dimension),
    nn.Linear(phase65a_representation_dimension, 96),
    nn.SiLU(inplace=False),
    nn.Linear(96, 7),
).to(phase65a_device)
phase65a_pretrain_optimizer = torch.optim.AdamW(
    list(phase65a_pretrain_model.encoder.parameters())
    + list(phase65a_auxiliary_head.parameters()),
    lr=PHASE65A_CONFIG["synthetic_learning_rate"],
    weight_decay=PHASE65A_CONFIG["synthetic_weight_decay"],
    betas=(0.9, 0.95),
)
phase65a_synthetic_generator = torch.Generator(device=phase65a_device)
phase65a_synthetic_generator.manual_seed(PHASE65A_CONFIG["seed"] + 20)
phase65a_pretrain_loss_sum = 0.0
phase65a_pretrain_classification_sum = 0.0
phase65a_pretrain_auxiliary_sum = 0.0
phase65a_pretrain_start_step = 0
synthetic_volume = synthetic_label = synthetic_target = None
representation = output = None

if phase65b_pretrain_checkpoint_path.is_file():
    try:
        phase65b_pretrain_checkpoint = torch.load(
            phase65b_pretrain_checkpoint_path,
            map_location=phase65a_device,
            weights_only=True,
        )
        phase65b_require(
            isinstance(phase65b_pretrain_checkpoint, dict)
            and phase65b_pretrain_checkpoint.get("run_sha256")
            == phase65b_run_sha256,
            "pretraining_checkpoint_run_mismatch",
        )
        phase65a_pretrain_start_step = int(
            phase65b_pretrain_checkpoint["step"]
        )
        phase65b_require(
            0 <= phase65a_pretrain_start_step
            <= PHASE65A_CONFIG["synthetic_pretraining_steps"],
            "pretraining_checkpoint_step_invalid",
        )
        phase65a_pretrain_model.encoder.load_state_dict(
            phase65b_pretrain_checkpoint["encoder_state"], strict=True
        )
        phase65a_auxiliary_head.load_state_dict(
            phase65b_pretrain_checkpoint["auxiliary_head_state"], strict=True
        )
        phase65a_pretrain_optimizer.load_state_dict(
            phase65b_pretrain_checkpoint["optimizer_state"]
        )
        phase65a_synthetic_generator.set_state(
            phase65b_pretrain_checkpoint["generator_state"]
        )
        phase65a_pretrain_loss_sum = float(
            phase65b_pretrain_checkpoint["total_loss_sum"]
        )
        phase65a_pretrain_classification_sum = float(
            phase65b_pretrain_checkpoint["classification_loss_sum"]
        )
        phase65a_pretrain_auxiliary_sum = float(
            phase65b_pretrain_checkpoint["auxiliary_loss_sum"]
        )
        phase65b_require(
            all(math.isfinite(value) for value in (
                phase65a_pretrain_loss_sum,
                phase65a_pretrain_classification_sum,
                phase65a_pretrain_auxiliary_sum,
            )),
            "pretraining_checkpoint_loss_invalid",
        )
        phase65b_pretraining_resumed_from_step = phase65a_pretrain_start_step
    except Exception:
        phase65b_invalid_checkpoint_count += 1
        phase65a_pretrain_start_step = 0
        phase65a_pretrain_model = phase65a_build_model(
            PHASE65A_CONFIG["seed"] + 10, 0.5
        )
        phase65a_auxiliary_head = nn.Sequential(
            nn.LayerNorm(phase65a_representation_dimension),
            nn.Linear(phase65a_representation_dimension, 96),
            nn.SiLU(inplace=False),
            nn.Linear(96, 7),
        ).to(phase65a_device)
        phase65a_pretrain_optimizer = torch.optim.AdamW(
            list(phase65a_pretrain_model.encoder.parameters())
            + list(phase65a_auxiliary_head.parameters()),
            lr=PHASE65A_CONFIG["synthetic_learning_rate"],
            weight_decay=PHASE65A_CONFIG["synthetic_weight_decay"],
            betas=(0.9, 0.95),
        )
        phase65a_synthetic_generator = torch.Generator(device=phase65a_device)
        phase65a_synthetic_generator.manual_seed(PHASE65A_CONFIG["seed"] + 20)
        phase65a_pretrain_loss_sum = 0.0
        phase65a_pretrain_classification_sum = 0.0
        phase65a_pretrain_auxiliary_sum = 0.0

phase65a_pretrain_model.train()
phase65a_auxiliary_head.train()
for phase65a_step in range(
    phase65a_pretrain_start_step + 1,
    PHASE65A_CONFIG["synthetic_pretraining_steps"] + 1,
):
    synthetic_volume, synthetic_label, synthetic_target = (
        phase65a_synthetic_batch(
            PHASE65A_CONFIG["synthetic_batch_size"],
            phase65a_synthetic_generator,
        )
    )
    phase65a_pretrain_optimizer.zero_grad(set_to_none=True)
    with torch.amp.autocast(
        device_type=phase65a_device.type,
        dtype=phase65a_amp_dtype,
        enabled=phase65a_cuda,
    ):
        representation = phase65a_representation(
            phase65a_pretrain_model, synthetic_volume
        )
        output = phase65a_auxiliary_head(representation).float()
        classification_loss = F.binary_cross_entropy_with_logits(
            output[:, 0], synthetic_label
        )
        auxiliary_loss = F.smooth_l1_loss(
            torch.sigmoid(output[:, 1:]), synthetic_target, beta=0.10
        )
        total_loss = (
            classification_loss
            + PHASE65A_CONFIG["synthetic_auxiliary_weight"] * auxiliary_loss
        )
    total_loss.backward()
    gradient_norm = torch.nn.utils.clip_grad_norm_(
        list(phase65a_pretrain_model.encoder.parameters())
        + list(phase65a_auxiliary_head.parameters()),
        PHASE65A_CONFIG["gradient_clip"],
    )
    assert torch.isfinite(gradient_norm)
    phase65a_pretrain_optimizer.step()
    phase65a_pretrain_loss_sum += float(total_loss.detach().cpu())
    phase65a_pretrain_classification_sum += float(
        classification_loss.detach().cpu()
    )
    phase65a_pretrain_auxiliary_sum += float(auxiliary_loss.detach().cpu())
    if (
        phase65a_step == PHASE65A_CONFIG["synthetic_pretraining_steps"]
        or phase65a_step % 200 == 0
    ):
        phase65b_atomic_torch_save({
            "schema_version": PHASE65A_CONFIG["schema_version"],
            "run_sha256": phase65b_run_sha256,
            "step": int(phase65a_step),
            "encoder_state": phase65a_pretrain_model.encoder.state_dict(),
            "auxiliary_head_state": phase65a_auxiliary_head.state_dict(),
            "optimizer_state": phase65a_pretrain_optimizer.state_dict(),
            "generator_state": phase65a_synthetic_generator.get_state(),
            "total_loss_sum": float(phase65a_pretrain_loss_sum),
            "classification_loss_sum": float(
                phase65a_pretrain_classification_sum
            ),
            "auxiliary_loss_sum": float(phase65a_pretrain_auxiliary_sum),
        }, phase65b_pretrain_checkpoint_path)
        print(
            f"Phase65B synthetic pretrain {phase65a_step}/"
            f"{PHASE65A_CONFIG['synthetic_pretraining_steps']}: "
            f"loss={phase65a_pretrain_loss_sum / phase65a_step:.5f}",
            flush=True,
        )

phase65a_pretrained_encoder_state = {
    name: value.detach().cpu().clone()
    for name, value in phase65a_pretrain_model.encoder.state_dict().items()
}
del (
    phase65a_pretrain_model,
    phase65a_auxiliary_head,
    phase65a_pretrain_optimizer,
    synthetic_volume,
    synthetic_label,
    synthetic_target,
    representation,
    output,
)
gc.collect()
if phase65a_cuda:
    torch.cuda.empty_cache()


def phase65a_update_ema(ema_model, model, decay):
    with torch.no_grad():
        source_parameters = dict(model.named_parameters())
        for name, parameter in ema_model.named_parameters():
            parameter.mul_(decay).add_(
                source_parameters[name], alpha=1.0 - decay
            )
        source_buffers = dict(model.named_buffers())
        for name, buffer in ema_model.named_buffers():
            buffer.copy_(source_buffers[name])


@torch.inference_mode()
def phase65a_predict(model, loader, expected_indices):
    model.eval()
    probability_parts = []
    index_parts = []
    for batch in loader:
        volume = batch["volume"].to(
            phase65a_device, non_blocking=phase65a_cuda
        )
        zero_anchor = torch.zeros(
            volume.shape[0], device=phase65a_device, dtype=torch.float32
        )
        with torch.amp.autocast(
            device_type=phase65a_device.type,
            dtype=phase65a_amp_dtype,
            enabled=phase65a_cuda,
        ):
            output = model(volume, zero_anchor)
        probability_parts.append(
            output["probability"].float().cpu().numpy()
        )
        index_parts.append(batch["case_index"].numpy())
    probability = np.concatenate(probability_parts).astype(np.float64)
    indices = np.concatenate(index_parts).astype(np.int64)
    assert np.array_equal(indices, expected_indices)
    return phase65a_clip(probability)


phase65a_expert_oof = np.full(phase65a_expected_n, np.nan, dtype=np.float64)
phase65a_fold_training_rows = []

for phase65a_fold in range(3):
    train_indices = np.flatnonzero(phase65a_original_fold != phase65a_fold)
    valid_indices = np.flatnonzero(phase65a_original_fold == phase65a_fold)
    assert train_indices.size > 0 and valid_indices.size > 0
    assert not set(phase65a_groups[train_indices].tolist()) & set(
        phase65a_groups[valid_indices].tolist()
    )
    phase65b_fold_checkpoint_path = (
        phase65b_checkpoint_directory / f"fold_{phase65a_fold}_oof.npz"
    )
    phase65b_fold_resumed = False
    if phase65b_fold_checkpoint_path.is_file():
        try:
            with np.load(
                phase65b_fold_checkpoint_path, allow_pickle=False
            ) as phase65b_fold_checkpoint:
                checkpoint_run = str(np.asarray(
                    phase65b_fold_checkpoint["run_sha256"]
                ).item())
                checkpoint_indices = np.asarray(
                    phase65b_fold_checkpoint["valid_indices"], dtype=np.int64
                ).reshape(-1)
                checkpoint_probability = np.asarray(
                    phase65b_fold_checkpoint["probability"], dtype=np.float64
                ).reshape(-1)
                checkpoint_loss = float(np.asarray(
                    phase65b_fold_checkpoint["final_training_loss"]
                ).item())
            phase65b_require(
                checkpoint_run == phase65b_run_sha256,
                "fold_checkpoint_run_mismatch",
            )
            phase65b_require(
                np.array_equal(checkpoint_indices, valid_indices),
                "fold_checkpoint_indices_mismatch",
            )
            phase65b_require(
                checkpoint_probability.shape == valid_indices.shape
                and np.all(np.isfinite(checkpoint_probability))
                and np.all((checkpoint_probability > 0.0)
                           & (checkpoint_probability < 1.0))
                and math.isfinite(checkpoint_loss),
                "fold_checkpoint_values_invalid",
            )
            phase65a_expert_oof[valid_indices] = checkpoint_probability
            phase65b_completed_folds_resumed.append(int(phase65a_fold))
            phase65a_fold_training_rows.append({
                "fold": int(phase65a_fold),
                "train_n": int(train_indices.size),
                "valid_n": int(valid_indices.size),
                "train_group_count": int(np.unique(
                    phase65a_groups[train_indices]
                ).size),
                "valid_group_count": int(np.unique(
                    phase65a_groups[valid_indices]
                ).size),
                "final_training_loss": checkpoint_loss,
                "resumed_from_matching_private_checkpoint": True,
                "outer_images_used_for_fitting": False,
                "outer_labels_used_for_fitting": False,
            })
            phase65b_fold_resumed = True
        except Exception:
            phase65b_invalid_checkpoint_count += 1
            phase65b_fold_resumed = False
    if phase65b_fold_resumed:
        print(
            f"Phase65B fold {phase65a_fold}/2 restored from matching checkpoint",
            flush=True,
        )
        continue

    train_dataset = Phase65RealDataset(
        train_indices, augment=not phase65a_is_test,
        seed=PHASE65A_CONFIG["seed"] + 1000 * phase65a_fold,
    )
    valid_dataset = Phase65RealDataset(
        valid_indices, augment=False,
        seed=PHASE65A_CONFIG["seed"] + 1000 * phase65a_fold + 1,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=PHASE65A_CONFIG["real_batch_size"],
        shuffle=False,
        num_workers=PHASE65A_CONFIG["real_worker_count"],
        pin_memory=phase65a_cuda,
        persistent_workers=False,
        drop_last=False,
    )

    prevalence = float(np.mean(phase65a_labels[train_indices]))
    model = phase65a_build_model(
        PHASE65A_CONFIG["seed"] + 100 + phase65a_fold, prevalence
    )
    model.encoder.load_state_dict(
        phase65a_pretrained_encoder_state, strict=True
    )
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.AdamW([
        {
            "params": list(model.encoder.parameters()),
            "lr": PHASE65A_CONFIG["encoder_learning_rate"],
        },
        {
            "params": list(model.head.parameters()),
            "lr": PHASE65A_CONFIG["head_learning_rate"],
        },
    ], weight_decay=PHASE65A_CONFIG["weight_decay"], betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(
        train_indices.size / PHASE65A_CONFIG["real_batch_size"]
    )
    total_steps = steps_per_epoch * PHASE65A_CONFIG["real_epoch_count"]
    warmup_steps = max(1, int(round(
        total_steps * PHASE65A_CONFIG["warmup_fraction"]
    )))

    def phase65a_lr_multiplier(step):
        if step < warmup_steps:
            return float((step + 1) / warmup_steps)
        progress = (
            (step - warmup_steps)
            / max(1, total_steps - warmup_steps - 1)
        )
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = PHASE65A_CONFIG["minimum_learning_rate_multiplier"]
        return float(floor + (1.0 - floor) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=phase65a_lr_multiplier
    )
    group_weight = torch.from_numpy(
        phase65a_group_weight(phase65a_groups[train_indices])
    ).to(phase65a_device)
    fold_loss_history = []

    for epoch in range(1, PHASE65A_CONFIG["real_epoch_count"] + 1):
        train_dataset.set_epoch(epoch)
        loader_generator = torch.Generator(device="cpu")
        loader_generator.manual_seed(
            PHASE65A_CONFIG["seed"]
            + phase65a_fold * 10_000 + epoch
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=PHASE65A_CONFIG["real_batch_size"],
            shuffle=True,
            generator=loader_generator,
            num_workers=PHASE65A_CONFIG["real_worker_count"],
            pin_memory=phase65a_cuda,
            persistent_workers=False,
            drop_last=False,
        )
        model.train()
        loss_sum = 0.0
        case_count = 0
        for batch in train_loader:
            volume = batch["volume"].to(
                phase65a_device, non_blocking=phase65a_cuda
            )
            label = batch["label"].to(
                phase65a_device, non_blocking=phase65a_cuda
            )
            group = batch["group"].to(
                phase65a_device, non_blocking=phase65a_cuda
            )
            zero_anchor = torch.zeros(
                volume.shape[0], device=phase65a_device,
                dtype=torch.float32,
            )
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=phase65a_device.type,
                dtype=phase65a_amp_dtype,
                enabled=phase65a_cuda,
            ):
                output = model(volume, zero_anchor)
                per_case = F.binary_cross_entropy_with_logits(
                    output["logit"].float(), label.float(), reduction="none"
                )
                ordinary_loss = per_case.mean()
                balanced_loss = torch.mean(per_case * group_weight[group])
                mix = PHASE65A_CONFIG["group_balance_mix"]
                loss = (1.0 - mix) * ordinary_loss + mix * balanced_loss
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), PHASE65A_CONFIG["gradient_clip"]
            )
            assert torch.isfinite(gradient_norm)
            optimizer.step()
            scheduler.step()
            phase65a_update_ema(
                ema_model, model, PHASE65A_CONFIG["ema_decay"]
            )
            batch_n = int(volume.shape[0])
            loss_sum += float(loss.detach().cpu()) * batch_n
            case_count += batch_n
        fold_loss_history.append(loss_sum / case_count)
        if (
            epoch == PHASE65A_CONFIG["real_epoch_count"]
            or epoch % 3 == 0
        ):
            print(
                f"Phase65B real fold {phase65a_fold}/2 epoch {epoch}/"
                f"{PHASE65A_CONFIG['real_epoch_count']}: "
                f"loss={fold_loss_history[-1]:.5f}",
                flush=True,
            )

    phase65a_fold_probability = phase65a_predict(
        ema_model, valid_loader, valid_indices
    )
    phase65a_expert_oof[valid_indices] = phase65a_fold_probability
    phase65b_atomic_npz(
        phase65b_fold_checkpoint_path,
        run_sha256=np.asarray(phase65b_run_sha256),
        valid_indices=valid_indices.astype(np.int64),
        probability=phase65a_fold_probability.astype(np.float64),
        final_training_loss=np.asarray(fold_loss_history[-1], dtype=np.float64),
    )
    phase65a_fold_training_rows.append({
        "fold": int(phase65a_fold),
        "train_n": int(train_indices.size),
        "valid_n": int(valid_indices.size),
        "train_group_count": int(np.unique(
            phase65a_groups[train_indices]
        ).size),
        "valid_group_count": int(np.unique(
            phase65a_groups[valid_indices]
        ).size),
        "final_training_loss": float(fold_loss_history[-1]),
        "resumed_from_matching_private_checkpoint": False,
        "outer_images_used_for_fitting": False,
        "outer_labels_used_for_fitting": False,
    })
    del (
        train_dataset,
        valid_dataset,
        valid_loader,
        model,
        ema_model,
        optimizer,
        scheduler,
        group_weight,
    )
    gc.collect()
    if phase65a_cuda:
        torch.cuda.empty_cache()

assert np.all(np.isfinite(phase65a_expert_oof))
assert np.all((phase65a_expert_oof > 0.0) & (phase65a_expert_oof < 1.0))
phase65a_candidate = phase65a_sigmoid(
    PHASE65A_CONFIG["anchor_logit_weight"] * phase65a_logit(phase65a_anchor)
    + PHASE65A_CONFIG["synthetic_expert_logit_weight"]
    * phase65a_logit(phase65a_expert_oof)
)
phase65a_anchor_metrics = phase65a_metrics(phase65a_labels, phase65a_anchor)
phase65a_expert_metrics = phase65a_metrics(
    phase65a_labels, phase65a_expert_oof
)
phase65a_pooled = phase65a_compare(
    phase65a_labels, phase65a_anchor, phase65a_candidate
)

phase65a_fold_rows = []
for phase65a_fold in range(3):
    mask = phase65a_original_fold == phase65a_fold
    phase65a_fold_rows.append({
        "fold": int(phase65a_fold),
        **phase65a_compare(
            phase65a_labels[mask],
            phase65a_anchor[mask],
            phase65a_candidate[mask],
        ),
    })

phase65a_major_mask = np.isin(
    phase65a_groups, PHASE65A_CONFIG["major_groups"]
)
phase65a_major_combined = phase65a_compare(
    phase65a_labels[phase65a_major_mask],
    phase65a_anchor[phase65a_major_mask],
    phase65a_candidate[phase65a_major_mask],
)
phase65a_major_rows = []
phase65a_major_regrets = []
for phase65a_group in PHASE65A_CONFIG["major_groups"]:
    mask = phase65a_groups == phase65a_group
    comparison = phase65a_compare(
        phase65a_labels[mask], phase65a_anchor[mask],
        phase65a_candidate[mask],
    )
    phase65a_major_rows.append({
        "group": int(phase65a_group), **comparison
    })
    phase65a_major_regrets.append(max(0.0, -comparison["log_loss_gain"]))

phase65a_domain_rows = []
phase65a_domain_gains = []
for domain, definition in enumerate(phase65a_domain_definitions):
    mask = phase65a_domain_id == domain
    assert int(np.sum(mask)) > 0
    comparison = phase65a_compare(
        phase65a_labels[mask], phase65a_anchor[mask],
        phase65a_candidate[mask],
    )
    row = {
        "domain": str(definition["name"]),
        "n": int(np.sum(mask)),
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    }
    phase65a_domain_rows.append(row)
    phase65a_domain_gains.append(comparison["log_loss_gain"])

phase65a_domain_gains = np.asarray(
    phase65a_domain_gains, dtype=np.float64
)
phase65a_bootstrap_rng = np.random.default_rng(
    PHASE65A_CONFIG["domain_bootstrap_seed"]
)
phase65a_bootstrap = np.mean(
    phase65a_domain_gains[phase65a_bootstrap_rng.integers(
        0, phase65a_domain_gains.size,
        size=(
            PHASE65A_CONFIG["domain_bootstrap_replicates"],
            phase65a_domain_gains.size,
        ),
    )],
    axis=1,
)
phase65a_q025, phase65a_q50, phase65a_q975 = np.quantile(
    phase65a_bootstrap, [0.025, 0.50, 0.975]
).tolist()
phase65a_domain_rows_sorted = sorted(
    phase65a_domain_rows, key=lambda row: row["log_loss_gain"]
)

phase65a_fold_log_loss_gains = np.asarray([
    row["log_loss_gain"] for row in phase65a_fold_rows
], dtype=np.float64)
phase65a_fold_auroc_gains = np.asarray([
    row["auroc_gain"] for row in phase65a_fold_rows
], dtype=np.float64)
phase65a_max_fold_regret = float(max(
    0.0, -np.min(phase65a_fold_log_loss_gains)
))
phase65a_max_major_regret = float(max(phase65a_major_regrets))
phase65a_gate = PHASE65A_CONFIG["advancement_gate"]
phase65a_advanced = bool(
    phase65a_pooled["log_loss_gain"]
    >= phase65a_gate["minimum_pooled_log_loss_gain"]
    and phase65a_pooled["auroc_gain"]
    >= phase65a_gate["minimum_pooled_auroc_gain"]
    and int(np.sum(phase65a_fold_log_loss_gains > 0.0))
    >= phase65a_gate["minimum_fold_log_loss_wins"]
    and int(np.sum(phase65a_fold_auroc_gains > 0.0))
    >= phase65a_gate["minimum_fold_auroc_wins"]
    and phase65a_max_fold_regret
    <= phase65a_gate["maximum_fold_log_loss_regret"]
    and phase65a_major_combined["log_loss_gain"]
    >= phase65a_gate["minimum_major_groups_combined_log_loss_gain"]
    and phase65a_max_major_regret
    <= phase65a_gate["maximum_individual_major_group_log_loss_regret"]
    and phase65a_q025
    >= phase65a_gate["minimum_domain_bootstrap_lower_95_log_loss_gain"]
    and phase65a_pooled["brier_gain"]
    >= -phase65a_gate["brier_regret_allowed"]
)
phase65a_status = (
    "phase65b_physics_synthetic_gate_passed_ready_for_logo_confirmation"
    if phase65a_advanced
    else "phase65b_physics_synthetic_gate_failed_stop_candidate"
)

phase65a_contract_core = {
    "schema_version": PHASE65A_CONFIG["schema_version"],
    "status": phase65a_status,
    "source_contract_sha256": str(
        phase65a_validation["contract_sha256"]
    ),
    "candidate": (
        "0.75_anchor_logit_plus_0.25_physics_synthetic_pretrained_3d_logit"
    ),
    "single_predeclared_candidate": True,
    "synthetic_pretraining_uses_real_images": False,
    "synthetic_pretraining_uses_real_labels": False,
    "outer_images_used_for_fitting": False,
    "outer_labels_used_for_fitting": False,
    "test_data_read": False,
    "standalone_production_restore": bool(PHASE65B_PRODUCTION_RESTORE_USED),
    "requires_previous_notebook_cells": False,
    "phase56_submission_sha256": (
        PHASE65B_RESTORE_SUMMARY_PRIVATE.get("submission_sha256")
        if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict)
        else None
    ),
    "frozen_run_sha256": phase65b_run_sha256,
}
phase65a_contract_sha256 = hashlib.sha256(json.dumps(
    phase65a_contract_core, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
phase65a_contract_path = Path(PHASE65A_CONFIG["contract_file"])
phase65a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase65a_contract_temporary = phase65a_contract_path.with_suffix(
    phase65a_contract_path.suffix + ".tmp"
)
phase65a_contract_temporary.write_text(json.dumps({
    **phase65a_contract_core,
    "contract_sha256": phase65a_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
    "contains_embeddings": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase65a_contract_temporary, phase65a_contract_path)

PHASE65B_PHYSICS_SYNTHETIC_STATE_PRIVATE = {
    "status": phase65a_status,
    "contract_sha256": phase65a_contract_sha256,
    "expert_oof_probability": phase65a_expert_oof.copy(),
    "candidate_oof_probability": phase65a_candidate.copy(),
}

phase65a_report = {
    "phase": "phase65b_standalone_physics_synthetic_pretraining_gate",
    "status": phase65a_status,
    "restart_safety": {
        "requires_previous_notebook_cells": False,
        "requires_live_kernel_variables_in_production": False,
        "production_restore_used": bool(PHASE65B_PRODUCTION_RESTORE_USED),
        "training_state_reconstructed_from_persistent_artifacts": bool(
            PHASE65B_PRODUCTION_RESTORE_USED
        ),
        "frozen_run_sha256": phase65b_run_sha256,
        "procedural_pretraining_resumed_from_step": int(
            phase65b_pretraining_resumed_from_step
        ),
        "completed_folds_resumed": phase65b_completed_folds_resumed,
        "invalid_or_stale_private_checkpoint_count": int(
            phase65b_invalid_checkpoint_count
        ),
        "private_checkpoint_paths_exported": False,
    },
    "source_integrity": {
        "phase56_submission_sha256": (
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("submission_sha256")
            if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict)
            else None
        ),
        "phase62_contract_sha256": (
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("phase62_contract_sha256")
            if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict)
            else None
        ),
        "phase64_rejection_preserved": True,
        "phase65a_fallback_rejection_preserved": (
            True if PHASE65B_PRODUCTION_RESTORE_USED else None
        ),
        "phase65ar4_offline_integrity_preserved": (
            True if PHASE65B_PRODUCTION_RESTORE_USED else None
        ),
        "training_label_cache_row_order_verified": bool(
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("row_order_verified", False)
        ),
        "router_group_fold_mapping_verified": bool(
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("router_fold_verified", False)
        ),
        "highres_cache_contract_verified": bool(
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("cache_contract_verified", False)
        ),
    },
    "candidate": {
        "single_predeclared_candidate": True,
        "formula": (
            "sigmoid(0.75*anchor_logit+0.25*physics_expert_logit)"
        ),
        "numeric_advanced": phase65a_advanced,
    },
    "procedural_pretraining": {
        "real_images_used": False,
        "real_labels_used": False,
        "external_weights_or_datasets_used": False,
        "generated_case_count": int(
            PHASE65A_CONFIG["synthetic_batch_size"]
            * PHASE65A_CONFIG["synthetic_pretraining_steps"]
        ),
        "steps": int(PHASE65A_CONFIG["synthetic_pretraining_steps"]),
        "targets": [
            "abnormality", "mean_putamen_uptake", "minimum_putamen_uptake",
            "putamen_asymmetry", "mean_caudate_uptake",
            "background_uptake", "resolution_blur",
        ],
        "mean_total_loss": float(
            phase65a_pretrain_loss_sum
            / PHASE65A_CONFIG["synthetic_pretraining_steps"]
        ),
        "mean_classification_loss": float(
            phase65a_pretrain_classification_sum
            / PHASE65A_CONFIG["synthetic_pretraining_steps"]
        ),
        "mean_auxiliary_loss": float(
            phase65a_pretrain_auxiliary_sum
            / PHASE65A_CONFIG["synthetic_pretraining_steps"]
        ),
    },
    "validation": {
        "fold_count": 3,
        "held_out_unit": "complete_acquisition_groups",
        "epoch_selection_on_outer_labels": False,
        "blend_selection_on_outer_labels": False,
        "fold_training": phase65a_fold_training_rows,
    },
    "anchor": phase65a_anchor_metrics,
    "physics_expert_alone": phase65a_expert_metrics,
    "eligible_candidate": phase65a_pooled,
    "original_folds": phase65a_fold_rows,
    "major_groups": {
        "combined": phase65a_major_combined,
        "individual": phase65a_major_rows,
        "maximum_log_loss_regret": phase65a_max_major_regret,
    },
    "stress_domain_extremes": {
        "worst_three": phase65a_domain_rows_sorted[:3],
        "best_three": phase65a_domain_rows_sorted[-3:][::-1],
    },
    "domain_stability": {
        "domain_macro_log_loss_gain": float(np.mean(phase65a_domain_gains)),
        "domain_win_count": int(np.sum(phase65a_domain_gains > 0.0)),
        "domain_count": int(phase65a_domain_gains.size),
        "bootstrap_replicates": int(
            PHASE65A_CONFIG["domain_bootstrap_replicates"]
        ),
        "bootstrap_lower_95_log_loss_gain": float(phase65a_q025),
        "bootstrap_median_log_loss_gain": float(phase65a_q50),
        "bootstrap_upper_95_log_loss_gain": float(phase65a_q975),
    },
    "gate_summary": {
        "fold_log_loss_wins": int(np.sum(
            phase65a_fold_log_loss_gains > 0.0
        )),
        "fold_auroc_wins": int(np.sum(
            phase65a_fold_auroc_gains > 0.0
        )),
        "maximum_fold_log_loss_regret": phase65a_max_fold_regret,
        "maximum_major_group_log_loss_regret": phase65a_max_major_regret,
        "domain_bootstrap_lower_95_log_loss_gain": float(phase65a_q025),
        "thresholds": dict(phase65a_gate),
    },
    "interpretation_contract": {
        "phase64_diffusion_candidate_remains_rejected": True,
        "procedural_generator_fixed_before_real_oof_evaluation": True,
        "synthetic_pretraining_independent_of_challenge_data": True,
        "each_real_oof_model_excludes_held_out_groups": True,
        "test_cases_processed_independently": True,
        "no_test_retraining_adaptation_or_batch_statistics": True,
        "passing_result_requires_logo_confirmation": True,
    },
    "training_performed": True,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "case_level_predictions_exported_in_sanitized_output": False,
    "models_exported_in_sanitized_output": False,
    "private_restart_checkpoints_written": True,
    "private_case_level_oof_checkpoints_written": True,
    "private_checkpoint_contents_exported_in_sanitized_report": False,
    "synthetic_test_mode": phase65a_is_test,
    "contract_sha256": phase65a_contract_sha256,
    "contract_file": str(phase65a_contract_path),
    "elapsed_seconds": float(time.perf_counter() - phase65a_started),
}

PHASE65B_PHYSICS_SYNTHETIC_REPORT_PRIVATE = phase65a_report

# Compatibility aliases are private in-memory only. They allow a subsequent
# confirmation cell to consume the result without weakening the Phase65B
# contract or requiring a rerun.
PHASE65A_PHYSICS_SYNTHETIC_STATE_PRIVATE = (
    PHASE65B_PHYSICS_SYNTHETIC_STATE_PRIVATE
)
PHASE65A_PHYSICS_SYNTHETIC_REPORT_PRIVATE = (
    PHASE65B_PHYSICS_SYNTHETIC_REPORT_PRIVATE
)

print("BEGIN SANITIZED_PHASE65B_STANDALONE_PHYSICS_SYNTHETIC_PRETRAINING_GATE")
print(json.dumps(phase65a_report, indent=2, sort_keys=False))
print("END SANITIZED_PHASE65B_STANDALONE_PHYSICS_SYNTHETIC_PRETRAINING_GATE")

del phase65a_pretrained_encoder_state
gc.collect()
if phase65a_cuda:
    torch.cuda.empty_cache()
