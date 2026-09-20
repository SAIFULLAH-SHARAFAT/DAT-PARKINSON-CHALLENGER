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
import tempfile
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


# Cell 165C — standalone class-conditional domain-adversarial bilateral 3D gate.
#
# This cell requires no prior notebook cells or live variables in production.
# It reconstructs the exact Phase56/Phase62 state from persistent artifacts,
# verifies that Phase65B was rejected, and trains one frozen candidate using
# only the training cache. Every OOF model excludes the complete acquisition
# groups assigned to its validation fold. The candidate, optimizer, epochs,
# adversarial strength, augmentation, EMA, and anchor blend are fixed before
# any outer-fold predictions are scored.

phase65c_started = time.perf_counter()

for phase65c_required_name in (
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
    phase65b_require(
        phase65c_required_name in globals(),
        f"missing_live_boundary_{phase65c_required_name}",
    )

phase65b_require(
    PHASE64A_DIFFUSION_REPORT_PRIVATE.get("status")
    == "repeated_group_blocked_gate_failed_stop_diffusion_candidate",
    "phase64_rejection_not_preserved_in_phase65c",
)

phase65c_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE
phase65c_engine = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase65c_labels = np.asarray(
    phase65c_validation["labels"], dtype=np.int64
).reshape(-1)
phase65c_groups = np.asarray(
    phase65c_validation["groups"], dtype=np.int64
).reshape(-1)
phase65c_original_fold = np.asarray(
    phase65c_validation["original_fold"], dtype=np.int64
).reshape(-1)
phase65c_anchor = np.asarray(
    phase65c_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase65c_domain_id = np.asarray(
    phase65c_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase65c_domain_definitions = phase65c_validation[
    "stress_domain_definitions"
]

phase65c_test_override = globals().get(
    "PHASE65C_SYNTHETIC_TEST_OVERRIDE_PRIVATE",
    globals().get("PHASE65A_SYNTHETIC_TEST_OVERRIDE_PRIVATE"),
)
phase65c_is_test = isinstance(phase65c_test_override, dict)
phase65c_expected_n = (
    int(phase65c_test_override["case_count"])
    if phase65c_is_test else 1362
)
phase65c_expected_group_count = (
    int(phase65c_test_override.get("group_count", 15))
    if phase65c_is_test else 15
)

phase65b_require(
    phase65c_labels.shape
    == phase65c_groups.shape
    == phase65c_original_fold.shape
    == phase65c_anchor.shape
    == phase65c_domain_id.shape
    == (phase65c_expected_n,),
    "phase65c_vector_shape_mismatch",
)
phase65b_require(
    set(np.unique(phase65c_labels).tolist()) == {0, 1},
    "phase65c_labels_not_binary",
)
phase65b_require(
    np.array_equal(
        np.unique(phase65c_groups),
        np.arange(phase65c_expected_group_count, dtype=np.int64),
    ),
    "phase65c_group_set_mismatch",
)
phase65b_require(
    np.array_equal(np.unique(phase65c_original_fold), np.arange(3)),
    "phase65c_fold_set_mismatch",
)
phase65b_require(
    np.all(np.isfinite(phase65c_anchor))
    and np.all((phase65c_anchor > 0.0) & (phase65c_anchor < 1.0)),
    "phase65c_anchor_probability_invalid",
)
for phase65c_group in range(phase65c_expected_group_count):
    phase65b_require(
        np.unique(
            phase65c_original_fold[phase65c_groups == phase65c_group]
        ).size == 1,
        "phase65c_group_split_leakage",
    )


PHASE65C_CONFIG = {
    "schema_version": "phase65c_standalone_class_conditional_domain_adversarial_v1",
    "seed": 650265,
    "input_shape": [80, 80, 80],
    "batch_size": 8,
    "worker_count": 2,
    "epoch_count": 14,
    "encoder_learning_rate": 6.0e-5,
    "disease_head_learning_rate": 3.0e-4,
    "domain_head_learning_rate": 3.0e-4,
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
    "anchor_logit_weight": 0.85,
    "domain_invariant_expert_logit_weight": 0.15,
    "major_groups": [1, 3],
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 650266,
    "probability_clip": 1.0e-7,
    "advancement_gate": {
        "minimum_pooled_log_loss_gain": 0.003,
        "minimum_pooled_auroc_gain": 0.001,
        "minimum_fold_log_loss_wins": 2,
        "minimum_fold_auroc_wins": 2,
        "maximum_fold_log_loss_regret": 0.001,
        "minimum_major_groups_combined_log_loss_gain": 0.003,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
        "brier_regret_allowed": 0.0,
    },
    "contract_file": str(
        PHASE65B_WORKING_ROOT
        / "phase65c_domain_invariant_bilateral_3d_gate_contract.json"
    ),
    "checkpoint_directory": str(
        PHASE65B_WORKING_ROOT / "phase65c_private_checkpoint"
    ),
}

if phase65c_is_test:
    PHASE65C_CONFIG = {
        **PHASE65C_CONFIG,
        "input_shape": list(phase65c_test_override.get(
            "input_shape", [16, 16, 16]
        )),
        "batch_size": int(phase65c_test_override.get("batch_size", 6)),
        "worker_count": 0,
        "epoch_count": int(phase65c_test_override.get("epoch_count", 2)),
        "domain_bootstrap_replicates": int(
            phase65c_test_override.get("domain_bootstrap_replicates", 200)
        ),
        "contract_file": str(phase65c_test_override.get(
            "contract_file", os.path.join(tempfile.gettempdir(), "phase65c_synthetic_contract.json")
        )),
        "checkpoint_directory": str(phase65c_test_override.get(
            "checkpoint_directory", os.path.join(tempfile.gettempdir(), "phase65c_private_checkpoint")
        )),
    }

phase65b_require(
    abs(
        PHASE65C_CONFIG["anchor_logit_weight"]
        + PHASE65C_CONFIG["domain_invariant_expert_logit_weight"]
        - 1.0
    ) <= 1.0e-15,
    "phase65c_blend_weights_do_not_sum_to_one",
)
phase65b_require(
    PHASE65C_CONFIG["epoch_count"] >= 1,
    "phase65c_epoch_count_invalid",
)

phase65c_device = phase57c_device
phase65c_cuda = bool(phase57c_cuda)
phase65c_amp_dtype = phase57c_amp_dtype
phase65c_cache_file = str(phase65c_engine["highres_cache_file"])
phase65c_cache_path = Path(phase65c_cache_file)
phase65b_require(phase65c_cache_path.is_file(), "phase65c_cache_missing")
phase65c_cache_contract = np.load(
    phase65c_cache_path, mmap_mode="r", allow_pickle=False
)
phase65b_require(
    phase65c_cache_contract.shape
    == (phase65c_expected_n, *PHASE65C_CONFIG["input_shape"])
    and phase65c_cache_contract.dtype == np.float16,
    "phase65c_cache_contract_mismatch",
)
del phase65c_cache_contract


def phase65c_array_sha256(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(value.tobytes()).hexdigest()


phase65c_phase65b_contract_sha256 = None
if not phase65c_is_test:
    phase65c_phase65b_contract = phase65b_json(
        PHASE65B_WORKING_ROOT
        / "phase65b_physics_synthetic_pretraining_gate_contract.json",
        "phase65b_rejection_contract",
    )
    phase65b_require(
        phase65c_phase65b_contract.get("status")
        == "phase65b_physics_synthetic_gate_failed_stop_candidate",
        "phase65b_rejection_not_preserved",
    )
    phase65b_require(
        phase65b_contract_core_hash_valid(phase65c_phase65b_contract),
        "phase65b_rejection_contract_hash_mismatch",
    )
    phase65c_phase65b_contract_sha256 = str(
        phase65c_phase65b_contract["contract_sha256"]
    )

phase65c_run_contract_core = {
    "schema_version": PHASE65C_CONFIG["schema_version"],
    "config": {
        key: value
        for key, value in PHASE65C_CONFIG.items()
        if key not in {"contract_file", "checkpoint_directory"}
    },
    "architecture": phase57c_architecture_config,
    "source_contract_sha256": str(phase65c_validation["contract_sha256"]),
    "phase65b_rejection_contract_sha256": phase65c_phase65b_contract_sha256,
    "label_sha256": phase65c_array_sha256(phase65c_labels.astype(np.int8)),
    "group_sha256": phase65c_array_sha256(phase65c_groups.astype(np.int8)),
    "fold_sha256": phase65c_array_sha256(
        phase65c_original_fold.astype(np.int8)
    ),
    "anchor_sha256": phase65c_array_sha256(
        phase65c_anchor.astype(np.float64)
    ),
    "cache_shape": [phase65c_expected_n, *PHASE65C_CONFIG["input_shape"]],
    "cache_dtype": "float16",
}
phase65c_run_sha256 = hashlib.sha256(json.dumps(
    phase65c_run_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()
phase65c_checkpoint_directory = (
    Path(PHASE65C_CONFIG["checkpoint_directory"])
    / phase65c_run_sha256[:16]
)
phase65c_checkpoint_directory.mkdir(parents=True, exist_ok=True)
phase65c_completed_folds_resumed = []
phase65c_fold_epoch_resume = {}
phase65c_invalid_checkpoint_count = 0


def phase65c_atomic_torch_save(payload, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def phase65c_atomic_npz(path, **arrays):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def phase65c_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE65C_CONFIG["probability_clip"],
        1.0 - PHASE65C_CONFIG["probability_clip"],
    )


def phase65c_logit(probability):
    probability = phase65c_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase65c_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return phase65c_clip(output)


def phase65c_auc(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    phase65b_require(labels.shape == score.shape, "phase65c_auc_shape")
    positive_n = int(np.sum(labels == 1))
    negative_n = int(np.sum(labels == 0))
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
        rank[order[start:stop]] = 0.5 * ((start + 1) + stop)
        start = stop
    statistic = float(np.sum(rank[labels == 1]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def phase65c_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = phase65c_clip(probability).reshape(-1)
    phase65b_require(
        labels.shape == probability.shape,
        "phase65c_metric_shape_mismatch",
    )
    loss = -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(loss)),
        "auroc": phase65c_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase65c_compare(labels, anchor, candidate):
    anchor_metrics = phase65c_metrics(labels, anchor)
    candidate_metrics = phase65c_metrics(labels, candidate)
    anchor_auc = anchor_metrics["auroc"]
    candidate_auc = candidate_metrics["auroc"]
    auroc_gain = (
        None if anchor_auc is None or candidate_auc is None
        else float(candidate_auc - anchor_auc)
    )
    return {
        "n": anchor_metrics["n"],
        "anchor_log_loss": anchor_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            anchor_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "anchor_auroc": anchor_auc,
        "candidate_auroc": candidate_auc,
        "auroc_gain": auroc_gain,
        "brier_gain": float(
            anchor_metrics["brier"] - candidate_metrics["brier"]
        ),
    }


def phase65c_group_weight(groups):
    groups = np.asarray(groups, dtype=np.int64)
    counts = np.bincount(
        groups, minlength=phase65c_expected_group_count
    ).astype(np.float64)
    lookup = np.ones(phase65c_expected_group_count, dtype=np.float64)
    active = counts > 0.0
    lookup[active] = np.power(counts[active], -0.5)
    lookup /= float(np.mean(lookup[groups]))
    return lookup.astype(np.float32)


class Phase65CGradientReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, strength):
        ctx.strength = float(strength)
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.strength * gradient, None


class Phase65CDomainInvariantBilateral3D(nn.Module):
    def __init__(self, seed, prevalence, domain_class_count):
        super().__init__()
        torch.manual_seed(int(seed))
        if phase65c_cuda:
            torch.cuda.manual_seed_all(int(seed))
        self.backbone = phase57c_model_class(
            channels=phase57c_architecture_config["channels"],
            blocks_per_stage=phase57c_architecture_config["blocks_per_stage"],
            projection_dimension=phase57c_architecture_config[
                "multiscale_projection_dimension"
            ],
            head_hidden_dimension=phase57c_architecture_config[
                "head_hidden_dimension"
            ],
            residual_cap=PHASE65C_CONFIG["independent_logit_cap"],
        ).to(phase65c_device)
        representation_dimension = int(
            self.backbone.head[1].in_features
        )
        hidden = max(32, min(96, representation_dimension // 2))
        self.domain_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(representation_dimension),
                nn.Linear(representation_dimension, hidden),
                nn.SiLU(inplace=False),
                nn.Linear(hidden, int(domain_class_count)),
            )
            for _ in range(2)
        ]).to(phase65c_device)
        prevalence = float(np.clip(prevalence, 1.0e-4, 1.0 - 1.0e-4))
        prevalence_logit = math.log(prevalence / (1.0 - prevalence))
        raw_bias = np.arctanh(np.clip(
            prevalence_logit / PHASE65C_CONFIG["independent_logit_cap"],
            -0.999,
            0.999,
        ))
        with torch.no_grad():
            self.backbone.head[-1].bias.fill_(float(raw_bias))
        phase65b_require(
            not any(isinstance(module, (
                nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d
            )) for module in self.modules()),
            "phase65c_batch_normalization_forbidden",
        )
        phase65b_require(
            sum(parameter.numel() for parameter in self.parameters())
            < 6_000_000,
            "phase65c_model_parameter_ceiling_exceeded",
        )

    def representation(self, volume):
        original, reflected = self.backbone.make_bilateral_inputs(volume)
        encoded = self.backbone.encoder(torch.cat([original, reflected], dim=0))
        original_embedding, reflected_embedding = encoded.chunk(2, dim=0)
        return torch.cat([
            0.5 * (original_embedding + reflected_embedding),
            torch.abs(original_embedding - reflected_embedding),
        ], dim=1)

    def disease(self, representation):
        raw = self.backbone.head(representation).reshape(-1).float()
        logit = PHASE65C_CONFIG["independent_logit_cap"] * torch.tanh(raw)
        return logit

    def forward(self, volume):
        representation = self.representation(volume)
        logit = self.disease(representation)
        return {
            "representation": representation,
            "logit": logit,
            "probability": torch.sigmoid(logit),
        }

    def domain_logits(self, representation, labels, reversal_strength):
        reversed_representation = Phase65CGradientReverse.apply(
            representation, float(reversal_strength)
        )
        all_logits = torch.stack([
            head(reversed_representation) for head in self.domain_heads
        ], dim=1)
        row = torch.arange(
            labels.numel(), device=labels.device, dtype=torch.long
        )
        return all_logits[row, labels.long().reshape(-1)]


class Phase65CRealDataset(Dataset):
    def __init__(self, indices, two_views, seed):
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.two_views = bool(two_views)
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
                phase65c_cache_file, mmap_mode="r", allow_pickle=False
            )
        return self.cache

    def __getitem__(self, item):
        case_index = int(self.indices[int(item)])
        volume = np.asarray(
            self.resolve_cache()[case_index], dtype=np.float32
        ).copy()
        tensor = torch.from_numpy(volume)[None]
        if self.two_views:
            base_seed = (
                self.seed + self.epoch * 1_000_003 + case_index * 193
            )
            view_a = phase57c_augment_one(
                tensor[None], base_seed + 17,
                phase57c_architecture_config["scanner_augmentation"],
            )[0]
            view_b = phase57c_augment_one(
                tensor[None], base_seed + 71,
                phase57c_architecture_config["scanner_augmentation"],
            )[0]
        else:
            view_a = tensor
            view_b = tensor
        return {
            "view_a": view_a,
            "view_b": view_b,
            "label": torch.tensor(
                phase65c_labels[case_index], dtype=torch.float32
            ),
            "group": torch.tensor(
                phase65c_groups[case_index], dtype=torch.long
            ),
            "case_index": torch.tensor(case_index, dtype=torch.long),
        }


def phase65c_loader(dataset, shuffle, seed):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=PHASE65C_CONFIG["batch_size"],
        shuffle=bool(shuffle),
        num_workers=PHASE65C_CONFIG["worker_count"],
        pin_memory=phase65c_cuda,
        drop_last=False,
        persistent_workers=False,
        generator=generator,
    )


def phase65c_update_ema(ema_model, model, decay):
    with torch.no_grad():
        model_parameters = dict(model.named_parameters())
        for name, ema_parameter in ema_model.named_parameters():
            ema_parameter.mul_(decay).add_(
                model_parameters[name], alpha=1.0 - decay
            )
        model_buffers = dict(model.named_buffers())
        for name, ema_buffer in ema_model.named_buffers():
            ema_buffer.copy_(model_buffers[name])


def phase65c_lr_multiplier(step, total_steps):
    total_steps = max(1, int(total_steps))
    warmup_steps = max(
        1, int(round(PHASE65C_CONFIG["warmup_fraction"] * total_steps))
    )
    if step < warmup_steps:
        return float(step + 1) / float(warmup_steps)
    progress = float(step - warmup_steps) / float(
        max(1, total_steps - warmup_steps)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))
    minimum = PHASE65C_CONFIG["minimum_learning_rate_multiplier"]
    return float(minimum + (1.0 - minimum) * cosine)


def phase65c_predict(model, indices, seed):
    dataset = Phase65CRealDataset(indices, two_views=False, seed=seed)
    loader = phase65c_loader(dataset, shuffle=False, seed=seed)
    model.eval()
    probability = []
    observed_indices = []
    with torch.no_grad():
        for batch in loader:
            volume = batch["view_a"].to(
                phase65c_device, non_blocking=phase65c_cuda
            )
            with torch.autocast(
                device_type=phase65c_device.type,
                dtype=phase65c_amp_dtype,
                enabled=phase65c_cuda,
            ):
                output = model(volume)
            probability.append(
                output["probability"].float().cpu().numpy()
            )
            observed_indices.append(batch["case_index"].numpy())
    probability = np.concatenate(probability).astype(np.float64)
    observed_indices = np.concatenate(observed_indices).astype(np.int64)
    phase65b_require(
        np.array_equal(observed_indices, np.asarray(indices, dtype=np.int64)),
        "phase65c_prediction_row_order_mismatch",
    )
    return phase65c_clip(probability)


phase65c_expert_oof = np.full(
    phase65c_expected_n, np.nan, dtype=np.float64
)
phase65c_fold_training_rows = []

for phase65c_fold in range(3):
    train_indices = np.flatnonzero(
        phase65c_original_fold != phase65c_fold
    )
    valid_indices = np.flatnonzero(
        phase65c_original_fold == phase65c_fold
    )
    train_groups = np.unique(phase65c_groups[train_indices]).astype(np.int64)
    valid_groups = np.unique(phase65c_groups[valid_indices]).astype(np.int64)
    phase65b_require(
        not set(train_groups.tolist()) & set(valid_groups.tolist()),
        "phase65c_outer_group_leakage",
    )
    local_group_lookup = np.full(
        phase65c_expected_group_count, -1, dtype=np.int64
    )
    local_group_lookup[train_groups] = np.arange(
        train_groups.size, dtype=np.int64
    )
    local_group_lookup_torch = torch.from_numpy(
        local_group_lookup
    ).to(phase65c_device)

    fold_oof_path = (
        phase65c_checkpoint_directory
        / f"fold_{phase65c_fold}_oof.npz"
    )
    fold_model_path = (
        phase65c_checkpoint_directory
        / f"fold_{phase65c_fold}_model.pt"
    )
    fold_training_path = (
        phase65c_checkpoint_directory
        / f"fold_{phase65c_fold}_training.pt"
    )

    restored_complete = False
    if fold_oof_path.is_file() and fold_model_path.is_file():
        try:
            with np.load(fold_oof_path, allow_pickle=False) as checkpoint:
                checkpoint_run = str(np.asarray(
                    checkpoint["run_sha256"]
                ).item())
                checkpoint_indices = np.asarray(
                    checkpoint["valid_indices"], dtype=np.int64
                )
                checkpoint_probability = np.asarray(
                    checkpoint["probability"], dtype=np.float64
                )
                checkpoint_loss = np.asarray(
                    checkpoint["loss_history"], dtype=np.float64
                )
                checkpoint_domain_accuracy = float(np.asarray(
                    checkpoint["final_domain_accuracy"]
                ).item())
            model_checkpoint = torch.load(
                fold_model_path, map_location="cpu", weights_only=False
            )
            phase65b_require(
                checkpoint_run == phase65c_run_sha256
                and model_checkpoint.get("run_sha256")
                == phase65c_run_sha256,
                "phase65c_completed_checkpoint_run_mismatch",
            )
            phase65b_require(
                np.array_equal(checkpoint_indices, valid_indices),
                "phase65c_completed_checkpoint_indices_mismatch",
            )
            phase65b_require(
                checkpoint_probability.shape == (valid_indices.size,)
                and np.all(np.isfinite(checkpoint_probability))
                and np.all(
                    (checkpoint_probability > 0.0)
                    & (checkpoint_probability < 1.0)
                ),
                "phase65c_completed_checkpoint_probability_invalid",
            )
            phase65b_require(
                checkpoint_loss.shape
                == (PHASE65C_CONFIG["epoch_count"],)
                and np.all(np.isfinite(checkpoint_loss)),
                "phase65c_completed_checkpoint_loss_invalid",
            )
            phase65c_expert_oof[valid_indices] = checkpoint_probability
            phase65c_completed_folds_resumed.append(int(phase65c_fold))
            phase65c_fold_training_rows.append({
                "fold": int(phase65c_fold),
                "train_n": int(train_indices.size),
                "valid_n": int(valid_indices.size),
                "train_group_count": int(train_groups.size),
                "valid_group_count": int(valid_groups.size),
                "resumed_complete": True,
                "resumed_from_epoch": int(PHASE65C_CONFIG["epoch_count"]),
                "final_total_loss": float(checkpoint_loss[-1]),
                "final_domain_accuracy": checkpoint_domain_accuracy,
            })
            restored_complete = True
        except Exception:
            phase65c_invalid_checkpoint_count += 1
            restored_complete = False
    if restored_complete:
        print(
            f"Phase65C fold {phase65c_fold}/2 restored from matching checkpoint",
            flush=True,
        )
        continue

    prevalence = float(np.mean(phase65c_labels[train_indices]))
    model = Phase65CDomainInvariantBilateral3D(
        seed=PHASE65C_CONFIG["seed"] + 1000 * phase65c_fold,
        prevalence=prevalence,
        domain_class_count=int(train_groups.size),
    )
    ema_model = copy.deepcopy(model).eval()
    for parameter in ema_model.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW([
        {
            "params": model.backbone.encoder.parameters(),
            "lr": PHASE65C_CONFIG["encoder_learning_rate"],
        },
        {
            "params": model.backbone.head.parameters(),
            "lr": PHASE65C_CONFIG["disease_head_learning_rate"],
        },
        {
            "params": model.domain_heads.parameters(),
            "lr": PHASE65C_CONFIG["domain_head_learning_rate"],
        },
    ], weight_decay=PHASE65C_CONFIG["weight_decay"])

    train_dataset = Phase65CRealDataset(
        train_indices,
        two_views=True,
        seed=PHASE65C_CONFIG["seed"] + 20_000 * phase65c_fold,
    )
    steps_per_epoch = int(math.ceil(
        train_indices.size / PHASE65C_CONFIG["batch_size"]
    ))
    total_steps = steps_per_epoch * PHASE65C_CONFIG["epoch_count"]
    group_weight = torch.from_numpy(
        phase65c_group_weight(phase65c_groups[train_indices])
    ).to(phase65c_device)
    loss_history = []
    classification_history = []
    domain_history = []
    consistency_history = []
    domain_accuracy_history = []
    start_epoch = 0
    global_step = 0

    if fold_training_path.is_file():
        try:
            training_checkpoint = torch.load(
                fold_training_path,
                map_location=phase65c_device,
                weights_only=False,
            )
            phase65b_require(
                training_checkpoint.get("run_sha256")
                == phase65c_run_sha256
                and int(training_checkpoint.get("fold", -1))
                == phase65c_fold,
                "phase65c_training_checkpoint_run_mismatch",
            )
            completed_epoch = int(training_checkpoint["completed_epoch"])
            phase65b_require(
                0 <= completed_epoch < PHASE65C_CONFIG["epoch_count"],
                "phase65c_training_checkpoint_epoch_invalid",
            )
            model.load_state_dict(training_checkpoint["model_state"], strict=True)
            ema_model.load_state_dict(
                training_checkpoint["ema_state"], strict=True
            )
            optimizer.load_state_dict(training_checkpoint["optimizer_state"])
            loss_history = [float(value) for value in training_checkpoint[
                "loss_history"
            ]]
            classification_history = [
                float(value) for value in training_checkpoint[
                    "classification_history"
                ]
            ]
            domain_history = [float(value) for value in training_checkpoint[
                "domain_history"
            ]]
            consistency_history = [
                float(value) for value in training_checkpoint[
                    "consistency_history"
                ]
            ]
            domain_accuracy_history = [
                float(value) for value in training_checkpoint[
                    "domain_accuracy_history"
                ]
            ]
            phase65b_require(
                len(loss_history) == completed_epoch + 1
                and len(classification_history) == completed_epoch + 1
                and len(domain_history) == completed_epoch + 1
                and len(consistency_history) == completed_epoch + 1
                and len(domain_accuracy_history) == completed_epoch + 1,
                "phase65c_training_checkpoint_history_invalid",
            )
            start_epoch = completed_epoch + 1
            global_step = start_epoch * steps_per_epoch
            phase65c_fold_epoch_resume[int(phase65c_fold)] = int(start_epoch)
        except Exception:
            phase65c_invalid_checkpoint_count += 1
            model = Phase65CDomainInvariantBilateral3D(
                seed=PHASE65C_CONFIG["seed"] + 1000 * phase65c_fold,
                prevalence=prevalence,
                domain_class_count=int(train_groups.size),
            )
            ema_model = copy.deepcopy(model).eval()
            for parameter in ema_model.parameters():
                parameter.requires_grad_(False)
            optimizer = torch.optim.AdamW([
                {
                    "params": model.backbone.encoder.parameters(),
                    "lr": PHASE65C_CONFIG["encoder_learning_rate"],
                },
                {
                    "params": model.backbone.head.parameters(),
                    "lr": PHASE65C_CONFIG["disease_head_learning_rate"],
                },
                {
                    "params": model.domain_heads.parameters(),
                    "lr": PHASE65C_CONFIG["domain_head_learning_rate"],
                },
            ], weight_decay=PHASE65C_CONFIG["weight_decay"])
            loss_history = []
            classification_history = []
            domain_history = []
            consistency_history = []
            domain_accuracy_history = []
            start_epoch = 0
            global_step = 0

    base_learning_rates = [
        PHASE65C_CONFIG["encoder_learning_rate"],
        PHASE65C_CONFIG["disease_head_learning_rate"],
        PHASE65C_CONFIG["domain_head_learning_rate"],
    ]
    for epoch in range(start_epoch, PHASE65C_CONFIG["epoch_count"]):
        train_dataset.set_epoch(epoch)
        train_loader = phase65c_loader(
            train_dataset,
            shuffle=True,
            seed=PHASE65C_CONFIG["seed"]
            + phase65c_fold * 100_000
            + epoch,
        )
        model.train()
        epoch_total = 0.0
        epoch_classification = 0.0
        epoch_domain = 0.0
        epoch_consistency = 0.0
        epoch_domain_correct = 0
        epoch_domain_count = 0
        epoch_case_count = 0

        for batch in train_loader:
            view_a = batch["view_a"].to(
                phase65c_device, non_blocking=phase65c_cuda
            )
            view_b = batch["view_b"].to(
                phase65c_device, non_blocking=phase65c_cuda
            )
            label = batch["label"].to(
                phase65c_device, non_blocking=phase65c_cuda
            )
            group = batch["group"].to(
                phase65c_device, non_blocking=phase65c_cuda
            )
            local_group = local_group_lookup_torch[group]
            phase65b_require(
                bool(torch.all(local_group >= 0).item()),
                "phase65c_training_group_not_in_fold_map",
            )
            progress = float(global_step) / float(max(1, total_steps - 1))
            reversal_strength = PHASE65C_CONFIG[
                "maximum_gradient_reversal"
            ] * (
                2.0 / (
                    1.0 + math.exp(
                        -PHASE65C_CONFIG["gradient_reversal_ramp"] * progress
                    )
                ) - 1.0
            )
            learning_rate_multiplier = phase65c_lr_multiplier(
                global_step, total_steps
            )
            for parameter_group, base_lr in zip(
                optimizer.param_groups, base_learning_rates
            ):
                parameter_group["lr"] = base_lr * learning_rate_multiplier

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=phase65c_device.type,
                dtype=phase65c_amp_dtype,
                enabled=phase65c_cuda,
            ):
                output_a = model(view_a)
                output_b = model(view_b)
                per_case = 0.5 * (
                    F.binary_cross_entropy_with_logits(
                        output_a["logit"], label, reduction="none"
                    )
                    + F.binary_cross_entropy_with_logits(
                        output_b["logit"], label, reduction="none"
                    )
                )
                balanced = torch.mean(per_case * group_weight[group])
                mix = PHASE65C_CONFIG["group_balance_mix"]
                classification_loss = (
                    (1.0 - mix) * torch.mean(per_case) + mix * balanced
                )
                logit_consistency = torch.mean(torch.square(
                    output_a["logit"] - output_b["logit"]
                ))
                representation_consistency = torch.mean(
                    1.0 - F.cosine_similarity(
                        output_a["representation"].float(),
                        output_b["representation"].float(),
                        dim=1,
                        eps=1.0e-6,
                    )
                )
                domain_logits_a = model.domain_logits(
                    output_a["representation"], label, reversal_strength
                )
                domain_logits_b = model.domain_logits(
                    output_b["representation"], label, reversal_strength
                )
                domain_loss = 0.5 * (
                    F.cross_entropy(domain_logits_a, local_group)
                    + F.cross_entropy(domain_logits_b, local_group)
                )
                consistency_loss = (
                    PHASE65C_CONFIG["logit_view_consistency_weight"]
                    * logit_consistency
                    + PHASE65C_CONFIG[
                        "representation_view_consistency_weight"
                    ] * representation_consistency
                )
                total_loss = classification_loss + consistency_loss + domain_loss

            phase65b_require(
                bool(torch.isfinite(total_loss).item()),
                "phase65c_nonfinite_training_loss",
            )
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), PHASE65C_CONFIG["gradient_clip"]
            )
            optimizer.step()
            phase65c_update_ema(
                ema_model, model, PHASE65C_CONFIG["ema_decay"]
            )

            batch_n = int(label.numel())
            epoch_total += float(total_loss.detach().cpu()) * batch_n
            epoch_classification += float(
                classification_loss.detach().cpu()
            ) * batch_n
            epoch_domain += float(domain_loss.detach().cpu()) * batch_n
            epoch_consistency += float(
                consistency_loss.detach().cpu()
            ) * batch_n
            epoch_domain_correct += int(torch.sum(
                torch.argmax(domain_logits_a.detach(), dim=1) == local_group
            ).cpu())
            epoch_domain_count += batch_n
            epoch_case_count += batch_n
            global_step += 1

        loss_history.append(epoch_total / max(1, epoch_case_count))
        classification_history.append(
            epoch_classification / max(1, epoch_case_count)
        )
        domain_history.append(epoch_domain / max(1, epoch_case_count))
        consistency_history.append(
            epoch_consistency / max(1, epoch_case_count)
        )
        domain_accuracy_history.append(
            epoch_domain_correct / max(1, epoch_domain_count)
        )
        phase65c_atomic_torch_save({
            "run_sha256": phase65c_run_sha256,
            "fold": int(phase65c_fold),
            "completed_epoch": int(epoch),
            "train_group_ids": train_groups.copy(),
            "model_state": model.state_dict(),
            "ema_state": ema_model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "loss_history": list(loss_history),
            "classification_history": list(classification_history),
            "domain_history": list(domain_history),
            "consistency_history": list(consistency_history),
            "domain_accuracy_history": list(domain_accuracy_history),
        }, fold_training_path)
        print(
            f"Phase65C fold {phase65c_fold}/2 epoch {epoch + 1}/"
            f"{PHASE65C_CONFIG['epoch_count']}: "
            f"class={classification_history[-1]:.5f}, "
            f"domain_acc={domain_accuracy_history[-1]:.3f}",
            flush=True,
        )

    fold_probability = phase65c_predict(
        ema_model,
        valid_indices,
        PHASE65C_CONFIG["seed"] + 900_000 + phase65c_fold,
    )
    phase65c_expert_oof[valid_indices] = fold_probability
    phase65c_atomic_torch_save({
        "run_sha256": phase65c_run_sha256,
        "fold": int(phase65c_fold),
        "train_group_ids": train_groups.copy(),
        "valid_group_ids": valid_groups.copy(),
        "ema_model_state": {
            key: value.detach().cpu()
            for key, value in ema_model.state_dict().items()
        },
        "architecture_config": json.loads(json.dumps(
            phase57c_architecture_config
        )),
        "phase65c_config": {
            key: value for key, value in PHASE65C_CONFIG.items()
            if key not in {"contract_file", "checkpoint_directory"}
        },
    }, fold_model_path)
    phase65c_atomic_npz(
        fold_oof_path,
        run_sha256=np.asarray(phase65c_run_sha256),
        valid_indices=valid_indices.astype(np.int64),
        probability=fold_probability.astype(np.float64),
        loss_history=np.asarray(loss_history, dtype=np.float64),
        final_domain_accuracy=np.asarray(
            domain_accuracy_history[-1], dtype=np.float64
        ),
    )
    phase65c_fold_training_rows.append({
        "fold": int(phase65c_fold),
        "train_n": int(train_indices.size),
        "valid_n": int(valid_indices.size),
        "train_group_count": int(train_groups.size),
        "valid_group_count": int(valid_groups.size),
        "resumed_complete": False,
        "resumed_from_epoch": int(start_epoch),
        "final_total_loss": float(loss_history[-1]),
        "final_classification_loss": float(classification_history[-1]),
        "final_domain_loss": float(domain_history[-1]),
        "final_consistency_loss": float(consistency_history[-1]),
        "final_domain_accuracy": float(domain_accuracy_history[-1]),
        "chance_domain_accuracy": float(1.0 / train_groups.size),
    })

    del (
        model, ema_model, optimizer, train_dataset, train_loader,
        group_weight, local_group_lookup_torch
    )
    gc.collect()
    if phase65c_cuda:
        torch.cuda.empty_cache()

phase65b_require(
    np.all(np.isfinite(phase65c_expert_oof))
    and np.all((phase65c_expert_oof > 0.0) & (phase65c_expert_oof < 1.0)),
    "phase65c_expert_oof_incomplete",
)
phase65c_candidate = phase65c_sigmoid(
    PHASE65C_CONFIG["anchor_logit_weight"]
    * phase65c_logit(phase65c_anchor)
    + PHASE65C_CONFIG["domain_invariant_expert_logit_weight"]
    * phase65c_logit(phase65c_expert_oof)
)
phase65c_anchor_metrics = phase65c_metrics(
    phase65c_labels, phase65c_anchor
)
phase65c_expert_metrics = phase65c_metrics(
    phase65c_labels, phase65c_expert_oof
)
phase65c_pooled = phase65c_compare(
    phase65c_labels, phase65c_anchor, phase65c_candidate
)

phase65c_fold_rows = []
for phase65c_fold in range(3):
    mask = phase65c_original_fold == phase65c_fold
    phase65c_fold_rows.append({
        "fold": int(phase65c_fold),
        **phase65c_compare(
            phase65c_labels[mask],
            phase65c_anchor[mask],
            phase65c_candidate[mask],
        ),
    })

phase65c_major_mask = np.isin(
    phase65c_groups, PHASE65C_CONFIG["major_groups"]
)
phase65c_major_combined = phase65c_compare(
    phase65c_labels[phase65c_major_mask],
    phase65c_anchor[phase65c_major_mask],
    phase65c_candidate[phase65c_major_mask],
)
phase65c_major_rows = []
phase65c_major_regrets = []
for phase65c_group in PHASE65C_CONFIG["major_groups"]:
    mask = phase65c_groups == phase65c_group
    comparison = phase65c_compare(
        phase65c_labels[mask],
        phase65c_anchor[mask],
        phase65c_candidate[mask],
    )
    phase65c_major_rows.append({
        "group": int(phase65c_group), **comparison
    })
    phase65c_major_regrets.append(max(0.0, -comparison["log_loss_gain"]))

phase65c_domain_rows = []
phase65c_domain_gains = []
for domain, definition in enumerate(phase65c_domain_definitions):
    mask = phase65c_domain_id == domain
    phase65b_require(int(np.sum(mask)) > 0, "phase65c_empty_domain")
    comparison = phase65c_compare(
        phase65c_labels[mask],
        phase65c_anchor[mask],
        phase65c_candidate[mask],
    )
    row = {
        "domain": str(definition["name"]),
        "n": int(np.sum(mask)),
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    }
    phase65c_domain_rows.append(row)
    phase65c_domain_gains.append(comparison["log_loss_gain"])

phase65c_domain_gains = np.asarray(
    phase65c_domain_gains, dtype=np.float64
)
phase65c_bootstrap_rng = np.random.default_rng(
    PHASE65C_CONFIG["domain_bootstrap_seed"]
)
phase65c_bootstrap = np.mean(
    phase65c_domain_gains[phase65c_bootstrap_rng.integers(
        0,
        phase65c_domain_gains.size,
        size=(
            PHASE65C_CONFIG["domain_bootstrap_replicates"],
            phase65c_domain_gains.size,
        ),
    )],
    axis=1,
)
phase65c_q025, phase65c_q50, phase65c_q975 = np.quantile(
    phase65c_bootstrap, [0.025, 0.50, 0.975]
).tolist()
phase65c_domain_rows_sorted = sorted(
    phase65c_domain_rows, key=lambda row: row["log_loss_gain"]
)

phase65c_fold_log_loss_gains = np.asarray([
    row["log_loss_gain"] for row in phase65c_fold_rows
], dtype=np.float64)
phase65c_fold_auroc_gains = np.asarray([
    row["auroc_gain"] for row in phase65c_fold_rows
], dtype=np.float64)
phase65c_max_fold_regret = float(max(
    0.0, -np.min(phase65c_fold_log_loss_gains)
))
phase65c_max_major_regret = float(max(phase65c_major_regrets))
phase65c_gate = PHASE65C_CONFIG["advancement_gate"]
phase65c_advanced = bool(
    phase65c_pooled["log_loss_gain"]
    >= phase65c_gate["minimum_pooled_log_loss_gain"]
    and phase65c_pooled["auroc_gain"]
    >= phase65c_gate["minimum_pooled_auroc_gain"]
    and int(np.sum(phase65c_fold_log_loss_gains > 0.0))
    >= phase65c_gate["minimum_fold_log_loss_wins"]
    and int(np.sum(phase65c_fold_auroc_gains > 0.0))
    >= phase65c_gate["minimum_fold_auroc_wins"]
    and phase65c_max_fold_regret
    <= phase65c_gate["maximum_fold_log_loss_regret"]
    and phase65c_major_combined["log_loss_gain"]
    >= phase65c_gate["minimum_major_groups_combined_log_loss_gain"]
    and phase65c_max_major_regret
    <= phase65c_gate["maximum_individual_major_group_log_loss_regret"]
    and phase65c_q025
    >= phase65c_gate["minimum_domain_bootstrap_lower_95_log_loss_gain"]
    and phase65c_pooled["brier_gain"]
    >= -phase65c_gate["brier_regret_allowed"]
)
phase65c_status = (
    "phase65c_domain_invariant_gate_passed_ready_for_repeated_group_confirmation"
    if phase65c_advanced
    else "phase65c_domain_invariant_gate_failed_stop_candidate"
)

phase65c_contract_core = {
    "schema_version": PHASE65C_CONFIG["schema_version"],
    "status": phase65c_status,
    "source_contract_sha256": str(
        phase65c_validation["contract_sha256"]
    ),
    "phase65b_rejection_contract_sha256": phase65c_phase65b_contract_sha256,
    "candidate": (
        "0.85_anchor_logit_plus_0.15_class_conditional_"
        "domain_adversarial_bilateral_3d_logit"
    ),
    "single_predeclared_candidate": True,
    "outer_images_used_for_fitting": False,
    "outer_labels_used_for_fitting": False,
    "test_data_read": False,
    "standalone_production_restore": bool(PHASE65B_PRODUCTION_RESTORE_USED),
    "requires_previous_notebook_cells": False,
    "phase56_submission_sha256": (
        PHASE65B_RESTORE_SUMMARY_PRIVATE.get("submission_sha256")
        if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict) else None
    ),
    "frozen_run_sha256": phase65c_run_sha256,
}
phase65c_contract_sha256 = hashlib.sha256(json.dumps(
    phase65c_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()
phase65c_contract_path = Path(PHASE65C_CONFIG["contract_file"])
phase65c_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase65c_contract_temporary = phase65c_contract_path.with_suffix(
    phase65c_contract_path.suffix + ".tmp"
)
phase65c_contract_temporary.write_text(json.dumps({
    **phase65c_contract_core,
    "contract_sha256": phase65c_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
    "contains_embeddings": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase65c_contract_temporary, phase65c_contract_path)

PHASE65C_DOMAIN_INVARIANT_STATE_PRIVATE = {
    "status": phase65c_status,
    "contract_sha256": phase65c_contract_sha256,
    "expert_oof_probability": phase65c_expert_oof.copy(),
    "candidate_oof_probability": phase65c_candidate.copy(),
}

phase65c_report = {
    "phase": "phase65c_standalone_class_conditional_domain_invariant_bilateral_3d_gate",
    "status": phase65c_status,
    "restart_safety": {
        "requires_previous_notebook_cells": False,
        "requires_live_kernel_variables_in_production": False,
        "production_restore_used": bool(PHASE65B_PRODUCTION_RESTORE_USED),
        "training_state_reconstructed_from_persistent_artifacts": bool(
            PHASE65B_PRODUCTION_RESTORE_USED
        ),
        "frozen_run_sha256": phase65c_run_sha256,
        "completed_folds_resumed": phase65c_completed_folds_resumed,
        "fold_epoch_resume": phase65c_fold_epoch_resume,
        "invalid_or_stale_private_checkpoint_count": int(
            phase65c_invalid_checkpoint_count
        ),
        "private_checkpoint_paths_exported": False,
    },
    "source_integrity": {
        "phase56_submission_sha256": (
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("submission_sha256")
            if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict) else None
        ),
        "phase62_contract_sha256": (
            PHASE65B_RESTORE_SUMMARY_PRIVATE.get("phase62_contract_sha256")
            if isinstance(PHASE65B_RESTORE_SUMMARY_PRIVATE, dict) else None
        ),
        "phase65b_rejection_contract_sha256": (
            phase65c_phase65b_contract_sha256
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
    "hypothesis": {
        "missing_signal": (
            "disease discrimination stable after suppressing acquisition-group "
            "information and enforcing scanner-view consistency"
        ),
        "novelty_vs_phase65b": (
            "real training-only class-conditional domain confusion; no "
            "procedural anatomy simulator and no synthetic pretraining"
        ),
    },
    "candidate": {
        "single_predeclared_candidate": True,
        "formula": (
            "sigmoid(0.85*anchor_logit+0.15*domain_invariant_expert_logit)"
        ),
        "numeric_advanced": phase65c_advanced,
        "class_conditional_domain_adversary": True,
        "two_scanner_randomized_training_views": True,
        "batch_normalization_used": False,
    },
    "validation": {
        "fold_count": 3,
        "held_out_unit": "complete_acquisition_groups",
        "epoch_selection_on_outer_labels": False,
        "blend_selection_on_outer_labels": False,
        "fold_training": phase65c_fold_training_rows,
    },
    "anchor": phase65c_anchor_metrics,
    "domain_invariant_expert_alone": phase65c_expert_metrics,
    "eligible_candidate": phase65c_pooled,
    "original_folds": phase65c_fold_rows,
    "major_groups": {
        "combined": phase65c_major_combined,
        "individual": phase65c_major_rows,
        "maximum_log_loss_regret": phase65c_max_major_regret,
    },
    "stress_domain_extremes": {
        "worst_three": phase65c_domain_rows_sorted[:3],
        "best_three": phase65c_domain_rows_sorted[-3:][::-1],
    },
    "domain_stability": {
        "domain_macro_log_loss_gain": float(np.mean(phase65c_domain_gains)),
        "domain_win_count": int(np.sum(phase65c_domain_gains > 0.0)),
        "domain_count": int(phase65c_domain_gains.size),
        "bootstrap_replicates": int(
            PHASE65C_CONFIG["domain_bootstrap_replicates"]
        ),
        "bootstrap_lower_95_log_loss_gain": float(phase65c_q025),
        "bootstrap_median_log_loss_gain": float(phase65c_q50),
        "bootstrap_upper_95_log_loss_gain": float(phase65c_q975),
    },
    "gate_summary": {
        "fold_log_loss_wins": int(np.sum(
            phase65c_fold_log_loss_gains > 0.0
        )),
        "fold_auroc_wins": int(np.sum(
            phase65c_fold_auroc_gains > 0.0
        )),
        "maximum_fold_log_loss_regret": phase65c_max_fold_regret,
        "maximum_major_group_log_loss_regret": phase65c_max_major_regret,
        "domain_bootstrap_lower_95_log_loss_gain": float(phase65c_q025),
        "thresholds": dict(phase65c_gate),
    },
    "interpretation_contract": {
        "phase60_independent_3d_candidate_remains_rejected": True,
        "phase65b_physics_synthetic_candidate_remains_rejected": True,
        "each_real_oof_model_excludes_held_out_groups": True,
        "domain_adversary_uses_training_groups_only": True,
        "outer_predictions_scored_only_after_candidate_frozen": True,
        "test_cases_processed_independently": True,
        "no_test_retraining_adaptation_or_batch_statistics": True,
        "passing_result_requires_repeated_group_confirmation": True,
    },
    "training_performed": True,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "case_level_predictions_exported_in_sanitized_output": False,
    "models_exported_in_sanitized_output": False,
    "private_restart_checkpoints_written": True,
    "private_case_level_oof_checkpoints_written": True,
    "private_model_checkpoints_written": True,
    "private_checkpoint_contents_exported_in_sanitized_report": False,
    "synthetic_test_mode": phase65c_is_test,
    "contract_sha256": phase65c_contract_sha256,
    "contract_file": str(phase65c_contract_path),
    "elapsed_seconds": float(time.perf_counter() - phase65c_started),
}

PHASE65C_DOMAIN_INVARIANT_REPORT_PRIVATE = phase65c_report

print("BEGIN SANITIZED_PHASE65C_STANDALONE_DOMAIN_INVARIANT_BILATERAL_3D_GATE")
print(json.dumps(phase65c_report, indent=2, sort_keys=False))
print("END SANITIZED_PHASE65C_STANDALONE_DOMAIN_INVARIANT_BILATERAL_3D_GATE")

gc.collect()
if phase65c_cuda:
    torch.cuda.empty_cache()
