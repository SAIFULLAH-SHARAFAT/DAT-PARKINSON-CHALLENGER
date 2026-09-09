from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Sampler


# Cell 157C — Phase57 three-fold training-engine pilot.
#
# Run accepted Cells 157A and 157B first. This cell validates the complete
# training machinery on an internal partition drawn only from the fit side of
# original group-held-out fold 0. It never reads fold-0 outer-validation images
# or labels, and it performs no candidate selection or performance claim.

phase57c_started = time.perf_counter()

assert isinstance(
    globals().get("PHASE57_SCANNER_ROBUST_STATE_PRIVATE"), dict
), {
    "message": (
        "Cell 157C requires PHASE57_SCANNER_ROBUST_STATE_PRIVATE. "
        "Run accepted Phase57 Cell 157B in this kernel first."
    )
}
assert isinstance(
    globals().get("PHASE57_SCANNER_ROBUST_CONFIG_PRIVATE"), dict
), {
    "message": "The accepted Phase57 architecture configuration is missing."
}

phase57c_state = PHASE57_SCANNER_ROBUST_STATE_PRIVATE
phase57c_architecture_config = PHASE57_SCANNER_ROBUST_CONFIG_PRIVATE

phase57c_labels = np.asarray(
    phase57c_state["labels"], dtype=np.int64
).reshape(-1)
phase57c_groups = np.asarray(
    phase57c_state["groups"], dtype=np.int64
).reshape(-1)
phase57c_original_fold = np.asarray(
    phase57c_state["original_fold"], dtype=np.int64
).reshape(-1)
phase57c_anchor_probability = np.asarray(
    phase57c_state["anchor_probability"], dtype=np.float64
).reshape(-1)
phase57c_highres_path = Path(phase57c_state["highres_cache_file"])
phase57c_model_class = phase57c_state["model_class"]
phase57c_augment_one = phase57c_state["augment_one"]

assert phase57c_labels.shape == (1362,)
assert phase57c_groups.shape == (1362,)
assert phase57c_original_fold.shape == (1362,)
assert phase57c_anchor_probability.shape == (1362,)
assert phase57c_highres_path.is_file()
assert np.all(np.isin(phase57c_original_fold, [0, 1, 2]))
assert np.all(np.isfinite(phase57c_anchor_probability))
assert np.all(
    (phase57c_anchor_probability > 0.0)
    & (phase57c_anchor_probability < 1.0)
)


PHASE57C_CONFIG = {
    "seed": 570257,
    "outer_fold": 0,
    "internal_monitor_fraction": 0.12,
    "pilot_training_case_count": 256,
    "batch_size": 8,
    "worker_count": 2,
    "epoch_count": 2,
    "width_multiplier": 1.25,
    "residual_cap": 1.0,
    "rank_loss_weight": 0.05,
    "group_dro_weight": 0.05,
    "group_dro_eta": 0.05,
    "maximum_rank_pairs_per_batch_group": 64,
    "encoder_learning_rate": 2.0e-4,
    "head_learning_rate": 8.0e-4,
    "minimum_learning_rate_multiplier": 0.10,
    "warmup_fraction": 0.10,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "architecture_screen": (
        phase57c_architecture_config["planned_screen"]
    ),
}

assert PHASE57C_CONFIG["architecture_screen"]["candidate_count"] == (
    len(PHASE57C_CONFIG["architecture_screen"]["width_multipliers"])
    * len(PHASE57C_CONFIG["architecture_screen"]["residual_caps"])
    * len(PHASE57C_CONFIG["architecture_screen"]["rank_loss_weights"])
    * len(PHASE57C_CONFIG["architecture_screen"]["group_dro_values"])
)

phase57c_device = torch.device(
    "cuda:0" if torch.cuda.is_available() else "cpu"
)
phase57c_cuda = phase57c_device.type == "cuda"
phase57c_amp_dtype = torch.bfloat16

random.seed(PHASE57C_CONFIG["seed"])
np.random.seed(PHASE57C_CONFIG["seed"])
torch.manual_seed(PHASE57C_CONFIG["seed"])
if phase57c_cuda:
    torch.cuda.manual_seed_all(PHASE57C_CONFIG["seed"])


def phase57c_logit(probability):
    probability = np.clip(
        np.asarray(probability, dtype=np.float64),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    return np.log(probability) - np.log1p(-probability)


def phase57c_log_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        1.0e-7,
        1.0 - 1.0e-7,
    )
    assert labels.shape == probability.shape
    return float(np.mean(-(
        labels * np.log(probability)
        + (1.0 - labels) * np.log1p(-probability)
    )))


def phase57c_auroc(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    assert labels.shape == score.shape
    assert set(np.unique(labels).tolist()) == {0, 1}
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
    positive = labels == 1
    positive_n = int(np.sum(positive))
    negative_n = int(labels.size - positive_n)
    assert positive_n > 0 and negative_n > 0
    statistic = (
        float(np.sum(ranks[positive]))
        - positive_n * (positive_n + 1) / 2.0
    )
    return float(statistic / (positive_n * negative_n))


def phase57c_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.asarray(probability, dtype=np.float64).reshape(-1)
    assert labels.shape == probability.shape
    assert np.all(np.isfinite(probability))
    return {
        "n": int(labels.size),
        "log_loss": phase57c_log_loss(labels, probability),
        "auroc": phase57c_auroc(labels, probability),
        "mean_probability": float(np.mean(probability)),
    }


def phase57c_stratified_monitor_split(indices, fraction, seed):
    """Deterministic split within acquisition-group and label strata."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    rng = np.random.default_rng(int(seed))
    fit_parts = []
    monitor_parts = []
    for group in np.unique(phase57c_groups[indices]):
        for label in (0, 1):
            stratum = indices[
                (phase57c_groups[indices] == group)
                & (phase57c_labels[indices] == label)
            ].copy()
            if stratum.size == 0:
                continue
            rng.shuffle(stratum)
            if stratum.size >= 4:
                monitor_n = int(round(float(fraction) * stratum.size))
                monitor_n = min(max(monitor_n, 1), stratum.size - 2)
            else:
                monitor_n = 0
            monitor_parts.append(stratum[:monitor_n])
            fit_parts.append(stratum[monitor_n:])
    fit = np.sort(np.concatenate(fit_parts)).astype(np.int64)
    monitor = np.sort(np.concatenate([
        part for part in monitor_parts if part.size > 0
    ])).astype(np.int64)
    assert fit.size + monitor.size == indices.size
    assert np.intersect1d(fit, monitor).size == 0
    assert np.array_equal(np.sort(np.concatenate([fit, monitor])), indices)
    return fit, monitor


def phase57c_balanced_subset(indices, count, seed):
    """Round-robin across group-label strata without replacement."""
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    assert 0 < int(count) <= indices.size
    rng = np.random.default_rng(int(seed))
    buckets = []
    for group in np.unique(phase57c_groups[indices]):
        for label in (0, 1):
            bucket = indices[
                (phase57c_groups[indices] == group)
                & (phase57c_labels[indices] == label)
            ].copy()
            if bucket.size:
                rng.shuffle(bucket)
                buckets.append(bucket.tolist())
    rng.shuffle(buckets)
    selected = []
    cursor = 0
    while len(selected) < int(count):
        added = False
        for bucket in buckets:
            if cursor < len(bucket):
                selected.append(bucket[cursor])
                added = True
                if len(selected) == int(count):
                    break
        assert added, "Balanced subset exhausted before reaching target size."
        cursor += 1
    selected = np.asarray(selected, dtype=np.int64)
    assert selected.size == np.unique(selected).size == int(count)
    assert np.all(np.isin(selected, indices))
    return selected


phase57c_outer_train_indices = np.flatnonzero(
    phase57c_original_fold != PHASE57C_CONFIG["outer_fold"]
).astype(np.int64)
phase57c_outer_valid_indices = np.flatnonzero(
    phase57c_original_fold == PHASE57C_CONFIG["outer_fold"]
).astype(np.int64)
assert phase57c_outer_train_indices.size == 895
assert phase57c_outer_valid_indices.size == 467
assert np.intersect1d(
    phase57c_groups[phase57c_outer_train_indices],
    phase57c_groups[phase57c_outer_valid_indices],
).size == 0

(
    phase57c_internal_fit_indices,
    phase57c_internal_monitor_indices,
) = phase57c_stratified_monitor_split(
    phase57c_outer_train_indices,
    PHASE57C_CONFIG["internal_monitor_fraction"],
    PHASE57C_CONFIG["seed"],
)
phase57c_pilot_train_indices = phase57c_balanced_subset(
    phase57c_internal_fit_indices,
    PHASE57C_CONFIG["pilot_training_case_count"],
    PHASE57C_CONFIG["seed"] + 1,
)

assert np.intersect1d(
    phase57c_pilot_train_indices,
    phase57c_internal_monitor_indices,
).size == 0
assert np.intersect1d(
    phase57c_pilot_train_indices,
    phase57c_outer_valid_indices,
).size == 0
assert np.intersect1d(
    phase57c_internal_monitor_indices,
    phase57c_outer_valid_indices,
).size == 0
assert set(np.unique(
    phase57c_labels[phase57c_pilot_train_indices]
).tolist()) == {0, 1}
assert set(np.unique(
    phase57c_labels[phase57c_internal_monitor_indices]
).tolist()) == {0, 1}


class Phase57PilotDataset(Dataset):
    def __init__(
        self,
        cache_file,
        indices,
        labels,
        groups,
        anchor_probability,
        augment,
        augmentation_function,
        augmentation_config,
        seed,
    ):
        self.cache_file = str(cache_file)
        self.indices = np.asarray(indices, dtype=np.int64).copy()
        self.labels = np.asarray(labels, dtype=np.int64).copy()
        self.groups = np.asarray(groups, dtype=np.int64).copy()
        self.anchor_logit = phase57c_logit(anchor_probability).astype(
            np.float32
        )
        self.augment = bool(augment)
        self.augmentation_function = augmentation_function
        self.augmentation_config = json.loads(json.dumps(
            augmentation_config
        ))
        self.seed = int(seed)
        self.epoch = 0
        self._cache = None
        assert self.indices.shape == self.labels.shape
        assert self.indices.shape == self.groups.shape
        assert self.indices.shape == self.anchor_logit.shape

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def __len__(self):
        return int(self.indices.size)

    def _resolve_cache(self):
        if self._cache is None:
            self._cache = np.load(self.cache_file, mmap_mode="r")
            assert self._cache.shape == (1362, 80, 80, 80)
            assert self._cache.dtype == np.float16
        return self._cache

    def __getitem__(self, item):
        item = int(item)
        case_index = int(self.indices[item])
        volume = np.asarray(
            self._resolve_cache()[case_index], dtype=np.float32
        ).copy()
        assert volume.shape == (80, 80, 80)
        assert np.all(np.isfinite(volume))
        tensor = torch.from_numpy(volume)[None]
        if self.augment:
            augmentation_seed = (
                self.seed
                + self.epoch * 1_000_003
                + case_index * 97
            )
            tensor = self.augmentation_function(
                tensor[None],
                augmentation_seed,
                self.augmentation_config,
            )[0]
        return {
            "volume": tensor,
            "label": torch.tensor(self.labels[item], dtype=torch.float32),
            "group": torch.tensor(self.groups[item], dtype=torch.long),
            "anchor_logit": torch.tensor(
                self.anchor_logit[item], dtype=torch.float32
            ),
            "case_index": torch.tensor(case_index, dtype=torch.long),
        }


def phase57c_worker_init(worker_id):
    worker_seed = PHASE57C_CONFIG["seed"] + 10_000 * int(worker_id)
    random.seed(worker_seed)
    np.random.seed(worker_seed % (2 ** 32 - 1))
    torch.manual_seed(worker_seed)
    torch.set_num_threads(1)


class Phase57GroupPairOrderSampler(Sampler):
    """One-use ordering that keeps group-local opposite-label pairs intact."""

    def __init__(self, labels, groups, batch_size, seed):
        self.labels = np.asarray(labels, dtype=np.int64).reshape(-1)
        self.groups = np.asarray(groups, dtype=np.int64).reshape(-1)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        assert self.labels.shape == self.groups.shape
        assert self.batch_size > 0 and self.batch_size % 2 == 0

    def __len__(self):
        return int(self.labels.size)

    def __iter__(self):
        rng = np.random.default_rng(self.seed)
        paired_units = []
        unpaired = []
        all_positions = np.arange(self.labels.size, dtype=np.int64)
        for group in np.unique(self.groups):
            positive = all_positions[
                (self.groups == group) & (self.labels == 1)
            ].copy()
            negative = all_positions[
                (self.groups == group) & (self.labels == 0)
            ].copy()
            rng.shuffle(positive)
            rng.shuffle(negative)
            pair_n = min(positive.size, negative.size)
            for pair_index in range(pair_n):
                if rng.random() < 0.5:
                    paired_units.append([
                        int(positive[pair_index]),
                        int(negative[pair_index]),
                    ])
                else:
                    paired_units.append([
                        int(negative[pair_index]),
                        int(positive[pair_index]),
                    ])
            unpaired.extend(positive[pair_n:].astype(int).tolist())
            unpaired.extend(negative[pair_n:].astype(int).tolist())

        rng.shuffle(paired_units)
        rng.shuffle(unpaired)
        # Paired units occupy the prefix. Because batch_size is even, no pair
        # crosses a batch boundary. Remaining unpaired cases follow randomly.
        order = [item for pair in paired_units for item in pair] + unpaired
        assert len(order) == self.labels.size
        assert len(set(order)) == self.labels.size
        return iter(order)


phase57c_train_dataset = Phase57PilotDataset(
    cache_file=phase57c_highres_path,
    indices=phase57c_pilot_train_indices,
    labels=phase57c_labels[phase57c_pilot_train_indices],
    groups=phase57c_groups[phase57c_pilot_train_indices],
    anchor_probability=phase57c_anchor_probability[
        phase57c_pilot_train_indices
    ],
    augment=True,
    augmentation_function=phase57c_augment_one,
    augmentation_config=phase57c_architecture_config[
        "scanner_augmentation"
    ],
    seed=PHASE57C_CONFIG["seed"] + 100,
)
phase57c_monitor_dataset = Phase57PilotDataset(
    cache_file=phase57c_highres_path,
    indices=phase57c_internal_monitor_indices,
    labels=phase57c_labels[phase57c_internal_monitor_indices],
    groups=phase57c_groups[phase57c_internal_monitor_indices],
    anchor_probability=phase57c_anchor_probability[
        phase57c_internal_monitor_indices
    ],
    augment=False,
    augmentation_function=phase57c_augment_one,
    augmentation_config=phase57c_architecture_config[
        "scanner_augmentation"
    ],
    seed=PHASE57C_CONFIG["seed"] + 200,
)


def phase57c_make_train_loader(epoch):
    phase57c_train_dataset.set_epoch(epoch)
    sampler = Phase57GroupPairOrderSampler(
        labels=phase57c_labels[phase57c_pilot_train_indices],
        groups=phase57c_groups[phase57c_pilot_train_indices],
        batch_size=PHASE57C_CONFIG["batch_size"],
        seed=PHASE57C_CONFIG["seed"] + 1000 + int(epoch),
    )
    return DataLoader(
        phase57c_train_dataset,
        batch_size=PHASE57C_CONFIG["batch_size"],
        shuffle=False,
        sampler=sampler,
        num_workers=PHASE57C_CONFIG["worker_count"],
        pin_memory=phase57c_cuda,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=phase57c_worker_init,
    )


phase57c_monitor_loader = DataLoader(
    phase57c_monitor_dataset,
    batch_size=PHASE57C_CONFIG["batch_size"],
    shuffle=False,
    num_workers=PHASE57C_CONFIG["worker_count"],
    pin_memory=phase57c_cuda,
    persistent_workers=False,
    drop_last=False,
    worker_init_fn=phase57c_worker_init,
)


def phase57c_scaled_channels(base_channels, multiplier):
    result = []
    for channel in base_channels:
        scaled = int(round(float(channel) * float(multiplier) / 4.0) * 4)
        result.append(max(8, scaled))
    assert len(result) == 4
    return result


phase57c_pilot_channels = phase57c_scaled_channels(
    phase57c_architecture_config["channels"],
    PHASE57C_CONFIG["width_multiplier"],
)

# Release the architecture-contract model and tensors retained by Cell 157B.
for phase57c_stale_name in (
    "phase57b_model",
    "phase57b_optimizer",
    "phase57b_contract_volume",
    "phase57b_contract_label",
    "phase57b_contract_anchor_logit",
    "phase57b_initial",
    "phase57b_reflected",
    "phase57b_batch_float32",
    "phase57b_reflected_float32",
    "phase57b_single_float32",
):
    globals().pop(phase57c_stale_name, None)
gc.collect()
if phase57c_cuda:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(phase57c_device)


def phase57c_build_model(seed):
    torch.manual_seed(int(seed))
    if phase57c_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model = phase57c_model_class(
        channels=phase57c_pilot_channels,
        blocks_per_stage=phase57c_architecture_config[
            "blocks_per_stage"
        ],
        projection_dimension=phase57c_architecture_config[
            "multiscale_projection_dimension"
        ],
        head_hidden_dimension=phase57c_architecture_config[
            "head_hidden_dimension"
        ],
        residual_cap=PHASE57C_CONFIG["residual_cap"],
    )
    return model.to(phase57c_device)


phase57c_model = phase57c_build_model(PHASE57C_CONFIG["seed"] + 300)
phase57c_model_parameter_count = sum(
    parameter.numel() for parameter in phase57c_model.parameters()
)
assert phase57c_model_parameter_count < 6_000_000


phase57c_training_group_counts = np.bincount(
    phase57c_groups[phase57c_pilot_train_indices], minlength=15
).astype(np.float64)
phase57c_active_training_groups = np.flatnonzero(
    phase57c_training_group_counts > 0
).astype(np.int64)
phase57c_case_weight_by_group = np.zeros(15, dtype=np.float32)
phase57c_case_weight_by_group[phase57c_active_training_groups] = (
    1.0 / phase57c_training_group_counts[phase57c_active_training_groups]
).astype(np.float32)
phase57c_training_case_weights = phase57c_case_weight_by_group[
    phase57c_groups[phase57c_pilot_train_indices]
]
phase57c_case_weight_by_group *= (
    1.0 / float(np.mean(phase57c_training_case_weights))
)
phase57c_case_weight_by_group_tensor = torch.from_numpy(
    phase57c_case_weight_by_group
).to(phase57c_device)

phase57c_dro_weight = torch.zeros(
    15, dtype=torch.float32, device=phase57c_device
)
phase57c_dro_weight[torch.from_numpy(
    phase57c_active_training_groups
).to(phase57c_device)] = 1.0 / float(
    phase57c_active_training_groups.size
)


def phase57c_rank_loss(logit, label, group, maximum_pairs):
    losses = []
    pair_count = 0
    active_group_count = 0
    for group_value in torch.unique(group).tolist():
        mask = group == int(group_value)
        positive = logit[mask & (label > 0.5)]
        negative = logit[mask & (label < 0.5)]
        if positive.numel() == 0 or negative.numel() == 0:
            continue
        differences = (
            positive[:, None] - negative[None, :]
        ).reshape(-1)
        differences = differences[:int(maximum_pairs)]
        losses.append(F.softplus(-differences).mean())
        pair_count += int(differences.numel())
        active_group_count += 1
    if not losses:
        return logit.sum() * 0.0, 0, 0
    return torch.stack(losses).mean(), pair_count, active_group_count


def phase57c_training_objective(logit, label, group):
    per_case_bce = F.binary_cross_entropy_with_logits(
        logit.float(), label.float(), reduction="none"
    )
    case_weight = phase57c_case_weight_by_group_tensor[group]
    balanced_bce = torch.mean(per_case_bce * case_weight)

    present_groups = torch.unique(group)
    group_losses = []
    for group_value in present_groups.tolist():
        group_losses.append(per_case_bce[group == int(group_value)].mean())
    group_losses = torch.stack(group_losses)

    with torch.no_grad():
        phase57c_dro_weight[present_groups] *= torch.exp(
            PHASE57C_CONFIG["group_dro_eta"]
            * group_losses.detach()
        )
        phase57c_dro_weight[phase57c_dro_weight < 0.0] = 0.0
        phase57c_dro_weight[:] /= torch.clamp(
            phase57c_dro_weight.sum(), min=1.0e-12
        )

    present_weight = phase57c_dro_weight[present_groups]
    present_weight = present_weight / torch.clamp(
        present_weight.sum(), min=1.0e-12
    )
    dro_loss = torch.sum(present_weight * group_losses)

    rank_loss, rank_pair_count, rank_group_count = phase57c_rank_loss(
        logit.float(),
        label.float(),
        group,
        PHASE57C_CONFIG["maximum_rank_pairs_per_batch_group"],
    )
    total = (
        balanced_bce
        + PHASE57C_CONFIG["group_dro_weight"] * dro_loss
        + PHASE57C_CONFIG["rank_loss_weight"] * rank_loss
    )
    return {
        "total": total,
        "balanced_bce": balanced_bce,
        "dro": dro_loss,
        "rank": rank_loss,
        "rank_pair_count": rank_pair_count,
        "rank_group_count": rank_group_count,
    }


@torch.inference_mode()
def phase57c_evaluate(model):
    model.eval()
    probabilities = []
    labels = []
    groups = []
    case_indices = []
    residuals = []
    for batch in phase57c_monitor_loader:
        volume = batch["volume"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        anchor_logit = batch["anchor_logit"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        with torch.amp.autocast(
            device_type=phase57c_device.type,
            dtype=phase57c_amp_dtype,
            enabled=phase57c_cuda,
        ):
            output = model(volume, anchor_logit)
        probabilities.append(
            output["probability"].float().cpu().numpy()
        )
        residuals.append(output["residual"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
        case_indices.append(batch["case_index"].numpy())

    probability = np.concatenate(probabilities).astype(np.float64)
    residual = np.concatenate(residuals).astype(np.float64)
    label = np.concatenate(labels).astype(np.int64)
    group = np.concatenate(groups).astype(np.int64)
    case_index = np.concatenate(case_indices).astype(np.int64)
    assert np.array_equal(case_index, phase57c_internal_monitor_indices)
    assert np.array_equal(label, phase57c_labels[case_index])
    assert np.array_equal(group, phase57c_groups[case_index])
    assert np.all(np.isfinite(probability))
    assert np.all((probability > 0.0) & (probability < 1.0))

    result = phase57c_metrics(label, probability)
    result.update({
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "mean_absolute_residual": float(np.mean(np.abs(residual))),
    })
    return result, probability


phase57c_optimizer = torch.optim.AdamW([
    {
        "params": list(phase57c_model.encoder.parameters()),
        "lr": PHASE57C_CONFIG["encoder_learning_rate"],
    },
    {
        "params": list(phase57c_model.head.parameters()),
        "lr": PHASE57C_CONFIG["head_learning_rate"],
    },
], weight_decay=PHASE57C_CONFIG["weight_decay"], betas=(0.9, 0.95))

phase57c_steps_per_epoch = math.ceil(
    len(phase57c_train_dataset) / PHASE57C_CONFIG["batch_size"]
)
phase57c_total_steps = (
    PHASE57C_CONFIG["epoch_count"] * phase57c_steps_per_epoch
)
phase57c_warmup_steps = max(
    1,
    int(round(
        PHASE57C_CONFIG["warmup_fraction"] * phase57c_total_steps
    )),
)


def phase57c_learning_rate_multiplier(step):
    step = int(step)
    if step < phase57c_warmup_steps:
        return float((step + 1) / phase57c_warmup_steps)
    progress = (
        (step - phase57c_warmup_steps)
        / max(1, phase57c_total_steps - phase57c_warmup_steps - 1)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = PHASE57C_CONFIG["minimum_learning_rate_multiplier"]
    return float(minimum + (1.0 - minimum) * cosine)


phase57c_scheduler = torch.optim.lr_scheduler.LambdaLR(
    phase57c_optimizer, lr_lambda=phase57c_learning_rate_multiplier
)

phase57c_epoch_zero_metrics, phase57c_epoch_zero_probability = (
    phase57c_evaluate(phase57c_model)
)
phase57c_monitor_anchor_probability = phase57c_anchor_probability[
    phase57c_internal_monitor_indices
]
phase57c_epoch_zero_probability_error = float(np.max(np.abs(
    phase57c_epoch_zero_probability
    - phase57c_monitor_anchor_probability
)))
assert phase57c_epoch_zero_probability_error <= 1.0e-6, {
    "epoch_zero_anchor_probability_error": (
        phase57c_epoch_zero_probability_error
    )
}


def phase57c_clone_state_dict(model):
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
    }


phase57c_best_epoch = 0
phase57c_best_metrics = dict(phase57c_epoch_zero_metrics)
phase57c_best_probability = phase57c_epoch_zero_probability.copy()
phase57c_best_state = phase57c_clone_state_dict(phase57c_model)
phase57c_history = []
phase57c_global_step = 0
phase57c_all_gradient_norms_finite = True

for phase57c_epoch in range(1, PHASE57C_CONFIG["epoch_count"] + 1):
    phase57c_epoch_started = time.perf_counter()
    phase57c_model.train()
    phase57c_loader = phase57c_make_train_loader(phase57c_epoch)
    phase57c_seen_case_indices = []
    phase57c_loss_sums = {
        "total": 0.0,
        "balanced_bce": 0.0,
        "dro": 0.0,
        "rank": 0.0,
    }
    phase57c_case_count = 0
    phase57c_rank_pair_count = 0
    phase57c_rank_active_batch_count = 0
    phase57c_maximum_gradient_norm = 0.0

    for phase57c_batch in phase57c_loader:
        phase57c_volume = phase57c_batch["volume"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        phase57c_label = phase57c_batch["label"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        phase57c_group = phase57c_batch["group"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        phase57c_anchor_logit = phase57c_batch["anchor_logit"].to(
            phase57c_device, non_blocking=phase57c_cuda
        )
        phase57c_optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(
            device_type=phase57c_device.type,
            dtype=phase57c_amp_dtype,
            enabled=phase57c_cuda,
        ):
            phase57c_output = phase57c_model(
                phase57c_volume, phase57c_anchor_logit
            )
            phase57c_losses = phase57c_training_objective(
                phase57c_output["logit"],
                phase57c_label,
                phase57c_group,
            )

        phase57c_losses["total"].backward()
        phase57c_gradient_norm = torch.nn.utils.clip_grad_norm_(
            phase57c_model.parameters(),
            PHASE57C_CONFIG["gradient_clip"],
        )
        phase57c_gradient_norm_value = float(
            phase57c_gradient_norm.detach().float().item()
        )
        phase57c_all_gradient_norms_finite = (
            phase57c_all_gradient_norms_finite
            and math.isfinite(phase57c_gradient_norm_value)
        )
        assert math.isfinite(phase57c_gradient_norm_value)
        phase57c_optimizer.step()
        phase57c_scheduler.step()
        phase57c_global_step += 1

        phase57c_batch_n = int(phase57c_label.numel())
        phase57c_case_count += phase57c_batch_n
        for phase57c_loss_name in phase57c_loss_sums:
            phase57c_loss_sums[phase57c_loss_name] += (
                float(phase57c_losses[phase57c_loss_name].detach().item())
                * phase57c_batch_n
            )
        phase57c_rank_pair_count += int(
            phase57c_losses["rank_pair_count"]
        )
        phase57c_rank_active_batch_count += int(
            phase57c_losses["rank_pair_count"] > 0
        )
        phase57c_maximum_gradient_norm = max(
            phase57c_maximum_gradient_norm,
            phase57c_gradient_norm_value,
        )
        phase57c_seen_case_indices.extend(
            phase57c_batch["case_index"].numpy().astype(int).tolist()
        )

    phase57c_seen_case_indices = np.asarray(
        phase57c_seen_case_indices, dtype=np.int64
    )
    assert phase57c_case_count == phase57c_pilot_train_indices.size
    assert phase57c_rank_active_batch_count > 0
    assert phase57c_rank_pair_count > 0
    assert phase57c_seen_case_indices.size == np.unique(
        phase57c_seen_case_indices
    ).size
    assert np.array_equal(
        np.sort(phase57c_seen_case_indices),
        np.sort(phase57c_pilot_train_indices),
    )

    phase57c_valid_metrics, phase57c_valid_probability = (
        phase57c_evaluate(phase57c_model)
    )
    phase57c_improved = (
        phase57c_valid_metrics["log_loss"]
        < phase57c_best_metrics["log_loss"] - 1.0e-12
    ) or (
        abs(
            phase57c_valid_metrics["log_loss"]
            - phase57c_best_metrics["log_loss"]
        ) <= 1.0e-12
        and phase57c_valid_metrics["auroc"]
        > phase57c_best_metrics["auroc"]
    )
    if phase57c_improved:
        phase57c_best_epoch = phase57c_epoch
        phase57c_best_metrics = dict(phase57c_valid_metrics)
        phase57c_best_probability = phase57c_valid_probability.copy()
        phase57c_best_state = phase57c_clone_state_dict(phase57c_model)

    phase57c_epoch_record = {
        "epoch": phase57c_epoch,
        "train": {
            name: value / phase57c_case_count
            for name, value in phase57c_loss_sums.items()
        },
        "rank_pair_count": phase57c_rank_pair_count,
        "rank_active_batch_count": phase57c_rank_active_batch_count,
        "validation": phase57c_valid_metrics,
        "maximum_gradient_norm": phase57c_maximum_gradient_norm,
        "learning_rate_multiplier_end": (
            phase57c_learning_rate_multiplier(phase57c_global_step - 1)
        ),
        "each_training_case_used_once": True,
        "improved_over_previous_best": bool(phase57c_improved),
        "elapsed_seconds": round(
            time.perf_counter() - phase57c_epoch_started, 3
        ),
    }
    phase57c_history.append(phase57c_epoch_record)
    print(
        f"Phase57 training-engine pilot epoch {phase57c_epoch}/"
        f"{PHASE57C_CONFIG['epoch_count']}: "
        f"valid_log_loss={phase57c_valid_metrics['log_loss']:.6f}, "
        f"valid_auroc={phase57c_valid_metrics['auroc']:.6f}"
    )

assert phase57c_global_step == phase57c_total_steps
assert phase57c_all_gradient_norms_finite
assert len(phase57c_history) == PHASE57C_CONFIG["epoch_count"]

# Restore the selected checkpoint and verify exact prediction parity.
phase57c_model.load_state_dict(phase57c_best_state, strict=True)
phase57c_restored_metrics, phase57c_restored_probability = (
    phase57c_evaluate(phase57c_model)
)
phase57c_restoration_error = float(np.max(np.abs(
    phase57c_restored_probability - phase57c_best_probability
)))
assert phase57c_restoration_error == 0.0, {
    "checkpoint_restoration_probability_error": phase57c_restoration_error,
}
assert abs(
    phase57c_restored_metrics["log_loss"]
    - phase57c_best_metrics["log_loss"]
) <= 1.0e-12
assert abs(
    phase57c_restored_metrics["auroc"]
    - phase57c_best_metrics["auroc"]
) <= 1.0e-12

# After the head has moved away from zero, quantify the numerical effect of
# BF16 on reflection consistency and re-check per-case batch independence.
phase57c_parity_batch = next(iter(phase57c_monitor_loader))
phase57c_parity_volume = phase57c_parity_batch["volume"].to(
    phase57c_device, non_blocking=phase57c_cuda
)
phase57c_parity_anchor_logit = phase57c_parity_batch[
    "anchor_logit"
].to(phase57c_device, non_blocking=phase57c_cuda)
phase57c_model.eval()
with torch.inference_mode():
    with torch.amp.autocast(
        device_type=phase57c_device.type,
        dtype=phase57c_amp_dtype,
        enabled=phase57c_cuda,
    ):
        phase57c_parity_original = phase57c_model(
            phase57c_parity_volume, phase57c_parity_anchor_logit
        )
        phase57c_parity_reflected = phase57c_model(
            torch.flip(phase57c_parity_volume, dims=[2]),
            phase57c_parity_anchor_logit,
        )
    phase57c_parity_batch_float32 = phase57c_model(
        phase57c_parity_volume, phase57c_parity_anchor_logit
    )
    phase57c_parity_single_float32 = phase57c_model(
        phase57c_parity_volume[:1], phase57c_parity_anchor_logit[:1]
    )

phase57c_restored_amp_reflection_probability_error = float(torch.max(
    torch.abs(
        phase57c_parity_original["probability"].float()
        - phase57c_parity_reflected["probability"].float()
    )
).item())
phase57c_restored_batch_independence_probability_error = float(torch.max(
    torch.abs(
        phase57c_parity_batch_float32["probability"][:1].float()
        - phase57c_parity_single_float32["probability"].float()
    )
).item())
assert phase57c_restored_amp_reflection_probability_error <= 2.0e-3, {
    "restored_amp_reflection_probability_error": (
        phase57c_restored_amp_reflection_probability_error
    ),
}
assert phase57c_restored_batch_independence_probability_error <= 1.0e-4, {
    "restored_float32_batch_independence_probability_error": (
        phase57c_restored_batch_independence_probability_error
    ),
}
phase57c_peak_vram_mb = 0.0
if phase57c_cuda:
    phase57c_peak_vram_mb = float(
        torch.cuda.max_memory_allocated(phase57c_device) / (1024 ** 2)
    )

phase57c_contract_core = {
    "schema_version": "phase57_training_engine_pilot_v1",
    "architecture_contract_sha256": phase57c_state["contract_sha256"],
    "partition": {
        "outer_fold": PHASE57C_CONFIG["outer_fold"],
        "outer_train_n": int(phase57c_outer_train_indices.size),
        "outer_valid_n": int(phase57c_outer_valid_indices.size),
        "pilot_train_n": int(phase57c_pilot_train_indices.size),
        "internal_monitor_n": int(
            phase57c_internal_monitor_indices.size
        ),
        "outer_validation_images_used": False,
        "outer_validation_labels_used": False,
    },
    "pilot_candidate": {
        "width_multiplier": PHASE57C_CONFIG["width_multiplier"],
        "channels": phase57c_pilot_channels,
        "residual_cap": PHASE57C_CONFIG["residual_cap"],
        "rank_loss_weight": PHASE57C_CONFIG["rank_loss_weight"],
        "group_dro_weight": PHASE57C_CONFIG["group_dro_weight"],
    },
    "engine": {
        "group_balanced_case_weights": True,
        "online_group_dro": True,
        "within_batch_group_pairwise_rank_loss": True,
        "group_local_opposite_label_pairs_kept_in_same_batch": True,
        "each_training_case_used_once_per_epoch": True,
        "deterministic_per_case_augmentation": True,
        "epoch_zero_anchor_fallback": True,
        "checkpoint_restoration_verified": True,
        "mixed_precision": (
            "bfloat16" if phase57c_cuda else "disabled_on_cpu"
        ),
    },
    "screen": PHASE57C_CONFIG["architecture_screen"],
}
phase57c_contract_sha256 = hashlib.sha256(json.dumps(
    phase57c_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

phase57c_contract_path = (
    phase57c_highres_path.parent
    / "phase57_training_engine_contract.json"
)
phase57c_temporary_path = phase57c_contract_path.with_suffix(
    phase57c_contract_path.suffix + ".tmp"
)
phase57c_temporary_path.write_text(json.dumps({
    **phase57c_contract_core,
    "contract_sha256": phase57c_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase57c_temporary_path, phase57c_contract_path)

PHASE57_TRAINING_ENGINE_CONFIG_PRIVATE = json.loads(json.dumps(
    PHASE57C_CONFIG
))
PHASE57_TRAINING_ENGINE_STATE_PRIVATE = {
    "contract_sha256": phase57c_contract_sha256,
    "architecture_contract_sha256": phase57c_state["contract_sha256"],
    "dataset_class": Phase57PilotDataset,
    "build_model": phase57c_build_model,
    "model_class": phase57c_model_class,
    "augment_one": phase57c_augment_one,
    "labels": phase57c_labels.copy(),
    "groups": phase57c_groups.copy(),
    "original_fold": phase57c_original_fold.copy(),
    "anchor_probability": phase57c_anchor_probability.copy(),
    "highres_cache_file": str(phase57c_highres_path),
    "architecture_config": json.loads(json.dumps(
        phase57c_architecture_config
    )),
}

phase57c_report = {
    "phase": "phase57_scanner_robust_three_fold_training_engine_pilot",
    "status": "accepted_ready_for_frozen_architecture_screen",
    "purpose": (
        "validate_complete_training_engine_before_72_model_fits"
    ),
    "partition": {
        "outer_fold": PHASE57C_CONFIG["outer_fold"],
        "outer_train_n": int(phase57c_outer_train_indices.size),
        "outer_valid_n": int(phase57c_outer_valid_indices.size),
        "pilot_train_n": int(phase57c_pilot_train_indices.size),
        "internal_monitor_n": int(
            phase57c_internal_monitor_indices.size
        ),
        "pilot_train_group_count": int(np.unique(
            phase57c_groups[phase57c_pilot_train_indices]
        ).size),
        "internal_monitor_group_count": int(np.unique(
            phase57c_groups[phase57c_internal_monitor_indices]
        ).size),
        "outer_validation_images_used": False,
        "outer_validation_labels_used": False,
    },
    "pilot_candidate": {
        "width_multiplier": PHASE57C_CONFIG["width_multiplier"],
        "channels": phase57c_pilot_channels,
        "residual_cap": PHASE57C_CONFIG["residual_cap"],
        "rank_loss_weight": PHASE57C_CONFIG["rank_loss_weight"],
        "group_dro_weight": PHASE57C_CONFIG["group_dro_weight"],
        "parameter_count": phase57c_model_parameter_count,
    },
    "engine_contract": phase57c_contract_core["engine"],
    "epoch_zero": {
        "validation": phase57c_epoch_zero_metrics,
        "anchor_probability_maximum_error": (
            phase57c_epoch_zero_probability_error
        ),
    },
    "training": {
        "epoch_count": PHASE57C_CONFIG["epoch_count"],
        "steps_per_epoch": phase57c_steps_per_epoch,
        "completed_optimizer_steps": phase57c_global_step,
        "history": phase57c_history,
        "all_gradient_norms_finite": (
            phase57c_all_gradient_norms_finite
        ),
    },
    "checkpoint_selection": {
        "selected_epoch": phase57c_best_epoch,
        "selected_validation": phase57c_best_metrics,
        "restored_probability_maximum_error": phase57c_restoration_error,
        "restored_amp_reflection_probability_error": (
            phase57c_restored_amp_reflection_probability_error
        ),
        "restored_float32_batch_independence_probability_error": (
            phase57c_restored_batch_independence_probability_error
        ),
        "epoch_zero_remained_selected": phase57c_best_epoch == 0,
    },
    "next_screen": {
        **PHASE57C_CONFIG["architecture_screen"],
        "fold_count": 3,
        "planned_model_fit_count": 72,
        "outer_validation_used_only_after_each_fold_model_is_frozen": True,
        "logo_confirmation_used_only_after_shared_candidate_is_frozen": True,
    },
    "contract_sha256": phase57c_contract_sha256,
    "device": str(phase57c_device),
    "peak_vram_mb": phase57c_peak_vram_mb,
    "fit_labels_used": True,
    "internal_monitor_labels_used_for_engine_test": True,
    "outer_validation_labels_used": False,
    "logo_labels_used": False,
    "public_leaderboard_used": False,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "case_level_indices_displayed": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase57c_started, 3),
}

PHASE57_TRAINING_ENGINE_REPORT_PRIVATE = dict(phase57c_report)

print("BEGIN SANITIZED_PHASE57_TRAINING_ENGINE")
print(json.dumps(phase57c_report, indent=2))
print("END SANITIZED_PHASE57_TRAINING_ENGINE")
