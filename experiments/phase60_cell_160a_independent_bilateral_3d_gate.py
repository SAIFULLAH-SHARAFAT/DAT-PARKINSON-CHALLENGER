from __future__ import annotations

import copy
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
from torch.utils.data import DataLoader


# Cell 160A — independent bilateral 3D classifier with staged outer-fold gate.
# Run after accepted Cell 159B in the same kernel. The candidate is trained and
# selected using internal-fit/internal-monitor only. Outer fold 0 is constructed
# and read only if the frozen internal candidate advances.

phase60a_started = time.perf_counter()

for phase60a_name in (
    "PHASE59_CALIBRATION_CEILING_REPORT_PRIVATE",
    "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE",
    "phase57c_model_class",
    "phase57c_architecture_config",
    "phase57c_augment_one",
    "Phase57PilotDataset",
    "Phase57GroupPairOrderSampler",
    "phase57c_worker_init",
    "phase57c_rank_loss",
    "phase57c_metrics",
    "phase57c_device",
    "phase57c_cuda",
    "phase57c_amp_dtype",
):
    assert phase60a_name in globals(), {"missing": phase60a_name}

assert PHASE59_CALIBRATION_CEILING_REPORT_PRIVATE["status"] == (
    "calibration_secondary_transport_discrimination_primary"
)

PHASE60A_CONFIG = {
    "schema_version": "phase60_independent_bilateral_3d_gate_v1",
    "seed": 600360,
    "epoch_count": 20,
    "batch_size": 8,
    "worker_count": 2,
    "width_multiplier": 1.0,
    "logit_cap": 8.0,
    "encoder_learning_rate": 1.0e-4,
    "head_learning_rate": 3.0e-4,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "warmup_fraction": 0.08,
    "minimum_learning_rate_multiplier": 0.05,
    "group_balance_mix": 0.25,
    "rank_loss_weight": 0.02,
    "maximum_rank_pairs_per_batch_group": 64,
    "ema_decay": 0.995,
    "blend_alpha_grid": [0.0, 0.10, 0.20, 0.30, 0.40, 0.50],
    "internal_gate": {
        "minimum_log_loss_gain": 0.0015,
        "minimum_auroc_gain": 0.0010,
        "maximum_log_loss_excess_for_auroc_gain": 0.0005,
    },
    "outer_fold_zero_gate": {
        "minimum_log_loss_gain": 0.005,
        "minimum_auroc_gain": 0.002,
    },
    "contract_file": "/kaggle/working/phase60_independent_bilateral_3d_gate_contract.json",
}

phase60a_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase60a_labels = np.asarray(phase60a_state["labels"], dtype=np.int64)
phase60a_groups = np.asarray(phase60a_state["groups"], dtype=np.int64)
phase60a_anchor_probability = np.asarray(
    phase60a_state["anchor_probability"], dtype=np.float64
)
phase60a_internal_fit_indices = np.asarray(
    phase60a_state["internal_fit_indices"], dtype=np.int64
)
phase60a_monitor_indices = np.asarray(
    phase60a_state["monitor_indices"], dtype=np.int64
)
phase60a_outer_indices = np.asarray(phase57c_outer_valid_indices, dtype=np.int64)
phase60a_cache_file = str(phase60a_state["highres_cache_file"])
phase60a_device = phase57c_device
phase60a_cuda = phase57c_cuda
phase60a_amp_dtype = phase57c_amp_dtype

assert phase60a_labels.shape == phase60a_groups.shape == phase60a_anchor_probability.shape == (1362,)
assert phase60a_internal_fit_indices.size == 786
assert phase60a_monitor_indices.size == 109
assert phase60a_outer_indices.size == 467
assert np.intersect1d(phase60a_internal_fit_indices, phase60a_monitor_indices).size == 0
assert np.intersect1d(phase60a_internal_fit_indices, phase60a_outer_indices).size == 0
assert np.intersect1d(phase60a_monitor_indices, phase60a_outer_indices).size == 0

random.seed(PHASE60A_CONFIG["seed"])
np.random.seed(PHASE60A_CONFIG["seed"])
torch.manual_seed(PHASE60A_CONFIG["seed"])
if phase60a_cuda:
    torch.cuda.manual_seed_all(PHASE60A_CONFIG["seed"])


def phase60a_scaled_channels(channels, multiplier):
    return [max(8, int(round(float(c) * float(multiplier) / 4.0) * 4)) for c in channels]


phase60a_channels = phase60a_scaled_channels(
    phase57c_architecture_config["channels"], PHASE60A_CONFIG["width_multiplier"]
)


def phase60a_build_model(seed, prevalence):
    torch.manual_seed(int(seed))
    if phase60a_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model = phase57c_model_class(
        channels=phase60a_channels,
        blocks_per_stage=phase57c_architecture_config["blocks_per_stage"],
        projection_dimension=phase57c_architecture_config["multiscale_projection_dimension"],
        head_hidden_dimension=phase57c_architecture_config["head_hidden_dimension"],
        residual_cap=PHASE60A_CONFIG["logit_cap"],
    ).to(phase60a_device)
    prevalence_logit = math.log(float(prevalence) / (1.0 - float(prevalence)))
    raw_bias = np.arctanh(prevalence_logit / PHASE60A_CONFIG["logit_cap"])
    with torch.no_grad():
        model.head[-1].bias.fill_(float(raw_bias))
    model.head[-1].bias.requires_grad_(True)
    return model


dummy_probability_fit = np.full(phase60a_internal_fit_indices.size, 0.5, dtype=np.float64)
dummy_probability_monitor = np.full(phase60a_monitor_indices.size, 0.5, dtype=np.float64)

phase60a_train_dataset = Phase57PilotDataset(
    cache_file=phase60a_cache_file,
    indices=phase60a_internal_fit_indices,
    labels=phase60a_labels[phase60a_internal_fit_indices],
    groups=phase60a_groups[phase60a_internal_fit_indices],
    anchor_probability=dummy_probability_fit,
    augment=True,
    augmentation_function=phase57c_augment_one,
    augmentation_config=phase57c_architecture_config["scanner_augmentation"],
    seed=PHASE60A_CONFIG["seed"] + 100,
)
phase60a_monitor_dataset = Phase57PilotDataset(
    cache_file=phase60a_cache_file,
    indices=phase60a_monitor_indices,
    labels=phase60a_labels[phase60a_monitor_indices],
    groups=phase60a_groups[phase60a_monitor_indices],
    anchor_probability=dummy_probability_monitor,
    augment=False,
    augmentation_function=phase57c_augment_one,
    augmentation_config=phase57c_architecture_config["scanner_augmentation"],
    seed=PHASE60A_CONFIG["seed"] + 200,
)


def phase60a_train_loader(epoch):
    phase60a_train_dataset.set_epoch(epoch)
    sampler = Phase57GroupPairOrderSampler(
        labels=phase60a_labels[phase60a_internal_fit_indices],
        groups=phase60a_groups[phase60a_internal_fit_indices],
        batch_size=PHASE60A_CONFIG["batch_size"],
        seed=PHASE60A_CONFIG["seed"] + 1000 + int(epoch),
    )
    return DataLoader(
        phase60a_train_dataset,
        batch_size=PHASE60A_CONFIG["batch_size"],
        sampler=sampler,
        shuffle=False,
        num_workers=PHASE60A_CONFIG["worker_count"],
        pin_memory=phase60a_cuda,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=phase57c_worker_init,
    )


phase60a_monitor_loader = DataLoader(
    phase60a_monitor_dataset,
    batch_size=PHASE60A_CONFIG["batch_size"],
    shuffle=False,
    num_workers=PHASE60A_CONFIG["worker_count"],
    pin_memory=phase60a_cuda,
    persistent_workers=False,
    drop_last=False,
    worker_init_fn=phase57c_worker_init,
)

fit_group_counts = np.bincount(
    phase60a_groups[phase60a_internal_fit_indices], minlength=15
).astype(np.float64)
active_groups = fit_group_counts > 0
sqrt_lookup = np.ones(15, dtype=np.float64)
sqrt_lookup[active_groups] = np.power(fit_group_counts[active_groups], -0.5)
case_weights = sqrt_lookup[phase60a_groups[phase60a_internal_fit_indices]]
sqrt_lookup /= float(np.mean(case_weights))
phase60a_group_weight = torch.from_numpy(sqrt_lookup.astype(np.float32)).to(phase60a_device)


def phase60a_objective(logit, label, group):
    per_case = F.binary_cross_entropy_with_logits(logit.float(), label.float(), reduction="none")
    ordinary = per_case.mean()
    balanced = torch.mean(per_case * phase60a_group_weight[group])
    mix = PHASE60A_CONFIG["group_balance_mix"]
    classification = (1.0 - mix) * ordinary + mix * balanced
    rank, pair_count, rank_group_count = phase57c_rank_loss(
        logit.float(), label.float(), group,
        PHASE60A_CONFIG["maximum_rank_pairs_per_batch_group"],
    )
    total = classification + PHASE60A_CONFIG["rank_loss_weight"] * rank
    return total, classification, rank, pair_count, rank_group_count


@torch.inference_mode()
def phase60a_predict(model, loader, expected_indices):
    model.eval()
    probability, logit, labels, groups, indices = [], [], [], [], []
    for batch in loader:
        volume = batch["volume"].to(phase60a_device, non_blocking=phase60a_cuda)
        zero_anchor = torch.zeros(volume.shape[0], device=phase60a_device, dtype=torch.float32)
        with torch.amp.autocast(
            device_type=phase60a_device.type,
            dtype=phase60a_amp_dtype,
            enabled=phase60a_cuda,
        ):
            output = model(volume, zero_anchor)
        probability.append(output["probability"].float().cpu().numpy())
        logit.append(output["logit"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
        indices.append(batch["case_index"].numpy())
    probability = np.concatenate(probability).astype(np.float64)
    logit = np.concatenate(logit).astype(np.float64)
    labels = np.concatenate(labels).astype(np.int64)
    groups = np.concatenate(groups).astype(np.int64)
    indices = np.concatenate(indices).astype(np.int64)
    assert np.array_equal(indices, expected_indices)
    assert np.array_equal(labels, phase60a_labels[indices])
    assert np.array_equal(groups, phase60a_groups[indices])
    return {"probability": probability, "logit": logit, "labels": labels, "groups": groups, "indices": indices}


def phase60a_logit(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-7, 1 - 1e-7)
    return np.log(probability) - np.log1p(-probability)


def phase60a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    result = np.empty_like(logit)
    positive = logit >= 0
    result[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def phase60a_blend_records(prediction, anchor_probability):
    anchor_logit = phase60a_logit(anchor_probability)
    records = []
    for alpha in PHASE60A_CONFIG["blend_alpha_grid"]:
        probability = phase60a_sigmoid(
            (1.0 - float(alpha)) * anchor_logit + float(alpha) * prediction["logit"]
        )
        records.append({
            "alpha": float(alpha),
            "metrics": phase57c_metrics(prediction["labels"], probability),
        })
    return records


def phase60a_update_ema(ema_model, model, decay):
    with torch.no_grad():
        source = dict(model.named_parameters())
        for name, parameter in ema_model.named_parameters():
            parameter.mul_(decay).add_(source[name], alpha=1.0 - decay)
        source_buffers = dict(model.named_buffers())
        for name, buffer in ema_model.named_buffers():
            buffer.copy_(source_buffers[name])


prevalence = float(np.mean(phase60a_labels[phase60a_internal_fit_indices]))
phase60a_model = phase60a_build_model(PHASE60A_CONFIG["seed"] + 300, prevalence)
phase60a_ema_model = copy.deepcopy(phase60a_model).eval()
for parameter in phase60a_ema_model.parameters():
    parameter.requires_grad_(False)

phase60a_optimizer = torch.optim.AdamW([
    {"params": list(phase60a_model.encoder.parameters()), "lr": PHASE60A_CONFIG["encoder_learning_rate"]},
    {"params": list(phase60a_model.head.parameters()), "lr": PHASE60A_CONFIG["head_learning_rate"]},
], weight_decay=PHASE60A_CONFIG["weight_decay"], betas=(0.9, 0.95))

steps_per_epoch = math.ceil(phase60a_internal_fit_indices.size / PHASE60A_CONFIG["batch_size"])
total_steps = steps_per_epoch * PHASE60A_CONFIG["epoch_count"]
warmup_steps = max(1, int(round(total_steps * PHASE60A_CONFIG["warmup_fraction"])))


def phase60a_lr_multiplier(step):
    if step < warmup_steps:
        return float((step + 1) / warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    floor = PHASE60A_CONFIG["minimum_learning_rate_multiplier"]
    return float(floor + (1.0 - floor) * cosine)


phase60a_scheduler = torch.optim.lr_scheduler.LambdaLR(
    phase60a_optimizer, lr_lambda=phase60a_lr_multiplier
)

monitor_anchor = phase60a_anchor_probability[phase60a_monitor_indices]
monitor_anchor_metrics = phase57c_metrics(
    phase60a_labels[phase60a_monitor_indices], monitor_anchor
)
phase60a_history = []
phase60a_best = {
    "epoch": 0,
    "variant": "identity_anchor",
    "alpha": 0.0,
    "metrics": monitor_anchor_metrics,
    "state_dict": None,
}

for epoch in range(1, PHASE60A_CONFIG["epoch_count"] + 1):
    phase60a_model.train()
    loader = phase60a_train_loader(epoch)
    loss_sum = 0.0
    case_count = 0
    pair_count = 0
    maximum_gradient_norm = 0.0
    for batch in loader:
        volume = batch["volume"].to(phase60a_device, non_blocking=phase60a_cuda)
        label = batch["label"].to(phase60a_device, non_blocking=phase60a_cuda)
        group = batch["group"].to(phase60a_device, non_blocking=phase60a_cuda)
        zero_anchor = torch.zeros(volume.shape[0], device=phase60a_device, dtype=torch.float32)
        phase60a_optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast(
            device_type=phase60a_device.type,
            dtype=phase60a_amp_dtype,
            enabled=phase60a_cuda,
        ):
            output = phase60a_model(volume, zero_anchor)
            total, classification, rank, batch_pairs, _ = phase60a_objective(
                output["logit"], label, group
            )
        total.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            phase60a_model.parameters(), PHASE60A_CONFIG["gradient_clip"]
        )
        assert torch.isfinite(gradient_norm)
        maximum_gradient_norm = max(maximum_gradient_norm, float(gradient_norm.detach().cpu()))
        phase60a_optimizer.step()
        phase60a_scheduler.step()
        phase60a_update_ema(phase60a_ema_model, phase60a_model, PHASE60A_CONFIG["ema_decay"])
        n = int(volume.shape[0])
        loss_sum += float(total.detach().cpu()) * n
        case_count += n
        pair_count += int(batch_pairs)

    epoch_record = {
        "epoch": epoch,
        "training_loss": loss_sum / case_count,
        "rank_pair_count": pair_count,
        "maximum_gradient_norm": maximum_gradient_norm,
        "variants": [],
    }
    for variant, evaluation_model in (("raw", phase60a_model), ("ema", phase60a_ema_model)):
        prediction = phase60a_predict(
            evaluation_model, phase60a_monitor_loader, phase60a_monitor_indices
        )
        independent_metrics = phase57c_metrics(prediction["labels"], prediction["probability"])
        blend_records = phase60a_blend_records(prediction, monitor_anchor)
        best_blend = min(blend_records, key=lambda r: (r["metrics"]["log_loss"], -r["metrics"]["auroc"]))
        epoch_record["variants"].append({
            "variant": variant,
            "independent_metrics": independent_metrics,
            "best_blend": best_blend,
            "all_blends": blend_records,
        })
        candidate = {
            "epoch": epoch,
            "variant": variant,
            "alpha": best_blend["alpha"],
            "metrics": best_blend["metrics"],
        }
        if (candidate["metrics"]["log_loss"], -candidate["metrics"]["auroc"]) < (
            phase60a_best["metrics"]["log_loss"], -phase60a_best["metrics"]["auroc"]
        ):
            phase60a_best = {
                **candidate,
                "state_dict": {
                    name: value.detach().cpu().clone()
                    for name, value in evaluation_model.state_dict().items()
                },
            }
    phase60a_history.append(epoch_record)
    print(
        f"Phase60 independent 3D epoch {epoch}/{PHASE60A_CONFIG['epoch_count']}: "
        f"best_log_loss={phase60a_best['metrics']['log_loss']:.6f}, "
        f"best_auroc={phase60a_best['metrics']['auroc']:.6f}, "
        f"alpha={phase60a_best['alpha']:.2f}"
    )

internal_log_loss_gain = monitor_anchor_metrics["log_loss"] - phase60a_best["metrics"]["log_loss"]
internal_auroc_gain = phase60a_best["metrics"]["auroc"] - monitor_anchor_metrics["auroc"]
internal_gate = PHASE60A_CONFIG["internal_gate"]
phase60a_internal_advanced = bool(
    phase60a_best["alpha"] > 0.0
    and (
        (internal_log_loss_gain >= internal_gate["minimum_log_loss_gain"] and internal_auroc_gain >= 0.0)
        or (
            internal_auroc_gain >= internal_gate["minimum_auroc_gain"]
            and -internal_log_loss_gain <= internal_gate["maximum_log_loss_excess_for_auroc_gain"]
        )
    )
)

phase60a_outer_report = {
    "evaluated": False,
    "reason": "internal_gate_failed",
    "outer_validation_images_used": False,
    "outer_validation_labels_used": False,
}
phase60a_outer_advanced = False

if phase60a_internal_advanced:
    frozen_model = phase60a_build_model(PHASE60A_CONFIG["seed"] + 999, prevalence)
    frozen_model.load_state_dict(phase60a_best["state_dict"], strict=True)
    outer_dataset = Phase57PilotDataset(
        cache_file=phase60a_cache_file,
        indices=phase60a_outer_indices,
        labels=phase60a_labels[phase60a_outer_indices],
        groups=phase60a_groups[phase60a_outer_indices],
        anchor_probability=np.full(phase60a_outer_indices.size, 0.5, dtype=np.float64),
        augment=False,
        augmentation_function=phase57c_augment_one,
        augmentation_config=phase57c_architecture_config["scanner_augmentation"],
        seed=PHASE60A_CONFIG["seed"] + 500,
    )
    outer_loader = DataLoader(
        outer_dataset,
        batch_size=PHASE60A_CONFIG["batch_size"],
        shuffle=False,
        num_workers=PHASE60A_CONFIG["worker_count"],
        pin_memory=phase60a_cuda,
        persistent_workers=False,
        drop_last=False,
        worker_init_fn=phase57c_worker_init,
    )
    outer_prediction = phase60a_predict(frozen_model, outer_loader, phase60a_outer_indices)
    outer_anchor = phase60a_anchor_probability[phase60a_outer_indices]
    outer_anchor_metrics = phase57c_metrics(outer_prediction["labels"], outer_anchor)
    alpha = float(phase60a_best["alpha"])
    outer_blend = phase60a_sigmoid(
        (1.0 - alpha) * phase60a_logit(outer_anchor) + alpha * outer_prediction["logit"]
    )
    outer_candidate_metrics = phase57c_metrics(outer_prediction["labels"], outer_blend)
    outer_log_loss_gain = outer_anchor_metrics["log_loss"] - outer_candidate_metrics["log_loss"]
    outer_auroc_gain = outer_candidate_metrics["auroc"] - outer_anchor_metrics["auroc"]
    outer_gate = PHASE60A_CONFIG["outer_fold_zero_gate"]
    phase60a_outer_advanced = bool(
        outer_log_loss_gain >= outer_gate["minimum_log_loss_gain"]
        and outer_auroc_gain >= outer_gate["minimum_auroc_gain"]
    )
    phase60a_outer_report = {
        "evaluated": True,
        "reason": "internal_candidate_frozen_before_outer_read",
        "n": int(phase60a_outer_indices.size),
        "anchor": outer_anchor_metrics,
        "candidate": outer_candidate_metrics,
        "log_loss_gain": float(outer_log_loss_gain),
        "auroc_gain": float(outer_auroc_gain),
        "advanced": phase60a_outer_advanced,
        "outer_validation_images_used": True,
        "outer_validation_labels_used": True,
    }
    del frozen_model, outer_dataset, outer_loader, outer_prediction, outer_blend

if phase60a_outer_advanced:
    phase60a_status = "accepted_ready_for_frozen_three_fold_training"
elif phase60a_internal_advanced:
    phase60a_status = "outer_fold_transport_gate_failed_stop_candidate"
else:
    phase60a_status = "internal_gate_failed_stop_independent_bilateral_candidate"

contract_core = {
    "schema_version": PHASE60A_CONFIG["schema_version"],
    "status": phase60a_status,
    "config": PHASE60A_CONFIG,
    "outer_fold_read_conditional_on_internal_gate": True,
    "test_data_used": False,
}
contract_sha256 = hashlib.sha256(json.dumps(
    contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE60A_CONFIG["contract_file"])
contract_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = contract_path.with_suffix(contract_path.suffix + ".tmp")
temporary_path.write_text(json.dumps({
    **contract_core,
    "contract_sha256": contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, contract_path)

PHASE60_INDEPENDENT_3D_STATE_PRIVATE = {
    "status": phase60a_status,
    "contract_sha256": contract_sha256,
    "selected_epoch": phase60a_best["epoch"],
    "selected_variant": phase60a_best["variant"],
    "selected_alpha": phase60a_best["alpha"],
    "selected_state_dict": phase60a_best["state_dict"] if phase60a_outer_advanced else None,
}

phase60a_report = {
    "phase": "phase60_independent_bilateral_3d_staged_gate",
    "status": phase60a_status,
    "partition": {
        "internal_fit_n": int(phase60a_internal_fit_indices.size),
        "internal_monitor_n": int(phase60a_monitor_indices.size),
        "outer_fold_zero_n": int(phase60a_outer_indices.size),
    },
    "architecture": {
        "channels": phase60a_channels,
        "logit_cap": PHASE60A_CONFIG["logit_cap"],
        "parameter_count": int(sum(p.numel() for p in phase60a_model.parameters())),
        "normalization": "per_case_plus_group_norm_no_batch_norm",
        "independent_classifier_not_anchor_residual": True,
    },
    "internal_monitor": {
        "anchor": monitor_anchor_metrics,
        "selected_epoch": int(phase60a_best["epoch"]),
        "selected_variant": phase60a_best["variant"],
        "selected_alpha": float(phase60a_best["alpha"]),
        "selected_metrics": phase60a_best["metrics"],
        "log_loss_gain": float(internal_log_loss_gain),
        "auroc_gain": float(internal_auroc_gain),
        "advanced": phase60a_internal_advanced,
    },
    "outer_fold_zero": phase60a_outer_report,
    "training_curve": {
        "epoch": [int(record["epoch"]) for record in phase60a_history],
        "training_loss": [float(record["training_loss"]) for record in phase60a_history],
        "best_monitor_log_loss": [float(min(
            variant["best_blend"]["metrics"]["log_loss"]
            for variant in record["variants"]
        )) for record in phase60a_history],
        "best_monitor_auroc_within_epoch": [float(max(
            variant["best_blend"]["metrics"]["auroc"]
            for variant in record["variants"]
        )) for record in phase60a_history],
    },
    "contract_sha256": contract_sha256,
    "contract_file": str(contract_path),
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase60a_started, 3),
}
PHASE60_INDEPENDENT_3D_REPORT_PRIVATE = phase60a_report

print("BEGIN SANITIZED_PHASE60_INDEPENDENT_BILATERAL_3D_GATE")
print(json.dumps(phase60a_report, indent=2))
print("END SANITIZED_PHASE60_INDEPENDENT_BILATERAL_3D_GATE")

del phase60a_model, phase60a_ema_model, phase60a_optimizer, phase60a_scheduler
gc.collect()
if phase60a_cuda:
    torch.cuda.empty_cache()
