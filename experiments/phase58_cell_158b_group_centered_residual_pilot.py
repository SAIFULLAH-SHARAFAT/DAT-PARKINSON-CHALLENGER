from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


# Cell 158B — group-centered logistic-gradient residual pilot.
# Run immediately after accepted Cell 158A in the same live kernel.
# No outer-fold-0 image/label, test image, smoke image, or leaderboard signal
# is read. Epoch zero is the exact frozen anchor and is always selectable.

phase58b_started = time.perf_counter()

for phase58b_name in (
    "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE",
    "PHASE58_GRADIENT_TRANSPORT_REPORT_PRIVATE",
    "phase57c_model_class",
    "phase57c_architecture_config",
    "phase57c_make_train_loader",
    "phase57c_monitor_loader",
    "phase57c_device",
    "phase57c_cuda",
    "phase57c_amp_dtype",
    "phase57c_clone_state_dict",
    "phase57c_metrics",
):
    assert phase58b_name in globals(), {"missing": phase58b_name}

phase58b_state = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase58b_audit = PHASE58_GRADIENT_TRANSPORT_REPORT_PRIVATE
assert phase58b_audit["status"] == (
    "accepted_positive_intercept_pressure_confirmed_ready_for_"
    "group_centered_gradient_residual_pilot"
)

PHASE58B_CONFIG = {
    "schema_version": "phase58_group_centered_residual_pilot_v1",
    "seed": 580358,
    "epoch_count": 6,
    "batch_size": 8,
    "width_multiplier": 1.0,
    "residual_cap": 0.5,
    "encoder_learning_rate": 2.5e-5,
    "head_learning_rate": 1.0e-4,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "warmup_fraction": 0.10,
    "minimum_learning_rate_multiplier": 0.10,
    "huber_beta": 0.10,
    "global_center_weight": 4.0,
    "batch_group_center_weight": 1.0,
    "residual_l2_weight": 0.05,
    "arms": (
        {"label": "centered_gradient_half", "target_scale": 0.5},
        {"label": "centered_gradient_full", "target_scale": 1.0},
    ),
    "selection_log_loss_band": 0.001,
    "minimum_log_loss_gain": 0.001,
    "minimum_auroc_gain": 0.001,
    "maximum_log_loss_excess_for_auroc_gain": 0.0005,
    "maximum_absolute_mean_residual": 0.02,
    "contract_file": "/kaggle/working/phase58_group_centered_residual_pilot_contract.json",
}

phase58b_device = phase57c_device
phase58b_cuda = phase57c_cuda
phase58b_amp_dtype = phase57c_amp_dtype
phase58b_labels = np.asarray(phase58b_state["labels"], dtype=np.int64)
phase58b_groups = np.asarray(phase58b_state["groups"], dtype=np.int64)
phase58b_anchor_probability = np.asarray(
    phase58b_state["anchor_probability"], dtype=np.float64
)
phase58b_pilot_indices = np.asarray(
    phase58b_state["pilot_indices"], dtype=np.int64
)
phase58b_monitor_indices = np.asarray(
    phase58b_state["monitor_indices"], dtype=np.int64
)
phase58b_pilot_target = np.asarray(
    phase58b_state["pilot_centered_first_order_target"], dtype=np.float32
)
assert phase58b_labels.shape == phase58b_groups.shape == (1362,)
assert phase58b_pilot_indices.shape == phase58b_pilot_target.shape == (256,)
assert phase58b_monitor_indices.shape == (109,)
assert np.intersect1d(phase58b_pilot_indices, phase58b_monitor_indices).size == 0

phase58b_target_lookup = torch.full(
    (1362,), float("nan"), dtype=torch.float32, device=phase58b_device
)
phase58b_target_lookup[torch.from_numpy(phase58b_pilot_indices).to(
    phase58b_device
)] = torch.from_numpy(phase58b_pilot_target).to(phase58b_device)


def phase58b_scaled_channels(channels, multiplier):
    return [
        max(8, int(round(float(c) * float(multiplier) / 4.0) * 4))
        for c in channels
    ]


phase58b_channels = phase58b_scaled_channels(
    phase57c_architecture_config["channels"],
    PHASE58B_CONFIG["width_multiplier"],
)


def phase58b_build_model(seed):
    torch.manual_seed(int(seed))
    if phase58b_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model = phase57c_model_class(
        channels=phase58b_channels,
        blocks_per_stage=phase57c_architecture_config["blocks_per_stage"],
        projection_dimension=phase57c_architecture_config[
            "multiscale_projection_dimension"
        ],
        head_hidden_dimension=phase57c_architecture_config[
            "head_hidden_dimension"
        ],
        residual_cap=PHASE58B_CONFIG["residual_cap"],
    ).to(phase58b_device)
    model.head[-1].bias.requires_grad_(False)
    with torch.no_grad():
        model.head[-1].bias.zero_()
    return model


def phase58b_center_penalty(residual, group):
    global_term = torch.square(residual.float().mean())
    group_terms = []
    for group_value in torch.unique(group).tolist():
        group_terms.append(torch.square(
            residual[group == int(group_value)].float().mean()
        ))
    group_term = torch.stack(group_terms).mean() if group_terms else global_term * 0
    return global_term, group_term


@torch.inference_mode()
def phase58b_evaluate(model):
    model.eval()
    probabilities, residuals, labels, groups, indices = [], [], [], [], []
    for batch in phase57c_monitor_loader:
        volume = batch["volume"].to(phase58b_device, non_blocking=phase58b_cuda)
        anchor_logit = batch["anchor_logit"].to(
            phase58b_device, non_blocking=phase58b_cuda
        )
        with torch.amp.autocast(
            device_type=phase58b_device.type,
            dtype=phase58b_amp_dtype,
            enabled=phase58b_cuda,
        ):
            output = model(volume, anchor_logit)
        probabilities.append(output["probability"].float().cpu().numpy())
        residuals.append(output["residual"].float().cpu().numpy())
        labels.append(batch["label"].numpy())
        groups.append(batch["group"].numpy())
        indices.append(batch["case_index"].numpy())
    probability = np.concatenate(probabilities).astype(np.float64)
    residual = np.concatenate(residuals).astype(np.float64)
    label = np.concatenate(labels).astype(np.int64)
    group = np.concatenate(groups).astype(np.int64)
    index = np.concatenate(indices).astype(np.int64)
    assert np.array_equal(index, phase58b_monitor_indices)
    assert np.array_equal(label, phase58b_labels[index])
    metrics = phase57c_metrics(label, probability)
    group_means = [float(np.mean(residual[group == g])) for g in np.unique(group)]
    metrics.update({
        "mean_residual": float(np.mean(residual)),
        "mean_absolute_residual": float(np.mean(np.abs(residual))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
        "maximum_absolute_group_mean_residual": float(np.max(np.abs(group_means))),
    })
    return metrics, probability


phase58b_anchor_metrics = phase57c_metrics(
    phase58b_labels[phase58b_monitor_indices],
    phase58b_anchor_probability[phase58b_monitor_indices],
)
phase58b_results = []
phase58b_checkpoints = {}

for phase58b_arm_index, phase58b_arm in enumerate(PHASE58B_CONFIG["arms"]):
    seed = PHASE58B_CONFIG["seed"] + 1000 * phase58b_arm_index
    model = phase58b_build_model(seed)
    epoch_zero_metrics, epoch_zero_probability = phase58b_evaluate(model)
    probability_error = float(np.max(np.abs(
        epoch_zero_probability
        - phase58b_anchor_probability[phase58b_monitor_indices]
    )))
    assert probability_error <= 1.0e-6

    optimizer = torch.optim.AdamW([
        {"params": list(model.encoder.parameters()),
         "lr": PHASE58B_CONFIG["encoder_learning_rate"]},
        {"params": list(model.head.parameters()),
         "lr": PHASE58B_CONFIG["head_learning_rate"]},
    ], weight_decay=PHASE58B_CONFIG["weight_decay"], betas=(0.9, 0.95))
    steps_per_epoch = math.ceil(256 / PHASE58B_CONFIG["batch_size"])
    total_steps = PHASE58B_CONFIG["epoch_count"] * steps_per_epoch
    warmup_steps = max(1, int(round(PHASE58B_CONFIG["warmup_fraction"] * total_steps)))

    def lr_multiplier(step):
        if step < warmup_steps:
            return float((step + 1) / warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps - 1)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        floor = PHASE58B_CONFIG["minimum_learning_rate_multiplier"]
        return float(floor + (1.0 - floor) * cosine)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_multiplier)
    history = [{"epoch": 0, "validation": epoch_zero_metrics}]
    best_epoch = 0
    best_metrics = dict(epoch_zero_metrics)
    best_state = phase57c_clone_state_dict(model)
    global_step = 0

    for epoch in range(1, PHASE58B_CONFIG["epoch_count"] + 1):
        model.train()
        loader = phase57c_make_train_loader(epoch)
        loss_sum = 0.0
        case_count = 0
        for batch in loader:
            volume = batch["volume"].to(phase58b_device, non_blocking=phase58b_cuda)
            group = batch["group"].to(phase58b_device, non_blocking=phase58b_cuda)
            anchor_logit = batch["anchor_logit"].to(
                phase58b_device, non_blocking=phase58b_cuda
            )
            case_index = batch["case_index"].to(phase58b_device)
            target = phase58b_target_lookup[case_index] * float(
                phase58b_arm["target_scale"]
            )
            assert torch.all(torch.isfinite(target))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=phase58b_device.type,
                dtype=phase58b_amp_dtype,
                enabled=phase58b_cuda,
            ):
                output = model(volume, anchor_logit)
                residual = output["residual"].float()
                regression = F.smooth_l1_loss(
                    residual, target.float(), beta=PHASE58B_CONFIG["huber_beta"]
                )
                global_center, group_center = phase58b_center_penalty(residual, group)
                loss = (
                    regression
                    + PHASE58B_CONFIG["global_center_weight"] * global_center
                    + PHASE58B_CONFIG["batch_group_center_weight"] * group_center
                    + PHASE58B_CONFIG["residual_l2_weight"] * torch.square(residual).mean()
                )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), PHASE58B_CONFIG["gradient_clip"])
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                model.head[-1].bias.zero_()
            n = int(volume.shape[0])
            loss_sum += float(loss.detach().cpu()) * n
            case_count += n
            global_step += 1
        metrics, _ = phase58b_evaluate(model)
        history.append({
            "epoch": epoch,
            "training_loss": loss_sum / case_count,
            "validation": metrics,
        })
        candidates = [record for record in history if abs(
            record["validation"].get("mean_residual", 0.0)
        ) <= PHASE58B_CONFIG["maximum_absolute_mean_residual"]]
        selected = min(candidates, key=lambda record: (
            record["validation"]["log_loss"],
            -record["validation"]["auroc"],
            record["epoch"],
        ))
        if selected["epoch"] == epoch:
            best_epoch = epoch
            best_metrics = dict(metrics)
            best_state = phase57c_clone_state_dict(model)

    model.load_state_dict(best_state, strict=True)
    log_loss_gain = phase58b_anchor_metrics["log_loss"] - best_metrics["log_loss"]
    auroc_gain = best_metrics["auroc"] - phase58b_anchor_metrics["auroc"]
    advanced = bool(best_epoch > 0 and (
        log_loss_gain >= PHASE58B_CONFIG["minimum_log_loss_gain"]
        or (
            auroc_gain >= PHASE58B_CONFIG["minimum_auroc_gain"]
            and -log_loss_gain <= PHASE58B_CONFIG[
                "maximum_log_loss_excess_for_auroc_gain"
            ]
        )
    ) and abs(best_metrics["mean_residual"]) <= PHASE58B_CONFIG[
        "maximum_absolute_mean_residual"
    ])
    phase58b_results.append({
        "arm_index": phase58b_arm_index,
        "label": phase58b_arm["label"],
        "target_scale": phase58b_arm["target_scale"],
        "selected_epoch": best_epoch,
        "selected_validation": best_metrics,
        "log_loss_gain": float(log_loss_gain),
        "auroc_gain": float(auroc_gain),
        "advanced": advanced,
        "history": history,
    })
    if advanced:
        phase58b_checkpoints[phase58b_arm["label"]] = best_state
    del model, optimizer, scheduler, loader
    gc.collect()
    if phase58b_cuda:
        torch.cuda.empty_cache()

phase58b_advanced = [r for r in phase58b_results if r["advanced"]]
phase58b_selected = None if not phase58b_advanced else min(
    phase58b_advanced,
    key=lambda r: (r["selected_validation"]["log_loss"],
                   -r["selected_validation"]["auroc"]),
)
phase58b_status = (
    "accepted_candidate_ready_for_full_nested_three_fold_confirmation"
    if phase58b_selected is not None
    else "no_update_selected_stop_group_centered_residual_path"
)

phase58b_contract_core = {
    "schema_version": PHASE58B_CONFIG["schema_version"],
    "status": phase58b_status,
    "source_contract_sha256": phase58b_audit["contract_sha256"],
    "config": PHASE58B_CONFIG,
    "selected_arm": None if phase58b_selected is None else phase58b_selected["label"],
    "outer_fold_zero_untouched": True,
    "test_data_used": False,
}
phase58b_contract_sha256 = hashlib.sha256(json.dumps(
    phase58b_contract_core, sort_keys=True, separators=(",", ":")
).encode()).hexdigest()
contract_path = Path(PHASE58B_CONFIG["contract_file"])
contract_path.parent.mkdir(parents=True, exist_ok=True)
temporary_path = contract_path.with_suffix(contract_path.suffix + ".tmp")
temporary_path.write_text(json.dumps({
    **phase58b_contract_core,
    "contract_sha256": phase58b_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n")
os.replace(temporary_path, contract_path)

PHASE58_GROUP_CENTERED_PILOT_STATE_PRIVATE = {
    "status": phase58b_status,
    "contract_sha256": phase58b_contract_sha256,
    "selected_record": phase58b_selected,
    "selected_checkpoint": None if phase58b_selected is None else (
        phase58b_checkpoints[phase58b_selected["label"]]
    ),
    "config": PHASE58B_CONFIG,
}

phase58b_report = {
    "phase": "phase58_group_centered_logistic_gradient_residual_pilot",
    "status": phase58b_status,
    "source_contract_sha256": phase58b_audit["contract_sha256"],
    "partition": {
        "pilot_train_n": 256,
        "internal_monitor_n": 109,
        "outer_validation_images_used": False,
        "outer_validation_labels_used": False,
    },
    "anchor": phase58b_anchor_metrics,
    "arms": phase58b_results,
    "selected_arm": None if phase58b_selected is None else phase58b_selected["label"],
    "contract_sha256": phase58b_contract_sha256,
    "contract_file": str(contract_path),
    "training_target": "training_group_centered_y_minus_anchor_probability",
    "monitor_labels_used_for_checkpoint_selection": True,
    "outer_fold_zero_untouched": True,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase58b_started, 3),
}
PHASE58_GROUP_CENTERED_PILOT_REPORT_PRIVATE = phase58b_report

print("BEGIN SANITIZED_PHASE58_GROUP_CENTERED_RESIDUAL_PILOT")
print(json.dumps(phase58b_report, indent=2))
print("END SANITIZED_PHASE58_GROUP_CENTERED_RESIDUAL_PILOT")
