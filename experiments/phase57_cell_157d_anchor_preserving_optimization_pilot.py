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


# Cell 157D — anchor-preserving optimization pilot.
#
# Cell 157C proved the engine works but exposed an intercept-like residual
# drift. This controlled pilot compares four predeclared optimization arms on
# exactly the same training cases, augmentations, initialization, and internal
# monitor set. Original fold-0 validation remains completely untouched.

phase57d_started = time.perf_counter()

assert isinstance(
    globals().get("PHASE57_TRAINING_ENGINE_STATE_PRIVATE"), dict
), {
    "message": (
        "Cell 157D requires PHASE57_TRAINING_ENGINE_STATE_PRIVATE. "
        "Run accepted Phase57 Cell 157C in this kernel first."
    )
}
assert isinstance(
    globals().get("PHASE57_TRAINING_ENGINE_REPORT_PRIVATE"), dict
), {
    "message": "The accepted Phase57 Cell 157C report is missing."
}
for phase57d_required_name in (
    "phase57c_model_class",
    "phase57c_architecture_config",
    "phase57c_pilot_train_indices",
    "phase57c_internal_monitor_indices",
    "phase57c_train_dataset",
    "phase57c_monitor_loader",
    "phase57c_make_train_loader",
    "phase57c_rank_loss",
    "phase57c_metrics",
    "phase57c_clone_state_dict",
    "phase57c_labels",
    "phase57c_groups",
    "phase57c_anchor_probability",
    "phase57c_device",
    "phase57c_cuda",
    "phase57c_amp_dtype",
):
    assert phase57d_required_name in globals(), {
        "message": "A required accepted Cell 157C object is missing.",
        "name": phase57d_required_name,
    }

phase57d_state = PHASE57_TRAINING_ENGINE_STATE_PRIVATE
phase57d_training_engine_report = PHASE57_TRAINING_ENGINE_REPORT_PRIVATE
assert phase57d_training_engine_report["status"] == (
    "accepted_ready_for_frozen_architecture_screen"
)
assert phase57d_training_engine_report["checkpoint_selection"][
    "selected_epoch"
] == 0
phase57d_device = phase57c_device
phase57d_cuda = phase57c_cuda
phase57d_amp_dtype = phase57c_amp_dtype


PHASE57D_CONFIG = {
    "seed": 570357,
    "epoch_count": 4,
    "width_multiplier": 1.0,
    "residual_cap": 0.5,
    "batch_size": 8,
    "warmup_fraction": 0.10,
    "minimum_learning_rate_multiplier": 0.10,
    "weight_decay": 0.02,
    "gradient_clip": 1.0,
    "regret_temperature": 0.02,
    "maximum_rank_pairs_per_batch_group": 64,
    "arms": [
        {
            "label": "plain_conservative",
            "encoder_learning_rate": 2.5e-5,
            "head_learning_rate": 1.0e-4,
            "group_balance_mix": 0.0,
            "anchor_regret_weight": 0.5,
            "residual_l2_weight": 0.10,
            "residual_center_weight": 1.0,
            "rank_loss_weight": 0.0,
        },
        {
            "label": "mixed_moderate",
            "encoder_learning_rate": 5.0e-5,
            "head_learning_rate": 2.0e-4,
            "group_balance_mix": 0.25,
            "anchor_regret_weight": 0.5,
            "residual_l2_weight": 0.05,
            "residual_center_weight": 0.5,
            "rank_loss_weight": 0.02,
        },
        {
            "label": "plain_rank_safe",
            "encoder_learning_rate": 5.0e-5,
            "head_learning_rate": 1.5e-4,
            "group_balance_mix": 0.0,
            "anchor_regret_weight": 1.0,
            "residual_l2_weight": 0.10,
            "residual_center_weight": 1.0,
            "rank_loss_weight": 0.05,
        },
        {
            "label": "light_group_balance",
            "encoder_learning_rate": 2.5e-5,
            "head_learning_rate": 1.0e-4,
            "group_balance_mix": 0.10,
            "anchor_regret_weight": 1.0,
            "residual_l2_weight": 0.20,
            "residual_center_weight": 2.0,
            "rank_loss_weight": 0.02,
        },
    ],
    "selection_log_loss_band": 0.001,
    "safe_maximum_log_loss_excess": 0.001,
    "safe_maximum_auroc_deficit": 0.0005,
    "advance_minimum_log_loss_gain": 0.001,
    "advance_minimum_auroc_gain": 0.001,
    "advance_maximum_log_loss_excess_for_auroc": 0.0005,
    "architecture_screen_if_advanced": {
        "width_multipliers": [0.75, 1.0, 1.25],
        "residual_caps": [0.5, 1.0],
        "candidate_count": 6,
        "fold_count": 3,
        "planned_fit_count": 18,
    },
}

assert len(PHASE57D_CONFIG["arms"]) == 4
assert len({arm["label"] for arm in PHASE57D_CONFIG["arms"]}) == 4
assert PHASE57D_CONFIG["architecture_screen_if_advanced"][
    "candidate_count"
] == (
    len(PHASE57D_CONFIG["architecture_screen_if_advanced"][
        "width_multipliers"
    ])
    * len(PHASE57D_CONFIG["architecture_screen_if_advanced"][
        "residual_caps"
    ])
)


def phase57d_scaled_channels(base_channels, multiplier):
    return [
        max(8, int(round(float(value) * float(multiplier) / 4.0) * 4))
        for value in base_channels
    ]


phase57d_channels = phase57d_scaled_channels(
    phase57c_architecture_config["channels"],
    PHASE57D_CONFIG["width_multiplier"],
)

phase57d_training_groups = phase57c_groups[
    phase57c_pilot_train_indices
]
phase57d_group_counts = np.bincount(
    phase57d_training_groups, minlength=15
).astype(np.float64)
phase57d_active_groups = np.flatnonzero(
    phase57d_group_counts > 0
).astype(np.int64)
phase57d_group_case_weight = np.zeros(15, dtype=np.float32)
phase57d_group_case_weight[phase57d_active_groups] = (
    1.0 / phase57d_group_counts[phase57d_active_groups]
).astype(np.float32)
phase57d_raw_training_weights = phase57d_group_case_weight[
    phase57d_training_groups
]
phase57d_group_case_weight /= float(np.mean(
    phase57d_raw_training_weights
))
phase57d_group_case_weight_tensor = torch.from_numpy(
    phase57d_group_case_weight
).to(phase57d_device)


def phase57d_build_model(seed):
    torch.manual_seed(int(seed))
    if phase57d_cuda:
        torch.cuda.manual_seed_all(int(seed))
    model = phase57c_model_class(
        channels=phase57d_channels,
        blocks_per_stage=phase57c_architecture_config[
            "blocks_per_stage"
        ],
        projection_dimension=phase57c_architecture_config[
            "multiscale_projection_dimension"
        ],
        head_hidden_dimension=phase57c_architecture_config[
            "head_hidden_dimension"
        ],
        residual_cap=PHASE57D_CONFIG["residual_cap"],
    ).to(phase57d_device)
    # A free intercept caused the Phase57C drift. It is fixed at exact zero.
    model.head[-1].bias.requires_grad_(False)
    assert torch.count_nonzero(model.head[-1].bias.detach()).item() == 0
    return model


def phase57d_center_penalty(residual, group):
    global_center = torch.square(torch.mean(residual.float()))
    group_centers = []
    for group_value in torch.unique(group).tolist():
        group_centers.append(torch.square(torch.mean(
            residual[group == int(group_value)].float()
        )))
    group_center = torch.stack(group_centers).mean()
    return global_center + group_center


def phase57d_objective(output, anchor_logit, label, group, arm):
    logit = output["logit"].float()
    residual = output["residual"].float()
    label = label.float()
    anchor_logit = anchor_logit.float()

    per_case = F.binary_cross_entropy_with_logits(
        logit, label, reduction="none"
    )
    anchor_per_case = F.binary_cross_entropy_with_logits(
        anchor_logit, label, reduction="none"
    ).detach()
    ordinary_bce = per_case.mean()
    balanced_bce = torch.mean(
        per_case * phase57d_group_case_weight_tensor[group]
    )
    mix = float(arm["group_balance_mix"])
    classification = (1.0 - mix) * ordinary_bce + mix * balanced_bce

    group_regrets = []
    for group_value in torch.unique(group).tolist():
        mask = group == int(group_value)
        group_regrets.append(
            per_case[mask].mean() - anchor_per_case[mask].mean()
        )
    group_regrets = torch.stack(group_regrets)
    temperature = float(PHASE57D_CONFIG["regret_temperature"])
    # Smooth positive-part penalty. Subtracting the constant leaves gradients
    # unchanged and makes the identity checkpoint report zero regret penalty.
    regret_penalty = torch.mean(
        temperature * F.softplus(group_regrets / temperature)
        - temperature * math.log(2.0)
    )
    residual_l2 = torch.mean(torch.square(residual))
    residual_center = phase57d_center_penalty(residual, group)
    rank, rank_pair_count, rank_group_count = phase57c_rank_loss(
        logit,
        label,
        group,
        PHASE57D_CONFIG["maximum_rank_pairs_per_batch_group"],
    )
    total = (
        classification
        + float(arm["anchor_regret_weight"]) * regret_penalty
        + float(arm["residual_l2_weight"]) * residual_l2
        + float(arm["residual_center_weight"]) * residual_center
        + float(arm["rank_loss_weight"]) * rank
    )
    return {
        "total": total,
        "ordinary_bce": ordinary_bce,
        "balanced_bce": balanced_bce,
        "regret_penalty": regret_penalty,
        "residual_l2": residual_l2,
        "residual_center": residual_center,
        "rank": rank,
        "rank_pair_count": rank_pair_count,
        "rank_group_count": rank_group_count,
    }


@torch.inference_mode()
def phase57d_evaluate(model):
    model.eval()
    probability_parts = []
    residual_parts = []
    label_parts = []
    group_parts = []
    index_parts = []
    for batch in phase57c_monitor_loader:
        volume = batch["volume"].to(
            phase57d_device, non_blocking=phase57d_cuda
        )
        anchor_logit = batch["anchor_logit"].to(
            phase57d_device, non_blocking=phase57d_cuda
        )
        with torch.amp.autocast(
            device_type=phase57d_device.type,
            dtype=phase57d_amp_dtype,
            enabled=phase57d_cuda,
        ):
            output = model(volume, anchor_logit)
        probability_parts.append(
            output["probability"].float().cpu().numpy()
        )
        residual_parts.append(output["residual"].float().cpu().numpy())
        label_parts.append(batch["label"].numpy())
        group_parts.append(batch["group"].numpy())
        index_parts.append(batch["case_index"].numpy())

    probability = np.concatenate(probability_parts).astype(np.float64)
    residual = np.concatenate(residual_parts).astype(np.float64)
    labels = np.concatenate(label_parts).astype(np.int64)
    groups = np.concatenate(group_parts).astype(np.int64)
    indices = np.concatenate(index_parts).astype(np.int64)
    assert np.array_equal(indices, phase57c_internal_monitor_indices)
    metrics = phase57c_metrics(labels, probability)
    metrics.update({
        "mean_residual": float(np.mean(residual)),
        "mean_absolute_residual": float(np.mean(np.abs(residual))),
        "maximum_absolute_residual": float(np.max(np.abs(residual))),
    })
    return metrics, probability, residual, groups


phase57d_anchor_metrics = phase57c_metrics(
    phase57c_labels[phase57c_internal_monitor_indices],
    phase57c_anchor_probability[phase57c_internal_monitor_indices],
)


def phase57d_lr_multiplier(step, total_steps, warmup_steps):
    if step < warmup_steps:
        return float((step + 1) / warmup_steps)
    progress = (
        (step - warmup_steps)
        / max(1, total_steps - warmup_steps - 1)
    )
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    minimum = PHASE57D_CONFIG["minimum_learning_rate_multiplier"]
    return float(minimum + (1.0 - minimum) * cosine)


for phase57d_stale_name in (
    "phase57c_model",
    "phase57c_optimizer",
    "phase57c_best_state",
    "phase57c_best_probability",
):
    globals().pop(phase57d_stale_name, None)
gc.collect()
if phase57d_cuda:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(phase57d_device)

phase57d_arm_records = []
phase57d_checkpoint_records = []
phase57d_steps_per_epoch = math.ceil(
    phase57c_pilot_train_indices.size / PHASE57D_CONFIG["batch_size"]
)
phase57d_total_steps = (
    PHASE57D_CONFIG["epoch_count"] * phase57d_steps_per_epoch
)
phase57d_warmup_steps = max(1, int(round(
    PHASE57D_CONFIG["warmup_fraction"] * phase57d_total_steps
)))

for phase57d_arm_index, phase57d_arm in enumerate(
    PHASE57D_CONFIG["arms"]
):
    phase57d_arm_started = time.perf_counter()
    # Identical initialization across arms isolates the objective/optimizer.
    phase57d_model = phase57d_build_model(PHASE57D_CONFIG["seed"] + 100)
    phase57d_optimizer = torch.optim.AdamW([
        {
            "params": list(phase57d_model.encoder.parameters()),
            "lr": float(phase57d_arm["encoder_learning_rate"]),
        },
        {
            "params": [
                parameter
                for parameter in phase57d_model.head.parameters()
                if parameter.requires_grad
            ],
            "lr": float(phase57d_arm["head_learning_rate"]),
        },
    ], weight_decay=PHASE57D_CONFIG["weight_decay"], betas=(0.9, 0.95))
    phase57d_scheduler = torch.optim.lr_scheduler.LambdaLR(
        phase57d_optimizer,
        lr_lambda=lambda step: phase57d_lr_multiplier(
            step, phase57d_total_steps, phase57d_warmup_steps
        ),
    )

    phase57d_initial_metrics, phase57d_initial_probability, _, _ = (
        phase57d_evaluate(phase57d_model)
    )
    phase57d_identity_error = float(np.max(np.abs(
        phase57d_initial_probability
        - phase57c_anchor_probability[phase57c_internal_monitor_indices]
    )))
    assert phase57d_identity_error <= 1.0e-6

    phase57d_history = []
    phase57d_global_step = 0
    phase57d_best_epoch = 0
    phase57d_best_metrics = dict(phase57d_initial_metrics)
    phase57d_best_probability = phase57d_initial_probability.copy()
    phase57d_best_state = phase57c_clone_state_dict(phase57d_model)

    for phase57d_epoch in range(1, PHASE57D_CONFIG["epoch_count"] + 1):
        phase57d_epoch_started = time.perf_counter()
        phase57d_model.train()
        loader = phase57c_make_train_loader(phase57d_epoch)
        sums = {
            "total": 0.0,
            "ordinary_bce": 0.0,
            "balanced_bce": 0.0,
            "regret_penalty": 0.0,
            "residual_l2": 0.0,
            "residual_center": 0.0,
            "rank": 0.0,
        }
        seen = []
        case_count = 0
        rank_pair_count = 0
        maximum_gradient_norm = 0.0
        for batch in loader:
            volume = batch["volume"].to(
                phase57d_device, non_blocking=phase57d_cuda
            )
            label = batch["label"].to(
                phase57d_device, non_blocking=phase57d_cuda
            )
            group = batch["group"].to(
                phase57d_device, non_blocking=phase57d_cuda
            )
            anchor_logit = batch["anchor_logit"].to(
                phase57d_device, non_blocking=phase57d_cuda
            )
            phase57d_optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(
                device_type=phase57d_device.type,
                dtype=phase57d_amp_dtype,
                enabled=phase57d_cuda,
            ):
                output = phase57d_model(volume, anchor_logit)
                losses = phase57d_objective(
                    output, anchor_logit, label, group, phase57d_arm
                )
            losses["total"].backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                phase57d_model.parameters(),
                PHASE57D_CONFIG["gradient_clip"],
            )
            gradient_norm_value = float(
                gradient_norm.detach().float().item()
            )
            assert math.isfinite(gradient_norm_value)
            phase57d_optimizer.step()
            phase57d_scheduler.step()
            phase57d_global_step += 1

            batch_n = int(label.numel())
            case_count += batch_n
            for name in sums:
                sums[name] += float(losses[name].detach().item()) * batch_n
            rank_pair_count += int(losses["rank_pair_count"])
            maximum_gradient_norm = max(
                maximum_gradient_norm, gradient_norm_value
            )
            seen.extend(batch["case_index"].numpy().astype(int).tolist())

        seen = np.asarray(seen, dtype=np.int64)
        assert case_count == phase57c_pilot_train_indices.size
        assert seen.size == np.unique(seen).size
        assert np.array_equal(
            np.sort(seen), np.sort(phase57c_pilot_train_indices)
        )
        assert rank_pair_count > 0

        metrics, probability, _, _ = phase57d_evaluate(phase57d_model)
        improved = (
            metrics["log_loss"]
            < phase57d_best_metrics["log_loss"] - 1.0e-12
        ) or (
            abs(
                metrics["log_loss"] - phase57d_best_metrics["log_loss"]
            ) <= 1.0e-12
            and metrics["auroc"] > phase57d_best_metrics["auroc"]
        )
        if improved:
            phase57d_best_epoch = phase57d_epoch
            phase57d_best_metrics = dict(metrics)
            phase57d_best_probability = probability.copy()
            phase57d_best_state = phase57c_clone_state_dict(phase57d_model)

        phase57d_history.append({
            "epoch": phase57d_epoch,
            "train": {
                name: value / case_count for name, value in sums.items()
            },
            "rank_pair_count": rank_pair_count,
            "validation": metrics,
            "maximum_gradient_norm": maximum_gradient_norm,
            "improved_over_previous_best": bool(improved),
            "elapsed_seconds": round(
                time.perf_counter() - phase57d_epoch_started, 3
            ),
        })

    assert phase57d_global_step == phase57d_total_steps
    phase57d_model.load_state_dict(phase57d_best_state, strict=True)
    restored_metrics, restored_probability, _, _ = phase57d_evaluate(
        phase57d_model
    )
    restoration_error = float(np.max(np.abs(
        restored_probability - phase57d_best_probability
    )))
    assert restoration_error == 0.0

    phase57d_record = {
        "arm_index": phase57d_arm_index,
        "specification": dict(phase57d_arm),
        "selected_epoch": phase57d_best_epoch,
        "selected_metrics": phase57d_best_metrics,
        "log_loss_gain": (
            phase57d_anchor_metrics["log_loss"]
            - phase57d_best_metrics["log_loss"]
        ),
        "auroc_gain": (
            phase57d_best_metrics["auroc"]
            - phase57d_anchor_metrics["auroc"]
        ),
        "identity_error": phase57d_identity_error,
        "restoration_error": restoration_error,
        "history": phase57d_history,
        "elapsed_seconds": round(
            time.perf_counter() - phase57d_arm_started, 3
        ),
    }
    phase57d_arm_records.append(phase57d_record)
    phase57d_checkpoint_records.append({
        "arm_index": phase57d_arm_index,
        "epoch": phase57d_best_epoch,
        "state_dict": phase57d_best_state,
        "probability": phase57d_best_probability,
    })
    print(
        f"Phase57 anchor-preserving pilot arm "
        f"{phase57d_arm_index + 1}/{len(PHASE57D_CONFIG['arms'])}: "
        f"epoch={phase57d_best_epoch}, "
        f"log_loss={phase57d_best_metrics['log_loss']:.6f}, "
        f"auroc={phase57d_best_metrics['auroc']:.6f}"
    )
    del phase57d_model, phase57d_optimizer, phase57d_scheduler
    gc.collect()
    if phase57d_cuda:
        torch.cuda.empty_cache()


phase57d_nonzero_records = [
    record for record in phase57d_arm_records
    if record["selected_epoch"] > 0
]
phase57d_safe_records = [
    record for record in phase57d_nonzero_records
    if record["selected_metrics"]["log_loss"]
    <= phase57d_anchor_metrics["log_loss"]
    + PHASE57D_CONFIG["safe_maximum_log_loss_excess"]
    and record["selected_metrics"]["auroc"]
    >= phase57d_anchor_metrics["auroc"]
    - PHASE57D_CONFIG["safe_maximum_auroc_deficit"]
]

phase57d_selected_record = None
if phase57d_safe_records:
    minimum_log_loss = min(
        record["selected_metrics"]["log_loss"]
        for record in phase57d_safe_records
    )
    band = [
        record for record in phase57d_safe_records
        if record["selected_metrics"]["log_loss"]
        <= minimum_log_loss + PHASE57D_CONFIG["selection_log_loss_band"]
    ]
    phase57d_selected_record = max(
        band,
        key=lambda record: (
            record["selected_metrics"]["auroc"],
            -record["selected_metrics"]["log_loss"],
            -record["arm_index"],
        ),
    )

phase57d_advance_by_log_loss = bool(
    phase57d_selected_record is not None
    and phase57d_selected_record["log_loss_gain"]
    >= PHASE57D_CONFIG["advance_minimum_log_loss_gain"]
    and phase57d_selected_record["auroc_gain"] >= 0.0
)
phase57d_advance_by_auroc = bool(
    phase57d_selected_record is not None
    and phase57d_selected_record["auroc_gain"]
    >= PHASE57D_CONFIG["advance_minimum_auroc_gain"]
    and phase57d_selected_record["log_loss_gain"]
    >= -PHASE57D_CONFIG["advance_maximum_log_loss_excess_for_auroc"]
)
phase57d_gate_advanced = (
    phase57d_advance_by_log_loss or phase57d_advance_by_auroc
)

phase57d_selected_checkpoint = None
if phase57d_selected_record is not None:
    phase57d_selected_checkpoint = next(
        checkpoint for checkpoint in phase57d_checkpoint_records
        if checkpoint["arm_index"] == phase57d_selected_record["arm_index"]
    )

phase57d_post_training_parity = None
phase57d_post_training_parity_passed = True
if phase57d_selected_checkpoint is not None:
    phase57d_parity_model = phase57d_build_model(
        PHASE57D_CONFIG["seed"] + 100
    )
    phase57d_parity_model.load_state_dict(
        phase57d_selected_checkpoint["state_dict"], strict=True
    )
    phase57d_parity_model.eval()
    phase57d_parity_batch = next(iter(phase57c_monitor_loader))
    phase57d_parity_volume = phase57d_parity_batch["volume"].to(
        phase57d_device, non_blocking=phase57d_cuda
    )
    phase57d_parity_anchor = phase57d_parity_batch["anchor_logit"].to(
        phase57d_device, non_blocking=phase57d_cuda
    )
    with torch.inference_mode():
        with torch.amp.autocast(
            device_type=phase57d_device.type,
            dtype=phase57d_amp_dtype,
            enabled=phase57d_cuda,
        ):
            phase57d_parity_original = phase57d_parity_model(
                phase57d_parity_volume, phase57d_parity_anchor
            )
            phase57d_parity_reflected = phase57d_parity_model(
                torch.flip(phase57d_parity_volume, dims=[2]),
                phase57d_parity_anchor,
            )
        phase57d_parity_batch_float32 = phase57d_parity_model(
            phase57d_parity_volume, phase57d_parity_anchor
        )
        phase57d_parity_single_float32 = phase57d_parity_model(
            phase57d_parity_volume[:1], phase57d_parity_anchor[:1]
        )
    phase57d_amp_reflection_probability_error = float(torch.max(
        torch.abs(
            phase57d_parity_original["probability"].float()
            - phase57d_parity_reflected["probability"].float()
        )
    ).item())
    phase57d_batch_independence_probability_error = float(torch.max(
        torch.abs(
            phase57d_parity_batch_float32["probability"][:1].float()
            - phase57d_parity_single_float32["probability"].float()
        )
    ).item())
    phase57d_post_training_parity_passed = bool(
        phase57d_amp_reflection_probability_error <= 2.0e-3
        and phase57d_batch_independence_probability_error <= 1.0e-4
    )
    phase57d_post_training_parity = {
        "amp_reflection_probability_error": (
            phase57d_amp_reflection_probability_error
        ),
        "float32_batch_independence_probability_error": (
            phase57d_batch_independence_probability_error
        ),
        "passed": phase57d_post_training_parity_passed,
    }
    del phase57d_parity_model
    gc.collect()
    if phase57d_cuda:
        torch.cuda.empty_cache()

phase57d_gate_advanced = bool(
    phase57d_gate_advanced and phase57d_post_training_parity_passed
)

phase57d_contract_core = {
    "schema_version": "phase57_anchor_preserving_optimization_v1",
    "training_engine_contract_sha256": phase57d_state["contract_sha256"],
    "arms": PHASE57D_CONFIG["arms"],
    "selection_rule": (
        "maximum_auroc_within_0p001_of_safe_nonzero_log_loss_minimum"
    ),
    "advance_rule": (
        "log_loss_gain_0p001_without_auroc_loss_or_auroc_gain_0p001_"
        "with_at_most_0p0005_log_loss_excess"
    ),
    "architecture_screen_if_advanced": PHASE57D_CONFIG[
        "architecture_screen_if_advanced"
    ],
}
phase57d_contract_sha256 = hashlib.sha256(json.dumps(
    phase57d_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

phase57d_contract_path = Path(
    phase57d_state["highres_cache_file"]
).parent / "phase57_anchor_preserving_optimization_contract.json"
phase57d_temporary_path = phase57d_contract_path.with_suffix(
    phase57d_contract_path.suffix + ".tmp"
)
phase57d_temporary_path.write_text(json.dumps({
    **phase57d_contract_core,
    "contract_sha256": phase57d_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase57d_temporary_path, phase57d_contract_path)

PHASE57_ANCHOR_OPTIMIZATION_STATE_PRIVATE = {
    "contract_sha256": phase57d_contract_sha256,
    "training_engine_contract_sha256": phase57d_state["contract_sha256"],
    "gate_advanced": phase57d_gate_advanced,
    "selected_record": phase57d_selected_record,
    "selected_checkpoint": phase57d_selected_checkpoint,
    "config": json.loads(json.dumps(PHASE57D_CONFIG)),
}

phase57d_peak_vram_mb = 0.0
if phase57d_cuda:
    phase57d_peak_vram_mb = float(
        torch.cuda.max_memory_allocated(phase57d_device) / (1024 ** 2)
    )

phase57d_report = {
    "phase": "phase57_anchor_preserving_optimization_pilot",
    "status": (
        "optimization_arm_frozen_ready_for_architecture_screen"
        if phase57d_gate_advanced
        else "no_update_selected_stop_before_architecture_screen"
    ),
    "reason_for_intervention": {
        "phase57c_selected_epoch": int(
            phase57d_training_engine_report["checkpoint_selection"][
                "selected_epoch"
            ]
        ),
        "phase57c_epoch_1_mean_absolute_residual": float(
            phase57d_training_engine_report["training"]["history"][0][
                "validation"
            ]["mean_absolute_residual"]
        ),
        "phase57c_epoch_2_mean_absolute_residual": float(
            phase57d_training_engine_report["training"]["history"][1][
                "validation"
            ]["mean_absolute_residual"]
        ),
        "identified_failure": "positive_intercept_like_residual_drift",
    },
    "controlled_changes": [
        "freeze_final_residual_bias_at_zero",
        "reduce_encoder_and_head_learning_rates",
        "blend_ordinary_and_group_balanced_bce",
        "penalize_anchor_relative_group_regret",
        "penalize_residual_magnitude_and_group_mean",
    ],
    "partition": {
        "pilot_train_n": int(phase57c_pilot_train_indices.size),
        "internal_monitor_n": int(
            phase57c_internal_monitor_indices.size
        ),
        "outer_validation_images_used": False,
        "outer_validation_labels_used": False,
    },
    "anchor": phase57d_anchor_metrics,
    "search": {
        "arm_count": len(PHASE57D_CONFIG["arms"]),
        "epoch_count_per_arm": PHASE57D_CONFIG["epoch_count"],
        "nonzero_selected_arm_count": len(phase57d_nonzero_records),
        "safe_nonzero_arm_count": len(phase57d_safe_records),
    },
    "arms": phase57d_arm_records,
    "selected": phase57d_selected_record,
    "gate": {
        "advance_by_log_loss": phase57d_advance_by_log_loss,
        "advance_by_auroc": phase57d_advance_by_auroc,
        "post_training_parity_passed": (
            phase57d_post_training_parity_passed
        ),
        "advanced": phase57d_gate_advanced,
    },
    "post_training_parity": phase57d_post_training_parity,
    "next_screen_if_advanced": PHASE57D_CONFIG[
        "architecture_screen_if_advanced"
    ],
    "contract_sha256": phase57d_contract_sha256,
    "device": str(phase57d_device),
    "peak_vram_mb": phase57d_peak_vram_mb,
    "fit_labels_used": True,
    "internal_monitor_labels_used_for_controlled_selection": True,
    "outer_validation_labels_used": False,
    "logo_labels_used": False,
    "public_leaderboard_used": False,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "smoke_data_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "elapsed_seconds": round(time.perf_counter() - phase57d_started, 3),
}

PHASE57_ANCHOR_OPTIMIZATION_REPORT_PRIVATE = dict(phase57d_report)

print("BEGIN SANITIZED_PHASE57_ANCHOR_OPTIMIZATION")
print(json.dumps(phase57d_report, indent=2))
print("END SANITIZED_PHASE57_ANCHOR_OPTIMIZATION")
