from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.fft import dctn
from scipy.spatial.distance import cdist
from sklearn.linear_model import LogisticRegression
from sklearn.utils.extmath import randomized_svd


# Cell 164A — repeated acquisition-group-blocked diffusion-map gate.
#
# Run after Cell 163A in the same live kernel. This is one predeclared new
# representation family, not another router or residual learner. A fixed
# multiscale bilateral DCT representation is constructed independently per
# case. For every outer split, all fitted normalization, PCA, self-tuning
# diffusion-map quantities, and the classifier use only outer-training groups.
# Nyström extension maps each held-out case using only its distances to the
# fitted training manifold. No held-out image or label fits any parameter.

phase64a_started = time.perf_counter()

assert isinstance(globals().get("PHASE63A_SPARSE_RISK_REPORT_PRIVATE"), dict)
assert PHASE63A_SPARSE_RISK_REPORT_PRIVATE["status"] == (
    "sparse_risk_localization_rejected_stop_router_family"
)
assert isinstance(globals().get("PHASE62_VALIDATION_RESET_STATE_PRIVATE"), dict)
assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)

phase64a_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE
phase64a_engine = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE

phase64a_labels = np.asarray(
    phase64a_validation["labels"], dtype=np.int64
).reshape(-1)
phase64a_groups = np.asarray(
    phase64a_validation["groups"], dtype=np.int64
).reshape(-1)
phase64a_anchor = np.asarray(
    phase64a_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase64a_domain_id = np.asarray(
    phase64a_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase64a_domain_definitions = phase64a_validation["stress_domain_definitions"]
phase64a_repeat_fold = np.asarray(
    phase64a_validation["group_blocked_repeat_case_fold"], dtype=np.int64
)

phase64a_synthetic_override = globals().get(
    "PHASE64A_SYNTHETIC_TEST_OVERRIDE_PRIVATE"
)
phase64a_is_synthetic = isinstance(phase64a_synthetic_override, dict)
phase64a_expected_n = (
    int(phase64a_synthetic_override["case_count"])
    if phase64a_is_synthetic else 1362
)
phase64a_expected_group_count = (
    int(phase64a_synthetic_override.get("group_count", 15))
    if phase64a_is_synthetic else 15
)
phase64a_expected_repeat_count = (
    int(phase64a_synthetic_override.get("repeat_count", 2))
    if phase64a_is_synthetic else 5
)

assert phase64a_labels.shape == (phase64a_expected_n,)
assert phase64a_groups.shape == (phase64a_expected_n,)
assert phase64a_anchor.shape == (phase64a_expected_n,)
assert phase64a_domain_id.shape == (phase64a_expected_n,)
assert phase64a_repeat_fold.shape == (
    phase64a_expected_repeat_count, phase64a_expected_n
)
assert set(np.unique(phase64a_labels).tolist()) == {0, 1}
assert np.array_equal(
    np.unique(phase64a_groups), np.arange(phase64a_expected_group_count)
)
assert np.all(np.isfinite(phase64a_anchor))
assert np.all((phase64a_anchor > 0.0) & (phase64a_anchor < 1.0))
assert np.all(np.isin(phase64a_repeat_fold, [0, 1, 2]))
for repeat in range(phase64a_expected_repeat_count):
    for group in range(phase64a_expected_group_count):
        group_folds = np.unique(
            phase64a_repeat_fold[repeat, phase64a_groups == group]
        )
        assert group_folds.size == 1


PHASE64A_CONFIG = {
    "schema_version": "phase64_group_blocked_diffusion_nystrom_gate_v1",
    "seed": 640264,
    "probability_clip": 1.0e-7,
    "input_shape": [80, 80, 80],
    "pooled_shape": [20, 20, 20],
    "dct_keep_shape": [8, 8, 8],
    "positive_lower_quantile": 0.10,
    "positive_upper_quantile": 0.995,
    "log_compression": 4.0,
    "pca_dimension": 64,
    "pca_scale_floor_fraction": 0.10,
    "diffusion_neighbor_rank": 15,
    "diffusion_alpha": 1.0,
    "diffusion_dimension": 24,
    "minimum_diffusion_eigenvalue": 1.0e-6,
    "logistic_c": 0.20,
    "maximum_logistic_iterations": 3000,
    "anchor_logit_weight": 0.80,
    "diffusion_logit_weight": 0.20,
    "major_groups": [1, 3],
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 640265,
    "advancement_gate": {
        "minimum_averaged_oof_log_loss_gain": 0.005,
        "minimum_averaged_oof_auroc_gain": 0.0015,
        "minimum_repeat_log_loss_wins": 4,
        "minimum_repeat_auroc_wins": 3,
        "maximum_repeat_log_loss_regret": 0.002,
        "minimum_major_groups_combined_log_loss_gain": 0.005,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
        "brier_regret_allowed": 0.0,
    },
    "contract_file": (
        "/kaggle/working/phase64_group_blocked_diffusion_nystrom_contract.json"
    ),
}

if phase64a_is_synthetic:
    PHASE64A_CONFIG = {
        **PHASE64A_CONFIG,
        "input_shape": list(phase64a_synthetic_override.get(
            "input_shape", [16, 16, 16]
        )),
        "pooled_shape": list(phase64a_synthetic_override.get(
            "pooled_shape", [4, 4, 4]
        )),
        "dct_keep_shape": list(phase64a_synthetic_override.get(
            "dct_keep_shape", [4, 4, 4]
        )),
        "pca_dimension": int(phase64a_synthetic_override.get(
            "pca_dimension", 12
        )),
        "diffusion_neighbor_rank": int(phase64a_synthetic_override.get(
            "diffusion_neighbor_rank", 5
        )),
        "diffusion_dimension": int(phase64a_synthetic_override.get(
            "diffusion_dimension", 6
        )),
        "domain_bootstrap_replicates": int(
            phase64a_synthetic_override.get("domain_bootstrap_replicates", 200)
        ),
        "contract_file": str(phase64a_synthetic_override.get(
            "contract_file",
            os.path.join(tempfile.gettempdir(), "phase64_synthetic_contract.json"),
        )),
    }

assert abs(
    PHASE64A_CONFIG["anchor_logit_weight"]
    + PHASE64A_CONFIG["diffusion_logit_weight"]
    - 1.0
) <= 1.0e-15


def phase64a_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE64A_CONFIG["probability_clip"],
        1.0 - PHASE64A_CONFIG["probability_clip"],
    )


def phase64a_logit(probability):
    probability = phase64a_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase64a_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase64a_auc(labels, score):
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
    u_statistic = float(np.sum(ranks[labels == 1]))
    u_statistic -= positive_n * (positive_n + 1) / 2.0
    return float(u_statistic / (positive_n * negative_n))


def phase64a_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase64a_clip(probability)
    return -(
        labels * np.log(probability)
        + (1 - labels) * np.log1p(-probability)
    )


def phase64a_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase64a_clip(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase64a_case_loss(labels, probability))),
        "auroc": phase64a_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def phase64a_compare(labels, anchor, candidate):
    anchor_metrics = phase64a_metrics(labels, anchor)
    candidate_metrics = phase64a_metrics(labels, candidate)
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


def phase64a_group_balanced_weight(groups):
    groups = np.asarray(groups, dtype=np.int64)
    unique, counts = np.unique(groups, return_counts=True)
    count_map = {int(group): int(count) for group, count in zip(unique, counts)}
    weight = np.asarray(
        [1.0 / count_map[int(group)] for group in groups], dtype=np.float64
    )
    weight *= weight.size / np.sum(weight)
    return weight


def phase64a_weighted_mean_scale(features, weights):
    features = np.asarray(features, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    total = float(np.sum(weights))
    mean = np.sum(features * weights[:, None], axis=0) / total
    centered = features - mean
    variance = np.sum(
        np.square(centered) * weights[:, None], axis=0
    ) / total
    raw_scale = np.sqrt(np.maximum(variance, 0.0))
    positive_scale = raw_scale[raw_scale > 1.0e-10]
    reference = (
        float(np.median(positive_scale)) if positive_scale.size else 1.0
    )
    scale = np.maximum(raw_scale, 0.10 * reference)
    scale = np.maximum(scale, 1.0e-6)
    return mean, scale


def phase64a_extract_case_feature(volume):
    volume = np.asarray(volume, dtype=np.float32)
    assert list(volume.shape) == PHASE64A_CONFIG["input_shape"]
    positive = volume[volume > 0.0]
    if positive.size == 0:
        normalized = np.zeros_like(volume, dtype=np.float32)
    else:
        lower = float(np.quantile(
            positive, PHASE64A_CONFIG["positive_lower_quantile"]
        ))
        upper = float(np.quantile(
            positive, PHASE64A_CONFIG["positive_upper_quantile"]
        ))
        scale = max(upper - lower, 1.0e-6)
        normalized = np.clip((volume - lower) / scale, 0.0, 1.0)
    compression = float(PHASE64A_CONFIG["log_compression"])
    normalized = np.log1p(compression * normalized) / np.log1p(compression)

    input_shape = np.asarray(PHASE64A_CONFIG["input_shape"], dtype=np.int64)
    pooled_shape = np.asarray(PHASE64A_CONFIG["pooled_shape"], dtype=np.int64)
    assert np.all(input_shape % pooled_shape == 0)
    block = input_shape // pooled_shape
    pooled = normalized.reshape(
        int(pooled_shape[0]), int(block[0]),
        int(pooled_shape[1]), int(block[1]),
        int(pooled_shape[2]), int(block[2]),
    ).mean(axis=(1, 3, 5))

    reflected = pooled[::-1, :, :]
    symmetric = 0.5 * (pooled + reflected)
    asymmetric = np.abs(pooled - reflected)
    keep = tuple(int(value) for value in PHASE64A_CONFIG["dct_keep_shape"])
    symmetric_dct = dctn(symmetric, norm="ortho")[:keep[0], :keep[1], :keep[2]]
    asymmetric_dct = dctn(asymmetric, norm="ortho")[:keep[0], :keep[1], :keep[2]]
    return np.concatenate([
        symmetric_dct.reshape(-1),
        asymmetric_dct.reshape(-1),
    ]).astype(np.float32, copy=False)


phase64a_cache_path = Path(phase64a_engine["highres_cache_file"])
assert phase64a_cache_path.is_file(), {"missing": str(phase64a_cache_path)}
phase64a_cache = np.load(phase64a_cache_path, mmap_mode="r", allow_pickle=False)
assert phase64a_cache.shape == (
    phase64a_expected_n, *PHASE64A_CONFIG["input_shape"]
)
assert phase64a_cache.dtype == np.float16

phase64a_first_feature = phase64a_extract_case_feature(phase64a_cache[0])
phase64a_feature_count = int(phase64a_first_feature.size)
phase64a_features = np.empty(
    (phase64a_expected_n, phase64a_feature_count), dtype=np.float32
)
phase64a_features[0] = phase64a_first_feature
for index in range(1, phase64a_expected_n):
    phase64a_features[index] = phase64a_extract_case_feature(
        phase64a_cache[index]
    )
assert np.all(np.isfinite(phase64a_features))


def phase64a_fit_weighted_pca(train_features, train_weights, seed):
    train_features = np.asarray(train_features, dtype=np.float64)
    train_weights = np.asarray(train_weights, dtype=np.float64)
    mean, scale = phase64a_weighted_mean_scale(train_features, train_weights)
    standardized = (train_features - mean) / scale
    dimension = min(
        int(PHASE64A_CONFIG["pca_dimension"]),
        standardized.shape[0] - 2,
        standardized.shape[1],
    )
    assert dimension >= int(PHASE64A_CONFIG["diffusion_dimension"]) + 2
    weighted = standardized * np.sqrt(train_weights[:, None])
    _, singular, components = randomized_svd(
        weighted,
        n_components=dimension,
        n_iter=5,
        random_state=int(seed),
        flip_sign=True,
    )
    eigen_scale = singular / np.sqrt(max(np.sum(train_weights) - 1.0, 1.0))
    reference = float(np.median(eigen_scale[eigen_scale > 1.0e-10]))
    floor = PHASE64A_CONFIG["pca_scale_floor_fraction"] * reference
    eigen_scale = np.maximum(eigen_scale, max(floor, 1.0e-6))
    return {
        "mean": mean,
        "scale": scale,
        "components": components,
        "eigen_scale": eigen_scale,
    }


def phase64a_apply_weighted_pca(features, state):
    standardized = (
        np.asarray(features, dtype=np.float64) - state["mean"]
    ) / state["scale"]
    projected = standardized @ state["components"].T
    return projected / state["eigen_scale"]


def phase64a_pairwise_squared(left, right):
    distance = cdist(
        np.asarray(left, dtype=np.float64),
        np.asarray(right, dtype=np.float64),
        metric="sqeuclidean",
    )
    return np.maximum(distance, 0.0)


def phase64a_positive_scale(values, fallback):
    values = np.asarray(values, dtype=np.float64)
    positive = values[values > 1.0e-12]
    if positive.size:
        return float(np.median(positive))
    return float(max(fallback, 1.0e-6))


def phase64a_fit_diffusion(train_projected):
    train_projected = np.asarray(train_projected, dtype=np.float64)
    train_n = train_projected.shape[0]
    neighbor_rank = min(
        int(PHASE64A_CONFIG["diffusion_neighbor_rank"]), train_n - 2
    )
    assert neighbor_rank >= 2
    squared = phase64a_pairwise_squared(train_projected, train_projected)
    np.fill_diagonal(squared, np.inf)
    local_squared = np.partition(squared, neighbor_rank - 1, axis=1)[
        :, neighbor_rank - 1
    ]
    finite_distance = squared[np.isfinite(squared)]
    fallback = np.sqrt(phase64a_positive_scale(finite_distance, 1.0))
    local_scale = np.sqrt(np.maximum(local_squared, 0.0))
    local_scale = np.maximum(local_scale, max(1.0e-6, fallback * 1.0e-3))
    np.fill_diagonal(squared, 0.0)

    denominator = local_scale[:, None] * local_scale[None, :]
    kernel = np.exp(-squared / np.maximum(denominator, 1.0e-12))
    density = np.maximum(np.sum(kernel, axis=1), 1.0e-12)
    alpha = float(PHASE64A_CONFIG["diffusion_alpha"])
    normalized = kernel / (
        np.power(density[:, None], alpha)
        * np.power(density[None, :], alpha)
    )
    degree = np.maximum(np.sum(normalized, axis=1), 1.0e-12)
    symmetric_operator = normalized / np.sqrt(
        degree[:, None] * degree[None, :]
    )
    symmetric_operator = 0.5 * (
        symmetric_operator + symmetric_operator.T
    )

    eigenvalue, eigenvector = np.linalg.eigh(symmetric_operator)
    order = np.argsort(eigenvalue)[::-1]
    eigenvalue = eigenvalue[order]
    eigenvector = eigenvector[:, order]
    requested = int(PHASE64A_CONFIG["diffusion_dimension"])
    usable = np.flatnonzero(
        eigenvalue[1:] > PHASE64A_CONFIG["minimum_diffusion_eigenvalue"]
    ) + 1
    assert usable.size >= requested, {
        "usable": int(usable.size), "requested": requested
    }
    selected = usable[:requested]
    selected_eigenvalue = eigenvalue[selected]
    psi = eigenvector[:, selected] / np.sqrt(degree[:, None])
    train_coordinate = psi * selected_eigenvalue[None, :]
    return {
        "train_projected": train_projected,
        "local_scale": local_scale,
        "density": density,
        "degree": degree,
        "psi": psi,
        "eigenvalue": selected_eigenvalue,
        "train_coordinate": train_coordinate,
        "neighbor_rank": int(neighbor_rank),
    }


def phase64a_nystrom(valid_projected, state):
    valid_projected = np.asarray(valid_projected, dtype=np.float64)
    squared = phase64a_pairwise_squared(
        valid_projected, state["train_projected"]
    )
    neighbor_rank = int(state["neighbor_rank"])
    local_squared = np.partition(squared, neighbor_rank - 1, axis=1)[
        :, neighbor_rank - 1
    ]
    fallback = phase64a_positive_scale(state["local_scale"], 1.0)
    local_scale = np.sqrt(np.maximum(local_squared, 0.0))
    local_scale = np.maximum(local_scale, max(1.0e-6, fallback * 1.0e-3))
    denominator = local_scale[:, None] * state["local_scale"][None, :]
    kernel = np.exp(-squared / np.maximum(denominator, 1.0e-12))
    density = np.maximum(np.sum(kernel, axis=1), 1.0e-12)
    alpha = float(PHASE64A_CONFIG["diffusion_alpha"])
    normalized = kernel / (
        np.power(density[:, None], alpha)
        * np.power(state["density"][None, :], alpha)
    )
    degree = np.maximum(np.sum(normalized, axis=1), 1.0e-12)
    transition = normalized / degree[:, None]
    return transition @ state["psi"]


def phase64a_fit_predict(
    train_indices, valid_indices, fit_seed
):
    train_indices = np.asarray(train_indices, dtype=np.int64)
    valid_indices = np.asarray(valid_indices, dtype=np.int64)
    train_groups = phase64a_groups[train_indices]
    valid_groups = phase64a_groups[valid_indices]
    assert not set(train_groups.tolist()) & set(valid_groups.tolist())
    train_labels = phase64a_labels[train_indices]
    assert np.unique(train_labels).tolist() == [0, 1]
    train_weights = phase64a_group_balanced_weight(train_groups)

    pca = phase64a_fit_weighted_pca(
        phase64a_features[train_indices], train_weights, fit_seed
    )
    train_projected = phase64a_apply_weighted_pca(
        phase64a_features[train_indices], pca
    )
    valid_projected = phase64a_apply_weighted_pca(
        phase64a_features[valid_indices], pca
    )
    diffusion = phase64a_fit_diffusion(train_projected)
    train_coordinate = diffusion["train_coordinate"]
    valid_coordinate = phase64a_nystrom(valid_projected, diffusion)

    coordinate_mean, coordinate_scale = phase64a_weighted_mean_scale(
        train_coordinate, train_weights
    )
    train_standardized = (
        train_coordinate - coordinate_mean
    ) / coordinate_scale
    valid_standardized = (
        valid_coordinate - coordinate_mean
    ) / coordinate_scale

    classifier = LogisticRegression(
        C=float(PHASE64A_CONFIG["logistic_c"]),
        solver="lbfgs",
        max_iter=int(PHASE64A_CONFIG["maximum_logistic_iterations"]),
        random_state=int(fit_seed) + 1,
    )
    classifier.fit(
        train_standardized, train_labels, sample_weight=train_weights
    )
    valid_logit = classifier.decision_function(valid_standardized)

    weighted_prevalence = float(
        np.sum(train_weights * train_labels) / np.sum(train_weights)
    )
    ordinary_prevalence = float(np.mean(train_labels))
    prior_correction = float(
        phase64a_logit(ordinary_prevalence)
        - phase64a_logit(weighted_prevalence)
    )
    valid_probability = phase64a_sigmoid(valid_logit + prior_correction)
    diagnostic = {
        "train_n": int(train_indices.size),
        "valid_n": int(valid_indices.size),
        "train_group_count": int(np.unique(train_groups).size),
        "valid_group_count": int(np.unique(valid_groups).size),
        "pca_dimension": int(pca["components"].shape[0]),
        "diffusion_dimension": int(diffusion["psi"].shape[1]),
        "largest_nontrivial_eigenvalue": float(diffusion["eigenvalue"][0]),
        "smallest_retained_eigenvalue": float(diffusion["eigenvalue"][-1]),
        "ordinary_training_prevalence": ordinary_prevalence,
        "group_balanced_training_prevalence": weighted_prevalence,
        "prior_correction": prior_correction,
        "outer_images_used_for_fitting": False,
        "outer_labels_used_for_fitting": False,
    }
    return phase64a_clip(valid_probability), diagnostic


phase64a_repeat_count = phase64a_repeat_fold.shape[0]
phase64a_expert_by_repeat = np.full(
    (phase64a_repeat_count, phase64a_expected_n), np.nan, dtype=np.float64
)
phase64a_candidate_by_repeat = np.full_like(
    phase64a_expert_by_repeat, np.nan
)
phase64a_fit_records = []
phase64a_anchor_logit = phase64a_logit(phase64a_anchor)

for repeat in range(phase64a_repeat_count):
    for fold in range(3):
        valid_indices = np.flatnonzero(phase64a_repeat_fold[repeat] == fold)
        train_indices = np.flatnonzero(phase64a_repeat_fold[repeat] != fold)
        assert train_indices.size > 0 and valid_indices.size > 0
        expert_probability, diagnostic = phase64a_fit_predict(
            train_indices,
            valid_indices,
            PHASE64A_CONFIG["seed"] + 100 * repeat + fold,
        )
        phase64a_expert_by_repeat[repeat, valid_indices] = expert_probability
        candidate_logit = (
            PHASE64A_CONFIG["anchor_logit_weight"]
            * phase64a_anchor_logit[valid_indices]
            + PHASE64A_CONFIG["diffusion_logit_weight"]
            * phase64a_logit(expert_probability)
        )
        phase64a_candidate_by_repeat[repeat, valid_indices] = (
            phase64a_sigmoid(candidate_logit)
        )
        phase64a_fit_records.append({
            "repeat": int(repeat),
            "fold": int(fold),
            **diagnostic,
        })
        print(
            f"Phase64 diffusion fit {len(phase64a_fit_records)}/"
            f"{phase64a_repeat_count * 3}: repeat={repeat}, fold={fold}",
            flush=True,
        )

assert np.all(np.isfinite(phase64a_expert_by_repeat))
assert np.all(np.isfinite(phase64a_candidate_by_repeat))
assert np.all(
    (phase64a_candidate_by_repeat > 0.0)
    & (phase64a_candidate_by_repeat < 1.0)
)

phase64a_repeat_records = []
phase64a_repeat_report_rows = []
for repeat in range(phase64a_repeat_count):
    expert_metrics = phase64a_metrics(
        phase64a_labels, phase64a_expert_by_repeat[repeat]
    )
    comparison = phase64a_compare(
        phase64a_labels,
        phase64a_anchor,
        phase64a_candidate_by_repeat[repeat],
    )
    phase64a_repeat_records.append({
        "repeat": int(repeat),
        "diffusion_expert": expert_metrics,
        "eligible_blend": comparison,
    })
    phase64a_repeat_report_rows.append({
        "repeat": int(repeat),
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    })

phase64a_mean_expert = np.mean(phase64a_expert_by_repeat, axis=0)
phase64a_mean_candidate = np.mean(phase64a_candidate_by_repeat, axis=0)
phase64a_anchor_metrics = phase64a_metrics(phase64a_labels, phase64a_anchor)
phase64a_expert_metrics = phase64a_metrics(
    phase64a_labels, phase64a_mean_expert
)
phase64a_pooled = phase64a_compare(
    phase64a_labels, phase64a_anchor, phase64a_mean_candidate
)

phase64a_major_mask = np.isin(
    phase64a_groups, PHASE64A_CONFIG["major_groups"]
)
phase64a_major_combined = phase64a_compare(
    phase64a_labels[phase64a_major_mask],
    phase64a_anchor[phase64a_major_mask],
    phase64a_mean_candidate[phase64a_major_mask],
)
phase64a_major_rows = []
phase64a_major_regrets = []
for group in PHASE64A_CONFIG["major_groups"]:
    mask = phase64a_groups == group
    comparison = phase64a_compare(
        phase64a_labels[mask],
        phase64a_anchor[mask],
        phase64a_mean_candidate[mask],
    )
    phase64a_major_rows.append({
        "group": int(group),
        **comparison,
    })
    phase64a_major_regrets.append(max(0.0, -comparison["log_loss_gain"]))

phase64a_domain_rows = []
phase64a_domain_gains = []
for domain, definition in enumerate(phase64a_domain_definitions):
    mask = phase64a_domain_id == domain
    assert int(np.sum(mask)) > 0
    assert set(np.unique(phase64a_groups[mask]).tolist()) == set(
        int(value) for value in definition["groups"]
    )
    comparison = phase64a_compare(
        phase64a_labels[mask],
        phase64a_anchor[mask],
        phase64a_mean_candidate[mask],
    )
    phase64a_domain_rows.append({
        "domain": str(definition["name"]),
        "groups": [int(value) for value in definition["groups"]],
        "n": int(np.sum(mask)),
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    })
    phase64a_domain_gains.append(comparison["log_loss_gain"])

phase64a_domain_gains = np.asarray(phase64a_domain_gains, dtype=np.float64)
phase64a_domain_rows_sorted = sorted(
    phase64a_domain_rows, key=lambda row: row["log_loss_gain"]
)
phase64a_domain_extremes = {
    "worst_three": phase64a_domain_rows_sorted[:3],
    "best_three": phase64a_domain_rows_sorted[-3:][::-1],
}
phase64a_bootstrap_rng = np.random.default_rng(
    PHASE64A_CONFIG["domain_bootstrap_seed"]
)
phase64a_bootstrap = np.mean(
    phase64a_domain_gains[phase64a_bootstrap_rng.integers(
        0,
        phase64a_domain_gains.size,
        size=(
            PHASE64A_CONFIG["domain_bootstrap_replicates"],
            phase64a_domain_gains.size,
        ),
    )],
    axis=1,
)
phase64a_bootstrap_q025, phase64a_bootstrap_median, phase64a_bootstrap_q975 = (
    np.quantile(phase64a_bootstrap, [0.025, 0.50, 0.975]).tolist()
)

phase64a_repeat_gains = np.asarray([
    record["eligible_blend"]["log_loss_gain"]
    for record in phase64a_repeat_records
], dtype=np.float64)
phase64a_repeat_auc_gains = np.asarray([
    record["eligible_blend"]["auroc_gain"]
    for record in phase64a_repeat_records
], dtype=np.float64)
phase64a_gate = PHASE64A_CONFIG["advancement_gate"]
phase64a_max_repeat_regret = float(max(0.0, -np.min(phase64a_repeat_gains)))
phase64a_max_major_regret = float(max(phase64a_major_regrets))
phase64a_advanced = bool(
    phase64a_pooled["log_loss_gain"]
    >= phase64a_gate["minimum_averaged_oof_log_loss_gain"]
    and phase64a_pooled["auroc_gain"]
    >= phase64a_gate["minimum_averaged_oof_auroc_gain"]
    and int(np.sum(phase64a_repeat_gains > 0.0))
    >= phase64a_gate["minimum_repeat_log_loss_wins"]
    and int(np.sum(phase64a_repeat_auc_gains > 0.0))
    >= phase64a_gate["minimum_repeat_auroc_wins"]
    and phase64a_max_repeat_regret
    <= phase64a_gate["maximum_repeat_log_loss_regret"]
    and phase64a_major_combined["log_loss_gain"]
    >= phase64a_gate["minimum_major_groups_combined_log_loss_gain"]
    and phase64a_max_major_regret
    <= phase64a_gate["maximum_individual_major_group_log_loss_regret"]
    and phase64a_bootstrap_q025
    >= phase64a_gate["minimum_domain_bootstrap_lower_95_log_loss_gain"]
    and phase64a_pooled["brier_gain"]
    >= -phase64a_gate["brier_regret_allowed"]
)

phase64a_status = (
    "repeated_group_blocked_gate_passed_ready_for_phase64b_deployment_refit"
    if phase64a_advanced
    else "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
)

phase64a_fit_summary = {
    "fit_count": int(len(phase64a_fit_records)),
    "train_n_range": [
        int(min(row["train_n"] for row in phase64a_fit_records)),
        int(max(row["train_n"] for row in phase64a_fit_records)),
    ],
    "valid_n_range": [
        int(min(row["valid_n"] for row in phase64a_fit_records)),
        int(max(row["valid_n"] for row in phase64a_fit_records)),
    ],
    "train_group_count_range": [
        int(min(row["train_group_count"] for row in phase64a_fit_records)),
        int(max(row["train_group_count"] for row in phase64a_fit_records)),
    ],
    "valid_group_count_range": [
        int(min(row["valid_group_count"] for row in phase64a_fit_records)),
        int(max(row["valid_group_count"] for row in phase64a_fit_records)),
    ],
    "prior_logit_correction_range": [
        float(min(row["prior_correction"] for row in phase64a_fit_records)),
        float(max(row["prior_correction"] for row in phase64a_fit_records)),
    ],
}

phase64a_contract_core = {
    "schema_version": PHASE64A_CONFIG["schema_version"],
    "status": phase64a_status,
    "source_contract_sha256": str(phase64a_validation["contract_sha256"]),
    "candidate": (
        "0.80_anchor_logit_plus_0.20_group_blocked_diffusion_nystrom_logit"
    ),
    "single_predeclared_candidate": True,
    "outer_images_used_for_fitting": False,
    "outer_labels_used_for_fitting": False,
    "test_data_read": False,
}
phase64a_contract_sha256 = hashlib.sha256(json.dumps(
    phase64a_contract_core,
    sort_keys=True,
    separators=(",", ":"),
).encode("utf-8")).hexdigest()

phase64a_contract_path = Path(PHASE64A_CONFIG["contract_file"])
phase64a_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase64a_temporary_path = phase64a_contract_path.with_suffix(
    phase64a_contract_path.suffix + ".tmp"
)
phase64a_temporary_path.write_text(json.dumps({
    **phase64a_contract_core,
    "contract_sha256": phase64a_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
    "contains_embeddings": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase64a_temporary_path, phase64a_contract_path)

PHASE64A_DIFFUSION_REPORT_PRIVATE = None
PHASE64A_DIFFUSION_STATE_PRIVATE = {
    "contract_sha256": phase64a_contract_sha256,
    "status": phase64a_status,
    "mean_diffusion_expert_oof_probability": phase64a_mean_expert.copy(),
    "mean_candidate_oof_probability": phase64a_mean_candidate.copy(),
}

phase64a_report = {
    "phase": "phase64_group_blocked_diffusion_nystrom_gate",
    "status": phase64a_status,
    "candidate": {
        "single_predeclared_candidate": True,
        "formula": (
            "sigmoid(0.80*anchor_logit+0.20*diffusion_expert_logit)"
        ),
        "numeric_advanced": phase64a_advanced,
    },
    "representation": {
        "case_level_positive_quantile_normalization": True,
        "case_level_log_compression": True,
        "bilateral_symmetric_and_absolute_asymmetric_channels": True,
        "pooled_shape": list(PHASE64A_CONFIG["pooled_shape"]),
        "dct_keep_shape": list(PHASE64A_CONFIG["dct_keep_shape"]),
        "feature_count": phase64a_feature_count,
        "fold_local_group_weighted_pca_dimension": int(
            PHASE64A_CONFIG["pca_dimension"]
        ),
        "self_tuning_neighbor_rank": int(
            PHASE64A_CONFIG["diffusion_neighbor_rank"]
        ),
        "diffusion_alpha": float(PHASE64A_CONFIG["diffusion_alpha"]),
        "diffusion_dimension": int(PHASE64A_CONFIG["diffusion_dimension"]),
        "nystrom_out_of_sample_extension": True,
    },
    "validation": {
        "repeat_count": int(phase64a_repeat_count),
        "fold_count_per_repeat": 3,
        "fit_count": int(len(phase64a_fit_records)),
        "whole_acquisition_groups_held_out": True,
        "outer_images_used_for_fitting": False,
        "outer_labels_used_for_fitting": False,
        "original_three_fold_oof_used_for_selection": False,
        "historically_pristine_holdout_claimed": False,
    },
    "anchor": phase64a_anchor_metrics,
    "averaged_diffusion_expert_alone": phase64a_expert_metrics,
    "averaged_eligible_candidate": phase64a_pooled,
    "repeats": phase64a_repeat_report_rows,
    "major_groups": {
        "combined": phase64a_major_combined,
        "individual": phase64a_major_rows,
        "maximum_log_loss_regret": phase64a_max_major_regret,
    },
    "stress_domain_extremes": phase64a_domain_extremes,
    "domain_stability": {
        "domain_macro_log_loss_gain": float(np.mean(phase64a_domain_gains)),
        "domain_win_count": int(np.sum(phase64a_domain_gains > 0.0)),
        "domain_count": int(phase64a_domain_gains.size),
        "bootstrap_replicates": int(
            PHASE64A_CONFIG["domain_bootstrap_replicates"]
        ),
        "bootstrap_lower_95_log_loss_gain": float(phase64a_bootstrap_q025),
        "bootstrap_median_log_loss_gain": float(phase64a_bootstrap_median),
        "bootstrap_upper_95_log_loss_gain": float(phase64a_bootstrap_q975),
    },
    "gate_summary": {
        "repeat_log_loss_wins": int(np.sum(phase64a_repeat_gains > 0.0)),
        "repeat_auroc_wins": int(np.sum(phase64a_repeat_auc_gains > 0.0)),
        "maximum_repeat_log_loss_regret": phase64a_max_repeat_regret,
        "maximum_major_group_log_loss_regret": phase64a_max_major_regret,
        "domain_bootstrap_lower_95_log_loss_gain": float(
            phase64a_bootstrap_q025
        ),
        "thresholds": dict(phase64a_gate),
    },
    "fit_diagnostics": phase64a_fit_summary,
    "interpretation_contract": {
        "phase63_router_family_remains_rejected": True,
        "candidate_hyperparameters_frozen_before_outer_predictions": True,
        "no_candidate_grid_or_outer_label_selection": True,
        "test_case_inference_uses_training_manifold_only": True,
        "test_cases_processed_independently": True,
        "no_test_retraining_or_adaptation": True,
        "passing_result_requires_phase64b_full_training_refit": True,
    },
    "training_performed": True,
    "training_voxel_cache_read": True,
    "training_nifti_files_read": False,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "models_exported": False,
    "synthetic_test_mode": phase64a_is_synthetic,
    "contract_sha256": phase64a_contract_sha256,
    "contract_file": str(phase64a_contract_path),
    "elapsed_seconds": float(time.perf_counter() - phase64a_started),
}

PHASE64A_DIFFUSION_REPORT_PRIVATE = phase64a_report

print("BEGIN SANITIZED_PHASE64_GROUP_BLOCKED_DIFFUSION_NYSTROM_GATE")
print(json.dumps(phase64a_report, indent=2, sort_keys=False))
print("END SANITIZED_PHASE64_GROUP_BLOCKED_DIFFUSION_NYSTROM_GATE")
