from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import numpy as np
from sklearn.cluster import KMeans
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler


# Cell 162B — fold-local multi-template anatomy and uncertainty expert.
#
# Run after Cell 162A in the same live kernel. This is the one candidate frozen
# by Phase62A. It reads only the accepted 80^3 training cache. All templates,
# scalers, classifiers, calibration parameters, novelty thresholds, and router
# quantities are fit within each outer-training partition. Each fold's model
# excludes that fold's labels; final evaluation starts only after all three OOF
# prediction blocks have been frozen.

phase62b_started = time.perf_counter()

assert isinstance(globals().get("PHASE62_VALIDATION_RESET_REPORT_PRIVATE"), dict)
assert PHASE62_VALIDATION_RESET_REPORT_PRIVATE["status"] == (
    "accepted_validation_reset_ready_for_phase62b"
)
assert isinstance(globals().get("PHASE62_VALIDATION_RESET_STATE_PRIVATE"), dict)
assert isinstance(globals().get("PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE"), dict)

phase62b_validation = PHASE62_VALIDATION_RESET_STATE_PRIVATE
phase62b_engine = PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE
phase62b_labels = np.asarray(phase62b_validation["labels"], dtype=np.int64).reshape(-1)
phase62b_groups = np.asarray(phase62b_validation["groups"], dtype=np.int64).reshape(-1)
phase62b_folds = np.asarray(
    phase62b_validation["original_fold"], dtype=np.int64
).reshape(-1)
phase62b_anchor = np.asarray(
    phase62b_validation["anchor_probability"], dtype=np.float64
).reshape(-1)
phase62b_domain_id = np.asarray(
    phase62b_validation["stress_domain_id"], dtype=np.int64
).reshape(-1)
phase62b_domain_definitions = phase62b_validation["stress_domain_definitions"]

phase62b_synthetic_override = globals().get(
    "PHASE62B_SYNTHETIC_TEST_OVERRIDE_PRIVATE"
)
phase62b_expected_n = (
    int(phase62b_synthetic_override["case_count"])
    if isinstance(phase62b_synthetic_override, dict)
    else 1362
)

assert phase62b_labels.shape == phase62b_groups.shape == phase62b_folds.shape
assert phase62b_labels.shape == phase62b_anchor.shape == phase62b_domain_id.shape
assert phase62b_labels.shape == (phase62b_expected_n,)
assert set(np.unique(phase62b_labels).tolist()) == {0, 1}
assert np.array_equal(np.unique(phase62b_groups), np.arange(15))
assert np.array_equal(np.unique(phase62b_folds), np.arange(3))
assert np.all(np.isfinite(phase62b_anchor))
assert np.all((phase62b_anchor > 0.0) & (phase62b_anchor < 1.0))


PHASE62B_CONFIG = {
    "schema_version": "phase62_multitemplate_anatomy_uncertainty_gate_v1",
    "seed": 620262,
    "probability_clip": 1.0e-7,
    "input_shape": [80, 80, 80],
    "left_right_axis": 0,
    "low_resolution_shape": [20, 20, 20],
    "template_shape": [10, 10, 10],
    "templates_per_class": 3,
    "inner_fold_count": 3,
    "major_groups": [1, 3],
    "confidence_threshold": 0.80,
    "feature_contract": {
        "per_case_robust_normalization": True,
        "bilateral_mirrored_statistics": True,
        "posterior_anterior_profiles": True,
        "fixed_threshold_shape_statistics": True,
        "fold_local_class_template_kmeans": True,
        "template_sample_weighting": "equal_acquisition_group_mass",
    },
    "standard_head": {
        "logistic_c": 0.08,
        "hist_learning_rate": 0.04,
        "hist_iterations": 160,
        "hist_max_leaf_nodes": 7,
        "hist_min_samples_leaf": 25,
        "hist_l2": 8.0,
        "logit_ensemble": [0.50, 0.50],
    },
    "sensitivity_positive_weight": 1.75,
    "specificity_negative_weight": 1.75,
    "platt_c": 0.50,
    "router": {
        "minimum_logit_conflict": 0.50,
        "conflict_width": 2.00,
        "uncertainty_width": 0.35,
        "anchor_confidence_start": 1.00,
        "anchor_confidence_width": 2.00,
        "maximum_anatomy_alpha": 0.45,
        "maximum_uncertainty_shrink": 0.25,
        "maximum_anatomy_residual": 1.50,
    },
    "domain_bootstrap_replicates": 10000,
    "domain_bootstrap_seed": 620263,
    "advancement_gate": dict(
        PHASE62_VALIDATION_RESET_CONFIG_PRIVATE["advancement_gate"]
    ),
    "contract_file": (
        "/kaggle/working/phase62_multitemplate_anatomy_uncertainty_contract.json"
    ),
}


def phase62b_clip(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64),
        PHASE62B_CONFIG["probability_clip"],
        1.0 - PHASE62B_CONFIG["probability_clip"],
    )


def phase62b_logit(probability):
    probability = phase62b_clip(probability)
    return np.log(probability) - np.log1p(-probability)


def phase62b_sigmoid(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def phase62b_auc(labels, score):
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


def phase62b_case_loss(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase62b_clip(probability)
    return -(labels * np.log(probability) + (1 - labels) * np.log1p(-probability))


def phase62b_masked_mean(value, mask):
    value = np.asarray(value, dtype=np.float64)
    mask = np.asarray(mask, dtype=bool)
    return None if int(np.sum(mask)) == 0 else float(np.mean(value[mask]))


def phase62b_metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64)
    probability = phase62b_clip(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(phase62b_case_loss(labels, probability))),
        "auroc": phase62b_auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
    }


def phase62b_compare(labels, anchor, candidate):
    anchor_metrics = phase62b_metrics(labels, anchor)
    candidate_metrics = phase62b_metrics(labels, candidate)
    return {
        "n": int(np.asarray(labels).size),
        "anchor_log_loss": anchor_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": anchor_metrics["log_loss"] - candidate_metrics["log_loss"],
        "anchor_auroc": anchor_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": (
            None
            if anchor_metrics["auroc"] is None or candidate_metrics["auroc"] is None
            else candidate_metrics["auroc"] - anchor_metrics["auroc"]
        ),
        "brier_gain": anchor_metrics["brier"] - candidate_metrics["brier"],
    }


def phase62b_group_balanced_weight(groups):
    groups = np.asarray(groups, dtype=np.int64)
    unique, counts = np.unique(groups, return_counts=True)
    lookup = {int(group): float(len(groups) / (len(unique) * count))
              for group, count in zip(unique, counts)}
    weight = np.asarray([lookup[int(group)] for group in groups], dtype=np.float64)
    return weight / np.mean(weight)


def phase62b_weighted_summary(array):
    array = np.asarray(array, dtype=np.float32)
    flat = array.reshape(-1)
    quantiles = np.quantile(flat, [0.50, 0.75, 0.90, 0.95, 0.975, 0.99, 0.995])
    features = [float(value) for value in quantiles]
    count = flat.size
    for fraction in [0.001, 0.005, 0.01, 0.02, 0.05, 0.10]:
        top_n = max(1, int(np.ceil(count * fraction)))
        threshold_index = count - top_n
        top_values = np.partition(flat, threshold_index)[threshold_index:]
        features.append(float(np.mean(top_values)))
    positive = np.maximum(array, 0.0)
    mass = float(np.sum(positive)) + 1.0e-8
    coordinates = np.meshgrid(*[
        np.linspace(-1.0, 1.0, size, dtype=np.float32)
        for size in array.shape
    ], indexing="ij")
    for coordinate in coordinates:
        center = float(np.sum(positive * coordinate) / mass)
        variance = float(np.sum(positive * np.square(coordinate - center)) / mass)
        features.extend([center, float(np.sqrt(max(variance, 0.0)))])
    features.extend([float(np.mean(positive)), float(np.std(positive)), float(mass / count)])
    return features


def phase62b_profile_features(array, axis):
    reduce_axes = tuple(index for index in range(3) if index != axis)
    profile = np.sum(np.maximum(array, 0.0), axis=reduce_axes).astype(np.float64)
    total = float(np.sum(profile)) + 1.0e-12
    probability = profile / total
    coordinate = np.linspace(-1.0, 1.0, profile.size)
    center = float(np.sum(probability * coordinate))
    variance = float(np.sum(probability * np.square(coordinate - center)))
    skew = float(np.sum(probability * np.power(coordinate - center, 3)))
    entropy = float(-np.sum(probability * np.log(np.clip(probability, 1.0e-12, None))))
    peak = float(coordinate[int(np.argmax(profile))])
    posterior = float(np.sum(probability[: profile.size // 2]))
    return [center, np.sqrt(max(variance, 0.0)), skew, entropy, peak, posterior]


def phase62b_extract_case(volume):
    value = np.asarray(volume, dtype=np.float32)
    value = np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
    value = np.maximum(value, 0.0)
    scale = float(np.quantile(value, 0.995))
    if not np.isfinite(scale) or scale <= 1.0e-7:
        scale = 1.0
    value = np.clip(value / scale, 0.0, 1.5)
    low = value.reshape(20, 4, 20, 4, 20, 4).mean(axis=(1, 3, 5))
    central = low[2:18, 2:18, 2:18]
    left = central[:8]
    right = central[8:][::-1]

    features = []
    features.extend(phase62b_weighted_summary(central))
    features.extend(phase62b_weighted_summary(left))
    features.extend(phase62b_weighted_summary(right))
    for axis in range(3):
        features.extend(phase62b_profile_features(central, axis))
    for half in [left, right]:
        for start, stop in [(0, 5), (5, 11), (11, 16)]:
            segment = half[:, start:stop, :]
            segment_flat = segment.reshape(-1)
            top_n = max(1, int(np.ceil(0.05 * segment_flat.size)))
            top = np.partition(segment_flat, segment_flat.size - top_n)[-top_n:]
            features.extend([
                float(np.mean(segment)),
                float(np.mean(top)),
                float(np.sum(segment) / (np.sum(half) + 1.0e-8)),
            ])
    mirrored_difference = np.abs(left - right)
    features.extend([
        float(np.mean(mirrored_difference)),
        float(np.quantile(mirrored_difference, 0.90)),
        float(np.quantile(mirrored_difference, 0.99)),
        float(np.sqrt(np.mean(np.square(mirrored_difference)))),
        float(np.corrcoef(left.reshape(-1), right.reshape(-1))[0, 1]),
    ])
    maximum = float(np.max(central)) + 1.0e-8
    for threshold in [0.25, 0.40, 0.55, 0.70, 0.85]:
        mask = central >= threshold * maximum
        left_occupancy = float(np.mean(mask[:8]))
        right_occupancy = float(np.mean(mask[8:]))
        features.extend([
            float(np.mean(mask)),
            left_occupancy,
            right_occupancy,
            abs(left_occupancy - right_occupancy),
        ])

    template = low.reshape(10, 2, 10, 2, 10, 2).mean(axis=(1, 3, 5))
    template = np.maximum(template - np.median(template), 0.0)
    template_scale = float(np.quantile(template, 0.99))
    if template_scale > 1.0e-7:
        template = np.clip(template / template_scale, 0.0, 1.5)
    template = template.reshape(-1).astype(np.float32)
    template /= float(np.linalg.norm(template) + 1.0e-8)
    feature = np.nan_to_num(
        np.asarray(features, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
    )
    return feature, template


def phase62b_extract_all(cache):
    static_rows = []
    template_rows = []
    for case_index in range(phase62b_expected_n):
        feature, template = phase62b_extract_case(cache[case_index])
        static_rows.append(feature)
        template_rows.append(template)
    static = np.stack(static_rows).astype(np.float32)
    template = np.stack(template_rows).astype(np.float32)
    assert np.all(np.isfinite(static)) and np.all(np.isfinite(template))
    return static, template


def phase62b_fit_templates(template_base, labels, groups, seed):
    centers = []
    template_weight = phase62b_group_balanced_weight(groups)
    for label in [0, 1]:
        mask = labels == label
        assert int(np.sum(mask)) >= PHASE62B_CONFIG["templates_per_class"]
        model = KMeans(
            n_clusters=PHASE62B_CONFIG["templates_per_class"],
            n_init=10,
            random_state=int(seed + label),
        )
        model.fit(template_base[mask], sample_weight=template_weight[mask])
        class_centers = np.asarray(model.cluster_centers_, dtype=np.float32)
        class_centers /= np.linalg.norm(class_centers, axis=1, keepdims=True) + 1.0e-8
        centers.append(class_centers)
    return np.concatenate(centers, axis=0)


def phase62b_template_features(template_base, centers):
    cosine = np.clip(np.asarray(template_base) @ centers.T, -1.0, 1.0)
    distance = np.sqrt(np.maximum(2.0 - 2.0 * cosine, 0.0))
    per_class = PHASE62B_CONFIG["templates_per_class"]
    normal_distance = np.min(distance[:, :per_class], axis=1)
    pathologic_distance = np.min(distance[:, per_class:], axis=1)
    normal_similarity = np.max(cosine[:, :per_class], axis=1)
    pathologic_similarity = np.max(cosine[:, per_class:], axis=1)
    summary = np.column_stack([
        cosine,
        distance,
        normal_distance,
        pathologic_distance,
        normal_distance - pathologic_distance,
        normal_similarity,
        pathologic_similarity,
        pathologic_similarity - normal_similarity,
        np.mean(distance[:, :per_class], axis=1),
        np.mean(distance[:, per_class:], axis=1),
    ])
    novelty = np.min(distance, axis=1)
    return np.asarray(summary, dtype=np.float32), np.asarray(novelty, dtype=np.float64)


def phase62b_fit_expert(features, labels, groups, seed):
    group_weight = phase62b_group_balanced_weight(groups)
    scaler = StandardScaler().fit(features)
    standardized = scaler.transform(features)
    logistic = LogisticRegression(
        C=PHASE62B_CONFIG["standard_head"]["logistic_c"],
        solver="lbfgs",
        max_iter=2000,
        random_state=int(seed),
    ).fit(standardized, labels, sample_weight=group_weight)

    def fit_hist(class_weight, offset):
        sample_weight = group_weight * np.where(
            labels == 1, class_weight[1], class_weight[0]
        )
        return HistGradientBoostingClassifier(
            loss="log_loss",
            learning_rate=PHASE62B_CONFIG["standard_head"]["hist_learning_rate"],
            max_iter=PHASE62B_CONFIG["standard_head"]["hist_iterations"],
            max_leaf_nodes=PHASE62B_CONFIG["standard_head"]["hist_max_leaf_nodes"],
            min_samples_leaf=PHASE62B_CONFIG["standard_head"]["hist_min_samples_leaf"],
            l2_regularization=PHASE62B_CONFIG["standard_head"]["hist_l2"],
            early_stopping=False,
            random_state=int(seed + offset),
        ).fit(standardized, labels, sample_weight=sample_weight)

    standard_hist = fit_hist((1.0, 1.0), 11)
    sensitivity_hist = fit_hist(
        (1.0, PHASE62B_CONFIG["sensitivity_positive_weight"]), 17
    )
    specificity_hist = fit_hist(
        (PHASE62B_CONFIG["specificity_negative_weight"], 1.0), 23
    )
    return {
        "scaler": scaler,
        "logistic": logistic,
        "standard_hist": standard_hist,
        "sensitivity_hist": sensitivity_hist,
        "specificity_hist": specificity_hist,
    }


def phase62b_predict_expert(expert, features):
    standardized = expert["scaler"].transform(features)
    logistic_probability = expert["logistic"].predict_proba(standardized)[:, 1]
    hist_probability = expert["standard_hist"].predict_proba(standardized)[:, 1]
    standard_logit = (
        PHASE62B_CONFIG["standard_head"]["logit_ensemble"][0]
        * phase62b_logit(logistic_probability)
        + PHASE62B_CONFIG["standard_head"]["logit_ensemble"][1]
        * phase62b_logit(hist_probability)
    )
    return {
        "standard": phase62b_sigmoid(standard_logit),
        "sensitivity": expert["sensitivity_hist"].predict_proba(standardized)[:, 1],
        "specificity": expert["specificity_hist"].predict_proba(standardized)[:, 1],
    }


def phase62b_fit_platt(raw_probability, labels, groups):
    feature = phase62b_logit(raw_probability).reshape(-1, 1)
    calibrator = LogisticRegression(
        C=PHASE62B_CONFIG["platt_c"],
        solver="lbfgs",
        max_iter=2000,
        random_state=PHASE62B_CONFIG["seed"] + 41,
    )
    calibrator.fit(
        feature, labels, sample_weight=phase62b_group_balanced_weight(groups)
    )
    return calibrator


def phase62b_apply_router(anchor, anatomy, sensitivity, specificity, novelty_scaled):
    router = PHASE62B_CONFIG["router"]
    anchor_logit = phase62b_logit(anchor)
    anatomy_logit = phase62b_logit(anatomy)
    conflict = np.clip(
        (np.abs(anatomy_logit - anchor_logit) - router["minimum_logit_conflict"])
        / router["conflict_width"],
        0.0,
        1.0,
    )
    uncertainty = np.clip(
        np.abs(sensitivity - specificity) / router["uncertainty_width"],
        0.0,
        1.0,
    )
    anchor_confidence = np.clip(
        (np.abs(anchor_logit) - router["anchor_confidence_start"])
        / router["anchor_confidence_width"],
        0.0,
        1.0,
    )
    anatomy_trust = (1.0 - uncertainty) * (1.0 - 0.50 * novelty_scaled)
    anatomy_alpha = router["maximum_anatomy_alpha"] * conflict * anatomy_trust
    uncertainty_shrink = (
        router["maximum_uncertainty_shrink"]
        * conflict * anchor_confidence * uncertainty
    )
    anatomy_residual = np.clip(
        anatomy_logit - anchor_logit,
        -router["maximum_anatomy_residual"],
        router["maximum_anatomy_residual"],
    )
    blended_logit = anchor_logit + anatomy_alpha * anatomy_residual
    final_logit = blended_logit * (1.0 - uncertainty_shrink)
    return phase62b_sigmoid(final_logit), {
        "anatomy_alpha": anatomy_alpha,
        "uncertainty_shrink": uncertainty_shrink,
        "uncertainty": uncertainty,
        "conflict": conflict,
    }


phase62b_cache_path = Path(phase62b_engine["highres_cache_file"])
assert phase62b_cache_path.is_file(), {"missing": str(phase62b_cache_path)}
phase62b_cache = np.load(phase62b_cache_path, mmap_mode="r")
assert phase62b_cache.shape == (phase62b_expected_n, 80, 80, 80)
assert phase62b_cache.dtype == np.float16

phase62b_static, phase62b_template_base = phase62b_extract_all(phase62b_cache)
assert phase62b_static.shape[0] == phase62b_expected_n
assert phase62b_template_base.shape == (phase62b_expected_n, 1000)

phase62b_anatomy_oof = np.full(phase62b_expected_n, np.nan, dtype=np.float64)
phase62b_sensitivity_oof = np.full(phase62b_expected_n, np.nan, dtype=np.float64)
phase62b_specificity_oof = np.full(phase62b_expected_n, np.nan, dtype=np.float64)
phase62b_candidate_oof = np.full(phase62b_expected_n, np.nan, dtype=np.float64)
phase62b_router_arrays = {
    name: np.full(phase62b_expected_n, np.nan, dtype=np.float64)
    for name in ["anatomy_alpha", "uncertainty_shrink", "uncertainty", "conflict"]
}
phase62b_fold_models = []
phase62b_training_records = []

for outer_fold in range(3):
    outer_train = np.flatnonzero(phase62b_folds != outer_fold)
    outer_valid = np.flatnonzero(phase62b_folds == outer_fold)
    train_labels = phase62b_labels[outer_train]
    train_groups = phase62b_groups[outer_train]

    inner_splitter = StratifiedGroupKFold(
        n_splits=PHASE62B_CONFIG["inner_fold_count"],
        shuffle=True,
        random_state=PHASE62B_CONFIG["seed"] + 100 * outer_fold,
    )
    inner_raw = np.full(outer_train.size, np.nan, dtype=np.float64)
    inner_split_count = 0
    for inner_train_local, inner_valid_local in inner_splitter.split(
        outer_train, train_labels, train_groups
    ):
        inner_split_count += 1
        assert not set(train_groups[inner_train_local]).intersection(
            set(train_groups[inner_valid_local])
        )
        template_state = phase62b_fit_templates(
            phase62b_template_base[outer_train[inner_train_local]],
            train_labels[inner_train_local],
            train_groups[inner_train_local],
            PHASE62B_CONFIG["seed"] + 1000 * outer_fold + inner_split_count,
        )
        train_template_feature, _ = phase62b_template_features(
            phase62b_template_base[outer_train[inner_train_local]], template_state
        )
        valid_template_feature, _ = phase62b_template_features(
            phase62b_template_base[outer_train[inner_valid_local]], template_state
        )
        inner_train_feature = np.concatenate([
            phase62b_static[outer_train[inner_train_local]], train_template_feature
        ], axis=1)
        inner_valid_feature = np.concatenate([
            phase62b_static[outer_train[inner_valid_local]], valid_template_feature
        ], axis=1)
        inner_expert = phase62b_fit_expert(
            inner_train_feature,
            train_labels[inner_train_local],
            train_groups[inner_train_local],
            PHASE62B_CONFIG["seed"] + 2000 * outer_fold + inner_split_count,
        )
        inner_raw[inner_valid_local] = phase62b_predict_expert(
            inner_expert, inner_valid_feature
        )["standard"]
    assert inner_split_count == PHASE62B_CONFIG["inner_fold_count"]
    assert np.all(np.isfinite(inner_raw))
    calibrator = phase62b_fit_platt(inner_raw, train_labels, train_groups)

    template_state = phase62b_fit_templates(
        phase62b_template_base[outer_train],
        train_labels,
        train_groups,
        PHASE62B_CONFIG["seed"] + 3000 + outer_fold,
    )
    train_template_feature, train_novelty = phase62b_template_features(
        phase62b_template_base[outer_train], template_state
    )
    valid_template_feature, valid_novelty = phase62b_template_features(
        phase62b_template_base[outer_valid], template_state
    )
    outer_train_feature = np.concatenate([
        phase62b_static[outer_train], train_template_feature
    ], axis=1)
    outer_valid_feature = np.concatenate([
        phase62b_static[outer_valid], valid_template_feature
    ], axis=1)
    expert = phase62b_fit_expert(
        outer_train_feature,
        train_labels,
        train_groups,
        PHASE62B_CONFIG["seed"] + 4000 + outer_fold,
    )
    prediction = phase62b_predict_expert(expert, outer_valid_feature)
    anatomy_probability = calibrator.predict_proba(
        phase62b_logit(prediction["standard"]).reshape(-1, 1)
    )[:, 1]
    novelty_q75, novelty_q95 = np.quantile(train_novelty, [0.75, 0.95])
    novelty_scaled = np.clip(
        (valid_novelty - novelty_q75) / max(novelty_q95 - novelty_q75, 1.0e-6),
        0.0,
        1.0,
    )
    candidate_probability, router_values = phase62b_apply_router(
        phase62b_anchor[outer_valid],
        anatomy_probability,
        prediction["sensitivity"],
        prediction["specificity"],
        novelty_scaled,
    )

    # Freeze this complete outer prediction block before any outer label slice.
    phase62b_anatomy_oof[outer_valid] = anatomy_probability
    phase62b_sensitivity_oof[outer_valid] = prediction["sensitivity"]
    phase62b_specificity_oof[outer_valid] = prediction["specificity"]
    phase62b_candidate_oof[outer_valid] = candidate_probability
    for name, value in router_values.items():
        phase62b_router_arrays[name][outer_valid] = value
    phase62b_fold_models.append({
        "outer_fold": outer_fold,
        "template_centers": template_state,
        "expert": expert,
        "calibrator": calibrator,
        "novelty_q75": float(novelty_q75),
        "novelty_q95": float(novelty_q95),
    })
    phase62b_training_records.append({
        "outer_fold": outer_fold,
        "train_n": int(outer_train.size),
        "valid_n": int(outer_valid.size),
        "train_group_count": int(np.unique(train_groups).size),
        "feature_count": int(outer_train_feature.shape[1]),
        "inner_fold_count": inner_split_count,
        "outer_labels_read_during_fit_or_prediction": False,
    })
    print(
        f"Phase62B outer fold {outer_fold + 1}/3 frozen: "
        f"train_n={outer_train.size}, valid_n={outer_valid.size}"
    )

assert np.all(np.isfinite(phase62b_anatomy_oof))
assert np.all(np.isfinite(phase62b_candidate_oof))
assert np.all((phase62b_candidate_oof > 0.0) & (phase62b_candidate_oof < 1.0))
assert all(np.all(np.isfinite(value)) for value in phase62b_router_arrays.values())


# Only now are outer labels used for evaluation.
phase62b_pooled = phase62b_compare(
    phase62b_labels, phase62b_anchor, phase62b_candidate_oof
)
phase62b_anatomy_metrics = phase62b_metrics(
    phase62b_labels, phase62b_anatomy_oof
)
phase62b_fold_records = []
for fold in range(3):
    mask = phase62b_folds == fold
    comparison = phase62b_compare(
        phase62b_labels[mask], phase62b_anchor[mask], phase62b_candidate_oof[mask]
    )
    phase62b_fold_records.append({
        "fold": fold,
        "n": comparison["n"],
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    })

phase62b_domain_records = []
phase62b_domain_gains = []
for domain_index, definition in enumerate(phase62b_domain_definitions):
    mask = phase62b_domain_id == domain_index
    comparison = phase62b_compare(
        phase62b_labels[mask], phase62b_anchor[mask], phase62b_candidate_oof[mask]
    )
    phase62b_domain_gains.append(comparison["log_loss_gain"])
    phase62b_domain_records.append({
        "domain": definition["name"],
        "groups": definition["groups"],
        "n": comparison["n"],
        "log_loss_gain": comparison["log_loss_gain"],
        "auroc_gain": comparison["auroc_gain"],
        "brier_gain": comparison["brier_gain"],
    })
phase62b_domain_gains = np.asarray(phase62b_domain_gains, dtype=np.float64)

phase62b_bootstrap_rng = np.random.default_rng(
    PHASE62B_CONFIG["domain_bootstrap_seed"]
)
phase62b_bootstrap_gain = np.empty(
    PHASE62B_CONFIG["domain_bootstrap_replicates"], dtype=np.float64
)
for replicate in range(PHASE62B_CONFIG["domain_bootstrap_replicates"]):
    sampled = phase62b_bootstrap_rng.integers(
        0, len(phase62b_domain_gains), size=len(phase62b_domain_gains)
    )
    phase62b_bootstrap_gain[replicate] = float(
        np.mean(phase62b_domain_gains[sampled])
    )

phase62b_major_mask = np.isin(phase62b_groups, PHASE62B_CONFIG["major_groups"])
phase62b_major_comparison = phase62b_compare(
    phase62b_labels[phase62b_major_mask],
    phase62b_anchor[phase62b_major_mask],
    phase62b_candidate_oof[phase62b_major_mask],
)
phase62b_individual_major_gains = {}
for group in PHASE62B_CONFIG["major_groups"]:
    mask = phase62b_groups == group
    phase62b_individual_major_gains[str(group)] = phase62b_compare(
        phase62b_labels[mask], phase62b_anchor[mask], phase62b_candidate_oof[mask]
    )["log_loss_gain"]

phase62b_anchor_predicted = (phase62b_anchor >= 0.5).astype(np.int64)
phase62b_confident = (
    (phase62b_anchor >= PHASE62B_CONFIG["confidence_threshold"])
    | (phase62b_anchor <= 1.0 - PHASE62B_CONFIG["confidence_threshold"])
)
phase62b_confident_error = phase62b_confident & (
    phase62b_anchor_predicted != phase62b_labels
)
phase62b_anchor_loss = phase62b_case_loss(phase62b_labels, phase62b_anchor)
phase62b_candidate_loss = phase62b_case_loss(
    phase62b_labels, phase62b_candidate_oof
)
phase62b_confidence_report = {
    "confident_error_count": int(np.sum(phase62b_confident_error)),
    "confident_error_mean_log_loss_gain": phase62b_masked_mean(
        phase62b_anchor_loss - phase62b_candidate_loss,
        phase62b_confident_error,
    ),
    "remaining_case_mean_log_loss_gain": phase62b_masked_mean(
        phase62b_anchor_loss - phase62b_candidate_loss,
        ~phase62b_confident_error,
    ),
    "mean_anatomy_alpha": float(np.mean(phase62b_router_arrays["anatomy_alpha"])),
    "maximum_anatomy_alpha": float(np.max(phase62b_router_arrays["anatomy_alpha"])),
    "mean_uncertainty_shrink": float(np.mean(
        phase62b_router_arrays["uncertainty_shrink"]
    )),
    "intervention_fraction": float(np.mean(
        (phase62b_router_arrays["anatomy_alpha"] > 1.0e-6)
        | (phase62b_router_arrays["uncertainty_shrink"] > 1.0e-6)
    )),
}

gate = PHASE62B_CONFIG["advancement_gate"]
maximum_fold_regret = float(max(
    -record["log_loss_gain"] for record in phase62b_fold_records
))
maximum_major_regret = float(max(
    -gain for gain in phase62b_individual_major_gains.values()
))
bootstrap_lower = float(np.quantile(phase62b_bootstrap_gain, 0.025))
numeric_advanced = bool(
    phase62b_pooled["log_loss_gain"] >= gate["minimum_pooled_log_loss_gain"]
    and phase62b_pooled["auroc_gain"] >= gate["minimum_pooled_auroc_gain"]
    and phase62b_major_comparison["log_loss_gain"]
    >= gate["minimum_major_groups_combined_log_loss_gain"]
    and maximum_major_regret
    <= gate["maximum_individual_major_group_log_loss_regret"]
    and maximum_fold_regret <= gate["maximum_original_fold_log_loss_regret"]
    and bootstrap_lower >= gate["minimum_domain_bootstrap_lower_95_log_loss_gain"]
    and phase62b_confidence_report["remaining_case_mean_log_loss_gain"]
    >= -gate["maximum_non_confident_case_log_loss_regret"]
    and phase62b_pooled["brier_gain"] >= -gate["brier_regret_allowed"]
)
phase62b_status = (
    "numeric_gate_passed_ready_for_runtime_reconstruction"
    if numeric_advanced
    else "validation_gate_failed_stop_multitemplate_candidate"
)

phase62b_contract_core = {
    "schema_version": PHASE62B_CONFIG["schema_version"],
    "status": phase62b_status,
    "feature_contract": PHASE62B_CONFIG["feature_contract"],
    "model_contract": PHASE62B_CONFIG["standard_head"],
    "router_contract": PHASE62B_CONFIG["router"],
    "advancement_gate": gate,
    "numeric_advanced": numeric_advanced,
    "each_outer_prediction_block_excludes_its_outer_labels": True,
    "test_data_used": False,
}
phase62b_contract_sha256 = hashlib.sha256(json.dumps(
    phase62b_contract_core, sort_keys=True, separators=(",", ":")
).encode("utf-8")).hexdigest()
phase62b_contract_path = Path(PHASE62B_CONFIG["contract_file"])
phase62b_contract_path.parent.mkdir(parents=True, exist_ok=True)
phase62b_temporary_path = phase62b_contract_path.with_suffix(
    phase62b_contract_path.suffix + ".tmp"
)
phase62b_temporary_path.write_text(json.dumps({
    **phase62b_contract_core,
    "contract_sha256": phase62b_contract_sha256,
    "contains_labels": False,
    "contains_probabilities": False,
    "contains_case_indices": False,
    "contains_voxel_data": False,
    "contains_embeddings": False,
}, indent=2, sort_keys=True) + "\n", encoding="utf-8")
os.replace(phase62b_temporary_path, phase62b_contract_path)

PHASE62B_MULTITEMPLATE_MODELS_PRIVATE = phase62b_fold_models
PHASE62B_MULTITEMPLATE_STATE_PRIVATE = {
    "contract_sha256": phase62b_contract_sha256,
    "status": phase62b_status,
    "anatomy_oof_probability": phase62b_anatomy_oof.copy(),
    "candidate_oof_probability": phase62b_candidate_oof.copy(),
    "sensitivity_oof_probability": phase62b_sensitivity_oof.copy(),
    "specificity_oof_probability": phase62b_specificity_oof.copy(),
}

phase62b_report = {
    "phase": "phase62_fold_local_multitemplate_anatomy_uncertainty_gate",
    "status": phase62b_status,
    "candidate": {
        "single_predeclared_candidate": True,
        "numeric_advanced": numeric_advanced,
        "formula": (
            "bounded_anchor_plus_trusted_anatomy_residual_then_"
            "sensitivity_specificity_uncertainty_shrink"
        ),
    },
    "features": {
        "static_feature_count": int(phase62b_static.shape[1]),
        "template_base_dimension": int(phase62b_template_base.shape[1]),
        "templates_per_class": PHASE62B_CONFIG["templates_per_class"],
        "training_only_fold_local_templates": True,
    },
    "diagnostic_anatomy_expert_alone": phase62b_anatomy_metrics,
    "eligible_routed_candidate": phase62b_pooled,
    "training_partitions": phase62b_training_records,
    "original_folds": phase62b_fold_records,
    "major_groups": {
        "combined_log_loss_gain": phase62b_major_comparison["log_loss_gain"],
        "combined_auroc_gain": phase62b_major_comparison["auroc_gain"],
        "individual_log_loss_gains": phase62b_individual_major_gains,
        "maximum_log_loss_regret": maximum_major_regret,
    },
    "stress_domains": phase62b_domain_records,
    "domain_stability": {
        "domain_macro_log_loss_gain": float(np.mean(phase62b_domain_gains)),
        "domain_win_count": int(np.sum(phase62b_domain_gains > 0.0)),
        "domain_count": int(len(phase62b_domain_gains)),
        "bootstrap_replicates": int(phase62b_bootstrap_gain.size),
        "bootstrap_lower_95_log_loss_gain": bootstrap_lower,
        "bootstrap_median_log_loss_gain": float(np.median(phase62b_bootstrap_gain)),
        "bootstrap_upper_95_log_loss_gain": float(
            np.quantile(phase62b_bootstrap_gain, 0.975)
        ),
    },
    "confidence_and_router": phase62b_confidence_report,
    "gate_summary": {
        "maximum_original_fold_log_loss_regret": maximum_fold_regret,
        "maximum_major_group_log_loss_regret": maximum_major_regret,
        "domain_bootstrap_lower_95_log_loss_gain": bootstrap_lower,
        "thresholds": gate,
    },
    "each_outer_prediction_block_excludes_its_outer_labels": True,
    "training_nifti_files_read": False,
    "training_voxel_cache_read": True,
    "test_data_read": False,
    "case_level_predictions_exported": False,
    "models_exported": False,
    "contract_sha256": phase62b_contract_sha256,
    "contract_file": str(phase62b_contract_path),
    "elapsed_seconds": round(time.perf_counter() - phase62b_started, 3),
}

PHASE62B_MULTITEMPLATE_REPORT_PRIVATE = phase62b_report

phase62b_serialized_report = json.dumps(phase62b_report, indent=2)
phase62b_output_lines = phase62b_serialized_report.splitlines()
assert len(phase62b_output_lines) + 5 <= 300
assert max(len(line) for line in phase62b_output_lines) <= 300

print("BEGIN SANITIZED_PHASE62_MULTITEMPLATE_ANATOMY_UNCERTAINTY_GATE")
print(phase62b_serialized_report)
print("END SANITIZED_PHASE62_MULTITEMPLATE_ANATOMY_UNCERTAINTY_GATE")
