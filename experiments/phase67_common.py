
"""Phase67 shared restore, metric, bootstrap and contract helpers.

Phases 57-66 were preserved as standalone notebook cells that each re-declared
the same restore and metric code. Phase67 runs five coordinated cells instead of
one, so the shared contract lives here once and every cell imports it by path:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import phase67_common as common

"Standalone" keeps its established meaning: no earlier notebook cell and no live
in-memory variable is required. Authorized private artifacts, installed NumPy
and (for the training cell) PyTorch are still required.

Nothing here reads test data, writes case-level values, or mutates the accepted
Phase56 archive.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import zipfile
from pathlib import Path

import numpy as np

PHASE67_COMMON_VERSION = "phase67_common_v1"

# Frozen development constants. A mismatch is a stop, never a silent repair.
CASE_COUNT = 1362
GROUP_COUNT = 15
NORMAL_COUNT = 615
PATHOLOGIC_COUNT = 747
EXPECTED_GROUP_SIZES = [39, 456, 145, 255, 76, 7, 49, 208, 32, 4, 35, 10, 32, 9, 5]
EXPECTED_ORIGINAL_FOLD_SIZES = [467, 443, 452]
EXPECTED_PHASE56_SHA = (
    "d012af37307f27a8f92bf95eecc271b7a2153ede6231480670f2e769f4d6c78c"
)
ANCHOR_LOG_LOSS = 0.29016628416289664
ANCHOR_AUROC = 0.9464742438589044
ANCHOR_MEAN_PROBABILITY = 0.5500570231688339
PHASE52_REGENERATION_SEEDS = [520101, 520102, 520103, 520104, 520105]
PROBABILITY_CLIP = 1.0e-7
INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N = 30
MAJOR_GROUPS = [1, 3]

# Accepted upstream statuses. Phase67 preserves the recorded chain rather than
# asserting its own conclusions about earlier phases.
ACCEPTED_PHASE65AR4_STATUS = (
    "accepted_corrected_offline_deployment_integrity_numeric_parity_not_"
    "available_ready_for_corrected_phase65b",
    "accepted_corrected_offline_and_exact_phase56_numeric_parity_ready_for_"
    "corrected_phase65b",
)
ACCEPTED_PHASE62_STATUS = "accepted_validation_reset_ready_for_phase62b"
ACCEPTED_PHASE64_STATUS = (
    "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
)
ACCEPTED_PHASE65A_STATUS = (
    "fallback_numeric_gate_failed_retain_phase12c_unknown_fallback"
)


class Phase67Stop(RuntimeError):
    """Sanitized stop. Carries a stage code only, never a path or a value."""


def require(condition, stage):
    if not bool(condition):
        raise Phase67Stop(str(stage))


# ---------------------------------------------------------------------------
# Digests, JSON and contract self-hashes
# ---------------------------------------------------------------------------

def sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(str(value.shape).encode("utf-8"))
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def read_json(path, stage):
    path = Path(path)
    require(path.is_file(), f"missing_{stage}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise Phase67Stop(f"invalid_json_{stage}") from exc
    require(isinstance(value, dict), f"non_object_{stage}")
    return value


def contract_core_hash_valid(payload):
    require(isinstance(payload, dict), "invalid_contract_payload")
    claimed = payload.get("contract_sha256")
    core = {
        key: value
        for key, value in payload.items()
        if key != "contract_sha256" and not key.startswith("contains_")
    }
    return isinstance(claimed, str) and claimed == canonical_hash(core)


def report_hash_valid(payload):
    require(isinstance(payload, dict), "invalid_report_payload")
    claimed = payload.get("contract_sha256")
    core = dict(payload)
    core.pop("contract_sha256", None)
    core.pop("contract_file", None)
    recomputed = hashlib.sha256(
        json.dumps(
            core, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        ).encode("utf-8")
    ).hexdigest()
    return isinstance(claimed, str) and claimed == recomputed


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_npz(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def emit_contract(path, core):
    """Write a self-hashed contract. The contains_* flags are appended after
    hashing, which is exactly the set contract_core_hash_valid strips."""
    payload = {
        **core,
        "contract_sha256": canonical_hash(core),
        "contains_labels": False,
        "contains_probabilities": False,
        "contains_case_indices": False,
        "contains_voxel_data": False,
        "contains_embeddings": False,
        "contains_uids": False,
    }
    atomic_json(path, payload)
    return payload["contract_sha256"]


def print_sanitized(tag, report):
    print(f"BEGIN SANITIZED_{tag}")
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"END SANITIZED_{tag}")


# ---------------------------------------------------------------------------
# Metrics. Hand-rolled float64 to match every earlier phase bit for bit.
# ---------------------------------------------------------------------------

def clip_probability(probability):
    return np.clip(
        np.asarray(probability, dtype=np.float64).reshape(-1),
        PROBABILITY_CLIP,
        1.0 - PROBABILITY_CLIP,
    )


def logit(probability):
    probability = clip_probability(probability)
    return np.log(probability) - np.log1p(-probability)


def sigmoid(value):
    value = np.asarray(value, dtype=np.float64)
    output = np.empty_like(value)
    positive = value >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    exponential = np.exp(value[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output


def auc(labels, score):
    """Mann-Whitney U with mid-rank tie handling. Returns None on a
    single-class slice, so small stress domains cannot raise."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    require(labels.shape == score.shape, "metric_shape_mismatch")
    positive_n = int(np.sum(labels == 1))
    negative_n = int(labels.size - positive_n)
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
        rank[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    statistic = float(np.sum(rank[labels == 1]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))


def loss_vector(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = clip_probability(probability)
    require(labels.shape == probability.shape, "metric_shape_mismatch")
    return -(
        labels * np.log(probability) + (1 - labels) * np.log1p(-probability)
    )


def metrics(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = clip_probability(probability)
    return {
        "n": int(labels.size),
        "log_loss": float(np.mean(loss_vector(labels, probability))),
        "auroc": auc(labels, probability),
        "brier": float(np.mean(np.square(probability - labels))),
        "mean_probability": float(np.mean(probability)),
    }


def compare(labels, baseline, candidate):
    """Positive gain always means the candidate is better, matching every
    earlier phase's *_compare contract."""
    baseline_metrics = metrics(labels, baseline)
    candidate_metrics = metrics(labels, candidate)
    auroc_gain = None
    if (
        baseline_metrics["auroc"] is not None
        and candidate_metrics["auroc"] is not None
    ):
        auroc_gain = float(candidate_metrics["auroc"] - baseline_metrics["auroc"])
    return {
        "n": int(np.asarray(labels).size),
        "baseline_log_loss": baseline_metrics["log_loss"],
        "candidate_log_loss": candidate_metrics["log_loss"],
        "log_loss_gain": float(
            baseline_metrics["log_loss"] - candidate_metrics["log_loss"]
        ),
        "baseline_auroc": baseline_metrics["auroc"],
        "candidate_auroc": candidate_metrics["auroc"],
        "auroc_gain": auroc_gain,
        "baseline_brier": baseline_metrics["brier"],
        "candidate_brier": candidate_metrics["brier"],
        "brier_gain": float(baseline_metrics["brier"] - candidate_metrics["brier"]),
    }


def regret(gain):
    """Regret is a non-negative loss. phase62_cell_162b omitted this clamp and
    reported negative 'regrets'; Phase67 does not repeat that."""
    return float(max(0.0, -float(gain)))


def expected_calibration_error(labels, probability, bins=15):
    labels = np.asarray(labels, dtype=np.float64).reshape(-1)
    probability = clip_probability(probability)
    edges = np.linspace(0.0, 1.0, int(bins) + 1)
    index = np.clip(np.digitize(probability, edges[1:-1], right=False), 0, bins - 1)
    total = 0.0
    maximum_gap = 0.0
    occupied = 0
    for bucket in range(int(bins)):
        mask = index == bucket
        count = int(np.sum(mask))
        if count == 0:
            continue
        occupied += 1
        gap = abs(float(np.mean(probability[mask])) - float(np.mean(labels[mask])))
        maximum_gap = max(maximum_gap, gap)
        total += gap * count / labels.size
    return {
        "ece": float(total),
        "maximum_bin_gap": float(maximum_gap),
        "occupied_bins": int(occupied),
    }


# ---------------------------------------------------------------------------
# Stress domains and the two bootstraps
# ---------------------------------------------------------------------------

def build_domain_definitions(groups, minimum_n=INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N):
    """Label-blind: every group with n>=minimum_n is its own stress domain and
    all smaller groups collapse into one pooled domain. For the frozen group
    sizes this yields 10 individual domains plus one pooled domain = 11."""
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    sizes = np.bincount(groups, minlength=GROUP_COUNT)
    individual = [
        int(group)
        for group, count in enumerate(sizes)
        if int(count) >= int(minimum_n)
    ]
    tiny = [group for group in range(GROUP_COUNT) if group not in individual]
    require(
        set(individual + tiny) == set(range(GROUP_COUNT)),
        "domain_partition_incomplete",
    )
    require(
        set(MAJOR_GROUPS).issubset(individual),
        "major_group_not_individual_domain",
    )
    definitions = [
        {"name": f"group_{group}", "groups": [group]} for group in individual
    ]
    definitions.append({"name": "tiny_groups_pooled", "groups": tiny})
    return definitions


def assign_domain_ids(groups, definitions):
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    domain_id = np.full(groups.size, -1, dtype=np.int64)
    for index, definition in enumerate(definitions):
        domain_id[np.isin(groups, definition["groups"])] = index
    require(np.all(domain_id >= 0), "domain_assignment_incomplete")
    return domain_id


def per_domain_gains(labels, domain_id, baseline, candidate, expected_domains=None):
    """Mean log-loss gain per stress domain.

    Only the domains actually present in domain_id are used, because inner
    group-held selection folds legitimately exclude whole domains. Pass
    expected_domains to require complete coverage, which is what the outer
    adjudication does.
    """
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    domain_id = np.asarray(domain_id, dtype=np.int64).reshape(-1)
    gain = loss_vector(labels, baseline) - loss_vector(labels, candidate)
    present = np.unique(domain_id)
    if expected_domains is not None:
        require(
            np.array_equal(present, np.arange(int(expected_domains))),
            "domain_coverage_incomplete",
        )
    return np.asarray(
        [float(np.mean(gain[domain_id == domain])) for domain in present],
        dtype=np.float64,
    )


def macro_domain_bootstrap(domain_gain, replicates, seed):
    """Secondary instrument, preserved unchanged from phase65a_domain_bootstrap.

    Resamples the 11 stress domains with replacement and takes quantiles of the
    UNWEIGHTED mean, so it weights a 32-case domain like a 456-case one. That is
    intentional as a robustness probe, but it is not the estimand the
    competition scores, so Phase67 reports it beside the case-weighted bound.
    """
    domain_gain = np.asarray(domain_gain, dtype=np.float64).reshape(-1)
    require(domain_gain.size > 0, "empty_domain_gain_vector")
    rng = np.random.default_rng(int(seed))
    index = rng.integers(
        0, domain_gain.size, size=(int(replicates), domain_gain.size)
    )
    replicate = np.mean(domain_gain[index], axis=1)
    lower, median, upper = np.quantile(replicate, [0.025, 0.50, 0.975])
    return {
        "estimator": "unweighted_mean_of_per_domain_mean_log_loss_gain",
        "units": int(domain_gain.size),
        "replicates": int(replicates),
        "seed": int(seed),
        "point": float(np.mean(domain_gain)),
        "standard_deviation_across_domains": (
            float(np.std(domain_gain, ddof=1)) if domain_gain.size > 1 else 0.0
        ),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "fraction_positive": float(np.mean(replicate > 0.0)),
    }


def group_cluster_bootstrap(labels, groups, baseline, candidate, replicates, seed):
    """Primary instrument. Resamples whole acquisition groups with replacement
    and CONCATENATES their case indices, so group size and multiplicity are
    respected. The statistic is therefore the case-weighted pooled log-loss
    gain, which is what the competition metric actually measures. Ported from
    phase61_cell_161b_phase31_fixed_blend_stability_gate.py:298."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    baseline_loss = loss_vector(labels, baseline)
    candidate_loss = loss_vector(labels, candidate)
    group_indices = [np.flatnonzero(groups == group) for group in range(GROUP_COUNT)]
    require(all(item.size > 0 for item in group_indices), "empty_acquisition_group")
    rng = np.random.default_rng(int(seed))
    replicate = np.empty(int(replicates), dtype=np.float64)
    for draw in range(int(replicates)):
        sampled = rng.integers(0, GROUP_COUNT, size=GROUP_COUNT)
        indices = np.concatenate([group_indices[int(item)] for item in sampled])
        replicate[draw] = float(
            np.mean(baseline_loss[indices] - candidate_loss[indices])
        )
    lower, median, upper = np.quantile(replicate, [0.025, 0.50, 0.975])
    return {
        "estimator": "case_weighted_pooled_log_loss_gain_over_resampled_groups",
        "units": int(GROUP_COUNT),
        "replicates": int(replicates),
        "seed": int(seed),
        "point": float(np.mean(baseline_loss - candidate_loss)),
        "lower_95": float(lower),
        "median": float(median),
        "upper_95": float(upper),
        "fraction_positive": float(np.mean(replicate > 0.0)),
    }


def stability_rows(labels, groups, original_fold, domain_id, baseline, candidate):
    """Per-fold, per-group, per-domain and major-group slices in one call."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    original_fold = np.asarray(original_fold, dtype=np.int64).reshape(-1)
    domain_id = np.asarray(domain_id, dtype=np.int64).reshape(-1)

    fold_rows = []
    for fold in range(3):
        mask = original_fold == fold
        row = compare(labels[mask], baseline[mask], candidate[mask])
        row["fold"] = int(fold)
        fold_rows.append(row)

    group_rows = []
    for group in range(GROUP_COUNT):
        mask = groups == group
        row = compare(labels[mask], baseline[mask], candidate[mask])
        row["group"] = int(group)
        group_rows.append(row)

    domain_rows = []
    for domain in np.unique(domain_id):
        mask = domain_id == domain
        row = compare(labels[mask], baseline[mask], candidate[mask])
        row["domain"] = int(domain)
        domain_rows.append(row)

    major_mask = np.isin(groups, MAJOR_GROUPS)
    major_combined = compare(
        labels[major_mask], baseline[major_mask], candidate[major_mask]
    )

    return {
        "pooled": compare(labels, baseline, candidate),
        "folds": fold_rows,
        "groups": group_rows,
        "domains": sorted(domain_rows, key=lambda item: item["log_loss_gain"]),
        "major_groups_combined": major_combined,
        "maximum_fold_log_loss_regret": max(
            regret(row["log_loss_gain"]) for row in fold_rows
        ),
        "maximum_major_group_log_loss_regret": max(
            regret(row["log_loss_gain"])
            for row in group_rows
            if row["group"] in MAJOR_GROUPS
        ),
        "maximum_powered_domain_log_loss_regret": max(
            regret(row["log_loss_gain"])
            for row in domain_rows
            if row["n"] >= INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N
        ),
        "improved_group_fraction": float(
            np.mean([row["log_loss_gain"] > 0.0 for row in group_rows])
        ),
        "improved_domain_fraction": float(
            np.mean([row["log_loss_gain"] > 0.0 for row in domain_rows])
        ),
    }


# ---------------------------------------------------------------------------
# Private artifact restore
# ---------------------------------------------------------------------------

def load_vector(description, candidates, expected_n=CASE_COUNT):
    existing = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen and candidate.is_file():
            seen.add(key)
            existing.append(candidate)
    require(bool(existing), f"missing_vector_{description}")
    reference = np.asarray(
        np.load(existing[0], allow_pickle=False), dtype=np.float64
    ).reshape(-1)
    require(reference.shape == (expected_n,), f"shape_vector_{description}")
    require(np.all(np.isfinite(reference)), f"finite_vector_{description}")
    for duplicate_path in existing[1:]:
        duplicate = np.asarray(
            np.load(duplicate_path, allow_pickle=False), dtype=np.float64
        ).reshape(-1)
        require(
            duplicate.shape == reference.shape
            and float(np.max(np.abs(duplicate - reference))) <= 1.0e-12,
            f"duplicate_vector_mismatch_{description}",
        )
    return reference


def optional_vector(description, candidates, expected_n=CASE_COUNT):
    """Same checks, but a missing file returns None instead of stopping. Used
    for diagnostic-only historical vectors whose provenance is unverified."""
    if not any(Path(candidate).is_file() for candidate in candidates):
        return None
    return load_vector(description, candidates, expected_n=expected_n)


def restore_labels(root, labels_root=None):
    root = Path(root)
    candidates = []
    seen = set()
    private_root = labels_root or os.environ.get("DAT_PRIVATE_ROOT") or ""
    if private_root:
        candidate = Path(private_root) / "train_labels.csv"
        if candidate.is_file():
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
        if len(rows) != CASE_COUNT:
            continue
        uids = [str(row["uid"]) for row in rows]
        if len(set(uids)) != CASE_COUNT or any(not uid for uid in uids):
            continue
        try:
            raw = np.asarray([float(row["is_pathologic"]) for row in rows])
        except Exception:
            continue
        if not np.all(np.isin(raw, [0.0, 1.0])):
            continue
        records.append((uids, raw.astype(np.int64)))
    require(bool(records), "training_labels_not_resolved")
    reference_uids, reference_labels = records[0]
    for duplicate_uids, duplicate_labels in records[1:]:
        require(
            duplicate_uids == reference_uids,
            "training_label_duplicate_uid_order_mismatch",
        )
        require(
            np.array_equal(duplicate_labels, reference_labels),
            "training_label_duplicate_value_mismatch",
        )
    require(
        int(np.sum(reference_labels == 0)) == NORMAL_COUNT
        and int(np.sum(reference_labels == 1)) == PATHOLOGIC_COUNT,
        "training_label_count_mismatch",
    )
    return reference_labels


def read_archive_json(root, member, stage):
    with zipfile.ZipFile(Path(root) / "phase56_submission.zip") as archive:
        value = json.loads(archive.read(member))
    require(isinstance(value, dict), f"archive_json_type_{stage}")
    return value


def read_archive_npz(root, member):
    """Read an npz member out of the accepted archive without extracting it."""
    with zipfile.ZipFile(Path(root) / "phase56_submission.zip") as archive:
        payload = archive.read(member)
    return np.load(io.BytesIO(payload), allow_pickle=False)


def read_phase42_gate(root):
    return read_archive_json(
        root, "phase42_assets/phase42_hierarchical_gate.json", "phase42_gate"
    )


def read_router(root):
    """Full acquisition router, not only fold_for_group. Stage 0 needs the
    prototypes and the standardization to reason about protocol coverage."""
    with read_archive_npz(root, "models/phase30_acquisition_router.npz") as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def read_router_fold(root):
    return np.asarray(
        read_router(root)["fold_for_group"], dtype=np.int64
    ).reshape(-1)


def restore_production_state(artifact_root, labels_root=None, require_chain=True):
    """Restore labels, groups, original folds, the exact Phase43 anchor and the
    stress-domain assignment from private artifacts.

    Ported from phase65b_restore_production_state (identical arithmetic and
    identical parity assertions). The upstream chain-status guards are kept by
    default because Phase67 must not silently proceed on a mutated record; pass
    require_chain=False only in a synthetic test.
    """
    root = Path(artifact_root)
    provenance = {}

    ar4 = read_json(
        root / "phase65ar4_corrected_offline_gate_adjudication_contract.json",
        "phase65ar4_contract",
    )
    require(report_hash_valid(ar4), "phase65ar4_hash_mismatch")
    if require_chain:
        require(
            ar4.get("status") in ACCEPTED_PHASE65AR4_STATUS,
            "phase65ar4_not_accepted",
        )
        require(
            ar4.get("gates", {}).get(
                "corrected_offline_deployment_integrity_passed"
            )
            is True,
            "phase65ar4_offline_gate_not_passed",
        )

    submission_zip = root / "phase56_submission.zip"
    require(submission_zip.is_file(), "missing_phase56_submission")
    submission_sha = sha_file(submission_zip)
    require(
        ar4.get("artifact_provenance", {}).get("submission_zip", {}).get("sha256")
        == submission_sha,
        "phase56_digest_changed_after_phase65ar4",
    )
    if require_chain:
        require(
            submission_sha == EXPECTED_PHASE56_SHA,
            "phase56_digest_not_the_accepted_archive",
        )

    phase62 = read_json(
        root / "phase62_acquisition_domain_validation_contract.json",
        "phase62_contract",
    )
    require(contract_core_hash_valid(phase62), "phase62_contract_hash_mismatch")
    phase64 = read_json(
        root / "phase64_group_blocked_diffusion_nystrom_contract.json",
        "phase64_contract",
    )
    require(contract_core_hash_valid(phase64), "phase64_contract_hash_mismatch")
    phase65a = read_json(
        root / "phase65a_corrected_deployment_fallback_contract.json",
        "phase65a_contract",
    )
    require(contract_core_hash_valid(phase65a), "phase65a_contract_hash_mismatch")
    if require_chain:
        require(
            phase62.get("status") == ACCEPTED_PHASE62_STATUS,
            "phase62_validation_not_accepted",
        )
        require(
            phase64.get("status") == ACCEPTED_PHASE64_STATUS,
            "phase64_rejection_not_preserved",
        )
        require(
            phase65a.get("status") == ACCEPTED_PHASE65A_STATUS,
            "phase65a_fallback_decision_not_preserved",
        )

    labels = restore_labels(root, labels_root=labels_root)

    phase50_path = root / "phase50_private_validation/phase50_partitions.npz"
    phase52_path = root / "phase52_repeated_partition.npz"
    phase52_contract = read_json(
        root / "phase52_support_gate_contract.json", "phase52_contract"
    )
    require(phase50_path.is_file(), "missing_phase50_partition")
    require(phase52_path.is_file(), "missing_phase52_partition")
    with np.load(phase50_path, allow_pickle=False) as partition50:
        groups = np.asarray(
            partition50["acquisition_group"], dtype=np.int64
        ).reshape(-1)
    with np.load(phase52_path, allow_pickle=False) as partition52:
        fold_assignment = np.asarray(partition52["fold_assignment"], dtype=np.int64)
        phase52_sha = str(np.asarray(partition52["contract_sha256"]).item())
    require(groups.shape == (CASE_COUNT,), "phase50_group_shape_mismatch")
    require(
        np.bincount(groups, minlength=GROUP_COUNT).tolist() == EXPECTED_GROUP_SIZES,
        "phase50_group_counts_mismatch",
    )
    require(fold_assignment.shape == (5, CASE_COUNT), "phase52_fold_shape_mismatch")
    require(
        phase52_sha == str(phase52_contract.get("contract_sha256")),
        "phase52_contract_link_mismatch",
    )
    require(
        hashlib.sha256(
            np.ascontiguousarray(fold_assignment.astype(np.int8)).tobytes()
        ).hexdigest()
        == str(phase52_contract.get("phase52_partition_sha256")),
        "phase52_partition_digest_mismatch",
    )

    # Regenerating the Phase52 partition from its seeds is the row-order proof
    # that links the label CSV order to the voxel cache order.
    regenerated = np.full((5, CASE_COUNT), -1, dtype=np.int64)
    for repeat, seed in enumerate(PHASE52_REGENERATION_SEEDS):
        rng = np.random.default_rng(seed)
        for group in range(GROUP_COUNT):
            for label in (0, 1):
                indices = np.flatnonzero(
                    (groups == group) & (labels == label)
                ).copy()
                if not indices.size:
                    continue
                rng.shuffle(indices)
                cycle = (
                    np.arange(indices.size, dtype=np.int64)
                    + int(rng.integers(0, 5))
                ) % 5
                regenerated[repeat, indices] = cycle
    require(
        np.array_equal(regenerated, fold_assignment),
        "training_label_cache_row_order_mismatch",
    )

    phase12 = load_vector(
        "phase12c_oof", [root / "phase32_phase12c_oof_float64.npy"]
    )
    phase33 = load_vector(
        "phase33_component_oof",
        [
            root / "phase33_private_checkpoint/phase33_component_oof_float64.npy",
            root / "phase33_component_oof_float64.npy",
        ],
    )
    combined39 = load_vector(
        "phase39_combined_residual",
        [
            root / "phase39_private_checkpoint/phase39_combined_residual_float64.npy",
            root / "phase39_combined_residual_float64.npy",
        ],
    )
    phase39 = load_vector(
        "phase39_oof",
        [
            root / "phase39_private_checkpoint/phase39_oof_float64.npy",
            root / "phase39_oof_float64.npy",
        ],
    )

    phase12_logit = logit(phase12)
    phase33_residual = logit(phase33) - phase12_logit
    phase39_reconstructed = sigmoid(phase12_logit + np.clip(combined39, -2.0, 2.0))
    require(
        float(np.max(np.abs(phase39_reconstructed - phase39))) <= 2.0e-12,
        "phase39_reconstruction_mismatch",
    )

    gate42 = read_phase42_gate(root)
    require(
        float(gate42.get("phase36_raw_residual_weight")) == 0.75
        and float(gate42.get("phase33_logit_residual_weight")) == 0.125
        and float(gate42.get("residual_cap")) == 1.0,
        "phase42_gate_constants_mismatch",
    )
    alpha_map = {
        int(group): float(alpha)
        for group, alpha in gate42["known_group_alpha"].items()
    }
    require(set(alpha_map) == set(range(GROUP_COUNT)), "phase42_alpha_map_mismatch")
    group_alpha = np.asarray([alpha_map[int(group)] for group in groups])

    # Phase39 combined residual is 0.75*r36 + 0.25*r33; Phase42 uses
    # 0.75*r36 + 0.125*r33, hence the 0.125*r33 subtraction. The reconstruction
    # cap is 1.0. The separate 0.5 residual cap in the ledger is an unrelated
    # Phase57 model-head hyperparameter and must not be substituted here.
    phase42_uncapped = combined39 - 0.125 * phase33_residual
    anchor = sigmoid(
        phase12_logit + group_alpha * np.clip(phase42_uncapped, -1.0, 1.0)
    )
    anchor_metrics = metrics(labels, anchor)
    require(
        abs(anchor_metrics["log_loss"] - ANCHOR_LOG_LOSS) <= 2.0e-10
        and abs(anchor_metrics["auroc"] - ANCHOR_AUROC) <= 2.0e-12
        and abs(anchor_metrics["mean_probability"] - ANCHOR_MEAN_PROBABILITY)
        <= 2.0e-10,
        "phase43_anchor_metric_parity_mismatch",
    )

    logo_path = root / "phase57_logo_partition.npz"
    phase57_contract = read_json(
        root / "phase57_transport_contract.json", "phase57_transport_contract"
    )
    require(logo_path.is_file(), "missing_phase57_logo_partition")
    with np.load(logo_path, allow_pickle=False) as logo:
        fold_for_group = np.asarray(
            logo["original_fold_for_group"], dtype=np.int64
        ).reshape(-1)
        logo_contract_sha = str(np.asarray(logo["contract_sha256"]).item())
    require(
        contract_core_hash_valid(phase57_contract),
        "phase57_transport_contract_hash_mismatch",
    )
    require(
        logo_contract_sha == phase57_contract.get("contract_sha256"),
        "phase57_logo_contract_link_mismatch",
    )
    router_fold = read_router_fold(root)
    require(
        fold_for_group.shape == (GROUP_COUNT,)
        and np.array_equal(fold_for_group, router_fold),
        "phase57_router_fold_mismatch",
    )
    original_fold = fold_for_group[groups]
    require(
        np.bincount(original_fold, minlength=3).tolist()
        == EXPECTED_ORIGINAL_FOLD_SIZES,
        "phase57_original_fold_counts_mismatch",
    )
    for group in range(GROUP_COUNT):
        require(
            np.unique(original_fold[groups == group]).size == 1,
            "phase57_group_split_leakage",
        )

    definitions = build_domain_definitions(groups)
    require(
        definitions == phase62.get("stress_domain_definitions"),
        "phase62_stress_domain_definition_mismatch",
    )
    domain_id = assign_domain_ids(groups, definitions)

    cache_path = root / "phase31_highres_float16.npy"
    require(cache_path.is_file(), "missing_phase31_highres_cache")
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    require(
        cache.shape == (CASE_COUNT, 80, 80, 80) and cache.dtype == np.float16,
        "phase31_highres_cache_contract_mismatch",
    )
    del cache

    provenance.update(
        {
            "phase56_submission_sha256": submission_sha,
            "phase62_contract_sha256": str(phase62["contract_sha256"]),
            "phase64_contract_sha256": str(phase64["contract_sha256"]),
            "phase65a_contract_sha256": str(phase65a["contract_sha256"]),
            "phase65ar4_contract_sha256": str(ar4["contract_sha256"]),
            "phase57_transport_contract_sha256": str(
                phase57_contract["contract_sha256"]
            ),
            "row_order_verified": True,
            "router_fold_verified": True,
            "cache_contract_verified": True,
            "anchor_parity_verified": True,
        }
    )

    return {
        "labels": labels.copy(),
        "groups": groups.copy(),
        "original_fold": original_fold.copy(),
        "anchor_probability": anchor.copy(),
        "stress_domain_id": domain_id.copy(),
        "stress_domain_definitions": json.loads(json.dumps(definitions)),
        "phase52_fold_assignment": fold_assignment.copy(),
        "phase12c_probability": phase12.copy(),
        "phase33_component_probability": phase33.copy(),
        "phase39_probability": phase39.copy(),
        "group_alpha": group_alpha.copy(),
        "anchor_metrics": anchor_metrics,
        "highres_cache_file": str(cache_path),
        "phase62_contract": phase62,
        "phase65a_contract": phase65a,
        "provenance": provenance,
        "contract_sha256": str(phase62["contract_sha256"]),
    }


def restore_repeat_partition(artifact_root, groups, phase62_contract=None):
    """The 5 x 3 whole-acquisition-group repeated partitions, re-derived from the
    frozen Phase62 contract so no live kernel state is needed. Diagnostic only:
    repeated partitions reuse the same patients."""
    root = Path(artifact_root)
    if phase62_contract is None:
        phase62_contract = read_json(
            root / "phase62_acquisition_domain_validation_contract.json",
            "phase62_repeat_contract",
        )
    require(
        contract_core_hash_valid(phase62_contract),
        "phase62_repeat_contract_hash_mismatch",
    )
    group_to_fold = np.asarray(
        phase62_contract["group_blocked_repeat_group_to_fold"], dtype=np.int64
    )
    require(group_to_fold.shape == (5, GROUP_COUNT), "phase62_repeat_shape_mismatch")
    groups = np.asarray(groups, dtype=np.int64).reshape(-1)
    case_fold = group_to_fold[:, groups]
    for repeat in range(group_to_fold.shape[0]):
        require(
            np.array_equal(np.unique(case_fold[repeat]), np.arange(3)),
            "phase62_repeat_fold_incomplete",
        )
        for group in range(GROUP_COUNT):
            require(
                np.unique(case_fold[repeat, groups == group]).size == 1,
                "phase62_repeat_group_leakage",
            )
    return group_to_fold, case_fold
