"""Phase67A - deployment coverage and cache parity audit.

Two unresolved questions could each cost more leaderboard log loss than any new
model, and neither requires training:

  1. Router coverage. Phase56 routes a scan to one of 15 acquisition groups by
     comparing standardized header features against 137 prototypes at
     rounding_decimals=6. Anything it cannot place is an unknown protocol and
     falls back to Phase12c alone, whose development log loss is 0.307616
     against the anchor's 0.290166. If that lookup is exact rather than
     nearest-prototype, a scan from the same scanner with slightly different
     header rounding lands on the weaker branch. This cell measures how brittle
     the placement is instead of assuming either way.

  2. Unknown-protocol fallback. Phase65A recorded
     fallback_numeric_gate_failed_retain_phase12c_unknown_fallback, yet the
     pooled gain available from swapping Phase12c for the full anchor is about
     0.0175 - far above that gate's minimum_pooled_log_loss_gain of 0.005. A
     different criterion failed. This cell re-evaluates all ten criteria and
     names the ones that are false, then asks the question that actually
     matters for a branch which by construction never sees a known group.

It also carries the parity harness for the third blocking question:
phase31_highres_float16.npy is hash-bound but has never been shown to reproduce
from raw NIfTI. Every expert Phase67C and Phase67 Stage 3 would train is built
on that cache, so a failure here blocks them regardless of development score.
The recomputation itself needs the preprocessing implementation, which lives in
the private submission archive; supply it through DAT_PREPROCESS_CALLABLE and
DAT_NIFTI_ROOT and this cell will run and score the comparison. Without them it
reports the check as not performed and says exactly what is missing, rather
than reporting a pass it did not earn.

Read-only. No training, no test data, no case index, UID or path exported.
"""
from __future__ import annotations

import importlib
import json
import os
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402

PHASE67A_CONFIG = {
    "schema_version": "phase67_coverage_parity_audit_v1",
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get(
        "DAT_OUTPUT_ROOT", "/kaggle/working/phase67_private"
    ),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),
    # Optional parity inputs. Both must be present for the recomputation to run.
    "nifti_root": os.environ.get("DAT_NIFTI_ROOT", ""),
    # "package.module:function" taking a NIfTI path and returning an 80^3 array.
    "preprocess_callable": os.environ.get("DAT_PREPROCESS_CALLABLE", ""),
    "seed": 670101,
    # Relative jitter magnitudes applied to standardized header features. The
    # smallest is the size of a rounding step at six decimals; the larger ones
    # stand for genuine acquisition or reconstruction drift.
    "jitter_scales": [1.0e-6, 1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2],
    "jitter_replicates": 200,
    "parity_sample_per_group": 4,
    "parity_tolerance": 1.0e-3,
    "macro_bootstrap_replicates": 10000,
    "macro_bootstrap_seed": 660699,
    "cluster_bootstrap_replicates": 10000,
    "cluster_bootstrap_seed": 611161,
    # Reproduced verbatim from phase65_cell_165a so the re-evaluation is
    # comparable to the recorded decision rather than a new, looser gate.
    "phase65a_fallback_gate": {
        "minimum_pooled_log_loss_gain": 0.005,
        "minimum_pooled_auroc_gain": 0.002,
        "brier_regret_allowed": 0.0,
        "maximum_individual_group_log_loss_regret": 0.005,
        "maximum_individual_major_group_log_loss_regret": 0.0,
        "maximum_individual_domain_log_loss_regret": 0.005,
        "maximum_original_fold_log_loss_regret": 0.0,
        "minimum_improved_group_fraction": 0.75,
        "minimum_repeat_partition_log_loss_win_fraction": 0.85,
        "minimum_domain_bootstrap_lower_95_log_loss_gain": 0.0,
    },
    "contract_file": "phase67a_coverage_parity_audit_contract.json",
}

# Tokens whose presence in the archived routing sources distinguishes an exact
# table lookup from a nearest-prototype assignment. Only counts are reported;
# no archived source text is exported.
ROUTING_TOKENS = {
    "exact_lookup": ["round(", "np.round", "tobytes", "dict(", "lookup", "=="],
    "nearest_prototype": [
        "argmin", "cdist", "linalg.norm", "nearest", "distance", "kdtree"
    ],
    "threshold": ["threshold", "maximum_distance", "tolerance", "unknown"],
}


def describe_router(router):
    """Report the router's shape and content without exporting any values that
    could identify a scanner or an institution."""
    description = {
        "keys": sorted(router.keys()),
        "shapes": {key: list(np.asarray(value).shape) for key, value in router.items()},
        "dtypes": {key: str(np.asarray(value).dtype) for key, value in router.items()},
    }
    if "rounding_decimals" in router:
        description["rounding_decimals"] = int(
            np.asarray(router["rounding_decimals"]).reshape(-1)[0]
        )
    if "prototype_groups" in router:
        prototype_groups = np.asarray(
            router["prototype_groups"], dtype=np.int64
        ).reshape(-1)
        description["prototype_count"] = int(prototype_groups.size)
        description["prototypes_per_group"] = np.bincount(
            prototype_groups, minlength=common.GROUP_COUNT
        ).tolist()
        description["groups_with_a_single_prototype"] = [
            int(group)
            for group, count in enumerate(description["prototypes_per_group"])
            if int(count) == 1
        ]
    return description


def standardize(prototypes, router):
    """Apply the router's own standardization if it ships one."""
    values = np.asarray(prototypes, dtype=np.float64)
    if "feature_mean" in router and "feature_scale" in router:
        mean = np.asarray(router["feature_mean"], dtype=np.float64).reshape(-1)
        scale = np.asarray(router["feature_scale"], dtype=np.float64).reshape(-1)
        if mean.size == values.shape[1] and scale.size == values.shape[1]:
            safe = np.where(np.abs(scale) < 1.0e-12, 1.0, scale)
            return (values - mean) / safe, True
    return values, False


def prototype_separation(prototypes):
    """Nearest-neighbour distance between distinct prototypes.

    If prototypes sit far apart relative to plausible header drift, a
    nearest-prototype router is safe; if they are packed together, a distance
    ceiling matters. Either way the exact-match question is separate, and the
    jitter probe answers that one.
    """
    values = np.asarray(prototypes, dtype=np.float64)
    count = values.shape[0]
    if count < 2:
        return {"prototype_count": int(count), "computed": False}
    difference = values[:, None, :] - values[None, :, :]
    distance = np.sqrt(np.sum(np.square(difference), axis=2))
    np.fill_diagonal(distance, np.inf)
    nearest = np.min(distance, axis=1)
    return {
        "prototype_count": int(count),
        "computed": True,
        "nearest_neighbour_distance_minimum": float(np.min(nearest)),
        "nearest_neighbour_distance_median": float(np.median(nearest)),
        "nearest_neighbour_distance_maximum": float(np.max(nearest)),
        "exact_duplicate_prototype_pairs": int(
            np.sum(distance < 1.0e-12) // 2
        ),
    }


def jitter_probe(prototypes, prototype_groups, router, config, rng):
    """Perturb each prototype and ask whether it is still placed correctly.

    Two placement rules are scored side by side. Exact match is what a rounded
    table lookup does; nearest prototype is what a distance-based router would
    do. The gap between the two curves is the coverage a router change would
    recover, expressed as a fraction of protocols rather than of test cases,
    because the test protocol mix is unknown and must not be guessed.
    """
    values = np.asarray(prototypes, dtype=np.float64)
    groups = np.asarray(prototype_groups, dtype=np.int64).reshape(-1)
    decimals = int(np.asarray(router.get("rounding_decimals", 6)).reshape(-1)[0])
    rounded = np.round(values, decimals)
    table = {row.tobytes(): index for index, row in enumerate(rounded)}
    magnitude = float(np.median(np.abs(values))) or 1.0
    rows = []
    for scale in config["jitter_scales"]:
        exact_hits = 0
        nearest_group_hits = 0
        total = 0
        for _ in range(int(config["jitter_replicates"])):
            index = int(rng.integers(0, values.shape[0]))
            noise = rng.normal(0.0, float(scale) * magnitude, values.shape[1])
            probe = values[index] + noise
            total += 1
            if np.round(probe, decimals).tobytes() in table:
                exact_hits += 1
            distance = np.sqrt(np.sum(np.square(values - probe), axis=1))
            if int(groups[int(np.argmin(distance))]) == int(groups[index]):
                nearest_group_hits += 1
        rows.append(
            {
                "relative_jitter": float(scale),
                "replicates": int(total),
                "exact_match_placement_rate": float(exact_hits / max(total, 1)),
                "nearest_prototype_correct_group_rate": float(
                    nearest_group_hits / max(total, 1)
                ),
            }
        )
    return {
        "rounding_decimals": decimals,
        "rounded_prototypes_are_unique": bool(len(table) == values.shape[0]),
        "rows": rows,
        "interpretation": (
            "an_exact_match_rate_that_collapses_while_the_nearest_prototype_"
            "rate_stays_high_means_placement_is_brittle_and_a_distance_based_"
            "router_would_recover_coverage;_this_measures_protocols_not_test_"
            "cases_and_the_test_protocol_mix_is_unknown"
        ),
    }


def scan_routing_sources(artifact_root):
    """Count routing-relevant tokens in the archived Python sources.

    Static and conservative: token counts do not prove which branch executes,
    exactly as the Phase65A-R2 import-graph audit did not prove dynamic
    reachability. No archived source text leaves this cell.
    """
    archive_path = Path(artifact_root) / "phase56_submission.zip"
    if not archive_path.is_file():
        return {"available": False, "reason": "missing_phase56_submission"}
    counts = {family: 0 for family in ROUTING_TOKENS}
    inspected = 0
    router_referencing_files = 0
    with zipfile.ZipFile(archive_path) as archive:
        for name in archive.namelist():
            if not name.endswith(".py"):
                continue
            try:
                text = archive.read(name).decode("utf-8", errors="replace")
            except Exception:
                continue
            inspected += 1
            if "acquisition_router" not in text and "fold_for_group" not in text:
                continue
            router_referencing_files += 1
            for family, tokens in ROUTING_TOKENS.items():
                counts[family] += sum(text.count(token) for token in tokens)
    return {
        "available": True,
        "python_files_inspected": int(inspected),
        "router_referencing_files": int(router_referencing_files),
        "token_family_counts": counts,
        "limitation": (
            "token_counts_are_static_evidence_only_and_do_not_establish_which_"
            "branch_executes_at_inference_time"
        ),
    }


def fallback_reassessment(state, config):
    """Re-evaluate the Phase65A fallback gate and name the failing criteria.

    Then ask the question that criterion set does not: for a branch that only
    ever sees protocols absent from the router, is the full anchor better than
    Phase12c alone? A zero-regret requirement on major groups is the wrong test
    for that branch, because by construction it never handles them.
    """
    labels = state["labels"]
    groups = state["groups"]
    original_fold = state["original_fold"]
    domain_id = state["stress_domain_id"]
    anchor = state["anchor_probability"]
    phase12c = state["phase12c_probability"]
    gate = config["phase65a_fallback_gate"]

    stability = common.stability_rows(
        labels, groups, original_fold, domain_id, phase12c, anchor
    )
    pooled = stability["pooled"]
    domain_gains = common.per_domain_gains(
        labels, domain_id, phase12c, anchor,
        expected_domains=len(state["stress_domain_definitions"]),
    )
    macro = common.macro_domain_bootstrap(
        domain_gains,
        config["macro_bootstrap_replicates"],
        config["macro_bootstrap_seed"],
    )
    cluster = common.group_cluster_bootstrap(
        labels,
        groups,
        phase12c,
        anchor,
        config["cluster_bootstrap_replicates"],
        config["cluster_bootstrap_seed"],
    )

    repeat_fraction = None
    try:
        _, repeat_case_fold = common.restore_repeat_partition(
            state["artifact_root"], groups, phase62_contract=state["phase62_contract"]
        )
        wins = []
        for repeat in range(repeat_case_fold.shape[0]):
            for fold in range(3):
                mask = repeat_case_fold[repeat] == fold
                row = common.compare(labels[mask], phase12c[mask], anchor[mask])
                wins.append(row["log_loss_gain"] > 0.0)
        repeat_fraction = float(np.mean(wins))
    except common.Phase67Stop:
        repeat_fraction = None

    criteria = {
        "minimum_pooled_log_loss_gain": bool(
            pooled["log_loss_gain"] >= gate["minimum_pooled_log_loss_gain"]
        ),
        "minimum_pooled_auroc_gain": bool(
            pooled["auroc_gain"] is not None
            and pooled["auroc_gain"] >= gate["minimum_pooled_auroc_gain"]
        ),
        "brier_regret_allowed": bool(
            pooled["brier_gain"] >= -gate["brier_regret_allowed"]
        ),
        "maximum_individual_group_log_loss_regret": bool(
            max(common.regret(row["log_loss_gain"]) for row in stability["groups"])
            <= gate["maximum_individual_group_log_loss_regret"]
        ),
        "maximum_individual_major_group_log_loss_regret": bool(
            stability["maximum_major_group_log_loss_regret"]
            <= gate["maximum_individual_major_group_log_loss_regret"]
        ),
        "maximum_individual_domain_log_loss_regret": bool(
            max(common.regret(row["log_loss_gain"]) for row in stability["domains"])
            <= gate["maximum_individual_domain_log_loss_regret"]
        ),
        "maximum_original_fold_log_loss_regret": bool(
            stability["maximum_fold_log_loss_regret"]
            <= gate["maximum_original_fold_log_loss_regret"]
        ),
        "minimum_improved_group_fraction": bool(
            stability["improved_group_fraction"]
            >= gate["minimum_improved_group_fraction"]
        ),
        "minimum_repeat_partition_log_loss_win_fraction": bool(
            repeat_fraction is not None
            and repeat_fraction
            >= gate["minimum_repeat_partition_log_loss_win_fraction"]
        ),
        "minimum_domain_bootstrap_lower_95_log_loss_gain": bool(
            macro["lower_95"]
            >= gate["minimum_domain_bootstrap_lower_95_log_loss_gain"]
        ),
    }

    # The branch-appropriate question: treat each group in turn as if it were
    # an unknown protocol and ask which vector serves it better.
    per_group = []
    for group in range(common.GROUP_COUNT):
        mask = groups == group
        row = common.compare(labels[mask], phase12c[mask], anchor[mask])
        row["group"] = int(group)
        row["anchor_better"] = bool(row["log_loss_gain"] > 0.0)
        per_group.append(row)
    anchor_better_groups = [row["group"] for row in per_group if row["anchor_better"]]
    anchor_better_cases = int(
        sum(row["n"] for row in per_group if row["anchor_better"])
    )

    return {
        "comparison": "phase12c_only_baseline_versus_full_phase43_anchor",
        "pooled": pooled,
        "phase65a_gate": gate,
        "phase65a_criteria": criteria,
        "phase65a_failing_criteria": sorted(
            name for name, passed in criteria.items() if not passed
        ),
        "phase65a_gate_passes": bool(all(criteria.values())),
        "repeat_partition_log_loss_win_fraction": repeat_fraction,
        "macro_domain_bootstrap": macro,
        "case_weighted_group_cluster_bootstrap": cluster,
        "per_group_scored_as_if_unknown": per_group,
        "groups_where_anchor_beats_phase12c": anchor_better_groups,
        "group_count_where_anchor_wins": len(anchor_better_groups),
        "case_count_where_anchor_wins": anchor_better_cases,
        "interpretation": (
            "a_zero_regret_requirement_on_major_groups_is_not_the_right_test_"
            "for_a_branch_that_by_construction_never_handles_a_known_group;_"
            "the_per_group_scored_as_if_unknown_rows_are_the_branch_relevant_"
            "evidence_and_the_unknown_protocol_share_of_the_test_set_remains_"
            "unmeasured"
        ),
        "unknown_protocol_share_of_test_set_estimated": False,
    }


def cache_parity(state, config, rng):
    """Recompute the 80^3 cache from raw NIfTI for a stratified sample.

    Runs only when both the NIfTI root and a preprocessing callable are
    supplied, because the preprocessing implementation lives in the private
    archive and is not reconstructable from this repository. When it cannot
    run, this reports not_performed and names the missing input rather than
    letting an unmeasured field read as a pass.
    """
    cache_path = Path(state["highres_cache_file"])
    result = {
        "cache_sha256": common.sha_file(cache_path),
        "cache_contract_verified": True,
        "recomputation_performed": False,
    }
    nifti_root = str(config.get("nifti_root") or "").strip()
    callable_path = str(config.get("preprocess_callable") or "").strip()
    if not nifti_root or not callable_path:
        result["status"] = "not_performed"
        result["missing_inputs"] = [
            name
            for name, value in (
                ("DAT_NIFTI_ROOT", nifti_root),
                ("DAT_PREPROCESS_CALLABLE", callable_path),
            )
            if not value
        ]
        result["consequence"] = (
            "every_expert_trained_on_this_cache_remains_blocked_for_"
            "deployment_until_raw_nifti_equivalence_is_shown"
        )
        return result
    if ":" not in callable_path:
        result["status"] = "preprocess_callable_must_be_module_colon_function"
        return result
    module_name, function_name = callable_path.split(":", 1)
    try:
        module = importlib.import_module(module_name)
        preprocess = getattr(module, function_name)
    except Exception:
        result["status"] = "preprocess_callable_not_importable"
        return result

    root = Path(nifti_root)
    if not root.is_dir():
        result["status"] = "nifti_root_not_a_directory"
        return result

    groups = state["groups"]
    cache = np.load(cache_path, mmap_mode="r", allow_pickle=False)
    per_group = int(config["parity_sample_per_group"])
    selected = []
    for group in range(common.GROUP_COUNT):
        indices = np.flatnonzero(groups == group)
        take = min(per_group, indices.size)
        selected.extend(
            rng.choice(indices, size=take, replace=False).tolist()
        )

    deviations = []
    failures = 0
    for index in selected:
        try:
            recomputed = np.asarray(preprocess(int(index)), dtype=np.float32)
        except Exception:
            failures += 1
            continue
        if recomputed.shape != (80, 80, 80):
            failures += 1
            continue
        stored = np.asarray(cache[int(index)], dtype=np.float32)
        deviations.append(float(np.max(np.abs(recomputed - stored))))
    del cache

    result["recomputation_performed"] = True
    result["sampled_case_count"] = len(selected)
    result["compared_case_count"] = len(deviations)
    result["failed_case_count"] = int(failures)
    if deviations:
        result["maximum_absolute_deviation"] = float(np.max(deviations))
        result["median_absolute_deviation"] = float(np.median(deviations))
        result["within_tolerance"] = bool(
            float(np.max(deviations)) <= float(config["parity_tolerance"])
        )
        result["status"] = (
            "parity_within_tolerance"
            if result["within_tolerance"]
            else "parity_deviation_exceeds_tolerance"
        )
    else:
        result["status"] = "no_case_could_be_recomputed"
    return result


def main():
    started = time.perf_counter()
    config = PHASE67A_CONFIG
    artifact_root = Path(config["artifact_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(config["seed"])

    state = common.restore_production_state(
        artifact_root, labels_root=config["labels_root"] or None
    )
    state["artifact_root"] = artifact_root

    router = common.read_router(artifact_root)
    router_description = describe_router(router)
    routing_sources = scan_routing_sources(artifact_root)

    coverage = {"router": router_description, "static_source_scan": routing_sources}
    if "prototypes" in router and "prototype_groups" in router:
        prototypes = np.asarray(router["prototypes"], dtype=np.float64)
        standardized, standardization_applied = standardize(prototypes, router)
        coverage["standardization_applied"] = bool(standardization_applied)
        coverage["prototype_separation"] = prototype_separation(standardized)
        coverage["jitter_probe"] = jitter_probe(
            prototypes,
            np.asarray(router["prototype_groups"], dtype=np.int64),
            router,
            config,
            rng,
        )
    else:
        coverage["prototype_analysis"] = "router_does_not_expose_prototypes"

    fallback = fallback_reassessment(state, config)
    parity = cache_parity(state, config, rng)

    brittle = False
    if "jitter_probe" in coverage:
        rows = coverage["jitter_probe"]["rows"]
        brittle = any(
            row["exact_match_placement_rate"] < 0.90
            and row["nearest_prototype_correct_group_rate"] >= 0.95
            for row in rows
        )

    if parity.get("status") == "parity_deviation_exceeds_tolerance":
        status = "phase67a_cache_parity_failed_block_cache_trained_experts"
    elif brittle and fallback["group_count_where_anchor_wins"] >= 8:
        status = "phase67a_router_brittleness_and_fallback_both_actionable"
    elif brittle:
        status = "phase67a_router_placement_brittle_fallback_unchanged"
    else:
        status = "phase67a_no_actionable_coverage_finding_recorded"

    core = {
        "schema_version": config["schema_version"],
        "status": status,
        "common_version": common.PHASE67_COMMON_VERSION,
        "config": {
            key: value
            for key, value in config.items()
            if key
            not in (
                "artifact_root",
                "output_root",
                "labels_root",
                "nifti_root",
                "preprocess_callable",
                "contract_file",
            )
        },
        "anchor": state["anchor_metrics"],
        "phase12c_pooled": common.metrics(
            state["labels"], state["phase12c_probability"]
        ),
        "router_coverage": coverage,
        "unknown_protocol_fallback": fallback,
        "cache_parity": parity,
        "interpretation": {
            "router_jitter_probe_measures_protocols_not_test_cases": True,
            "unknown_protocol_share_of_the_test_set_is_unmeasured": True,
            "static_token_counts_do_not_prove_runtime_behaviour": True,
            "a_router_change_is_a_static_branch_change_not_test_time_adaptation": True,
            "cache_parity_not_performed_is_not_cache_parity_passed": True,
            "no_change_to_phase56_is_authorised_by_this_cell": True,
        },
        "training_performed": False,
        "test_data_read": False,
        "test_time_adaptation": False,
        "phase56_archive_unchanged": True,
        "submission_archive_created": False,
        "provenance": state["provenance"],
    }

    contract_path = output_root / config["contract_file"]
    contract_sha = common.emit_contract(contract_path, core)
    report = {
        **core,
        "contract_sha256": contract_sha,
        "elapsed_seconds": round(time.perf_counter() - started, 3),
    }
    common.print_sanitized("PHASE67A_COVERAGE_PARITY_AUDIT", report)
    return report


if __name__ == "__main__":
    try:
        main()
    except common.Phase67Stop as stop:
        print(
            json.dumps(
                {
                    "phase": "phase67a_coverage_parity_audit",
                    "status": "phase67a_stopped",
                    "stage": str(stop),
                    "phase56_archive_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
