"""Phase67E - single adjudication against the predeclared Phase67 gate.

Runs once, after every Phase67 candidate vector exists. It reads the upstream
contracts, verifies their self-hashes and their link back to the same Phase62
validation constitution, assembles every available candidate, and scores each
one against thresholds fixed before any outer number was read.

Why the gate is shaped the way it is.

The competition scores case-weighted pooled log loss. The bound that rejected
Phase65C and Phase65D is a bootstrap over the 11 stress domains that takes the
UNWEIGHTED mean, so it weights a 32-case domain exactly like a 456-case one.
Both instruments are reported here, but the case-weighted group cluster
bootstrap is primary because it is the quantity the leaderboard measures. The
macro-domain bound is retained as a secondary robustness requirement rather
than discarded, so a candidate that passes Phase67 also passes the older rule.

Neither bootstrap has much power. Inverting Phase65D's recorded interval gives
an across-domain standard deviation near 0.0079 and a standard error near
0.00239, so its 0.0034 mean gain needed to be above roughly 0.0047 to clear
zero. With only 15 acquisition groups, no group-resampling bootstrap can
certify a 0.004 effect. Phase67 therefore asks for a larger and flatter effect
instead of a weaker rule, and the declared thresholds below say so explicitly.

Harm caps apply only to stress domains with at least 30 cases. On the frozen
partition every one of the 11 domains already clears that bar - the five groups
smaller than 30 are pooled into a single 35-case domain - so the guard is
currently inert and is kept so the rule stays correct if the domain definition
ever changes. The weighting problem the primary bound fixes is not tiny
domains; it is that group12 with 32 cases and group1 with 456 count equally in
the macro mean. Any domain below the bar is reported in full and never vetoes
on its own.

Passing this gate is a development result. It authorises the deployment work -
raw NIfTI preprocessing equivalence, case-independent inference, local weight
loading, output schema, offline runtime rehearsal - and nothing beyond that.
The accepted Phase56 archive is not modified here under any outcome.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import phase67_common as common  # noqa: E402

PHASE67E_CONFIG = {
    "schema_version": "phase67_adjudication_v1",
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get(
        "DAT_OUTPUT_ROOT", "/kaggle/working/phase67_private"
    ),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),
    "macro_bootstrap_replicates": 10000,
    "macro_bootstrap_seed": 660699,
    "cluster_bootstrap_replicates": 10000,
    "cluster_bootstrap_seed": 611161,
    "powered_domain_minimum_n": common.INDIVIDUAL_STRESS_DOMAIN_MINIMUM_N,
    "gate": {
        "minimum_pooled_log_loss_gain": 0.006,
        "minimum_pooled_auroc_gain": 0.0025,
        "brier_must_improve": True,
        "minimum_case_weighted_cluster_bootstrap_lower_95": 0.0,
        "minimum_macro_domain_bootstrap_lower_95": 0.0,
        "maximum_fold_log_loss_regret": 0.001,
        "maximum_powered_domain_log_loss_regret": 0.002,
        "maximum_permutation_null_fraction": 0.25,
        "every_seed_must_improve": True,
    },
    # Separate, stronger flag. Reported, never a promotion requirement.
    "development_target": {"log_loss": 0.24, "auroc": 0.95},
    "contract_file": "phase67e_adjudication_contract.json",
}

UPSTREAM_CONTRACTS = [
    ("phase67a", "phase67a_coverage_parity_audit_contract.json", False),
    ("phase67b", "phase67b_nested_multi_expert_stack_contract.json", True),
    ("phase67c", "phase67c_multiseed_domain_invariant_expert_contract.json", False),
    ("phase67d", "phase67d_shift_robust_calibration_contract.json", False),
]


def load_upstream(output_root):
    records = {}
    for name, filename, required in UPSTREAM_CONTRACTS:
        path = Path(output_root) / filename
        if not path.is_file():
            common.require(not required, f"missing_required_contract_{name}")
            records[name] = {"available": False}
            continue
        payload = common.read_json(path, f"{name}_contract")
        common.require(
            common.contract_core_hash_valid(payload),
            f"{name}_contract_hash_mismatch",
        )
        records[name] = {
            "available": True,
            "status": payload.get("status"),
            "contract_sha256": payload.get("contract_sha256"),
            "schema_version": payload.get("schema_version"),
            "payload": payload,
        }
    return records


def read_candidate_vectors(output_root, upstream):
    """Collect every candidate the earlier cells left in the private output
    directory. Each carries the provenance needed to interpret its gate row."""
    output_root = Path(output_root)
    candidates = []
    seed_vectors = None

    stack_path = output_root / "phase67b_stacked_development_oof.npz"
    if stack_path.is_file():
        with np.load(stack_path, allow_pickle=False) as payload:
            if "primary" in payload.files:
                candidates.append(
                    {
                        "name": "phase67b_nested_multi_expert_stack",
                        "probability": np.asarray(
                            payload["primary"], dtype=np.float64
                        ),
                        "source": "phase67b",
                        "protocol": "nested_original_three_whole_group_folds",
                    }
                )
            if "nested_recalibration_baseline" in payload.files:
                candidates.append(
                    {
                        "name": "phase67b_nested_recalibration_baseline",
                        "probability": np.asarray(
                            payload["nested_recalibration_baseline"],
                            dtype=np.float64,
                        ),
                        "source": "phase67b",
                        "protocol": "nested_anchor_recalibration_only",
                    }
                )

    calibrated_path = output_root / "phase67d_calibrated_development_oof.npz"
    if calibrated_path.is_file():
        with np.load(calibrated_path, allow_pickle=False) as payload:
            for key in payload.files:
                if not key.startswith("nested_"):
                    continue
                candidates.append(
                    {
                        "name": f"phase67d_{key}",
                        "probability": np.asarray(payload[key], dtype=np.float64),
                        "source": "phase67d",
                        "protocol": "nested_shift_robust_calibration",
                    }
                )

    expert_path = output_root / "phase67c_multiseed_expert_oof.npz"
    if expert_path.is_file():
        with np.load(expert_path, allow_pickle=False) as payload:
            if "candidate" in payload.files:
                candidates.append(
                    {
                        "name": "phase67c_multiseed_expert_blend",
                        "probability": np.asarray(
                            payload["candidate"], dtype=np.float64
                        ),
                        "source": "phase67c",
                        "protocol": "five_seed_average_over_original_three_folds",
                    }
                )
            if "seed_probability" in payload.files:
                seed_vectors = np.asarray(
                    payload["seed_probability"], dtype=np.float64
                )

    valid = []
    for candidate in candidates:
        probability = candidate["probability"]
        if probability.shape != (common.CASE_COUNT,):
            continue
        if not bool(np.all(np.isfinite(probability))):
            continue
        if not bool(np.all((probability > 0.0) & (probability < 1.0))):
            continue
        valid.append(candidate)
    return valid, seed_vectors


def adjudicate(
    labels,
    groups,
    original_fold,
    domain_id,
    anchor,
    candidate,
    config,
    domain_count,
    seed_vectors=None,
    permutation_fraction=None,
):
    """Score one candidate against the predeclared gate.

    Returns named per-criterion booleans rather than a single collapsed
    verdict, following phase66f's adjudicate(), so a failure says which
    requirement failed.
    """
    gate = config["gate"]
    stability = common.stability_rows(
        labels, groups, original_fold, domain_id, anchor, candidate
    )
    pooled = stability["pooled"]
    domain_gains = common.per_domain_gains(
        labels, domain_id, anchor, candidate, expected_domains=domain_count
    )
    macro = common.macro_domain_bootstrap(
        domain_gains,
        config["macro_bootstrap_replicates"],
        config["macro_bootstrap_seed"],
    )
    cluster = common.group_cluster_bootstrap(
        labels,
        groups,
        anchor,
        candidate,
        config["cluster_bootstrap_replicates"],
        config["cluster_bootstrap_seed"],
    )

    seed_rows = []
    every_seed_improves = None
    if seed_vectors is not None and seed_vectors.ndim == 2:
        anchor_metrics = common.metrics(labels, anchor)
        for index in range(seed_vectors.shape[0]):
            seed_metrics = common.metrics(labels, seed_vectors[index])
            seed_rows.append({"seed_index": int(index), **seed_metrics})
        every_seed_improves = bool(
            all(
                row["log_loss"] < anchor_metrics["log_loss"]
                and row["auroc"] is not None
                and row["auroc"] > anchor_metrics["auroc"]
                for row in seed_rows
            )
        )

    criteria = {
        "minimum_pooled_log_loss_gain": bool(
            pooled["log_loss_gain"] >= gate["minimum_pooled_log_loss_gain"]
        ),
        "minimum_pooled_auroc_gain": bool(
            pooled["auroc_gain"] is not None
            and pooled["auroc_gain"] >= gate["minimum_pooled_auroc_gain"]
        ),
        "brier_improves": bool(pooled["brier_gain"] > 0.0),
        "case_weighted_cluster_bootstrap_lower_positive": bool(
            cluster["lower_95"]
            > gate["minimum_case_weighted_cluster_bootstrap_lower_95"]
        ),
        "macro_domain_bootstrap_lower_positive": bool(
            macro["lower_95"] > gate["minimum_macro_domain_bootstrap_lower_95"]
        ),
        "fold_regret_within_bound": bool(
            stability["maximum_fold_log_loss_regret"]
            <= gate["maximum_fold_log_loss_regret"]
        ),
        "powered_domain_regret_within_bound": bool(
            stability["maximum_powered_domain_log_loss_regret"]
            <= gate["maximum_powered_domain_log_loss_regret"]
        ),
    }
    if permutation_fraction is not None:
        criteria["permutation_null_bounded"] = bool(
            permutation_fraction <= gate["maximum_permutation_null_fraction"]
        )
    if every_seed_improves is not None and gate["every_seed_must_improve"]:
        criteria["every_seed_improves"] = every_seed_improves

    tiny_rows = [
        row
        for row in stability["domains"]
        if row["n"] < config["powered_domain_minimum_n"]
    ]
    target = config["development_target"]
    candidate_metrics = common.metrics(labels, candidate)

    return {
        "pooled": pooled,
        "folds": stability["folds"],
        "domains": stability["domains"],
        "groups": stability["groups"],
        "major_groups_combined": stability["major_groups_combined"],
        "maximum_fold_log_loss_regret": stability["maximum_fold_log_loss_regret"],
        "maximum_powered_domain_log_loss_regret": stability[
            "maximum_powered_domain_log_loss_regret"
        ],
        "improved_group_fraction": stability["improved_group_fraction"],
        "improved_domain_fraction": stability["improved_domain_fraction"],
        "case_weighted_group_cluster_bootstrap": cluster,
        "macro_domain_bootstrap": macro,
        "underpowered_domains_reported_not_vetoing": tiny_rows,
        "seed_metrics": seed_rows,
        "permutation_null_fraction_of_expert_gain": permutation_fraction,
        "gates": criteria,
        "gate_passes": bool(all(criteria.values())),
        "development_target_met": bool(
            candidate_metrics["log_loss"] <= target["log_loss"]
            and candidate_metrics["auroc"] is not None
            and candidate_metrics["auroc"] >= target["auroc"]
        ),
        "candidate_metrics": candidate_metrics,
    }


def main():
    started = time.perf_counter()
    config = PHASE67E_CONFIG
    artifact_root = Path(config["artifact_root"])
    output_root = Path(config["output_root"])
    output_root.mkdir(parents=True, exist_ok=True)

    state = common.restore_production_state(
        artifact_root, labels_root=config["labels_root"] or None
    )
    labels = state["labels"]
    groups = state["groups"]
    original_fold = state["original_fold"]
    domain_id = state["stress_domain_id"]
    anchor = state["anchor_probability"]
    domain_count = len(state["stress_domain_definitions"])

    upstream = load_upstream(output_root)
    # Every upstream cell must have restored the same validation constitution.
    for name, record in upstream.items():
        if not record.get("available"):
            continue
        provenance = record["payload"].get("provenance", {})
        if "phase62_contract_sha256" in provenance:
            common.require(
                provenance["phase62_contract_sha256"]
                == state["provenance"]["phase62_contract_sha256"],
                f"{name}_validation_constitution_mismatch",
            )

    permutation_fraction = None
    if upstream["phase67b"].get("available"):
        permutation_fraction = upstream["phase67b"]["payload"].get(
            "permutation_null_fraction_of_observed_gain"
        )

    candidates, seed_vectors = read_candidate_vectors(output_root, upstream)
    common.require(bool(candidates), "phase67e_no_candidate_vector_available")

    rows = []
    for candidate in candidates:
        applies_permutation = candidate["source"] in ("phase67b", "phase67d")
        applies_seeds = candidate["source"] == "phase67c"
        rows.append(
            {
                "name": candidate["name"],
                "source": candidate["source"],
                "protocol": candidate["protocol"],
                **adjudicate(
                    labels,
                    groups,
                    original_fold,
                    domain_id,
                    anchor,
                    candidate["probability"],
                    config,
                    domain_count,
                    seed_vectors=seed_vectors if applies_seeds else None,
                    permutation_fraction=(
                        permutation_fraction if applies_permutation else None
                    ),
                ),
            }
        )

    passing = [row for row in rows if row["gate_passes"]]
    # Among passing candidates the decision rule is the scored metric, with
    # AUROC breaking ties. Selecting here is legitimate only because every
    # candidate was produced under a group-held protocol declared in advance.
    best = (
        min(
            passing,
            key=lambda row: (
                row["candidate_metrics"]["log_loss"],
                -(row["candidate_metrics"]["auroc"] or 0.0),
            ),
        )
        if passing
        else None
    )

    parity_blocking = False
    parity_status = None
    if upstream["phase67a"].get("available"):
        parity = upstream["phase67a"]["payload"].get("cache_parity", {})
        parity_status = parity.get("status")
        parity_blocking = parity_status != "parity_within_tolerance"

    cache_trained = {"phase67c"}
    if best is None:
        status = "phase67e_no_candidate_cleared_the_declared_gate_retain_phase56"
    elif best["source"] in cache_trained and parity_blocking:
        status = (
            "phase67e_gate_passed_but_cache_parity_unproven_deployment_blocked"
        )
    else:
        status = "phase67e_gate_passed_deployment_verification_required"

    core = {
        "schema_version": config["schema_version"],
        "status": status,
        "common_version": common.PHASE67_COMMON_VERSION,
        "config": {
            key: value
            for key, value in config.items()
            if key
            not in ("artifact_root", "output_root", "labels_root", "contract_file")
        },
        "anchor": state["anchor_metrics"],
        "upstream_contracts": {
            name: {
                key: value
                for key, value in record.items()
                if key != "payload"
            }
            for name, record in upstream.items()
        },
        "candidates": rows,
        "passing_candidate_names": [row["name"] for row in passing],
        "selected_candidate": best["name"] if best else None,
        "cache_parity_status": parity_status,
        "cache_parity_blocks_cache_trained_candidates": parity_blocking,
        "required_before_any_submission": [
            "raw_nifti_to_80_cubed_cache_preprocessing_equivalence",
            "case_independent_inference_on_each_test_case",
            "local_weight_loading_with_no_network_access",
            "required_output_schema_and_row_count",
            "offline_runtime_rehearsal_following_the_phase65ar3_pattern",
            "fresh_archive_digest_recorded_before_replacing_anything",
        ],
        "interpretation": {
            "competition_metric_is_case_weighted_pooled_log_loss": True,
            "macro_domain_bootstrap_weights_a_32_case_domain_like_a_456_case_one": True,
            "case_weighted_cluster_bootstrap_is_primary_macro_is_secondary": True,
            "fifteen_groups_cannot_certify_a_0_004_effect_at_95_percent": True,
            "harm_caps_apply_only_to_domains_with_at_least_30_cases": True,
            "underpowered_domains_are_reported_and_never_veto_alone": True,
            "bootstrap_is_descriptive_after_many_prior_experiments": True,
            "repeated_partitions_reuse_the_same_patients": True,
            "development_and_leaderboard_populations_are_not_comparable": True,
            "a_passing_gate_is_not_a_leaderboard_claim": True,
            "a_passing_gate_does_not_establish_deployment_parity": True,
            "development_target_is_a_flag_not_a_promotion_requirement": True,
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
    common.print_sanitized("PHASE67E_ADJUDICATION", report)
    return report


if __name__ == "__main__":
    try:
        main()
    except common.Phase67Stop as stop:
        print(
            json.dumps(
                {
                    "phase": "phase67e_adjudication",
                    "status": "phase67e_stopped",
                    "stage": str(stop),
                    "phase56_archive_unchanged": True,
                },
                indent=2,
                sort_keys=True,
            )
        )
        raise SystemExit(2)
