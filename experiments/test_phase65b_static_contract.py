import ast
import hashlib
from pathlib import Path


CELL = Path(__file__).with_name(
    "phase65_cell_165b_standalone_physics_synthetic_pretraining_gate.py"
)
source = CELL.read_text(encoding="utf-8")
tree = ast.parse(source, filename=str(CELL))


def literals(node_type=ast.Constant):
    for node in ast.walk(tree):
        if isinstance(node, node_type):
            yield node


strings = [
    node.value for node in literals()
    if isinstance(getattr(node, "value", None), str)
]
joined = "\n".join(strings)

# Standalone persistent restoration boundary.
for required in (
    "phase65ar4_corrected_offline_gate_adjudication_contract.json",
    "phase62_acquisition_domain_validation_contract.json",
    "phase64_group_blocked_diffusion_nystrom_contract.json",
    "phase65a_corrected_deployment_fallback_contract.json",
    "phase50_partitions.npz",
    "phase52_repeated_partition.npz",
    "phase57_logo_partition.npz",
    "phase31_highres_float16.npy",
    "phase56_submission.zip",
):
    assert required in joined, required

# Fixed single candidate and frozen advancement thresholds.
for required in (
    "phase65b_standalone_physics_synthetic_pretraining_gate_v2",
    "0.75_anchor_logit_plus_0.25_physics_synthetic_pretrained_3d_logit",
    "minimum_pooled_log_loss_gain",
    "minimum_pooled_auroc_gain",
    "minimum_fold_log_loss_wins",
    "maximum_fold_log_loss_regret",
    "minimum_major_groups_combined_log_loss_gain",
    "minimum_domain_bootstrap_lower_95_log_loss_gain",
    "brier_regret_allowed",
):
    assert required in source, required

assert '"synthetic_pretraining_steps": 800' in source
assert '"real_epoch_count": 12' in source
assert '"anchor_logit_weight": 0.75' in source
assert '"synthetic_expert_logit_weight": 0.25' in source
assert "for phase65a_fold in range(3):" in source
assert "phase65a_original_fold != phase65a_fold" in source
assert "phase65a_original_fold == phase65a_fold" in source
assert '"outer_images_used_for_fitting": False' in source
assert '"outer_labels_used_for_fitting": False' in source

# No BatchNorm, network client, test-runtime path, or external pretrained asset.
assert "batch_normalization_forbidden" in source
assert not any(
    isinstance(node, ast.Call)
    and isinstance(node.func, ast.Attribute)
    and node.func.attr in {"BatchNorm1d", "BatchNorm2d", "BatchNorm3d"}
    for node in ast.walk(tree)
)
assert "/code_execution" not in source
assert "torch.hub" not in source
assert "load_state_dict_from_url" not in source
for forbidden_import in ("requests", "httpx", "urllib", "socket"):
    assert not any(
        isinstance(node, (ast.Import, ast.ImportFrom))
        and (
            any(alias.name.split(".")[0] == forbidden_import for alias in node.names)
            if isinstance(node, ast.Import)
            else (node.module or "").split(".")[0] == forbidden_import
        )
        for node in ast.walk(tree)
    ), forbidden_import

# Restart checkpoints are bound to a frozen data/config digest and are private.
for required in (
    "phase65b_run_sha256",
    "procedural_pretraining.pt",
    "fold_{phase65a_fold}_oof.npz",
    "phase65b_atomic_torch_save",
    "phase65b_atomic_npz",
    "private_checkpoint_paths_exported",
    "completed_folds_resumed",
):
    assert required in source, required

# Sanitized output contract and no case-level content in the JSON contract.
assert "BEGIN SANITIZED_PHASE65B_STANDALONE_PHYSICS_SYNTHETIC_PRETRAINING_GATE" in source
assert '"contains_labels": False' in source
assert '"contains_probabilities": False' in source
assert '"contains_case_indices": False' in source
assert '"test_data_read": False' in source

compile(source, str(CELL), "exec")
print(
    "phase65b static contract validation passed",
    hashlib.sha256(source.encode("utf-8")).hexdigest(),
)
