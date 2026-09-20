from pathlib import Path
import gc
import runpy
import tempfile

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TinyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv3d(3, 4, kernel_size=3, padding=1)
        self.output_dimension = 8

    def forward(self, value):
        value = F.silu(self.conv(value))
        average = F.adaptive_avg_pool3d(value, 1).flatten(1)
        maximum = F.adaptive_max_pool3d(value, 1).flatten(1)
        return torch.cat([average, maximum], dim=1)


class TinyBilateralModel(nn.Module):
    def __init__(
        self, channels, blocks_per_stage, projection_dimension,
        head_hidden_dimension, residual_cap,
    ):
        super().__init__()
        self.residual_cap = float(residual_cap)
        self.encoder = TinyEncoder()
        self.head = nn.Sequential(
            nn.LayerNorm(16), nn.Linear(16, 8), nn.SiLU(), nn.Linear(8, 1)
        )

    @staticmethod
    def normalize_per_case(volume):
        mean = volume.mean(dim=(2, 3, 4), keepdim=True)
        scale = volume.std(dim=(2, 3, 4), keepdim=True).clamp_min(1.0e-3)
        return (volume - mean) / scale

    def make_bilateral_inputs(self, volume):
        value = self.normalize_per_case(volume)
        reflected = torch.flip(value, dims=[2])
        symmetric = 0.5 * (value + reflected)
        asymmetric = torch.abs(value - reflected)
        return (
            torch.cat([value, symmetric, asymmetric], dim=1),
            torch.cat([reflected, symmetric, asymmetric], dim=1),
        )

    def forward(self, volume, anchor_logit):
        original, reflected = self.make_bilateral_inputs(volume)
        encoded = self.encoder(torch.cat([original, reflected], dim=0))
        left, right = encoded.chunk(2, dim=0)
        representation = torch.cat([
            0.5 * (left + right), torch.abs(left - right)
        ], dim=1)
        raw = self.head(representation).reshape(-1)
        logit = anchor_logit.reshape(-1) + self.residual_cap * torch.tanh(raw)
        return {"logit": logit, "probability": torch.sigmoid(logit)}


def unused_augmentation(volume, seed, config):
    return volume


rng = np.random.default_rng(650165)
n = 90
groups = np.repeat(np.arange(15, dtype=np.int64), 6)
labels = np.tile(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64), 15)
original_fold = (groups % 3).astype(np.int64)
anchor_logit = 1.7 * (labels - 0.5) + rng.normal(0.0, 1.0, n)
anchor = 1.0 / (1.0 + np.exp(-anchor_logit))
domain_id = groups.copy()
domain_definitions = [
    {"name": f"group_{group}", "groups": [group]}
    for group in range(15)
]

with tempfile.TemporaryDirectory(prefix="phase65_synthetic_", ignore_cleanup_errors=True) as directory:
    directory = Path(directory)
    cache_path = directory / "cache.npy"
    contract_path = directory / "contract.json"
    volume = rng.gamma(2.0, 0.15, size=(n, 16, 16, 16)).astype(np.float32)
    volume[:, 5:11, 5:11, 5:11] += labels[:, None, None, None] * 0.45
    np.save(cache_path, volume.astype(np.float16), allow_pickle=False)

    state = {
        "PHASE64A_DIFFUSION_REPORT_PRIVATE": {
            "status": "repeated_group_blocked_gate_failed_stop_diffusion_candidate"
        },
        "PHASE62_VALIDATION_RESET_STATE_PRIVATE": {
            "contract_sha256": "synthetic_phase62_contract",
            "labels": labels,
            "groups": groups,
            "original_fold": original_fold,
            "anchor_probability": anchor,
            "stress_domain_id": domain_id,
            "stress_domain_definitions": domain_definitions,
        },
        "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE": {
            "highres_cache_file": str(cache_path)
        },
        "phase57c_model_class": TinyBilateralModel,
        "phase57c_architecture_config": {
            "channels": [4, 4, 4, 4],
            "blocks_per_stage": [1, 1, 1, 1],
            "multiscale_projection_dimension": 2,
            "head_hidden_dimension": 8,
            "scanner_augmentation": {},
        },
        "phase57c_augment_one": unused_augmentation,
        "phase57c_device": torch.device("cpu"),
        "phase57c_cuda": False,
        "phase57c_amp_dtype": torch.bfloat16,
        "PHASE65A_SYNTHETIC_TEST_OVERRIDE_PRIVATE": {
            "case_count": n,
            "group_count": 15,
            "input_shape": [16, 16, 16],
            "synthetic_batch_size": 4,
            "synthetic_pretraining_steps": 2,
            "real_batch_size": 6,
            "real_epoch_count": 1,
            "domain_bootstrap_replicates": 200,
            "contract_file": str(contract_path),
            "checkpoint_directory": str(directory / "checkpoints"),
        },
    }
    result = runpy.run_path(
        str(Path(__file__).with_name(
            "phase65_cell_165b_standalone_physics_synthetic_pretraining_gate.py"
        )),
        init_globals=state,
    )
    assert contract_path.is_file()
    report = result["PHASE65B_PHYSICS_SYNTHETIC_REPORT_PRIVATE"]
    private_state = result["PHASE65B_PHYSICS_SYNTHETIC_STATE_PRIVATE"]
    assert report["synthetic_test_mode"] is True
    assert report["restart_safety"]["requires_previous_notebook_cells"] is False
    assert report["restart_safety"]["completed_folds_resumed"] == []
    assert len(report["original_folds"]) == 3
    assert private_state["expert_oof_probability"].shape == (n,)
    assert private_state["candidate_oof_probability"].shape == (n,)

    resumed = runpy.run_path(
        str(Path(__file__).with_name(
            "phase65_cell_165b_standalone_physics_synthetic_pretraining_gate.py"
        )),
        init_globals=state,
    )
    resumed_report = resumed["PHASE65B_PHYSICS_SYNTHETIC_REPORT_PRIVATE"]
    resumed_state = resumed["PHASE65B_PHYSICS_SYNTHETIC_STATE_PRIVATE"]
    assert resumed_report["restart_safety"][
        "procedural_pretraining_resumed_from_step"
    ] == 2
    assert resumed_report["restart_safety"]["completed_folds_resumed"] == [0, 1, 2]
    assert np.array_equal(
        private_state["expert_oof_probability"],
        resumed_state["expert_oof_probability"],
    )
    assert np.array_equal(
        private_state["candidate_oof_probability"],
        resumed_state["candidate_oof_probability"],
    )
    print("PHASE65B_STANDALONE_AND_RESUME_VALIDATION_PASSED")
    # Release the memory-mapped cache the cell left in its globals
    # before the temporary directory is removed (Windows locks mappings).
    result.clear()
    del result
    gc.collect()
