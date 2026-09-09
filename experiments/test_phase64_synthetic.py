from pathlib import Path
import runpy
import tempfile

import numpy as np


rng = np.random.default_rng(640264)
n = 90
group_count = 15
labels = np.tile(np.asarray([0, 0, 0, 1, 1, 1], dtype=np.int64), group_count)
groups = np.repeat(np.arange(group_count, dtype=np.int64), 6)
repeat_fold = np.stack([
    (groups + repeat) % 3 for repeat in range(2)
], axis=0).astype(np.int64)
anchor_logit = (2.1 * (labels - 0.5) + rng.normal(0.0, 1.0, n))
anchor = 1.0 / (1.0 + np.exp(-anchor_logit))

domain_id = groups.copy()
domain_definitions = [
    {
        "domain_id": int(group),
        "name": f"group_{group}",
        "groups": [int(group)],
        "n": int(np.sum(groups == group)),
    }
    for group in range(group_count)
]

with tempfile.TemporaryDirectory(prefix="phase64_synthetic_") as directory:
    directory = Path(directory)
    cache_path = directory / "cache.npy"
    contract_path = directory / "contract.json"
    volume = rng.gamma(2.0, 0.5, size=(n, 16, 16, 16)).astype(np.float32)
    volume[:, 5:11, 5:11, 5:11] += labels[:, None, None, None] * 0.8
    np.save(cache_path, volume.astype(np.float16), allow_pickle=False)

    state = {
        "PHASE63A_SPARSE_RISK_REPORT_PRIVATE": {
            "status": "sparse_risk_localization_rejected_stop_router_family",
        },
        "PHASE62_VALIDATION_RESET_STATE_PRIVATE": {
            "contract_sha256": "synthetic_phase62_contract",
            "labels": labels,
            "groups": groups,
            "anchor_probability": anchor,
            "stress_domain_id": domain_id,
            "stress_domain_definitions": domain_definitions,
            "group_blocked_repeat_case_fold": repeat_fold,
        },
        "PHASE58_GRADIENT_TRANSPORT_STATE_PRIVATE": {
            "highres_cache_file": str(cache_path),
        },
        "PHASE64A_SYNTHETIC_TEST_OVERRIDE_PRIVATE": {
            "case_count": n,
            "group_count": group_count,
            "repeat_count": 2,
            "input_shape": [16, 16, 16],
            "pooled_shape": [4, 4, 4],
            "dct_keep_shape": [4, 4, 4],
            "pca_dimension": 12,
            "diffusion_neighbor_rank": 5,
            "diffusion_dimension": 6,
            "domain_bootstrap_replicates": 200,
            "contract_file": str(contract_path),
        },
    }
    result = runpy.run_path(
        str(Path(__file__).with_name("phase64_cell_164a_group_blocked_diffusion_nystrom_gate.py")),
        init_globals=state,
    )
    assert contract_path.is_file()
    report = result["PHASE64A_DIFFUSION_REPORT_PRIVATE"]
    assert report["synthetic_test_mode"] is True
    assert report["validation"]["fit_count"] == 6
    assert report["fit_diagnostics"]["fit_count"] == 6
    assert result["PHASE64A_DIFFUSION_STATE_PRIVATE"][
        "mean_candidate_oof_probability"
    ].shape == (n,)
    print("PHASE64_SYNTHETIC_VALIDATION_PASSED")
