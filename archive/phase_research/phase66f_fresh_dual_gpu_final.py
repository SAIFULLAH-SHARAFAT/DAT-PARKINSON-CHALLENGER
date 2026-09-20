"""Phase66: one fixed DINOv2 volume-adaptation experiment.

Standalone means no prior notebook cells, imports from prior phase scripts,
or in-memory variables. Persistent development artifacts and installed NumPy
and PyTorch are still required. This evaluates a candidate; it is not a
submission ZIP and does not establish development-to-deployment parity.
"""
from __future__ import annotations
import copy
import csv
import gc
import hashlib
import io
import json
import math
import os
import random
import time
import zipfile
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.checkpoint import checkpoint

# Edit paths here if persistent artifacts were restored somewhere else.
CONFIG = {
    "artifact_root": os.environ.get("DAT_ARTIFACT_ROOT", "/kaggle/working"),
    "output_root": os.environ.get("DAT_OUTPUT_ROOT", "/kaggle/working/phase66f_fresh_dualgpu_private"),
    "labels_root": os.environ.get("DAT_LABELS_ROOT", ""),  # private training-label directory
    "weights_file": os.environ.get("DAT_DINOV2_WEIGHTS", ""), # local public checkpoint
    "epochs": 14,
    "seeds": [660601, 660602],
    "slices_per_axis": 12,
    "image_size": 224,
    "rank": 8,
    "alpha": 16.0,
    "effective_batch": 16,
    "adapter_lr": 0.0001,
    "head_lr": 0.0003,
    "weight_decay": 0.05,
    "ema_decay": 0.99,
    "gradient_clip": 1.0,
    "bootstrap_replicates": 10000,
    "historical_phase66_budget_hours": 12.0,
    "per_fit_budget_hours": 6.0,
    "gate": {
        "minimum_log_loss_gain": 0.01,
        "minimum_auroc_gain": 0.002,
        "maximum_fold_log_loss_regret": 0.005,
        "maximum_stress_domain_log_loss_regret": 0.01,
        "minimum_macro_domain_bootstrap_lower": 0.0,
    },
}
EXPECTED_PHASE56_SHA = 'd012af37307f27a8f92bf95eecc271b7a2153ede6231480670f2e769f4d6c78c'
ALGORITHM_VERSION = 'phase66f_fresh_consistent_dinov2_dualgpu_v1'
EXPECTED_DINOV2_VITS14_TENSOR_SHA = '951a04ca2ace0206b9ed058c9364113162b78b7a9e113492697d36c68160d04c'
P66_OWNED_SYMBOLS = ('Phase65BRestoreStop', 'phase65b_require', 'phase65b_sha_file', 'phase65b_json', 'phase65b_contract_core_hash_valid', 'phase65b_report_hash_valid', 'phase65b_logit_restore', 'phase65b_sigmoid_restore', 'phase65b_auc_restore', 'phase65b_metrics_restore', 'phase65b_load_vector', 'phase65b_restore_labels', 'phase65b_read_phase42_gate', 'phase65b_read_router_fold', 'phase65b_restore_production_state', 'LowRankLinear', 'LowRankPatch', 'projected', 'PatchEmbed', 'LayerScale', 'Attention', 'FeedForward', 'TransformerBlock', 'DinoSmall', 'VolumeClassifier', 'make_model', 'trainable_state', 'load_trainable', 'prepare_volume', 'canonical_hash', 'array_hash', 'implementation_hash', 'atomic_json', 'atomic_checkpoint', 'checked_checkpoint', 'locate_weights', 'CacheDataset', 'seed_everything', 'amp_context', 'training_spent', 'predict', 'fit_one', 'metrics', 'loss_vector', 'adjudicate', 'emit_report', 'run_phase66')

class Phase65BRestoreStop(RuntimeError):
    pass

PHASE65B_WORKING_ROOT = Path(CONFIG['artifact_root']).resolve()


def phase65b_require(condition, stage):
    if not bool(condition):
        raise Phase65BRestoreStop(str(stage))

def phase65b_sha_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        while True:
            block = handle.read(8 * 1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()

def phase65b_json(path, stage):
    path = Path(path)
    phase65b_require(path.is_file(), f'missing_{stage}')
    try:
        value = json.loads(path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise Phase65BRestoreStop(f'invalid_json_{stage}') from exc
    phase65b_require(isinstance(value, dict), f'non_object_{stage}')
    return value

def phase65b_contract_core_hash_valid(payload):
    phase65b_require(isinstance(payload, dict), 'invalid_contract_payload')
    claimed = payload.get('contract_sha256')
    core = {key: value for key, value in payload.items() if key != 'contract_sha256' and (not key.startswith('contains_'))}
    recomputed = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()
    return isinstance(claimed, str) and claimed == recomputed

def phase65b_report_hash_valid(payload):
    phase65b_require(isinstance(payload, dict), 'invalid_report_payload')
    claimed = payload.get('contract_sha256')
    core = dict(payload)
    core.pop('contract_sha256', None)
    core.pop('contract_file', None)
    recomputed = hashlib.sha256(json.dumps(core, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()
    return isinstance(claimed, str) and claimed == recomputed

def phase65b_logit_restore(probability):
    probability = np.clip(np.asarray(probability, dtype=np.float64), 1e-07, 1.0 - 1e-07)
    return np.log(probability) - np.log1p(-probability)

def phase65b_sigmoid_restore(logit):
    logit = np.asarray(logit, dtype=np.float64)
    output = np.empty_like(logit)
    positive = logit >= 0.0
    output[positive] = 1.0 / (1.0 + np.exp(-logit[positive]))
    exponential = np.exp(logit[~positive])
    output[~positive] = exponential / (1.0 + exponential)
    return output

def phase65b_auc_restore(labels, score):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    score = np.asarray(score, dtype=np.float64).reshape(-1)
    phase65b_require(labels.shape == score.shape, 'restore_metric_shape')
    order = np.argsort(score, kind='mergesort')
    sorted_score = score[order]
    rank = np.empty(score.size, dtype=np.float64)
    start = 0
    while start < score.size:
        stop = start + 1
        while stop < score.size and sorted_score[stop] == sorted_score[start]:
            stop += 1
        rank[order[start:stop]] = 0.5 * (start + 1 + stop)
        start = stop
    positive = labels == 1
    positive_n = int(np.sum(positive))
    negative_n = int(labels.size - positive_n)
    phase65b_require(positive_n > 0 and negative_n > 0, 'restore_metric_classes')
    statistic = float(np.sum(rank[positive]))
    statistic -= positive_n * (positive_n + 1) / 2.0
    return float(statistic / (positive_n * negative_n))

def phase65b_metrics_restore(labels, probability):
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    probability = np.clip(np.asarray(probability, dtype=np.float64).reshape(-1), 1e-07, 1.0 - 1e-07)
    phase65b_require(labels.shape == probability.shape, 'restore_metric_shape')
    loss = -(labels * np.log(probability) + (1 - labels) * np.log1p(-probability))
    return {'log_loss': float(np.mean(loss)), 'auroc': phase65b_auc_restore(labels, probability), 'mean_probability': float(np.mean(probability))}

def phase65b_load_vector(description, candidates, expected_n=1362):
    existing = []
    seen = set()
    for candidate in candidates:
        candidate = Path(candidate)
        key = str(candidate)
        if key not in seen and candidate.is_file():
            seen.add(key)
            existing.append(candidate)
    phase65b_require(bool(existing), f'missing_vector_{description}')
    reference = np.asarray(np.load(existing[0], allow_pickle=False), dtype=np.float64).reshape(-1)
    phase65b_require(reference.shape == (expected_n,), f'shape_vector_{description}')
    phase65b_require(np.all(np.isfinite(reference)), f'finite_vector_{description}')
    for duplicate_path in existing[1:]:
        duplicate = np.asarray(np.load(duplicate_path, allow_pickle=False), dtype=np.float64).reshape(-1)
        phase65b_require(duplicate.shape == reference.shape and float(np.max(np.abs(duplicate - reference))) <= 1e-12, f'duplicate_vector_mismatch_{description}')
    return reference

def phase65b_restore_labels(root):
    root_candidates = []
    private_root = CONFIG.get('labels_root') or os.environ.get('DAT_PRIVATE_ROOT')
    if private_root:
        root_candidates.append(Path(private_root))
    candidates = []
    seen = set()
    for candidate_root in root_candidates:
        candidate = candidate_root / 'train_labels.csv'
        if candidate.is_file() and str(candidate.resolve()) not in seen:
            seen.add(str(candidate.resolve()))
            candidates.append(candidate)
    if not candidates and Path('/kaggle/input').is_dir():
        for candidate in sorted(Path('/kaggle/input').glob('**/train_labels.csv')):
            lowered = [part.lower() for part in candidate.parts]
            if any(('test' in part or 'smoke' in part for part in lowered)):
                continue
            resolved = str(candidate.resolve())
            if resolved not in seen:
                seen.add(resolved)
                candidates.append(candidate)
    records = []
    for candidate in candidates:
        try:
            with candidate.open('r', newline='', encoding='utf-8') as handle:
                rows = list(csv.DictReader(handle))
        except Exception:
            continue
        if not rows or list(rows[0].keys()) != ['uid', 'is_pathologic']:
            continue
        if len(rows) != 1362:
            continue
        uids = [str(row['uid']) for row in rows]
        if len(set(uids)) != 1362 or any((not uid for uid in uids)):
            continue
        try:
            raw_labels = np.asarray([float(row['is_pathologic']) for row in rows])
            if not np.all(np.isin(raw_labels, [0.0, 1.0])):
                continue
            labels = raw_labels.astype(np.int64)
        except Exception:
            continue
        if not np.all(np.isin(labels, [0, 1])):
            continue
        records.append((uids, labels))
    phase65b_require(bool(records), 'training_labels_not_resolved')
    reference_uids, reference_labels = records[0]
    for duplicate_uids, duplicate_labels in records[1:]:
        phase65b_require(duplicate_uids == reference_uids, 'training_label_duplicate_uid_order_mismatch')
        phase65b_require(np.array_equal(duplicate_labels, reference_labels), 'training_label_duplicate_value_mismatch')
    phase65b_require(int(np.sum(reference_labels == 0)) == 615 and int(np.sum(reference_labels == 1)) == 747, 'training_label_count_mismatch')
    return reference_labels

def phase65b_read_phase42_gate(root):
    with zipfile.ZipFile(root / 'phase56_submission.zip') as z:
        value = json.loads(z.read('phase42_assets/phase42_hierarchical_gate.json'))
    phase65b_require(isinstance(value, dict), 'phase42_gate_type')
    return value

def phase65b_read_router_fold(root):
    with zipfile.ZipFile(root / 'phase56_submission.zip') as z:
        with np.load(io.BytesIO(z.read('models/phase30_acquisition_router.npz')), allow_pickle=False) as r:
            return np.asarray(r['fold_for_group'], dtype=np.int64)

def phase65b_restore_production_state():
    root = PHASE65B_WORKING_ROOT
    ar4 = phase65b_json(root / 'phase65ar4_corrected_offline_gate_adjudication_contract.json', 'phase65ar4_contract')
    phase65b_require(ar4.get('status') == 'accepted_corrected_offline_deployment_integrity_numeric_parity_not_available_ready_for_corrected_phase65b' or ar4.get('status') == 'accepted_corrected_offline_and_exact_phase56_numeric_parity_ready_for_corrected_phase65b', 'phase65ar4_not_accepted')
    phase65b_require(phase65b_report_hash_valid(ar4), 'phase65ar4_hash_mismatch')
    phase65b_require(ar4.get('gates', {}).get('corrected_offline_deployment_integrity_passed') is True, 'phase65ar4_offline_gate_not_passed')
    submission_zip = root / 'phase56_submission.zip'
    phase65b_require(submission_zip.is_file(), 'missing_phase56_submission')
    submission_sha = phase65b_sha_file(submission_zip)
    phase65b_require(ar4.get('artifact_provenance', {}).get('submission_zip', {}).get('sha256') == submission_sha, 'phase56_digest_changed_after_phase65ar4')
    phase62 = phase65b_json(root / 'phase62_acquisition_domain_validation_contract.json', 'phase62_contract')
    phase65b_require(phase62.get('status') == 'accepted_validation_reset_ready_for_phase62b', 'phase62_validation_not_accepted')
    phase65b_require(phase65b_contract_core_hash_valid(phase62), 'phase62_contract_hash_mismatch')
    phase64 = phase65b_json(root / 'phase64_group_blocked_diffusion_nystrom_contract.json', 'phase64_contract')
    phase65b_require(phase64.get('status') == 'repeated_group_blocked_gate_failed_stop_diffusion_candidate', 'phase64_rejection_not_preserved')
    phase65b_require(phase65b_contract_core_hash_valid(phase64), 'phase64_contract_hash_mismatch')
    phase65a = phase65b_json(root / 'phase65a_corrected_deployment_fallback_contract.json', 'phase65a_contract')
    phase65b_require(phase65a.get('status') == 'fallback_numeric_gate_failed_retain_phase12c_unknown_fallback', 'phase65a_fallback_decision_not_preserved')
    phase65b_require(phase65b_contract_core_hash_valid(phase65a), 'phase65a_contract_hash_mismatch')
    labels = phase65b_restore_labels(root)
    phase50_path = root / 'phase50_private_validation/phase50_partitions.npz'
    phase52_path = root / 'phase52_repeated_partition.npz'
    phase52_contract = phase65b_json(root / 'phase52_support_gate_contract.json', 'phase52_contract')
    phase65b_require(phase50_path.is_file(), 'missing_phase50_partition')
    phase65b_require(phase52_path.is_file(), 'missing_phase52_partition')
    with np.load(phase50_path, allow_pickle=False) as partition50:
        groups = np.asarray(partition50['acquisition_group'], dtype=np.int64).reshape(-1)
    with np.load(phase52_path, allow_pickle=False) as partition52:
        fold_assignment = np.asarray(partition52['fold_assignment'], dtype=np.int64)
        phase52_sha = str(np.asarray(partition52['contract_sha256']).item())
    phase65b_require(groups.shape == (1362,), 'phase50_group_shape_mismatch')
    expected_group_sizes = [39, 456, 145, 255, 76, 7, 49, 208, 32, 4, 35, 10, 32, 9, 5]
    phase65b_require(np.bincount(groups, minlength=15).tolist() == expected_group_sizes, 'phase50_group_counts_mismatch')
    phase65b_require(fold_assignment.shape == (5, 1362), 'phase52_fold_shape_mismatch')
    phase65b_require(phase52_sha == str(phase52_contract.get('contract_sha256')), 'phase52_contract_link_mismatch')
    partition_digest = hashlib.sha256(np.ascontiguousarray(fold_assignment.astype(np.int8)).tobytes()).hexdigest()
    phase65b_require(partition_digest == str(phase52_contract.get('phase52_partition_sha256')), 'phase52_partition_digest_mismatch')
    regenerated = np.full((5, 1362), -1, dtype=np.int64)
    for repeat, seed in enumerate([520101, 520102, 520103, 520104, 520105]):
        rng = np.random.default_rng(seed)
        for group in range(15):
            for label in (0, 1):
                indices = np.flatnonzero((groups == group) & (labels == label)).copy()
                if not indices.size:
                    continue
                rng.shuffle(indices)
                cycle = (np.arange(indices.size, dtype=np.int64) + int(rng.integers(0, 5))) % 5
                regenerated[repeat, indices] = cycle
    phase65b_require(np.array_equal(regenerated, fold_assignment), 'training_label_cache_row_order_mismatch')
    phase12 = phase65b_load_vector('phase12c_oof', [root / 'phase32_phase12c_oof_float64.npy'])
    phase33 = phase65b_load_vector('phase33_component_oof', [root / 'phase33_private_checkpoint/phase33_component_oof_float64.npy', root / 'phase33_component_oof_float64.npy'])
    combined39 = phase65b_load_vector('phase39_combined_residual', [root / 'phase39_private_checkpoint/phase39_combined_residual_float64.npy', root / 'phase39_combined_residual_float64.npy'])
    phase39 = phase65b_load_vector('phase39_oof', [root / 'phase39_private_checkpoint/phase39_oof_float64.npy', root / 'phase39_oof_float64.npy'])
    phase12_logit = phase65b_logit_restore(phase12)
    phase33_residual = phase65b_logit_restore(phase33) - phase12_logit
    phase39_reconstructed = phase65b_sigmoid_restore(phase12_logit + np.clip(combined39, -2.0, 2.0))
    phase65b_require(float(np.max(np.abs(phase39_reconstructed - phase39))) <= 2e-12, 'phase39_reconstruction_mismatch')
    gate42 = phase65b_read_phase42_gate(root)
    phase65b_require(float(gate42.get('phase36_raw_residual_weight')) == 0.75 and float(gate42.get('phase33_logit_residual_weight')) == 0.125 and (float(gate42.get('residual_cap')) == 1.0), 'phase42_gate_constants_mismatch')
    alpha_map = {int(group): float(alpha) for group, alpha in gate42['known_group_alpha'].items()}
    phase65b_require(set(alpha_map) == set(range(15)), 'phase42_alpha_map_mismatch')
    group_alpha = np.asarray([alpha_map[int(group)] for group in groups])
    phase42_uncapped = combined39 - 0.125 * phase33_residual
    anchor = phase65b_sigmoid_restore(phase12_logit + group_alpha * np.clip(phase42_uncapped, -1.0, 1.0))
    anchor_metrics = phase65b_metrics_restore(labels, anchor)
    phase65b_require(abs(anchor_metrics['log_loss'] - 0.29016628416289664) <= 2e-10 and abs(anchor_metrics['auroc'] - 0.9464742438589044) <= 2e-12 and (abs(anchor_metrics['mean_probability'] - 0.5500570231688339) <= 2e-10), 'phase43_anchor_metric_parity_mismatch')
    logo_path = root / 'phase57_logo_partition.npz'
    phase57_contract = phase65b_json(root / 'phase57_transport_contract.json', 'phase57_transport_contract')
    phase65b_require(logo_path.is_file(), 'missing_phase57_logo_partition')
    with np.load(logo_path, allow_pickle=False) as logo:
        fold_for_group = np.asarray(logo['original_fold_for_group'], dtype=np.int64).reshape(-1)
        logo_contract_sha = str(np.asarray(logo['contract_sha256']).item())
    phase65b_require(phase65b_contract_core_hash_valid(phase57_contract), 'phase57_transport_contract_hash_mismatch')
    phase65b_require(logo_contract_sha == phase57_contract.get('contract_sha256'), 'phase57_logo_contract_link_mismatch')
    router_fold = phase65b_read_router_fold(root).reshape(-1)
    phase65b_require(fold_for_group.shape == (15,) and np.array_equal(fold_for_group, router_fold), 'phase57_router_fold_mismatch')
    original_fold = fold_for_group[groups]
    phase65b_require(np.bincount(original_fold, minlength=3).tolist() == [467, 443, 452], 'phase57_original_fold_counts_mismatch')
    for group in range(15):
        phase65b_require(np.unique(original_fold[groups == group]).size == 1, 'phase57_group_split_leakage')
    group_sizes = np.bincount(groups, minlength=15)
    individual_groups = [int(group) for group, count in enumerate(group_sizes) if int(count) >= 30]
    tiny_groups = [group for group in range(15) if group not in individual_groups]
    domain_definitions = [{'name': f'group_{group}', 'groups': [group]} for group in individual_groups] + [{'name': 'tiny_groups_pooled', 'groups': tiny_groups}]
    phase65b_require(domain_definitions == phase62.get('stress_domain_definitions'), 'phase62_stress_domain_definition_mismatch')
    domain_id = np.full(1362, -1, dtype=np.int64)
    for domain, definition in enumerate(domain_definitions):
        domain_id[np.isin(groups, definition['groups'])] = domain
    phase65b_require(np.all(domain_id >= 0), 'phase62_domain_assignment_incomplete')
    cache_path = root / 'phase31_highres_float16.npy'
    phase65b_require(cache_path.is_file(), 'missing_phase31_highres_cache')
    cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
    phase65b_require(cache.shape == (1362, 80, 80, 80) and cache.dtype == np.float16, 'phase31_highres_cache_contract_mismatch')
    del cache
    validation_state = {'contract_sha256': str(phase62['contract_sha256']), 'labels': labels.copy(), 'groups': groups.copy(), 'original_fold': original_fold.copy(), 'anchor_probability': anchor.copy(), 'stress_domain_id': domain_id.copy(), 'stress_domain_definitions': json.loads(json.dumps(domain_definitions))}
    return {'validation_state': validation_state, 'engine_state': {'highres_cache_file': str(cache_path)}, 'phase64_report': {'status': 'repeated_group_blocked_gate_failed_stop_diffusion_candidate'}, 'submission_sha256': submission_sha, 'anchor_metrics': anchor_metrics, 'phase62_contract_sha256': str(phase62['contract_sha256']), 'phase64_contract_sha256': str(phase64['contract_sha256']), 'phase65a_contract_sha256': str(phase65a['contract_sha256']), 'phase65ar4_contract_sha256': str(ar4['contract_sha256']), 'row_order_verified': True, 'router_fold_verified': True, 'cache_contract_verified': True}

class LowRankLinear(nn.Module):
    def __init__(self, base, rank, alpha, views=3):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.ParameterList([nn.Parameter(torch.empty(rank, base.in_features)) for _ in range(views)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(base.out_features, rank)) for _ in range(views)])
        for a in self.a:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x, view):
        return self.base(x) + self.scale * F.linear(F.linear(x, self.a[view]), self.b[view])


class LowRankPatch(nn.Module):
    def __init__(self, base, rank, alpha, views=3):
        super().__init__()
        self.base = base
        self.scale = alpha / rank
        self.a = nn.ParameterList([nn.Parameter(torch.empty(rank, *base.weight.shape[1:])) for _ in range(views)])
        self.b = nn.ParameterList([nn.Parameter(torch.zeros(base.out_channels, rank, 1, 1)) for _ in range(views)])
        for a in self.a:
            nn.init.kaiming_uniform_(a, a=math.sqrt(5))

    def forward(self, x, view):
        return self.base(x) + self.scale * F.conv2d(
            F.conv2d(x, self.a[view], stride=self.base.stride), self.b[view])


def projected(layer, x, view):
    return layer(x, view) if isinstance(layer, (LowRankLinear, LowRankPatch)) else layer(x)


class PatchEmbed(nn.Module):
    def __init__(self, width, patch):
        super().__init__()
        self.proj = nn.Conv2d(3, width, patch, stride=patch)

    def forward(self, x, view):
        return projected(self.proj, x, view).flatten(2).transpose(1, 2)


class LayerScale(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(width))

    def forward(self, x):
        return x * self.gamma


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.proj = nn.Linear(width, width)

    def forward(self, x, view):
        b, n, d = x.shape
        q, k, v = projected(self.qkv, x, view).reshape(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        return projected(self.proj, out.transpose(1, 2).reshape(b, n, d), view)


class FeedForward(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Linear(width, 4 * width)
        self.fc2 = nn.Linear(4 * width, width)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, eps=1e-6)
        self.attn = Attention(width, heads)
        self.ls1 = LayerScale(width)
        self.norm2 = nn.LayerNorm(width, eps=1e-6)
        self.mlp = FeedForward(width)
        self.ls2 = LayerScale(width)

    def forward(self, x, view):
        x = x + self.ls1(self.attn(self.norm1(x), view))
        return x + self.ls2(self.mlp(self.norm2(x)))


class DinoSmall(nn.Module):
    """Non-register DINOv2 ViT-S/14; base checkpoint keys are checked strictly."""
    def __init__(self, width=384, depth=12, heads=6, patch=14, grid=37):
        super().__init__()
        self.width, self.patch, self.grid = width, patch, grid
        self.cls_token = nn.Parameter(torch.zeros(1, 1, width))
        self.pos_embed = nn.Parameter(torch.zeros(1, grid * grid + 1, width))
        self.mask_token = nn.Parameter(torch.zeros(1, width))
        self.patch_embed = PatchEmbed(width, patch)
        self.blocks = nn.ModuleList([TransformerBlock(width, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(width, eps=1e-6)
        self.checkpoint_blocks = True

    def add_adapters(self, rank, alpha):
        for p in self.parameters():
            p.requires_grad_(False)
        self.patch_embed.proj = LowRankPatch(self.patch_embed.proj, rank, alpha)
        for block in self.blocks:
            block.attn.qkv = LowRankLinear(block.attn.qkv, rank, alpha)
            block.attn.proj = LowRankLinear(block.attn.proj, rank, alpha)

    def position(self, h, w, dtype):
        gh, gw = h // self.patch, w // self.patch
        if gh == self.grid and gw == self.grid:
            return self.pos_embed.to(dtype)
        spatial = self.pos_embed[:, 1:].float().reshape(1, self.grid, self.grid, self.width).permute(0, 3, 1, 2)
        spatial = F.interpolate(spatial, scale_factor=((gh + 0.1) / self.grid, (gw + 0.1) / self.grid),
                                mode='bicubic', align_corners=False, antialias=False)
        phase65b_require(spatial.shape[-2:] == (gh, gw), 'positional_interpolation_shape')
        return torch.cat([self.pos_embed[:, :1].float(), spatial.flatten(2).transpose(1, 2)], dim=1).to(dtype)

    def forward(self, image, view=0):
        x = self.patch_embed(image, view)
        x = torch.cat([self.cls_token.expand(x.shape[0], -1, -1), x], dim=1)
        x = x + self.position(image.shape[-2], image.shape[-1], x.dtype)
        for block in self.blocks:
            if self.training and self.checkpoint_blocks:
                # Pass view explicitly: recomputation must not use mutable view state.
                x = checkpoint(block, x, view, use_reentrant=False)
            else:
                x = block(x, view)
        return self.norm(x)[:, 0]


class VolumeClassifier(nn.Module):
    def __init__(self, backbone, config):
        super().__init__()
        self.backbone = backbone
        self.image_size = config['image_size']
        self.slices = config['slices_per_axis']
        d = backbone.width
        self.slice_query = nn.Parameter(torch.empty(3, d))
        self.view_query = nn.Parameter(torch.empty(d))
        nn.init.normal_(self.slice_query, std=0.02)
        nn.init.normal_(self.view_query, std=0.02)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.Dropout(0.1), nn.Linear(d, 1))
        self.register_buffer('pixel_mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('pixel_std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, volume, jitter=False):
        # Cache axes are used consistently; no atlas, population registration,
        # per-slice normalization, or cross-case statistics are fitted here.
        b = volume.shape[0]
        views = []
        for axis in range(3):
            v = volume.movedim(axis + 2, 2)  # [B,1,slice,H,W]
            size = v.shape[2]
            centers = ((torch.arange(self.slices, device=v.device) + 0.5) * size / self.slices).long()
            if jitter:
                centers = (centers + torch.randint(-2, 3, centers.shape, device=v.device)).clamp(1, size - 2)
            # A 3-voxel slab suppresses single-slice noise, without changing scale by case.
            slab = sum(v.index_select(2, (centers + offset).clamp(0, size - 1)) for offset in (-1, 0, 1)) / 3
            images = slab.transpose(1, 2).reshape(b * self.slices, 1, *v.shape[-2:])
            images = F.interpolate(images, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            images = images.expand(-1, 3, -1, -1)
            images = (images - self.pixel_mean) / self.pixel_std
            encoded = self.backbone(images, axis).reshape(b, self.slices, -1)
            attention = torch.softmax((encoded * self.slice_query[axis]).sum(-1) / math.sqrt(encoded.shape[-1]), dim=1)
            views.append((attention.unsqueeze(-1) * encoded).sum(1))
        views = torch.stack(views, dim=1)
        attention = torch.softmax((views * self.view_query).sum(-1) / math.sqrt(views.shape[-1]), dim=1)
        return self.head((attention.unsqueeze(-1) * views).sum(1)).squeeze(-1)


def make_model(base_state, config, prevalence=0.5, tiny=False):
    backbone = DinoSmall(width=24, depth=2, heads=3, patch=4, grid=4) if tiny else DinoSmall()
    if base_state is not None:
        backbone.load_state_dict(base_state, strict=True)
    backbone.add_adapters(config['rank'], config['alpha'])
    model = VolumeClassifier(backbone, config)
    with torch.no_grad():
        model.head[-1].weight.mul_(0.1)
        model.head[-1].bias.fill_(math.log(prevalence / (1 - prevalence)))
    return model


def trainable_state(model):
    return {name: p.detach().cpu().clone() for name, p in model.named_parameters() if p.requires_grad}


def load_trainable(model, state):
    current = {name: p for name, p in model.named_parameters() if p.requires_grad}
    phase65b_require(set(current) == set(state), 'adapter_checkpoint_keys')
    with torch.no_grad():
        for name, parameter in current.items():
            phase65b_require(parameter.shape == state[name].shape, 'adapter_checkpoint_shape')
            parameter.copy_(state[name])


def prepare_volume(volume, training=False):
    x = volume.float().clamp(0, 1.5) / 1.5
    if training:
        if torch.rand((), device=x.device) < 0.5:
            x = x.flip(2)  # left/right exchange only
        gamma = torch.empty((x.shape[0], 1, 1, 1, 1), device=x.device).uniform_(0.9, 1.1)
        x = x.clamp_min(1e-6).pow(gamma)
        mix = torch.rand((), device=x.device) * 0.25
        x = (1 - mix) * x + mix * F.avg_pool3d(x, 3, stride=1, padding=1)
        x = (x + torch.randn_like(x) * 0.006).clamp(0, 1)
    return x


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def array_hash(value):
    value = np.ascontiguousarray(value)
    return hashlib.sha256(str(value.shape).encode() + str(value.dtype).encode() + value.tobytes()).hexdigest()


def implementation_hash():
    import types
    def encode_code(code):
        return {
            'bytecode': code.co_code.hex(), 'names': code.co_names,
            'vars': code.co_varnames, 'args': code.co_argcount,
            'constants': [encode_code(x) if isinstance(x, types.CodeType) else repr(x) for x in code.co_consts],
        }
    records = {}
    for name, obj in list(globals().items()):
        if name not in P66_OWNED_SYMBOLS:
            continue
        if isinstance(obj, types.FunctionType) and obj.__module__ == __name__:
            records[name] = encode_code(obj.__code__)
        elif isinstance(obj, type) and obj.__module__ == __name__:
            records[name] = {k: encode_code(v.__code__) for k, v in obj.__dict__.items()
                             if isinstance(v, types.FunctionType)}
    return canonical_hash(records)


def atomic_json(path, value):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('w') as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def atomic_checkpoint(path, payload):
    tmp = Path(str(path) + '.tmp')
    with tmp.open('wb') as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def checked_checkpoint(path, run_hash):
    try:
        value = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as exc:
        raise Phase65BRestoreStop('private_checkpoint_unreadable') from exc
    phase65b_require(value.get('run_hash') == run_hash, 'private_checkpoint_run_mismatch')
    return value


def locate_weights(root):
    if CONFIG['weights_file']:
        candidates = [Path(CONFIG['weights_file'])]
    else:
        candidates = [root / name for name in ('dinov2_vits14_lvd142m.pth', 'dinov2_vits14_pretrain.pth')]
        if not any(p.is_file() for p in candidates):
            for name in ('dinov2_vits14_lvd142m.pth', 'dinov2_vits14_pretrain.pth'):
                candidates += list(Path('/kaggle/input').glob('**/' + name))
    candidates = list(dict.fromkeys(p.resolve() for p in candidates if p.is_file()))
    phase65b_require(bool(candidates), 'missing_local_dinov2_vits14_weights')
    digests = [phase65b_sha_file(p) for p in candidates]
    phase65b_require(len(set(digests)) == 1, 'ambiguous_dinov2_weights_set_weights_file')
    try:
        loaded = torch.load(candidates[0], map_location='cpu', weights_only=True)
    except Exception as exc:
        raise Phase65BRestoreStop('dinov2_restricted_load_failed') from exc

    # Accept the raw official state dictionary or the same dictionary inside a
    # small set of common checkpoint wrappers. Nothing is dropped or partially
    # loaded. The exact tensor fingerprint below prevents a merely compatible
    # fine-tuned or otherwise substituted checkpoint from entering the run.
    probe = DinoSmall()
    expected = probe.state_dict()
    expected_keys = set(expected)
    queue = [(loaded, 0)]
    visited = set()
    compatible = []
    while queue:
        value, depth = queue.pop(0)
        if not isinstance(value, dict) or id(value) in visited or depth > 3:
            continue
        visited.add(id(value))
        if value and set(value) == expected_keys and all(torch.is_tensor(v) for v in value.values()):
            if all(tuple(value[k].shape) == tuple(expected[k].shape) for k in expected_keys):
                compatible.append((value, depth))
        for wrapper in ('state_dict', 'model', 'teacher', 'student', 'backbone'):
            if wrapper in value:
                queue.append((value[wrapper], depth + 1))
    phase65b_require(len(compatible) == 1, 'dinov2_expected_one_complete_vits14_state_dict')
    state, wrapper_depth = compatible[0]
    phase65b_require(all(v.dtype == torch.float32 for v in state.values()), 'dinov2_expected_float32_tensors')
    phase65b_require(all(torch.isfinite(v).all().item() for v in state.values()), 'nonfinite_dinov2_weights')

    tensor_digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        metadata = json.dumps([key, str(tensor.dtype), list(tensor.shape)],
                              separators=(',', ':')).encode('utf-8')
        raw = tensor.numpy().tobytes()
        tensor_digest.update(len(metadata).to_bytes(8, 'big'))
        tensor_digest.update(metadata)
        tensor_digest.update(len(raw).to_bytes(8, 'big'))
        tensor_digest.update(raw)
    tensor_sha = tensor_digest.hexdigest()
    phase65b_require(tensor_sha == EXPECTED_DINOV2_VITS14_TENSOR_SHA,
                     'dinov2_tensor_fingerprint_not_official_reference')
    try:
        probe.load_state_dict(state, strict=True)
    except Exception as exc:
        raise Phase65BRestoreStop('dinov2_exact_model_strict_load_failed') from exc
    del probe
    return state, tensor_sha, {
        'tensor_fingerprint_verified': True,
        'input_was_wrapped': bool(wrapper_depth),
        'wrapper_depth': int(wrapper_depth),
        'restricted_loader_used': True,
        'strict_model_load_passed': True,
    }


class CacheDataset(Dataset):
    def __init__(self, cache, labels, indices):
        self.cache, self.labels, self.indices = cache, labels, np.asarray(indices)

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, offset):
        index = int(self.indices[offset])
        x = torch.from_numpy(np.array(self.cache[index], dtype=np.float32, copy=True)).unsqueeze(0)
        return x, torch.tensor(float(self.labels[index]), dtype=torch.float32)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def amp_context(device):
    dtype = torch.bfloat16 if device.type == 'cuda' and torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype, enabled=device.type == 'cuda')


def training_spent(output, run_hash):
    return sum(float(checked_checkpoint(p, run_hash)['spent_seconds']) for p in output.glob('fit_*_*.pt'))


def predict(model, cache, labels, indices, device):
    model.eval()
    values = []
    with torch.inference_mode():
        for x, _ in DataLoader(CacheDataset(cache, labels, indices), batch_size=1, shuffle=False):
            with amp_context(device):
                logit = model(prepare_volume(x.to(device)), jitter=False)
            values.append(logit.float().cpu())
    result = torch.cat(values).numpy().astype(np.float64)
    phase65b_require(np.isfinite(result).all(), 'nonfinite_prediction')
    return result


def fit_one(base, cache, labels, groups, folds, fold, seed_index, output, run_hash, config, device, tiny=False):
    seed = int(config['seeds'][seed_index]) + 1009 * int(fold)
    path = output / f'fit_{seed_index}_{fold}.pt'
    train = np.flatnonzero(folds != fold)
    valid = np.flatnonzero(folds == fold)
    phase65b_require(not set(groups[train]) & set(groups[valid]), 'acquisition_group_leakage')
    phase65b_require(set(labels[train]) == {0, 1}, 'one_class_training_fold')
    prior = checked_checkpoint(path, run_hash) if path.exists() else None
    if prior is not None:
        phase65b_require(prior['train_indices_sha'] == array_hash(train) and prior['valid_indices_sha'] == array_hash(valid),
                         'checkpoint_partition_mismatch')
        if prior['complete']:
            logit = prior['valid_logits'].numpy().astype(np.float64)
            phase65b_require(logit.shape == (len(valid),) and np.isfinite(logit).all(), 'completed_prediction_invalid')
            print(f'Phase66 seed {seed_index + 1} fold {fold + 1}: restored completed fit', flush=True)
            return valid, logit, prior['spent_seconds']
    seed_everything(seed)
    model = make_model(base, config, float(labels[train].mean()), tiny=tiny).to(device)
    parameters = dict((n, p) for n, p in model.named_parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW([
        {'params': [p for n,p in parameters.items() if n.startswith('backbone.')], 'lr': config['adapter_lr']},
        {'params': [p for n,p in parameters.items() if not n.startswith('backbone.')], 'lr': config['head_lr']},
    ], weight_decay=config['weight_decay'])
    scaler = torch.amp.GradScaler('cuda', enabled=device.type == 'cuda' and not torch.cuda.is_bf16_supported())
    ema = {n: p.detach().clone() for n, p in parameters.items()}
    start_epoch, spent, step = 0, 0.0, 0
    if prior is not None:
        load_trainable(model, prior['model'])
        optimizer.load_state_dict(prior['optimizer'])
        scaler.load_state_dict(prior['scaler'])
        ema = {n: t.to(device) for n, t in prior['ema'].items()}
        start_epoch, spent, step = prior['epoch'], prior['spent_seconds'], prior['step']
        torch.set_rng_state(prior['torch_rng'])
        if device.type == 'cuda':
            torch.cuda.set_rng_state_all(prior['cuda_rng'])
    total_steps = config['epochs'] * math.ceil(len(train) / config['effective_batch'])
    payload = prior
    for epoch in range(start_epoch, config['epochs']):
        if spent >= config['per_fit_budget_hours'] * 3600:
            raise Phase65BRestoreStop('per_fit_six_hour_budget_exhausted_retain_phase56')
        started = time.perf_counter()
        generator = torch.Generator().manual_seed(seed + 7919 * epoch)
        loader = DataLoader(CacheDataset(cache, labels, train), batch_size=1, shuffle=True,
                            generator=generator, num_workers=0)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss_sum, batch_count = 0.0, 0
        for micro, (x, y) in enumerate(loader):
            block_start = (micro // config['effective_batch']) * config['effective_batch']
            divisor = min(config['effective_batch'], len(loader) - block_start)
            with amp_context(device):
                logit = model(prepare_volume(x.to(device), training=True), jitter=True)
                loss = F.binary_cross_entropy_with_logits(logit.float(), y.to(device))
            phase65b_require(torch.isfinite(loss).item(), 'nonfinite_training_loss')
            scaler.scale(loss / divisor).backward()
            loss_sum += float(loss.detach())
            batch_count += 1
            if (micro + 1) % config['effective_batch'] == 0 or micro + 1 == len(loader):
                scaler.unscale_(optimizer)
                norm = torch.nn.utils.clip_grad_norm_(list(parameters.values()), config['gradient_clip'])
                phase65b_require(torch.isfinite(norm).item(), 'nonfinite_training_gradient')
                fraction = step / max(1, total_steps - 1)
                warmup = min(1.0, (step + 1) / max(1, total_steps / config['epochs']))
                multiplier = warmup * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))
                for pg, initial in zip(optimizer.param_groups, [config['adapter_lr'], config['head_lr']]):
                    pg['lr'] = initial * multiplier
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                with torch.no_grad():
                    for name, p in parameters.items():
                        ema[name].lerp_(p, 1 - config['ema_decay'])
                step += 1
        if device.type == 'cuda':
            torch.cuda.synchronize()
        spent += time.perf_counter() - started
        payload = {
            'run_hash': run_hash, 'train_indices_sha': array_hash(train), 'valid_indices_sha': array_hash(valid),
            'model': trainable_state(model), 'ema': {n: t.detach().cpu() for n, t in ema.items()},
            'optimizer': optimizer.state_dict(), 'scaler': scaler.state_dict(),
            'torch_rng': torch.get_rng_state(), 'cuda_rng': torch.cuda.get_rng_state_all() if device.type == 'cuda' else [],
            'epoch': epoch + 1, 'step': step, 'spent_seconds': spent, 'complete': False,
        }
        atomic_checkpoint(path, payload)
        print(f'Phase66 seed {seed_index + 1}/{len(config["seeds"])} fold {fold + 1}/3 '
              f'epoch {epoch + 1}/{config["epochs"]}: train_BCE={loss_sum / batch_count:.5f}', flush=True)
    phase65b_require(payload is not None, 'fit_state_missing')
    load_trainable(model, payload['ema'])
    logit = predict(model, cache, labels, valid, device)
    payload['valid_logits'] = torch.tensor(logit)
    payload['complete'] = True
    atomic_checkpoint(path, payload)
    del model, optimizer, ema, payload
    gc.collect()
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    return valid, logit, spent


def metrics(labels, probability):
    result = phase65b_metrics_restore(labels, probability)
    result['brier'] = float(np.mean((probability - labels) ** 2))
    return result


def loss_vector(labels, probability):
    p = np.clip(probability, 1e-7, 1 - 1e-7)
    return -(labels * np.log(p) + (1 - labels) * np.log1p(-p))


def adjudicate(labels, anchor, logits_by_seed, groups, folds, domain_ids, config):
    seed_probability = phase65b_sigmoid_restore(logits_by_seed)
    # This is explicitly a two-seed ensemble on ONE original partition.
    # It is not a claim of independent validation repetitions.
    candidate = seed_probability.mean(axis=0)
    a, c = metrics(labels, anchor), metrics(labels, candidate)
    gain = loss_vector(labels, anchor) - loss_vector(labels, candidate)
    fold_gain = [float(gain[folds == f].mean()) for f in np.unique(folds)]
    domain_gain = [float(gain[domain_ids == d].mean()) for d in np.unique(domain_ids)]
    rng = np.random.default_rng(660699)
    bootstrap = np.mean(np.asarray(domain_gain)[rng.integers(0, len(domain_gain),
                        size=(config['bootstrap_replicates'], len(domain_gain)))], axis=1)
    lower, upper = (float(x) for x in np.quantile(bootstrap, [0.025, 0.975]))
    seed_metrics = [metrics(labels, p) for p in seed_probability]
    g = config['gate']
    gates = {
        'minimum_log_loss_gain': a['log_loss'] - c['log_loss'] >= g['minimum_log_loss_gain'],
        'minimum_auroc_gain': c['auroc'] - a['auroc'] >= g['minimum_auroc_gain'],
        'brier_improves': c['brier'] < a['brier'],
        'every_seed_improves_log_loss': all(m['log_loss'] < a['log_loss'] for m in seed_metrics),
        'every_seed_improves_auroc': all(m['auroc'] > a['auroc'] for m in seed_metrics),
        'fold_regret_within_bound': min(fold_gain) >= -g['maximum_fold_log_loss_regret'],
        'stress_domain_regret_within_bound': min(domain_gain) >= -g['maximum_stress_domain_log_loss_regret'],
        'macro_domain_bootstrap_lower_positive': lower > g['minimum_macro_domain_bootstrap_lower'],
    }
    return {
        'status': ('phase66_fixed_candidate_development_gate_passed_deployment_parity_pending'
                   if all(gates.values()) else 'phase66_final_attempt_failed_retain_phase56_stop_model_search'),
        'anchor_phase43_development_comparator': a,
        'primary_two_seed_foundation_expert': c,
        'individual_seed_metrics': seed_metrics,
        'log_loss_gain': a['log_loss'] - c['log_loss'],
        'auroc_gain': c['auroc'] - a['auroc'],
        'fold_log_loss_gains': fold_gain,
        'stress_domains': [{'domain': int(d), 'n': int(np.sum(domain_ids == d)), 'log_loss_gain': delta}
                           for d, delta in zip(np.unique(domain_ids), domain_gain)],
        'macro_domain_bootstrap_95': [lower, upper],
        'bootstrap_is_descriptive_after_many_prior_experiments': True,
        'gates': gates,
        'development_target_met': c['log_loss'] <= 0.24 and c['auroc'] >= 0.95,
        'leaderboard_target_met': None,
        'blends_or_calibrators_selected': False,
        'phase56_complete_sparse_adapter_oof_available': False,
        'full_phase56_numeric_or_offline_parity_claimed': False,
    }


def emit_report(output, report):
    report = dict(report)
    report['contract_sha256'] = canonical_hash(report)
    path = output / 'phase66_final_attempt_contract.json'
    atomic_json(path, report)
    print('BEGIN SANITIZED_PHASE66_FINAL_DINOV2_VOLUME_ADAPTATION')
    print(json.dumps(report, indent=2, allow_nan=False))
    print('END SANITIZED_PHASE66_FINAL_DINOV2_VOLUME_ADAPTATION')


def run_phase66():
    global PHASE65B_WORKING_ROOT
    import fcntl
    started = time.perf_counter()
    output = Path(CONFIG['output_root']).resolve()
    output.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_umask = os.umask(0o077)
    lock = (output / 'run.lock').open('a')
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.close()
        os.umask(old_umask)
        print('Phase66 is already running in this output directory.')
        return
    stage = 'environment'
    try:
        phase65b_require(torch.cuda.is_available(), 'cuda_required')
        phase65b_require(CONFIG['image_size'] % 14 == 0, 'image_size_multiple_of_14_required')
        phase65b_require(len(CONFIG['seeds']) == 2 and len(set(CONFIG['seeds'])) == 2, 'two_distinct_fixed_seeds_required')
        phase65b_require(CONFIG['epochs'] >= 1 and CONFIG['effective_batch'] >= 1, 'invalid_training_settings')
        PHASE65B_WORKING_ROOT = Path(CONFIG['artifact_root']).resolve()
        if CONFIG['labels_root']:
            os.environ['DAT_PRIVATE_ROOT'] = CONFIG['labels_root']
        stage = 'persistent_artifact_restore'
        restored = phase65b_restore_production_state()
        phase65b_require(restored['submission_sha256'] == EXPECTED_PHASE56_SHA, 'unexpected_phase56_digest')
        state = restored['validation_state']
        labels, groups, folds, anchor, domains = [np.asarray(state[k]) for k in
            ['labels', 'groups', 'original_fold', 'anchor_probability', 'stress_domain_id']]
        cache_path = Path(restored['engine_state']['highres_cache_file'])
        cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
        print('Phase66: original folds and development comparator restored; checking local weights and cache.', flush=True)
        stage = 'pretrained_weights_and_cache'
        base, weight_hash, weight_evidence = locate_weights(PHASE65B_WORKING_ROOT)
        for start in range(0, len(cache), 16):
            phase65b_require(np.isfinite(cache[start:start+16]).all(), 'nonfinite_voxel_cache')
        mathematical_config = {k: v for k, v in CONFIG.items() if not k.endswith('root') and k != 'weights_file'}
        freeze = {
            'algorithm': ALGORITHM_VERSION, 'implementation_sha': implementation_hash(),
            'config': mathematical_config, 'weights_tensor_sha': weight_hash,
            'cache_sha': phase65b_sha_file(cache_path),
            'labels_sha': array_hash(labels), 'groups_sha': array_hash(groups), 'folds_sha': array_hash(folds),
            'anchor_sha': array_hash(anchor), 'domain_ids_sha': array_hash(domains),
            'phase56_sha': restored['submission_sha256'],
            'torch': str(torch.__version__), 'numpy': str(np.__version__),
            'cuda_version': str(torch.version.cuda),
            'device_type': torch.cuda.get_device_name(0),
        }
        run_hash = canonical_hash(freeze)
        stage = 'frozen_manifest'
        manifest_path = output / 'frozen_run.json'
        if manifest_path.exists():
            frozen = phase65b_json(manifest_path, 'frozen_run')
            phase65b_require(frozen == freeze, 'frozen_inputs_code_settings_or_runtime_changed')
        else:
            phase65b_require(not list(output.glob('fit_*_*.pt')), 'orphan_checkpoints_without_manifest')
            atomic_json(manifest_path, freeze)
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(min(8, os.cpu_count() or 1))
        device = torch.device('cuda:0')
        logits = np.full((len(CONFIG['seeds']), len(labels)), np.nan, dtype=np.float64)
        stage = 'fixed_training_and_oof_prediction'
        # The outer outcomes are scored only after every predeclared fit finishes.
        for seed_index in range(len(CONFIG['seeds'])):
            for fold in range(3):
                valid, prediction, _ = fit_one(base, cache, labels, groups, folds, fold, seed_index,
                                                output, run_hash, CONFIG, device)
                logits[seed_index, valid] = prediction
        phase65b_require(np.isfinite(logits).all(), 'incomplete_oof_predictions')
        stage = 'final_adjudication'
        phase65b_require(phase65b_sha_file(PHASE65B_WORKING_ROOT / 'phase56_submission.zip') == EXPECTED_PHASE56_SHA,
                         'phase56_changed_during_run')
        report = adjudicate(labels, anchor, logits, groups, folds, domains, CONFIG)
        report.update({
            'phase': 'phase66_final_dinov2_volume_adaptation', 'frozen_run_sha256': run_hash,
            'restart_safe': True, 'previous_cells_required': False,
            'fit_count': 6, 'partition_count': 1, 'seeds_per_fold': 2,
            'preprocessing': 'fixed_existing_80cubed_cache_to_three_axes_12_slabs_each_224px',
            'method_is_an_adaptation_of_anymc3d_not_a_paper_reproduction': True,
            'group_held_out_for_every_expert_fit': True,
            'validation_labels_used_for_epoch_or_hyperparameter_selection': False,
            'fresh_unseen_validation_cohort': False,
            'prior_experiment_decisions_reversed': False,
            'phase56_archive_unchanged': True,
            'pretrained_weights': weight_evidence,
            'training_seconds_completed_epochs': training_spent(output, run_hash),
            'current_invocation_seconds': time.perf_counter() - started,
            'training_labels_and_cached_training_volumes_read_locally': True,
            'test_data_read': False, 'network_downloads_performed_by_cell': False,
            'private_paths_uids_predictions_or_weights_exported_in_report': False,
            'case_independent_inference_design': True,
            'submission_archive_created': False,
            'deployment_next_step_if_passed': 'verify exact raw-NIfTI preprocessing and held-out inference export parity before packaging',
        })
        emit_report(output, report)
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, Phase65BRestoreStop) else type(exc).__name__
        # Exception messages and tracebacks can contain private paths: export only our stage code.
        emit_report(output, {'phase': 'phase66_final_dinov2_volume_adaptation',
                    'status': 'phase66_interrupted_or_setup_runtime_stop_retain_phase56',
                    'stage': stage, 'reason': reason, 'restart_from_last_complete_epoch': True,
                    'unfinished_epoch_may_repeat': True, 'case_values_or_private_logs_exported': False})
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()
        os.umask(old_umask)


# Phase66F: fresh, source-consistent six-fit dual-GPU experiment. It never
# imports predictions or checkpoints from Phase66/Phase66R1/R2/R3.
import subprocess
import sys

PHASE66F_OUTPUT = Path('/kaggle/working/phase66f_fresh_dualgpu_private')
PHASE66F_VERSION = 'phase66f_fresh_six_fit_dual_gpu_coordinator_v1'


def phase66f_context(expected_freeze=None):
    global PHASE65B_WORKING_ROOT
    phase65b_require(torch.cuda.is_available(), 'cuda_required')
    PHASE65B_WORKING_ROOT = Path(CONFIG['artifact_root']).resolve()
    if CONFIG['labels_root']:
        os.environ['DAT_PRIVATE_ROOT'] = CONFIG['labels_root']
    restored = phase65b_restore_production_state()
    phase65b_require(restored['submission_sha256'] == EXPECTED_PHASE56_SHA,
                     'unexpected_phase56_digest')
    state = restored['validation_state']
    labels, groups, folds, anchor, domains = [np.asarray(state[k]) for k in
        ['labels', 'groups', 'original_fold', 'anchor_probability', 'stress_domain_id']]
    cache_path = Path(restored['engine_state']['highres_cache_file'])
    phase65b_require(cache_path.is_file(), 'missing_highres_cache')
    cache = np.load(cache_path, mmap_mode='r', allow_pickle=False)
    base_state, weight_hash, weight_evidence = locate_weights(PHASE65B_WORKING_ROOT)
    if expected_freeze is None:
        for start in range(0, len(cache), 16):
            phase65b_require(np.isfinite(cache[start:start + 16]).all(),
                             'nonfinite_voxel_cache')
        cache_sha = phase65b_sha_file(cache_path)
    else:
        phase65b_require(isinstance(expected_freeze, dict), 'invalid_expected_freeze')
        phase65b_require(cache_path.stat().st_size == expected_freeze.get('cache_bytes'),
                         'worker_cache_size_changed')
        cache_sha = expected_freeze.get('cache_sha')
    mathematical_config = {k: v for k, v in CONFIG.items()
                           if not k.endswith('root') and k != 'weights_file'}
    freeze = {
        'algorithm': ALGORITHM_VERSION,
        'coordinator_version': PHASE66F_VERSION,
        'implementation_sha': implementation_hash(),
        'config': mathematical_config,
        'weights_tensor_sha': weight_hash,
        'cache_sha': cache_sha,
        'cache_bytes': cache_path.stat().st_size,
        'labels_sha': array_hash(labels),
        'groups_sha': array_hash(groups),
        'folds_sha': array_hash(folds),
        'anchor_sha': array_hash(anchor),
        'domain_ids_sha': array_hash(domains),
        'phase56_sha': restored['submission_sha256'],
        'torch': str(torch.__version__),
        'numpy': str(np.__version__),
        'cuda_version': str(torch.version.cuda),
        'device_type': torch.cuda.get_device_name(0),
    }
    if expected_freeze is not None:
        phase65b_require(freeze == expected_freeze,
                         'fresh_run_source_inputs_settings_or_runtime_changed')
    return {
        'restored': restored, 'labels': labels, 'groups': groups, 'folds': folds,
        'anchor': anchor, 'domains': domains, 'cache': cache,
        'base_state': base_state, 'weight_evidence': weight_evidence,
        'freeze': freeze, 'run_hash': canonical_hash(freeze),
    }


def phase66f_read_complete_fit(seed_index, fold, context):
    path = PHASE66F_OUTPUT / f'fit_{seed_index}_{fold}.pt'
    if not path.is_file():
        return None
    value = checked_checkpoint(path, context['run_hash'])
    train = np.flatnonzero(context['folds'] != fold)
    valid = np.flatnonzero(context['folds'] == fold)
    phase65b_require(value.get('train_indices_sha') == array_hash(train),
                     'checkpoint_train_partition_mismatch')
    phase65b_require(value.get('valid_indices_sha') == array_hash(valid),
                     'checkpoint_valid_partition_mismatch')
    if not bool(value.get('complete')):
        return None
    logit = value.get('valid_logits')
    phase65b_require(torch.is_tensor(logit), 'completed_logits_type')
    logit = logit.numpy().astype(np.float64)
    phase65b_require(logit.shape == (len(valid),) and np.isfinite(logit).all(),
                     'completed_logits_invalid')
    return valid, logit, float(value['spent_seconds'])


def phase66f_worker(seed_index, fold):
    try:
        manifest = phase65b_json(PHASE66F_OUTPUT / 'frozen_scientific_run.json',
                                 'phase66f_manifest')
        context = phase66f_context(expected_freeze=manifest)
        device = torch.device('cuda:0')
        torch.backends.cudnn.benchmark = False
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.set_num_threads(min(2, os.cpu_count() or 1))
        fit_one(context['base_state'], context['cache'], context['labels'],
                context['groups'], context['folds'], fold, seed_index,
                PHASE66F_OUTPUT, context['run_hash'], CONFIG, device)
        phase65b_require(phase66f_read_complete_fit(seed_index, fold, context) is not None,
                         'worker_fit_not_complete')
        print(f'Phase66F worker completed seed {seed_index + 1} fold {fold + 1}',
              flush=True)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, Phase65BRestoreStop) else type(exc).__name__
        print(f'PHASE66F_WORKER_STOP seed={seed_index + 1} '
              f'fold={fold + 1} reason={reason}', flush=True)
        return 2


def phase66f_emit(report):
    report = dict(report)
    report['contract_sha256'] = canonical_hash(report)
    atomic_json(PHASE66F_OUTPUT / 'phase66f_fresh_dualgpu_contract.json', report)
    print('BEGIN SANITIZED_PHASE66F_FRESH_DUAL_GPU')
    print(json.dumps(report, indent=2, allow_nan=False))
    print('END SANITIZED_PHASE66F_FRESH_DUAL_GPU')


def run_phase66f():
    import fcntl
    started = time.perf_counter()
    PHASE66F_OUTPUT.mkdir(parents=True, exist_ok=True, mode=0o700)
    old_umask = os.umask(0o077)
    lock = None
    locked = False
    stage = 'locks'
    context = None
    succeeded = False
    active_processes = []
    try:
        lock = (PHASE66F_OUTPUT / 'coordinator.lock').open('a')
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        locked = True
        stage = 'environment_and_frozen_manifest'
        phase65b_require(torch.cuda.device_count() == 2,
                         'exactly_two_visible_cuda_devices_required')
        phase65b_require(torch.cuda.get_device_name(0) == torch.cuda.get_device_name(1),
                         'two_gpu_models_must_match')
        context = phase66f_context()
        manifest_path = PHASE66F_OUTPUT / 'frozen_scientific_run.json'
        if manifest_path.exists():
            phase65b_require(phase65b_json(manifest_path, 'phase66f_manifest') ==
                             context['freeze'], 'fresh_run_manifest_changed')
        else:
            phase65b_require(not list(PHASE66F_OUTPUT.glob('fit_*_*.pt')),
                             'orphan_checkpoints_without_manifest')
            atomic_json(manifest_path, context['freeze'])

        complete = []
        remaining = []
        for seed_index in range(len(CONFIG['seeds'])):
            for fold in range(3):
                if phase66f_read_complete_fit(seed_index, fold, context) is None:
                    remaining.append((seed_index, fold))
                else:
                    complete.append((seed_index, fold))
        print(f'Phase66F: source-consistent fresh run; complete={len(complete)}, '
              f'remaining={len(remaining)}; launching at most two fits per wave.',
              flush=True)

        stage = 'parallel_six_fit_training'
        script_path = Path(__file__).resolve()
        for start in range(0, len(remaining), 2):
            wave = remaining[start:start + 2]
            processes = []
            for gpu_index, (seed_index, fold) in enumerate(wave):
                environment = dict(os.environ)
                environment['CUDA_VISIBLE_DEVICES'] = str(gpu_index)
                environment['PYTHONUNBUFFERED'] = '1'
                command = [sys.executable, str(script_path), '--phase66f-worker',
                           str(seed_index), str(fold)]
                process = subprocess.Popen(command, env=environment, close_fds=True)
                active_processes.append(process)
                processes.append((seed_index, fold, process))
            for seed_index, fold, process in processes:
                return_code = process.wait()
                if process in active_processes:
                    active_processes.remove(process)
                phase65b_require(return_code == 0,
                                 f'worker_failed_seed_{seed_index + 1}_fold_{fold + 1}')

        stage = 'fresh_combined_adjudication'
        logits = np.full((len(CONFIG['seeds']), len(context['labels'])), np.nan,
                         dtype=np.float64)
        fit_seconds = 0.0
        for seed_index in range(len(CONFIG['seeds'])):
            for fold in range(3):
                fit = phase66f_read_complete_fit(seed_index, fold, context)
                phase65b_require(fit is not None, 'incomplete_six_fit_set')
                valid, prediction, seconds = fit
                logits[seed_index, valid] = prediction
                fit_seconds += seconds
        phase65b_require(np.isfinite(logits).all(), 'incomplete_oof_predictions')
        phase65b_require(phase65b_sha_file(
            Path(CONFIG['artifact_root']) / 'phase56_submission.zip') ==
            EXPECTED_PHASE56_SHA, 'phase56_changed_during_phase66f')
        report = adjudicate(context['labels'], context['anchor'], logits,
                            context['groups'], context['folds'], context['domains'], CONFIG)
        report.update({
            'phase': 'phase66f_fresh_source_consistent_dual_gpu_rerun',
            'coordinator_version': PHASE66F_VERSION,
            'fresh_run_sha256': context['run_hash'],
            'fresh_fit_count': 6,
            'two_seeds_three_original_group_folds': True,
            'fits_launched_at_most_two_in_parallel': True,
            'parallelism': 'one_independent_fold_seed_fit_per_gpu',
            'ddp_used': False,
            'old_phase66_checkpoints_or_predictions_used': False,
            'old_phase66_outputs_modified': False,
            'fresh_implementation_consistent_across_all_six_fits': True,
            'reason_for_fresh_run': 'exact_historical_phase66_implementation_not_recoverable',
            'historical_phase66_gpu_budget_hours':
                float(CONFIG['historical_phase66_budget_hours']),
            'fresh_per_fit_budget_hours': float(CONFIG['per_fit_budget_hours']),
            'fresh_completed_epoch_gpu_seconds': fit_seconds,
            'current_invocation_wall_seconds': time.perf_counter() - started,
            'pretrained_weights': context['weight_evidence'],
            'phase56_archive_unchanged': True,
            'training_labels_and_cached_training_volumes_read_locally': True,
            'test_data_read': False,
            'test_time_adaptation': False,
            'case_values_private_paths_uids_predictions_or_weights_exported': False,
            'submission_archive_created': False,
            'deployment_next_step_if_passed':
                'prove_raw_nifti_preprocessing_and_held_out_inference_export_parity',
        })
        phase66f_emit(report)
        succeeded = True
    except BlockingIOError:
        phase66f_emit({
            'phase': 'phase66f_fresh_source_consistent_dual_gpu_rerun',
            'status': 'phase66f_runtime_stop_retain_phase56',
            'stage': 'locks', 'reason': 'phase66f_coordinator_already_running',
            'restart_safe': True, 'old_phase66_outputs_modified': False,
            'case_values_private_paths_uids_predictions_or_logs_exported': False,
        })
    except (Exception, KeyboardInterrupt) as exc:
        reason = str(exc) if isinstance(exc, Phase65BRestoreStop) else type(exc).__name__
        phase66f_emit({
            'phase': 'phase66f_fresh_source_consistent_dual_gpu_rerun',
            'status': 'phase66f_interrupted_or_runtime_stop_retain_phase56',
            'stage': stage, 'reason': reason,
            'restart_safe': True, 'unfinished_epoch_may_repeat': True,
            'old_phase66_outputs_modified': False,
            'case_values_private_paths_uids_predictions_or_logs_exported': False,
        })
    finally:
        for process in active_processes:
            try:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait(timeout=10)
            except Exception:
                pass
        if locked:
            fcntl.flock(lock, fcntl.LOCK_UN)
        if lock is not None:
            lock.close()
        os.umask(old_umask)
    return succeeded


if __name__ == '__main__':
    if len(sys.argv) == 4 and sys.argv[1] == '--phase66f-worker':
        raise SystemExit(phase66f_worker(int(sys.argv[2]), int(sys.argv[3])))
    run_phase66f()
