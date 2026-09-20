"""Experiment 1 - is the residual loss a SYNDROME problem, not a severity problem?

HYPOTHESIS
    The loss concentrates in pathological scans that look healthy to the model,
    and that overlap is asymmetric: pathological cases bleed into the healthy
    cluster, healthy cases do not bleed the other way. If the cause is symmetric
    bilateral reduction -- the pattern of atypical parkinsonian syndromes rather
    than typical Parkinson's -- then the binary head is being asked to learn one
    decision boundary for two different-looking positive populations.

    Test: split the pathological class into asymmetric-typical and
    symmetric-atypical using the scans themselves, supervise that split as an
    AUXILIARY head, and keep the shipped output binary. If the hypothesis holds,
    loss falls in the contested stratum without rising anywhere else.

WHAT MAKES THIS DIFFERENT FROM A THREE-CLASS RESKIN
    1. The stratum is derived inside the training partition of each outer fold
       and applied unchanged to the held-out fold. No held-out case informs its
       own stratum.
    2. There is a MATCHED NULL: the identical model with the identical auxiliary
       head, trained on PERMUTED stratum labels. If the real split does not beat
       the permuted one, the auxiliary head was buying capacity, not structure.
       This is the control that separates a finding from a parameter count.
    3. Loss is reported decomposed by stratum, so a gain in the contested tail
       cannot be hidden by the easy majority, and harm in the easy majority
       cannot be hidden by the contested tail.

NO ABSOLUTE THRESHOLDS
    Every descriptor is a ratio against the scan's own reference. Under a
    randomised global gain an absolute intensity cut is wrong by the gain factor
    during training and exactly right at validation, which manufactures a leak
    that looks like a result.

INPUTS (one local directory, from --data-root or $DAT_DATA_ROOT)
    train_labels.csv        uid,is_pathologic
    volume_cache.npy        (cases, 80, 80, 80)
    acquisition_group.npy   acquisition cluster per case
    original_fold.npy       outer fold per case
    Each can be overridden individually on the command line.

OUTPUT
    A sanitized JSON contract in --output-root (or $DAT_OUTPUT_ROOT). No
    case-level values, no identifiers, no paths.

    python run.py --help
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import datcore as dat  # noqa: E402

CONFIG_FILE = Path(__file__).with_name("config.json")


# --------------------------------------------------------------------------
# Descriptors - ratios only
# --------------------------------------------------------------------------

def window_slices(shape, window):
    out = []
    for size, (lo, hi) in zip(shape, (window["lr"], window["ap"], window["si"])):
        out.append(slice(int(round(lo * size)), max(int(round(hi * size)), int(round(lo * size)) + 2)))
    return tuple(out)


def describe(volume, config):
    """Lateralisation and antero-posterior descriptors for one canonical volume.

    Returns a dict of dimensionless ratios. `asymmetry` is the quantity the
    stratum split uses: 0 means the two sides are equal, 1 means one side carries
    all of the specific signal.
    """
    window = config["striatum_window"]
    eps = float(config["descriptor"]["epsilon"])
    region = volume[window_slices(volume.shape, window)]
    reference = float(np.quantile(volume, config["descriptor"]["reference_quantile"]))
    scaled = region.astype(np.float64) / (reference + eps)

    mid = scaled.shape[0] // 2
    left, right = scaled[:mid], scaled[mid:]

    def top_mean(block):
        flat = np.sort(block.reshape(-1))[::-1]
        take = max(int(round(float(config["descriptor"]["top_fraction"]) * flat.size)), 1)
        return float(np.mean(flat[:take]))

    left_uptake, right_uptake = top_mean(left), top_mean(right)
    strong = max(left_uptake, right_uptake)
    weak = min(left_uptake, right_uptake)
    asymmetry = float((strong - weak) / (strong + weak + eps))

    # Antero-posterior gradient inside the region: putaminal tail loss shows up
    # as the posterior half falling relative to the anterior half.
    ap_axis = 1
    halves = np.array_split(scaled, 2, axis=ap_axis)
    anterior, posterior = top_mean(halves[0]), top_mean(halves[1])
    ap_ratio = float(posterior / (anterior + eps))

    return {
        "uptake_strong": strong,
        "uptake_weak": weak,
        "asymmetry": asymmetry,
        "ap_ratio": ap_ratio,
        "bilateral_level": float(0.5 * (left_uptake + right_uptake)),
    }


def describe_all(cache, config):
    rows = [describe(np.asarray(cache[i]), config) for i in range(cache.shape[0])]
    keys = sorted(rows[0])
    return {k: np.asarray([r[k] for r in rows], dtype=np.float64) for k in keys}


# --------------------------------------------------------------------------
# Fold-local stratum assignment
# --------------------------------------------------------------------------

def assign_strata(descriptors, labels, train_mask, config):
    """Split pathological cases into asymmetric-typical and symmetric-atypical.

    The threshold is a quantile of the asymmetry index computed over PATHOLOGICAL
    TRAINING cases only, then applied unchanged to every case. Returns stratum
    ids: 0 healthy, 1 asymmetric-typical, 2 symmetric-atypical.
    """
    asymmetry = descriptors["asymmetry"]
    pathological_train = (labels > 0.5) & train_mask
    dat.require(
        int(pathological_train.sum()) >= 2 * int(config["stratum"]["minimum_per_stratum"]),
        "too_few_pathological_training_cases_to_split",
    )
    threshold = float(
        np.quantile(asymmetry[pathological_train], config["stratum"]["asymmetry_quantile"])
    )
    strata = np.zeros(labels.shape[0], dtype=np.int64)
    pathological = labels > 0.5
    # Below the threshold the two sides are close: symmetric bilateral reduction.
    strata[pathological & (asymmetry <= threshold)] = 2
    strata[pathological & (asymmetry > threshold)] = 1

    # `minimum_per_stratum` has to be enforced on the RESULT, not just on the
    # input count. If the asymmetry index is degenerate -- every pathological
    # case at the same value, say -- the quantile still returns a number and the
    # split still "succeeds", but one stratum comes out empty. The auxiliary head
    # then trains on a dead class and the gate reports on a hypothesis that was
    # never actually tested. Fail loudly instead.
    minimum = int(config["stratum"]["minimum_per_stratum"])
    for index in (1, 2):
        in_training = int(((strata == index) & train_mask).sum())
        dat.require(
            in_training >= minimum,
            f"stratum_{config['stratum']['classes'][index]}_has_only_"
            f"{in_training}_training_cases_minimum_is_{minimum}",
        )
    return strata, threshold


def permute_strata(strata, labels, seed):
    """The matched null: shuffle stratum identity WITHIN the pathological cases.

    Class balance, stratum sizes, auxiliary head width and parameter count are
    all preserved. Only the correspondence between a scan and its stratum is
    destroyed.
    """
    rng = np.random.default_rng(int(seed))
    out = strata.copy()
    index = np.flatnonzero(labels > 0.5)
    out[index] = rng.permutation(strata[index])
    return out


# --------------------------------------------------------------------------
# Model
# --------------------------------------------------------------------------

def build_model(config, prevalence):
    import torch
    import torch.nn as nn

    from datcore.models3d import MultiscaleEncoder3D

    spec = config["model"]

    class SyndromeAwareNet(nn.Module):
        """One trunk, two heads.

        The binary head is what ships. The auxiliary stratum head exists only to
        shape the trunk during training and is discarded at inference, so the
        deployed output is exactly the binary probability it has always been.
        """

        def __init__(self):
            super().__init__()
            blocks = spec["blocks_per_stage"]
            if isinstance(blocks, int):
                blocks = [blocks] * len(spec["channels"])
            self.encoder = MultiscaleEncoder3D(
                int(spec["input_channels"]),
                list(spec["channels"]),
                [int(b) for b in blocks],
                int(spec["projection"]),
            )
            width = int(self.encoder.output_dimension)
            self.dropout = nn.Dropout(float(spec["dropout"]))
            self.binary_head = nn.Linear(width, 1)
            self.stratum_head = nn.Linear(width, len(config["stratum"]["classes"]))
            nn.init.zeros_(self.binary_head.weight)
            nn.init.constant_(
                self.binary_head.bias, float(np.log(prevalence / (1.0 - prevalence)))
            )

        def forward(self, volume):
            features = self.dropout(self.encoder(volume))
            return self.binary_head(features).squeeze(-1), self.stratum_head(features)

    return SyndromeAwareNet()


def train_fold(cache, labels, strata, train_index, valid_index, config, seed, device):
    """One outer fold, one seed. Returns held-out probabilities."""
    import torch
    import torch.nn.functional as F

    torch.manual_seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    import random as _random
    _random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    if hasattr(torch.backends.cudnn, "allow_tf32"):
        torch.backends.cudnn.allow_tf32 = False

    spec = config["training"]
    prevalence = float(np.mean(labels[train_index]))
    model = build_model(config, prevalence).to(device)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=float(spec["learning_rate"]),
        weight_decay=float(spec["weight_decay"]),
    )
    batch = int(spec["batch_size"])
    steps_per_epoch = max(len(train_index) // batch, 1)
    total_steps = steps_per_epoch * int(spec["epochs"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=float(spec["learning_rate"]),
        total_steps=total_steps, pct_start=float(spec["warmup_fraction"]),
    )
    use_amp = bool(spec["amp"]) and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    rng = np.random.default_rng(seed)

    model.train()
    for _ in range(int(spec["epochs"])):
        order = rng.permutation(train_index)
        for start in range(0, steps_per_epoch * batch, batch):
            picked = order[start:start + batch]
            if picked.size == 0:
                continue
            volume = torch.from_numpy(
                np.asarray(cache[picked], dtype=np.float32)
            ).unsqueeze(1).to(device)
            target = torch.from_numpy(labels[picked].astype(np.float32)).to(device)
            stratum = torch.from_numpy(strata[picked].astype(np.int64)).to(device)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logit, stratum_logits = model(volume)
                loss = F.binary_cross_entropy_with_logits(logit, target)
                loss = loss + float(spec["auxiliary_weight"]) * F.cross_entropy(
                    stratum_logits, stratum
                )
            optimiser.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimiser)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(spec["gradient_clip"]))
            previous = scaler.get_scale()
            scaler.step(optimiser)
            scaler.update()
            if scaler.get_scale() >= previous:
                scheduler.step()

    model.eval()
    out = np.zeros(valid_index.size, dtype=np.float64)
    with torch.no_grad():
        for start in range(0, valid_index.size, batch):
            picked = valid_index[start:start + batch]
            volume = torch.from_numpy(
                np.asarray(cache[picked], dtype=np.float32)
            ).unsqueeze(1).to(device)
            logit, _ = model(volume)
            out[start:start + picked.size] = (
                torch.sigmoid(logit.float()).cpu().numpy().reshape(-1)
            )
    return np.clip(out, config["probability_clip"], 1.0 - config["probability_clip"])


def out_of_fold(cache, labels, descriptors, folds, config, seed, device, permuted=False):
    probability = np.zeros(labels.shape[0], dtype=np.float64)
    thresholds = []
    for fold in sorted(set(int(f) for f in folds)):
        valid_mask = folds == fold
        train_mask = ~valid_mask
        strata, threshold = assign_strata(descriptors, labels, train_mask, config)
        thresholds.append(threshold)
        if permuted:
            strata = permute_strata(strata, labels, config["matched_null"]["seed"] + fold)
        probability[valid_mask] = train_fold(
            cache, labels, strata,
            np.flatnonzero(train_mask), np.flatnonzero(valid_mask),
            config, seed + 100 * fold, device,
        )
    return probability, thresholds


# --------------------------------------------------------------------------
# Evaluation
# --------------------------------------------------------------------------

def decompose(labels, probability, strata, config):
    """Loss per stratum, so a tail gain cannot hide majority harm."""
    rows = []
    for index, name in enumerate(config["stratum"]["classes"]):
        mask = strata == index
        if not mask.any():
            continue
        rows.append(
            {
                "stratum": name,
                "n": int(mask.sum()),
                **dat.metrics(labels[mask], probability[mask]),
            }
        )
    return rows


def evaluate(labels, candidate, anchor, strata, folds, config):
    pooled = dat.compare(labels, anchor, candidate)
    per_stratum = []
    for index, name in enumerate(config["stratum"]["classes"]):
        mask = strata == index
        if int(mask.sum()) < 5:
            continue
        row = dat.compare(labels[mask], anchor[mask], candidate[mask])
        per_stratum.append({"stratum": name, "n": int(mask.sum()), **row})
    per_fold = []
    for fold in sorted(set(int(f) for f in folds)):
        mask = folds == fold
        per_fold.append(
            {"fold": int(fold), "n": int(mask.sum()),
             **dat.compare(labels[mask], anchor[mask], candidate[mask])}
        )
    return pooled, per_stratum, per_fold


# --------------------------------------------------------------------------

def load_inputs(args, config):
    data_root = dat.resolve_root(args.data_root, "DAT_DATA_ROOT")
    labels = dat.load_labels(args.labels or data_root / "train_labels.csv")
    cache = dat.load_cache(args.cache or data_root / "volume_cache.npy")
    groups = dat.load_vector(args.groups or data_root / "acquisition_group.npy",
                             "acquisition_group.npy")
    folds = dat.load_vector(args.folds or data_root / "original_fold.npy", "original_fold.npy")
    return cache, labels, groups, folds


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data-root", default="",
                        help="directory holding the inputs (default: $DAT_DATA_ROOT)")
    parser.add_argument("--output-root", default=os.environ.get("DAT_OUTPUT_ROOT", ""),
                        help="writable directory for the contract (default: $DAT_OUTPUT_ROOT)")
    parser.add_argument("--labels", default="", help="override: train_labels.csv")
    parser.add_argument("--cache", default="", help="volume cache .npy")
    parser.add_argument("--groups", default="", help="acquisition cluster .npy")
    parser.add_argument("--folds", default="", help="outer fold .npy")
    parser.add_argument("--config", default=str(CONFIG_FILE))
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    import torch

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    dat.require(bool(args.output_root), "DAT_OUTPUT_ROOT_not_set")
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    started = time.time()
    cache, labels, groups, folds = load_inputs(args, config)
    descriptors = describe_all(cache, config)

    # The stratum used for REPORTING is fitted on the whole training population.
    # It is never used to fit a model; the models only ever see fold-local strata.
    reporting_strata, reporting_threshold = assign_strata(
        descriptors, labels, np.ones(labels.shape[0], dtype=bool), config
    )

    seeds = list(config["seeds"]["values"])
    candidate_runs, null_runs = [], []
    for seed in seeds:
        probability, _ = out_of_fold(cache, labels, descriptors, folds, config, seed, device)
        candidate_runs.append(probability)
        if config["matched_null"]["enabled"]:
            null_probability, _ = out_of_fold(
                cache, labels, descriptors, folds, config, seed, device, permuted=True
            )
            null_runs.append(null_probability)

    candidate = np.mean(candidate_runs, axis=0)
    null_mean = np.mean(null_runs, axis=0) if null_runs else None

    seed_log_losses = [dat.metrics(labels, p)["log_loss"] for p in candidate_runs]
    noise_floor = float(np.std(seed_log_losses, ddof=1)) if len(seed_log_losses) > 1 else None

    anchor = null_mean if null_mean is not None else np.full_like(candidate, float(labels.mean()))
    pooled, per_stratum, per_fold = evaluate(
        labels, candidate, anchor, reporting_strata, folds, config
    )

    bar = config["ship_bar"]
    stratum_regret = max((-row["log_loss_gain"] for row in per_stratum), default=0.0)
    fold_regret = max((-row["log_loss_gain"] for row in per_fold), default=0.0)
    seed_signs = [
        dat.compare(labels, anchor, p)["log_loss_gain"] > 0.0 for p in candidate_runs
    ]
    gates = {
        "minimum_pooled_log_loss_gain": bool(
            pooled["log_loss_gain"] >= bar["minimum_pooled_log_loss_gain"]
        ),
        "minimum_auroc_gain": bool(
            (pooled["auroc_gain"] or 0.0) >= bar["minimum_auroc_gain"]
        ),
        "beats_matched_null": bool(
            null_mean is not None
            and pooled["log_loss_gain"] >= bar["minimum_gain_over_matched_null"]
        ),
        "stratum_regret_within_bound": bool(
            stratum_regret <= bar["maximum_stratum_log_loss_regret"]
        ),
        "fold_regret_within_bound": bool(fold_regret <= bar["maximum_fold_log_loss_regret"]),
        "gain_exceeds_measured_seed_noise": bool(
            noise_floor is not None and pooled["log_loss_gain"] >= 3.0 * noise_floor
        ),
    }
    if bar["require_all_seeds_same_sign"]:
        gates["all_seeds_same_sign"] = bool(all(seed_signs) or not any(seed_signs))

    passed = all(gates.values())
    core = {
        "experiment": config["experiment"],
        "schema_version": config["schema_version"],
        "status": (
            "exp1_syndrome_structure_supported"
            if passed
            else "exp1_syndrome_structure_not_established_retain_binary_head"
        ),
        "stratum_definition": {
            "asymmetry_quantile": config["stratum"]["asymmetry_quantile"],
            "reporting_threshold": reporting_threshold,
            "sizes": {
                name: int((reporting_strata == i).sum())
                for i, name in enumerate(config["stratum"]["classes"])
            },
            "fitted_inside_training_partition_only": True,
        },
        "measured_seed_noise_floor": noise_floor,
        "seed_log_losses": seed_log_losses,
        "candidate_metrics": dat.metrics(labels, candidate),
        "matched_null_metrics": (
            dat.metrics(labels, null_mean) if null_mean is not None else None
        ),
        "loss_decomposition": decompose(labels, candidate, reporting_strata, config),
        "pooled_vs_null": pooled,
        "per_stratum_vs_null": per_stratum,
        "per_fold_vs_null": per_fold,
        "gates": gates,
        "declared_ship_bar": bar,
        "interpretation": {
            "shipped_output_is_binary": True,
            "auxiliary_head_discarded_at_inference": True,
            "strata_never_derived_from_held_out_cases": True,
            "comparison_is_against_a_permuted_stratum_control_not_a_published_number": True,
            "test_data_read": False,
            "case_level_values_exported": False,
        },
        "elapsed_seconds": round(time.time() - started, 3),
    }
    contract_path = output_root / config["contract_file"]
    contract_sha = dat.emit_contract(contract_path, core)
    dat.print_sanitized(
        "EXP1_ATYPICAL_SYNDROME_STRUCTURE", {**core, "contract_sha256": contract_sha}
    )
    return 0 if passed else 2


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except dat.DatStop as stop:
        print(f"EXP1 STOP: {stop}", file=sys.stderr)
        raise SystemExit(2)
