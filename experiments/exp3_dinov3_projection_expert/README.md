# Experiment 3 — can a foundation model contribute on a *physical* representation?

This is the SOTA arm. It is built so that it can win honestly, or lose conclusively.

## Read this before running it

DINOv3 has never been tested on this task. What *has* been tested, and falsified four times, is a
foundation model applied to a **raw volume**:

| where | result |
|---|---|
| Phase66F (this repo), DINOv2 ViT-S/14 on the volume | **−0.091 log loss, −0.027 AUROC**, rejected |
| 6th place, DINOv2 | **+0.05 log loss** |
| 6th place, `eva02_base` | 0.2328 on one acquisition cluster, **0.2874 on another** |
| 10th place, frozen DINOv2 features | AUROC 0.93 alone, **−0.0002** to the ensemble |

The 6th-place team's conclusion was that pretrained extractors on this modality are
**acquisition-conditional**. The physical reason is that a ~9 mm point-spread function at 2 mm
sampling leaves the volume roughly twice oversampled — there is less information in a voxel than
its size suggests, so extra capacity has little to consume.

**This experiment therefore changes the input, not the model size.** The volume is reduced to four
physically meaningful 2D channels first, and the foundation model reads *that*. It is the one form
in which a foundation model has not yet been falsified on this data.

## The arm that matters is the control

Three arms, one harness, identical input:

| arm | what it is |
|---|---|
| `dinov3_frozen` | DINOv3 features, trunk frozen |
| `dinov3_finetuned` | DINOv3 fine-tuned at 0.1× the head learning rate |
| **`scratch_control`** | **the same head on the same input, no pretraining at all** |

The scratch control is mandatory — `run.py` refuses to start without it. It is the **matched
null**: if all three arms land together, the *representation* did the work and DINOv3 contributed
nothing. That is precisely the control which turned a published "C8 equivariance works" result
into "C8 steerability contributes nothing — the decorrelation came from a missing input branch."

**The declared bar is DINOv3 minus the scratch control, not DINOv3 minus Phase66F.** Beating the
old raw-volume baseline would prove only that the projection works.

## The projection

`physical_projection()` reduces the superior-inferior axis to four channels and holds **zero
learnable parameters** — the synthetic test asserts it declares no `nn.Parameter` and no buffer. A
learnable version of an equivalent projection collapsed to near rank-1 in a published solution, so
"fixed" here is a measured optimum, not a shortcut.

| channel | what it is |
|---|---|
| `peak` | τ-softened log-mean-exp over S-I; τ → 0 gives the mean, τ → ∞ the max |
| `mean` | arithmetic mean over S-I |
| `anisotropy_times_uptake` | local second-moment anisotropy × relative uptake |
| `normalised_displacement_tail` | eroded soft indicator of what stays bright relative to the in-image maximum |

Each scan is first divided by **its own** reference (the mean of voxels above 0.15 × its 99.9th
percentile). That is not decoration: without it, τ responds to the absolute level, and the 0.46×–2.3×
global gain applied by the gamma augmentation would move every downstream channel — the projection
would be measuring the camera rather than the patient. With it, the gain cancels exactly, and the
test asserts invariance at both ends of the range.

## Physics-consistent augmentation, and the rule it forces

Left-right flip (the only label-preserving mirror), affine ±31.7°, Poisson counts at 25–175
(the correct noise model for SPECT, whose acquisition varies fivefold in signal-to-noise across
centres), and a gamma operation that on normalised input is a **0.46×–2.3× global gain**.

From which: **no absolute threshold may appear anywhere in the pipeline.** A level computed on the
clean image is wrong by up to 2.3× on a third of batches and exactly right at validation time.
Every 3D channel that failed in the published ablations binarised the volume first. The synthetic
test measures the realised gain span to prove this is a live constraint, not a comment.

## Declared gates

In `config.json`, written before the first run. Applied to each pretrained arm against the control:

| gate | bar |
|---|---|
| `beats_scratch_control` | ≥ 0.005 log loss |
| `auroc_not_worse` | ≥ 0 |
| `all_seeds_same_sign` | required, 3 seeds |
| `gain_exceeds_measured_seed_noise` | ≥ 3 × the **measured** seed standard deviation |
| `no_cluster_collapse` | no acquisition cluster worse by > 0.04 |
| `cluster_regret_within_bound` | no cluster worse by > 0.010 |

The cluster read is the instrument **neither top-ten solution had** — both validated with every
acquisition cluster present in every fold. The `eva02_base` collapse above is exactly what it
exists to catch.

## Running it

```bash
# one-off, or put these in config/local.env
export DAT_DATA_ROOT="/path/to/data"      # labels + prepared arrays
export DAT_OUTPUT_ROOT="/path/to/runs"    # writable, for the contract
export DAT_DINOV3_WEIGHTS="/path/to/dinov3_checkpoint.pth"

python run.py --device cuda
```

Weights are loaded **locally and never downloaded**; the checkpoint digest is recorded in the
contract. With `DAT_DINOV3_WEIGHTS` unset the run stops rather than reaching for the network — the
synthetic test asserts this.

Nine training runs (3 arms × 3 seeds) over the outer folds. `--arms scratch_control,dinov3_frozen`
runs a subset; the control cannot be omitted. Exit code 0 only if a pretrained arm clears every
gate.

## Validate it without data or weights

```bash
python test_synthetic.py
```

## The honest expectation

**The scratch control is likely to match or beat both DINOv3 arms.** Four independent measurements
point that way and none point the other. The experiment is built so that outcome is a clean,
citable negative result obtained in nine runs — rather than another Phase66F, which cost two GPUs
and six fits to learn the same thing less precisely.

If DINOv3 *does* clear the bar, this is the first evidence on this task that a foundation
model helps, with a matched null and an out-of-cluster read already attached to it.
