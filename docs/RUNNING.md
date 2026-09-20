# Running the experiments locally

Everything here runs on a single workstation. There is no notebook host, no cloud mount and no
path discovery: every location is an explicit flag or an environment variable.

## Environment

```bash
python -m venv .venv
. .venv/bin/activate          # Windows PowerShell: .venv/Scripts/Activate.ps1
pip install -r requirements.txt
```

Python 3.12 is the reference interpreter; 3.13 also works. Install the PyTorch build matching your
CUDA version from pytorch.org first if you want GPU training — experiment 2 needs no GPU at all,
and experiments 1 and 3 fall back to CPU automatically.

## Configuration

```bash
cp config/local.env.example config/local.env
set -a && . config/local.env && set +a
```

`config/local.env` is gitignored. Three variables:

| variable | meaning |
|---|---|
| `DAT_DATA_ROOT` | directory holding the labels and prepared arrays |
| `DAT_OUTPUT_ROOT` | writable directory for the sanitized JSON contracts |
| `DAT_DINOV3_WEIGHTS` | experiment 3 only; a local checkpoint, never downloaded |

## Inputs

The competition data is not distributed here. Once you have it, `DAT_DATA_ROOT` should contain:

| file | shape | notes |
|---|---|---|
| `train_labels.csv` | 1,362 rows | columns `uid,is_pathologic`, exactly as distributed |
| `volume_cache.npy` | `(1362, 80, 80, 80)` | preprocessed volumes, canonical case order |
| `acquisition_group.npy` | `(1362,)` int | acquisition cluster per case |
| `original_fold.npy` | `(1362,)` int | outer fold per case |
| `oof_probability.npy` | `(1362,)` float | out-of-fold probability; experiment 2 only |

Every loader length-checks its input and refuses a file of the wrong shape. A silently truncated or
reordered array produces a different experiment that still runs to completion and still prints a
plausible number, which is the most expensive mistake available here.

Any file can be overridden individually: `--labels`, `--cache`, `--groups`, `--folds`, `--oof`.

## Running

```bash
python experiments/exp2_shift_robust_calibration_bracket/run.py
python experiments/exp1_atypical_syndrome_structure/run.py --device cuda
python experiments/exp3_dinov3_projection_expert/run.py --device cuda
```

Exit code `0` when every declared gate passes, `2` when one does not. A gate failure is a result,
not a crash: it means the change did not earn promotion.

Each run writes one sanitized JSON contract to `DAT_OUTPUT_ROOT`, carrying metrics, gate outcomes
and a self-hash. Contracts contain no case-level values, no identifiers and no paths.

## Validating without any data

Every experiment ships a synthetic test that builds populations whose correct answer is known by
construction and asserts the measurement recovers it — including a case where the answer is "there
is nothing here", so a test cannot pass by always saying yes.

```bash
python experiments/exp1_atypical_syndrome_structure/test_synthetic.py
python experiments/exp2_shift_robust_calibration_bracket/test_synthetic.py
python experiments/exp3_dinov3_projection_expert/test_synthetic.py
```

## Checking the repository itself

```bash
python scripts/verify_release.py
python -m compileall -q experiments datcore scripts archive
python scripts/verify_release.py
```

The verifier passes both times: build artefacts are excluded from the walk, so the documented
command order is repeatable. After any deliberate edit, regenerate the manifest with
`python scripts/update_manifest.py`. It refuses to write if the privacy checks fail, so the manifest
cannot be used to bless a leak.

## The archived research

`archive/phase_research/` holds the original Phase57–67 notebook-cell exports. They were written for
a hosted notebook environment, hardcode cloud-style paths, and depend on private artefacts that are
not distributed. They are kept as history and are not expected to run here. See
[`../archive/README.md`](../archive/README.md).
