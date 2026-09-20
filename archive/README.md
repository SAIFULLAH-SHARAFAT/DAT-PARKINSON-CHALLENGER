# Archived phase research

The original Phase57–67 sources, as exported from the notebook cells that produced the competition
results. They are kept because the experiment ledger in
[`../docs/EXPERIMENT_HISTORY.md`](../docs/EXPERIMENT_HISTORY.md) refers to them and because a
negative result is only citable if the code that produced it still exists.

**They are not expected to run here.** Specifically:

- They were written for a hosted notebook environment and hardcode cloud-style working paths.
- They depend on private artefacts — fold contracts, OOF vectors, checkpoints, a submission archive
  — that are not distributed and never will be.
- They are notebook-cell exports, so most have no `if __name__ == "__main__"` guard and expect to
  be executed top to bottom with variables already in scope.
- `phase67_common.py` is a shared module, so the Phase67 cells are not standalone; run them with
  this directory importable.

The supported code lives in [`../datcore/`](../datcore/) and
[`../experiments/`](../experiments/), is configured for a local workstation, and carries tests that
run with no data at all.

## What is in here

| pattern | phase | subject |
|---|---|---|
| `phase57_*` | 57 | transport reset, scanner-robust architecture, engine pilot, anchor-preserving optimisation |
| `phase58_*` | 58 | gradient-pressure audit, group-centred pseudo-residuals |
| `phase59_*` | 59 | out-of-fold failure anatomy, calibration ceiling |
| `phase60_*` | 60 | independent bilateral 3D classifier |
| `phase61_*` | 61 | historical complementarity, fixed Phase31 blend stability |
| `phase62_*` | 62 | acquisition-domain validation reset, multitemplate expert |
| `phase63_*` | 63 | sparse confidence-risk ceiling |
| `phase64_*` | 64 | group-blocked diffusion / Nyström expert |
| `phase65_cell_165a_*`, `phase65_cell_165ar*` | 65A | deployment fallback, archive provenance, offline runtime, Torch Hub locality |
| `phase65_cell_165b/c/d_*` | 65B–D | physics-synthetic pretraining, domain adversary, repeated-group confirmation |
| `phase66_*`, `phase66f_*` | 66 / 66F | DINOv2 volume adaptation, weight diagnostic, final dual-GPU run |
| `phase67_common.py`, `phase67_cell_167*` | 67 | shared module and five cells, **unreviewed work in progress** |
| `test_phase6*_synthetic.py`, `test_phase65b_static_contract.py` | — | synthetic and static contract checks |
| `PHASE66_RESEARCH.md` | 66 | literature review, superseded by the Phase66F result |

## Phase67 carries known open defects

Recorded here so nobody reads a Phase67 contract and believes more than it says:

1. `167A`'s cache-parity recomputation cannot run without an operator-supplied preprocessing
   callable and NIfTI root. Until it does, every cache-trained candidate stays deployment-blocked,
   and both `167C` and `167E` gate on that verdict.
2. `167E`'s per-seed criterion compares raw per-seed expert probabilities against the anchor while
   the candidate is the blend, so it cannot be satisfied. The contract reports the mismatch rather
   than emitting an unreachable `False`; choosing the correct statistic is a research decision that
   has not been made.
3. `167A` and `167D` have no tests. `167E` is only partially covered.

## Running the archived tests

Two of them need real symlinks to reproduce the official runtime layout, so they skip with an
explicit message on a platform that refuses symlink creation (Windows without Developer Mode). One
targets a cell that was never part of the public release and says so. The rest run:

```bash
python archive/phase_research/test_phase64_synthetic.py
python archive/phase_research/test_phase65b_static_contract.py
python archive/phase_research/test_phase67b_synthetic.py
python archive/phase_research/test_phase67c_synthetic.py
```
