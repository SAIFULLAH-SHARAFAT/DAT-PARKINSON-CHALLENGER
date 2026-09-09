# Script index

All scripts are under `experiments/`. They are preserved research cells, not a single installable training package.

| Phase | Files | Role | Recorded decision |
|---|---|---|---|
| 57 | `phase57_*` | Transport reset, scanner-robust architecture, engine pilot, residual optimization, fresh-kernel restore | Trained arms selected epoch-zero identity; retain anchor |
| 58 | `phase58_*` | Gradient-pressure audit and group-centered pseudo-residual pilot | Gains too small; reject |
| 59 | `phase59_*` | OOF failure anatomy and calibration ceiling | Calibration cannot close target gap |
| 60 | `phase60_*` | Independent bilateral 3D classifier | Monitor gain reversed on outer fold; reject |
| 61 | `phase61_*` | Historical complementarity and fixed Phase31 blend stability | Aggregate gain, failed stability/provenance gate |
| 62 | `phase62_*` | Acquisition-domain validation reset and multitemplate expert | Domain harm; reject |
| 63 | `phase63_*` | Sparse confidence-risk ceiling | Risk score did not locate confident errors; reject |
| 64 | `phase64_*` | Group-blocked diffusion/Nyström expert | Worse overall and by major domains; reject |
| 65A | `phase65_cell_165a_*` | Conservative deployment fallback audit | Retain Phase12c unknown-protocol fallback |
| 65A-R | `phase65_cell_165ar*` | Archive provenance, local Torch Hub, offline runtime, corrected output-path adjudication | Offline deployment integrity accepted; exact development parity unclaimed |
| 65B | `phase65_cell_165b_*` | Physics-synthetic pretraining gate | Reject |
| 65C | `phase65_cell_165c_*` | Class-conditional acquisition adversary | Small average gain, failed domain gate |
| 65D | `phase65_cell_165d_*` | Repeated whole-group confirmation | Five repeat wins, negative lower domain bound; reject |
| 66 | `phase66_final_*`, `phase66_weights_*` | Original fixed DINOv2 attempt and read-only checkpoint diagnostic | Setup/restart work only |
| 66F | `phase66f_fresh_*` | Fresh, source-consistent dual-GPU DINOv2 run | All six fits complete; materially worse; reject |

Files beginning with `test_` are synthetic or static contract checks. Some source files reconstruct private state by filename and digest; those dependencies are deliberately absent.

The failed Phase66R/R2/R3 continuation implementations are not shipped as runnable paths. Their failures and rationale are documented in `EXPERIMENT_HISTORY.md`; Phase66F is the superseding source-consistent implementation.

