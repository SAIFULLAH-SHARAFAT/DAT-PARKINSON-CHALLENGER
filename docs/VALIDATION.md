# Release validation

The public release was validated on 9 September 2026.

## Passed locally

- Python 3.12 syntax compilation for every file in `experiments/` and `scripts/`.
- Phase64 synthetic group-blocked diffusion/Nyström test.
- Phase65A-R2 synthetic provenance/replay acceptance and rejection paths.
- Phase65A-R3 synthetic static, offline, exact-parity, and mismatch paths.
- Phase65A-R4 synthetic corrected-adjudication acceptance and rejection paths.
- Phase65B static contract test.
- Release manifest, excluded-extension, size, secret-pattern, and user-path scan.

## Not rerun in the release-building environment

The Phase65 and Phase65B synthetic training tests import PyTorch. PyTorch was not installed in the release-building environment, so those tests were syntax-compiled but not executed there. Their historical validated status is described in the experiment record. Run them in the declared Kaggle/PyTorch environment:

```bash
python experiments/test_phase65_synthetic.py
python experiments/test_phase65b_standalone_synthetic.py
```

No private data, private artifacts, model weights, or leaderboard submissions were accessed while assembling this release. Consequently, public-release validation does not recompute any private or leaderboard metric.

