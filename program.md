# Autoresearch Program

Goal: reduce the `score` produced by `scripts/autoresearch_eval.sh`.

Interpretation:
- Lower `score` is better.
- Primary metric is `benchmark_rmse_mean_current`.
- A run gets a large penalty if `strict_holdout_passed=false`.
- A run gets an extra penalty if the benchmark gate fails.
- Complexity is only a tiny tie-break.

Fixed evaluation command:

```bash
scripts/autoresearch_eval.sh
```

What this runs:
- `cargo run -p threebody-cli -- quickstart --profile autoresearch`
- Numeric-only quickstart profile.
- Deterministic benchmark-first evaluation.
- Score artifacts written to the run directory and one row appended to `results.tsv`.

Allowed edit scope:
- `threebody-discover/src/library.rs`
- `threebody-discover/src/sparse.rs`
- `threebody-discover/src/ga.rs`
- `threebody-cli/src/eval/`

Do not edit:
- `program.md`
- `scripts/autoresearch_eval.sh`
- quickstart scoring or benchmark-harness code in `threebody-cli/src/main.rs`
- `results.tsv`

Working rules:
1. Make one small change at a time.
2. Run `scripts/autoresearch_eval.sh`.
3. Keep the change only if the new row in `results.tsv` has a lower `score`.
4. Prefer changes that improve benchmark RMSE without increasing complexity much.
5. Do not optimize by weakening the benchmark or editing the evaluator.

Useful artifacts after each run:
- `results.tsv`
- `results/<run>/RESULTS.md`
- `results/<run>/autoresearch_score.json`
- `results/<run>/factory/benchmark_eval.json`
- `results/<run>/factory/strict_holdout_report.json`
