# Fast-run audit — 2026-07-07

## Scope

Audit target: the completed `fast` run (`seed=0`, `R=200`) and the current toy package. The audit covered output completeness, estimator execution, interval construction, PPI++ selection diagnostics, input-cache reproducibility, plotting, and serial/parallel invariance.

## Verified after correction

- Aggregate output has the expected **273** rows; interval-level output has **54,600** rows; diagnostic output has **7,800** rows. There are no duplicate keys.
- All 7,800 estimator calls converged. No non-finite estimates, standard errors, or interval endpoints; no negative standard errors or widths; no reversed intervals.
- Re-aggregation from interval-level records reproduces the saved mean CI widths to machine precision and empirical coverages exactly.
- Every PPI++ diagnostic has all 21 grid candidates, and the recorded `lambda_hat` minimizes the stored covariance-trace criterion.
- Every saved PPI++ record with `lambda_hat=0` equals its Classic counterpart exactly. This verifies the endpoint implementation in the produced output.
- All 200 cached inputs regenerate exactly from their `(seed, replicate_id, stream_id)` specification. The proxy checks hold across all cached replicates: P2/P3 have equal pointwise squared error; P1/P4 share the same underlying random component; logistic pseudo outcomes remain in `[0,1]`.
- A serial (`WORKERS=1`) and parallel (`WORKERS=2`) test with the same 12-replicate design produced identical metric, interval, diagnostic, and input-manifest files.

## Corrections made before regenerating fast output

1. **Coverage boundary instability — substantive correction.**
   Some Linear Regression Naive-ML cells have analytically exact coefficients and numerically zero standard errors. The previous condition `lower <= truth <= upper` made their coverage depend on round-off at approximately `1e-16`; the same mathematical interval could be reported as covered or uncovered across numerical environments. A floating-point-scale boundary tolerance is now used. The fast outputs were regenerated after this change.

2. **Fast figure metadata and layout.**
   Fast figures no longer claim to display seed 1--29 IQR. The title and legend no longer overlap. P1--P4 are plotted as categorical regimes rather than connected trajectories, avoiding an artificial monotonicity implication.

3. **Portable input manifest.**
   Cached-file paths are now written in POSIX form, instead of inheriting Windows path separators.

4. **CPU control.**
   `config.WORKERS` now controls the number of independent replicate processes. It defaults to `1`; changing it preserves random streams and output ordering.

5. **Audit notebook and tests.**
   The notebook no longer treats fast and full aggregate tables as values that should match exactly, because they use different `R`. It now checks whether full seed 0 reproduces the first fast-run replicate records. A lightweight `unittest` suite covers preflight and the numerical coverage guard.

## Interpretation constraints remaining

- `fast` is a debugging/sanity-check output, not evidence for final comparative claims. With `R=200`, the Monte Carlo standard error of a nominal 95% coverage estimate is approximately 0.015; apparent deviations of a few percentage points are not decisive.
- PPI++ currently tunes **one scalar lambda by covariance trace**. This is a vector-level objective. It need not make every coefficient's interval shorter in every replicate, even though it includes Classic (`lambda=0`) among candidates.
- The current `fit_ppi` implementation is the unweighted, `lambda=1` Wald-estimation path. If the manuscript intends to compare against the original test-inversion PPI confidence set, relabel this method as `PPI (lambda=1, Wald)` or separately implement the original confidence-set construction.
- The present LR and logistic DGPs make the learner proxy the conditional mean of the same covariates used by the correctly specified inferential model. Under this design, a first-order efficiency gain over the correctly specified labelled-only regression is not generally expected. Thus, the weak LR/GLM gain is structurally consistent with the DGP, not necessarily an implementation failure. If the toy experiment is intended to demonstrate regression/GLM efficiency gains, introduce predictive information available to the learner but not fully represented in the inferential covariates, or redefine the estimand accordingly.
- `ACTIVE_METHODS` is currently a descriptive configuration field; `main.py` still executes the fixed four-method contract. Do not rely on changing that field to disable a method until the orchestration is made method-selective.

## Release status

The regenerated fast package is internally consistent for its stated controlled experiment. Do not start the full run until the intended interpretation of the LR/GLM scenarios and the `PPI` method label have been fixed in the experimental protocol.

## Post-audit display revision — 2026-07-07

The fast simulation was re-rendered with a revised figure contract; no data-generating, estimator, interval, or aggregation code was changed.

- Main figures are now limited to the 95% confidence level and pair empirical coverage with CI width relative to Classic.
- Naive ML remains in coverage panels but is excluded from relative-width panels, because invalid near-zero intervals do not constitute inferential efficiency.
- All three nominal levels (90%, 95%, 97.5%) are consolidated into calibration figures that show empirical coverage minus nominal coverage by proxy regime. Catastrophic off-scale Naive-ML failures are marked with boundary triangles so calibrated methods remain readable; uncensored values remain in `output/fast/table/plot_summary.csv`.
- The full-mode plotting path now computes point summaries and IQRs from the same all-seed metric population: median across seeds as the marker and cross-seed IQR as the vertical interval.
- The fast run was regenerated after introducing plotting configuration fields. The regenerated `metrics_seed0.csv` is byte-identical to the preceding audited fast metric table.
- `pytest -q` now runs correctly from the project root through `tests/conftest.py`; all three tests pass.
