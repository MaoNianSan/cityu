# toyv3

Controlled preliminary simulation package for Prediction-Powered Inference (PPI).

The current experiment compares **Classic**, **Naive ML**, **PPI**, **PPI++V1**, and **PPI++V2** under three scenarios:

1. mean estimation;
2. linear regression;
3. logistic GLM.

Cross-PPI is retained only as an interface in `cross_ppi.py` and is not run in the current preliminary experiment.

## 1. Fixed experiment contract

### Data generation

$$
X=(1,X_1,X_2)^\top,
\qquad X_1\sim\mathrm{Uniform}(-1,1),
\qquad X_2\sim\mathrm{Bernoulli}(0.5).
$$

The common linear signal is

$$
\eta(X)=0.25+0.9X_1-0.6X_2.
$$

- Mean estimation / LR:
  
  $$
  Y=\eta(X)+\epsilon,\qquad \epsilon\sim N(0,1).
  $$
- **Logistic GLM:**
  
  $$
  Y\mid X\sim\mathrm{Bernoulli}\{\operatorname{expit}(\eta(X))\}.
  $$

The default data sizes are `n=60` labelled observations and `N=600` unlabelled observations.

### Controlled proxy-output regimes

For Gaussian mean estimation and LR, the additive proxy is \(f(X)=z(X)+e(X)\), and P1/P4 share one \(Z\sim N(0,1)\) realization:

- `P1`: \(e=0.02Z\);
- `P2`: \(e=0.06\);
- `P3`: $e=0.06(2X_2-1)$;
- `P4`: \(e=0.20Z\).

Hence Gaussian P4 error is pointwise ten times P1 error, and 0.02/0.20 are standard-deviation scales. For logistic GLM,

\[
f_{P1}(X)=\operatorname{expit}\{\eta(X)+0.02Z\},\qquad
f_{P4}(X)=\operatorname{expit}\{\eta(X)+0.20Z\}.
\]

The reported logistic error is still \(f(X)-z(X)\). No clipping is used. The probability-scale errors need not have a 10:1 ratio, but the latent perturbations do; P2/P3 retain their existing definitions.

These are controlled pseudo learner outputs, not separately trained ML algorithms.

### Inference methods

1. **Classic:** uses only labeled gold data.
2. **Naive ML:** treats learner predictions on the unlabeled sample as true labels.
3. **PPI:** standard prediction-powered inference with fixed \(\lambda=1\).
4. **PPI++V1:** calls the official `ppi-python` package directly with `lam=None`; this is the package-standard PPI++ reference.
5. **PPI++V2:** internal transparent implementation; searches over a fixed lambda grid and selects the lambda minimizing the plug-in sandwich covariance trace.

For PPI++V2, the selected scalar \(\lambda\in[0,1]\) is chosen per replicate/scenario/profile over the grid in `config.py`. This is a vector-level criterion: it does not guarantee a smaller interval for every individual coefficient in every replicate.

$$
\lambda=0\Rightarrow\text{Classic},
\qquad
\lambda=1\Rightarrow\text{PPI}.
$$

Implementation note:

- PPI++V1 calls `ppi-python` separately at 90%, 95%, and 97.5%; formal coverage and width use these direct package intervals.
- A diagonal covariance reconstructed from the direct 95% interval remains only for compatibility and diagnostics.
- With `lam=None`, package-internal lambda is not exposed and no numeric replicate-level lambda is claimed.
- PPI++V2 keeps full internal diagnostics for lambda grid, covariance trace, and selected lambda.
- V1 and V2 are not expected to be numerically identical, because their tuning and covariance construction differ.

## 2. PPI Methods

### 2.1 PPI

For mean estimation, PPI uses

$$
\widehat\theta_{\mathrm{PPI}}
=
\frac{1}{n}\sum_{i=1}^{n}Y_i
+
\left(
\frac{1}{N}\sum_{j=1}^{N}f(X_j^u)
-
\frac{1}{n}\sum_{i=1}^{n}f(X_i)
\right).
$$

The first term uses labelled data. The bracketed term adds prediction information from the unlabelled sample and corrects it using the labelled sample.

For linear regression and logistic GLM, the same idea is implemented at the estimating-equation level.

### 2.2 PPI++V1

PPI++V1 can be written as

$$
\widehat\theta_{\mathrm{PPI++V1}}
=
\frac{1}{n}\sum_{i=1}^{n}Y_i
+
\widehat\lambda_{\mathrm{py}}
\left(
\frac{1}{N}\sum_{j=1}^{N}f(X_j^u)
-
\frac{1}{n}\sum_{i=1}^{n}f(X_i)
\right),
$$

where $\widehat\lambda_{\mathrm{py}}$ is the package-selected power-tuning parameter returned internally when `lam=None`.

For a fixed $\lambda$, the weighted estimating equation has the form

$$
0
=
\frac{1}{n}
\sum_{i=1}^{n}
\left\{
\psi_i^Y(\theta)
-
\lambda \psi_i^{\hat Y}(\theta)
\right\}
+
\frac{\lambda}{N}
\sum_{j=1}^{N}
\psi_j^{\hat Y,U}(\theta).
$$

PPI++V1 chooses $\lambda$ using the package's internal optimal-lambda rule. Define the centered score matrices

$$
G_c = G-\bar G,
\qquad
\widehat G_c = \widehat G-\bar{\widehat G},
$$

where $G$ collects the labelled true-label score contributions $\psi_i^Y$, and $\widehat G$ collects the labelled prediction score contributions $\psi_i^{\hat Y}$. The package forms

$$
\widehat C
=
\frac{1}{n}
\left(
G_c^\top \widehat G_c
+
\widehat G_c^\top G_c
\right).
$$

It also forms a prediction-score covariance matrix by combining the labelled and unlabelled prediction-score contributions:

$$
\widehat S
=
\widehat{\operatorname{Cov}}
\left(
\begin{bmatrix}
\widehat G \\
\widehat G^{U}
\end{bmatrix}
\right),
$$

where $\widehat G^{U}$ collects the unlabelled prediction-powered score contributions $\psi_j^{\hat Y,U}$.

Let $\widehat H^{-1}$ be the inverse empirical Hessian or Jacobian used for the corresponding estimating equation. With `coord=None`, PPI++V1 optimizes the total variance over all coordinates and computes

$$
\widehat\lambda_{\mathrm{py}}
=
\Pi_{[0,1]}
\left[
\frac{
\operatorname{tr}
\left(
\widehat H^{-1}
\widehat C
\widehat H^{-T}
\right)
}{
2(1+n/N)
\operatorname{tr}
\left(
\widehat H^{-1}
\widehat S
\widehat H^{-T}
\right)
}
\right],
$$

where $\Pi_{[0,1]}(\cdot)$ denotes clipping to the interval $[0,1]$.

For mean estimation, $\widehat H^{-1}=I$. In this case, the score contributions reduce to centered versions of $Y_i$, $f(X_i)$, and $f(X_j^u)$ around the preliminary PPI point estimate.

For linear regression and logistic GLM, the same formula is applied to the corresponding score or gradient contributions.

The package returns marginal confidence intervals rather than a full covariance matrix. To keep the same result interface as the other methods, toyv3 reconstructs a diagonal covariance matrix from the returned marginal intervals. If the package returns the interval $[L_k,U_k]$ for coefficient $k$ at confidence level $c$, then

$$
z_c
=
\Phi^{-1}
\left(
\frac{1+c}{2}
\right),
$$

$$
\widehat{\mathrm{se}}_k
=
\frac{U_k-L_k}{2z_c},
$$

and toyv3 constructs

$$
\widehat V_{\mathrm{PPI++V1}}
=
\operatorname{diag}
\left(
\widehat{\mathrm{se}}_1^2,
\ldots,
\widehat{\mathrm{se}}_p^2
\right).
$$

This diagonal covariance is only an interface reconstruction and is not used for formal CI width or coverage. The 90%, 95%, and 97.5% intervals are requested directly using two-sided alpha 0.10, 0.05, and 0.025 respectively.

### 2.3 PPI++V2

PPI++V2 is the internal variance-minimizing weighted PPI implementation used in this toy experiment. It is not the paper-based standard PPI++ covariance-to-variance-ratio rule.

PPI++V2 considers a weighted PPI estimator indexed by $\lambda\in[0,1]$:

$$
\widehat\theta_{\lambda}
=
\frac{1}{n}\sum_{i=1}^{n}Y_i
+
\lambda
\left(
\frac{1}{N}\sum_{j=1}^{N}f(X_j^u)
-
\frac{1}{n}\sum_{i=1}^{n}f(X_i)
\right).
$$

The boundary cases are

$$
\lambda=0\Rightarrow\text{Classic},
\qquad
\lambda=1\Rightarrow\text{PPI}.
$$

For a fixed $\lambda$, the variance estimator in the mean-estimation case is

$$
\widehat{\operatorname{Var}}(\widehat\theta_{\lambda})
=
\frac{
\widehat{\operatorname{Var}}\{Y_i-\lambda f(X_i)\}_{i=1}^{n}
}{n}
+
\lambda^2
\frac{
\widehat{\operatorname{Var}}\{f(X_j^u)\}_{j=1}^{N}
}{N}.
$$

The first term is the labelled correction uncertainty. It contains the variance of the true labels, the variance of the learner output, and the covariance between the true label and the learner output, because

$$
\operatorname{Var}\{Y-\lambda f(X)\}
=
\operatorname{Var}(Y)
+
\lambda^2\operatorname{Var}\{f(X)\}
-
2\lambda\operatorname{Cov}\{Y,f(X)\}.
$$

The second term is the uncertainty from the unlabelled prediction-powered component. It depends on the variation of $f(X^u)$ on the unlabelled sample and is scaled by $\lambda^2$.

For linear regression and logistic GLM, the same idea is implemented at the estimating-equation level. For each fixed $\lambda$, toyv3 first solves the corresponding weighted estimating equation and then computes a sandwich covariance matrix,

$$
\widehat V_\lambda
=
\widehat H_\lambda^{-1}
\widehat\Omega_\lambda
\widehat H_\lambda^{-T}.
$$

Here, $\widehat H_\lambda$ is the empirical Hessian or Jacobian of the weighted estimating equation. The middle term $\widehat\Omega_\lambda$ collects the variation from the labelled correction contribution and the unlabelled prediction-powered contribution.

PPI++V2 searches over the grid $\Lambda=\texttt{PPI\_PLUS\_PLUS\_V2\_LAMBDA\_GRID}$ and selects

$$
\widehat{\lambda}
=
\arg\min_{\lambda\in\Lambda}
\operatorname{tr}
\left(
\widehat V_\lambda
\right).
$$

For mean estimation, $\operatorname{tr}(\widehat V_\lambda)$ is the scalar estimated variance. For linear regression and logistic GLM, $\widehat V_\lambda$ is a covariance matrix for all coefficients, so the trace criterion minimizes the sum of estimated coefficient variances.

This is a vector-level variance criterion. It can reduce the overall covariance trace, but it does not guarantee a shorter confidence interval for every individual coefficient in every replicate.

## 3. File structure

```text
config.py             # Numerical parameters, run modes, and worker count
formulation.py         # Scenario definitions, targets, and outcome contracts
data_generation.py     # Paired labelled/unlabelled data generation and input caching, but the exact generation form depend on formulation.py
learner_proxy.py       # P1--P4 controlled proxy outputs
baselines.py           # Classic and Naive ML inference
ppi.py                 # Standard PPI
ppiplusplus.py         # PPI++V1 official-package wrapper and PPI++V2 internal grid search
cross_ppi.py           # Cross-PPI
plotting.py            # 
main.py                # Simulation orchestration and output writing
checks.py              # Mathematical and numerical preflight checks
ipy/result_audit.ipynb # Read-only audit of fast/full outputs
tests/test_core.py      # Lightweight regression tests for core numerical guards
```

## 4. Installation

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 5. Run modes

```bash
python main.py --mode fast
python main.py --mode full

# Set WORKERS in config.py to control the number of replicate processes.
```

All numerical values and the CPU allocation (`WORKERS`) are controlled in `config.py`.
`WORKERS=1` is serial execution. Because each replicate has its own deterministic
random-number streams, changing `WORKERS` changes runtime but not the generated
data, point estimates, intervals, or aggregate tables.

- **fast:** `R=200`, seed `0`; saves full replicate records and generated input files for inspection.
- **full:** `R=2000`, seeds `0`--`29`; seed `0` is the primary result and seeds `1`--`29` are used for robustness summaries. Detailed replicate records and generated input files are retained only for seed `0`.

Optional debugging switches:

```bash
python main.py --mode fast --skip-plots
python main.py --mode fast --skip-checks
```

## 6. Outputs

```text
output/
├── fast/
│   ├── figure/
│   ├── table/
│   └── other/
└── full/
    ├── figure/
    ├── table/
    └── other/
```

Primary displayed metrics are:

$$
\text{Average CI Width}
\qquad\text{and}\qquad
\text{Empirical Coverage}.
$$

- `fast/table/metrics_seed0.csv`: seed-0 aggregate metrics;
- `full/table/seed0_metrics.csv`: seed-0 primary metrics;
- `full/table/all_seed_metrics.csv`: aggregate metrics for seeds 0--29;
- `full/table/robustness_summary.csv`: seed-0 values plus seed 1--29 median, IQR, min, and max;
- `other/replicate_results_seed0.parquet`: seed-0 interval-level audit records;
- `other/diagnostics_seed0.parquet`: convergence, Hessian condition number, and PPI++V1/PPI++V2 diagnostics.

If `pyarrow` is unavailable, the two detailed audit files are written as compressed `.csv.gz` files instead; aggregate result tables are always CSV.

### Result display and interpretation

The outputs are evaluated mainly by empirical coverage and confidence-interval width.

A useful PPI-family method should maintain empirical coverage close to the nominal confidence level while reducing CI width relative to Classic inference.

The main 95% figures report empirical coverage and relative CI width. The calibration figures report empirical coverage minus nominal coverage at 90%, 95%, and 97.5% confidence levels.

Naive ML is included only as a diagnostic baseline. It is excluded from the relative-width comparison because an extremely narrow interval with invalid coverage is not meaningful inferential efficiency.

### Figure contract

Each scenario produces two figure types:

- `*_main_cl_950`: the main 95% figure. Its left column reports empirical coverage for Classic, Naive ML, PPI, PPI++V1, and PPI++V2. Its right column reports CI width relative to Classic for PPI, PPI++V1, and PPI++V2 only. Naive ML is excluded from the relative-width comparison because a near-zero interval with invalid coverage is not inferential efficiency.
- `*_calibration`: appendix-oriented calibration panels for 90%, 95%, and 97.5% confidence levels. They plot empirical coverage minus nominal coverage by P1--P4. Extreme Naive-ML failures are represented by boundary triangles so that the calibration of valid methods remains visible; uncensored values remain in `table/plot_summary.csv`.

The grey reference regions/bars are 95% binomial Monte-Carlo ranges under exact calibration. Fast-mode figures show seed 0 with `R=200`. Full-mode figures show the median across all outer seeds, with vertical IQR bars computed from that same seed population. `plot_summary.csv` records the exact aggregate quantities used for plotting, including uncensored coverage error and relative CI width.

## 7. Integrity checks

`checks.py` runs before `main.py` by default. It verifies, among other conditions:

- deterministic regeneration under the same seed;
- `P2` and `P3` equal pointwise squared error;
- the shared P1/P4 random component and 10× amplitude ratio;
- logistic pseudo outcomes remain in `[0,1]`;
- `lambda=0` exactly recovers Classic;
- `lambda=1` exactly recovers PPI;
- boundary coverage is stable when an analytically exact interval has only floating-point-scale width.

## 8. Notebook scope

`ipy/result_audit.ipynb` only reads existing `output/fast/` and `output/full/` files. It does not generate data, refit estimators, or overwrite formal output.
