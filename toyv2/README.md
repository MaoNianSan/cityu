# toyv2

Controlled preliminary simulation package for Prediction-Powered Inference (PPI).

The current experiment compares **Classic**, **Naive ML**, **PPI**, and **PPI++** under three scenarios:

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

For each scenario, let \(z(X)\) be the true conditional mean. The proxy output is  $f(X)=z(X)+e(X).$

- `P1`: \(e=0.02U\),   $ U \sim \mathrm{U}(-1,1)$
- `P2`: \(e=0.06\);
- `P3`: \(e=0.06 $ (2X_2-1) \in \{-0.06,0.06\} $ ) This form can create +/- e value equally.
- `P4`: \(e=0.20U\).

## 2. PPI Methods

These are controlled pseudo learner outputs, not separately trained ML algorithms.

For this part, we use the parameter name uesed in the code

For each scenario, the labelled and unlabelled covariates are

$$
x_{\mathrm{labeled}}=\texttt{data.x\_labeled},
\qquad
x_{\mathrm{unlabeled}}=\texttt{data.x\_unlabeled}.
$$

The labelled outcome is

$$
y_{\mathrm{labeled}}=\texttt{data.outcome\_for(scenario)}.
$$

The controlled proxy outputs are

$$
f_{\mathrm{labeled}}=\texttt{prediction.f\_labeled},
\qquad
f_{\mathrm{unlabeled}}=\texttt{prediction.f\_unlabeled}.
$$

The labelled and unlabelled sample sizes are

$$
n=\texttt{x\_labeled.shape[0]},
\qquad
N=\texttt{x\_unlabeled.shape[0]}.
$$

### PPI

PPI uses prediction outputs on the large unlabelled sample and corrects the prediction bias using the labelled sample.

For mean estimation, the PPI estimator is

$$
\widehat\theta_{\mathrm{PPI}}
=
\frac{1}{n}\sum_{i=1}^{n} y_{\mathrm{labeled},i}
+
\left(
\frac{1}{N}\sum_{j=1}^{N} f_{\mathrm{unlabeled},j}
-
\frac{1}{n}\sum_{i=1}^{n} f_{\mathrm{labeled},i}
\right).
$$

The corresponding variance estimator is

$$
\widehat{\operatorname{Var}}
(\widehat\theta_{\mathrm{PPI}})
=
\frac{
\widehat{\operatorname{Var}}
(y_{\mathrm{labeled}}-f_{\mathrm{labeled}})
}{n}
+
\frac{
\widehat{\operatorname{Var}}
(f_{\mathrm{unlabeled}})
}{N}.
$$


For linear regression and logistic GLM, the same idea is applied at the estimating-equation level.

For linear regression, PPI solves

$$
0
=
\frac{1}{N}
x_{\mathrm{unlabeled}}^\top
\left(
x_{\mathrm{unlabeled}}\beta
-
f_{\mathrm{unlabeled}}
\right)
+
\frac{1}{n}
x_{\mathrm{labeled}}^\top
\left(
f_{\mathrm{labeled}}
-
y_{\mathrm{labeled}}
\right).
$$

For logistic GLM, with

$$
\mu_\beta(x)=\operatorname{expit}(x^\top\beta),
$$

PPI solves

$$
0
=
\frac{1}{N}
x_{\mathrm{unlabeled}}^\top
\left(
\mu_\beta(x_{\mathrm{unlabeled}})
-
f_{\mathrm{unlabeled}}
\right)
+
\frac{1}{n}
x_{\mathrm{labeled}}^\top
\left(
f_{\mathrm{labeled}}
-
y_{\mathrm{labeled}}
\right).
$$

### PPI++V2

The toyv2 implementation of PPI++ is a variance-minimizing weighted PPI variant.
It evaluates a family of estimators indexed by

$$
\texttt{lambda\_}\in[0,1].
$$

The two boundary cases are

$$
\texttt{lambda\_}=0
\Rightarrow
\text{Classic},
\qquad
\texttt{lambda\_}=1
\Rightarrow
\text{PPI}.
$$

For mean estimation, the weighted estimator is

$$
\widehat\theta_{\lambda}
=
\frac{1}{n}\sum_{i=1}^{n} y_{\mathrm{labeled},i}
+
\lambda
\left(
\frac{1}{N}\sum_{j=1}^{N} f_{\mathrm{unlabeled},j}
-
\frac{1}{n}\sum_{i=1}^{n} f_{\mathrm{labeled},i}
\right).
$$

The corresponding variance estimator is

$$
\widehat{\operatorname{Var}}
(\widehat\theta_{\lambda})
=
\frac{
\widehat{\operatorname{Var}}
(y_{\mathrm{labeled}}-\lambda f_{\mathrm{labeled}})
}{n}
+
\lambda^2
\frac{
\widehat{\operatorname{Var}}
(f_{\mathrm{unlabeled}})
}{N}.
$$

For linear regression, the weighted estimating equation is

$$
0
=
\frac{1}{n}
x_{\mathrm{labeled}}^\top
\left[
(1-\lambda)x_{\mathrm{labeled}}\beta
-
y_{\mathrm{labeled}}
+
\lambda f_{\mathrm{labeled}}
\right]
+
\frac{\lambda}{N}
x_{\mathrm{unlabeled}}^\top
\left(
x_{\mathrm{unlabeled}}\beta
-
f_{\mathrm{unlabeled}}
\right).
$$

Equivalently, the code constructs

$$
\texttt{hessian}
=
(1-\lambda)
\frac{x_{\mathrm{labeled}}^\top x_{\mathrm{labeled}}}{n}
+
\lambda
\frac{x_{\mathrm{unlabeled}}^\top x_{\mathrm{unlabeled}}}{N},
$$

and

$$
\texttt{rhs}
=
\frac{
x_{\mathrm{labeled}}^\top
(y_{\mathrm{labeled}}-\lambda f_{\mathrm{labeled}})
}{n}
+
\lambda
\frac{
x_{\mathrm{unlabeled}}^\top f_{\mathrm{unlabeled}}
}{N}.
$$

The estimate is obtained by solving

$$
\widehat\beta_{\lambda}
=
\texttt{hessian}^{-1}
\texttt{rhs}.
$$

For logistic GLM, the weighted estimating equation is

$$
0
=
\frac{1}{n}
x_{\mathrm{labeled}}^\top
\left[
(1-\lambda)\mu_\beta(x_{\mathrm{labeled}})
-
y_{\mathrm{labeled}}
+
\lambda f_{\mathrm{labeled}}
\right]
+
\frac{\lambda}{N}
x_{\mathrm{unlabeled}}^\top
\left[
\mu_\beta(x_{\mathrm{unlabeled}})
-
f_{\mathrm{unlabeled}}
\right].
$$

For each candidate `lambda_`, the code computes a sandwich covariance matrix,

$$
\texttt{covariance}
=
\texttt{hessian}^{-1}
\texttt{omega}
\texttt{hessian}^{-T},
$$

where `omega` is formed from `psi_labeled` and `psi_unlabeled`.

PPI++V2 selects the value of `lambda_` that minimizes the trace of this covariance matrix over the grid specified by `PPI_PLUS_PLUS_LAMBDA_GRID` in `config.py`:

$$
\widehat{\lambda}
=
\arg\min_{\lambda\in\texttt{PPI\_PLUS\_PLUS\_LAMBDA\_GRID}}
\operatorname{tr}
\left(
\texttt{covariance}_{\lambda}
\right).
$$

This is a vector-level variance criterion. It can reduce the overall sandwich covariance trace, but it does not guarantee a shorter confidence interval for every individual coefficient in every replicate.

This implementation is not the paper-based standard PPI++ covariance-to-variance-ratio rule. It is retained as a separate variance-minimizing variant and is therefore referred to as **PPI++V2**.

$$
\lambda=0\Rightarrow\text{Classic},
\qquad
\lambda=1\Rightarrow\text{PPI}.
$$

## 3. File structure

```text
config.py             # Numerical parameters, run modes, and worker count
formulation.py         # Scenario definitions, targets, and outcome contracts
data_generation.py     # Paired labelled/unlabelled data generation and input caching, but the exact generation form depend on formulation.py
learner_proxy.py       # P1--P4 controlled proxy outputs
baselines.py           # Classic and Naive ML inference
ppi.py                 # Standard PPI
ppiplusplus.py         # PPI++
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
- `other/diagnostics_seed0.parquet`: convergence, Hessian condition number, and PPI++ lambda diagnostics.

If `pyarrow` is unavailable, the two detailed audit files are written as compressed `.csv.gz` files instead; aggregate result tables are always CSV.

### Figure contract

Each scenario produces two figure types:

- `*_main_cl_950`: the main 95% figure. Its left column reports empirical coverage for Classic, Naive ML, PPI, and PPI++. Its right column reports CI width relative to Classic for PPI and PPI++ only. Naive ML is excluded from the relative-width comparison because a near-zero interval with invalid coverage is not inferential efficiency.
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

