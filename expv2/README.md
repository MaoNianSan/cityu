# Real-data EXP1: learner-type comparison within PPI

## 1. Experiment contract

This package compares LIN, DT, RF, and GB learners under one fixed real-data semi-supervised protocol. Learner quality is observed after training and is never assigned by perturbing predictions. The inference methods are Classic, Naive ML, PPI, PPI++V1, PPI++V2, and Cross-PPI.

Throughout this README:

- $X^{\mathrm{tar}}$ is the low-dimensional target-design matrix used to define the inferential parameter.
- $X^{\mathrm{learner}}$ is the richer feature set used to train a prediction model.
- $\widehat f(X^{\mathrm{learner}})$ is the learner prediction.
- $\theta^\star$ or $\beta^\star$ is the inferential target.

To maximize compatibility across GitHub Markdown, Jupyter Markdown, VS Code preview, and common MathJax/KaTeX renderers, all display equations use `$$ ... $$`, all inline equations use `$ ... $`, and raw column names remain in code formatting rather than inside LaTeX commands.

## 2. Datasets, targets, and learner quality

### 2.1 `a_mean`: Galaxy Zoo 2 mean estimation

The binary outcome is the column `t01_smooth_or_features_a01_smooth_flag`, coded as $Y_i \in \{0,1\}$. The empirical target is

$$
\theta^\star = \frac{1}{M}\sum_{i=1}^{M}Y_i.
$$

Here, $M$ is the complete cleaned dataset size. Its full-sample mean is used as the simulation-style empirical truth. There is no nontrivial target-design vector for mean estimation.

Learners use `ra`, `dec`, `sample`, `total_classifications`, and `total_votes`. Target-derived variables are excluded. For a binary learner, $\widehat f_i$ is a predicted probability.

Learner quality is measured on `PPI_inf` by the Brier score

$$
Q_{\mathrm{Brier}}(\widehat f)
=
\frac{1}{n}\sum_{i\in\mathcal I}
\left(Y_i-\widehat f_i\right)^2,
$$

where $\mathcal I$ denotes the ordinary-PPI inference subset and $n=|\mathcal I|$. Lower values indicate better predictive quality.

### 2.2 `b_lr`: ACS PUMS linear regression

The outcome is `PINCP`. Let $a_i$ denote `AGEP`, and let $s_i$ denote `SEX_MALE`, where `SEX_MALE` equals one when `SEX == 1` and zero otherwise. The target-design vector is

$$
x_i = (1,a_i,s_i)^\top.
$$

The empirical OLS target is

$$
\beta^\star
=
\underset{\beta}{\mathrm{arg\,min}}
\frac{1}{M}\sum_{i=1}^{M}
\left(Y_i-x_i^\top\beta\right)^2.
$$

Equivalently,

$$
\beta^\star
=
\left(X^\top X\right)^{+}X^\top Y
=
\left(\beta_0^\star,\beta_{\mathrm{AGEP}}^\star,\beta_{\mathrm{SEX}}^\star\right)^\top,
$$

where $(\cdot)^{+}$ denotes the Moore-Penrose pseudoinverse used by the code.

The target model uses only `AGEP` and `SEX_MALE`. The learner uses the richer covariate set specified in `config.py` to predict income. The main display contains the `AGEP` and `SEX_MALE` coefficients; the intercept remains in the result files.

Learner quality is measured on `PPI_inf` by

$$
Q_{\mathrm{MSE}}(\widehat f)
=
\frac{1}{n}\sum_{i\in\mathcal I}
\left(Y_i-\widehat f_i\right)^2.
$$

Lower values indicate better predictive quality.

### 2.3 `c_glm`: Adult logistic GLM

The binary outcome equals one when `income > 50000`. Let $a_i$, $e_i$, and $h_i$ denote `age`, `education-num`, and `hours-per-week`. The target-design vector is

$$
x_i=(1,a_i,e_i,h_i)^\top.
$$

The target probability model is

$$
\mu_\beta(x_i)
=
\frac{1}{1+\exp(-x_i^\top\beta)}.
$$

The empirical logistic target is

$$
\beta^\star
=
\underset{\beta}{\mathrm{arg\,min}}
\left[
-\frac{1}{M}\sum_{i=1}^{M}
\left\{
Y_i\log\mu_\beta(x_i)
+
(1-Y_i)\log\left(1-\mu_\beta(x_i)\right)
\right\}
\right].
$$

Equivalently, $\beta^\star$ solves

$$
\frac{1}{M}\sum_{i=1}^{M}
x_i\left(\mu_{\beta^\star}(x_i)-Y_i\right)=0.
$$

Learners use the complete numerical and categorical feature set in `config.py`. Learner quality is the Brier score defined in Section 2.1. The main display contains the three non-intercept coefficients.

## 3. Learner training protocol

| ID | Binary estimator | Continuous estimator | Numerical preprocessing | Categorical preprocessing | Tuned parameters |
|---|---|---|---|---|---|
| LIN | `LogisticRegression(max_iter=1000, solver="lbfgs")` | `Ridge(solver="lsqr")` | median imputation and standardization | most-frequent imputation and one-hot encoding; unknown categories ignored | `C` in `{0.1, 1, 10}` or `alpha` in `{0.1, 1, 10}` |
| DT | `DecisionTreeClassifier` | `DecisionTreeRegressor` | median imputation | most-frequent imputation and ordinal encoding; unknown value `-1` | `max_depth` in `{4, 8}` and `min_samples_leaf` in `{10, 50}` |
| RF | `RandomForestClassifier` | `RandomForestRegressor` | median imputation | most-frequent imputation and ordinal encoding; unknown value `-1` | `max_depth` in `{8, 12}` |
| GB | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` | median imputation | most-frequent imputation and ordinal encoding; unknown value `-1` | `max_iter` in `{50, 100}` |

RF fixes `n_jobs=1`, `max_features=0.7`, `max_samples=0.8`, `n_estimators=20`, and `min_samples_leaf=20`. GB fixes `learning_rate=0.1`, `max_leaf_nodes=15`, and `early_stopping=False`.

For learner class $a$ and candidate hyperparameter value $\eta$, the ordinary-PPI training subset is divided into fitting and validation subsets:

$$
\mathcal T
=
\mathcal F\cup\mathcal V,
\qquad
\mathcal F\cap\mathcal V=\varnothing,
\qquad
|\mathcal V|=0.25|\mathcal T|.
$$

The selected hyperparameter is

$$
\widehat\eta_a
=
\underset{\eta\in\mathcal H_a}{\mathrm{arg\,min}}
Q_{\mathcal V}(\widehat f_{a,\eta}).
$$

Each candidate is fitted only on $\mathcal F$. Validation Brier score or MSE selects the hyperparameters. The selected learner is then refitted on all of $\mathcal T$. Its formal split-learner quality is evaluated independently on $\mathcal I$.

PPI, PPI++V1, PPI++V2, and Naive ML share this split-trained learner.

For Cross-PPI, the labelled set is partitioned into $K=5$ folds:

$$
\mathcal L
=
\mathcal L_1\cup\cdots\cup\mathcal L_K.
$$

For fold $k$, tuning and refitting use only

$$
\mathcal L_{-k}=\mathcal L\setminus\mathcal L_k,
$$

and the holdout fold $\mathcal L_k$ receives predictions from a learner that was not trained on that fold. Every fold-specific learner also predicts every unlabelled observation, producing an $N\times K$ matrix.

Ordinary PPI trains one learner on 20% of the labelled observations. Each five-fold Cross-PPI learner trains on approximately 80% of the labelled observations. Their empirical difference therefore combines cross-fitting with a larger effective learner-training sample.

## 4. Sampling and inference formulas

### 4.1 Sampling notation

For each replicate:

- $\mathcal L$ is the labelled set, with size $n_L$.
- $\mathcal U$ is the unlabelled set, with size $N$.
- $\mathcal T$ is the ordinary-PPI learner-training subset.
- $\mathcal I$ is the ordinary-PPI inference subset, with size $n$.
- $\mathcal L=\mathcal T\cup\mathcal I$ and $\mathcal T\cap\mathcal I=\varnothing$.

The configured proportions are

$$
\frac{n_L}{M}=0.10,
\qquad
\frac{N}{M}=0.90,
\qquad
\frac{|\mathcal T|}{n_L}=0.20,
\qquad
\frac{n}{n_L}=0.80.
$$

For ordinary PPI, define

$$
\widehat f_i=\widehat f(X_i^{\mathrm{learner}}),
\quad i\in\mathcal I,
$$

and

$$
\widehat f_j^u=\widehat f(X_j^{u,\mathrm{learner}}),
\quad j\in\mathcal U.
$$

For regression targets, let $x_i$ and $x_j^u$ denote rows of the target-design matrices $X^{\mathrm{tar}}_{\mathcal I}$ and $X^{\mathrm{tar}}_{\mathcal U}$.

### 4.2 Classic and Naive ML

Classic inference uses all labelled observations, not only `PPI_inf`.

For mean estimation,

$$
\widehat\theta_{\mathrm{Classic}}
=
\frac{1}{n_L}\sum_{i\in\mathcal L}Y_i.
$$

For OLS and logistic GLM, Classic applies the target formulation to $(X^{\mathrm{tar}}_{\mathcal L},Y_{\mathcal L})$.

Naive ML treats unlabelled predictions as if they were observed outcomes. For mean estimation,

$$
\widehat\theta_{\mathrm{Naive}}
=
\frac{1}{N}\sum_{j\in\mathcal U}\widehat f_j^u.
$$

For OLS and logistic GLM, Naive ML applies the target formulation to $(X^{\mathrm{tar}}_{\mathcal U},\widehat f_{\mathcal U})$. It has no labelled correction and is used only as a validity diagnostic.

### 4.3 Internal PPI implementation

#### Mean estimation

The internal PPI mean estimator is

$$
\widehat\theta_{\mathrm{PPI}}
=
\frac{1}{N}\sum_{j\in\mathcal U}\widehat f_j^u
+
\frac{1}{n}\sum_{i\in\mathcal I}
\left(Y_i-\widehat f_i\right).
$$

Its implemented variance estimator is

$$
\widehat{\mathrm{Var}}
\left(\widehat\theta_{\mathrm{PPI}}\right)
=
\frac{\widehat{\mathrm{Var}}(\widehat f^u)}{N}
+
\frac{\widehat{\mathrm{Var}}(Y-\widehat f)}{n}.
$$

#### Linear regression

Define

$$
H_U
=
\frac{1}{N}\sum_{j\in\mathcal U}x_j^u{x_j^u}^\top.
$$

The internal PPI coefficient estimator is

$$
\widehat\beta_{\mathrm{PPI}}
=
H_U^{+}
\left[
\frac{1}{N}\sum_{j\in\mathcal U}x_j^u\widehat f_j^u
+
\frac{1}{n}\sum_{i\in\mathcal I}x_i\left(Y_i-\widehat f_i\right)
\right].
$$

#### Logistic GLM

The internal PPI logistic estimator solves

$$
0
=
\frac{1}{N}\sum_{j\in\mathcal U}
x_j^u\left(\mu_\beta(x_j^u)-\widehat f_j^u\right)
+
\frac{1}{n}\sum_{i\in\mathcal I}
x_i\left(\widehat f_i-Y_i\right).
$$

For linear and logistic targets, the code uses a sandwich covariance matrix of the form

$$
\widehat\Sigma
=
\widehat H^{+}
\left(
\frac{\widehat V_U}{N}
+
\frac{\widehat V_I}{n}
\right)
\widehat H^{+}.
$$

### 4.4 PPI++V1: `ppi-python==0.2.3`

PPI++V1 is a strict wrapper around `ppi-python==0.2.3`. The wrapper passes `lam=None`, so the package estimates a power-tuning parameter internally.

For a fixed scalar $\lambda$, the package mean estimator has the form

$$
\widehat\theta_{\mathrm{V1}}(\lambda)
=
\frac{\lambda}{N}\sum_{j\in\mathcal U}\widehat f_j^u
+
\frac{1}{n}\sum_{i\in\mathcal I}
\left(Y_i-\lambda\widehat f_i\right).
$$

For OLS, define

$$
\mathrm{OLS}(X,z)
=
\left(X^\top X\right)^{+}X^\top z.
$$

The package point estimator is

$$
\widehat\beta_{\mathrm{V1}}(\lambda)
=
\mathrm{OLS}\left(X_U,\lambda\widehat f_U\right)
+
\mathrm{OLS}\left(X_I,Y_I-\lambda\widehat f_I\right).
$$

For logistic GLM, define the logistic pseudo-outcome loss

$$
\ell(\beta;x,z)
=
-zx^\top\beta
+
\log\left(1+\exp(x^\top\beta)\right).
$$

The package point estimator minimizes

$$
\widehat\beta_{\mathrm{V1}}(\lambda)
=
\underset{\beta}{\mathrm{arg\,min}}
\left[
\frac{\lambda}{N}\sum_{j\in\mathcal U}
\ell(\beta;x_j^u,\widehat f_j^u)
-
\frac{\lambda}{n}\sum_{i\in\mathcal I}
\ell(\beta;x_i,\widehat f_i)
+
\frac{1}{n}\sum_{i\in\mathcal I}
\ell(\beta;x_i,Y_i)
\right].
$$

For `coord=None`, the package chooses one scalar $\lambda$ to reduce total asymptotic variance across the full parameter vector. In simplified notation, let $g_i$ be labelled outcome gradients, let $\widehat g_i$ and $\widehat g_j^u$ be prediction-based gradients, and let $A$ be the inverse Hessian. Define

$$
C
=
\frac{1}{n}\sum_{i\in\mathcal I}
\left[
(g_i-\overline g)(\widehat g_i-\overline{\widehat g})^\top
+
(\widehat g_i-\overline{\widehat g})(g_i-\overline g)^\top
\right],
$$

and let $V$ be the covariance matrix of the combined labelled and unlabelled prediction gradients. The package's overall tuning rule has the form

$$
\widehat\lambda_{\mathrm{V1}}
=
\frac{\mathrm{tr}(ACA)}
{2\left(1+n/N\right)\mathrm{tr}(AVA)},
$$

followed by clipping to $[0,1]$.

The wrapper does not receive the selected value from the package, so `selected_lambda` is stored as missing and `lambda_source` is recorded as `package_internal_not_exposed`.

Important: V1 is not merely another search procedure for the project-specific V2 estimator. For mean estimation, the fixed-$\lambda$ point-estimator forms coincide. For OLS and logistic regression, V1 follows the package formulations above, while V2 follows the internal estimating equations in Section 4.5.

### 4.5 PPI++V2: internal covariance-trace grid

PPI++V2 searches

$$
\Lambda
=
\{0,0.025,0.050,\ldots,1\}.
$$

For each $\lambda\in\Lambda$, the code computes an estimate and covariance matrix, then selects

$$
\widehat\lambda_{\mathrm{V2}}
=
\underset{\lambda\in\Lambda}{\mathrm{arg\,min}}
\mathrm{tr}\left(\widehat\Sigma_\lambda\right).
$$

#### Mean estimation

$$
\widehat\theta_{\lambda}
=
\frac{1}{n}\sum_{i\in\mathcal I}Y_i
+
\lambda
\left[
\frac{1}{N}\sum_{j\in\mathcal U}\widehat f_j^u
-
\frac{1}{n}\sum_{i\in\mathcal I}\widehat f_i
\right].
$$

The implemented variance is

$$
\widehat{\mathrm{Var}}\left(\widehat\theta_\lambda\right)
=
\frac{\widehat{\mathrm{Var}}(Y-\lambda\widehat f)}{n}
+
\frac{\lambda^2\widehat{\mathrm{Var}}(\widehat f^u)}{N}.
$$

#### Linear regression

Define

$$
H_I
=
\frac{1}{n}\sum_{i\in\mathcal I}x_ix_i^\top,
\qquad
H_U
=
\frac{1}{N}\sum_{j\in\mathcal U}x_j^u{x_j^u}^\top.
$$

Then

$$
H_\lambda
=
(1-\lambda)H_I+\lambda H_U,
$$

and

$$
\widehat\beta_\lambda
=
H_\lambda^{+}
\left[
\frac{1}{n}\sum_{i\in\mathcal I}x_iY_i
+
\lambda
\left(
\frac{1}{N}\sum_{j\in\mathcal U}x_j^u\widehat f_j^u
-
\frac{1}{n}\sum_{i\in\mathcal I}x_i\widehat f_i
\right)
\right].
$$

#### Logistic GLM

The V2 estimator solves

$$
0
=
\frac{1}{n}\sum_{i\in\mathcal I}
x_i
\left[
(1-\lambda)\mu_\beta(x_i)-Y_i+\lambda\widehat f_i
\right]
+
\frac{\lambda}{N}\sum_{j\in\mathcal U}
x_j^u
\left[
\mu_\beta(x_j^u)-\widehat f_j^u
\right].
$$

The code explicitly audits the V2 boundaries:

At $\lambda=0$, V2 reduces to Classic fitted only on `PPI_inf`. At $\lambda=1$, V2 reduces to the internal PPI implementation in Section 4.3.

The $\lambda=0$ boundary is therefore split-Classic, not the main Classic baseline, because the main Classic baseline uses all labelled observations.

### 4.6 Cross-PPI

For each labelled observation $i\in\mathcal L_k$, let

$$
\widehat f_i^{\mathrm{oof}}
=
\widehat f_{-k}(X_i^{\mathrm{learner}})
$$

be its out-of-fold prediction. For each unlabelled observation, define the fold-averaged prediction

$$
\overline f_j^u
=
\frac{1}{K}\sum_{k=1}^{K}
\widehat f_{-k}(X_j^{u,\mathrm{learner}}).
$$

The package Cross-PPI mean point estimator is

$$
\widehat\theta_{\mathrm{Cross}}
=
\frac{1}{N}\sum_{j\in\mathcal U}\overline f_j^u
+
\frac{1}{n_L}\sum_{i\in\mathcal L}
\left(Y_i-\widehat f_i^{\mathrm{oof}}\right).
$$

The package Cross-PPI OLS point estimator is

$$
\widehat\beta_{\mathrm{Cross}}
=
\mathrm{OLS}(X_U,\overline f_U)
+
\mathrm{OLS}\left(X_L,Y_L-\widehat f_L^{\mathrm{oof}}\right).
$$

The package Cross-PPI logistic point estimator minimizes

$$
\widehat\beta_{\mathrm{Cross}}
=
\underset{\beta}{\mathrm{arg\,min}}
\left[
\frac{1}{N}\sum_{j\in\mathcal U}
\ell(\beta;x_j^u,\overline f_j^u)
-
\frac{1}{n_L}\sum_{i\in\mathcal L}
\ell(\beta;x_i,\widehat f_i^{\mathrm{oof}})
+
\frac{1}{n_L}\sum_{i\in\mathcal L}
\ell(\beta;x_i,Y_i)
\right].
$$

The wrapper passes the full $N\times K$ prediction matrix to `ppi-python`; the package averages its columns for point estimation and uses the package Cross-PPI variance routines for confidence intervals.

### 4.7 Confidence-interval construction in the current wrapper

For the internal Classic, Naive ML, PPI, and PPI++V2 methods, the code returns an estimate and standard error. For confidence level $1-\alpha$, `main.py` constructs

$$
\mathrm{CI}_{1-\alpha,j}
=
\left[
\widehat\theta_j-z_{1-\alpha/2}\widehat{\mathrm{SE}}_j,
\widehat\theta_j+z_{1-\alpha/2}\widehat{\mathrm{SE}}_j
\right].
$$

The current PPI++V1 and Cross-PPI wrappers call `ppi-python` once with `alpha=0.05`. They convert the returned 95% interval width into a Gaussian-equivalent standard error:

$$
\widehat{\mathrm{SE}}_j
=
\frac{U_j-L_j}{2z_{0.975}}.
$$

`main.py` then reconstructs the 90%, 95%, and 97.5% intervals from that standard error. Therefore, the result files do not contain three independent package calls at three confidence levels. This statement describes the current implementation and prevents the README from overstating what the wrapper does.

## 5. Running

```bash
python -m pip install -r requirements.txt
python -m compileall -q .
python main.py --mode fast --experiment a_mean
python main.py --mode fast
python main.py --mode full

# Regenerate figures from existing tables without rerunning inference
python replot.py --mode fast --experiment a_mean
python replot.py --mode fast
```

`fast=20` and `full=100`. Seeds, grids, features, sample proportions, and requested confidence levels are controlled by `config.py`. The fixed 95% package call used by the current PPI++V1 and Cross-PPI wrappers is defined in `ppiplusv1.py` and `crossppi.py`, not in `config.py`.

## 6. Outputs

Each experiment writes to `output/{mode}/{experiment}/{figure,table,other}`.

### 6.1 Tables

- `replicate_results.csv`
- `summary_by_confidence.csv`
- `summary_95.csv`
- `learner_quality.csv`
- `learner_training_summary.csv`
- `ppiv2_tuning_summary.csv`

`learner_quality.csv` is replicate-level and contains the split-trained learner's Brier score or MSE on `PPI_inf`. Cross-PPI OOF quality is stored separately in `diagnostics.csv` under `cross_oof_quality`.

`learner_training_summary.csv` reports tuned-parameter selection frequencies, validation-score summaries, formal split-learner quality summaries, and the ordinary-PPI training and inference sizes.

### 6.2 Figures

```text
figure/
    learner_quality.png/.pdf
    inference_performance_95.png/.pdf
    diagnostic_coverage_calibration_<parameter>.png/.pdf
```

`learner_quality` shows the replicate distribution of split-trained learner prediction error on `PPI_inf`.

In `inference_performance_95`, the left column shows

$$
\widehat{\mathrm{Coverage}}-0.95,
$$

and the right column shows the median and interquartile range of the replicate-level relative width

$$
R_{b,m}
=
\frac{W_{b,m}}{W_{b,\mathrm{Classic}}}.
$$

Hollow efficiency markers indicate that the corresponding coverage error lies outside the nominal Monte Carlo band. Parameter-specific calibration diagnostics show the 90%, 95%, and 97.5% nominal levels using horizontally offset, unconnected markers.

The PPI++V2 tuning table reports selected-$\lambda$ summaries and the paired-replicate ratio

$$
R_b^{\mathrm{V2/PPI}}
=
\frac{W_{b,\mathrm{PPI++V2}}}
{W_{b,\mathrm{PPI}}}.
$$

Values below, equal to, or above one mean that V2 is narrower than, approximately unchanged from, or wider than the internal PPI implementation. It also reports

$$
\Delta\mathrm{Coverage}
=
\widehat{\mathrm{Coverage}}_{\mathrm{V2}}
-
\widehat{\mathrm{Coverage}}_{\mathrm{PPI}}.
$$

For OLS and logistic GLM, V2 selects one shared scalar $\lambda$ for the full coefficient vector by minimizing covariance trace. The same selected value is therefore repeated across parameter rows.

Naive ML remains in the result CSV files as an uncorrected prediction-only validity diagnostic. It is excluded from relative-width figures, has no standalone figure, and is displayed only as a notebook table.

### 6.3 Other files

- `config_used.json`
- `diagnostics.csv`
- `warnings.csv`
- `run_log.txt`

`warnings.csv` is always written with fixed headers, including when no warning occurs.

## 7. Notebook scope

`ipy/result_audit.ipynb` is read-only. It prefers full outputs, falls back to fast outputs, reads configuration and generated files, and never fits learners, recomputes intervals, or writes formal output.

## 8. Implementation notes

1. Main Classic uses all labelled observations. The $\lambda=0$ V2 boundary uses only `PPI_inf`; these are deliberately distinguished.
2. Internal PPI, package PPI++V1, and package Cross-PPI must not be treated as one interchangeable finite-sample implementation, especially for OLS.
3. PPI++V1 and Cross-PPI selected or fold-specific learner details are retained through package outputs and diagnostics, but the V1 selected $\lambda$ is not exposed by the current wrapper.
4. Ordinary-PPI quality in `learner_quality.csv` and Cross-PPI OOF quality in `diagnostics.csv` refer to different prediction sets and should not be conflated.
5. All formulas in this README describe the current code rather than an abstract method with different sample splitting or different package calls.
