# Optuna PR #46 Benchmark Report

## Executive summary

PR #46 improves the complete numerical TPE workload from **2.383323 s** to
**1.852960 s** for 100 trials and 20 parameters.

| Metric | Result |
|---|---:|
| Total time saved | **0.530363 s** |
| Time saved per trial | **5.304 ms** |
| Overall speedup | **1.29x** |
| Runtime reduction | **22.3%** |

The largest estimated contribution is density/log-PDF evaluation, followed by
vectorized truncated-normal calculations.

## Compared revisions

| Label | Revision | Description |
|---|---|---|
| Base | `e1171f56f0118c2a1ec0881f101902b8b5e0965c` | PR #46 target revision |
| PR #46 | `9372230b8f9ae69b017adca878745c44e16bbac4` | PR head after 21 commits |

The benchmark was run in separate worktrees so the two implementations were
isolated.

## Whole-study benchmark

The benchmark used the same study configuration for both revisions:

- 100 TPE trials
- 20 numerical `FloatDistribution` parameters per trial
- `TPESampler(seed=123, n_startup_trials=10)`
- Identical objective, search space, and random seed
- In-memory study
- Python 3.12 on Windows
- Optuna logging disabled

```python
def objective(trial):
    return sum(
        (trial.suggest_float(f"x{i}", -5.0, 5.0) - 0.5) ** 2
        for i in range(20)
    )
```

### Measured data

| Implementation | Total study time | Time per trial | Best value |
|---|---:|---:|---:|
| Base | 2.383323 s | 23.833 ms | 47.282189668775 |
| PR #46 | 1.852960 s | 18.530 ms | 47.282189668775 |

The identical best value confirms that this deterministic numerical workload
produced the same optimization result in both revisions.

## Component benchmark

The component benchmark used 1,000 observations, 20 dimensions, and a batch
of 256 candidates. Each measurement used three warmups and 15 timed runs;
the table reports median wall-clock time.

| Operation | Base | PR #46 | Change |
|---|---:|---:|---:|
| Parzen construction | 1.321 ms | 1.757 ms | **33.0% slower** |
| Parzen sampling | 35.580 ms | 5.142 ms | **6.9x faster** |
| Parzen log-PDF | 813.626 ms | 155.207 ms | **5.2x faster** |
| Truncated-normal `ppf` | 144.079 ms | 22.102 ms | **6.5x faster** |
| Truncated-normal `rvs` | 142.174 ms | 27.752 ms | **5.1x faster** |

## Contribution to the total improvement

The complete study saved **5.304 ms per trial**. The following allocation uses
the relative component improvements to estimate how that observed whole-study
saving is divided among PR goals.

| Goal | 概要（日本語） | 変更した関数と変更内容（日本語） | Estimated saving per trial | Estimated saving over 100 trials | Contribution |
|---|---|---|---:|---:|---:|
| Density/log-PDF evaluation | 候補点が good/bad のモデル `l(x)` と `g(x)` の下でどれだけ尤もらしいかを、対数確率密度で評価する処理を高速化。 | `probability_distributions.py` の `_MixtureOfProductDistribution.log_pdf` で、`_truncnorm.logpdf` の呼び出しを、`_log_standard_norm_pdf` による NumPy 配列演算へ置き換え。候補点を1点ずつ処理する形から、複数候補・複数次元をまとめて計算する形に変更。 | 3.78 ms | 0.378 s | **71.2%** |
| Truncated-normal/vectorized sampling | パラメータ範囲内の切断正規分布から、候補となるパラメータ値をまとめて生成する処理をベクトル化。 | `_truncnorm.py` の `_ndtri_exp`、`ppf`、`rvs` と `_erf.py` の `ndtr_negative` / `log_ndtr_negative` を配列入力対応に変更。Python のスカラー処理を、NumPy の配列演算・マスク処理・ベクトル化 Newton 反復に置き換え。 | 1.36 ms | 0.136 s | **25.6%** |
| Sampling orchestration | どの混合分布成分から候補を生成するかを選び、各次元のサンプルを組み立てる処理のオーバーヘッドを削減。 | `probability_distributions.py` の `_MixtureOfProductDistribution.sample` で、候補ごとのループを削減。`rng.choice` で選んだ成分の `mu` / `sigma` を配列にまとめ、一括で `_truncnorm.rvs` を呼び出すように変更。 | 0.17 ms | 0.017 s | **3.3%** |
| Parzen construction | 過去の good/bad 試行から、各パラメータの `mu` と `sigma` を計算して Parzen モデルを構築する処理。 | `parzen_estimator.py` に `_calculate_numerical_distributions` を追加し、`mu` / `sigma` の計算をパラメータごとの処理からまとめた配列処理へ変更。`sampler.py` の `_get_internal_repr` でも観測値と log 変換を配列化。ただし構築単体では 33% 遅くなった。 | -0.002 ms | -0.0002 s | **-0.05%** |
| **Total** | **5.30 ms** | **0.530 s** | **100%** |

The truncated-normal `rvs` benchmark is part of candidate sampling, so those
measurements are not independent and must not be added twice. The non-overlap
interpretation is approximately:

- Density evaluation: **71%** of the improvement.
- Truncated-normal/vectorized sampling: **26%**.
- Sampling orchestration: **3%**.
- Parzen construction: effectively neutral, with a small regression.

These are contribution estimates, not direct per-function timings from inside
the whole-study run. The whole-study total is directly measured; the split is
derived from the component benchmark.

## Prototype comparison: vectorized `_log_ndtr`

The proposed NumPy-only prototype was also benchmarked against the scalar base
implementation and PR #46's vectorized implementation. Inputs were 100,000
values uniformly distributed over `[-40, 0]`, which matches the non-positive
domain used by the PR's tail implementation. Each result used five timed runs.

| Implementation | Median time | Maximum absolute error vs. base |
|---|---:|---:|
| Base scalar `_log_ndtr` | 156.149 ms | 0.0 |
| Proposed prototype | 23.273 ms | `1.14e-13` |
| PR #46 vectorized implementation | 11.233 ms | 0.0 |

Relative to the base implementation, the prototype is approximately **6.7x
faster** while remaining numerically equivalent to about `1e-13` absolute error.
However, on this non-positive input distribution it is approximately **2.1x
slower than PR #46's implementation**. The prototype is therefore a strong
vectorized replacement for the old scalar implementation, but it does not
improve on the specialized negative-domain implementation currently used by
PR #46.

The prototype benchmark is implemented in
`benchmarks/bench_log_ndtr_prototype.py`.

### Why the prototype has a small numerical error

The prototype and the base implementation use the same mathematical
piecewise scheme, but they do not perform every operation in the same order.
The prototype evaluates the FreeBSD rational approximations with vectorized
NumPy Horner operations, while the base implementation evaluates the
corresponding scalar expression element by element. Floating-point rounding
therefore differs slightly at branch boundaries and during the final `log`,
`exp`, and subtraction operations.

The maximum observed absolute difference was `1.14e-13`, which is consistent
with ordinary double-precision round-off. This is not an accumulated error or
an instability observed in the tested range. It should nevertheless be
validated with the full tail and boundary test suite before replacing the
existing implementation.

The PR implementation was reported with zero error only because the PR
implementation itself was used as the reference for that run; that value does
not prove that it is mathematically exact.

### Why the prototype loses to PR #46

The prototype is faster than the old scalar implementation but slower than
PR #46 because the two implementations optimize different scopes:

| Difference | Effect |
|---|---|
| Prototype supports all three scalar branches (`a > 6`, `-20 < a <= 6`, and `a <= -20`) | More work than PR #46's negative-domain-only path |
| Prototype computes `_ndtr_vec` through `_erfc_pos` and multiple Horner evaluations | Additional polynomial evaluation and temporary arrays |
| Prototype uses `searchsorted` plus several boolean masks | Extra classification and indexing overhead |
| Prototype iterates the asymptotic tail with an active mask | General per-element convergence handling adds loop overhead |
| PR #46 first exploits symmetry and only evaluates non-positive values | Avoids the positive-side branches entirely |
| PR #46 uses `count_nonzero` binning and direct specialized formulas | Fewer masks and fewer intermediate arrays |

In short, the prototype is a general drop-in replacement for the old
three-branch scalar function, whereas PR #46 is a narrower implementation
specialized for the way TPE calls it. The prototype's approximately 23.3 ms
runtime versus PR #46's 11.2 ms is therefore expected: the prototype pays for
generality and compatibility, while PR #46 pays less work per element.

## Changes grouped by goal

## Chronological walkthrough of the PR-touched path

This section follows one numerical TPE sampling request from start to finish.
The goal is to show when each changed function is called and what data it
produces.

### 1. Collect observations: `sampler.py`

The process begins in `TPESampler._sample`. The sampler has a search space and
historical trials. It divides the trials into two groups:

```text
good trials → below model (l)
bad trials  → above model (g)
```

For each group, the changed internal-representation logic collects numerical
parameter values into arrays. Log-scaled numerical parameters are transformed
with `np.log` in one batch.

Relevant code:

```text
optuna/samplers/_tpe/sampler.py
    TPESampler._sample
    TPESampler._get_internal_repr
```

At the end of this step, the data is conceptually:

```text
observations_l = [values from good trials]
observations_g = [values from bad trials]
```

### 2. Build the Parzen models: `parzen_estimator.py`

`TPESampler._sample` creates one `_ParzenEstimator` for each group. The
constructor transforms the observations into numerical distributions.

```text
_ParzenEstimator.__init__
    ↓
_calculate_numerical_distributions
```

For each numerical parameter, this calculates:

- `mu` / `mus`: the centers of Gaussian kernels
- `sigma` / `sigmas`: the widths of Gaussian kernels
- the prior kernel centered at the middle of the search range

For independent TPE, observations are sorted per dimension to determine local
kernel widths. For multivariate TPE, widths are calculated from the complete
search-space dimension.

The result is a `_MixtureOfProductDistribution` containing one distribution
per parameter dimension.

Relevant code:

```text
optuna/samplers/_tpe/parzen_estimator.py
    _ParzenEstimator.__init__
    _calculate_numerical_distributions
```

### 3. Generate candidate values: `probability_distributions.py`

The Parzen estimator's `sample` method asks its mixture distribution to
generate candidates.

```text
_ParzenEstimator.sample
    ↓
_MixtureOfProductDistribution.sample
```

The mixture sampler first chooses which Gaussian component should generate
each candidate using `rng`. It then gathers the selected `mus` and `sigmas`
into arrays and samples all dimensions in a batch.

For continuous parameters, the values come from truncated normal
distributions. For discrete parameters, the continuous result is rounded to
the parameter step and clipped to the valid range.

Relevant code:

```text
optuna/samplers/_tpe/probability_distributions.py
    _MixtureOfProductDistribution.sample
    _BatchedTruncNormDistributions
    _BatchedDiscreteTruncNormDistributions
```

### 4. Sample from a truncated normal: `_truncnorm.py`

For a numerical parameter, the mixture sampler calls `_truncnorm.rvs`.

```text
_MixtureOfProductDistribution.sample
    ↓
_truncnorm.rvs
    ↓
_truncnorm.ppf
```

`rvs` generates uniform random quantiles. `ppf` converts those quantiles into
values inside the interval `[a, b]`, where `a` and `b` are the normalized lower
and upper bounds.

`ppf` needs the Gaussian probability mass inside the interval, so it calls:

```text
_log_gauss_mass
    ↓
_log_ndtr_negative / _ndtr_negative
```

The PR changes this path from mostly scalar calculations to array-oriented
calculations.

Relevant code:

```text
optuna/samplers/_tpe/_truncnorm.py
    rvs
    ppf
    _log_gauss_mass
    _ndtri_exp
```

### 5. Compute normal CDF values: `_erf.py`

The lowest-level numerical work computes the normal CDF or its logarithm.

```text
_truncnorm._ndtr_negative
    ↓
_erf.ndtr_negative

_truncnorm._log_ndtr_negative
    ↓
_erf.log_ndtr_negative
```

These functions use piecewise approximations based on the magnitude of the
input. The PR uses symmetry to focus on non-positive values, because the
right-tail probability is more numerically stable when expressed through the
left tail.

The low-level implementation uses the rational approximations in `_erf.py` to
evaluate `erf`, `erfc`, and related normal-CDF expressions without SciPy.

Relevant code:

```text
optuna/samplers/_tpe/_erf.py
    _ndtr_negative_non_big
    ndtr_negative
    _log_ndtr_negative
    log_ndtr_negative
```

### 6. Score the candidates: `probability_distributions.py`

After candidates are generated, the Parzen estimator evaluates their density.

```text
_ParzenEstimator.log_pdf
    ↓
_MixtureOfProductDistribution.log_pdf
    ↓
_log_standard_norm_pdf
    ↓
_truncnorm._log_gauss_mass
```

For each candidate, this computes how probable it is under the mixture model.
The sampler evaluates both models:

```text
log l(x) - log g(x)
```

The candidate with the best ratio is selected. This density-evaluation stage
is where the benchmark measured the largest contribution to the PR's speedup.

### 7. Return the selected parameter set: `sampler.py`

Finally, `TPESampler._sample` selects the best candidate according to the
density ratio, converts internal numerical representations back to external
parameter values, and returns the parameter dictionary to the trial.

The complete numerical path is therefore:

```text
TPESampler._sample
  → _get_internal_repr
  → _ParzenEstimator.__init__
  → _calculate_numerical_distributions
  → _ParzenEstimator.sample
  → _MixtureOfProductDistribution.sample
  → _truncnorm.rvs
  → _truncnorm.ppf
  → _log_gauss_mass
  → _erf.ndtr_negative / log_ndtr_negative
  → _ParzenEstimator.log_pdf
  → _MixtureOfProductDistribution.log_pdf
  → select best candidate
```

### 1. Vectorize Gaussian and truncated-normal math

Files: `_erf.py`, `_truncnorm.py`

- Adds vectorized `ndtr_negative` and `log_ndtr_negative`.
- Replaces scalar loops and cached scalar functions with NumPy operations.
- Vectorizes truncated-normal inverse-CDF calculations.

Measured effect: approximately **5–6.5x faster** in the isolated
truncated-normal operations, contributing an estimated **25.6%** of the total
study speedup.

Representative diff:

```diff
-from optuna.samplers._tpe._erf import erf
+from optuna.samplers._tpe._erf import log_ndtr_negative as numpy_log_ndtr_negative
+from optuna.samplers._tpe._erf import ndtr_negative as numpy_ndtr_negative

-def _log_ndtr(a: np.ndarray) -> np.ndarray:
-    return np.frompyfunc(_log_ndtr_single, 1, 1)(a).astype(float)
+def _log_ndtr_negative(a: np.ndarray) -> np.ndarray:
+    if a.size < 300:
+        return np.asarray([_log_ndtr_negative_single(v) for v in a.ravel()])
+    return numpy_log_ndtr_negative(a)

-def _ndtri_exp_single(y: float) -> float:
+def _ndtri_exp(y: np.ndarray) -> np.ndarray:
+    # Vectorized Newton iterations over the whole input array.
+    ...
```

### 2. Batch candidate sampling

Files: `_truncnorm.py`, `probability_distributions.py`

- Samples mixture components in batches.
- Performs discrete rounding after vectorized continuous sampling.
- Reduces Python-level dispatch and repeated broadcasting.

Measured effect: Parzen sampling improved from **35.580 ms** to **5.142 ms**.

Representative diff:

```diff
-ret = np.empty((batch_size, len(self.distributions)), dtype=np.float64)
-for i in range(batch_size):
-    ...
+active_indices = rng.choice(len(self.weights), p=self.weights, size=batch_size)
+active_mus = np.concatenate(
+    [d.mu[active_indices, np.newaxis] for d in self.distributions], axis=1
+)
+active_sigmas = np.concatenate(
+    [d.sigma[active_indices, np.newaxis] for d in self.distributions], axis=1
+)
+ret = _truncnorm.rvs(a=..., b=..., loc=active_mus, scale=active_sigmas,
+                     random_state=rng)
```

### 3. Vectorize density evaluation

Files: `probability_distributions.py`, `_truncnorm.py`

- Adds direct vectorized standard-normal log-density evaluation.
- Avoids the previous scalar-oriented truncated-normal `logpdf` path.

Measured effect: log-PDF evaluation improved from **813.626 ms** to
**155.207 ms**, making it the largest estimated contributor at **71.2%**.

Representative diff:

```diff
-weighted_log_pdf += _truncnorm.logpdf(
-    x[:, np.newaxis, cont_inds],
-    a=..., b=..., loc=mus_cont, scale=sigmas_cont,
-).sum(axis=-1)
+weighted_log_pdf += _log_standard_norm_pdf(
+    (x[:, np.newaxis, cont_inds] - mus_cont) / sigmas_cont,
+    a=..., b=...,
+).sum(axis=-1) - np.log(sigmas_cont).sum(axis=-1)
```

The new helper is direct array arithmetic:

```diff
+def _log_standard_norm_pdf(z, a, b):
+    return -(z**2) / 2.0 - _truncnorm._norm_pdf_logC \
+        - _truncnorm._log_gauss_mass(a, b)
```

### 4. Vectorize Parzen-estimator construction

File: `parzen_estimator.py`

- Calculates numerical means and sigmas together.
- Vectorizes sorting, endpoint handling, clipping, and multivariate setup.

Measured effect: construction became **33% slower** in the isolated benchmark,
but this phase is small relative to density evaluation and sampling.

Representative diff:

```diff
+mus, sigmas = _calculate_numerical_distributions(
+    transformed_observations,
+    lows=np.array([dist.low for dist in search_space.values()]),
+    highs=np.array([dist.high for dist in search_space.values()]),
+    steps=np.array([dist.step or 0.0 for dist in search_space.values()]),
+    multivariate=parameters.multivariate,
+    consider_endpoints=parameters.consider_endpoints,
+    consider_magic_clip=parameters.consider_magic_clip,
+    search_space_dim=len(search_space),
+)

-self._calculate_distributions(
-    transformed_observations[:, i], param, search_space[param], parameters
-)
+_BatchedTruncNormDistributions(mus[:, i], sigmas[:, i], dist.low, dist.high)
```

### 5. Batch numerical parameter preprocessing

File: `sampler.py`

- Collects numerical observations into arrays.
- Applies log transformations in batches.

This was not isolated as a separate timing, but it is expected to be a small
contributor compared with density evaluation and truncated-normal operations.

Representative diff:

```diff
-values: dict[str, list[float]] = {param_name: [] for param_name in search_space}
+values: list[list[float]] = [[] for _ in search_space]
 ...
-    param = trial.params[param_name]
-    distribution = trial.distributions[param_name]
-    values[param_name].append(distribution.to_internal_repr(param))
+    values[i].append(trial.params[param_name])
 ...
+values_array = np.asarray(values)
+values_array[log_inds] = np.log(values_array[log_inds])
```

## Correctness and coverage findings

The benchmark workload contains only numerical parameters. Targeted TPE tests
on PR #46 produced:

```text
23 failed, 86 passed
```

The failures include categorical-distribution incompatibilities:

- `_ParzenEstimator` assumes every distribution has `low`, `high`, and `step`.
- `_MixtureOfProductDistribution.sample()` assumes every distribution has
  `mu` and `sigma`.

The truncated-normal tests could not be collected because SciPy was not
installed in the benchmark environment.

Therefore, the reported whole-study speedup applies to numerical-only TPE
studies. A mixed numerical/categorical benchmark cannot currently be treated
as valid until those PR failures are fixed.

## Reproduction

The benchmark branch is `benchmark/pr46` in the worktree:

```text
C:\Users\yen.142857.yen\pfn\optuna-pr46-benchmark
```

Scripts:

- `benchmarks/bench_pr46.py` — component benchmark
- `benchmarks/bench_pr46_study.py` — complete-study benchmark

