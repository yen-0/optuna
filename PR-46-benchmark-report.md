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

| Goal | Estimated saving per trial | Estimated saving over 100 trials | Contribution |
|---|---:|---:|---:|
| Density/log-PDF evaluation | 3.78 ms | 0.378 s | **71.2%** |
| Truncated-normal/vectorized sampling | 1.36 ms | 0.136 s | **25.6%** |
| Sampling orchestration | 0.17 ms | 0.017 s | **3.3%** |
| Parzen construction | -0.002 ms | -0.0002 s | **-0.05%** |
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

