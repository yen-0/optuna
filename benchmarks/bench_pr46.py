import argparse
import json
import os
import sys
import time

import numpy as np


def timed(fn, repeats=15, warmups=3):
    for _ in range(warmups):
        fn()
    values = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        values.append(time.perf_counter() - t0)
    return {
        "median_s": float(np.median(values)),
        "min_s": float(np.min(values)),
        "max_s": float(np.max(values)),
        "samples": repeats,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    repo = os.path.abspath(args.repo)
    sys.path.insert(0, repo)

    from optuna.distributions import FloatDistribution
    from optuna.samplers._tpe._truncnorm import ppf, rvs
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimator
    from optuna.samplers._tpe.parzen_estimator import _ParzenEstimatorParameters

    rng = np.random.RandomState(12345)
    n, d, batch = 1000, 20, 256
    search_space = {f"x{i}": FloatDistribution(-5.0, 5.0) for i in range(d)}
    observations = {k: rng.uniform(-5.0, 5.0, n) for k in search_space}
    params = _ParzenEstimatorParameters(
        prior_weight=1.0,
        consider_magic_clip=True,
        consider_endpoints=True,
        weights=lambda x: np.ones(x),
        multivariate=False,
        categorical_distance_func={"x": lambda a, b: 0.0},
    )
    estimator = _ParzenEstimator(observations, search_space, params)
    samples = {k: rng.uniform(-5.0, 5.0, batch) for k in search_space}

    a = rng.uniform(-4.0, 0.0, batch * d)
    b = rng.uniform(0.0, 4.0, batch * d)
    q = rng.uniform(1e-6, 1 - 1e-6, batch * d)
    loc = np.zeros_like(a)
    scale = np.ones_like(a)

    results = {
        "repo": repo,
        "commit": os.popen(f'git -C "{repo}" rev-parse HEAD').read().strip(),
        "config": {"observations": n, "dimensions": d, "batch": batch},
        "parzen_construction": timed(lambda: _ParzenEstimator(observations, search_space, params)),
        "parzen_sample": timed(lambda: estimator.sample(rng, batch)),
        "parzen_log_pdf": timed(lambda: estimator.log_pdf(samples)),
        "truncnorm_ppf": timed(lambda: ppf(q, a, b)),
        "truncnorm_rvs": timed(lambda: rvs(a, b, loc, scale, rng)),
    }
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
