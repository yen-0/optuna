import csv
import functools
import math
import os
import time

import numpy as np
import optuna
from optuna.samplers._tpe import _truncnorm


def issue_prototype(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    tail = a < -37
    if tail.any():
        t = a[tail]
        s = 1 / (t * t)
        series = s * (-1 + s * (3 + s * (-15 + s * (105 - 945 * s))))
        out[tail] = _truncnorm._norm_logpdf(t) - np.log(-t) + np.log1p(series)
    out[~tail] = [
        math.log(0.5 * math.erfc(x)) if x >= 0 else math.log1p(-0.5 * math.erfc(-x))
        for x in (a[~tail] / -(2**0.5)).tolist()
    ]
    return out


@functools.lru_cache(1000)
def _cached_scalar(x: float) -> float:
    return math.log(0.5 * math.erfc(x)) if x >= 0 else math.log1p(-0.5 * math.erfc(-x))


def cached_prototype(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    out = np.empty_like(a)
    tail = a < -37
    if tail.any():
        t = a[tail]
        s = 1 / (t * t)
        series = s * (-1 + s * (3 + s * (-15 + s * (105 - 945 * s))))
        out[tail] = _truncnorm._norm_logpdf(t) - np.log(-t) + np.log1p(series)
    out[~tail] = [_cached_scalar(x) for x in (a[~tail] / -(2**0.5)).tolist()]
    return out


def run(name: str, fn, n_trials: int, seed: int) -> float:
    old = _truncnorm._log_ndtr
    _truncnorm._log_ndtr = fn
    try:
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(sampler=sampler)

        def objective(trial):
            return sum((trial.suggest_float(str(i), -10, 10) - 5) ** 2 for i in range(40))

        start = time.perf_counter()
        study.optimize(objective, n_trials=n_trials)
        elapsed = time.perf_counter() - start
        with open(os.environ["RESULTS_FILE"], "a", newline="") as f:
            csv.writer(f).writerow([name, n_trials, seed, f"{elapsed:.6f}"])
        print(f"{name},{n_trials},{seed},{elapsed:.6f}", flush=True)
        return elapsed
    finally:
        _truncnorm._log_ndtr = old


if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.ERROR)
    variants = [("issue", issue_prototype), ("cached", cached_prototype), ("master", _truncnorm._log_ndtr)]
    with open(os.environ["RESULTS_FILE"], "w", newline="") as f:
        csv.writer(f).writerow(["variant", "trials", "seed", "seconds"])
    for repeat in range(3):
        for trials in (10, 100, 1000, 3000):
            for name, fn in variants:
                run(name, fn, trials, repeat)
