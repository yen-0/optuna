import math
import sys
import time

import numpy as np


def prototype(a, erf):
    sqrt12 = 1 / math.sqrt(2.0)
    half_log = 0.5 * math.log(2 * math.pi)
    eps = sys.float_info.epsilon
    edges = np.array([0.84375, 1.25, 1 / 0.35, 28.0])

    def horner(c, x):
        out = np.full_like(x, c[-1])
        for v in c[-2::-1]:
            out = out * x + v
        return out

    def erfc_pos(x):
        out = np.zeros_like(x)
        bins = np.searchsorted(edges, x, side="right")
        m = bins == 0
        out[m] = 1 - x[m] * (1 + horner(erf.pp.coef, x[m] ** 2) / horner(erf.qq.coef, x[m] ** 2))
        m = bins == 1
        s = x[m] - 1
        out[m] = 1 - erf.erx - horner(erf.pa.coef, s) / horner(erf.qa.coef, s)
        for b, r, q in ((2, erf.ra.coef, erf.sa.coef), (3, erf.rb.coef, erf.sb.coef)):
            m = bins == b
            xm = x[m]
            z = 1 / (xm * xm)
            out[m] = np.exp(-xm * xm - 0.5625 + horner(r, z) / horner(q, z)) / xm
        return out

    def ndtr(x):
        z = x * sqrt12
        out = np.empty_like(z)
        left, right = z < -sqrt12, z >= sqrt12
        mid = ~(left | right)
        out[left] = 0.5 * erfc_pos(-z[left])
        out[mid] = 0.5 + 0.5 * np.sign(z[mid]) * (np.abs(z[mid]) * (1 + horner(erf.pp.coef, z[mid] ** 2) / horner(erf.qq.coef, z[mid] ** 2)))
        out[right] = 1 - 0.5 * erfc_pos(z[right])
        return out

    out = np.full(a.shape, np.nan)
    big, mid, tail = a > 6, (a > -20) & ~(a > 6), a <= -20
    out[big] = -ndtr(-a[big])
    out[mid] = np.log(ndtr(a[mid]))
    x = a[tail]
    lhs = -0.5 * x**2 - np.log(-x) - half_log
    total, rhs, factor = np.zeros_like(x), np.ones_like(x), np.ones_like(x)
    cons, numerator, sign = 1 / x**2, 1.0, 1.0
    active = np.abs(total - rhs) > eps
    i = 0
    while active.any():
        i += 1; sign = -sign; numerator *= 2 * i - 1
        total[active] = rhs[active]; factor[active] *= cons[active]
        rhs[active] += sign * numerator * factor[active]
        active &= np.abs(total - rhs) > eps
    out[tail] = lhs + np.log(rhs)
    return out


def main():
    import optuna.samplers._tpe._erf as erf
    import optuna.samplers._tpe._truncnorm as truncnorm

    rng = np.random.RandomState(123)
    values = rng.uniform(-40, 0, 100_000)
    funcs = {
        "base_scalar": getattr(truncnorm, "_log_ndtr", None),
        "pr_vectorized": getattr(truncnorm, "_log_ndtr_negative", None),
        "prototype": lambda x: prototype(x, erf),
    }
    funcs = {k: v for k, v in funcs.items() if v is not None}
    reference = funcs.get("base_scalar", funcs.get("pr_vectorized"))
    for name, fn in funcs.items():
        for _ in range(2): fn(values)
        times = []
        for _ in range(5):
            start = time.perf_counter(); result = fn(values); times.append(time.perf_counter() - start)
        print(name, float(np.median(times)), float(np.nanmax(np.abs(result - reference(values)))))


if __name__ == "__main__":
    main()
