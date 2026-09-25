# Return Native Floats from Numerical Untransformation

## Motivation

`_SearchSpaceTransform.untransform()` is expected to return native Python
scalar values. This matters for sampler conformance checks and for consumers
that reject NumPy scalar types, even though NumPy scalars usually behave like
Python numbers in arithmetic and comparisons.

The latest upstream code already normalizes the logarithmic and stepped float
branches. However, the linear, non-stepped float branch still contained:

```python
param = min(trans_param, np.nextafter(d.high, d.high - 1))
```

When the upper-bound clamp is selected, `np.nextafter()` supplies a NumPy
scalar and `min()` returns it unchanged. The result can therefore be a
`numpy.float64` instead of a Python `float`.

## Change

The branch now explicitly converts the result:

```python
param = float(min(trans_param, np.nextafter(d.high, d.high - 1)))
```

This preserves the numerical behavior while making the output type
consistent with the other numerical branches:

```python
# Logarithmic float
param = float(np.clip(param, d.low, np.nextafter(d.high, d.high - 1)))

# Stepped float
param = float(np.clip(...))

# Linear non-stepped float
param = float(min(trans_param, np.nextafter(d.high, d.high - 1)))
```

## Verification

A regression test checks that the linear non-stepped float branch returns a
native `float` for both transform modes. The complete transform test module
passes:

```text
123 passed
```
