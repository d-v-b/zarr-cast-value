# Python API

## `cast_array`

```python
def cast_array(
    arr: numpy.ndarray,
    *,
    target_dtype: str | numpy.dtype | type[numpy.generic],
    rounding_mode: str,
    out_of_range_mode: str | None = None,
    scalar_map_entries: dict[float, float]
        | Iterable[tuple[float, float]]
        | None = None,
) -> numpy.ndarray
```

Cast a numpy array to a new dtype, allocating a new output array.

### Parameters

| Parameter | Type | Description |
|---|---|---|
| `arr` | `numpy.ndarray` | Input array (any supported numeric dtype) |
| `target_dtype` | `str \| numpy.dtype \| type` | Target dtype. Accepts a string name (e.g. `"uint8"`), a numpy dtype object (e.g. `np.dtype("uint8")`), or a numpy scalar type (e.g. `np.uint8`). |
| `rounding_mode` | `str` | One of: `"nearest-even"`, `"towards-zero"`, `"towards-positive"`, `"towards-negative"`, `"nearest-away"` |
| `out_of_range_mode` | `str \| None` | `"clamp"`, `"wrap"`, or `None` (error on out-of-range) |
| `scalar_map_entries` | `dict \| Iterable \| None` | Mapping of special source values to target values. Accepts a dict or any iterable of `(source, target)` pairs. |

### Returns

A new `numpy.ndarray` with the target dtype and the same shape as `arr`.

### Raises

- `ValueError` -- if a value cannot be converted (e.g. NaN to int without a scalar_map entry, or out-of-range without a mode set).
- `TypeError` -- if the source or target dtype is unsupported, or if a scalar_map entry has an incompatible type for the source or target dtype.

---

## `cast_array_into`

```python
def cast_array_into(
    arr: numpy.ndarray,
    out: numpy.ndarray,
    *,
    rounding_mode: str,
    out_of_range_mode: str | None = None,
    scalar_map_entries: dict[float, float]
        | Iterable[tuple[float, float]]
        | None = None,
) -> None
```

Cast a numpy array into a pre-allocated output array. The target dtype is
inferred from `out.dtype`.

### Parameters

Same as `cast_array`, except:

- `out` replaces `target_dtype` -- a pre-allocated output array with the desired dtype and same shape as `arr`.

### Returns

`None`. The output is written into `out`.

### Raises

- `ValueError` -- if shapes don't match, or a value cannot be converted.
- `TypeError` -- if the source or target dtype is unsupported.

---

## Rounding modes

| Mode | Behavior |
|---|---|
| `"nearest-even"` | Round to nearest, ties to even (IEEE 754 default) |
| `"towards-zero"` | Truncate towards zero |
| `"towards-positive"` | Round towards +infinity (ceiling) |
| `"towards-negative"` | Round towards -infinity (floor) |
| `"nearest-away"` | Round to nearest, ties away from zero |

## Out-of-range modes

| Mode | Behavior |
|---|---|
| `None` | Error on the first out-of-range value |
| `"clamp"` | Clamp to the target type's range |
| `"wrap"` | Modular arithmetic (wrapping) |

## Scalar map

Scalar map entries are applied before rounding and range checking. Each entry
maps a source value to a target value. NaN entries match any NaN value. First
match wins.

Scalar map values must be compatible with the source and target dtypes. For
example, when casting `float64` to `int32`, the source value should be a float
and the target value should be an integer:

```python
# Correct: float source, int target
scalar_map_entries=[(float("nan"), 0), (float("inf"), 2147483647)]

# Error: float target for int dtype
scalar_map_entries=[(float("nan"), 0.0)]  # TypeError
```

## Supported dtypes

`int8`, `int16`, `int32`, `int64`, `uint8`, `uint16`, `uint32`, `uint64`,
`float16`, `float32`, `float64`
