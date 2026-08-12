# cast-value-rs

Python bindings for [cast-value.rs](https://github.com/zarr-developers/cast-value.rs), a Rust
implementation of the [`cast_value` Zarr codec](https://github.com/zarr-developers/zarr-extensions/tree/main/codecs/cast_value).

The `cast_value` codec converts array elements between numeric data types as part of the Zarr v3
codec pipeline. This package exposes that conversion to Python as a pair of numpy-aware functions.

The hot path applies per-element work -- scalar map lookups, rounding, and range checking -- to
every value in the array. In pure Python/numpy this requires multiple passes over the data; the
Rust implementation fuses all steps into a single pass over contiguous memory, avoiding
intermediate allocations.

## Installation

```sh
pip install cast-value-rs
```

## Usage

```python
import numpy as np
from cast_value_rs import cast_array

# Convert float64 to uint8 with clamping.
data = np.array([1.7, 2.3, 300.0, -5.0], dtype=np.float64)
result = cast_array(
    data,
    target_dtype="uint8",
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
)
print(result)  # [2, 2, 255, 0]
```

`target_dtype` accepts a string name, a numpy dtype object, or a numpy scalar type:

```python
cast_array(data, target_dtype="uint8", ...)
cast_array(data, target_dtype=np.dtype("uint8"), ...)
cast_array(data, target_dtype=np.uint8, ...)
```

A scalar map handles special values such as NaN before conversion:

```python
from math import nan, inf

result = cast_array(
    np.array([1.0, nan, inf], dtype=np.float64),
    target_dtype="uint8",
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
    scalar_map_entries={nan: 0, inf: 255},
)
print(result)  # [1, 0, 255]
```

Use `cast_array_into` to write into a pre-allocated array instead of allocating a new one:

```python
from cast_value_rs import cast_array_into

src = np.array([1.0, 2.0, 3.0], dtype=np.float64)
dst = np.zeros(3, dtype=np.uint8)
cast_array_into(
    src, dst,
    rounding_mode="nearest-even",
    out_of_range_mode="clamp",
)
print(dst)  # [1, 2, 3]
```

## Supported types and modes

| Category | Values |
|---|---|
| Signed integers | `int8`, `int16`, `int32`, `int64` |
| Unsigned integers | `uint8`, `uint16`, `uint32`, `uint64` |
| Floats | `float16`, `float32`, `float64` |
| Rounding modes | `nearest-even`, `towards-zero`, `towards-positive`, `towards-negative`, `nearest-away` |
| Out-of-range modes | `clamp`, `wrap`, or `None` to raise an error |

## Documentation

Full documentation, including the Rust API, is at
<https://zarr-developers.github.io/cast-value.rs/>.

## License

Licensed under either of [Apache License, Version 2.0](https://github.com/zarr-developers/cast-value.rs/blob/main/LICENSE-APACHE)
or [MIT license](https://github.com/zarr-developers/cast-value.rs/blob/main/LICENSE-MIT) at your option.
