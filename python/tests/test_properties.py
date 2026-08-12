"""Property-based tests.

The layout-invariance property uses the function as its own oracle: casting
an arbitrarily-strided view must behave exactly like casting a C-contiguous
copy of it -- same values, same shape, or the same error. This needs no
reimplementation of the casting semantics, and it samples two spaces the
example-based tests only probe pointwise: the full (source, target) dtype
grid (121 monomorphized conversion paths) and the space of memory layouts
(dimensionality x axis permutations x slices with steps).
"""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as npst

from cast_value_rs import cast_array, cast_array_into

DTYPES = [
    "int8",
    "int16",
    "int32",
    "int64",
    "uint8",
    "uint16",
    "uint32",
    "uint64",
    "float16",
    "float32",
    "float64",
]

ROUNDING_MODES = [
    "nearest-even",
    "towards-zero",
    "towards-positive",
    "towards-negative",
    "nearest-away",
]


@st.composite
def views(draw: st.DrawFn) -> np.ndarray:
    """An array of any supported dtype, seen through a random transpose and
    basic-index view.

    Yields every layout class the bindings must handle: C-contiguous,
    F-contiguous, strided, negative-stride, zero-size, and 0-d.
    """
    dtype = draw(st.sampled_from(DTYPES))
    base = draw(
        npst.arrays(
            dtype=dtype,
            shape=npst.array_shapes(min_dims=0, max_dims=4, min_side=0, max_side=7),
        )
    )
    if base.ndim == 0:
        return base
    perm = draw(st.permutations(range(base.ndim)))
    transposed = base.transpose(perm)
    view = transposed[draw(npst.basic_indices(transposed.shape))]
    # A full-integer index yields a numpy scalar; re-wrap it as a 0-d array.
    return view if isinstance(view, np.ndarray) else np.asarray(view)


LAYOUT_ARGS = given(
    arr=views(),
    target_dtype=st.sampled_from(DTYPES),
    rounding_mode=st.sampled_from(ROUNDING_MODES),
)


@LAYOUT_ARGS
@settings(deadline=None)
def test_cast_array_layout_invariance(
    arr: np.ndarray, target_dtype: str, rounding_mode: str
) -> None:
    """Casting a view is equivalent to casting a C-contiguous copy of it."""
    kwargs = dict(
        target_dtype=target_dtype,
        rounding_mode=rounding_mode,
        out_of_range_mode="clamp",
    )
    try:
        expected = cast_array(arr.copy(order="C"), **kwargs)
    except ValueError:
        # e.g. NaN input with an integer target: the error must not depend
        # on the input's memory layout either.
        with pytest.raises(ValueError):
            cast_array(arr, **kwargs)
        return
    actual = cast_array(arr, **kwargs)
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    assert np.array_equal(actual, expected, equal_nan=actual.dtype.kind == "f")


@LAYOUT_ARGS
@settings(deadline=None)
def test_cast_array_into_matches_cast_array(
    arr: np.ndarray, target_dtype: str, rounding_mode: str
) -> None:
    """cast_array_into agrees with cast_array for any input memory layout."""
    kwargs = dict(rounding_mode=rounding_mode, out_of_range_mode="clamp")
    out = np.zeros(arr.shape, dtype=target_dtype)
    try:
        expected = cast_array(
            arr.copy(order="C"), target_dtype=target_dtype, **kwargs
        )
    except ValueError:
        with pytest.raises(ValueError):
            cast_array_into(arr, out, **kwargs)
        return
    cast_array_into(arr, out, **kwargs)
    assert np.array_equal(out, expected, equal_nan=out.dtype.kind == "f")
