import numpy as np
import pytest

from numpy.testing import assert_array_equal

from allisbns.dataset import CodeDataset
from allisbns.merge import (
    difference,
    intersection,
    merge,
    symmetric_difference,
    union,
)


@pytest.mark.parametrize(
    "x, expected",
    [
        # x1: oo.
        # x2: .o.
        # ex: oo.
        (
            [[2, 1], [0, 1, 1, 1]],
            [2, 1],
        ),
    ],
)
def test_merge(x, expected):
    result = merge([CodeDataset(xi) for xi in x], lambda a, b: np.logical_or(a, b))
    assert_array_equal(result.codes, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        # x1: oo.
        # x2: .o.
        # ex: oo.
        (
            [[2, 1], [0, 1, 1, 1]],
            [2, 1],
        ),
        # x1: oo.
        # x2: .o.
        # x3: ..o
        # ex: ooo
        (
            [[2, 1], [0, 1, 1, 1], [0, 2, 1]],
            [3],
        ),
    ],
)
def test_union(x, expected):
    result = union([CodeDataset(xi) for xi in x])
    assert_array_equal(result.codes, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        # x1: oo.
        # x2: .o.
        # ex: .o.
        (
            [[2, 1], [0, 1, 1, 1]],
            [0, 1, 1, 1],
        ),
    ],
)
def test_intersection(x, expected):
    result = intersection([CodeDataset(xi) for xi in x])
    assert_array_equal(result.codes, expected)


@pytest.mark.parametrize(
    "x, expected",
    [
        # x1: oo.
        # x2: .o.
        # ex: o..
        (
            [[2, 1], [0, 1, 1, 1]],
            [1, 2],
        ),
        # x1: .o.
        # x2: oo.
        # ex: ...
        (
            [[0, 1, 1, 1], [2, 1]],
            [0, 3],
        ),
        # x1: oo.
        # x2: .o.
        # x3: o..
        # ex: ...
        (
            [[2, 1], [0, 1, 1, 1], [1, 2]],
            [0, 3],
        ),
    ],
)
def test_difference(x, expected):
    result = difference([CodeDataset(xi) for xi in x])
    assert_array_equal(result.codes, expected)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        # a: oo.
        # b: .oo
        # e: o.o
        (
            [2, 1],
            [0, 1, 2],
            [1, 1, 1],
        ),
    ],
)
def test_symmetric_difference(a, b, expected):
    result = symmetric_difference(CodeDataset(a), CodeDataset(b))
    assert_array_equal(result.codes, expected)
