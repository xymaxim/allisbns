import numpy as np
import pytest

from numpy.testing import assert_array_equal

from allisbns.rearrange import rearrange_to_blocks, rearrange_to_rows


DEFAULT_ARGUMENT = object()


@pytest.mark.parametrize(
    "bins, width, pad, expected",
    [
        ([0, 1, 2, 3, 4, 5, 6, 7], 4, DEFAULT_ARGUMENT, [[0, 1, 2, 3], [4, 5, 6, 7]]),
        ([0, 1, 2, 3, 4, 5], 4, DEFAULT_ARGUMENT, [[0, 1, 2, 3], [4, 5, 0, 0]]),
        ([0, 1, 2, 3, 4, 5], 4, -1, [[0, 1, 2, 3], [4, 5, -1, -1]]),
    ],
)
def test_rearrange_to_rows(bins, width, pad, expected):
    if pad is DEFAULT_ARGUMENT:
        result = rearrange_to_rows(bins, bin_size=1, width=width)
    else:
        result = rearrange_to_rows(bins, bin_size=1, width=width, pad_value=pad)
    assert_array_equal(result, np.array(expected))


def test_rearrange_to_rows_with_not_divisable_width():
    with pytest.raises(ValueError):
        rearrange_to_rows([0, 1, 2, 3], bin_size=2, width=5)


@pytest.mark.parametrize(
    "bins, width, size, expected",
    [
        (
            [0, 1, 2, 3, 4, 5, 6, 7],
            2,
            4,
            [[0, 1, 4, 5], [2, 3, 6, 7]],
        ),
    ],
)
def test_rearrange_to_blocks(bins, width, size, expected):
    result = rearrange_to_blocks(bins, bin_size=1, block_width=width, block_size=size)
    assert_array_equal(result, np.array(expected))
