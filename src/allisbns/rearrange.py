"""Rearranges bins to binned images in various ways."""

import math

import numpy as np

from numpy.typing import ArrayLike, NDArray


def rearrange_to_rows(
    bins: ArrayLike, bin_size: int, width: int, pad_value: int = 0
) -> NDArray:
    """Rearranges bins into rows with a fixed width.

    Arguments:
        bins: An array of bins.
        bin_size: A size of bins.
        width: A width of rows.
        pad_value: A value used to fill the remaining space of the final row
            (when the number of bins is not divisible by width).

    Returns:
        Bins rearranged to rows with shape (*height*, *row width*).

    Example:
        Here is a fictitious example to demonstrate the filling order:

          >>> from allisbns.dataset import BinnedArray
          >>> binned = BinnedArray(range(8), bin_size=1)
          >>> rearrange_to_rows(
          ...      binned, bin_size=binned.bin_size, width=5
          ... )
          array([[0, 1, 2, 3, 4],
                 [5, 6, 7, 0, 0]])
    """
    if width % bin_size != 0:
        raise ValueError(
            f"row width ({width}) is not divisible by bin size ({bin_size})"
        )

    bins = np.asarray(bins)

    bins_in_row = width // bin_size
    height = math.ceil(len(bins) / bins_in_row)

    if (pad_width := bins_in_row * height - len(bins)) > 0:
        padded_bins = np.concatenate([bins, [pad_value] * pad_width])
    else:
        padded_bins = bins

    new_shape: tuple[int, int] | tuple[int, int, int]
    if bins.ndim == 1:
        new_shape = (height, bins_in_row)
    elif bins.ndim == 2:
        new_shape = (height, bins_in_row, bins.shape[1])

    return padded_bins.reshape(new_shape)


def rearrange_to_blocks(
    bins: ArrayLike,
    bin_size: int,
    block_width: int = int(1e5),
    block_size: int = int(5e7),
) -> NDArray:
    """Rearranges bins into blocks of fixed width and height.

    The arrangement is inspired by the "bookshelf" space-filling curve [1], but
    we treat it in a simpler way. So, it is basically an equivalent to the
    line-filling curve as in :meth:`rearranges_to_rows` with additional division
    into vertical blocks of the fixed size stacked horizontally afterwards.

    Arguments:
        bins: An array of bins.
        bin_size: A size of bins.
        block_width: A width of one block.
        block_size: A number of ISBNs in one block.
        pad_value: A value used to fill the remaining space of the final block
            (when the number of bins is not divisible by the block size).

    Returns:
        Bins rearranged into shape (*block height*, *block width * block count*).

    References:
        1. https://phiresky.github.io/blog/2025/visualizing-all-books-in-isbn-space/

    Example:
        Here is a fictitious example to demonstrate the filling order:

          >>> from allisbns.dataset import BinnedArray
          >>> binned = BinnedArray(range(18), bin_size=1)
          >>> rearrange_to_blocks(
          ...     binned, binned.bin_size, block_width=3, block_size=9
          ... )
          BinnedArray([[ 0,  1,  2,  9, 10, 11],
                       [ 3,  4,  5, 12, 13, 14],
                       [ 6,  7,  8, 15, 16, 17]])
    """
    if block_width % bin_size != 0:
        raise ValueError(
            f"block width ({block_width}) is not divisible by bin size ({bin_size})"
        )

    bins = np.asarray(bins)

    bins_in_row = (block_width + bin_size - 1) // bin_size
    rows_in_block = (block_size + block_width - 1) // block_width
    blocks = bins.reshape((-1, rows_in_block, bins_in_row))

    return blocks.transpose(1, 0, 2).reshape(rows_in_block, -1)
