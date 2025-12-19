"""Functions to merge datasets."""

from collections.abc import Iterable
from typing import Protocol, cast

import numpy as np

from allisbns.dataset import CodeDataset, UnpackedCodes


class MergeInPlaceFunction(Protocol):
    """Typing protocol for merge functions with in-place evaluation support.

    The protocol is compatible with NumPy's logical operation `functions
    <https://numpy.org/devdocs/reference/routines.logic.html#logical-operations>`__
    that take the ``out`` argument.

    """

    def __call__(
        self, a: UnpackedCodes, b: UnpackedCodes, /, out: UnpackedCodes | None
    ) -> UnpackedCodes: ...


class MergePureFunction(Protocol):
    """Typing protocol for merge out-of-place functions."""

    def __call__(self, a: UnpackedCodes, b: UnpackedCodes) -> UnpackedCodes: ...


def merge(
    datasets: Iterable[CodeDataset],
    function: MergeInPlaceFunction | MergePureFunction,
) -> CodeDataset:
    """Merges datasets using a merge function.

    The bounds of all datasets should be the same.

    Arguments:
        datasets: Datasets to merge.
        function: A merge function.

    Returns:
        A merged dataset.

    See Also:
        :func:`union`, :func:`intersection`, :func:`difference`,
        :func:`symmetric_difference`
    """
    dataset_iterator = iter(datasets)

    initial = next(dataset_iterator)
    merged_unpacked = initial.unpack_codes()

    for other in dataset_iterator:
        if other.bounds != initial.bounds:
            raise ValueError("dataset bounds must be the same")
        try:
            cast("MergeInPlaceFunction", function)(
                merged_unpacked, other.unpack_codes(), out=merged_unpacked
            )
        except TypeError:
            merged_unpacked[:] = cast("MergePureFunction", function)(
                merged_unpacked, other.unpack_codes()
            )

    return CodeDataset.from_unpacked(merged_unpacked, offset=initial.offset)


def union(x: Iterable[CodeDataset]) -> CodeDataset:
    return merge(x, np.logical_or)


def intersection(x: Iterable[CodeDataset]) -> CodeDataset:
    return merge(x, np.logical_and)


def difference(x: Iterable[CodeDataset]) -> CodeDataset:
    def merge_function(
        a: UnpackedCodes, b: UnpackedCodes, /, out: UnpackedCodes | None
    ) -> UnpackedCodes:
        return np.logical_and(a, np.invert(b, out=b), out=out)

    return merge(x, merge_function)


def symmetric_difference(x: CodeDataset, y: CodeDataset) -> CodeDataset:
    return merge([x, y], np.logical_xor)
