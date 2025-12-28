"""Working with datasets from HDF5 files."""

from __future__ import annotations

from typing import TYPE_CHECKING

import h5py

from allisbns.dataset import CodeDataset
from allisbns.errors import CollectionNotPresentError


if TYPE_CHECKING:
    from collections.abc import Generator
    from pathlib import Path

    from allisbns.isbn import ISBN12


def from_h5(
    path: Path, collection: str, fill_to_isbn: ISBN12 | None = None
) -> CodeDataset:
    """Creates a dataset from an HDF5 file.

    Arguments:
        path: A path to an HDF5 file.
        collection: A collection name to be read (e.g., 'md5', 'rgb', etc.)
        fill_to_isbn: An ISBN up to which to fill dataset codes.

    Returns:
        A dataset created from a source file.
    """
    with h5py.File(path, "r") as h5:
        try:
            return CodeDataset(h5[collection][:], fill_to_isbn=fill_to_isbn)
        except KeyError as e:
            raise CollectionNotPresentError(collection, list(h5.keys())) from e


def iterate_datasets(
    source: h5py.File,
    collections: list[str] | None = None,
    fill_to_isbn: ISBN12 | None = None,
) -> Generator[CodeDataset]:
    """Iterates over datasets created from a source HDF5 file.

    By default, iterates over all collections in a source file.

    Arguments:
        source: A source HDF5 file.
        collections: Collection names to read (e.g. 'md5', 'rgb', etc.) When
            `None` (default), iterates over all collections in a file.
        fill_to_isbn: An ISBN up to which to fill dataset codes.

    Returns:
        Yields datasets for all or selected collections.

    Example:
        Iterate over all datasets read from an HDF5 file::

          import h5py
          from allisbns.hdf5 import iterate_datasets

          with h5py.File("aa_isbn13_codes_20251222T170326Z.h5") as h5:
              for dataset in iterate_datasets(h5):
                  ...
    """
    if collections:
        for collection in collections:
            try:
                source[collection]
            except KeyError as e:
                raise CollectionNotPresentError(collection, list(source.keys())) from e
        for collection in collections:
            yield CodeDataset(source[collection][:], fill_to_isbn=fill_to_isbn)
    else:
        for collection in source:
            yield CodeDataset(source[collection][:], fill_to_isbn=fill_to_isbn)
