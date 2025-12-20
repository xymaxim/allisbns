"""Benchmark merge functions."""

import numpy as np
import pytest

from allisbns.dataset import CodeDataset, iterate_datasets, load_bencoded
from allisbns.isbn import LAST_ISBN
from allisbns.merge import union


@pytest.fixture
def input_data():
    input_path = "examples/aa_isbn13_codes_20251118T170842Z.benc.zst"
    with open(input_path, "rb") as f:
        return load_bencoded(f)


@pytest.fixture
def datasets_to_test(input_data):
    collections = [key.decode() for key in input_data]
    end_isbn = LAST_ISBN
    return [
        x.reframe(None, end_isbn)
        for x in iterate_datasets(
            input_data, collections=collections, fill_to_isbn=LAST_ISBN
        )
    ]


def bitwise_union(datasets):
    first_dataset, *other_datasets = datasets
    merged_bits = np.packbits(first_dataset.unpack_codes())
    for dataset in other_datasets:
        np.bitwise_or(merged_bits, np.packbits(dataset.unpack_codes()), out=merged_bits)
    total_isbns = first_dataset.bounds.end - first_dataset.bounds.start + 1
    return CodeDataset.from_unpacked(np.unpackbits(merged_bits)[:total_isbns])


def test_current_union(benchmark, datasets_to_test):
    benchmark(union, datasets_to_test)


def test_bitwise_union(benchmark, datasets_to_test):
    benchmark(bitwise_union, datasets_to_test)
