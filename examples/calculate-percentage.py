#!/usr/bin/env python
"""Calculates the percentage of ISBNs in Anna's Archive.

This script calculates the percentage of the target collection's ISBNs in all
known ISBNs. All ISBNs are derived by merging all collections available in the
input bencoded file with codes.

The output is the numbers of all and the collection's ISBNs and the percentage.

The script is another implementation of this one:
https://software.annas-archive.li/AnnaArchivist/annas-archive/-/blob/a3b9d1be4070f493cd0b40932ff3fee22d08bb36/isbn_images/calculate_percentage_md5.py

Usage: python calculate-percentage.py codes-file collection
"""

import sys

from allisbns.dataset import CodeDataset, iterate_datasets, load_bencoded, unpack_data
from allisbns.isbn import LAST_ISBN
from allisbns.merge import union


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(f"Usage: python {sys.argv[0]} codes-file collection")
        sys.exit(1)

    input_path = sys.argv[1]
    collection_name = sys.argv[2]

    with open(input_path, "rb") as f:
        input_data = load_bencoded(f)

    try:
        target = CodeDataset(unpack_data(input_data[collection_name.encode()]))
    except KeyError:
        print(
            f"Got invalid collection name: '{collection_name}'.\n\n"
            f"Available collections: {[x.decode() for x in input_data]}"
        )
        sys.exit(1)

    all_datasets = iterate_datasets(input_data, fill_to_isbn=LAST_ISBN)
    all_merged = union(all_datasets)

    all_isbn_count = all_merged.count_filled_isbns()
    target_isbn_count = target.count_filled_isbns()

    print(f"{all_isbn_count:,d}")
    print(f"{target_isbn_count:,d}")
    print(f"{target_isbn_count / all_isbn_count * 100:.2f}")
