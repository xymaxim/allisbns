#!/usr/bin/env python
"""This script converts bencoded files with ISBN codes to HDF5.

By default, the output file is saved in the current directory using the input
filename with a '.h5' suffix.

Usage: python convert-bencoded-to-h5.py [-h] bencoded-file [h5-file]
"""

import argparse

from pathlib import Path

import h5py
import hdf5plugin

from tqdm import tqdm

from allisbns.dataset import load_bencoded, unpack_data


def main(arguments):
    input_path = Path(arguments.input)
    with open(input_path, "rb") as f:
        input_data = load_bencoded(f)

    if arguments.output:
        output_path = Path(arguments.output)
    else:
        output_base = input_path.name.partition(".")[0]
        output_path = Path.cwd() / Path(output_base).with_suffix(".h5")

    with (
        h5py.File(output_path, "w") as h5,
        tqdm(total=len(input_data), ncols=80) as progress,
    ):
        for collection, data in input_data.items():
            collection_string = collection.decode()
            progress.set_postfix(collection=collection_string)

            h5.create_dataset(
                collection_string,
                data=unpack_data(data),
                compression=hdf5plugin.Blosc(
                    cname="zstd", clevel=5, shuffle=hdf5plugin.Blosc.SHUFFLE
                ),
            )

            progress.update()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts bencoded files with ISBN codes to HDF5"
    )
    parser.add_argument(
        "input", type=str, help="path to a file with codes", metavar="bencoded-file"
    )
    parser.add_argument(
        "output",
        type=str,
        nargs="?",
        help="save converted data to this file",
        metavar="h5-file",
    )
    main(parser.parse_args())
