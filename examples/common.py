"""Common functions and variables used in notebooks."""

from pathlib import Path

import numpy as np

from numpy.typing import ArrayLike, NDArray

from allisbns.isbn import FIRST_ISBN, get_prefix_bounds
from allisbns.ranges import REGISTRATION_GROUPS


CURRENT_DUMP_FILENAME = "aa_isbn13_codes_20251222T170326Z.benc.zst"


def get_data_directory() -> Path:
    """Returns the input data directory."""
    return Path(__file__).parent / "data"


def label_groups(bins: ArrayLike, bin_size: int, offset: int = FIRST_ISBN) -> NDArray:
    """Assigns each bin a group number according to the group bounds."""
    output = np.zeros_like(bins, dtype=np.float32)
    last_isbn = offset + bin_size * len(bins)

    group_bounds = {}
    for prefix in sorted(REGISTRATION_GROUPS.keys()):
        group_bounds[prefix] = get_prefix_bounds(prefix)

    for index, prefix in enumerate(group_bounds.keys(), 1):
        start_isbn, end_isbn = group_bounds[prefix]
        if start_isbn <= last_isbn:
            left = (start_isbn - offset) // bin_size
            right = (end_isbn - offset) // bin_size
            output[left : right + 1] = index
        else:
            break

    return output
