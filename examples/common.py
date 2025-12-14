import numpy as np

from numpy.typing import ArrayLike, NDArray

from allisbns.isbn import FIRST_ISBN, get_prefix_bounds
from allisbns.ranges import REGISTRATION_GROUPS


LATEST_DUMP_FILENAME = "aa_isbn13_codes_20251118T170842Z.benc.zst"


def label_groups(bins: ArrayLike, bin_size: int, offset: int = FIRST_ISBN) -> NDArray:
    """Assigns each bin a group number according to the group boundaries."""
    output = np.zeros_like(bins, dtype=np.float32)
    last_isbn = offset + bin_size * len(bins)

    group_boundaries = {}
    for prefix in sorted(REGISTRATION_GROUPS.keys()):
        group_boundaries[prefix] = get_prefix_bounds(prefix)

    for index, prefix in enumerate(group_boundaries.keys(), 1):
        start_isbn, end_isbn = group_boundaries[prefix]
        if start_isbn <= last_isbn:
            left = (start_isbn - offset) // bin_size
            right = (end_isbn - offset) // bin_size
            output[left : right + 1] = index
        else:
            break

    return output
