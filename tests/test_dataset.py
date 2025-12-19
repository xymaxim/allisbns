import io

import numpy as np
import pytest

from numpy.testing import assert_array_equal

from allisbns.dataset import CodeDataset


packed_and_unpacked_codes = [
    ([3, 2, 1], [True, True, True, False, False, True]),
    ([0, 2, 1], [False, False, True]),
    ([3, 2, 1, 2], [True, True, True, False, False, True, False, False]),
]


@pytest.fixture
def codes_from_streak():
    return np.asarray(
        [
            5,  # 978_000_000_000 -- 978_000_000_004
            1,  # 978_000_000_005
            2,  # 978_000_000_006, 978_000_000_007
            3,  # 978_000_000_008 -- 978_000_000_010
        ],
    )


@pytest.fixture
def codes_from_gap():
    return np.asarray(
        [
            0,
            1,  # 978_000_000_000
            2,  # 978_000_000_001, 978_000_000_002
            3,  # 978_000_000_003 -- 978_000_000_005
        ],
    )


@pytest.mark.parametrize(
    "codes,offset,last,expected",
    [
        ([5, 1, 2], 978_000_000_000, None, [5, 1, 2]),
        ([5, 1, 2], 978_000_000_000, 978_000_000_007, [5, 1, 2]),
        ([5, 1, 2], 978_000_000_000, 978_000_000_008, [5, 1, 2, 1]),
        ([5, 1, 2, 3], 978_000_000_000, 978_000_000_011, [5, 1, 2, 4]),
    ],
)
def test_fill_to_isbn(codes, offset, last, expected):
    assert_array_equal(CodeDataset(codes, offset, fill_to_isbn=last).codes, expected)


def test_fill_to_wrong_total():
    with pytest.raises(ValueError) as exc_info:
        CodeDataset([5, 1, 2], 978_000_000_000, fill_to_isbn=978_000_000_006)
    assert str(exc_info.value) == (
        "fill ISBN (978000000006) must be beyond right bound (978000000007)"
    )


@pytest.mark.parametrize(
    "data, offset, expected",
    [
        ([5, 3, 2, 1], 978_000_000_000, (978_000_000_000, 978_000_000_010)),
        ([5, 3, 2, 1], 979_000_000_000, (979_000_000_000, 979_000_000_010)),
        ([0, 1, 2, 3], 978_000_000_000, (978_000_000_000, 978_000_000_005)),
    ],
)
def test_bounds(data, offset, expected):
    assert expected == CodeDataset(data, offset).bounds


@pytest.mark.parametrize(
    "isbn, expected",
    [
        (978_000_000_000, (True, 0, 0)),
        (978_000_000_004, (True, 0, 4)),
        (978_000_000_005, (False, 1, 0)),
        (978_000_000_006, (True, 2, 0)),
        (978_000_000_007, (True, 2, 1)),
        (978_000_000_010, (False, 3, 2)),
    ],
)
def test_query_isbn_from_streak(codes_from_streak, isbn, expected):
    assert expected == CodeDataset(codes_from_streak).query_isbn(isbn)


@pytest.mark.parametrize(
    "isbn, expected",
    [
        (978_000_000_000, (False, 1, 0)),
        (978_000_000_001, (True, 2, 0)),
        (978_000_000_003, (False, 3, 0)),
    ],
)
def test_query_isbn_from_gap(codes_from_gap, isbn, expected):
    assert expected == CodeDataset(codes_from_gap).query_isbn(isbn)


@pytest.mark.parametrize("isbn", [977_999_999_999, 980_000_000_000])
def test_query_isbn_outside(codes_from_streak, isbn):
    with pytest.raises(ValueError):
        CodeDataset(codes_from_streak).query_isbn(isbn)


@pytest.mark.parametrize(
    "isbn, expected",
    [
        (979_000_000_000, (True, 0, 0)),
        (979_000_000_005, (False, 1, 0)),
    ],
)
def test_query_isbn_offsetted(codes_from_streak, isbn, expected):
    dataset = CodeDataset(codes_from_streak, offset=979_000_000_000)
    assert expected == dataset.query_isbn(isbn)


@pytest.mark.parametrize(
    "isbns, expected",
    [
        (
            (978_000_000_003, 978_000_000_004, 978_000_000_005, 978_000_000_006),
            (True, True, False, True),
        ),
        (
            (978_000_000_000 - 1, 978_999_999_999),
            (False, False),
        ),
    ],
)
def test_check_isbns(codes_from_streak, isbns, expected):
    dataset = CodeDataset(codes_from_streak)
    assert_array_equal(expected, dataset.check_isbns(isbns))


@pytest.mark.parametrize(
    "isbns, expected",
    [
        (
            (979_000_000_003, 979_000_000_004, 979_000_000_005, 979_000_000_006),
            (True, True, False, True),
        ),
    ],
)
def test_check_isbns_offsetted(codes_from_streak, isbns, expected):
    dataset = CodeDataset(codes_from_streak, offset=979_000_000_000)
    assert_array_equal(expected, dataset.check_isbns(isbns))


@pytest.mark.parametrize(
    "start, end, expected",
    [
        # From a streak to a streak
        (978_000_000_003, 978_000_000_006, ([2, 1, 1], 978_000_000_003)),
        # From a streak to a gap
        (978_000_000_003, 978_000_000_008, ([2, 1, 2, 1], 978_000_000_003)),
        # From a gap to a streak
        (978_000_000_005, 978_000_000_006, ([0, 1, 1], 978_000_000_005)),
        # From a gap to a gap
        (978_000_000_005, 978_000_000_008, ([0, 1, 2, 1], 978_000_000_005)),
        # Inside one streak segment
        (978_000_000_006, 978_000_000_007, ([2], 978_000_000_006)),
        # Single streak element
        (978_000_000_006, 978_000_000_006, ([1], 978_000_000_006)),
        # Inside a single gap element
        (978_000_000_008, 978_000_000_009, ([0, 2], 978_000_000_008)),
        # Single gap element
        (978_000_000_008, 978_000_000_008, ([0, 1], 978_000_000_008)),
        # Entire dataset
        (978_000_000_000, 978_000_000_010, ([5, 1, 2, 3], 978_000_000_000)),
        # Start and end are both outside
        (
            978_000_000_000 - 10,
            978_000_000_020,
            ([0, 10, 5, 1, 2, 13], 978_000_000_000 - 10),
        ),
        # Start is outside
        (978_000_000_000 - 10, 978_000_000_003, ([0, 10, 4], 978_000_000_000 - 10)),
        # End is outside
        (978_000_000_003, 978_000_000_010 + 1, ([2, 1, 2, 4], 978_000_000_003)),
        # Start and end are both to the left of bounds
        (978_000_000_000 - 10, 978_000_000_000 - 5, ([0, 6], 978_000_000_000 - 10)),
        # Start and end are both to the right of bounds
        (978_000_000_010 + 1, 978_000_000_010 + 6, ([0, 6], 978_000_000_010 + 1)),
    ],
)
def test_reframe_when_from_streak(codes_from_streak, start, end, expected):
    expected_codes, expected_offset = expected

    initial = CodeDataset(codes_from_streak)
    initial_codes = initial.codes.copy()

    reframed = initial.reframe(start, end)

    assert expected_offset == reframed.offset
    assert_array_equal(reframed.codes, expected_codes)
    assert_array_equal(initial_codes, initial.codes)


@pytest.mark.parametrize(
    "start, end, expected",
    [
        (978_000_000_000, 978_000_000_000, ([0, 1], 978_000_000_000)),
        (978_000_000_001, 978_000_000_002, ([2], 978_000_000_001)),
    ],
)
def test_reframe_when_from_gap(codes_from_gap, start, end, expected):
    expected_codes, expected_offset = expected

    initial = CodeDataset(codes_from_gap)
    initial_codes = initial.codes.copy()

    reframed = initial.reframe(start, end)

    assert expected_offset == reframed.offset
    assert_array_equal(reframed.codes, expected_codes)
    assert_array_equal(initial_codes, initial.codes)


def test_reframe_with_slicing(codes_from_streak):
    sliced = CodeDataset(codes_from_streak)[978_000_000_001:978_000_000_002]
    expected = CodeDataset([2], 978_000_000_001)
    assert sliced.codes == expected.codes
    assert sliced.offset == expected.offset


@pytest.mark.parametrize("codes, expected", packed_and_unpacked_codes)
def test_unpack_codes(codes, expected):
    assert_array_equal(CodeDataset(codes).unpack_codes(), expected)


@pytest.mark.parametrize("expected, unpacked_codes", packed_and_unpacked_codes)
def test_from_unpacked(expected, unpacked_codes):
    assert CodeDataset(expected) == CodeDataset.from_unpacked(unpacked_codes)


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (([3, 2, 1], 978_000_000_000), ([3, 2, 1], 978_000_000_000), True),
        (([3, 2, 1], 978_000_000_000), ([3, 2, 1], 999_999_999_999), False),
        (([3, 2, 1], 978_000_000_000), ([1, 1, 1], 978_000_000_000), False),
    ],
)
def test_equality(a, b, expected):
    assert expected == (CodeDataset(*a) == CodeDataset(*b))


@pytest.mark.parametrize(
    "codes, normalize, expected",
    [([5, 1, 2, 3], False, [5, 1, 2, 3]), ([5, 1, 2, 3], True, [5, 1, 2])],
)
def test_write_bencoded(codes, normalize, expected):
    with io.BytesIO() as stream:
        CodeDataset(codes).write_bencoded(stream, "new", normalize)
        stream.seek(0)
        actual_codes = CodeDataset.from_file(stream, "new").codes
    assert_array_equal(actual_codes, expected)


@pytest.mark.parametrize(
    "codes, expected",
    [
        (
            [5, 1, 2],
            [
                978_000_000_000,
                978_000_000_001,
                978_000_000_002,
                978_000_000_003,
                978_000_000_004,
                978_000_000_006,
                978_000_000_007,
            ],
        ),
    ],
)
def test_get_filled_isbns(codes, expected):
    assert_array_equal(expected, CodeDataset(codes).get_filled_isbns())


@pytest.mark.parametrize("codes, expected", [([5, 1, 2], 7)])
def test_count_filled_isbns(codes, expected):
    assert expected == CodeDataset(codes).count_filled_isbns()


@pytest.mark.parametrize(
    "subset, superset, expected",
    [
        (
            CodeDataset([1, 2, 3], offset=0),
            CodeDataset([1, 2, 3, 4], offset=0),
            True,
        ),
        # (
        #     CodeDataset([0, 2, 3], offset=1),
        #     CodeDataset([1, 2, 3], offset=0),
        #     True,
        # ),
        (
            CodeDataset([1, 2, 3], offset=0),
            CodeDataset([1, 2, 3], offset=1),
            False,
        ),
        (
            CodeDataset([1, 2, 3], offset=0),
            CodeDataset([1, 2], offset=0),
            False,
        ),
    ],
)
def test_is_subset(subset, superset, expected):
    assert expected is subset.is_subset(superset)


@pytest.mark.parametrize(
    "codes, offset, expected",
    [
        ([5, 1, 2], 0, [0, 5, 1, 2]),
        ([5, 1, 2], 1, [0, 5, 1, 2]),
    ],
)
def test_invert(codes, offset, expected):
    assert (
        CodeDataset(expected, offset=offset)
        == CodeDataset(codes, offset=offset).invert()
    )


@pytest.mark.parametrize(
    "codes, expected",
    [
        ([5, 1, 2], [0, 5, 1, 2]),
    ],
)
def test_invert_dunder(codes, expected):
    assert CodeDataset(expected) == ~CodeDataset(codes)
