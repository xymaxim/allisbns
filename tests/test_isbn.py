from itertools import islice

import pytest

from allisbns.errors import InvalidISBNReason
from allisbns.isbn import (
    CanonicalISBN,
    MaskedISBN,
    generate_isbns,
    normalize_isbn,
    validate_isbn,
)


@pytest.mark.parametrize(
    "isbn, expected",
    [
        ("9780000000002", "9780000000002"),
        ("9780000000003", "9780000000003"),
        ("0000000002", "9780000000002"),
        ("978000000000X", "978000000000X"),
        ("978000000000x", "978000000000X"),
    ],
)
def test_canonical_isbn(isbn, expected):
    assert expected == str(CanonicalISBN(isbn))


@pytest.mark.parametrize(
    "isbn, expected",
    [
        ("978-000-000-000-X", "9780000000002"),
        ("978-000-000-001-X", "9780000000019"),
        ("978-000-000-002-X", "9780000000026"),
    ],
)
def test_complete_canonical_isbn(isbn, expected):
    assert expected == str(CanonicalISBN(isbn).complete())


@pytest.mark.parametrize(
    "isbn, expected",
    [
        ("978000000000X", 978000000000),
    ],
)
def test_canonical_isbn_to_isbn12(isbn, expected):
    assert expected == CanonicalISBN(isbn).to_isbn12()


@pytest.mark.parametrize(
    "isbn, expected",
    [
        (978_000_000_000, "9780000000002"),
        ("978-000-000-000-X", "978000000000X"),
        ("978-000-000-000-3", "9780000000003"),
    ],
)
def test_normalize_isbn(isbn, expected):
    assert expected == str(normalize_isbn(isbn))


@pytest.mark.parametrize(
    "isbn, expected",
    [
        ("978-000-000-000-X", "978000000000X"),
        ("978-000-000-000-3", "9780000000002"),
    ],
)
def test_normalize_isbn_and_correct(isbn, expected):
    assert expected == str(normalize_isbn(isbn, correct=True))


@pytest.mark.parametrize(
    "isbn, masked_dict",
    [
        (
            "9780000000002",
            {
                "bookland": "978",
                "group": "0",
                "registrant": "00",
                "publication": "000000",
                "check_digit": "2",
            },
        ),
    ],
)
def test_masked_isbn_from_canonical(isbn, masked_dict):
    expected = MaskedISBN(**masked_dict)
    assert expected == MaskedISBN.from_canonical(CanonicalISBN(isbn))


@pytest.mark.parametrize(
    "isbn, expected",
    [
        ("9780000000002", "978-0-00-000000-2"),
    ],
)
def test_hyphenate_isbn(isbn, expected):
    assert expected == CanonicalISBN(isbn).hyphenate()


@pytest.mark.parametrize(
    "isbn, expected",
    [
        # Correct ISBN
        ("978000000000X", (True, [])),
        # ISBN from the reserved '979-0' group
        ("979000000000X", (False, [InvalidISBNReason.BAD_GROUP])),
        # ISBN from the undefined registrant range
        ("979800000000X", (False, [InvalidISBNReason.BAD_REGISTRANT])),
        # ISBN from the reserved group and with the incorrect check digit
        (
            "9790000000002",
            (False, [InvalidISBNReason.BAD_CHECK_DIGIT, InvalidISBNReason.BAD_GROUP]),
        ),
    ],
)
def test_validate_with_reasons(isbn, expected):
    assert expected == validate_isbn(CanonicalISBN(isbn), return_reasons=True)


@pytest.mark.parametrize(
    "start_isbn, limit, expected",
    [
        (None, None, ["9780000000002", "9780000000019", "9780000000026"]),
        (978_000_000_003, 3, ["9780000000033", "9780000000040", "9780000000057"]),
        (979_999_999_999, 0, []),
        (979_999_999_999, 1, ["9799999999990"]),
        (979_999_999_999, 2, ["9799999999990"]),
    ],
)
def test_generate_isbns(start_isbn, limit, expected):
    if limit is None:
        testing_generator = islice(generate_isbns(start_isbn), 3)
    else:
        testing_generator = generate_isbns(start_isbn, limit)
    expected_values = [CanonicalISBN._from_normalized(x) for x in expected]
    assert expected_values == list(testing_generator)


@pytest.mark.parametrize("start_isbn", [978_000_000_000 - 1, 979_999_999_999 + 1])
def test_generate_isbns_outside(start_isbn):
    with pytest.raises(ValueError):
        next(generate_isbns(start_isbn))
