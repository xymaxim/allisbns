"""Classes and functions to work with ISBNs."""

from __future__ import annotations

import re

from dataclasses import asdict, dataclass
from re import Pattern
from typing import TYPE_CHECKING, Final, Literal, Self, cast

import numpy as np

from allisbns.errors import (
    InvalidISBNError,
    InvalidISBNReason,
    NotISBN12Error,
    NotISBNError,
    UndefinedISBNRangeError,
)
from allisbns.ranges import REGISTRANT_RANGES, REGISTRATION_GROUPS


if TYPE_CHECKING:
    import sys

    if "sphinx.ext.autodoc" in sys.modules:
        from collections.abc import Generator

# Type aliases

#: A valid ISBN-12 integer number.
type ISBN12 = int | np.integer

#: A string that is expected to be an ISBN.
type MaybeISBN = str

BooklandElement = Literal["978", "979"]

# Constants

#: A simple pattern to match ISBN-10 and ISBN-13 values.
ISBN_PATTERN: Pattern[str] = re.compile(r"^(?:978|979)?\d{9}[\dXx]$")

#: First defined ISBN-12.
FIRST_ISBN: Final[ISBN12] = 978_000_000_000

#: Last defined ISBN-12.
LAST_ISBN: Final[ISBN12] = 979_999_999_999

#: Total number of ISBNs.
TOTAL_ISBNS: Final = LAST_ISBN - FIRST_ISBN + 1

#: The length of ISBN-12 numbers.
ISBN12_LENGTH = 12


class CanonicalISBN:
    """Represents a canonical (normalized) ISBN-13 string.

    Only the ISBN-10 and ISBN-13 :data:`pattern <ISBN_PATTERN>` check is
    performed upon creation. So canonical ISBN values are assumed to be valid
    even if: the registration group or registrant range are undefined, the
    correct digit is incorrectly calculated. For validation, see
    :func:`validate_isbn`.

    See Also:
        :func:`normalize_isbn`
    """

    def __init__(self, value: MaybeISBN):
        """Creates a canonical ISBN-13 object."""
        self._value = self._normalize_value(value)

    @classmethod
    def _from_normalized(cls, isbn: str) -> Self:
        """Creates a new object from an already normalized ISBN-13 string."""
        new = cls.__new__(cls)
        new._value = isbn
        return new

    def _normalize_value(self, value: MaybeISBN) -> str:
        # Remove dashes and whitespaces
        output_value = _clean_isbn_string(value)

        # Check if value is ISBN-10 or ISBN-13
        if not match_isbn(output_value):
            raise NotISBNError(value)

        # Convert ISBN-10 to ISBN-13 if needed
        if len(output_value) == 10:
            output_value = f"978{output_value}"

        # Ensure that the 'X' check digit is uppercase
        if output_value[-1] == "x":
            output_value = output_value.upper()

        return output_value

    @property
    def bookland(self) -> BooklandElement:
        """Returns the bookland element."""
        return self._value[:3]  # type: ignore[return-value]

    @property
    def check_digit(self) -> str:
        """Returns the check digit."""
        return self._value[-1]

    def complete(self) -> Self:
        correct_check_digit = calculate_check_digit(self)
        if self.check_digit == correct_check_digit:
            return self
        return self._from_normalized(f"{self._value[:-1]}{correct_check_digit}")

    def hyphenate(self) -> str:
        return MaskedISBN.from_canonical(self).hyphenate()

    def to_isbn12(self) -> ISBN12:
        """Converts to an :data:`ISBN12` number."""
        return int(self._value[:ISBN12_LENGTH])

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return self._value == other._value

    def __hash__(self) -> int:
        return hash(self._value)

    def __getitem__(self, key: int | slice) -> str:
        return self._value[key]

    def __str__(self) -> str:
        return self._value

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self._value})"


@dataclass(frozen=True)
class MaskedISBN:
    """Represents a masked structure of an ISBN-13 string.

    Example:
        It is more practical to create it from a canonical ISBN:
          >>> canonical = normalize_isbn("978-000-000-000-X")
          >>> MaskedISBN.from_canonical(canonical)
          MaskedISBN(
              bookland='978',
              group='0',
              registrant='00',
              publication='000000',
              check_digit='X',
          )
    """

    #: Bookland, or GS1 element.
    bookland: BooklandElement
    #: Registration group element.
    group: str
    #: Registrant element.
    registrant: str
    #: Publication element.
    publication: str
    #: Check digit.
    check_digit: str

    @classmethod
    def from_canonical(cls, isbn: CanonicalISBN) -> Self:
        """Creates a masked structure from a canonical ISBN-13 string."""
        max_registrant_length = 7

        bookland = isbn.bookland
        current_index = len(bookland)

        # Extract a registration group element
        group_prefix = extract_group_prefix(isbn)
        _, group = group_prefix.split("-")
        current_index += len(group)

        # Match a registrant element
        isbn_rest_part = isbn[current_index:]
        registrant_ranges = REGISTRANT_RANGES[group_prefix]
        for range_start, range_end, registrant_length in registrant_ranges:
            registrant_probe = isbn_rest_part[:max_registrant_length]
            # For some groups (e.g., '978-99998'), the length of ISBN-13 may not
            # be enough to get the probe
            registrant_probe = registrant_probe.ljust(max_registrant_length, "0")
            if range_start <= registrant_probe <= range_end:
                if registrant_length == 0:
                    message = (
                        "not defined for use registrant: "
                        f"'{group_prefix}-{registrant_probe}'"
                    )
                    raise UndefinedISBNRangeError(
                        message, InvalidISBNReason.BAD_REGISTRANT, isbn
                    )
                registrant = isbn_rest_part[:registrant_length]
                current_index += registrant_length
                break

        return cls(
            bookland=bookland,
            group=group,
            registrant=registrant,
            publication=isbn[current_index:-1],
            check_digit=isbn.check_digit,
        )

    @property
    def elements(self) -> dict[str, str | BooklandElement]:
        return asdict(self)

    def hyphenate(self) -> str:
        """Formats an ISBN string with elements delimited by a hyphen."""
        return "-".join(self.elements.values())

    def __getitem__(self, key: int | slice[int, int, None]) -> str:
        match key:
            case slice() as x if x.step is not None:
                raise ValueError("slice step is not supported")
            case slice() as x:
                return "-".join(list(self.elements.values())[x])
            case int() as index:
                return list(self.elements.values())[index]
            case _:
                raise TypeError("index must be integer or slice object")


def _clean_isbn_string(x: MaybeISBN) -> MaybeISBN:
    return x.replace("-", "").replace(" ", "")


def ensure_isbn12(value: int | np.integer) -> ISBN12:
    """Ensures that a value is an ISBN-12 number."""
    if not isinstance(value, (int, np.integer)):
        raise TypeError(f"value must be integer, got {type(value)}")
    if not (FIRST_ISBN <= value <= LAST_ISBN):
        raise NotISBN12Error(
            f"value must be in range [{FIRST_ISBN}, {LAST_ISBN}], got {value}"
        )
    return value


def calculate_check_digit(isbn: CanonicalISBN | ISBN12) -> int:
    """Calculates the check digit for an ISBN."""
    isbn_value = str(isbn)[:ISBN12_LENGTH]
    total = 0
    for i in range(len(isbn_value)):
        digit = int(isbn_value[i])
        if i % 2 == 0:
            total += digit
        else:
            total += digit * 3
    return (10 - (total % 10)) % 10


def normalize_isbn(value: MaybeISBN | ISBN12, correct: bool = False) -> CanonicalISBN:
    """Normalizes an ISBN to a canonical form.

    Arguments:
        value: An value to normalize.
        correct: Whether to check and replace the incorrect check digit with the
            correct one. Defaults to `False`.

    Examples:
        1) The input ISBN-13 string with the correct check digit:

          >>> normalize_isbn("978-000-000-000-2")
          CanonicalISBN(9780000000002)

        2) The input ISBN-13 string with the incorrect check digit:

          >>> # Keep it as it is
          >>> normalize_isbn("978-000-000-000-3")
          CanonicalISBN(9780000000003)

          >>> # Or, correct the check digit
          >>> normalize_isbn("978-000-000-000-3", correct=True)
          CanonicalISBN(9780000000002)

        3) The input :data:`ISBN12` integer value:

          >>> value: ISBN12 = 978_000_000_000
          >>> normalize_isbn(value)
          CanonicalISBN(9780000000002)
    """
    match value:
        case int() | np.integer():
            isbn12 = ensure_isbn12(value)
            normalized_isbn = CanonicalISBN._from_normalized(f"{isbn12}X").complete()
        case str():
            normalized_isbn = CanonicalISBN(cast("str", value))
            if normalized_isbn.check_digit != "X" and correct:
                normalized_isbn = normalized_isbn.complete()
        case _:
            pass
    return normalized_isbn


def match_isbn(x: MaybeISBN) -> bool:
    """Checks that the input string is an ISBN-10 or ISBN-13."""
    return bool(re.match(ISBN_PATTERN, _clean_isbn_string(x)))


def validate_isbn(
    value: CanonicalISBN, *, return_reasons: bool = False
) -> bool | tuple[bool, list[InvalidISBNReason]]:
    """Validates a canonical ISBN.

    The validation includes checks for: (1) an undefined registration group, (2) an
    undefined registrant, and (3) an incorrect check digit. To check if a string is an
    ISBN or not, see :meth:`match_isbn` instead.

    Arguments:
        value: A canonical ISBN to validate.
        return_reasons: Whether to return reasons for failed checks or not.

    Returns:
        A boolean value or, if ``return_reasons`` is requested, a boolean value
        and a list of reasons for failed checks.

    See Also:
        :meth:`match_isbn`

    Examples:
        If you need to check a list of unsure ISBNs for validity, you can
        combine this function with :meth:`match_isbn`::

          def check_isbn_for_errors(x: MaybeISBN) -> list[InvalidISBNReason]:
              if not match_isbn(x):
                  return [InvalidISBNReason.NOT_ISBN]
              else:
                  canonical_isbn = normalize_isbn(x)
                  _, errors = validate_isbn(canonical_isbn, return_reasons=True)
                  return errors
    """
    if not isinstance(value, CanonicalISBN):
        raise TypeError("value should be of type 'CanonicalISBN'")

    errors: list[InvalidISBNReason] = []

    # Check if the ISBN contains a correct check digit
    current_check_digit = value[-1]
    if current_check_digit.isnumeric():
        calculated_check_digit = calculate_check_digit(value)
        if current_check_digit != str(calculated_check_digit):
            if return_reasons:
                errors.append(InvalidISBNReason.BAD_CHECK_DIGIT)
            else:
                return False

    # Check if the ISBN is of a valid registration group and not in any
    # undefined registrant ranges
    try:
        MaskedISBN.from_canonical(value)
    except InvalidISBNError as e:
        if return_reasons:
            errors.append(e.reason)
        else:
            return False

    if return_reasons:
        return not bool(errors), errors

    return True


def get_prefix_bounds(prefix: str) -> tuple[ISBN12, ISBN12]:
    """Gets the bounds for an ISBN prefix."""
    prefix = prefix.replace("-", "")
    return (
        int(prefix.ljust(ISBN12_LENGTH, "0")),
        int(prefix.ljust(ISBN12_LENGTH, "9")),
    )


def get_prefix_capacity(prefix: str) -> int:
    """Gets the capacity of an ISBN prefix."""
    return 10 ** (ISBN12_LENGTH - len(prefix.replace("-", "")))


def extract_group_prefix(isbn: CanonicalISBN) -> str:
    """Extracts the group prefix from the ISBN.

    Arguments:
        isbn: A canonical ISBN-13 string.

    Returns:
        A found full ISBN group prefix, e.g. '978-1'.

    Raises:
        :class:`~allisbns.errors.InvalidISBNError`: If the ISBN does not
        contain valid group prefix.

    Note:
        Group ranges were periodically updated from the RangeMessage.xml file
        available from the International ISBN Agency and may not be the most
        recent.

    See Also:
        :class:`MaskedISBN`
    """
    bookland_length = 3
    max_group_length = 5

    isbn_string = str(isbn)

    for group_length in range(1, max_group_length + 1):
        group = isbn_string[bookland_length : bookland_length + group_length]
        group_prefix = f"{isbn_string[:bookland_length]}-{group}"
        if group_prefix in REGISTRATION_GROUPS:
            break
    else:
        raise UndefinedISBNRangeError(
            f"invalid (or unknown) registration group: {group_prefix}",
            InvalidISBNReason.BAD_GROUP,
            isbn,
        )

    return group_prefix


def generate_isbns(
    start_isbn: ISBN12 | None = None, limit: int | None = None
) -> Generator[CanonicalISBN]:
    """Yields canonical ISBN-13 string values.

    Arguments:
        start_isbn: A start ISBN-12 number. Defaults to `None`, meaning yields
            from the first ISBN-13, '978-000-000-000-2'.
        limit: A maximum number of ISBNs to yields. Defaults to `None`, meaning
            yields till the last ISBN-13, '979-999-999-999-0'.

    """
    if start_isbn is None:
        start_isbn = FIRST_ISBN
    start_isbn = int(start_isbn)

    if not (FIRST_ISBN <= start_isbn <= LAST_ISBN):
        raise ValueError(
            f"start value must be in range [{FIRST_ISBN:_d}, {LAST_ISBN:_d}], "
            f"got {start_isbn:_d}"
        )

    if limit is not None:
        end_isbn: ISBN12 = start_isbn + limit - 1
        end_isbn = min(end_isbn, LAST_ISBN)
    else:
        end_isbn = LAST_ISBN

    for isbn in range(int(start_isbn), int(end_isbn) + 1):
        yield normalize_isbn(isbn)
