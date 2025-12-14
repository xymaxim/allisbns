import pytest

from allisbns.plotting import BlockCoordinateConverter, RowCoordinateConverter


ROW_CONVERTER_CASES = [
    ((0, 0), 978_000_000_000),
    ((0, 150), 978_300_000_000),
    ((999, 499), 978_999_998_000),
]


BLOCK_CONVERTER_CASES = [
    ((0, 0), 978_000_000_000),
    ((300, 0), 978_300_000_000),
    ((999, 499), 978_999_998_000),
]


@pytest.fixture
def row_coordinate_converter():
    return RowCoordinateConverter(
        width=int(2e6),
        bin_size=2000,
        offset=978_000_000_000,
    )


@pytest.fixture
def block_coordinate_converter():
    return BlockCoordinateConverter(
        block_width=int(1e5),
        block_size=int(5e7),
        bin_size=2000,
        offset=978_000_000_000,
    )


@pytest.mark.parametrize("xy, isbn", ROW_CONVERTER_CASES)
def test_row_xy_to_isbn(xy, isbn, row_coordinate_converter):
    assert isbn == row_coordinate_converter.xy_to_isbn(*xy)


@pytest.mark.parametrize("xy, isbn", ROW_CONVERTER_CASES)
def test_row_isbn_to_xy(xy, isbn, row_coordinate_converter):
    assert xy == row_coordinate_converter.isbn_to_xy(isbn)


@pytest.mark.parametrize("xy, isbn", BLOCK_CONVERTER_CASES)
def test_block_xy_to_isbn(xy, isbn, block_coordinate_converter):
    assert isbn == block_coordinate_converter.xy_to_isbn(*xy)


@pytest.mark.parametrize("xy, isbn", BLOCK_CONVERTER_CASES)
def test_block_isbn_to_xy(xy, isbn, block_coordinate_converter):
    assert xy == block_coordinate_converter.isbn_to_xy(isbn)
