"""Classes and functions to work with datasets of packed ISBN codes."""

from __future__ import annotations

import math

from dataclasses import InitVar, dataclass, field
from pathlib import Path
from struct import unpack
from typing import IO, TYPE_CHECKING, BinaryIO, NamedTuple, Self

import numpy as np
import numpy.typing as npt

from bencodepy import bread  # type: ignore[import-untyped]
from zstandard import ZstdCompressor, ZstdDecompressor

from allisbns.isbn import FIRST_ISBN, ISBN12


if TYPE_CHECKING:
    from collections.abc import Generator, KeysView


#: Packed ISBN codes that represent ISBN availability.
type PackedCodes = npt.NDArray[np.int32]

#: Unpacked representation of codes.
type UnpackedCodes = npt.NDArray[np.bool]


def load_bencoded(source: IO[bytes]) -> dict[bytes, bytes]:
    """Opens a compressed source and reads a bencoded data."""
    with ZstdDecompressor().stream_reader(source) as decompressor:
        return bread(decompressor)


def unpack_data(data: bytes) -> PackedCodes:
    """Unpacks data that come from a bencoded source."""
    return np.array(unpack(f"{len(data) // 4}I", data), dtype=np.int32)


@dataclass(frozen=True)
class BinnedArray:
    """Represents the binned data."""

    #: Bins.
    bins: npt.ArrayLike
    #: A size of bins.
    bin_size: int

    def __post_init__(self):
        object.__setattr__(self, "bins", np.asarray(self.bins))

    def __array__(
        self, dtype: npt.DTypeLike | None = None, copy: bool | None = None
    ) -> npt.NDArray:
        """Returns the underlying bins for NumPy when requested."""
        return np.asarray(self.bins, dtype=dtype, copy=copy)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.bins!r}, bin_size={self.bin_size})"


class QueryResult(NamedTuple):
    """Represents the result of a query for an ISBN.

    It shows whether the ISBN is filled (falls in a streak segment) or absent
    (in a gap segment) and the corresponding position in a segment.
    """

    #: Is the number in a streak or gap?
    is_streak: bool
    #: The index of the corresponding segment.
    segment_index: int
    #: The position in the corresponding segment.
    position_in_segment: int


@dataclass(frozen=True)
class CodeDataset:
    """Represents a dataset of the packed ISBN codes.

    Examples:
        Create a dataset from the input codes:

          >>> dataset = CodeDataset(codes=[1, 2, 3])
          >>> dataset
          CodeDataset(array([1, 2, 3], dtype=int32), bounds=(978000000000,
          978000000005))
          >>> dataset.codes
          array([1, 2, 3], dtype=int32)

        With the custom offset and fill to some ISBN:

          >>> CodeDataset(
          ...     codes=[1, 2, 3],
          ...     offset=979_000_000_000,
          ...     fill_to_isbn=979_999_999_999
          ... )
          CodeDataset(array([ 1, 2, 3, 999999994]), bounds=(979000000000,
          979999999999))
    """

    #: Packed ISBN codes.
    codes: npt.NDArray[np.int32]

    #: First ISBN in the dataset.
    offset: ISBN12 = FIRST_ISBN

    #: ISBN up to which to fill dataset codes.
    fill_to_isbn: InitVar[ISBN12 | None] = None

    #: First and last ISBNs in the dataset.
    bounds: tuple[ISBN12, ISBN12] = field(init=False)

    #: Cumulative sums of ISBNs derived from the codes.
    _isbn_cumsums: npt.NDArray = field(init=False)

    def __post_init__(self, fill_to_isbn: ISBN12 | None) -> None:
        object.__setattr__(self, "codes", np.asarray(self.codes, dtype=np.int32))

        if fill_to_isbn:
            self._fill_to_isbn(fill_to_isbn)

        object.__setattr__(
            self, "_isbn_cumsums", int(self.offset) + np.cumsum(self.codes)
        )
        object.__setattr__(
            self, "bounds", (self.offset, int(self._isbn_cumsums[-1] - 1))
        )

    def _fill_to_isbn(self, isbn: ISBN12) -> None:
        right_bound = self.offset + np.sum(self.codes) - 1
        if isbn < right_bound:
            message = f"fill ISBN ({isbn}) must be beyond right bound ({right_bound})"
            raise ValueError(message)
        if isbn == right_bound:
            return

        # Check if codes end with a gap segment
        deficiency = isbn - right_bound
        if len(self.codes) % 2 == 0:
            self.codes[-1] += deficiency
        else:
            object.__setattr__(
                self, "codes", np.concatenate([self.codes, [deficiency]])
            )

    @classmethod
    def from_file(
        cls,
        source: str | Path | IO[bytes],
        collection: str,
        offset: ISBN12 = FIRST_ISBN,
        fill_to_isbn: ISBN12 | None = None,
    ) -> Self:
        """Creates a dataset from a source file or byte stream.

        Arguments:
            source: A path to a bencoded compressed file or a byte stream.
            collection: A collection name to be read (e.g., 'md5', 'rgb',
                etc.). Refers to ``aarecord_id_prefix`` in the original data
                format.
            offset: The first ISBN.
            fill_to_isbn: An ISBN up to which to fill dataset codes.

        Returns:
            A dataset created from a source.
        """
        if isinstance(source, (str, Path)):
            with open(source, "rb") as f:
                input_data = load_bencoded(f)
        else:
            input_data = load_bencoded(source)
        try:
            codes = unpack_data(input_data[collection.encode()])
        except KeyError as exc:
            message = (
                f"'{collection}' not present. "
                f"Available collections: {list(input_data.keys())}"
            )
            raise ValueError(message) from exc
        return cls(codes=codes, offset=offset, fill_to_isbn=fill_to_isbn)

    @classmethod
    def from_unpacked(
        cls, unpacked_codes: UnpackedCodes, offset: ISBN12 = FIRST_ISBN
    ) -> Self:
        """Creates a dataset from the unpacked codes.

        Arguments:
            unpacked_codes: An array of unpack codes (boolean values).
            collection: A dataset collection name.
            offset: The first ISBN.

        Returns:
            A dataset created from the unpacked codes.
        """
        # This marks elements where the values change, i.e. the starts of the segments.
        change_mask = np.zeros(len(unpacked_codes) + 1, dtype=np.bool)
        change_mask[[0, -1]] = True
        np.not_equal(unpacked_codes[:-1], unpacked_codes[1:], out=change_mask[1:-1])
        packed_codes = np.diff(
            np.flatnonzero(change_mask),
            # If data starts with a gap, then prepend the packed codes with 0.
            prepend=np._NoValue if unpacked_codes[0] else [unpacked_codes[0]],
        )
        return cls(codes=packed_codes, offset=offset)

    def unpack_codes(self) -> UnpackedCodes:
        """Unpacks codes into boolean values.

        Returns:
            An array of unpacked codes.
        """
        tiles = np.zeros(len(self.codes), dtype=bool)
        tiles[::2] = True
        return np.repeat(tiles, self.codes)

    def crop(self, start: ISBN12, end: ISBN12) -> Self:
        """Crops the dataset to the input start and end ISBNs.

        Arguments:
            start: An ISBN to crop the dataset from.
            end: An ISBN to crop the dataset until.

        Returns:
            A new cropped dataset.
        """
        if start > end:
            raise ValueError("start is ahead of end")

        try:
            is_start_streak, start_idx, _ = self.query_isbn(start)
        except ValueError:
            start = self.bounds[0]
            is_start_streak = self.codes[0] != 0
            start_idx = 0
        try:
            _, end_idx, end_position = self.query_isbn(end)
        except ValueError:
            end = self.bounds[1]
            end_idx = len(self.codes) - 1
            end_position = self.codes[end_idx] - 1

        if start_idx == end_idx:
            segment_length = end - start + 1
            cropped = np.array([segment_length], self.codes.dtype)
        else:
            cropped = self.codes[start_idx : end_idx + 1].copy()
            cropped[0] = self._isbn_cumsums[start_idx] - start
            cropped[-1] = end_position + 1

        if not is_start_streak:
            cropped = np.concatenate([[0], cropped])

        return self.__class__(cropped, offset=start)

    def invert(self) -> Self:
        """Inverts the dataset by making streak segments gap."""
        return self.__class__(
            codes=np.concatenate([[0], self.codes]),
            offset=self.offset,
        )

    def query_isbn(self, isbn: ISBN12) -> QueryResult:
        """Queries if the ISBN is filled in the dataset and its position.

        Arguments:
            isbn: An ISBN to query for.

        Returns:
            A query result.
        """
        if not (self.bounds[0] <= isbn <= self.bounds[1]):
            raise ValueError("input ISBN is outside of the data range")

        segment_index = int(np.searchsorted(self._isbn_cumsums, isbn, side="right"))
        is_streak = bool(segment_index % 2 == 0)
        position_in_segment = int(
            isbn - self._isbn_cumsums[segment_index] + self.codes[segment_index]
        )

        return QueryResult(is_streak, segment_index, position_in_segment)

    def check_isbns(self, isbns: list[ISBN12]) -> npt.NDArray[np.bool]:
        """Checks if ISBNs are filled in the dataset or not.

        Arguments:
            isbns: A list with ISBNs to check.

        Returns:
            An array of boolean values.
        """
        isbns_array = np.array(isbns)
        # Check if all input ISBNs are inside data range
        outside_mask = (isbns_array < self.bounds[0]) | (isbns_array > self.bounds[1])
        if np.any(outside_mask):
            raise ValueError(
                f"some ISBNs are outside of data range: {isbns_array[outside_mask]}"
            )

        segment_indices = np.searchsorted(self._isbn_cumsums, isbns_array, side="right")

        return segment_indices % 2 == 0

    def get_filled_isbns(self) -> npt.NDArray:
        """Gets filled ISBNs in the dataset.

        Returns:
            An array of filled ISBNs.

        Examples:
            To get unfilled ISBNs, invert the dataset first::

              dataset.invert().get_filled_isbns()
        """
        return np.flatnonzero(self.unpack_codes()) + self.offset

    def count_filled_isbns(self) -> int:
        """Counts the number of filled ISBNs in the dataset."""
        return int(np.sum(self.codes[::2]))

    def bin(
        self,
        bin_size: int = 2500,
        num_chunks: int = 4,
    ) -> BinnedArray:
        """Performs a fixed-size binning of codes into bins.

        The bin value is the number of filled ISBN values.

        Arguments:
            bin_size: A number of ISBNs in one bin.
            num_chunks: A number of chunks used to process the dataset.

        Returns:
            A binned array.
        """
        num_bins = math.ceil((self.bounds[1] - self.bounds[0] + 1) / bin_size)
        num_bins_per_chunk = math.ceil(num_bins / num_chunks)
        num_isbns_per_chunk = num_bins_per_chunk * bin_size

        all_bins = np.zeros(num_bins_per_chunk * num_chunks, dtype=np.int32)

        for chunk_idx in range(num_chunks):
            chunk_start_isbn = chunk_idx * num_isbns_per_chunk + self.offset
            chunk_end_isbn = chunk_start_isbn + num_isbns_per_chunk - 1
            unpacked_codes = self.crop(chunk_start_isbn, chunk_end_isbn).unpack_codes()
            unpacked_length = len(unpacked_codes)
            if unpacked_length != num_isbns_per_chunk:
                unpacked_codes = np.pad(
                    unpacked_codes,
                    (0, num_isbns_per_chunk - unpacked_length),
                    constant_values=0,
                )
            chunk_shift = chunk_idx * num_bins_per_chunk
            all_bins[chunk_shift : chunk_shift + num_bins_per_chunk] = np.sum(
                unpacked_codes.reshape((num_bins_per_chunk, bin_size)), axis=1
            )

        return BinnedArray(all_bins[:num_bins], bin_size)

    def is_subset(self, other: Self) -> bool:
        """Determines if a dataset is a subset of another."""
        if self.bounds[0] < other.bounds[0] or self.bounds[1] > other.bounds[1]:
            return False
        return self == other.crop(*self.bounds)

    def write_bencoded(
        self, file: str | Path | IO[bytes], prefix: str, normalize: bool = True
    ) -> None:
        """Writes ISBN codes to a bencoded compressed file.

        Arguments:
            file: A file path or file-like object to which the codes will be
                written.
            prefix: A dataset collection name. Refers to 'aarecord_id_prefix'
                in the original data format description.
            normalize: Whether to normalize codes so that the ending gap code,
                if present, is omitted (default) or not.
        """
        output_stream: IO[bytes] | BinaryIO
        if isinstance(file, (str | Path)):
            output_stream = open(file, "wb")  # noqa: SIM115
            needs_closing = True
        else:
            output_stream = file
            needs_closing = False

        try:
            codes_to_write = self.codes.copy()
            if normalize and len(self.codes) % 2 == 0:
                codes_to_write = codes_to_write[:-1]

            # Convert the codes as little-endian 4-bytes unsigned integers to bytes
            codes_as_bytes = codes_to_write.astype("<u4").tobytes()

            compressor = ZstdCompressor(level=22, threads=-1)
            with compressor.stream_writer(output_stream, closefd=False) as writer:
                writer.write(b"d")
                writer.write(f"{len(prefix)}:{prefix}".encode())
                writer.write(f"{len(codes_as_bytes)}:".encode())
                writer.write(codes_as_bytes)
                writer.write(b"e")
        finally:
            if needs_closing:
                output_stream.close()

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return NotImplemented
        return np.array_equal(self.codes, other.codes) and self.offset == other.offset

    def __hash__(self) -> int:
        return hash((self.codes, self.offset))

    def __getitem__(self, key: slice[ISBN12, ISBN12, None]) -> Self:
        if isinstance(key, slice):
            if key.step:
                raise ValueError("slice step is not supported")
            return self.crop(key.start, key.stop)
        raise ValueError("Object supports only slicing")

    def __invert__(self) -> Self:
        return self.invert()

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.codes!r}, "
            f"bounds=({self.bounds[0]}, {self.bounds[1]}))"
        )


def iterate_datasets(
    data: dict[bytes, bytes],
    collections: list[str] | None = None,
    fill_to_isbn: ISBN12 | None = None,
) -> Generator[CodeDataset]:
    """Iterates over datasets created from the loaded bencoded data.

    By default, iterates over all collections in the data.

    Arguments:
        data: Loaded bencoded data that contain collections to unpack.
        collections: Collection names to unpack (e.g. 'md5', 'rgb', ...). When
            `None` (default), iterates over all collections in the data.
        fill_to_isbn: An ISBN up to which to fill dataset codes.

    Returns:
        Yields datasets for the selected collections.

    Example:
        Iterate over all datasets and count filled ISBNs::

          from allisbns.dataset import load_bencoded, iterate_datasets

          with open("aa_isbn13_codes_20251118T170842Z.benc.zst", "rb") as f:
              input_data = load_bencoded(f)

          filled_counts: dict[str, int] = {}

          collections = [x.decode() for x in input_data.keys()]
          dataset_iterator = iterate_datasets(input_data, collections)
          for collection, dataset in zip(collections, dataset_iterator):
              filled_counts[collection] = dataset.count_filled_isbns()
    """
    keys: list[bytes] | KeysView[bytes]
    if collections:
        keys = [x.encode() for x in collections]
    else:
        keys = data.keys()

    for key in keys:
        yield CodeDataset(
            codes=unpack_data(data[key]),
            fill_to_isbn=fill_to_isbn,
        )
