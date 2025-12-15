"""Supports plotting of binned images."""

import math

from abc import ABC, abstractmethod
from typing import Any, Literal, cast

import matplotlib as mpl
import matplotlib.typing as mpt
import numpy as np
import numpy.typing as npt

from matplotlib.axes import Axes

from allisbns.dataset import BinnedArray
from allisbns.isbn import FIRST_ISBN, ISBN12, LAST_ISBN
from allisbns.rearrange import rearrange_to_blocks, rearrange_to_rows


type MatplotlibExtent = tuple[float, float, float, float]


def truncate_colormap(
    colormap, vmin: float = 0, vmax: float = 0.98
) -> mpl.colors.Colormap:
    return mpl.colors.ListedColormap(
        colormap(np.linspace(vmin, vmax, colormap.N)), f"{colormap.name}_truncated"
    )


def get_default_colormap(
    nan_color: mpt.ColorType = "0.6", under_color: mpt.ColorType = "0.45"
) -> mpl.colors.Colormap:
    colormap = truncate_colormap(mpl.colormaps.get_cmap("plasma"))
    colormap.set_bad(nan_color)
    colormap.set_under(under_color)
    return colormap


def tweak_colormap(
    colormap: mpl.colors.Colormap,
    nan_color: mpt.ColorType | None = None,
    under_color: mpt.ColorType | None = None,
) -> mpl.colors.Colormap:
    if nan_color:
        colormap.set_bad(nan_color)
    if under_color:
        colormap.set_under(under_color)
    return colormap


def translate_with_padding(ax, transform, *, x: float = 0, y: float = 0):
    physical_transform = mpl.transforms.ScaledTranslation(
        x, y, ax.get_figure().dpi_scale_trans
    )
    return transform + physical_transform


def pad_along_x(ax, value: float):
    transform = mpl.transforms.blended_transform_factory(ax.transAxes, ax.transData)
    return translate_with_padding(ax, transform, x=value)


def pad_along_y(ax, value: float):
    transform = mpl.transforms.blended_transform_factory(ax.transData, ax.transAxes)
    return translate_with_padding(ax, transform, y=value)


class CoordinateConverter(ABC):
    """Represents a coordinate converter for bins rearranged in various ways."""

    @abstractmethod
    def xy_to_isbn(self, x: int, y: int) -> ISBN12:
        """Converts coordinates to an ISBN."""

    @abstractmethod
    def isbn_to_xy(self, isbn: ISBN12) -> tuple[int, int]:
        """Converts an ISBN to coordinates."""


class RowCoordinateConverter(CoordinateConverter):
    """Represents a coordinate converter for bins rearranged into rows."""

    def __init__(self, width: int, bin_size: int, offset: ISBN12):
        """Creates a coordinate converter.

        Arguments:
            width: A width of rows.
            bin_size: A size of bins.
            offset: An ISBN offset.
        """
        self.width = int(width)
        self.bin_size = int(bin_size)
        self.offset = int(offset)

    def xy_to_isbn(self, x: int, y: int) -> ISBN12:
        """Converts image coordinates to an ISBN."""
        return self.offset + y * self.width + x * self.bin_size

    def isbn_to_xy(self, isbn: ISBN12) -> tuple[int, int]:
        """Converts an ISBN to image coordinates."""
        bin_index = (isbn - self.offset) // self.bin_size
        y, x = divmod(bin_index, self.width // self.bin_size)
        return x, y


class BlockCoordinateConverter(CoordinateConverter):
    """Represents a coordinate converter for bins rearranged into blocks."""

    def __init__(
        self, block_width: int, block_size: int, bin_size: int, offset: ISBN12
    ):
        """Creates a coordinate converter.

        Arguments:
            block_width: A width of one block.
            block_size: A number of ISBNs in one block.
            bin_size: A size of bins.
            offset: An ISBN offset.
        """
        self.block_width = int(block_width)
        self.block_size = int(block_size)
        self.bin_size = int(bin_size)
        self.offset = offset

        self._bins_per_row = (block_width + bin_size - 1) // bin_size
        self._rows_per_block = (block_size + block_width - 1) // block_width

    def xy_to_isbn(self, x: int, y: int) -> ISBN12:
        block_index, subcolumn_index = divmod(x, self._bins_per_row)

        block_offset = block_index * self.block_size
        row_offset = y * self.block_width
        subcolumn_offset = subcolumn_index * self.bin_size

        return self.offset + block_offset + row_offset + subcolumn_offset

    def isbn_to_xy(self, isbn: ISBN12) -> tuple[int, int]:
        bin_index = (isbn - self.offset) // self.bin_size

        bins_per_block = self._bins_per_row * self._rows_per_block
        block_index, bin_in_block = divmod(bin_index, bins_per_block)
        y, subcolumn_index = divmod(bin_in_block, self._bins_per_row)

        x = block_index * self._bins_per_row + subcolumn_index

        return x, y


class BinnedPlotter(ABC):
    """Represents a base plotter for bins."""

    def __init__(
        self,
        ax,
        *,
        bin_size: int,
        offset: ISBN12 = FIRST_ISBN,
        aspect: Literal["auto", "equal"] | float = "equal",
        decorate_axes: bool = True,
    ):
        """Creates a plotter for bins.

        Arguments:
            ax: An axis into which to plot.
            bin_size: A size of bins.
            offset: An ISBN offset.
            aspect: An aspect of the image.
            decorate_axes: Whether to decorate axes with the plotter's specific
                axis and tick labels.
        """
        self.ax = ax
        self.bin_size = int(bin_size)
        self.offset = offset
        self.aspect = aspect
        self.decorate_axes = decorate_axes

        self._extent: MatplotlibExtent | None = None

        vmin = 20
        self.default_imshow_kwargs = {
            "vmin": min(bin_size, vmin),
            "vmax": bin_size,
            "interpolation": "none",
        }
        self.default_colorbar_kwargs = {
            "ax": self.ax,
            "label": "ISBNs/bin",
            "orientation": "horizontal",
            "location": "top",
            "aspect": 28,
            "shrink": 0.28,
            "anchor": (0, 0),
            "pad": 0.07,
            "extendrect": True,
        }

    @property
    def extent(self) -> MatplotlibExtent | None:
        """The extent of the image as tuple ``(left, right, bottom, top)``."""
        return self._extent

    def _style_axes(self) -> None:
        self.ax.minorticks_on()
        self.ax.yaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(1))

        self.ax.spines[:].set_visible(False)

        self.ax.xaxis.tick_top()
        self.ax.xaxis.set_label_position("top")

        self.ax.xaxis.set_ticks_position("both")
        self.ax.yaxis.set_ticks_position("both")

    @abstractmethod
    def _decorate_axes(self):
        """Decorate axes with the plotter's specific axis and tick labels."""

    def _check_bin_size(self, binned: BinnedArray) -> None:
        if binned.bin_size != self.bin_size:
            raise ValueError(
                f"size of bins ({binned.bin_size}) should match "
                f"the plotter's value ({self.bin_size})"
            )

    def plot_image(
        self,
        image: npt.NDArray,
        *,
        colormap: mpl.colors.Colormap | None = None,
        value_to_nan: int | None = 0,
        show_colorbar: bool = False,
        colorbar_kwargs: dict[str, Any] | None = None,
        **imshow_kwargs,
    ):
        """Plots an image.

        Arguments:
            image: An image array to plot.
            colormap: A colormap.
            value_to_nan: A value that will be replaced by :attr:`numpy:numpy.nan`.
            show_colorbar: Whether to add a colorbar to an image.
            colorbar_kwargs: Additional arguments accepted by
                :func:`~matplotlib:matplotlib.pyplot.colorbar`.
            imshow_kwargs: Additional arguments accepted by
                :func:`~matplotlib:matplotlib.pyplot.imshow`.
        """
        imshow_kwargs = self.default_imshow_kwargs | imshow_kwargs

        if "norm" in imshow_kwargs:
            raise NotImplementedError("Currently, 'norm' argument is not supported")

        vmin = cast("int", imshow_kwargs["vmin"])
        vmax = cast("int", imshow_kwargs["vmax"])
        if vmax is not None and vmin > vmax:
            raise ValueError("vmin cannot be greater than vmax")

        colorbar_kwargs = self.default_colorbar_kwargs | (colorbar_kwargs or {})
        if "extend" not in colorbar_kwargs:
            colorbar_kwargs["extend"] = "min" if imshow_kwargs["vmin"] else "neither"

        image = image.astype(np.float32)

        # Update the extent
        image_height, image_width = image.shape[:2]
        image_extent = (0, image_width, image_height, 0)

        if self.extent is None:
            self._extent = image_extent
        elif self.extent != image_extent:
            raise ValueError(
                f"image extent, {image_extent}, should be equal"
                f"to plotter extent, {self.extent}"
            )

        # Draw the image
        if value_to_nan is not None:
            image[image == value_to_nan] = np.nan

        if colormap is None:
            colormap = get_default_colormap()

        im = self.ax.imshow(
            image,
            cmap=colormap,
            extent=self.extent,
            aspect=self.aspect,
            **imshow_kwargs,
        )

        # Show the colorbar
        if show_colorbar:
            colorbar = self.ax.get_figure().colorbar(im, **colorbar_kwargs)
            vmin = cast("int", imshow_kwargs["vmin"])
            vmax = cast("int", imshow_kwargs["vmax"])

            if vmin > 0:
                colorbar.ax.set_xticks([vmin, *colorbar.ax.get_xticks()[1:]])
                colorbar.ax.set_xticklabels(
                    [f"{vmin:.0f}", *colorbar.ax.get_xticklabels()[1:]]
                )

            colorbar.ax.tick_params(axis="x", direction="in")
            colorbar.ax.set_xlim(vmin, vmax)

        return im

    @abstractmethod
    def plot_bins(
        self,
        binned: BinnedArray,
        colormap: mpl.colors.Colormap | None = None,
        value_to_nan: int | None = 0,
        show_colorbar: bool = True,
        colorbar_kwargs: dict | None = None,
        **imshow_kwargs,
    ):
        pass

    @abstractmethod
    def define_extent(self, end_isbn: ISBN12 = LAST_ISBN) -> None:
        """Defines the extent without plotting bins or an image.

        Arguments:
            end_isbn: An end ISBN to expand an extent to from the plotter offset.
        """


class RowBinnedPlotter(BinnedPlotter):
    """Represents a plotter for bins with a fixed width of rows."""

    def __init__(self, ax: Axes, bin_size: int, width: int = int(2.5e6), **kwargs):
        """Creates a plotter for bins with a fixed width of rows.

        Internally, bins are rearranged into an image with the
        :func:`~allisbns.rearrange.rearrange_to_rows` method.

        Arguments:
            ax: An axis into which to plot.
            bin_size: A size of bins.
            width: A width of rows.
            kwargs: Additional arguments to pass to :class:`BinnedPlotter`.
        """
        super().__init__(ax, bin_size=bin_size, **kwargs)

        self.width = int(width)
        self.coordinate_converter = RowCoordinateConverter(
            self.width, self.bin_size, self.offset
        )

        self._style_axes()
        if self.decorate_axes:
            self._decorate_axes()

    def _decorate_axes(self):
        xticks_formatter = mpl.ticker.FuncFormatter(
            lambda x, _: f"{x * self.bin_size:,.0f}"
        )
        xticks_formatter.set_offset_string(
            rf"ISBN = {self.offset:,.0f}(X) + $x$ $\times$ ($y$ + 1)"
        )
        self.ax.xaxis.set_major_formatter(xticks_formatter)
        offset_text = self.ax.xaxis.get_offset_text()
        offset_text.set_x(0)
        offset_text.set_horizontalalignment("left")

        self.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{x:.0f}")
        )

        self.ax.xaxis.set_minor_locator(mpl.ticker.AutoMinorLocator(5))

        self.ax.set_xlabel("Relative ISBN, $x$")
        self.ax.set_ylabel("Row, $y$")

    def define_extent(self, end_isbn: ISBN12 = LAST_ISBN) -> None:
        """Defines the extent without plotting bins or an image.

        Arguments:
            end_isbn: An end ISBN to expand an extent to from the plotter offset.
        """
        x, y = self.coordinate_converter.isbn_to_xy(end_isbn)

        self._extent = (0, x + 1, y + 1, 0)
        x1, x2, y1, y2 = self._extent

        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)

        self.ax.set_aspect(self.aspect)

    def plot_bins(
        self,
        binned: BinnedArray,
        colormap: mpl.colors.Colormap | None = None,
        value_to_nan: int | None = 0,
        show_colorbar: bool = True,
        colorbar_kwargs: dict | None = None,
        **imshow_kwargs,
    ):
        self._check_bin_size(binned)
        image = rearrange_to_rows(binned, binned.bin_size, self.width)
        return self.plot_image(
            image,
            colormap=colormap,
            value_to_nan=value_to_nan,
            show_colorbar=show_colorbar,
            colorbar_kwargs=colorbar_kwargs,
            **imshow_kwargs,
        )


class BlockBinnedPlotter(BinnedPlotter):
    """Represents a plotter for bins stacked as vertical blocks."""

    def __init__(
        self,
        ax: Axes,
        bin_size: int,
        block_width: int = int(1e5),
        block_size: int = int(5e7),
        decorate_axes: bool = True,
        **kwargs,
    ):
        """Creates a plotter for bins stacked as vertical blocks.

        Internally, bins are rearranged into an image with the
        :func:`~allisbns.rearrange.rearrange_to_blocks` method.

        Arguments:
            ax: An axis into which to plot.
            bin_size: A size of bins.
            block_width: A width of one block.
            block_size: A number of ISBNs in one block.
            decorate_axes: Whether to decorate axes with the plotter's specific
                axis and tick labels.
            kwargs: Additional arguments to pass to :class:`BinnedPlotter`.
        """
        super().__init__(ax, bin_size=bin_size, **kwargs)

        self.block_width = int(block_width)
        self.block_size = int(block_size)

        self.coordinate_converter = BlockCoordinateConverter(
            self.block_width, self.block_size, self.bin_size, self.offset
        )

        self._style_axes()
        if decorate_axes:
            self._decorate_axes()

    def _draw_scale_bar(self) -> None:
        scale_bar_end = self.block_width / self.bin_size
        scale_bar_center = scale_bar_end / 2
        self.ax.plot(
            [0, scale_bar_end],
            [1, 1],
            ls="-",
            color="k",
            lw=2,
            transform=pad_along_y(self.ax, 0.15),
            clip_on=False,
        )
        self.ax.text(
            x=scale_bar_center,
            y=1,
            s=f"{self.block_width:,d} ISBNs",
            ha="center",
            va="bottom",
            transform=pad_along_y(self.ax, 0.2),
        )

    def _decorate_axes(
        self, xlabel: str = "Relative ISBN", ylabel: str = "Relative ISBN"
    ):
        self.ax.set_xticklabels([])

        self.ax.yaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda y, _: f"{y * self.block_width:.2g}")
        )

        self._draw_scale_bar()

        self.ax.set_xlabel(xlabel)
        self.ax.set_ylabel(ylabel)

        # Create a secondary axis for ISBN prefixes
        ax2 = self.ax.secondary_xaxis(
            location=1,
            functions=(
                lambda x: self.coordinate_converter.xy_to_isbn(x, 0),
                lambda x: self.coordinate_converter.isbn_to_xy(x)[0],
            ),
            transform=pad_along_y(self.ax, 0.4),
        )
        ax2.set_xlabel("ISBN prefix")
        ax2.spines[:].set_visible(False)

        ax2.xaxis.set_major_locator(mpl.ticker.MultipleLocator(self.block_size * 2))
        ax2.xaxis.set_major_formatter(
            mpl.ticker.FuncFormatter(lambda x, _: f"{str(x)[:3]}-{str(x)[3:4]}")
        )

    def define_extent(self, end_isbn: ISBN12 = LAST_ISBN) -> None:
        """Defines the extent without plotting bins or an image.

        Arguments:
            end_isbn: An end ISBN to expand an extent to from the plotter offset.
        """
        x, _ = self.coordinate_converter.isbn_to_xy(end_isbn)

        bins_per_row = math.ceil(self.block_width / self.bin_size)
        block_index = x // bins_per_row
        image_height = (self.block_size // self.bin_size) // bins_per_row
        image_width = (block_index + 1) * bins_per_row

        self._extent = (0, image_width, image_height, 0)

        x1, x2, y1, y2 = self._extent
        self.ax.set_xlim(x1, x2)
        self.ax.set_ylim(y1, y2)

        self.ax.set_aspect(self.aspect)

    def plot_bins(
        self,
        binned: BinnedArray,
        colormap: mpl.colors.Colormap | None = None,
        value_to_nan: int | None = 0,
        show_colorbar: bool = True,
        colorbar_kwargs: dict | None = None,
        **imshow_kwargs,
    ):
        self._check_bin_size(binned)
        image = rearrange_to_blocks(
            binned, binned.bin_size, self.block_width, self.block_size
        )
        return self.plot_image(
            image,
            colormap=colormap,
            value_to_nan=value_to_nan,
            show_colorbar=show_colorbar,
            colorbar_kwargs=colorbar_kwargs,
            **imshow_kwargs,
        )
