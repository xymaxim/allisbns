Overview
########

.. contents:: Contents
   :depth: 1
   :backlinks: top
   :local:

Working with datasets
*********************

Codes
=====

The *packed ISBN codes* are a set of integers that represent the length of the
alternating streak and gap segments, for example, ``3, 2, 1, ...``. It is
similar to the `run-length encoding
<https://en.wikipedia.org/wiki/Run-length_encoding>`__ and efficiently describes
information about ISBN availability.  See `here
<https://software.annas-archive.li/AnnaArchivist/annas-archive/-/blob/9c18fc80341376914da2fe0770b4ed13347143f0/allthethings/cli/views.py#L2007-2019>`__
on how they are generated and for the description of the binary format used to
store them.

In our package we use several terms that relate to ISBN availability. We can
distinguish between *packed* and *unpacked* codes:
:type:`~allisbns.dataset.PackedCodes` refer to the original presentation of the
codes and :type:`~allisbns.dataset.UnpackedCodes` to the decoded one as boolean
values. To distinguish between the individual ISBNs from streak and gap segments,
we call them *filled* and *unfilled* ISBNs, respectively.

The files with codes named as ``aa_isbn13_codes_*.benc.zst`` and occasionally
published by Anna and the team can be downloaded via the
``aa_derived_mirror_metadata`` torrent from `this page
<https://annas-archive.org/torrents/aa_derived_mirror_metadata>`__.

Read datasets
=============

There are several ways to read datasets from the bencoded compressed files.

The **first** way is quick and short. It uncompress the file and extract a single
dataset:

.. code-block:: python

    >>> from allisbns.dataset import CodeDataset
    >>>
    >>> CodeDataset.from_file(
    ...     source="aa_isbn13_codes_20251118T170842Z.benc.zst",
    ...     collection="md5",
    ... )
    CodeDataset(array([    6,     1,     9, ...,     1, 91739,     1],
      shape=(14737375,), dtype=int32), bounds=(978000000000, 979999468900))

The **second** way is more practical when you need to read several datasets from
the same file.

.. code-block:: python

   from allisbns.dataset import load_bencoded, unpack_data

   # Load the bencoded data
   input_path = "aa_isbn13_codes_20251118T170842Z.benc.zst"
   with open(input_path, "rb") as f:
       input_data = load_bencoded(f)

   # Extract the desired datasets
   md5 = CodeDataset(unpack_data(input_data[b"md5"]))
   rgb = CodeDataset(unpack_data(input_data[b"rgb"]))

The **third** way allows you to iterate over datasets in the input data. The
above example can be rewritten with :func:`allisbns.dataset.iterate_datasets` as
the following:

.. code-block:: python

   md5, rgb = iterate_datasets(input_data, ["md5", "rgb"])

See more examples in :func:`~allisbns.dataset.iterate_datasets`'s docstring or
:ref:`this <cookbook-iterate-datasets>` cookbook recipe.

Limit output
============

By design, datasets are considered immutable after creation. Moreover, there is
no way to limit the output of some methods. If you need, for example, to get all
filled ISBNs with prefix '979', then you will need to reframe a dataset before:

.. code-block:: python

   from allisbns.isbn import get_prefix_bounds
   md5.reframe(*get_prefix_bounds("979")).get_filled_isbns()

If you need to modify codes for some reason, you can copy codes and create a new
dataset after editing. For example, here is an equivalent of the
:meth:`~allisbns.dataset.CodeDataset.invert` method to inverse a dataset:

.. code-block:: python

   >>> CodeDataset(np.concatenate([[0], md5.codes]), offset=md5.offset)
   CodeDataset(array([     0,      6,      1, ...,  91739,      1],
      shape=(14737377,), dtype=int32), bounds=(978000000000, 979999468900))

Working with ISBNs
******************

The package provides classes and functions to work with ISBNs, both numeric and
string types.

To simplify things, the methods of :class:`~allisbns.dataset.CodeDataset` only
accepts ISBNs represented as *ISBN-12* values (without the check digit) of
integer types. We omit the range validation there, but you can use
:func:`~allisbns.isbn.ensure_isbn12` to make sure that your values are valid if
needed.

To work with strings, we have two classes, :class:`~allisbns.isbn.CanonicalISBN`
and :class:`~allisbns.isbn.MaskedISBN`.

Normalize ISBNs
===============

Let's say we have some ISBN that comes from anywhere.

.. code-block:: python

   isbn = "978-23-6590-117-X"

It might be delimited by hyphens (correctly or not), contain the incorrect check
digit, or even not be an ISBN at all.

First, let's try to normalize and complete it with the correct check digit if
needed with :func:`~allisbns.isbn.normalize_isbn`.

.. code-block:: python

   # Normalize the ISBN, keep the 'X' check digit
   >>> canonical = normalize_isbn(isbn)
   CanonicalISBN(978236590117X)

   # Complete the ISBN with the check digit
   >>> canonical.complete()
   CanonicalISBN(9782365901178)

Now we are sure that our ISBN value is valid. The output is
:class:`~allisbns.isbn.CanonicalISBN`. We can then, for example, convert it to
the ISBN-12 integer number:

.. code-block:: python

   >>> isbn12 = canonical.to_isbn12()
   >>> isbn12
   978236590117

   >>> # Can be safely used for querying the dataset
   >>> md5.query(isbn12)

Format ISBNs
============

The canonical ISBNs can be formatted with hyphens to separate their elements:

.. code-block:: python

   >>> canonical.hyphen()
   '978-2-36590-117-8'

We have the :class:`~allisbns.isbn.MaskedISBN` class underneath for that: it
validates ISBN ranges and splits ISBNs into distinct elements. In most cases you do
not need to initialize it directly, since creating it from canonical ISBNs is
more practical:

.. code-block:: python

   >>> masked = MaskedISBN.from_canonical(canonical)
   >>> masked
   MaskedISBN(
       bookland='978',
       group='2',
       registrant='36590',
       publication='117',
       check_digit='8',
   )


The masked ISBNs enable slicing to output formatted ISBNs in a granular manner:

.. code-block:: python

   >>> # Get the full publisher prefix
   >>> masked[:3]
   '978-2-36590'


ISBN ranges
===========

As you may notice, the formatted ISBN differs from our initial input string
('978-23-6590-117-X') that is incorrectly formatted. Masking ISBNs is possible
with knowing about the valid ISBN registration groups and registrant ranges. The
valid ranges are available from the `International ISBN Agency
<https://www.isbn-international.org/>`__ website. We store such values in the
auto-generated :mod:`allisbns.ranges` module. Given that, the ranges may change
and expire, which breaks validation: we will still consider previously undefined
for use ranges invalid.

Plotting binned images
**********************

Our package also provides a plotting functionality to visualize datasets as
binned images. While you can plot your binned datasets by yourself, for example,
with Matplotlib's :func:`~matplotlib:matplotlib.pyplot.imshow`, there are some
existing :class:`plotters <allisbns.plotting.BinnedPlotter>` available. Plotters
set up axes, rearrange bins in various ways (with the help of functions from
:mod:`allisbns.rearrange`), are aware of coordinate conversion (handling by
:class:`allisbns.plotting.CoordinateConverter`), and draw images.

Available plotters
==================

Currently, two plotters exist: :class:`~allisbns.plotting.RowBinnedPlotter`
(plots bins as rows of fixed width) and
:class:`~allisbns.plotting.BlockBinnedPlotter` (plots bins as vertical blocks of
fixed size stacked horizontally).

Simple example
==============

Let's draw a binned image with :class:`~allisbns.plotting.RowBinnedPlotter`.

.. code-block:: python

   import matplotlib.pyplot as plt
   from allisbns.plotting import RowBinnedPlotter

   fig, ax = plt.subplots(figsize=(12, 12), dpi=100)

   # Bin your dataset
   binned = dataset.bin(2500)

   # Set up a plotter
   plotter = RowBinnedPlotter(
       ax, width=int(2.5e6), bin_size=binned.bin_size
   )

   # Draw a binned image
   plotter.plot_bins(binned)

   plt.tight_layout()
   plt.show()

Setting up the plotter fixes the image width to the row width (in relative
ISBNs). While the image height (number of rows) is automatically determined
during drawing and depends on the number of bins and the selected ``aspect``
(provided via ``imshow_kwargs`` of
:meth:`~allisbns.plotting.RowBinnedPlotter.plot_bins`). Similarly,
:class:`~allisbns.plotting.BlockBinnedPlotter` fixes the image height according
to the input block width and capacity values.

The plotter insists on working with the fixed bin size, which seems reasonable
since plotting differently binned datasets (with different colormaps) breaks the
comparison.

Image extent
============

Once you have plotted bins with
:meth:`~allisbns.plotting.BinnedPlotter.plot_bins` or an image with
:meth:`~allisbns.plotting.BinnedPlotter.plot_image`, the :attr:`extent
<allisbns.plotting.BinnedPlotter.extent>` is set and cannot be changed
after. This is deliberate behavior in the current versions.


The way to plot different images with different extents with one plotter is the
following: (1) determine one overall size for all images, (2) create arrays with
the corresponding common shape, (3) assign the desired region with your
data, and (4) plot new images.

Define extent without plotting
==============================

What if you want to scatter plot ISBNs using a plotter to keep all axis
decorations and all else? There is the
:meth:`~allisbns.plotting.BinnedPlotter.define_extent` method to define the
extent based on the ISBN bounds without plotting bins or an image:

.. code-block:: python

   >>> plotter = RowBinnedPlotter(
   ...     ax,
   ...     # Imitate the resolution of one ISBN
   ...     bin_size=1,
   ...     # Width is in relative ISBNs
   ...     width=int(2.5e6),
   ...     # Let's start our extent with this offset
   ...     offset=978_300_000_000,
   ...     # Adjust the image aspect to the new resolution, or set
   ...     # to 'auto' to follow the defined figure size
   ...     aspect=int(2.5e3),
   ... )

   >>> # This will set all the required axis limits for the extent corresponding
   >>> # to the ISBN range from 978300000000X (offset) to 978399999999X
   >>> plotter.define_extent(end_isbn=978_399_999_999)
   >>> plotter.extent
   (0, 2500000, 160, 0)

   >>> # Can be useful for further general plotting
   >>> ax.scatter(...)
