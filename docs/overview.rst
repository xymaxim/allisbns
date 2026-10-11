Overview
########

.. contents:: Contents
   :depth: 1
   :backlinks: top
   :local:

Working with datasets
*********************

Source data
===========

Anna and the team periodically, as part of the derived metadata, publishes files
with ISBN codes named as ``aa_isbn13_codes_*.benc.zst``. They can be downloaded
via the ``aa_derived_mirror_metadata`` torrent from `this page
<https://annas-archive.org/torrents/aa_derived_mirror_metadata>`__.

Codes
=====

The *packed ISBN codes* are a set of integers that represent the length of the
alternating *streak* (present ISBNs) and *gap* (missing ISBNs) segments, for
example::

  3, 2, 1, ...

The codes are mapped to 2 million ISBNs covering both 978 and 979 prefixes. It
is similar to the `run-length encoding
<https://en.wikipedia.org/wiki/Run-length_encoding>`__ and efficiently describes
information about ISBN availability for a selected dataset. The above codes can
be expand to the following boolean mask::

  True, True, True, False, False, True, ...

It is supposed for all published datasets that the offset
is 9780000000002. Also, codes end with a streak segment and the last gap segment
is omitted. See `here
<https://software.annas-archive.li/AnnaArchivist/annas-archive/-/blob/9c18fc80341376914da2fe0770b4ed13347143f0/allthethings/cli/views.py#L2007-2019>`__
on how codes are generated and for the description of the binary format used to
store them.

In our package we use several terms that relate to ISBN availability. We can
distinguish between *packed* and *unpacked* codes:
:type:`~allisbns.dataset.PackedCodes` refer to the original presentation of the
codes and :type:`~allisbns.dataset.UnpackedCodes` to the decoded one as boolean
values. To distinguish between the individual ISBNs from streak and gap segments,
we call them *filled* and *unfilled* ISBNs, respectively.

Read datasets
=============

Several ways
------------

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

Extend codes
------------

By default, datasets are read as is and have different sizes in ISBNs. However,
sometimes it is required to bring datasets to the same size: for example, when
you merge or plot them. To extend a dataset to some ISBN value with a gap
segment, we can use the ``fill_to_isbn`` argument:

.. code-block:: python

    >>> from prettyisbns.isbn import LAST_ISBN
    >>>
    >>> CodeDataset.from_file(
    ...     source="aa_isbn13_codes_20251118T170842Z.benc.zst",
    ...     collection="md5",
    ...     fill_to_isbn=LAST_ISBN,
    ... )
    CodeDataset(array([     6,      1,      9, ...,  91739,      1, 531099],
      shape=(14737376,)), bounds=(978000000000, 979999999999))

Reframe datasets
================

After reading, it is possible to select a part of a dataset by reframing it. The
reframing includes both cropping existing segments and extending codes with
gap segments on both sides.

The use cases could include limiting the output of some methods. For example, if
you need to get all filled ISBNs with the '979' prefix, then you will need to
reframe a dataset in prior with :meth:`~allisbns.dataset.CodeDataset.reframe`:

.. code-block:: python

   from prettyisbns.isbn import get_prefix_bounds
   md5.reframe(*get_prefix_bounds("979")).get_filled_isbns()

Immutable datasets
==================

By design, datasets are considered immutable after creation. If you need to
modify codes for some reason, you can copy codes and create a new dataset after
editing. For example, here is an equivalent of the
:meth:`~allisbns.dataset.CodeDataset.invert` method to inverse a dataset:

.. code-block:: python

   >>> CodeDataset(np.concatenate([[0], md5.codes]), offset=md5.offset)
   CodeDataset(array([     0,      6,      1, ...,  91739,      1],
     shape=(14737377,), dtype=int32), bounds=(978000000000, 979999468900))

Working with ISBNs
******************

To simplify things, the methods of :class:`~allisbns.dataset.CodeDataset` only
accept ISBNs represented as :type:`~prettyisbns.isbn.ISBN12` values (ISBN-13
without the check digit) of integer types.

Everything else related to ISBNs (normalizing, fixing, and formatting ISBNs) is
provided by the :external+prettyisbns:doc:`index` package. Use
:func:`~prettyisbns.normalize` to turn an ISBN string into an ISBN-12 value:

.. code-block:: python

   from prettyisbns import normalize
   from prettyisbns.errors import NotISBNError

   try:
       # Normalize the ISBN string and convert it to an ISBN-12 value
       isbn12 = normalize("978-2-36590-117-X").to_isbn12()
   except NotISBNError:
       # Handle a value that is not an ISBN
       ...

   # Can be safely used for querying the dataset
   md5.query_isbn(isbn12)

Because :type:`~prettyisbns.isbn.ISBN12` is just an :class:`int`, plain integers
can be passed directly, as long as they fall between
:data:`~prettyisbns.isbn.FIRST_ISBN` and :data:`~prettyisbns.isbn.LAST_ISBN`:

.. code-block:: python

   from prettyisbns.isbn import FIRST_ISBN, LAST_ISBN

   isbn12 = 978_2_36590_117
   if FIRST_ISBN <= isbn12 <= LAST_ISBN:
       md5.query_isbn(isbn12)
   else:
       ...

See ``prettyisbns``'s :external+prettyisbns:doc:`usage` for the full usage guide.

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
