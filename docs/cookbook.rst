Cookbook
########

.. contents:: Contents
   :depth: 1
   :backlinks: top
   :local:

.. _cookbook-iterate-datasets:

Iterate and reframe datasets
****************************

Let's first load the bencoded data from the compressed file:

.. code-block:: python

   from allisbns.dataset import load_bencoded

   input_path = "aa_isbn13_codes_20251118T170842Z.benc.zst"
   with open(input_path, "rb") as f:
       input_data = load_bencoded(f)

Then create an iterator over all datasets and iterate:

.. code-block:: python

   from allisbns.dataset import iterate_datasets, CodeDataset
   from allisbns.isbn import LAST_ISBN

   for dataset in iterate_datasets(input_data):
       ...

The iterable datasets can be narrowed only to the selected ones:

.. code-block:: python

   for dataset in iterate_datasets(
       input_data, collections=["md5", "rgb"]
   ):
       ...

Also, the iterable datasets can be lazy reframed to some new bounds. For
example, let's iterate over the '978' region of all datasets:

.. code-block:: python

   from allisbns.isbn import get_prefix_bounds

   # Get the corresponding bounds
   start_isbn, end_isbn = *get_prefix_bounds("978")

   # Create the iterator, fill all datasets to the end ISBN
   iterator = iterate_datasets(input_data, fill_to_isbn=end_isbn)

   # Use the generator expression to lazy reframe datasets
   reframing = (x.reframe(start_isbn, end_isbn) for x in iterator)
   for reframed_dataset in reframing:
       ...

Merge all datasets
******************

Create the iterator as above and union all datasets together:

.. code-block:: python

   from allisbns.isbn import LAST_ISBN
   from allisbns.merge import union

   # The bounds must be the same
   iterator = iterate_datasets(input_data, fill_to_isbn=LAST_ISBN)

   all_merged = merge.union(iterator)

After merging, we can save the result codes to a file for later use. For
example, let's temporarily save it to a binary file in :mod:`NumPy format
<numpy:numpy.lib.format>`:

.. code-block:: python

   timestamp = str(input_path).split(".")[0].split("_")[-1]
   output_path = f"xy_isbn13_codes_{timestamp}_all.npy"

   with open(output_path, "wb") as f:
       np.save(f, all_merged.codes, allow_pickle=False)

To write it down in the original format with compression, we can use
:meth:`~allisbns.dataset.CodeDataset.write_bencoded`:

.. code-block:: python

   with open(output_path.with_suffix(".benc.zst"), "wb") as f:
       all_merged.write_bencoded(f, prefix="all")

Store datasets in HDF5
**********************

The original files with ISBN codes have a quite simple structure. All codes are
packed into a single bencoded dictionary and shipped compressed with Zstd. That
yields a compact distribution (~80MB) but forces full decompression (~700MB) to
read any subset, which can hurt interactive use a bit. An alternative for
working with datasets would be to store codes in a different container format
optimized for homogeneous arrays and partial access without uncompressing the
whole file, such as HDF5, NetCDF, or Zarr.

Here we experiment with `HDF5 <https://www.hdfgroup.org/solutions/hdf5/>`__ to
convert bencoded files and save grouped analysis results.

Current way
-----------

The current reading of codes for a single collection from a bencoded file
(without keeping uncompressed data in memory) can be written as follows:

.. code-block:: python

   import struct

   import bencodepy
   import zstandard

   def read_codes(path: str, collection: str) -> tuple[int]:
       with open(path, "rb") as f:
           with zstandard.ZstdDecompressor().stream_reader(f) as s:
               uncompressed_data = bencodepy.bread(s)
       packed_binary_codes = uncompressed_data[collection.encode()]
       return struct.unpack(
           f"{len(packed_binary_codes) // 4}I",
           packed_binary_codes
       )

On a non-performant laptop, I got this, which can be noticeable in interactive
sessions:

.. code-block:: ipython

   In [1]: %timeit read_codes("aa_isbn13_codes_20251118T170842Z.benc.zst", "md5")
   2.38 s ± 21.1 ms per loop (mean ± std. dev. of 7 runs, 1 loop each)

Convert bencoded files
----------------------

Alternatively, we provide the ``convert-bencoded-to-h5.py`` script (`link
<https://github.com/xymaxim/allisbns/tree/main/scripts/convert-bencoded-to-h5.py>`__)
for converting the ``*.benc.zst`` files to HDF5.

.. code-block:: shell-session

   $ uv run python scripts/convert-bencoded-to-h5.py \
       aa_isbn13_codes_20251118T170842Z.benc.zst
   $ ls -sh
   78M  aa_isbn13_codes_20251118T170842Z.benc.zst
   82M  aa_isbn13_codes_20251118T170842Z.h5

The conversion is pretty quick and produces comparable file sizes. For the
compression, we use `Blosc <https://www.blosc.org/>`__ (non-standard, available
via `hdf5plugin <https://github.com/silx-kit/hdf5plugin>`__) with the shuffle
and Zstd filters activated:

.. code-block:: python

   hdf5plugin.Blosc(
       cname="zstd",
       clevel=5,
       shuffle=hdf5plugin.Blosc.SHUFFLE
   )

After that, reading codes as NumPy arrays is simply as follows:

.. code-block:: python

   import h5py
   import numpy as np
   import numpy.typing as npt

   def read_codes(path: str, collection: str) -> npt.NDArray[np.int32]:
       with h5py.File(path, "r") as f:
           return f[collection][:]

.. code-block:: ipython

   In [1]: %timeit read_codes("aa_isbn13_codes_20251118T170842Z.benc.zst", "md5")
   104 ms ± 1.39 ms per loop (mean ± std. dev. of 7 runs, 10 loops each)

Basically, this setup preserves compact storage while enabling fast partial reads.

Groups and attributes
---------------------

The structured nature of HDF5 can be useful to store datasets and the
corresponding metadata in a single file. For example, see the `Compare dumps
<https://github.com/xymaxim/allisbns/blob/main/examples/compare-dumps.ipynb>`__
example where we compare 'md5' datasets from two latest dumps to find additions
and deletions and save results in an HDF5 file using groups.
