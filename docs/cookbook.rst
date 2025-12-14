Cookbook
########

.. contents:: Contents
   :depth: 1
   :backlinks: top
   :local:

.. _cookbook-iterate-datasets:

Iterate over datasets
*********************

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

Also, the iterable datasets can be lazy cropped to some boundaries. For example,
let's iterate over the '978' region of all datasets:

.. code-block:: python

   from allisbns.isbn import get_prefix_bounds

   # Get the corresponding boundaries
   start_isbn, end_isbn = *get_prefix_bounds("978")

   # Create the iterator, fill all datasets to the end ISBN
   iterator = iterate_datasets(input_data, fill_to_isbn=end_isbn)

   # Use the generator expression to lazy crop datasets
   iterator_with_cropping = (x.crop(start_isbn, end_isbn) for x in iterator)
   for cropped_dataset in iterator_with_cropping:
       ...

Merge all datasets
******************

Create the iterator as above and union all datasets together:

.. code-block:: python

   from allisbns.isbn import LAST_ISBN
   from allisbns.merge import union

   # The boundaries must be the same
   iterator = iterate_datasets(input_data, fill_to_isbn=LAST_ISBN)

   all_merged = merge.union(iterator)

After merging, we can save the result codes to a file for later use. For
example, let's temporarily save it to a binary file in :mod:`NumPy format
<numpy:numpy.lib.format>`:

.. code-block:: python

   timestamp = str(input_path).split(".")[0].split("_")[-1]
   output_path = f"ms_isbn13_codes_{timestamp}_all.npy"

   with open(output_path, "wb") as f:
       np.save(f, all_merged.codes, allow_pickle=False)

To write it down in the original format with compression, we can use
:meth:`~allisbns.dataset.CodeDataset.write_bencoded`:

.. code-block:: python

   with open(output_path.with_suffix(".benc.zst"), "wb") as f:
       all_merged.write_bencoded(f, prefix="all")
