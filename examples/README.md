# Examples

## List of examples

### Intro

- [Getting started](https://github.com/xymaxim/allisbns/blob/main/examples/getting-started.ipynb)
  — introduction to the package
- [Plot binned images](https://github.com/xymaxim/allisbns/blob/main/examples/plot-binned-images.ipynb)
  — how to plot binned images
- [Merge datasets](https://github.com/xymaxim/allisbns/blob/main/examples/merge-datasets.ipynb)
  — how to merge datasets

### Next step

* [Plot all groups](https://github.com/xymaxim/allisbns/blob/main/examples/plot-all-groups.ipynb)
  — plotting all ISBN groups
* [Plot all ISBNs](https://github.com/xymaxim/allisbns/blob/main/examples/plot-all-isbns.ipynb)
  — plotting all merged datasets
* [Compare dumps](https://github.com/xymaxim/allisbns/blob/main/examples/compare-dumps.ipynb)
  — comparing two ISBN code dump files
* [Plot cover image](https://github.com/xymaxim/allisbns/blob/main/examples/plot-cover-image.ipynb)
  — how is the cover image plotted
* [Find unique ISBNs](https://github.com/xymaxim/allisbns/blob/main/examples/find-unique-isbns.ipynb)
  — finding ISBNs unique to the dataset
* Calculate percentage ([Python](https://github.com/xymaxim/allisbns/blob/main/examples/calculate-percentage.py),
  [Julia](https://gist.github.com/xymaxim/b2af6c8d8540c1f9d79ff288f42c2968#file-calculate-percentage-jl))
  — scripts to calculate the percentage of ISBNs

## How to run notebooks?

1. Download `aa_isbn13_codes_*.benc.zst` from the
   [torrent](https://allisbns.readthedocs.io/en/stable/#download-data) to
   `examples/data/`
2. Update `CURRENT_DUMP_PATH` in `examples/common.py`
3. Install extra dependencies: `$ uv sync --extra examples`
4. Some examples require codes to be stored in HDF5
   files ([why?](https://allisbns.readthedocs.io/en/latest/cookbook.html#store-datasets-in-hdf5)):

       $ cd examples/data/
       $ uv run python ../../scripts/convert-bencoded-to-h5.py \
           aa_isbn13_codes...

5. Run Jupyter Notebook in `examples`: `$ uv run jupyter notebook`
