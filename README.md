# allisbns

![PyPI - Version](https://img.shields.io/pypi/v/allisbns)

allisbns is a Python package to work with the packed ISBN codes from [Anna's
Archive](https://annas-archive.org/). It helps you to examine, manipulate, and
plot such data that represent the largest fully open list of all known ISBNs.

*(This project is not affiliated with Anna's Archive.)*

[Source](https://github.com/xymaxim/allisbns)
[Documentation](https://allisbns.readthedocs.io/en/stable/)
[Changelog](https://allisbns.readthedocs.io/en/latest/changelog.html)

## Introduction

Anna's Archive, besides books and datasets, provides a large amount of metadata
from different sources (including
[WorldCat](https://annas-archive.org/blog/worldcat-scrape.html), Google Books,
the [Chinese
collections](https://annas-archive.org/blog/finished-chinese-release.html), and
many others). Such an extensive collection presumably
[represents](https://annas-archive.org/blog/all-isbns.html) the largest openly
available metadata about all known ISBNs ever published (see the figure below).

The derived metadata, periodically published by Anna and the team, includes the
*packed ISBN codes*, a very [compact
representation](https://annas-archive.org/blog/all-isbns.html#:~:text=a%20compact%20data%20format)
of all ISBNs with distinction of original data sources: it can tell you what
ISBNs are available in a dataset.

After the [visualization
contest](https://annas-archive.org/blog/all-isbns-winners.html), the beautiful
interactive [viewer](https://annas-archive.org/isbn-visualization) exists to
explore all ISBNs. However, sometimes you need more imperative control over the
available data: check many ISBNs at once, analyze selected regions, compare
different dumps, plot custom images, etc.


![Binned image of all ISBNs](https://media.githubusercontent.com/media/xymaxim/allisbns/refs/heads/main/images/allisbns-cover-readme.jpg)

*Binned image*
([hi-res version](https://media.githubusercontent.com/media/xymaxim/allisbns/refs/heads/main/images/allisbns-cover-high.jpg))
*of all known ISBNs (source: Anna's Archive). The defined ISBN registration
groups are underlaid in black. See*
[here](https://github.com/xymaxim/allisbns/blob/main/examples/plot-cover-image.ipynb)
*how it is plotted*.

## Installation

The package is available on PyPI:

    pip install allisbns

To include optional plotting support, install it as:

    pip install allisbns[plotting]

## Quickstart

### Download data

The package works with datasets provided as bencoded files named as
`aa_isbn13_codes_*.benc.zst`. Such files are located in the `codes_benc`
directory within the
[aa_derived_mirror_metadata](https://annas-archive.org/torrents/aa_derived_mirror_metadata)
torrents.

### Work with datasets

Creates a dataset from the downloaded file with ISBN codes:

```python
    >>> from allisbns.dataset import CodeDataset
    >>> md5 = CodeDataset.from_file(
    ...     source="aa_isbn13_codes_20251118T170842Z.benc.zst",
    ...     collection="md5",
    ... )
    >>> md5
    CodeDataset(array([    6,     1,     9, ...,     1, 91739,     1],
      shape=(14737375,), dtype=int32), bounds=(978000000000, 979999468900))
```

Here the `md5` collection represents files available for downloading in Anna's
Archive. All available collections are:

    'airitibooks', 'bloomsbury', 'cadal_ssno', 'cerlalc', 'chinese_architecture',
    'duxiu_ssid', 'edsebk', 'gbooks', 'goodreads', 'hathi', 'huawen_library', 'ia',
    'isbndb', 'isbngrp', 'kulturpass', 'libby', 'md5', 'nexusstc',
    'nexusstc_download', 'oclc', 'ol', 'ptpress', 'rgb', 'sciencereading', 'shukui',
    'sklib', 'trantor', 'wanfang', 'zjjd'

Query one ISBN:

```python
    >>> md5.query_isbn(978_2_36590_117)
    QueryResult(is_streak=True, segment_index=8652142, position_in_segment=0)
```

Check many ISBNs:

```python
    >>> md5.check_isbns(range(978_2_36590_000, 978_2_36590_999 + 1))
    array([ True, False, False, ..., False, False, False], shape=(1000,))
```

Get all filled ISBNs:

```python
    >>> md5.get_filled_isbns()
    array([978000000000, 978000000001, 978000000002, ..., 979999377030,
       979999377160, 979999468900], shape=(16916212,))
```

Crop the dataset to some ISBN region:

```python
    >>> from allisbns.isbn import get_prefix_bounds
    >>> start_isbn, end_isbn = get_prefix_bounds("978")
    >>> md5.crop(start_isbn, end_isbn)
    CodeDataset(array([6, 1, 9, ..., 1, 2, 2],
       shape=(14503001,), dtype=int32), bounds=(978000000000, 978999999999))
```

## Further reading

After installing, check out the
[documentation](https://allisbns.readthedocs.io/en/stable/). See
[Overview](https://allisbns.readthedocs.io/en/stable/overview.html) for the
first guidance. The [API
reference](https://allisbns.readthedocs.io/en/stable/api/index.html) describes
modules, classes, and functions. There are practical
[examples](https://allisbns.readthedocs.io/en/stable/examples.html) that will
demonstrate the main
usage. [Cookbook](https://allisbns.readthedocs.io/en/stable/cookbook.html) also
contains useful examples. You want to contribute code?
[Contributing](https://allisbns.readthedocs.io/en/latest/contributing.html)
tells how to participate.

## License

Creative Commons Zero v1.0 Universal.
