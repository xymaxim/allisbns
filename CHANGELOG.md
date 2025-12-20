# Changelog

Versions follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) with
the `MAJOR.MINOR.PATCH` scheme.

## v0.1.3 - 2025.12.20

- Add the `total_isbns` field to `CodeDataset`
- Make the `CodeDataset.codes` array read-only

## v0.1.2 - 2025-12-19

- Rework dataset cropping into reframing, which involves both cropping and
  expanding

## v0.1.1 - 2025-12-16

- Fix check for the `fill_to_isbn` argument being inside the dataset bounds
- Allow to check ISBNs outside the dataset bounds in `check_isbns()`
