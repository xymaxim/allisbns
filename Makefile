NOTEBOOKS := $(wildcard examples/*.ipynb)

.PHONY: docs
docs:
	uv run sphinx-build docs docs/_build

examples/data/%.benc.zst:
	$(error missing file $@)

examples/data/%.h5: examples/data/%.benc.zst
	uv run python scripts/convert-bencoded-to-h5.py $< $@

FORCE:

examples/%.ipynb: FORCE
	uv run jupyter nbconvert --to notebook --execute --inplace $@

rerun-notebooks: $(NOTEBOOKS)

plot-cover-images: examples/plot-cover-image.ipynb
