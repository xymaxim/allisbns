NOTEBOOKS := $(wildcard examples/*.ipynb)

.PHONY: docs
docs:
	uv run sphinx-build docs docs/_build

RangeMessage.xml:
	curl https://www.isbn-international.org/export_rangemessage.xml \
	  -o RangeMessage.xml

# If download failed and you already have the RangeMessage.xml file, then
# run only this target as: make -W RangeMessage.xml ranges.py
ranges.py: RangeMessage.xml
	uv run python scripts/create-ranges-file.py RangeMessage.xml \
	  > src/allisbns/ranges.py

examples/data/%.benc.zst:
	$(error missing file $@)

examples/data/%.h5: examples/data/%.benc.zst
	uv run python scripts/convert-bencoded-to-h5.py $< $@

.PHONY: examples/%.ipynb
examples/%.ipynb:
	uv run jupyter nbconvert --to notebook --execute --inplace $@

run-all-notebooks: $(NOTEBOOKS)

plot-cover-images: examples/plot-cover-image.ipynb
