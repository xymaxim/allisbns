.PHONY: docs
docs:
	uv run sphinx-build docs docs/_build

.PHONY: RangeMessage.xml
RangeMessage.xml:
	curl https://www.isbn-international.org/export_rangemessage.xml \
	  -o RangeMessage.xml

# If download failed and you already have the RangeMessage.xml file, then
# run only this target as: make -W RangeMessage.xml ranges.py
.PHONY: ranges.py
ranges.py: RangeMessage.xml
	uv run python scripts/create-ranges-file.py RangeMessage.xml \
	  > src/allisbns/ranges.py

.PHONY: allisbns-cover.jpg
allisbns-cover.jpg:
	uv run jupyter nbconvert --to notebook --execute --inplace \
	  examples/plot-cover-image.ipynb
