PYTHON ?= python

.PHONY: all build model test
all: build model
build:
	$(PYTHON) src/build_dataset.py
model:
	$(PYTHON) src/model.py
test:
	pytest -q
