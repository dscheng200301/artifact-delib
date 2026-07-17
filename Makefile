PYTHON ?= conda run -n histo-delib python

.PHONY: env install doctor lint typecheck test fixture smoke-mock smoke-api verify clean-generated

env:
	conda env create -f environment.yml

install:
	$(PYTHON) -m pip install -e .

doctor:
	$(PYTHON) -m histodelib.cli doctor

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy src

test:
	$(PYTHON) -m pytest

fixture:
	$(PYTHON) -m histodelib.cli fixture build
	$(PYTHON) -m histodelib.cli fixture validate

smoke-mock:
	$(PYTHON) -m histodelib.cli run --method histodelib_rule --config fixture

smoke-api:
	$(PYTHON) -m histodelib.cli run --method histodelib_rule --config smoke_api --allow-paid-calls

verify: lint typecheck test fixture smoke-mock

clean-generated:
	$(PYTHON) -m histodelib.cli clean-generated
