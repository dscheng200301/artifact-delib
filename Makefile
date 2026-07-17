PYTHON ?= conda run -n histo-delib python

.PHONY: env install doctor lint typecheck test fixture smoke-mock smoke-api verify clean-generated

env:
	conda run -n histo-delib python --version
	conda run -n histo-delib python -m pip check

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
	@echo REAL_API_SMOKE_TEST=NOT_RUN

verify: lint typecheck test fixture smoke-mock
	$(PYTHON) -m pip check
	$(PYTHON) scripts/build_smoke_report.py

clean-generated:
	$(PYTHON) -m histodelib.cli clean-generated
