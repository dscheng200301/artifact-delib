$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path (Get-Location) 'src'
$env:PYTHONIOENCODING = 'utf-8'
conda run --no-capture-output -n histo-delib python -m pytest
conda run --no-capture-output -n histo-delib python -m ruff check src tests
conda run --no-capture-output -n histo-delib python -m mypy src
conda run --no-capture-output -n histo-delib python -m pip check
conda run --no-capture-output -n histo-delib python -m histodelib.cli doctor
conda run --no-capture-output -n histo-delib python -m histodelib.cli fixture build
conda run --no-capture-output -n histo-delib python -m histodelib.cli fixture validate
conda run --no-capture-output -n histo-delib python -m histodelib.cli run --method histodelib_rule --config fixture
conda run --no-capture-output -n histo-delib python scripts/build_smoke_report.py
