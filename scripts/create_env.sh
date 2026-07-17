#!/usr/bin/env sh
set -eu
conda run -n histo-delib python --version
conda run -n histo-delib python -m pip install -r requirements/base.txt -r requirements/api.txt -r requirements/dev.txt
conda run -n histo-delib python -m pip check
