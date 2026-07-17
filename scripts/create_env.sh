#!/usr/bin/env sh
set -eu
conda env create -f environment.yml
conda run -n histo-delib python -m pip check
