$ErrorActionPreference = 'Stop'
conda env create -f environment.yml
conda run -n histo-delib python -m pip check
