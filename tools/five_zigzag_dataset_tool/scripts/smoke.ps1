$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
python -m unittest discover -s tests -v
python -m zigzag_miner.cli smoke --pattern examples/impulse_pullback.yaml
