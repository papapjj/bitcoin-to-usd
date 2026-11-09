param(
  [string]$Symbol = "XAUUSD",
  [string]$Timeframe = "M1",
  [int]$Bars = 5000000,
  [string]$Pattern = "examples/impulse_pullback.yaml",
  [string]$Raw = "data/XAUUSD_M1.npz",
  [string]$Output = "datasets/XAUUSD_impulse"
)
$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
Set-Location $Root
$env:PYTHONPATH = $Root
New-Item -ItemType Directory -Force -Path (Split-Path $Raw) | Out-Null
python -m zigzag_miner.cli mt5-fetch --symbol $Symbol --timeframe $Timeframe --bars $Bars --output $Raw
python -m zigzag_miner.cli mine --input $Raw --pattern $Pattern --output $Output --shard-size 10000
