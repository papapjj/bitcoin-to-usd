<#
.SYNOPSIS
    Runs the paired MT5 Model 4 OOS gate after the full teacher is ready.

.DESCRIPTION
    This is the last research-only stage of the teacher chain.  It waits for
    the geometry checkpoint, then runs candidate and frozen baseline in
    isolated MT5 clones through run_v118_isolated_oos.py.  The Python runner
    owns the per-window ledger, terminal-flat, reconciliation, feature-parity
    and pooled-LCB gates.  A failed gate is recorded as evidence and is never
    converted into a promotion.

    The supervisor never starts a second OOS process for the same output.  A
    partial output without summary.json is treated as a fail-closed condition
    so a later operator must inspect it instead of silently mixing runs.
#>
[CmdletBinding()]
param(
    [string]$Python = 'C:\Python314\python.exe',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [Parameter(Mandatory = $true)] [string]$GeometryCheckpoint,
    [Parameter(Mandatory = $true)] [string]$BaselineCheckpoint,
    [Parameter(Mandatory = $true)] [string]$Output,
    [string]$SourceRunner = 'mt5_v2_runner',
    [int]$Model = 4,
    [string]$Windows = '2025.05.29:2025.05.30,2025.05.30:2025.05.31,2025.05.31:2025.06.01',
    [int]$PollSeconds = 30
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-WorkspacePath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Workspace $Value))
}

if ($Model -ne 4) { throw 'OOS promotion chi cho phep MT5 Model 4 real ticks' }
if ($PollSeconds -lt 5) { throw 'PollSeconds phai >= 5' }

$workspacePath = [IO.Path]::GetFullPath($Workspace)
$pythonPath = [IO.Path]::GetFullPath($Python)
$runnerPath = Join-Path $workspacePath 'scripts\run_v118_isolated_oos.py'
$geometryPath = Resolve-WorkspacePath $GeometryCheckpoint
$baselinePath = Resolve-WorkspacePath $BaselineCheckpoint
$outputPath = Resolve-WorkspacePath $Output
$outputDirectory = Split-Path -Parent $outputPath
$logDirectory = $outputPath + '.supervisor'
$lockPath = $outputPath + '.supervisor.lock'
$summaryPath = Join-Path $outputPath 'summary.json'
$geometryLockPath = $geometryPath + '.supervisor.lock'

foreach ($requiredPath in @($workspacePath, $pythonPath, $runnerPath, $baselinePath, (Split-Path -Parent $geometryPath))) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Khong tim thay duong dan bat buoc: $requiredPath"
    }
}
if (Test-Path -LiteralPath $summaryPath) {
    Write-Output ("PAIRED_OOS_DONE summary={0}" -f $summaryPath)
    exit 0
}
if (Test-Path -LiteralPath $outputPath) {
    throw "OOS output ton tai nhung thieu summary.json; dung fail-closed: $outputPath"
}
if (Test-Path -LiteralPath $lockPath) {
    throw "Paired OOS supervisor dang chay: $lockPath"
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $lockPath -Force | Out-Null
@{ pid = $PID; started_at = (Get-Date).ToString('o'); output = $outputPath } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lockPath 'owner.json') -Encoding UTF8

try {
    Write-Output ("PAIRED_OOS_WAIT geometry={0}" -f $geometryPath)
    while (-not (Test-Path -LiteralPath $geometryPath)) {
        if (-not (Test-Path -LiteralPath $geometryLockPath)) {
            throw 'Geometry supervisor da ket thuc nhung khong tao full-teacher checkpoint'
        }
        Start-Sleep -Seconds $PollSeconds
    }
    if ((Get-Item -LiteralPath $geometryPath).Length -le 0) {
        throw 'Full-teacher checkpoint rong'
    }
    if (Test-Path -LiteralPath $geometryLockPath) {
        throw 'Full-teacher checkpoint van dang duoc ghi; khong duoc chay OOS'
    }

    $stdoutPath = Join-Path $logDirectory 'stdout.log'
    $stderrPath = Join-Path $logDirectory 'stderr.log'
    $env:V13_SEED_SYSTEM = $baselinePath
    $env:V13_V118_CANDIDATE = $geometryPath
    $env:V13_V118_OOS_OUT = $outputPath
    $env:V13_SOURCE_RUNNER = (Resolve-WorkspacePath $SourceRunner)
    $env:V13_MODEL = [string]$Model
    $env:V13_V118_OOS_WINDOWS = $Windows

    Write-Output ("PAIRED_OOS_START model={0} windows={1} output={2}" -f $Model, $Windows, $outputPath)
    $child = Start-Process -FilePath $pythonPath -WorkingDirectory $workspacePath `
        -ArgumentList 'scripts\run_v118_isolated_oos.py' -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath -PassThru
    Write-Output ("PAIRED_OOS_CHILD pid={0}" -f $child.Id)
    Wait-Process -Id $child.Id
    $exitCode = $child.ExitCode
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $summaryPath)) {
        throw "Paired OOS that bai: exit_code=$exitCode; xem $logDirectory"
    }
    $summary = Get-Content -LiteralPath $summaryPath -Raw | ConvertFrom-Json
    Write-Output ("PAIRED_OOS_DONE all_windows_eligible={0} summary={1}" -f [bool]$summary.all_windows_eligible, $summaryPath)
}
finally {
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Recurse -Force
    }
}
