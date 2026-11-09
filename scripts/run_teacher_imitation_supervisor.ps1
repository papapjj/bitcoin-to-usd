<#
.SYNOPSIS
    Runs teacher imitation as a single, resumable worker.

.DESCRIPTION
    The trainer writes a completed-epoch resume checkpoint and, when
    CheckpointEveryWindows is nonzero, restartable mid-epoch checkpoints.
    This supervisor never starts a second worker while the first one is alive.
    If the worker exits before producing the final checkpoint it records the
    attempt and restarts from the same resume file, up to MaxAttempts. Each
    attempt has its own stdout/stderr log.
#>
[CmdletBinding()]
param(
    [string]$Python = 'C:\Python314\python.exe',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs\he_chuyen_gia_mot_context_mot_policy.yaml',
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [string]$Dataset,
    [Parameter(Mandatory = $true)] [string]$Output,
    [int]$MaxAttempts = 3,
    [int]$RetryDelaySeconds = 15,
    [int]$Epochs = 80,
    [int]$BatchSize = 8,
    [int]$FeatureBatchSize = 32,
    [double]$LearningRate = 0.003,
    [double]$RecurrentLearningRate = 0.00001,
    [int]$Patience = 12,
    [int]$Seed = 0,
    [int]$SequenceLength = 64,
    [double]$FalseEntryPenalty = 0.50,
    [double]$RankingLossWeight = 0.50,
    [int]$MentorMinEpoch = 2,
    [int]$CheckpointEveryWindows = 128
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-WorkspacePath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $Workspace $Value))
}

function Quote-Argument([string]$Value) {
    if ($Value -notmatch '[\s"]') { return $Value }
    return '"' + ($Value -replace '(?<!\\)"', '\\"') + '"'
}

if ($MaxAttempts -lt 1) { throw 'MaxAttempts phai >= 1' }
if ($RetryDelaySeconds -lt 0) { throw 'RetryDelaySeconds phai >= 0' }
if ($CheckpointEveryWindows -lt 0) { throw 'CheckpointEveryWindows phai >= 0' }

$workspacePath = [IO.Path]::GetFullPath($Workspace)
$pythonPath = [IO.Path]::GetFullPath($Python)
$trainScript = Join-Path $workspacePath 'scripts\train_teacher_imitation.py'
$configPath = Resolve-WorkspacePath $Config
$checkpointPath = Resolve-WorkspacePath $Checkpoint
$datasetPath = Resolve-WorkspacePath $Dataset
$outputPath = Resolve-WorkspacePath $Output
$outputDirectory = Split-Path -Parent $outputPath
$resumePath = $outputPath + '.resume.pkl'
$lockPath = Join-Path $outputDirectory ((Split-Path -Leaf $outputPath) + '.supervisor.lock')
$attemptDirectory = Join-Path $outputDirectory ((Split-Path -Leaf $outputPath) + '.supervisor-attempts')

foreach ($requiredPath in @($workspacePath, $pythonPath, $trainScript, $configPath, $checkpointPath, $datasetPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Khong tim thay duong dan bat buoc: $requiredPath"
    }
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $attemptDirectory -Force | Out-Null

function Get-LockOwnerPid {
    $metadataPath = Join-Path $lockPath 'owner.json'
    if (-not (Test-Path -LiteralPath $metadataPath)) { return $null }
    try { return [int](Get-Content -LiteralPath $metadataPath -Raw | ConvertFrom-Json).pid }
    catch { return $null }
}

if (Test-Path -LiteralPath $lockPath) {
    $ownerPid = Get-LockOwnerPid
    $owner = if ($ownerPid) { Get-Process -Id $ownerPid -ErrorAction SilentlyContinue } else { $null }
    if ($owner) { throw "Supervisor dang chay (PID $ownerPid): $lockPath" }
    Remove-Item -LiteralPath $lockPath -Recurse -Force
}
New-Item -ItemType Directory -Path $lockPath -Force | Out-Null
@{ pid = $PID; started_at = (Get-Date).ToString('o'); output = $outputPath } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lockPath 'owner.json') -Encoding UTF8

$argumentValues = @(
    'scripts\train_teacher_imitation.py',
    '--config', $configPath,
    '--checkpoint', $checkpointPath,
    '--dataset', $datasetPath,
    '--output', $outputPath,
    '--action-only', '--mentor-guided', '--mentor-min-epoch', [string]$MentorMinEpoch,
    '--epochs', [string]$Epochs,
    '--batch-size', [string]$BatchSize,
    '--feature-batch-size', [string]$FeatureBatchSize,
    '--learning-rate', [string]$LearningRate,
    '--recurrent-learning-rate', [string]$RecurrentLearningRate,
    '--patience', [string]$Patience,
    '--seed', [string]$Seed,
    '--sequence-length', [string]$SequenceLength,
    '--false-entry-penalty', [string]$FalseEntryPenalty,
    '--ranking-loss-weight', [string]$RankingLossWeight,
    '--checkpoint-every-windows', [string]$CheckpointEveryWindows
)
$argumentText = (($argumentValues | ForEach-Object { Quote-Argument ([string]$_) }) -join ' ')

try {
    $existingAttemptNumbers = @(
        Get-ChildItem -LiteralPath $attemptDirectory -Directory -ErrorAction SilentlyContinue |
            ForEach-Object {
                if ($_.Name -match '^attempt_(\d+)$') { [int]$Matches[1] }
            }
    )
    $attemptNumberBase = if ($existingAttemptNumbers.Count -gt 0) {
        ([int](($existingAttemptNumbers | Measure-Object -Maximum).Maximum) + 1)
    } else {
        1
    }

    for ($attempt = 1; $attempt -le $MaxAttempts; $attempt++) {
        if (Test-Path -LiteralPath $outputPath) {
            Write-Output ("SUPERVISOR_DONE final_checkpoint={0}" -f $outputPath)
            break
        }

        $attemptNumber = $attemptNumberBase + $attempt - 1
        $attemptTag = 'attempt_{0:D2}' -f $attemptNumber
        $attemptPath = Join-Path $attemptDirectory $attemptTag
        New-Item -ItemType Directory -Path $attemptPath -Force | Out-Null
        $stdoutPath = Join-Path $attemptPath 'stdout.log'
        $stderrPath = Join-Path $attemptPath 'stderr.log'
        $startedAt = Get-Date
        Write-Output ("SUPERVISOR_START attempt={0} started_at={1} resume_exists={2}" -f $attempt, $startedAt.ToString('o'), (Test-Path -LiteralPath $resumePath))

        $child = Start-Process -FilePath $pythonPath -WorkingDirectory $workspacePath `
            -ArgumentList $argumentText -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -PassThru
        Write-Output ("SUPERVISOR_CHILD pid={0} attempt={1}" -f $child.Id, $attempt)
        Wait-Process -Id $child.Id
        $exitCode = $child.ExitCode
        $finishedAt = Get-Date
        $finalExists = Test-Path -LiteralPath $outputPath
        Write-Output ("SUPERVISOR_EXIT attempt={0} exit_code={1} final_exists={2} finished_at={3}" -f $attempt, $exitCode, $finalExists, $finishedAt.ToString('o'))

        if ($finalExists) {
            Write-Output ("SUPERVISOR_DONE final_checkpoint={0}" -f $outputPath)
            break
        }
        $stderrText = if (Test-Path -LiteralPath $stderrPath) {
            Get-Content -LiteralPath $stderrPath -Raw -ErrorAction SilentlyContinue
        } else {
            ''
        }
        $gateRejected = $stderrText -match '(?m)(validation gate failed|autonomous validation gate failed)'
        if ($gateRejected) {
            # A deliberate validation rejection is evidence that the lineage
            # did not meet its contract.  Retrying the same checkpoint cannot
            # change that outcome and would spend hours without new evidence.
            Write-Output ("SUPERVISOR_REJECT_GATE attempt={0} stderr={1}" -f $attempt, $stderrPath)
            throw "Trainer bi validation gate; khong tu dong retry. Xem $stderrPath"
        }
        if ($attempt -lt $MaxAttempts) {
            Write-Output ("SUPERVISOR_RETRY attempt={0} delay_seconds={1} resume_exists={2}" -f $attempt, $RetryDelaySeconds, (Test-Path -LiteralPath $resumePath))
            if ($RetryDelaySeconds -gt 0) { Start-Sleep -Seconds $RetryDelaySeconds }
        } else {
            throw "Trainer khong tao final checkpoint sau $MaxAttempts attempts; xem $attemptDirectory"
        }
    }
}
finally {
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Recurse -Force
    }
}
