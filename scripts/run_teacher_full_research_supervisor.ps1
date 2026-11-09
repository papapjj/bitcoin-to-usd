<#
.SYNOPSIS
    Chain the validated action-teacher run into the research-only geometry stage.

.DESCRIPTION
    The action stage is owned by run_teacher_imitation_supervisor.ps1 and may
    take many hours. This supervisor waits for its candidate checkpoint and
    current sidecar proof, then runs geometry-only on a different output path.
    It never starts geometry while action training is still writing the source,
    and it never presents the resulting full teacher as deployable: MT5/OOS
    gates remain a separate promotion decision.
#>
[CmdletBinding()]
param(
    [string]$Python = 'C:\Python314\python.exe',
    [string]$Workspace = (Split-Path -Parent $PSScriptRoot),
    [string]$Config = 'configs\he_chuyen_gia_mot_context_mot_policy.yaml',
    [Parameter(Mandatory = $true)] [string]$ActionCheckpoint,
    [Parameter(Mandatory = $true)] [string]$Output,
    [Parameter(Mandatory = $true)] [string]$Dataset,
    [int]$GeometryEpochs = 60,
    [int]$GeometryBatchSize = 512,
    [int]$FeatureBatchSize = 32,
    [double]$GeometryLearningRate = 0.002,
    [int]$GeometryPatience = 10,
    [int]$SequenceLength = 64,
    [int]$PollSeconds = 30
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

if ($GeometryEpochs -lt 1) { throw 'GeometryEpochs phai >= 1' }
if ($GeometryBatchSize -lt 1) { throw 'GeometryBatchSize phai >= 1' }
if ($FeatureBatchSize -lt 1) { throw 'FeatureBatchSize phai >= 1' }
if ($GeometryPatience -lt 1) { throw 'GeometryPatience phai >= 1' }
if ($SequenceLength -lt 2) { throw 'SequenceLength phai >= 2' }
if ($PollSeconds -lt 5) { throw 'PollSeconds phai >= 5' }

$workspacePath = [IO.Path]::GetFullPath($Workspace)
$pythonPath = [IO.Path]::GetFullPath($Python)
$trainScript = Join-Path $workspacePath 'scripts\train_teacher_imitation.py'
$configPath = Resolve-WorkspacePath $Config
$actionPath = Resolve-WorkspacePath $ActionCheckpoint
$datasetPath = Resolve-WorkspacePath $Dataset
$outputPath = Resolve-WorkspacePath $Output
$outputDirectory = Split-Path -Parent $outputPath
$actionLockPath = Join-Path (Split-Path -Parent $actionPath) ((Split-Path -Leaf $actionPath) + '.supervisor.lock')
$lockPath = Join-Path $outputDirectory ((Split-Path -Leaf $outputPath) + '.supervisor.lock')
$logDirectory = Join-Path $outputDirectory ((Split-Path -Leaf $outputPath) + '.supervisor')
$hiddenCachePath = $outputPath + '.hidden.f16.mmap'
$actionReportPath = $actionPath + '.imitation.json'

foreach ($requiredPath in @($workspacePath, $pythonPath, $trainScript, $configPath, $datasetPath)) {
    if (-not (Test-Path -LiteralPath $requiredPath)) {
        throw "Khong tim thay duong dan bat buoc: $requiredPath"
    }
}
if (-not (Test-Path -LiteralPath (Split-Path -Parent $actionPath))) {
    throw "Khong tim thay thu muc action output: $(Split-Path -Parent $actionPath)"
}
New-Item -ItemType Directory -Path $outputDirectory -Force | Out-Null
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

if (Test-Path -LiteralPath $outputPath) {
    Write-Output ("FULL_TEACHER_DONE output_exists={0}" -f $outputPath)
    exit 0
}
if (Test-Path -LiteralPath $lockPath) {
    $ownerPath = Join-Path $lockPath 'owner.json'
    $ownerPid = $null
    if (Test-Path -LiteralPath $ownerPath) {
        try {
            $ownerRecord = Get-Content -LiteralPath $ownerPath -Raw | ConvertFrom-Json
            if ($null -ne $ownerRecord.pid) {
                $ownerPid = [int]$ownerRecord.pid
            }
        } catch {
            Write-Output ("GEOMETRY_LOCK_OWNER_UNREADABLE path={0}" -f $ownerPath)
        }
    }
    $ownerAlive = $false
    if ($null -ne $ownerPid) {
        $ownerAlive = $null -ne (Get-Process -Id $ownerPid -ErrorAction SilentlyContinue)
    }
    if ($ownerAlive) {
        throw "Geometry supervisor dang chay pid=$ownerPid lock=$lockPath"
    }
    Remove-Item -LiteralPath $lockPath -Recurse -Force
    Write-Output ("GEOMETRY_STALE_LOCK_REMOVED pid={0} path={1}" -f $ownerPid, $lockPath)
}
New-Item -ItemType Directory -Path $lockPath -Force | Out-Null
@{ pid = $PID; started_at = (Get-Date).ToString('o'); output = $outputPath } |
    ConvertTo-Json | Set-Content -LiteralPath (Join-Path $lockPath 'owner.json') -Encoding UTF8

try {
    Write-Output ("FULL_TEACHER_WAIT action={0} report={1}" -f $actionPath, $actionReportPath)
    while (-not (Test-Path -LiteralPath $actionReportPath)) {
        if (-not (Test-Path -LiteralPath $actionLockPath)) {
            throw 'Action supervisor da ket thuc nhung khong tao sidecar candidate; geometry khong duoc phep chay'
        }
        Start-Sleep -Seconds $PollSeconds
    }

    if (-not (Test-Path -LiteralPath $actionPath)) {
        throw 'Action sidecar ton tai nhung checkpoint action bi thieu'
    }
    $report = Get-Content -LiteralPath $actionReportPath -Raw | ConvertFrom-Json
    $validation = $report.training.validation_gate
    $autonomous = $report.training.autonomous_validation_gate
    $history = @($report.training.history)
    $roleAwareHistory = @(
        $history | Where-Object {
            $names = @($_.PSObject.Properties.Name)
            ($names -contains 'validation_buy_agent_balanced_accuracy') -and
            ($names -contains 'validation_sell_agent_balanced_accuracy') -and
            ($names -contains 'validation_buy_agent_entry_precision') -and
            ($names -contains 'validation_sell_agent_entry_precision') -and
            ($names -contains 'validation_buy_agent_entry_recall') -and
            ($names -contains 'validation_sell_agent_entry_recall')
        }
    )
    if (
        [int]$report.schema_version -lt 12 -or
        [string]$report.status -ne 'candidate_only' -or
        [bool]$report.promotable -or
        [string]$report.mt5_integration.entry_scope -ne 'all_entry_mass_v2' -or
        -not [bool]$report.verification.candidate_reload_passed -or
        -not [bool]$validation.enforced -or
        -not [bool]$validation.passed -or
        -not [bool]$autonomous.enforced -or
        -not [bool]$autonomous.passed -or
        $roleAwareHistory.Count -eq 0
    ) {
        throw 'Action candidate chua qua validation/reload/self-fed/role-aware history gate; dung truoc geometry'
    }

    $argumentValues = @(
        'scripts\train_teacher_imitation.py',
        '--config', $configPath,
        '--checkpoint', $actionPath,
        '--dataset', $datasetPath,
        '--output', $outputPath,
        '--geometry-only',
        '--geometry-epochs', [string]$GeometryEpochs,
        '--geometry-learning-rate', [string]$GeometryLearningRate,
        '--geometry-patience', [string]$GeometryPatience,
        '--batch-size', [string]$GeometryBatchSize,
        '--feature-batch-size', [string]$FeatureBatchSize,
        '--sequence-length', [string]$SequenceLength,
        '--hidden-cache', $hiddenCachePath
    )
    $argumentText = (($argumentValues | ForEach-Object { Quote-Argument ([string]$_) }) -join ' ')
    $stdoutPath = Join-Path $logDirectory 'stdout.log'
    $stderrPath = Join-Path $logDirectory 'stderr.log'
    Write-Output ("FULL_TEACHER_START geometry_output={0}" -f $outputPath)
    $child = Start-Process -FilePath $pythonPath -WorkingDirectory $workspacePath `
        -ArgumentList $argumentText -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath -PassThru
    Write-Output ("FULL_TEACHER_CHILD pid={0}" -f $child.Id)
    Wait-Process -Id $child.Id
    $exitCode = $child.ExitCode
    if ($exitCode -ne 0 -or -not (Test-Path -LiteralPath $outputPath)) {
        throw "Geometry teacher that bai: exit_code=$exitCode; xem $logDirectory"
    }
    Write-Output ("FULL_TEACHER_DONE output={0}" -f $outputPath)
}
finally {
    if (Test-Path -LiteralPath $lockPath) {
        Remove-Item -LiteralPath $lockPath -Recurse -Force
    }
}
