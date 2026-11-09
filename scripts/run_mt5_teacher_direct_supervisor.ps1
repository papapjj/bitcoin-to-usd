<#
.SYNOPSIS
    Run MT5 only after a validated teacher-imitation checkpoint exists.

.DESCRIPTION
    This is a fail-closed bridge between the action-only teacher worker and
    the MT5 ablation runner.  A resume checkpoint is never accepted.  The
    final checkpoint must have its imitation sidecar, candidate reload proof,
    validation gate and autonomous self-fed gate.  The MT5 runner itself
    still performs frozen-first LCB gating before any online arm.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)] [string]$Checkpoint,
    [Parameter(Mandatory = $true)] [string]$Profile,
    [Parameter(Mandatory = $true)] [string]$Config,
    [Parameter(Mandatory = $true)] [string]$RunnerDir,
    [Parameter(Mandatory = $true)] [string]$OutputDir,
    [Parameter(Mandatory = $true)] [string]$FromDate,
    [Parameter(Mandatory = $true)] [string]$ToDate,
    [int]$Model = 4,
    [int]$Seed = 0,
    [int]$PollSeconds = 20,
    [string]$SupervisorLog = ''
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

if ($Model -ne 4) { throw 'Teacher-direct supervisor chi chap nhan MT5 Model 4.' }
if ($PollSeconds -lt 1) { throw 'PollSeconds phai >= 1.' }

$workspace = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '..'))
function Resolve-WorkspacePath([string]$Value) {
    if ([IO.Path]::IsPathRooted($Value)) { return [IO.Path]::GetFullPath($Value) }
    return [IO.Path]::GetFullPath((Join-Path $workspace $Value))
}

$checkpointPath = Resolve-WorkspacePath $Checkpoint
$profilePath = Resolve-WorkspacePath $Profile
$configPath = Resolve-WorkspacePath $Config
$outputPath = Resolve-WorkspacePath $OutputDir
$sidecarPath = $checkpointPath + '.imitation.json'
$resumePath = $checkpointPath + '.resume.pkl'
$supervisorLogPath = if ($SupervisorLog) { Resolve-WorkspacePath $SupervisorLog } else { $null }

foreach ($required in @($profilePath, $configPath)) {
    if (-not (Test-Path -LiteralPath $required -PathType Leaf)) {
        throw "Khong tim thay file bat buoc: $required"
    }
}

while (-not (Test-Path -LiteralPath $sidecarPath -PathType Leaf)) {
    if ($supervisorLogPath -and (Test-Path -LiteralPath $supervisorLogPath -PathType Leaf)) {
        $log = Get-Content -LiteralPath $supervisorLogPath -Raw -ErrorAction SilentlyContinue
        if ($log -match 'SUPERVISOR_REJECT_GATE') {
            throw 'Teacher supervisor da reject validation gate; MT5 khong duoc chay.'
        }
    }
    $checkpointState = if (Test-Path -LiteralPath $checkpointPath -PathType Leaf) { 'final_checkpoint_seen' } else { 'training' }
    Write-Output ("WAIT_TEACHER state={0} sidecar={1}" -f $checkpointState, $sidecarPath)
    Start-Sleep -Seconds $PollSeconds
}

for ($resumeWait = 0; $resumeWait -lt 30 -and (Test-Path -LiteralPath $resumePath -PathType Leaf); $resumeWait++) {
    Write-Output ("WAIT_TEACHER_FINALIZE resume={0}" -f $resumePath)
    Start-Sleep -Seconds $PollSeconds
}
if (Test-Path -LiteralPath $resumePath -PathType Leaf) {
    throw "Sidecar da co nhung resume checkpoint van con sau thoi gian cho; tu choi MT5: $resumePath"
}

$report = Get-Content -LiteralPath $sidecarPath -Raw | ConvertFrom-Json
$training = $report.training
$verification = $report.verification
$validationGate = $training.validation_gate
$autonomousGate = $training.autonomous_validation_gate
if (-not [bool]$verification.candidate_reload_passed) {
    throw 'Teacher sidecar khong co candidate_reload_passed=true.'
}
if (-not [bool]$validationGate.enforced -or -not [bool]$validationGate.passed) {
    throw 'Teacher validation gate chua pass.'
}
if (-not [bool]$autonomousGate.enforced -or -not [bool]$autonomousGate.passed) {
    throw 'Teacher autonomous self-fed gate chua pass.'
}
if (Test-Path -LiteralPath $outputPath) {
    throw "MT5 output da ton tai, khong ghi de: $outputPath"
}

Write-Output ("TEACHER_VALIDATED checkpoint={0} report={1}" -f $checkpointPath, $sidecarPath)
Write-Output ("MT5_START model={0} from={1} to={2} profile={3}" -f $Model, $FromDate, $ToDate, $profilePath)
& python (Join-Path $workspace 'scripts\run_mt5_ablation_v193.py') `
    --profile $profilePath `
    --config $configPath `
    --checkpoint $checkpointPath `
    --runner-dir (Resolve-WorkspacePath $RunnerDir) `
    --output-dir $outputPath `
    --from-date $FromDate `
    --to-date $ToDate `
    --model $Model `
    --seed $Seed
$exitCode = $LASTEXITCODE
Write-Output ("MT5_EXIT={0}" -f $exitCode)
exit $exitCode
