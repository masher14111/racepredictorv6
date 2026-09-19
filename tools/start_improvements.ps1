[CmdletBinding()]
param(
    [ValidateSet('run', 'watch', 'status', 'stop', 'doctor')]
    [string]$Action = 'run',
    [switch]$DryRun,
    [ValidateRange(1, 99)][int]$MaxSteps = 99
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $pythonPath)) {
    throw "The project Python environment is missing: $pythonPath"
}
$runnerPath = Join-Path $PSScriptRoot 'improvement_runner.py'
$env:PYTHONIOENCODING = 'utf-8'
$env:PYTHONUNBUFFERED = '1'
try { [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new($false) } catch {}
if (-not $env:OPENROUTER_API_KEY) {
    $storedKey = [Environment]::GetEnvironmentVariable('OPENROUTER_API_KEY', 'User')
    if ($storedKey) { $env:OPENROUTER_API_KEY = $storedKey }
    $storedKey = $null
}
Push-Location -LiteralPath $projectRoot
try {
    Write-Host ''
    Write-Host 'Race Predictor - Automatic Improvements' -ForegroundColor Cyan
    Write-Host 'Fresh CLI session per step; shared memory; verified handoffs.'
    Write-Host 'Use the Stop launcher to finish the current step and pause.'
    Write-Host 'Ctrl+C interrupts immediately. Start again to resume from saved evidence.'
    Write-Host ''
    $runnerArgs = @($runnerPath, $Action)
    if ($Action -eq 'run') {
        $runnerArgs += @('--max-steps', "$MaxSteps")
        if ($DryRun) { $runnerArgs += '--dry-run' }
    }
    & $pythonPath @runnerArgs
    $runnerExit = $LASTEXITCODE
} finally {
    Pop-Location
}
exit $runnerExit
