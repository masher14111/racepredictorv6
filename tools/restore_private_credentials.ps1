[CmdletBinding()]
param([switch]$VerifyOnly)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$envFile = Join-Path $projectRoot '.env'
if (-not (Test-Path -LiteralPath $envFile)) { throw 'The private .env file is missing.' }
$allowedNames = @('OPENROUTER_API_KEY', 'FIRECRAWL_API_KEY')
$importedNames = @()
foreach ($line in [System.IO.File]::ReadAllLines($envFile)) {
    if ([string]::IsNullOrWhiteSpace($line) -or $line.StartsWith('#')) { continue }
    $separator = $line.IndexOf('=')
    if ($separator -lt 1) { throw 'Invalid private environment file format.' }
    $keyName = $line.Substring(0, $separator)
    $keyValue = $line.Substring($separator + 1)
    if ($keyName -notin $allowedNames -or [string]::IsNullOrWhiteSpace($keyValue)) {
        throw 'Unrecognised or empty private credential entry.'
    }
    if (-not $VerifyOnly) {
        [Environment]::SetEnvironmentVariable($keyName, $keyValue, 'User')
        [Environment]::SetEnvironmentVariable($keyName, $keyValue, 'Process')
    }
    $importedNames += $keyName
    $keyValue = $null
}
if ($importedNames.Count -ne 2 -or ($importedNames | Select-Object -Unique).Count -ne 2) {
    throw 'Expected both predictor API keys exactly once.'
}
if ($VerifyOnly) { Write-Host 'Private credential file verified; no environment settings changed.' }
else { Write-Host 'Predictor API keys restored for this Windows user. Reopen the application terminal before starting the app.' }
Write-Host 'Proxy and service configuration is loaded by the application from config.local.yaml.'
