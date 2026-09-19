[CmdletBinding()]
param()
$ErrorActionPreference = 'Stop'
Write-Host 'Enter your OpenRouter API key here. It will not be displayed or sent to chat.'
Write-Host 'This saves OPENROUTER_API_KEY in your Windows user environment.'
Write-Host 'No request or purchase is made by this setup.'
$secureKey = Read-Host 'OpenRouter API key' -AsSecureString
$keyPointer = [IntPtr]::Zero
try {
    $keyPointer = [Runtime.InteropServices.Marshal]::SecureStringToBSTR($secureKey)
    $keyText = [Runtime.InteropServices.Marshal]::PtrToStringBSTR($keyPointer).Trim()
    if ([string]::IsNullOrWhiteSpace($keyText)) { throw 'No key entered; nothing changed.' }
    [Environment]::SetEnvironmentVariable('OPENROUTER_API_KEY', $keyText, 'User')
    $env:OPENROUTER_API_KEY = $keyText
    Write-Host 'Key saved locally. The hosted trial reads it at step 15.' -ForegroundColor Green
    Write-Host 'The trial is capped at US$5 total and 1,200 requests; existing account credit is required.'
} finally {
    $keyText = $null
    if ($keyPointer -ne [IntPtr]::Zero) {
        [Runtime.InteropServices.Marshal]::ZeroFreeBSTR($keyPointer)
    }
    if ($secureKey) { $secureKey.Dispose() }
}
